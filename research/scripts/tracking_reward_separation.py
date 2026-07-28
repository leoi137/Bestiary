"""What the command-tracking reward would pay at the tracking error we MEASURED.

THE QUESTION THIS ANSWERS

`learnings/011` established that the trained hound earns 0.00674/step more
tracking reward than a do-nothing policy and pays 0.06889/step of control cost
to get it -- 12.6x more than it earns. `nulls.jsonl` row 4 therefore refuses a
retry of this experiment unless the per-step inequality

    (Phi_v * Phi_w achieved while DRIVING) - (Phi_v * Phi_w achieved while
    STANDING)   >   w_ctrl * sum(a^2)

holds at ACHIEVABLE phi rather than ideal phi. This script evaluates candidate
rewards against exactly that inequality, holding the robot's PHYSICS fixed.

THE TRICK THAT MAKES IT POSSIBLE WITHOUT A GPU

A reward parameter change does not change what the robot did. The measurement
recorded per-cell mean Phi under the Cauchy kernel at sigma_v = 0.15 and
sigma_w = 0.10, and Phi is invertible:

    Phi = 1/(1+u^2)   =>   u = sqrt(1/Phi - 1)   =>   error = u * sigma

So the physical tracking errors -- m/s and rad/s, kernel-free -- are recoverable
from the committed measurement, and any other kernel can then be scored on them.
That is what every table below does.

THE ASSUMPTION, STATED LOUDLY BECAUSE IT IS THE WEAK POINT

This holds the ACHIEVED ERRORS FIXED while changing the reward that produced
them. A policy retrained under a new reward will not reproduce these errors --
if it did, the reward change would be pointless. So these numbers are a LOWER
BOUND on a good outcome and NOT a prediction of the retrained policy: they say
"even if the machine learns nothing new, does driving now beat standing?"

That is the right question for a release condition and the wrong one for a
performance forecast. Do not quote these as predicted returns.

A second-order caveat: inverting a MEAN Phi gives the error whose Phi equals the
mean, not the mean error, and the two differ by Jensen's inequality. Every
figure here is therefore a summary of a cell, not a per-step truth.

Run:  venv/bin/python research/scripts/tracking_reward_separation.py
"""
from __future__ import annotations

import json

import numpy as np

from bestiary import paths

# The measurement's own kernel parameters. Inverting with anything else would
# recover the wrong physical error, so these are read as constants of the DATA,
# not as design choices.
MEASURED_SIGMA_V = 0.15
MEASURED_SIGMA_W = 0.10

STOP_CELL = "(0.0, 0.0, 0.0)"
MEASUREMENT = "hound_track_desert_s0_final_sac.json"

# learnings/011, final unselected checkpoint, six drive cells.
CTRL_COST_PER_STEP = 0.06889
CTRL_SUM_A2 = CTRL_COST_PER_STEP / 0.01   # w_ctrl was 0.01; recover sum(a^2)

# docs/theory/command-tracking-reward.md Section 2. An unsteered but DRIVING
# machine yaws at this rate; the design bound says it must not score well, or
# nothing forces the policy to actively steer.
UNSTEERED_DRIVING_YAW = 0.12695
YAW_FREERIDE_CAP = 0.45


def cauchy(err: np.ndarray | float, sigma: float):
    return 1.0 / (1.0 + (np.asarray(err) / sigma) ** 2)


def gaussian(err: np.ndarray | float, sigma: float):
    return np.exp(-0.5 * (np.asarray(err) / sigma) ** 2)


KERNELS = {"cauchy": cauchy, "gauss": gaussian}


def recover_errors() -> list[tuple[str, float, float, float, float]]:
    """(cell, policy err_v, policy err_w, zero err_v, zero err_w), SI units."""
    data = json.loads((paths.RESEARCH / "measurements" / MEASUREMENT).read_text())
    pol, zero = data["trained"]["cells"], data["zero_action"]["cells"]

    def inv(phi: float, sigma: float) -> float:
        return float(np.sqrt(1.0 / phi - 1.0) * sigma)

    out = []
    for key in pol:
        if key == STOP_CELL:
            continue
        out.append((
            key,
            inv(pol[key]["mean_phi_v"], MEASURED_SIGMA_V),
            inv(pol[key]["mean_phi_w"], MEASURED_SIGMA_W),
            inv(zero[key]["mean_phi_v"], MEASURED_SIGMA_V),
            inv(zero[key]["mean_phi_w"], MEASURED_SIGMA_W),
        ))
    return out


def score(errs, kernel: str, sigma_v: float, sigma_w: float) -> dict:
    k = KERNELS[kernel]
    policy = np.array([k(pv, sigma_v) * k(pw, sigma_w) for _, pv, pw, _, _ in errs])
    standing = np.array([k(zv, sigma_v) * k(zw, sigma_w) for _, _, _, zv, zw in errs])
    gaps = policy - standing
    return {
        "gap": float(gaps.mean()),
        "standing_income": float(standing.mean()),
        "policy_income": float(policy.mean()),
        "worst_cell": float(gaps.min()),
        "cells_positive": int((gaps > 0).sum()),
        "n_cells": len(gaps),
        # The steering bound is a SEPARATION, not a level: what matters is how
        # much better active steering scores than passive drift, because that
        # difference is the entire incentive to steer.
        "unsteered_score": float(k(UNSTEERED_DRIVING_YAW, sigma_w)),
        "per_cell": list(zip([e[0] for e in errs], policy, standing, gaps)),
    }


def main() -> None:
    errs = recover_errors()

    print("PHYSICAL TRACKING ERROR, recovered from the committed measurement")
    print("(kernel-free: these are what the robot actually did)")
    print(f"{'cell':17s} {'pol err_v':>10s} {'pol err_w':>10s} "
          f"{'zero err_v':>11s} {'zero err_w':>11s}")
    for key, pv, pw, zv, zw in errs:
        print(f"{key:17s} {pv:10.4f} {pw:10.4f} {zv:11.4f} {zw:11.4f}")
    print("  units: err_v m/s, err_w rad/s")

    print(f"\nTHE BAR: control cost {CTRL_COST_PER_STEP:.5f}/step "
          f"at w_ctrl=0.01, i.e. sum(a^2) = {CTRL_SUM_A2:.2f}")

    print("\nWHERE THE STANDING INCOME COMES FROM TODAY (cauchy 0.15/0.10)")
    base = score(errs, "cauchy", MEASURED_SIGMA_V, MEASURED_SIGMA_W)
    total_standing = sum(s for _, _, s, _ in base["per_cell"])
    for key, _p, s, _g in base["per_cell"]:
        print(f"  {key:17s} {s:.5f}   {100 * s / total_standing:5.1f}% of it")
    print("  One cell carries most of it, and the backward command floor moved")
    print("  to 0.40 in a4e7ef5 specifically to shrink that cell.")

    print("\n" + "=" * 78)
    print("CANDIDATES against all three criteria")
    print("  gap            must exceed the control cost, or driving does not pay")
    print("  standing_inc   must stay SMALL, or the alive bonus is reborn under")
    print("                 a new name -- this is the criterion a gap-only scan")
    print("                 silently ignores, and lesson 003 is about it")
    print("  unsteered      must stay under 0.45, or nothing forces active")
    print("                 steering (theory note Section 2)")
    print("=" * 78)
    header = (f"{'kernel':7s} {'s_v':>5s} {'s_w':>5s} {'gap':>9s} {'stand':>8s} "
              f"{'policy':>8s} {'worst':>9s} {'+cells':>7s} {'unstrd':>7s}")
    print(header)
    grid = [(0.15, 0.10), (0.20, 0.10), (0.25, 0.10), (0.27, 0.10), (0.30, 0.10),
            (0.20, 0.15), (0.25, 0.15), (0.30, 0.15), (0.27, 0.20), (0.30, 0.20),
            (0.35, 0.40)]
    for kernel in ("cauchy", "gauss"):
        for sigma_v, sigma_w in grid:
            r = score(errs, kernel, sigma_v, sigma_w)
            beats = "OK " if r["gap"] > CTRL_COST_PER_STEP else "no "
            yaw = "" if r["unsteered_score"] <= YAW_FREERIDE_CAP else "  YAW-BOUND BROKEN"
            print(f"{kernel:7s} {sigma_v:5.2f} {sigma_w:5.2f} {r['gap']:+9.4f} "
                  f"{r['standing_income']:8.4f} {r['policy_income']:8.4f} "
                  f"{r['worst_cell']:+9.4f} {r['cells_positive']:4d}/{r['n_cells']} "
                  f"{r['unsteered_score']:7.3f} {beats}{yaw}")

    print("\nTHE TAIL IS NOT THE LEVER -- SIGMA IS. But the tail decides the PRICE")
    print("of widening sigma, which is the standing income it lets back in:")
    print(f"{'':16s} {'gap':>9s} {'standing':>9s} {'gap per unit standing':>23s}")
    for kernel, sigma_v, sigma_w in (("cauchy", 0.20, 0.15), ("gauss", 0.20, 0.15),
                                     ("cauchy", 0.25, 0.15), ("gauss", 0.25, 0.15)):
        r = score(errs, kernel, sigma_v, sigma_w)
        print(f"  {kernel:6s} {sigma_v:.2f}/{sigma_w:.2f} {r['gap']:+9.4f} "
              f"{r['standing_income']:9.4f} "
              f"{r['gap'] / r['standing_income']:23.2f}")
    print("  At matched standing income the light tail buys a strictly larger")
    print("  gap. That -- not the far-field leak on its own -- is what the")
    print("  kernel family is actually worth here.")

    print("\nSENSITIVITY TO w_ctrl (the bar moves too; nulls row 2 caps it at 0.02)")
    print(f"{'w_ctrl':>7s} {'bar/step':>10s}   which candidates clear it")
    for w_ctrl in (0.010, 0.005, 0.003, 0.002):
        bar = w_ctrl * CTRL_SUM_A2
        winners = [f"{k} {sv:.2f}/{sw:.2f}"
                   for k in ("cauchy", "gauss") for sv, sw in grid
                   if score(errs, k, sv, sw)["gap"] > bar
                   and score(errs, k, sv, sw)["unsteered_score"] <= YAW_FREERIDE_CAP]
        print(f"{w_ctrl:7.3f} {bar:10.5f}   "
              f"{', '.join(winners) if winners else 'NONE that also hold the yaw bound'}")
    print("\n  Read that last table before touching sigma_w: lowering w_ctrl and")
    print("  widening sigma_w both make driving pay, but only one of them keeps")
    print("  the incentive to steer.")


if __name__ == "__main__":
    main()
