"""Measure a command-tracking policy against zero action, on a stated command grid.

`record/greedy_eval.py` is the instrument for the OLD reward and it cannot
measure this one. Two reasons, and the second is the one that matters:

1. It has no notion of a command. It rolls episodes out and sums the reward,
   which under `HoundPDTrackDesert-v0` means summing over whatever commands the
   sampler happened to draw.
2. **"Does the policy beat zero action?" is undefined over a mixture containing
   STOP.** Zero action is near-optimal on STOP *by design* -- it scores 909.35
   there against a theoretical 1000 -- so a policy that never moves collects a
   tenth of the mixture's value legitimately, and a ratio computed over the
   mixture silently rewards standing again. The exploit metric is dead only
   relative to a stated protocol.

So the protocol is the instrument. `docs/theory/command-tracking-reward.md`
section 3 fixes a grid of commands, requires both arms be run on identical
command sequences paired by seed, and requires the ratio be reported
**excluding the (0,0,0) cell**, which is reported separately as a
stop-competence check where the policy should MATCH zero action rather than
beat it. Publish the grid with the number, always.

    venv/bin/python -m bestiary.record.track_eval --run hound_track_desert_s0
    venv/bin/python -m bestiary.record.track_eval --run hound_track_desert_s0 \
        --episodes 20 --json

With no `--run`, it measures the zero-action arm alone -- which is how the
committed denominator in `research/measurements/tracking_baseline_zero_action.json`
was produced.

WHAT TO READ, IN ORDER

`drive_grid_ratio` is the headline: the policy's mean over the six driving
cells divided by zero action's. Section 5 predicts >= 5x for a successful run,
against 1.128x for the reward this replaces.

`mean_track` per cell is the unitless one, E[Phi_v * Phi_w] in [0,1]. It
survives cost-coefficient retuning and episode-length changes, so it is the
number that will still mean something after the contact cost is fixed.

`command_gain` is the cheapest lie-detector in the whole design. It regresses
achieved forward velocity on commanded forward velocity across the drive
cells. A slope near 1 is tracking; a slope near 0 means the policy is ignoring
the three command slots entirely and has found some command-independent gait
that the mixture happens to pay for. A policy can post a respectable return
with a dead command input, and this is the only number here that notices.

DO NOT READ `drive_grid_ratio` WHEN EITHER ARM IS NEGATIVE

Returns here are not bounded below by zero: the costs are paid every step and
the tracking term can be ~0, so zero action itself scores **-4.72 and -4.69**
on the two (0.5, 0, +-0.4) cells. A ratio of two negative numbers is not a
performance ordering, and a ratio that crosses zero is not even monotone.
Measured against the live run at 82k steps -- 5% trained, crashing 21 of 21
episodes, unambiguously worse than doing nothing -- this reported a *ratio of
3.328* on the (0.5, 0, 0.4) cell, because both arms were negative and the
policy's was less so.

`drive_grid_track` does not have this failure: it is E[Phi_v * Phi_w], bounded
in [0, 1] by construction, and it read 0.0115 against zero action's 0.0652 on
that same measurement -- correctly saying "far worse". Section 5 already
recommends it as the metric that survives cost-coefficient retuning; it also
happens to be the only one of the two that is safe to quote unconditionally.

So: read the track score first, and quote the ratio only once both arms are
comfortably positive. Both are reported because the section 5 prediction
(">= 5x") is stated as a ratio and must be resolvable in its own terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bestiary import paths

# Section 3's eval grid. The stop cell is LAST and is excluded from the
# headline ratio.
EVAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.5, 0.0, 0.4),
    (0.5, 0.0, -0.4),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 0.0),      # stop-competence cell, reported separately
)
STOP_CELL = (0.0, 0.0, 0.0)
TRACK_ENV = "HoundPDTrackDesert-v0"


def rollout(env, seed: int, policy=None, forced_cmd=None) -> dict:
    """One episode. `policy=None` means zero action; `forced_cmd` pins the command.

    Deterministic on both arms: zero action is deterministic by definition and
    the policy is queried with `deterministic=True`, so the two are compared
    like with like. `learnings/007` is the record of what happens when a noisy
    statistic is compared against a clean one.
    """
    obs, _ = env.reset(seed=seed)
    if forced_cmd is not None:
        env.unwrapped._cmd = np.array(forced_cmd, dtype=float)
        # Push the resample past the horizon so the command is genuinely held
        # for the whole episode, then refresh the observation so the policy is
        # actually SHOWN the command it is being scored against.
        env.unwrapped._steps_until_resample = 10**9
        obs = env.unwrapped._get_obs()

    zero = np.zeros(env.action_space.shape[0])
    total = 0.0
    phi_v, phi_w, achieved, commanded = [], [], [], []
    steps, terminated = 0, False
    while True:
        action = zero if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(action)
        total += r
        phi_v.append(info["track_phi_v"])
        phi_w.append(info["track_phi_w"])
        achieved.append(info["achieved_vx"])
        commanded.append(info["cmd_vx"])
        steps += 1
        if term or trunc:
            terminated = term
            break
    return {
        "return": total,
        "steps": steps,
        "terminated": terminated,
        "mean_phi_v": float(np.mean(phi_v)),
        "mean_phi_w": float(np.mean(phi_w)),
        "mean_track": float(np.mean(np.array(phi_v) * np.array(phi_w))),
        "achieved_vx": float(np.mean(achieved)),
        "commanded_vx": float(np.mean(commanded)),
    }


def _arm(env, policy, episodes: int, seed0: int) -> dict:
    """Run one arm over the whole grid. Both arms get the same seeds."""
    cells = {}
    for cell in EVAL_GRID:
        eps = [rollout(env, seed=seed0 + i, policy=policy, forced_cmd=cell)
               for i in range(episodes)]
        rets = np.array([e["return"] for e in eps])
        cells[str(cell)] = {
            "command": list(cell),
            "mean": float(rets.mean()),
            "sd": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
            "mean_track": float(np.mean([e["mean_track"] for e in eps])),
            "mean_phi_v": float(np.mean([e["mean_phi_v"] for e in eps])),
            "mean_phi_w": float(np.mean([e["mean_phi_w"] for e in eps])),
            "achieved_vx": float(np.mean([e["achieved_vx"] for e in eps])),
            "commanded_vx": float(cell[0]),
            "crashes": int(sum(e["terminated"] for e in eps)),
            "episodes": len(eps),
        }
    drive = [c for c in cells.values() if tuple(c["command"]) != STOP_CELL]
    # Command gain: OLS slope of achieved on commanded forward velocity across
    # the drive cells. Failure mode 3 -- a dead command input reads ~0.
    x = np.array([c["commanded_vx"] for c in drive])
    y = np.array([c["achieved_vx"] for c in drive])
    slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else float("nan")
    return {
        "cells": cells,
        "drive_grid_mean": float(np.mean([c["mean"] for c in drive])),
        "drive_grid_track": float(np.mean([c["mean_track"] for c in drive])),
        "stop_cell_mean": cells[str(STOP_CELL)]["mean"],
        "command_gain": slope,
        "crashes": int(sum(c["crashes"] for c in cells.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run name under runs/; omit to measure zero action alone")
    ap.add_argument("--episodes", type=int, default=20, help="episodes PER GRID CELL")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--latest", action="store_true",
                    help="use ant_sac.zip instead of ant_sac_best.zip")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym

    import bestiary.envs  # noqa: F401

    env = gym.make(TRACK_ENV)

    policy, checkpoint = None, None
    if args.run:
        from stable_baselines3 import SAC
        run_dir: Path = paths.RUNS / args.run
        checkpoint = "ant_sac.zip" if args.latest else "ant_sac_best.zip"
        ckpt = run_dir / checkpoint
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        # Refuse to compare across a moved reward. The whole point of the
        # spec hash is that a number measured under one objective must not be
        # quoted against another.
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("env_id") != TRACK_ENV:
            raise SystemExit(
                f"{args.run} trained on {cfg.get('env_id')}, not {TRACK_ENV}. "
                "Its policy was trained with zero-filled command slots, so "
                "evaluating it under nonzero commands feeds it observations off "
                "its training manifold -- undefined behaviour, not a baseline. "
                "See docs/theory/command-tracking-reward.md section 5."
            )
        policy = SAC.load(ckpt, device="cpu")

    zero = _arm(env, None, args.episodes, args.seed0)
    result = {
        "env": TRACK_ENV,
        "episodes_per_cell": args.episodes,
        "seed0": args.seed0,
        "zero_action": zero,
    }
    if policy is not None:
        trained = _arm(env, policy, args.episodes, args.seed0)
        result["run"] = args.run
        result["checkpoint"] = checkpoint
        result["trained"] = trained
        result["drive_grid_ratio"] = (
            trained["drive_grid_mean"] / zero["drive_grid_mean"]
            if zero["drive_grid_mean"] else float("nan")
        )
        result["stop_cell_ratio"] = (
            trained["stop_cell_mean"] / zero["stop_cell_mean"]
            if zero["stop_cell_mean"] else float("nan")
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"{TRACK_ENV}   {args.episodes} episodes per cell, seeds "
          f"{args.seed0}-{args.seed0 + args.episodes - 1}, both arms identical")
    header = f"  {'command':>18}  {'zero':>9}"
    if policy is not None:
        header += f"  {'policy':>9}  {'ratio':>7}  {'track':>7}"
    print(header)
    for k in zero["cells"]:
        z = zero["cells"][k]
        line = f"  {k:>18}  {z['mean']:9.2f}"
        if policy is not None:
            t = result["trained"]["cells"][k]
            ratio = t["mean"] / z["mean"] if z["mean"] else float("nan")
            line += f"  {t['mean']:9.2f}  {ratio:7.3f}  {t['mean_track']:7.4f}"
        if tuple(z["command"]) == STOP_CELL:
            line += "   <- stop cell, EXCLUDED from the headline"
        print(line)

    print(f"\n  zero-action drive-grid mean   {zero['drive_grid_mean']:9.2f}")
    if policy is not None:
        t = result["trained"]
        print(f"  policy     drive-grid mean   {t['drive_grid_mean']:9.2f}")
        print(f"  DRIVE-GRID RATIO             {result['drive_grid_ratio']:9.3f}"
              "    (section 5 predicts >= 5x for a successful run)")
        print(f"  drive-grid track score       {t['drive_grid_track']:9.4f}"
              f"    (zero action {zero['drive_grid_track']:.4f}, max 1.0)")
        print(f"  command gain (slope)         {t['command_gain']:9.3f}"
              "    (~1 tracks, ~0 = dead command input)")
        print(f"  stop cell                    {t['stop_cell_mean']:9.2f}"
              f"    (zero action {zero['stop_cell_mean']:.2f} — MATCH, do not beat)")
        print(f"  crashes                      {t['crashes']:9d}"
              f" of {args.episodes * len(EVAL_GRID)}")


if __name__ == "__main__":
    main()
