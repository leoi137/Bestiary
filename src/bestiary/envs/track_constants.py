"""Every constant in a command-tracking reward, computed from its own bound.

WHY THIS MODULE EXISTS

`envs/hound_track.py` carries five constants that are not free choices -- each
one is the solution of an inequality, and each inequality references the
kernel, the tolerances, and the robot's measured passive motion:

    VX_MIN_BACKWARD   the smallest backward command at which a STANDING machine
                      cannot out-earn the freeride cap
    VX_MIN            the same, forward
    SIGMA_V, SIGMA_W  two-sided: wide enough that measurement noise scores well,
                      tight enough that passive drift does not
    TERMINATION_PENALTY   the discounted value of escaping a negative stream

They were solved once, by hand, for the Cauchy kernel at sigma_v = 0.15. Two
things then happened that this module exists to prevent happening again.

**The backward floor was derived for one kernel and is quoted as if general.**
`hound_track.py`'s comment solves Phi((|vx| - creep)/sigma_v) * 0.9678 <= 0.17
and gets 0.3756, so 0.40. That solution is Cauchy-specific: the same inequality
under a Gaussian at sigma_v = 0.27 needs |vx| >= 0.544. A kernel change that
leaves the floor at 0.40 reopens the exact exploit the floor was added to
close, and nothing in the code would say so.

**The termination penalty was derived from an assumed rate that was never
measured.** K = c/(1-gamma) = 0.10/0.01 = 10, where c = 0.10/step was assumed.
The measured net rate is -0.00619/step for the trained policy, and the assumed
contact cost inside it (0.045/step) measures 0.00928/step -- 4.85x smaller.
`docs/lessons/006` works this through; `research/anomalies.jsonl` records it.

So: no constant here is written down. Each is returned by the function that
solves for it, given the kernel and tolerances actually in use. Change sigma_v
and the backward floor moves with it, because it is the same expression.

WHAT THIS MODULE DOES NOT DO

It does not choose sigma_v, sigma_w or w_ctrl. Those are the design, and the
design belongs in `docs/theory/`. This module enforces the consequences of a
choice, which is a different and much more mechanical job.
"""
from __future__ import annotations

import math
from collections.abc import Callable

# --- Measured properties of the machine. Every one is a MEASUREMENT with a
# --- file behind it, never an estimate; that is the whole point.

# research/measurements/tracking_noise.json, wheel command 0.0 -- the machine
# creeps BACKWARD while doing nothing, which is why the two command floors are
# asymmetric: the creep subtracts from the error under a backward command and
# adds to it under a forward one.
STANDING_CREEP_MS = 0.03553

# Same file: yaw rate rms of a machine lying still, and of one merely rolling
# with nothing steering it. The gap between these two is the entire reason the
# heading term can be earned at all.
STANDING_YAW_RADS = 0.01823
UNSTEERED_DRIVING_YAW_RADS = 0.12695

# research/measurements/tracking_noise.json, planar velocity rms while standing.
STANDING_PLANAR_RMS_MS = 0.0361

# --- The bounds themselves. These ARE judgement, and they are stated once.

# docs/theory/command-tracking-reward.md Section 2. A standing machine must not
# collect more than this per step from any command in the distribution. It is a
# fraction of what a competent policy should earn -- which is exactly the thing
# that went wrong once already: the cap was set against an ASSUMED good-policy
# rate of 0.87/step, and the achieved rate was 0.0718/step, so a leak budgeted
# at 4% of return became 90% of it. Read `freeride_cap_for` before reusing 0.17.
FREERIDE_CAP_DEFAULT = 0.17

# Same section. An unsteered but DRIVING machine must score at most this on the
# heading factor, or nothing in the reward forces the policy to actively steer.
UNSTEERED_YAW_CAP = 0.45

# The noise side of the two-sided sigma bound: a machine that is tracking
# perfectly, and is off only by its own measurement noise, should still score
# near the top of the kernel.
NOISE_FLOOR_SCORE = 0.90


def cauchy(u: float) -> float:
    """Phi(u) = 1/(1+u^2). Fat tail: Phi(3.57) = 0.073, not 1e-4."""
    return 1.0 / (1.0 + u * u)


def gaussian(u: float) -> float:
    """Phi(u) = exp(-u^2/2). Light tail: Phi(3.57) = 1.7e-3."""
    return math.exp(-0.5 * u * u)


KERNELS: dict[str, Callable[[float], float]] = {
    "cauchy": cauchy,
    "gaussian": gaussian,
}


def _invert(kernel: str, score: float) -> float:
    """The u at which the kernel returns `score`. Both kernels are monotone."""
    if not 0.0 < score < 1.0:
        raise ValueError(f"score must be in (0,1), got {score}")
    if kernel == "cauchy":
        return math.sqrt(1.0 / score - 1.0)
    if kernel == "gaussian":
        return math.sqrt(-2.0 * math.log(score))
    raise ValueError(f"unknown kernel {kernel!r}; have {sorted(KERNELS)}")


def command_floor(
    kernel: str,
    sigma_v: float,
    sigma_w: float,
    *,
    backward: bool,
    freeride_cap: float = FREERIDE_CAP_DEFAULT,
) -> float:
    """Smallest |vx_cmd| at which a STANDING machine stays under the cap.

    A standing machine's velocity error under a command (vx, 0, 0) is
    |vx -/+ creep|, and its heading factor is whatever a still body scores, so
    its per-step take is

        Phi(|vx -/+ creep| / sigma_v) * Phi(standing_yaw / sigma_w)  <=  cap

    Solving for |vx| is the whole function. The sign matters because the creep
    is BACKWARD: it cancels error under a backward command, so the same |vx|
    hands a stationary machine a strictly larger take going backwards, and the
    backward floor is therefore always the higher of the two.

    Raises if the cap is unreachable at any command magnitude -- which happens
    when the heading factor alone already exceeds it, and is a real design
    error rather than a number to clamp.
    """
    yaw_factor = KERNELS[kernel](STANDING_YAW_RADS / sigma_w)
    if yaw_factor <= freeride_cap:
        # Standing is already under the cap on heading alone; no floor needed.
        return 0.0
    speed_budget = freeride_cap / yaw_factor
    if not 0.0 < speed_budget < 1.0:
        raise ValueError(
            f"freeride cap {freeride_cap} is unreachable: a standing machine "
            f"scores {yaw_factor:.4f} on heading alone at sigma_w={sigma_w}"
        )
    err = _invert(kernel, speed_budget) * sigma_v
    # Going backward the creep cancels error, so the command must be LARGER by
    # the creep to produce the same error. Going forward it adds, so smaller.
    return err + STANDING_CREEP_MS if backward else max(0.0, err - STANDING_CREEP_MS)


def unsteered_yaw_score(kernel: str, sigma_w: float) -> float:
    """What a DRIVING machine scores on heading with nothing steering it.

    Must stay under `UNSTEERED_YAW_CAP`, or the reward pays for rolling and not
    for holding a line. This is the bound that actually binds when sigma_w is
    widened to make driving profitable, and it is the one a gap-maximising
    search will quietly walk through.
    """
    return KERNELS[kernel](UNSTEERED_DRIVING_YAW_RADS / sigma_w)


def noise_score(kernel: str, sigma_v: float, sigma_w: float) -> tuple[float, float]:
    """What a perfect tracker scores when it is off only by measurement noise.

    Should be at least `NOISE_FLOOR_SCORE` on both channels. If it is not, the
    reward is punishing the machine for error it cannot remove, which hands the
    gradient to noise.
    """
    return (
        KERNELS[kernel](STANDING_PLANAR_RMS_MS / sigma_v),
        KERNELS[kernel](STANDING_YAW_RADS / sigma_w),
    )


def termination_penalty(net_rate_per_step: float, gamma: float = 0.99) -> float:
    """K = c/(1-gamma), the discounted value of escaping a negative stream.

    The SIGN is the part everyone gets backwards, including a briefing written
    for this cycle. K does not price the income death forfeits -- it cancels
    the SUICIDE INCENTIVE an early, high-entropy policy has while its running
    rate is negative. `docs/theory/command-tracking-reward.md` Section 4 is
    explicit: "the discounted value of escaping a -c/step stream is c/(1-gamma)".

    So `net_rate_per_step` must be the net rate of an EARLY policy, and it must
    be MEASURED. Passing an assumed rate is how the shipped 10.0 came to be
    16.2x the stream it prices (`docs/lessons/006`).

    A non-negative rate means there is no suicide incentive to cancel and K
    should be zero -- returning it rather than a floor, because silently
    keeping a penalty nobody derived is the failure this module exists for.
    """
    if net_rate_per_step >= 0.0:
        return 0.0
    return -net_rate_per_step / (1.0 - gamma)


def describe(kernel: str, sigma_v: float, sigma_w: float,
             freeride_cap: float = FREERIDE_CAP_DEFAULT) -> str:
    """Every derived constant for one design, with the bound each one solves."""
    fwd = command_floor(kernel, sigma_v, sigma_w,
                        backward=False, freeride_cap=freeride_cap)
    back = command_floor(kernel, sigma_v, sigma_w,
                         backward=True, freeride_cap=freeride_cap)
    unsteered = unsteered_yaw_score(kernel, sigma_w)
    noise_v, noise_w = noise_score(kernel, sigma_v, sigma_w)
    lines = [
        f"kernel={kernel}  sigma_v={sigma_v}  sigma_w={sigma_w}  "
        f"freeride_cap={freeride_cap}",
        f"  VX_MIN          >= {fwd:.4f}   (standing take under the cap, forward)",
        f"  VX_MIN_BACKWARD >= {back:.4f}   (same, backward; creep cancels error)",
        f"  unsteered yaw score {unsteered:.4f}  "
        f"{'OK' if unsteered <= UNSTEERED_YAW_CAP else 'BREAKS'} "
        f"the <= {UNSTEERED_YAW_CAP} steering bound",
        f"  noise scores  v {noise_v:.4f}  w {noise_w:.4f}  "
        f"{'OK' if min(noise_v, noise_w) >= NOISE_FLOOR_SCORE else 'BELOW'} "
        f"the >= {NOISE_FLOOR_SCORE} noise floor",
    ]
    return "\n".join(lines)


def shipped_values_clear_their_floors() -> list[str]:
    """Check the constants `hound_track.py` actually ships against these bounds.

    This is the assertion that matters. Re-deriving a floor is only useful if
    something compares the result against what the env samples from, so this
    reports both and says whether the shipped value clears.
    """
    from bestiary.envs import hound_track as ht

    out = []
    for shipped, backward, name in ((ht.VX_MIN, False, "VX_MIN"),
                                    (ht.VX_MIN_BACKWARD, True, "VX_MIN_BACKWARD")):
        need = command_floor("cauchy", ht.SIGMA_V, ht.SIGMA_W, backward=backward)
        out.append(f"  {name:16s} ships {shipped:.4f}, needs >= {need:.4f}  "
                   f"{'CLEARS' if shipped >= need else 'VIOLATES'}")
    # The theory note solves the backward case by hand and reports 0.3756; this
    # expression gives 0.3605. Both conclude 0.40 clears and 0.35 does not, so
    # the shipped constant is unaffected, but the 4% gap is real and is
    # recorded rather than reconciled -- the note does not show its
    # intermediate steps, so which one is wrong cannot be settled from here.
    note = command_floor("cauchy", 0.15, 0.10, backward=True)
    out.append(f"  theory note states 0.3756 for the same bound; this module "
               f"gives {note:.4f} (4% apart, same conclusion)")
    return out


if __name__ == "__main__":
    print("The shipped design, re-derived from its own bounds:")
    print(describe("cauchy", 0.15, 0.10))
    print()
    for line in shipped_values_clear_their_floors():
        print(line)
    print("\n  The FORWARD command is the binding one, which is easy to miss:")
    print("  the creep is backward, so it ADDS to the error under a forward")
    print("  command. At the shipped floors a standing machine takes 0.1612 on")
    print("  (0.3,0,0) and 0.1401 on (-0.4,0,0) -- guards/tracking_frame.py")
    print("  asserts the worst over both signs, after a first version looked")
    print("  only one way.")
    print("\nWhat the same bounds require under a light tail and a wider sigma_v:")
    for sigma_v in (0.20, 0.25, 0.27, 0.30):
        print(describe("gaussian", sigma_v, 0.09))
        print()
