"""Why one seed is a probe and not a result, worked on this repo's own numbers.

The seed rule in CLAUDE.md says no effect is claimed from one run. That reads
like bureaucratic caution until you put this project's actual measurements into
the arithmetic, at which point it stops being caution and becomes the
difference between a finding and a coin flip.

Two things are conflated when people say "n". This script separates them,
because the project has a sample-size problem at BOTH levels and they need
different fixes:

  EPISODES  -- how many rollouts you average to score one trained policy.
               Cheap. Fixed by measuring more.
  SEEDS     -- how many times you repeat the whole training run.
               Expensive. The only thing that estimates between-run spread,
               and the one the record has exactly one of per arm.

More episodes cannot substitute for more seeds. Averaging 10,000 episodes of
one training run measures that run's policy to arbitrary precision and says
nothing whatever about whether a second run would land anywhere near it.

    venv/bin/python research/scripts/seed_variance_math.py

Writes nothing, needs no GPU, and runs instantly.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Measurements. Every number below was produced by code in this repo, not by
# hand: record/greedy_eval.py at n=60, three disjoint seed blocks, greedy vs
# zero-action under one protocol. Reported in research/learnings/008.
# ---------------------------------------------------------------------------

# The torque hound, paired greedy-vs-zero-action difference at n=60 episodes.
# This is the interval record/greedy_eval.py reported; the point estimate and
# standard error are recovered from it below rather than re-asserted.
TORQUE_CI_LO = -52.1      # return points
TORQUE_CI_HI = +132.9     # return points
CI_Z = 1.96               # two-sided 95% normal quantile

# The PD hound, same protocol, n=60.
PD_GREEDY = 1078.2        # mean greedy return
PD_ZERO_ACTION = 955.5    # mean zero-action return on identical seeds

# Fraction of the torque hound's 60 greedy episodes that scored BELOW the
# do-nothing control. A policy is not "good with occasional lapses" at this
# rate -- it is bimodal.
TORQUE_BELOW_STANDING = 16 / 60

# The eval size that was standard in this repo before learnings/008.
OLD_EVAL_EPISODES = 5


def main() -> int:
    # -- 1. What a confidence interval containing zero actually says ---------
    point = (TORQUE_CI_HI + TORQUE_CI_LO) / 2.0
    half_width = (TORQUE_CI_HI - TORQUE_CI_LO) / 2.0
    stderr = half_width / CI_Z

    print("1. The torque hound, paired against doing nothing (n=60 episodes)")
    print(f"     95% CI            [{TORQUE_CI_LO:+.1f}, {TORQUE_CI_HI:+.1f}] return points")
    print(f"     point estimate    {point:+.1f}")
    print(f"     standard error    {stderr:.1f}")
    print(f"     contains zero?    {'YES' if TORQUE_CI_LO < 0 < TORQUE_CI_HI else 'no'}")
    print("     -> The trained policy might be 133 points better than doing")
    print("        nothing, or 52 points WORSE. Both are consistent with the")
    print("        data. 'It beats the baseline' is not an established claim.")
    print()

    # -- 2. How much more measurement would settle it -----------------------
    # Standard error scales as 1/sqrt(n), so to shrink the half-width to the
    # size of the effect itself, n must grow by the square of the ratio.
    target = abs(point)
    ratio = half_width / target
    n_needed = math.ceil(60 * ratio ** 2)

    print("2. What would it take to resolve an effect this small?")
    print(f"     to get the half-width ({half_width:.1f}) below the effect ({target:.1f}),")
    print(f"     n must grow by ({half_width:.1f}/{target:.1f})^2 = {ratio ** 2:.1f}x")
    print(f"     -> {n_needed} episodes, up from 60")
    print("     and that is only to establish the effect for THIS ONE trained")
    print("     policy. It says nothing about the next training run.")
    print()

    # -- 3. Why a small eval hides a bimodal policy -------------------------
    p_clean = (1 - TORQUE_BELOW_STANDING) ** OLD_EVAL_EPISODES
    print("3. Why the old 5-episode eval was worse than no eval")
    print(f"     the torque policy falls below standing {TORQUE_BELOW_STANDING:.1%} of episodes")
    print(f"     P(all {OLD_EVAL_EPISODES} draws look clean) = "
          f"(1 - {TORQUE_BELOW_STANDING:.3f})^{OLD_EVAL_EPISODES} = {p_clean:.3f}")
    print(f"     -> roughly {p_clean:.0%} of the time, a policy that fails one episode")
    print("        in four reports a spotless five-episode evaluation.")
    print()

    # -- 4. The level the record has no measurement of at all ---------------
    pd_ratio = PD_GREEDY / PD_ZERO_ACTION
    print("4. The gap this project cannot close by measuring harder")
    print(f"     PD hound: {PD_GREEDY:.1f} greedy vs {PD_ZERO_ACTION:.1f} standing "
          f"= x{pd_ratio:.3f}")
    print("     measured over 60 episodes -- of ONE training run, seed 0.")
    print("     Episodes estimate the policy. They cannot estimate the SPREAD")
    print("     between training runs, because there is only one training run.")
    print("     n=1 seed has no spread to report, so any comparison against it")
    print("     is one draw against one draw, however many episodes back each.")
    print()
    print("     This is why the seed rule asks for >=3 seeds per arm and one")
    print("     changed variable, and why a single-seed row is written up as")
    print("     provisional rather than as a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
