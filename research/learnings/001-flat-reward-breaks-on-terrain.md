---
triggers: [reward_change, new_terrain, new_robot, long_run]
guard: standing-control
last_confirmed: 2026-07-26
---

# 001 — A reward tuned on flat ground breaks on terrain

**Date:** 2026-07-25 · **From:** `runs/spyder_desert_v0`

## What happened

We measured the desert policy against a policy that does literally nothing
(zero action), 5 greedy episodes each:

| policy | reward | speed |
|---|---|---|
| `spyder_desert_v0` | 509 (best eval 832) | 0.37 m/s |
| **zero action** | **987** | 0 m/s |

Doing nothing won. On a full episode the policy earned +294 for moving forward
and paid −571 in control cost, against a 1000-point alive bonus. Moving was a
net loss of ~293.

## What we learned

`ctrl_cost_weight = 0.1` was fine on flat ground, where forward reward beats
control cost about 8.7 : 1 and the cost is noise. On terrain the ratio flips to
0.51 : 1. Same reward function, opposite incentive.

**The cost did not rise — the payoff collapsed.** Measured per step:

| | flat (v3) | desert (v0) |
|---|---|---|
| forward reward | 7.05 | 0.29 |
| effort cost | −0.81 | −0.57 |
| ratio | 8.7 : 1 | 0.51 : 1 |

Speed fell 24×; effort only fell 1.4×. (Effort went *down*, not up — creeping
slowly means smaller actions.) A cost weight calibrated against 7 m/s payoffs is
ruinous at 0.3 m/s.

The reward was telling the spider to stand still. It half-listened, which is why
it looked like slow progress instead of an obvious failure.

## What to do next time

**Run the standing check before trusting any terrain reward.** Roll a zero-action
policy on the env. If doing nothing scores higher than the trained policy, the
reward is wrong. It takes two minutes and would have caught this at 4M steps
instead of 5.75M.

Also: don't compare rewards across envs. Flat scored 7392 and desert 832, but
that gap is mostly "7 m/s × 1000 steps" — different reward regimes, not a
ranking. Compare against the do-nothing baseline **on the same env**.
