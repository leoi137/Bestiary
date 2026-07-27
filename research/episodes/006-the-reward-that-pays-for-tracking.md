# Episode 006 — The reward that pays for tracking

**Date:** 2026-07-27

## Thesis

The hound's reward paid it for existing. A machine that stands still collected
nearly all of it, so the optimizer correctly learned that existence pays and
locomotion barely does. This episode harvests the two-seed measurement of how
big that problem actually is, then replaces the reward with one whose only
positive term is a product of two tolerance kernels against a sampled velocity
command — and launches the first run under it.

## Diagnosis

Two seeds now exist on the PD arm, which is the first time any arm in this
record has more than one. Measured with `record/greedy_eval.py` at n=60 on
seeds 0–59, both arms under one protocol, deterministic actions on both sides:

| run | seed | greedy | zero-action | ratio | crashes |
|---|---|---|---|---|---|
| `hound_pd_desert_v0` | 0 | 1049.10 ± 324.36 | 955.58 ± 0.68 | ×1.0979 | 8/60 |
| `hound_pd_desert_s1` | 1 | **1140.65** ± 161.41 | 955.58 ± 0.68 | **×1.1937** | 2/60 |

Seed 1 clears the project's 1.18 advisory margin. No hound arm had done that
before, and the prediction that it would was written at p = 0.20.

**The re-measurement is the more useful result.** Episode 004 published 1078.2
for seed 0. Measured today, under the identical protocol, the same checkpoint
gives 1049.10. The first hypothesis — a nondeterministic instrument — is
wrong: repeated runs of `greedy_eval` return byte-identical means on both runs.
The 29-point gap is the *seed block*. Episode 004 used "three disjoint seed
blocks"; this used seeds 0–59.

That is not a discrepancy to reconcile, it is the size of the error bar. Seed
0's own spread implies a standard error of 324.36/√60 = **±41.8** on an n=60
mean, so two honest readings of one checkpoint differing by 29 points is
entirely ordinary — and completely invisible unless somebody re-measures.

**So the 1.18 clearance is a point-estimate pass, and is recorded as one.**
Seed 1's standard error is 161.41/√60 = ±20.8, which is ±0.022 on the ratio.
×1.1937 sits **0.6 standard errors** above 1.18. A different seed block could
as easily have put the same checkpoint below the line.

**The mean on this arm is mostly a crash-rate proxy.** 58 of 60 episodes sit at
~1170; the failures score 191 and 376. Six crashes of difference between the
two seeds moves the mean about 90 points, which is the entire two-seed spread
of 91.55. Read the crash count beside the mean or the mean means nothing.

## What happened

### The replacement reward, and what building it turned up

`HoundPDTrackDesert-v0`. The only positive term is
`1[healthy] · Φ(u_v) · Φ(u_w)` with `Φ(u) = 1/(1+u²)`, measured against a
command resampled every 200–300 steps. There is no term a non-tracking policy
can collect. The design is `docs/theory/command-tracking-reward.md`; this
episode is its implementation and first test.

The observation did **not** change. The three command values go into slots
`envs/hound.py` reserved and has been zero-filling since its first commit, so
the width stays 169, the hash does not move, and every existing hound
checkpoint still loads. This is the one case in the project where a feature
lands without touching the one-way door, and it is only possible because
someone paid for the slots up front.

**The zero-action baseline under the new reward, measured** — the only number
that survives the objective change (`research/scripts/tracking_baseline.py`):

| protocol | zero-action |
|---|---|
| mixture, as trained | 108.35 ± 116.59 |
| drive grid, stop cell excluded | **55.73** |
| stop cell (0,0,0) | 909.35 |

Two things fell out of measuring it.

**The design's tracking arithmetic is right; its cost assumption is not.**
Per-slice, measured against predicted: STOP 0.9242 vs 0.904, TURN 0.0460 vs
0.049, DRIVE 0.0467 vs 0.046, mixture E[track] 0.1175 vs 0.132. The kernel, the
tolerances and the command mixture all behave as derived. But §5 assumes a
standing machine pays ≈0.045/step in contact cost, and it pays **0.0092** —
five times less. Standing therefore scores ~108/episode rather than the
predicted 87, and the headline predicted separation falls from 8× to about
6.5×. The separation argument survives; the specific multiplier does not.

**Zero action scores 226.38 on the (−0.3, 0, 0) cell** against 61.41 on
(0.5, 0, 0), because the hound's passive backward creep of −0.0354 m/s
partially *satisfies* a reverse command. The creep `learnings/009` could not
explain is now also a small reward source. It is harmless while the command
distribution asks for forward 80% of the time, and it would not be if that
ever changed.

### Two bugs caught before they cost anything

**The world-frame velocity bug.** The velocity error must be computed in the
trunk's heading frame; in the world frame, a correctly driving body's velocity
rotates under a yaw command and the target becomes unsatisfiable. The design
ranks this failure mode 6 and suggests catching it from training logs, which
costs a run. `guards/tracking_frame.py` now asserts it in 0.01 s, and measuring
the cost sharpened it: at 40° of yaw a world-frame reading caps Φ_v at
**0.1613**, not the ~0.5 the note estimated.

**A diagnostic that lied once every 250 steps.** The first `step()` built its
`info` dict by re-reading the command *after* the resample, so the logged
command disagreed with the one the reward was scored against on exactly the
resample boundaries — correct on 99.6% of steps. That dict is the input to the
command-gain regression, which is the detector for a policy ignoring its
commands entirely. A smoke test caught it at step 217 of the first rollout.

### The run

`hound_track_desert_s0`, seed 0, 1,500,000 steps, launched 08:03 CDT under a
declared 4.25 h ceiling. One seed, so it is a **probe** and cannot claim an
effect — but the question it asks is structural (*does standing earn near-zero
on driving commands?*) rather than comparative, which is what a probe can
legitimately answer.

Early behaviour, at 44k steps: `ep_len_mean` falling 270 → 232 → 186 and
`ent_coef` collapsed to 0.000726. Both are flagged, neither is yet diagnostic —
the design's own spiral detector asks for a sustained collapse below 400 across
the first 300k steps, and this is 3% of the way in.

## How the prediction did

Cycle 005 made four calls on `hound_pd_desert_s1`. **All four resolved true**,
which is itself the finding — the interesting one is claim C.

| claim | p | outcome |
|---|---|---|
| `final_ep_rew_mean` in [950, 1250] | 0.65 | **true** — 1052.31 |
| greedy beats zero-action, ratio > 1.00 | 0.70 | **true** — ×1.1937 |
| ratio clears the 1.18 margin | **0.20** | **true**, by 0.6 standard errors |
| two-seed spread ≥ 50 points | 0.60 | **true** — 91.55 |

Eleven predictions are now resolved at a 73% hit rate. At n = 7 the reliability
table was noise; at n = 11 one band is not: **all five claims in the 60–80%
band have come true.** The loop is systematically under-confident when it is
fairly sure, and this episode's own predictions were written up rather than
hedged toward the middle on that basis.

Claim C is the counterweight and the reason the Brier score got worse (0.079 →
0.142) in a cycle that went 4/4: a 20% call landing true is a bad prediction,
not a good one. The 0–40% band now reads over-confident across four claims.
