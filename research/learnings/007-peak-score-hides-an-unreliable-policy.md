---
triggers: [comparison, metric_added]
guard: ledger-schema
last_confirmed: 2026-07-26
---

# 007 — A peak score hides an unreliable policy

**Date:** 2026-07-25 · **From:** comparing `hound_desert_v0` (torque) with
`hound_pd_desert_v0` (PD position targets) · **Robot:** hound

## What happened

Two runs of the same robot on the same terrain, differing only in how the
policy commands its legs. Compared on the number the repo had been recording
as *the* result — `best_eval_return`, the high-water mark:

| | torque, 3.75M steps | PD, 1.0M steps |
|---|---|---|
| **peak eval** | **1218.3** | 1176.7 |

Read that alone and the conclusion is clear: the torque run is better, and the
change was not worth making.

Then look at every eval, not just the best one:

| | torque | PD |
|---|---|---|
| peak eval | 1218.3 | 1176.7 |
| **mean eval after 400k** | **887.5** | **1113.1** |
| individual evals after 400k | 1218, 1208, 1207, **523**, 956, **390**, 1205, … | 1175, 1165, 1166, **483**, 1173, 1164, 1177, … |
| crashes (eval < 600) | frequent, throughout 3.75M | 2 of 19 |

The torque policy's 1218.3 is **a number it visited, not a number it held.**
It fell over repeatedly for the entire run — at 3.5M steps it was still
producing evals of 390. The PD policy scores slightly lower at its best and
holds ~1170 almost every time.

A policy that reliably scores 1170 is worth more than one that occasionally
scores 1218 and often scores 500. The metric we were reporting said the
opposite.

## Why it happened

`best_eval_return` is a **maximum over a noisy sequence**, and a maximum is
the order statistic most contaminated by variance. It cannot go down. Every
additional evaluation can only push it up, so it rewards two different things
with the same number: being good, and being evaluated many times while being
erratic.

The torque run had 3.75× the steps of the PD run, therefore more evaluations,
therefore more chances for one lucky rollout to set a high-water mark. Its
peak partly measures *how long we ran it*.

This is survivorship bias with extra steps. We kept the best sample and threw
away the distribution that produced it.

## The math

Let each evaluation be a draw `Xᵢ` from the policy's return distribution, and
let `Mₙ = max(X₁ … Xₙ)` after `n` evaluations.

`Mₙ` is non-decreasing in `n` by construction. Its expectation grows with both
the mean **and** the spread of `X`:

```
E[Mₙ] ≈ μ + σ · aₙ
```

where `μ` is the mean return, `σ` the standard deviation, and `aₙ` a factor
growing roughly like `√(2 ln n)` for light-tailed `X`.

**Both terms increase `E[Mₙ]`.** A policy can raise its peak by getting better
(`μ↑`) *or* by getting less reliable (`σ↑`). The statistic cannot tell those
apart.

Worked with this comparison's real numbers, using the post-400k evals:

```
torque:  μ = 887.5    peak = 1218.3    peak − μ = 330.8
PD:      μ = 1113.1   peak = 1176.7    peak − μ =  63.6
```

The torque run's peak sits **330.8** above its own mean; PD's sits **63.6**
above. That gap is a direct read-out of σ: the torque policy's spread is
roughly five times wider. And `n` differed too — 14 evals versus 19 — so the
two peaks were not even drawn under the same number of trials.

Ranking by `μ` reverses the ranking by peak:

```
peak:  torque 1218.3  >  PD 1176.7      (torque wins by 41.6)
mean:  PD     1113.1  >  torque 887.5   (PD wins by 225.6)
```

**In plain English: the torque policy is not better, it is noisier — and the
statistic we were using pays for noise.**

## What to do next time

**1. Never report a peak alone.** Report the mean and the spread of evals over
the converged region, and the crash rate. `research/ledger.jsonl` already has
`best_eval_return`; it needs `mean_eval_after_converge` and
`eval_crash_rate` alongside, or the ledger will keep encoding this bias into
every future comparison.

**2. Keep the checkpoint selection, change the reporting.** Saving the
best-scoring checkpoint is still right — that is a selection procedure and
picking the best sample is what it is for. The mistake is using that same
number to *compare runs*. Two different jobs, two different statistics.

**3. Never compare peaks across runs of different length.** `E[Mₙ]` grows with
`n`. A 3.75M-step run and a 1M-step run do not have comparable maxima, and
normalising by "per step" does not fix it. Compare at equal `n`, or compare
means.

**4. A high peak with a low mean is a diagnosis, not a disappointment.** The
gap `peak − μ` measures instability directly. The torque run's 330.8 was
visible in the data the whole time and would have flagged the reliability
problem 3M steps earlier than we noticed it.

## How we would know this is wrong

- If the eval distribution turns out to be bimodal for a *substantive* reason
  — say the terrain has two regimes and the policy genuinely solves one — then
  the mean is the misleading statistic and the modes should be reported
  separately.
- If a future run shows a high mean and a low peak, the framing here (peak
  overstates, mean is honest) is too simple and both statistics are needed for
  reasons this lesson does not capture.
- If eval noise turns out to come mostly from the *terrain seed* rather than
  the policy, then σ is a property of the evaluation protocol rather than the
  policy, and the fix is to fix evaluation — more episodes per eval, fixed
  seeds — not to change the reported statistic.

## See also

- `research/episodes/003-pd-result-cheaper-not-higher.md` — the comparison
- [006 — Our regression oracle covered the robot, not the trainer](006-the-oracle-only-covered-the-robot.md)
  — the other lesson from this cycle about a check that reported success while
  missing what mattered
