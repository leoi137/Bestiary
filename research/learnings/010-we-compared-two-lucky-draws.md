---
number: 010
title: We compared two seeds through the checkpoint we had already proven not to trust
date: 2026-07-27
from: runs/hound_pd_desert_v0, runs/hound_pd_desert_s1, research/scripts/checkpoint_selection_spread.py
robot: hound
supersedes: none
extends: 008-best-checkpoint-is-the-luckiest-episode.md
guard: src/bestiary/record/greedy_eval.py (both checkpoints measured by default)
triggers:
  - comparison
  - metric_added
last_confirmed: 2026-07-27
---

# 010 — We compared two seeds through the checkpoint we had already proven not to trust

**Date:** 2026-07-27 · **From:** `hound_pd_desert_v0` vs `hound_pd_desert_s1`
**Robot:** hound

## What we believed before

That the PD arm's two training seeds differed by **91.55 return points**, and
that seed 1 had become the first hound policy in this record to clear the
project's 1.18 advisory margin, at ×1.1937.

Both numbers came from a careful measurement, and that is the point. They used
one instrument (`record/greedy_eval.py`), one protocol, 60 deterministic
episodes, the *same* 60 seeds on both arms, and a zero-action control measured
live rather than quoted. Every methodological rule this project had written
down was followed. The measurement was even repeated to check the tool was
deterministic, and it was — byte-identical.

The belief was reasonable because the *comparison* was clean. What nobody
re-examined was the **object being compared**.

## What happened

Both readings were taken from `ant_sac_best.zip`. `learnings/008` — written
two cycles earlier, by this same loop — establishes that this file is selected
by `argmax` over **one-episode** evaluations, so on a bimodal policy it is
reliably a snapshot caught in its good mode. It is the luckiest episode, not
the better policy.

Running the identical instrument, on the identical 60 seeds, against the
checkpoints that were **not** selected by argmax (`ant_sac.zip`, the final
weights):

| checkpoint | seed 0 | seed 1 | spread | clears 1.18? |
|---|---|---|---|---|
| `ant_sac_best.zip` | 1049.10 | 1140.65 | **+91.55** | seed 1 only |
| `ant_sac.zip` | 1089.05 | 1082.18 | **−6.87** | **neither** |

The sign of the spread flips, the magnitude collapses, and the 1.18 clearance
disappears. Paired over the same 60 seeds the difference on the final
checkpoints is **−6.87 ± 44.19** (t = −0.16, 95% CI [−93.49, +79.75]) — a
confidence interval containing zero and nearly two hundred points wide.

**Checkpoint selection alone moved the quantity we called "the seed spread" by
98.42 points**, which is more than the spread itself.

## Why it happened

Not because the argmax is subtle. Because **the selection and the comparison
were reasoned about at different times, by different rules.**

`learnings/008` is filed under *how to read a single policy's score*. The
question being asked here was *how do two seeds differ* — a comparison
question, which felt like a different topic. The lesson's trigger list did not
name it, the guard that enforces 008 (`eval-sampling`) checks that a ledger row
records its sample size and says nothing about which checkpoint produced it,
and `greedy_eval` defaulted to `_best.zip` silently. So every mechanism that
existed to carry 008 forward was pointed at the wrong moment.

The deeper mechanism is that **argmax selection is a bias that grows with how
bimodal the policy is, and both these policies are extremely bimodal.** 58 of
seed 1's 60 episodes score ~1170; the two failures score 191 and 376. The mean
is therefore almost purely a crash counter, and `_best.zip` is exactly the
checkpoint chosen for having, on one draw, not crashed.

The two policies' crash counts under the best checkpoint were **8 and 2** — a
difference of six. Under the final checkpoint they are **6 and 7** — a
difference of one, in the other direction. The entire "seed effect" was a
selection artifact in the crash count.

This is the same shape as `learnings/001`: a quantity was verified in one
regime, the regime changed, and nobody re-checked. There the regime change was
flat ground → terrain. Here it was single-policy score → two-policy
comparison.

## The math

The mean of a bimodal policy is a crash counter. With `n` episodes, `c` of
which crash, mean return `g` in the good mode and `b` in the crash mode:

```
mean = ( (n − c)·g + c·b ) / n
```

- `n` = 60 episodes (dimensionless)
- `c` = crashes (dimensionless)
- `g` = 1170.20, mean return of the 58 non-crashed episodes (reward units)
- `b` = 283.54, mean return of the 2 crashed episodes (reward units)

Differentiating with respect to the crash count gives what one crash is worth:

```
d(mean)/dc = (b − g) / n = (283.54 − 1170.20) / 60 = −14.78 per crash
```

**One extra crash in 60 episodes moves the reported mean by 14.78 points.** So
the 91.55-point "seed spread" is

```
91.55 / 14.78 = 6.2 crashes' worth
```

and the observed crash difference under `_best.zip` was exactly **6** (8 vs 2).
The spread is not a partial explanation of the crash difference — it *is* the
crash difference, to within a rounding error.

Physically: we were not measuring how well two policies walk. We were measuring
how often each fell over, on a checkpoint chosen for not having fallen over.

Arithmetic: `research/scripts/checkpoint_selection_spread.py`.

## What to do next time

**Quote both checkpoints or neither.** `record/greedy_eval.py` now measures
both by default and prints the selection delta; taking one is `--best-only`,
whose help text says what you are asking for. On this very run the delta at
n=3 is ×1.225 → ×0.893.

**A comparison needs its object pinned as carefully as its protocol.** The
protocol here was faultless. Ask *what am I comparing* with the same
suspicion as *how am I comparing it*.

**When a mean is a crash counter, report the crash count beside it.** Better,
report the median: the medians here are 1175.58 and 1170.10, differ by 5.5
points, and would never have suggested a seed effect at all.

**A lesson that keeps being rediscovered needs a mechanism, not another
retelling.** 008 was written, taught (`docs/lessons/004`), and guarded, and was
still violated eight commits after being taught — by the cycle that taught it.
The fix that worked was changing a default.

## How we would know this is wrong

**If the seed spread reappears on the final checkpoints at ≥3 seeds.** This is
n=2 seeds and is a **probe**, not an effect — the project's own seed rule wants
≥3 per arm. The claim here is narrow and provisional: *the specific 91.55-point
figure was a selection artifact*, which is measured. It is **not** the claim
that the PD arm has no between-seed variance. The CI on the final-checkpoint
difference is [−93.49, +79.75], which is far too wide to exclude a real effect
of the size originally claimed.

**If `ant_sac.zip` turns out to be the biased one.** The final checkpoint is
whatever the weights happened to be at the last step, which on a policy still
oscillating is also a draw — just an unselected one. If a third checkpoint
(say, an average over the last N evals) disagreed with both, this lesson would
need rewriting rather than retiring.

**If the crash difference is itself the real seed effect.** 8 vs 2 crashes
under one checkpoint and 6 vs 7 under another is consistent with noise, but a
proper test at ≥3 seeds could show crash rate genuinely differs by seed — in
which case the mean was measuring something real, just very inefficiently.
