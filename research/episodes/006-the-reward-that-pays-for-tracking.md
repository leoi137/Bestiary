# Episode 006 — The reward that pays for tracking

**Date:** 2026-07-27

## Thesis

The hound's reward paid it for existing. A machine that stands still collected
nearly all of it, so the optimizer correctly learned that existence pays and
locomotion barely does. This episode harvests the two-seed measurement of how
big that problem actually is, then replaces the reward with one whose only
positive term is a product of two tolerance kernels against a sampled velocity
command — and launches the first run under it.

> ## ⚠️ Read the refutation section at the bottom before quoting anything here
>
> An independent refuter killed two of this episode's four claims **after** the
> Diagnosis section below was written, and the section is left standing as
> written rather than quietly corrected. The **91.55-point two-seed spread is
> a checkpoint-selection artifact** and the **±41.8 standard error argument is
> wrong**. `research/learnings/010` is the correction. The claim that survives
> is the narrow one: the PD hound beats doing nothing.

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

---

## Refutation — what an independent check killed

The Diagnosis section above was written before the refutation and is left
exactly as written. Two of its claims did not survive.

### Killed: the 91.55-point two-seed spread

It compared two `ant_sac_best.zip` files. `research/learnings/008`, written two
cycles earlier by this same loop, establishes that this checkpoint is selected
by `argmax` over **one-episode** evaluations — the luckiest episode, not the
better policy. Running the identical instrument on the identical 60 seeds
against the checkpoints *not* selected by argmax:

| checkpoint | seed 0 | seed 1 | spread | clears 1.18? |
|---|---|---|---|---|
| `ant_sac_best.zip` | 1049.10 | 1140.65 | **+91.55** | seed 1 only |
| `ant_sac.zip` | 1089.05 | 1082.18 | **−6.87** | **neither** |

Paired over the same seeds, the final-checkpoint difference is **−6.87 ± 44.19**
(t = −0.16, 95% CI [−93.49, +79.75]). The sign flips, the magnitude collapses,
and the 1.18 clearance disappears. **Checkpoint selection alone moved the
quantity by 98.42 points**, more than the spread itself.

The arithmetic is unambiguous: one extra crash in 60 episodes moves the mean by
(283.54 − 1170.20)/60 = **−14.78** points, so 91.55 points is **6.2 crashes'
worth** — and the crash difference under `_best.zip` was exactly 6 (8 vs 2).
The medians differ by 5.5 points and would never have suggested a seed effect.

Full treatment in `research/learnings/010`. `record/greedy_eval.py` now
measures both checkpoints by default.

### Killed: the ±41.8 standard-error argument

The Diagnosis explains episode 004's 1078.2 versus today's 1049.10 as ordinary
sampling error, citing a ±41.8 standard error. That is wrong, and wrong in the
direction that makes an unexplained gap look explained. Measuring the **same
checkpoint** on two disjoint 60-episode blocks (seeds 0–59 and 100–159) gives
**1049.10 vs 1045.76** — a **3.34**-point gap, not 29. Block-to-block
reproducibility of this instrument is 2–3 points, because the block mean is
essentially determined by the crash count and both blocks drew 8.

So the 29-point gap is **not** covered by sampling error, and because episode
004 never recorded which seeds it used, 1078.2 cannot be re-derived from the
record at all. This is now an open anomaly, not a resolved question.

### Survived

**"The PD hound beats doing nothing"** is robust: ratio 1.1947 pooled over 120
episodes, bootstrap 95% CI [1.1619, 1.2189], and it survives the median
(1.2245), conditioning on non-crashed episodes (1.2246), a fresh seed block
(1.1957), and the unselected checkpoint (1.1325).

**"Clears 1.18" does not survive** as anything more than a point estimate:
bootstrap P(ratio < 1.18) = 0.294 at n=60 and 0.163 at n=120. And the 1.18
threshold is applied by `guards/standing.py` to the ledger's
`final_ep_rew_mean`, where `hound_pd_desert_s1` reads ×1.10 and **fails** —
so the Diagnosis silently swapped in a different numerator, which
`greedy_eval`'s own docstring says would change published verdicts and needs an
operator.

**The zero-action baseline survived cleanly**, reproduced independently to
within 1.7 return points on every cell. Two reporting corrections: the
±116.59 is a per-episode SD, not an uncertainty on the mean (the SE is
**±15.05**); and the drive-grid mean is **60.4% one cell** — dropping
(−0.3, 0, 0) takes it from 55.73 to **21.78**.

**The 5× contact-cost finding survived** at 4.92×, but is incomplete: it
explains only 65% of §5's drive-grid error. The other 35% is a second,
independent mistake — §5 applies the DRIVE-*distribution* expectation to the
eval *grid*, whose mean track is 44% higher. And fixing contact moves §5's
standing prediction **up** from 87 to 124: the note under-predicted the
standing floor rather than over-predicting it.

### A guard shipped this cycle was found broken by the same check

`guards/tracking_frame.py` assertion 5 bounds a *standing* machine's freeride,
and used the yaw drift of the **driving** arm (0.127 rad/s) instead of the
standing arm (0.01823). It therefore computed 0.0624 against a 0.16 cap and
passed by 2.6×, while the quantity it claimed to bound is **0.1612** — over
that cap. The theory note's §2 uses 0.968 for that factor, which is the
standing figure; the note had it right and the guard did not. Both constants
are now read from `tracking_noise.json` by arm name, so the substitution is
impossible rather than merely corrected.

### The one-sentence version

This cycle wrote a teaching lesson about not trusting an argmax-selected
checkpoint, and then, eight commits later, compared two seeds through exactly
that artifact.
