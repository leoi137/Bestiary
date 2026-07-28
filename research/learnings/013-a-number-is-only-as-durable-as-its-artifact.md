---
triggers: [comparison, metric_added, resume, long_run, refactor]
guard: track-length-bias — src/bestiary/guards/track_length_bias.py (the metric half only; the artifact half is UNGUARDED, see below)
last_confirmed: 2026-07-28
---

# 013 — A number is only as durable as the artifact it was computed from

**Date:** 2026-07-28 · **From:** cycle 010, re-measuring `hound_track_desert_s0`
against `research/anomalies.jsonl` row 20
**Robot:** hound

## What we believed before

That the number rule was sufficient. The rule — *no number enters the record
unless code computed it* — was written to kill prose arithmetic, and it did.
Every figure in this folder can be traced to a script, a run log, or a
committed measurement JSON, and that traceability is what lets a later cycle
build on a number without re-deriving it.

The belief underneath it was: **a number that came from code, run on a named
artifact, is reproducible.** That is what "computed" was taken to mean. It is
the reason we write `research/measurements/*.json` at all.

It was a reasonable belief, and it was already known to be false in one
specific place. Cycle 007 recorded — `anomalies.jsonl` row 19 — that
`*_best.zip` is **overwritten in place** by the training loop every time an
eval beats the previous best, and it went further: it observed that hashing the
file before and after a measurement is insufficient *by construction*, because
the file can change between the measurement and either hash. So the defect was
not merely known, it was known to be unfixable by the obvious fix.

The gap was not knowledge. It was that a **known-mutable artifact was allowed
to underwrite a published conclusion**, and the fix — copy the checkpoint
somewhere immutable before measuring it — was never built.

## What happened

`anomalies.jsonl` row 20 (2026-07-27) recorded that `track_eval`'s `mean_track`
is **length-biased**: it is a mean over episodes of each episode's own per-step
mean, so an episode that crashes at step 300 and one that survives 1000 steps
carry equal weight even though they bank income in a 1:3.33 ratio. The headline
numbers were policy **0.0955** against zero action's **0.0650** — ×1.47 raw —
collapsing to **0.0644** vs **0.0650** (×0.992) once normalised to a common
1000-step horizon. The conclusion carried forward into `STATE.md` and the cycle
notes was: *length-normalised, the policy does not beat zero action on tracking
at all.*

Cycle 010 built the length correction properly (`track_eval` now reports
`mean_steps`, `track_income`, `reward_track` and `track_per_horizon`) and
re-measured under the **identical protocol**: same env `HoundPDTrackDesert-v0`,
same six-cell drive grid, n = 20 per cell, seeds 1000–1019, `deterministic=True`.

All numbers below are printed by `research/scripts/track_length_bias_math.py`
from `research/measurements/track_length_bias_s0.json` and
`..._s0_best.json`.

**Zero action reproduced bit-for-bit: `0.06500765521917337`.** That is the
control that matters. It proves the env, the terrain, the grid, the seeds and
the protocol did not move between the two measurements.

The policy did not reproduce:

| quantity | row 20 (2026-07-27) | today, `ant_sac.zip` | today, `ant_sac_best.zip` |
|---|---|---|---|
| raw `mean_track`, policy | 0.0955 | 0.0718 | 0.0695 |
| raw `mean_track`, zero | 0.0650 | 0.0650 | 0.0650 |
| raw ratio | ×1.4692 | ×1.1038 | — |
| normalised ratio | ×0.9908 | **×1.0321** | — |
| length bias on the policy arm | — | **+6.9%** | **+2.4%** |

Then the source was found. Row 20's 0.0955 came from
`research/measurements/hound_track_desert_s0_midrun_950k.json`, measured on
`ant_sac_best.zip` at the 950k-step mark, when that checkpoint crashed **71
times out of 120** drive episodes. That JSON was committed at **10:58:13** on
2026-07-27. The checkpoint file's mtime is **2026-07-27 11:07:01.732** — the
training run overwrote it in place **nine minutes later** with a better-scoring
policy. (Both timestamps are recorded in `anomalies.jsonl` row 26.)

So the ×1.47 was never a property of the metric. It was a property of **one
vanished checkpoint's crash rate**, and that checkpoint no longer exists on
disk in any form.

Two consequences, and the second is worse than the first:

1. **The published conclusion inverts.** Length-normalised, on both surviving
   checkpoints, the policy marginally *wins* on tracking: ratio **1.0321**,
   replicated at **1.0268** on an independent seed block (n = 10/cell, seeds
   5000–5009, `research/measurements/track_length_bias_s0_seed5000.json`).
   Not "does not beat zero action at all."
2. **Row 20's 0.0644 is permanently unverifiable.** The 950k JSON predates the
   `mean_steps` / `track_income` / `reward_track` fields, so 0.0644 was derived
   by hand from fields that no longer suffice to recompute it, on an artifact
   that no longer exists. It cannot be confirmed and it cannot be refuted. It
   is simply stranded.

## Why it happened

Three separate things had to line up, and each is worth seeing on its own.

**One — the checkpoint is a moving target with a fixed name.** `*_best.zip` is
saved whenever an eval beats the previous best. That is a good policy-selection
rule and a terrible provenance rule, because the *name* is what the measurement
records and the *contents* are what the measurement used. A JSON that says
`"checkpoint": "ant_sac_best.zip"` is naming a slot, not a thing. Nine minutes
of a live run separated the record from what it recorded.

**Two — the surviving evidence looked like agreement.** Zero action reproduced
to the last digit, which is exactly what a healthy re-measurement looks like.
If the whole measurement had drifted, the discrepancy would have been read as
"something moved" and chased immediately. Because the control was perfect, the
policy discrepancy looked like a fact about the policy rather than a fact about
the file — and the natural reading was "the metric was wrong", which is the
conclusion row 20 had already primed.

**Three — the bias is real but small, and it is carried by one cell.** This is
the part that makes the correction sharp rather than a wash. The length bias
does not spread across the grid. On `ant_sac.zip` it is carried entirely by the
cell `(0.5, 0.0, 0.4)` — 7 crashes in 20 episodes, mean length 713 steps. On
`ant_sac_best.zip` it is carried entirely by `(0.8, 0.0, 0.0)` — 1 crash in 20,
mean 959.5 steps. Over the five non-crashing cells the measured bias is
**exactly 1.000000 on both checkpoints**, which it must be: every episode there
runs the full 1000-step horizon, so there is no length difference to bias
anything.

Which means the bias magnitude is a **crash-rate reading**, not a metric
property. A checkpoint crashing 71/120 has an enormous bias. A checkpoint
crashing 7/120 has +6.9%. One crashing 1/120 has +2.4%. Row 20 measured the
first and reported it as though it described the instrument.

Note what does *not* follow: that the crash rate at 950k was itself wrong. It
was correctly measured on a real policy. The failure is narrower and nastier —
the artifact that made the number meaningful stopped existing, so no later
cycle can tell a correct measurement of a bad policy apart from a mistake.

## The math

Two quantities, on the same rollouts.

`mean_track` — the per-episode mean of each episode's own per-step tracking
score, over `N` episodes:

    mean_track = (1/N) · Σᵢ [ (1/Lᵢ) · Σₜ₌₁..Lᵢ Φ_v(t)·Φ_w(t) ]

`track_per_horizon` — the income actually banked, per fixed horizon:

    track_per_horizon = (1/N) · Σᵢ [ (1/H) · Σₜ₌₁..Lᵢ Φ_v(t)·Φ_w(t) ]

**Symbols.** `Φ_v` speed-match score, unitless, ∈[0,1]. `Φ_w` heading-match
score, unitless, ∈[0,1]. `t` a control step at 20 Hz, so one step is 0.05 s.
`Lᵢ` the length of episode `i` in steps (≤ H, shorter when the robot crashes).
`H = 1000` steps, the fixed horizon. `N = 20` episodes per cell. Both
quantities are unitless and both are per-step rates; they differ only in
whether the denominator is *how long the episode lived* or *how long it was
supposed to live*.

Define the bias as their ratio:

    bias = mean_track / track_per_horizon

**Worked on the grid aggregate for `ant_sac.zip`:**

    bias = 0.071753 / 0.067094 = 1.0694        (+6.9%)

and on the zero-action arm, which never crashes (all Lᵢ = H = 1000):

    bias = 0.065008 / 0.065008 = 1.0000        exactly, by construction

**Now the tempting identity, and where it breaks.** If every episode in a cell
tracked at the same rate, the two definitions would differ only by the fraction
of the horizon lived, giving `bias = H / L̄`. Test it on the cell that carries
the bias, `(0.5, 0.0, 0.4)` on `ant_sac.zip`, where `L̄ = 713.0`:

    predicted:  H / L̄        = 1000 / 713.0        = 1.4024
    measured:   0.05087 / 0.02292                  = 2.2196

**The identity is false**, and by a wide margin — 2.22 against 1.40. The same
check on `ant_sac_best.zip`'s crashing cell `(0.8, 0.0, 0.0)` gives measured
1.1641 against predicted 1.0422. The reason is that the episodes that crash are
not the surviving episodes cut short: they are *different episodes*, tracking at
a different rate, and here the short ones scored **higher** per step while alive.
`H / L̄` is a lower bound on the bias in this data, not an equality, and it must
not be quoted as one.

**Physically:** `mean_track` answers *"how well does it track while it is still
upright?"* and `track_per_horizon` answers *"how much tracking reward does it
actually bank?"* Both are legitimate; only the second is what the reward pays.
The gap between them is not a fixed correction factor — it depends jointly on
how often the policy crashes *and* on how the crashing episodes' tracking rate
compares to the surviving ones.

## What to do next time

**Before measuring a checkpoint, freeze it.** The concrete fix: copy the file
to a content-addressed path (`runs/<name>/frozen/<sha256>.zip`), measure *that*
copy, and write the sha256 into the measurement JSON alongside the filename.
Then a name in the record resolves to bytes, not to a slot, and a later cycle
can either reproduce the number or state exactly which artifact it cannot find.
This is why hashing before-and-after is not enough — the immutability has to
come from the copy, not from a check around a mutable read.

**This is not built.** It is recorded as an open anomaly dated 2026-07-28
(`research/anomalies.jsonl` row 26) and nothing enforces it today. Do not read
this learning as describing a solved problem.

**Re-measure a control arm alongside every re-measurement.** Zero action
reproducing to `0.06500765521917337` is the only reason this was diagnosable at
all. A discrepancy with no control arm is uninterpretable: it could be the
protocol, the env, the terrain, the seeds, or the artifact, and there is no way
to tell them apart after the fact.

**Report a bias as a bias, per checkpoint, with its crashing cells named.** The
grid-level number `1.0694` is nearly useless on its own; `+6.9%, all of it from
(0.5,0,0.4) at 7/20 crashes, 1.000000 over the other five cells` is the finding.
Related: `learnings/008` (the best checkpoint is the luckiest episode) and
`learnings/011` (which quotes the ×1.47 and now needs reading against this file).

**Guard status, stated honestly.** The *metric* half of this is enforced:
`track-length-bias` (`src/bestiary/guards/track_length_bias.py`, 7 assertions,
fast tier) asserts the arithmetic on synthetic episodes whose answers are known
by construction — that the three aggregates agree exactly when nothing crashes,
that halving survival at a fixed rate leaves `mean_track` untouched and halves
`track_per_horizon`, that the horizon is read from the env rather than
hardcoded, that it is recoverable from a committed JSON, and that `rollout`
refuses an env with no episode limit instead of guessing one. The *artifact*
half is enforced by nothing at all.

## How we would know this is wrong

- **The 950k checkpoint turns out to be recoverable.** If a copy of that exact
  file exists anywhere still in scope and re-measuring it reproduces 0.0955,
  then row 20 was a correct measurement of a real artifact and the only defect
  is that its provenance was not recorded — a weaker claim than the one made
  here.
- **A third checkpoint of this run shows a large normalised gap.** The
  normalised ratio is 1.0321 on `ant_sac.zip` and 1.0268 on an independent seed
  block, both of which are marginal wins. If a further checkpoint under the same
  protocol comes back well below 1.0, then "the policy marginally wins" is
  specific to these two artifacts and not a correction to row 20's conclusion.
- **The bias turns out to be diffuse on another run.** The claim that the bias
  is carried by exactly one cell rests on the non-crashing cells measuring
  exactly 1.000000, which is arithmetic and cannot fail. But if a policy crashes
  in four cells at once, the "name the cell" advice degrades into "name four
  cells" and the sharp version of this lesson does not transfer.
- **Freezing the artifact does not stop the recurrence.** If, after the
  content-addressed copy is built, a measurement still fails to reproduce, then
  the mutable checkpoint was not the mechanism and something else in the
  measurement path is nondeterministic.
- **Provisional on one run and one seed for the run itself.** Everything here is
  `hound_track_desert_s0`, seed 0. The artifact lesson is general; the specific
  ratios are a probe.
