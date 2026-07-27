# 005 — The fix that fixes nothing

**Date:** 2026-07-27
**Runs:** `hound_pd_desert_s1` (launched, seed 1, 1M steps on `HoundPDDesert-v0`)
**Status of the headline finding:** survived independent refutation — and was
*produced* by it.

---

## Thesis

The hound's reward pays for existing. A policy that stands still scores 955
against a trained 1078, so more than 90% of an episode's return is the alive
bonus and locomotion is a rounding error in the objective. Replacing that
reward with one that pays for *matching a commanded velocity* is the obvious
fix, and this stretch of work exists to test it.

Two things had to happen before that test could mean anything, and this episode
is about discovering that one of them was not what it appeared to be.

## Diagnosis going in

The plan called for regenerating the terrain heightfield at `GRID=2048` before
touching the reward. The reasoning was documented in four separate places and
had never been challenged: terrain cells are 7.82 cm across, the hound's wheel
radius is 8.5 cm, so every wheel permanently straddles a cell boundary,
resolves against two prisms whose contact normals disagree, and collects a
small net backward push. Halving the cell would put four cells under each
wheel and remove the mechanism. It was called "the real fix".

That reasoning is careful, physical, and specific. It is also wrong, and
nothing in the record had ever asked it for a number.

## What happened

**The seed.** The comparison the whole question rests on turned out to have no
control. Every arm in the ledger is a *single training seed*, so a
tracking-reward run compared against the existing baseline would be one draw
against one draw. The torque hound's paired 95% confidence interval against
doing nothing is `[−52.1, +132.9]` — it contains zero, meaning that after 3.75
million steps we cannot say that policy learned to do anything beyond not
falling over. So the first run launched was not the exciting one: it was
`hound_pd_desert_s1`, identical to the existing PD run except `seed=1`, for no
purpose other than turning n=1 into n=2.

**The terrain.** Before regenerating anything, we measured what regenerating
would do. At the same seed, the `GRID=2048` terrain block-averaged down to the
coarse grid correlates with the committed terrain at **+0.0610** (four seeds:
0.024–0.098), with an rms difference of 1.3662 m. It is not a finer sampling of
the same desert. It is a different desert.

The cause is one indexing decision in the spectral synthesis. `generate.py`
draws its random phase array as `(n, n)` and indexes it by **FFT bin** rather
than by physical frequency, so changing `n` hands every drawn phase to a
different wavelength. The phases are not redrawn — the random stream is
bit-identical — they are *re-indexed*. A genuine refinement, holding the
phase-to-frequency map fixed while changing everything else, correlates at
**+0.9997**, which is also the positive control proving the low number is a
property of the terrain and not of the metric.

**The refutation, which is where the episode actually happens.** An independent
review confirmed the number and then killed half of what had been concluded
from it, by measuring rather than arguing.

The claim had been that every existing number stops being a comparator. False,
and specifically so. `reset_model` never randomizes trunk xy: every episode
starts at exactly (0, 0), the spawn pad is exactly `h = 0` out to 2.50 m at
every seed and grid, and a do-nothing policy peaks at 2.18 m radius. It never
leaves bit-identical ground. Rebuilt in memory at both grids:

| | GRID=1024 | GRID=2048 |
|---|---|---|
| zero-action return | 955.58 ± 1.36 | 955.34 ± 0.29 |
| greedy policy return | 930.9 ± 450 | 800.2 ± 457 |

The zero-action baseline does not move — it *is* the recorded 955.5, to 0.2
points. Only numbers a **moving** policy produced are at risk, and even there
the greedy difference is a direction rather than a result at sd ≈ 450.

The terrain's statistics also survive better than claimed: mean slope moves
2.9%, p95 slope 0.05%, both smaller than the ±8% spread between *seeds*. What
does not survive is the single fixed corridor the robot actually drives — along
+x the first obstacle above 0.25 m moves from 13.80 m to 10.83 m.

**And then the finding that mattered more than the one we set out to make.**
The creep the regeneration was proposed to fix does not care about the grid.
Measured properly, 20 episodes per arm at the same seed
(`research/scripts/creep_vs_grid.py`):

| grid | cell | x displacement |
|---|---|---|
| 1024 | 7.812 cm | **−1.7761 ± 0.0276 m** |
| 2048 | 3.906 cm | **−1.7641 ± 0.0066 m** |

Paired difference **+11.99 ± 26.18 mm** — 0.675% of a creep that would need
+1776 mm to remove.

Halving the cell size does not fix it, and the documented mechanism was never
available in the first place: `generate.py` blends the field to exactly zero
inside a 2.5 m radius, the pad measures `max|h| = 0.000e+00` and peak-to-peak
`0.000e+00` over 3212 cells, and every contact point across all 40 episodes
landed within 2.034 m. On exactly flat ground every prism normal is `+z` at any
cell size — four cells under a wheel are four cells of the same plane as one.
A physical story that four documents repeated, that reads correctly, and that
nobody had ever asked for two minutes of measurement.

The instrument was validated rather than assumed: the in-memory GRID=1024
rebuild is bit-identical to the committed asset across all 1,048,576 samples,
and reproduces the persisted `mean_vx` to 0.011%. (This also corrected the
review's own figure, which had the 1024 arm at −1.763 m.)

See [`../learnings/009-the-cell-size-story-was-never-measured.md`](../learnings/009-the-cell-size-story-was-never-measured.md).

## How the prediction did

The four predictions committed before launch all concern `hound_pd_desert_s1`
and are **unresolved at the time of writing** — the run was still training when
this episode was written, and they are registered in `calibration.jsonl` with
`outcome: null` so the next cycle scores them rather than this one.

That is deliberate and worth stating plainly: the headline finding of this
episode was **not** predicted. Nothing in the four claims is about the terrain,
because at the moment the predictions were written the terrain regeneration was
a chore to be skipped, not a question. The finding came from measuring
something in order to justify *not* doing it, which is an uncomfortable thing
to notice about one's own process — the measurement was defensive, and it paid
anyway.

## What this changes

1. `GRID=2048` is **off the roadmap as a creep fix**. It may still be wanted for
   contact fidelity at speed, but that is a different argument that now has to
   be made on its own evidence.
2. The passive creep is **unexplained again**, and the four documents asserting
   the cell-size mechanism need correcting rather than extending.
3. The documented creep magnitude is also wrong: `build.py` says ~5 cm/s, the
   record elsewhere says −3 cm/s, and the measurement is **−3.55 cm/s**. The
   5 cm/s figure is not merely rounded — it is arithmetically impossible
   beside the measured 0.0361 m/s rms planar speed, because an rms speed can
   never be smaller than the magnitude of the mean velocity.
4. A guard now asserts the spawn pad is exactly flat
   (`src/bestiary/guards/spawn_pad.py`), because the entire finding rests on
   that and nothing recorded the dependency.
5. The command-tracking reward is derived and recorded
   ([`../../docs/theory/command-tracking-reward.md`](../../docs/theory/command-tracking-reward.md))
   but **not implemented and not validated** — its predicted separations are
   predictions.
