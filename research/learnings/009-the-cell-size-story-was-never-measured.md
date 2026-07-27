---
number: 009
title: The creep is real, the cell-size explanation for it was never measured
date: 2026-07-27
from: research/scripts/creep_vs_grid.py, research/measurements/tracking_noise.json
robot: hound
supersedes: none
extends: 005-standing-check-caught-it-on-a-second-robot.md
guard: src/bestiary/guards/spawn_pad.py
triggers:
  - new_terrain
  - proposing a terrain regeneration, or any change to GRID
  - explaining the hound's passive backward creep
  - citing a physical mechanism that no script in the repo measures
last_confirmed: 2026-07-27
---

# 009 — The creep is real, the cell-size explanation for it was never measured

**Date:** 2026-07-27 · **From:** `research/scripts/creep_vs_grid.py`
**Robot:** hound

## What we believed before

Standing on the desert heightfield with every motor at zero, the hound drifts
backwards. On a flat plane, under the identical model, it barely moves. So the
drift comes from the heightfield, and `robots/hound/build.py` gave it a
mechanism:

> MuJoCo collides a heightfield by testing the geom against a prism per terrain
> cell, and the desert's cells are 7.82 cm across — almost exactly the wheel's
> 8.5 cm radius. A wheel is therefore always straddling a cell boundary,
> resolving against two prisms whose contact normals do not quite agree, and
> the residual is a small net backward push.

This is a good explanation. It is specific, it names a real MuJoCo
implementation detail, it explains why a plane behaves differently, and it
comes with a numerical coincidence (7.82 vs 8.5 cm) that feels like evidence.
It also came with a fix — regenerate at `GRID=2048`, putting four cells under
each wheel — described as *"the real fix"*. Four places in the repo repeat it:
`robots/hound/build.py`, `ROADMAP.md:46`, `research/CORE_PLAN.md:317`, and
[`005-standing-check-caught-it-on-a-second-robot.md`](005-standing-check-caught-it-on-a-second-robot.md):49.

**Why holding it was reasonable.** The belief was supported by a genuine
controlled experiment — swap the hfield for a plane at the same height and the
drift vanishes — and that experiment really does implicate the heightfield
collider. The error was not in the evidence but in the *step after* it: from
"the collider does it" to "the collider does it **via cell size**" is an
additional claim, and that second claim was never measured. It inherited the
credibility of the first.

This is the transferable shape: **a well-evidenced finding acquires an
unevidenced mechanism, and the mechanism then gets treated as though the
evidence covered it.** The tell is that the mechanism came with a *remedy* —
and the remedy, not the mechanism, is what people then acted on.

## What happened

`creep_vs_grid.py` rolls 20 zero-action episodes of 1000 steps per arm, same
seed both arms, heightfield injected in memory so nothing on disk changes:

| grid | cell | cells per wheel radius | x displacement |
|---|---|---|---|
| 1024 | 7.812 cm | 1.09 | **−1.7761 ± 0.0276 m** |
| 2048 | 3.906 cm | 2.18 | **−1.7641 ± 0.0066 m** |

Paired difference, `2048 − 1024`: **+11.99 ± 26.18 mm** over 20 seeds — 0.675%
of the creep.

**Halving the cell size does not remove the creep.** Removing it would take
+1776 mm; the measurement buys +12 mm.

## Why it happened

The mechanism was never running where this machine stands.

`terrain/generate.py` multiplies the composed field by a blend that is exactly
zero inside a 2.5 m radius, to give the robot a flat place to spawn. Measured
on the pad, in the metric field *and* in the compiled array MuJoCo actually
triangulates:

    max|h|         0.000e+00 m
    peak-to-peak   0.000e+00 m     over 3212 cells (1024) / 12876 cells (2048)

Not "nearly flat". Exactly flat — every cell holds the identical value. And
across all 40 episodes, every contact point landed within **2.034 m** of the
origin (furthest body origin 2.175 m), comfortably inside the 2.5 m disk.

On exactly flat ground, every prism's normal is `+z` by construction, at *any*
cell size. Two prisms whose normals disagree cannot exist there. **Four cells
under a wheel are four cells of the same plane as one.** The straddling story
describes something that may well happen out on the dunes, but the creep is
generated entirely on ground where the mechanism is unavailable.

## The math

The claim "halving the cell fixes it" predicts the creep goes to roughly zero.
Write the creep as displacement per episode:

    Δx = v̄ₓ · T

where `v̄ₓ` is mean forward velocity (m/s, negative = backward) and `T` is
episode duration. At 20 Hz control over 1000 steps, `T = 1000 × 0.05 s = 50 s`.
Measured at GRID=1024:

    Δx = −0.03553 m/s × 50 s = −1.7765 m

against a directly measured displacement of −1.7761 m — consistent to 0.4 mm,
which is the cross-check that the velocity and the displacement describe the
same thing.

The fix hypothesis predicts `Δx(2048) ≈ 0`, i.e. a paired change of
`+1.7761 m`. The measured change is:

    δ = Δx(2048) − Δx(1024) = −1.7641 − (−1.7761) = +0.0120 m

As a fraction of the change a fix requires:

    0.0120 / 1.7761 = 0.00675  →  0.7%

Physically: cutting the cell size in half closed under one percent of the gap
to "no creep", so whatever is pushing the robot backwards does not care how
finely the ground is diced.

## What to do next time

1. **`GRID=2048` is off the table as a creep fix.** It may still be wanted for
   contact fidelity once a policy actually drives out onto sloped ground, but
   that is a different argument needing its own evidence — and it is expensive,
   because the regenerated desert correlates with the committed one at only
   **+0.061** (`research/scripts/compare_terrain_grids.py`). It is a different
   desert, not a sharper one, so it invalidates every terrain-specific number
   a moving policy produced.
2. **The creep is unexplained again.** `build.py`'s hfield-versus-plane
   experiment still stands and still implicates the collider — what died is the
   cell-size route, not the collider itself. Do not replace one unmeasured
   mechanism with another.
3. **The magnitude in the docs is also wrong, in two directions.** `build.py`
   says ~5 cm/s, [`005`](005-standing-check-caught-it-on-a-second-robot.md):49
   says −3 cm/s. Measured: **−3.55 cm/s**
   (`research/measurements/tracking_noise.json`). The 5 cm/s figure is not
   merely rounded — it is arithmetically impossible alongside the measured
   0.0361 m/s rms planar speed, since rms speed can never fall below |mean
   velocity|.
4. **When a mechanism arrives with a remedy attached, measure the remedy before
   scheduling it.** This one cost two minutes and would have cost a full
   re-measurement of the record.

## What this does NOT show

Preserved deliberately, because a finding that overstates itself is the same
failure it is correcting:

- **Zero-action only.** A policy that *drives* off the pad meets 7.82 cm cells
  with real slope. The scale collision could be entirely real there. Untested.
- **The residual is on a knife edge.** +11.99 mm against a two-standard-error
  bound of 11.71 mm — "distinguishable from zero" by a 2% margin. Another block
  of seeds could flip that. Treat it as real-but-tiny, not as a constant.
- **The finer grid does change the spread**, from ±27.6 mm to ±6.6 mm, a factor
  of 4.2. That is consistent with straddling acting on the contact *set* — how
  many prisms a wheel resolves against, which flips seed to seed — while
  leaving the mean push untouched. It is a **hypothesis this measurement does
  not test**, not a result.
- **One terrain seed (7).** Pad flatness is a construction of `generate.py`
  independent of seed, so this should generalize, but it was not swept.
- **The GRID=2048 arm has no independent oracle.** The 1024 arm is validated
  against two external facts; the 2048 arm rests on the script's own model
  verification. A bug affecting only the finer grid would not be caught.

## How we would know this is wrong

Any one of these overturns it:

- **A policy that drives off the pad shows a grid-dependent drift.** If a
  trained policy operating out at 5–15 m radius drifts measurably less at
  GRID=2048, then cell size does matter for locomotion and this learning is
  correct only for the standing case.
- **The pad stops being exactly flat.** The whole argument rests on
  `max|h| = 0.000e+00` inside 2.5 m. If `generate.py`'s blend changes and the
  pad acquires relief, the mechanism becomes available and the measurement must
  be redone. This is what `guards/spawn_pad.py` exists to catch.
- **A larger seed block turns the +12 mm residual into a real effect.** If the
  paired difference grows with more seeds rather than shrinking toward zero,
  cell size has a small genuine contribution that 20 seeds could not resolve.
- **The instrument is wrong.** The 1024 arm reproduces the persisted `mean_vx`
  to 0.011% and its injected heightfield is bit-identical to the committed
  asset over all 1,048,576 samples. If either check is later shown to be
  measuring the wrong model, everything here goes with it.
