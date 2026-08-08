# 0007 — Gentle v5 "smooth-ridge" is the terrain for every future arm and robot

**Date:** 2026-08-07 · **Status:** accepted · **Robot:** spyder, and every machine after it

## The decision

The gentle training terrain is regenerated as **v5**: the same world as v4
(seed 11, same layers, same layout, same 7.4 ft / 2.25 m span) with the crest
mathematics changed so the ground can physically exist. All future training
arms, and all new robots that train on the gentle track, use
`assets/terrain/gentle_v5_hfield.bin`. **v4
(`assets/terrain/gentle_hfield.bin`) stays committed and untouched** — every
Spyder policy to date trained on those exact bytes, and their runs, videos and
grid evals are only interpretable against that ground. Nothing re-points the
existing env cfgs in this decision; the switch to v5 happens the first time a
NEW arm launches, because terrain is a one-way door and swapping it under a
trained lineage would silently invalidate every comparison.

## Why

v4's dune and mountain layers were ridged fields — `1 - |noise|` puts a C0
knife-edge at every crest, and `^1.5`/`^1.8` sharpen it further. Measured on
the 0.5 m-smoothed surface, crests reached **47°** — past any angle of repose,
ground that loose material cannot hold. The defect was invisible in the
stretched preview panels and obvious the moment the robot was drawn to scale
(`assets/terrain/gentle_terrain_robot_compare.png`; robot silhouette taken
from `spyder12.xml`: 6.6 ft / 2.02 m leg span).

Four candidates were measured on the identical world before choosing
(`assets/terrain/gentle_v5_comparison.png`):

| candidate | slope P99 | max | crest radius | foot-step P99 |
|---|---|---|---|---|
| v4 | 34.4° | 47° | 6.2 ft (1.9 m) | 5.5 in (14 cm) |
| **C1 smooth-ridge (chosen)** | 33.5° | 43° | 7.2 ft (2.2 m) | 3.7 in (9.4 cm) |
| C2 billow | 32.0° | 37° | 7.4 ft (2.3 m) | 2.4 in (6.1 cm) |
| C3 heavy smooth | 24.4° | 33.5° | 16.8 ft (5.1 m) | 3.1 in (7.9 cm) |

C1 was chosen at robot scale: it keeps v4's character and difficulty and
removes only the physically impossible crests. C3 over-flattens; C2 loses
footstep texture for little slope gain.

## What v5 is, mechanically

`generate.build_height_m` gained two knobs whose defaults take the original
code paths byte-identically (terrain-spec oracle 41/41 before and after):
`ridge_eps=0.30` rounds every `1-|f|` crease into `1-sqrt(f²+eps²)`, and
`fine_additive=True` makes the fine ridged layer additive — the multiplicative
form re-sharpened every crest it decorated. `terrain/gentle.py` then enforces
a **36° repose cap** (`_repose_limit`, cone erosion to convergence) after the
span rescale, re-checks it loudly, and refuses to ship a field that violates
it. The committed v5 measures (printed by `--stats`): body slope P50/P90/P99
= 12.1/23.1/29.8°, foot-step P99 2.2 in (5.7 cm), span 2.25 m exact.

## What this does NOT claim

v5 is walkable-but-hard ground with a natural power-law spectrum; it is not a
planetary-realism claim. It still has no discrete rocks, no craters, and no
ledges — the dominant hazards at this robot's scale on real regolith — because
a continuous heightfield cannot express them.

## Reversal triggers

- **Too easy:** a v5-trained policy saturates the terrain curriculum (mean
  level pinned at max with falls ≈ 0). Then difficulty returns as *discrete
  obstacle geometry* (rocks first), not as sharper heightfield crests — the
  47° crests were rejected as unphysical, not as too hard.
- **Cap artifact:** the faint axis-aligned plateau texture the 36° cap leaves
  in the preview hillshade proves visible in contact behaviour or the Kit
  viewer. Then the cap moves from post-hoc erosion into the layer amplitudes.
- **v4 lineage ends:** once no live policy traces to v4, a cycle may retire
  the v4 files to keep the asset directory single-versioned.
