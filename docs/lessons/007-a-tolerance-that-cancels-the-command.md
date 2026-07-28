# 007 — When a tolerance scales with the command, the command cancels

**One sentence:** If you score a robot's error against a tolerance that is
*proportional* to what you asked for, a machine that does nothing scores the
same number for every possible command — so no command is ever hard.

Assumes [001 — what a reward function is](001-what-a-reward-function-is.md) and
[003 — why two rewards should be multiplied](003-add-or-multiply.md).

## The idea

The hound is given a command — say *"turn at 0.4 rad/s"* — and paid each step by
a **kernel**: a function of its error that is 1 when perfect and falls toward 0
as the error grows. Ours is `K(e) = exp(-(e/α)²)`. The **tolerance** α is the
only free parameter: it is how wrong you may be before the score collapses.

A *fixed* α was the old design, and it is unfair in an obvious way. Missing by
0.1 rad/s on a 0.6 rad/s turn is good driving; missing by 0.1 rad/s on a
0.2 rad/s turn is barely turning at all. A fixed tolerance cannot tell those
apart. So the new design made the tolerance **proportional to the command**:
α_w = k·|ω_cmd|, a *relative* tolerance — "be within k× of what I asked."

That is a real improvement, and it has a trapdoor. A machine doing nothing has
an error equal to the whole command, and then the command divides itself out.

## The math

`e` = yaw error, rad/s. `ω_cmd` = commanded turn rate, rad/s. `k` = unitless
tolerance fraction. A do-nothing machine's own yaw drift measures
**0.0177 rad/s** (stop cell, n=20, seeds 1000–1019,
`research/measurements/track_rel_zero_action.json`) — 17–34× smaller than the
0.3–0.6 rad/s we command, so `e ≈ |ω_cmd|`. Substitute:

    K = exp(-( |ω_cmd| / (k·|ω_cmd|) )²) = exp(-1/k²)

ω_cmd is gone. The score is **scale-free**: 0.169013 at every turn command when
k = 0.75, 0.018316 at every turn command when k = 0.5 — a **9.2×** difference,
and no dependence on the command in either case.

Physically: the tolerance grows exactly as fast as the error it is grading, so
a harder command is not harder. Ask for a slow turn or a violent one; a machine
that stands there is graded identically.

Multiply by the speed factor — a stander is correctly *not* moving on a
turn-in-place command, measured φ_v = **0.944955** — and at k = 0.75 doing
nothing collects **0.1597/step on every turn command**. The trained policy's
best straight-drive cell earned **0.14685/step**. Standing still would have
out-earned driving, and out-earned the old reward's **0.04474/step** for the
same free lunch. At k = 0.5 it is 0.01731/step. Across every command that asks
for motion, the stander's income is **0.00918/step** at k = 0.5, against
**0.02571/step** at k = 0.75 on the same draws — the constant alone cuts what
doing nothing pays by **2.8×**.

Be careful comparing that to the old reward's **0.06501/step**: *that* figure is
a six-cell drive-grid rate, not a mixture rate, so dividing one by the other is
not like-for-like. The honest same-commands comparison is the six-cell
`drive_grid_track`, **0.0652 → 0.01397, a 4.7× cut**. Two averages over two
different sets of commands is exactly the kind of thing that reads as one number
if nobody prints both, which is why the script prints both.

## Where it bites here

`src/bestiary/envs/hound_track_rel.py`, the comment above `BETA_W`. The first
value written there was 0.75, and the derivation that produced it asked for
*"the largest k that still clears the freeride cap"*. **That is the bug, and it
is the transferable half of this lesson: a safety cap is a ceiling, not a
target.** Every increase in k hands the do-nothing machine income at exp(-1/k²),
and a bound you tune yourself up against has no margin left to protect you.
k = 0.5 clears the same cap with room.

A run launched at 03:48:54 on 2026-07-28 with k = 0.75 and was killed 114
seconds later, when the arithmetic above was checked.

Arithmetic: `scripts/007_scale_free_tolerance.py`.

## If you want to go deeper

[`../theory/command-tracking-reward.md`](../theory/command-tracking-reward.md) —
where the freeride cap comes from and why a tolerance must be bounded from both
sides.
