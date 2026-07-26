---
triggers: [new_robot, new_terrain, reward_change]
guard: standing-control
last_confirmed: 2026-07-26
---

# 005 — The standing check caught it again, on a different robot, from scratch

**Date:** 2026-07-25 · **From:** `runs/hound_desert_test150k`

## What happened

Hound (16-DoF wheel-legged dog) trained on `HoundDesert-v0` for 150k steps —
**from scratch**, no warm-start from anything. Ran the standing check from
[[001-flat-reward-breaks-on-terrain]]:

| HoundDesert-v0 | reward | episode length |
|---|---|---|
| trained best (150k) | 109 (probe: 8–161) | dies at 21–141 steps |
| **zero action** | **961** | **survives all 1000** |

Doing nothing scored **9× better**. Forward vs effort on its best episode: +88.7
vs −67.2 (1.3 : 1, against flat ground's 8.7 : 1).

## What we learned

**This isolated the variable.** Spyder's `spyder_desert_v0` had two candidate
causes: a reward built for flat ground, *and* a warm-start across a reward change
that wrecked the critic. We could not tell how much each contributed.

Hound had no warm-start, no stale critic, no annealed entropy coefficient — and
failed the same way. So the reward pathology is **sufficient on its own**. The
warm-start only made Spyder's version worse.

Two robots, two different morphologies, two different training setups, one bug —
and it is in the reward, not in either body.

That is also why the fix is worth sharing across robots: `CORE_PLAN.md` is a
*recipe*, and the recipe is the right unit of reuse even when the policies are
separate.

## What to do next time

Run the standing check on **every new robot and every new terrain**, in the first
30k steps, before spending a night on it. It costs two minutes and it has now
caught the same bug twice.

Second finding, specific to Hound: zero action **drifts −1.5 m per episode**
(~−3 cm/s backward) on the desert — the terrain's 7.82 cm cells match the 8.5 cm
wheels, the limitation already noted in `ROADMAP.md`. Under command tracking,
"commanded zero, actually drifting backward" is precisely what the reward term
measures, so this corrupts the reward directly. Regenerate the terrain at
`GRID=2048` before any serious Hound run.
