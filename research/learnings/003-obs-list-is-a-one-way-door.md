---
triggers: [obs_change, resume, long_run, new_robot]
guard: checkpoint-width
last_confirmed: 2026-07-26
---

# 003 — Changing the observation list throws away every checkpoint

**Date:** 2026-07-25 · **From:** planning `CORE_PLAN.md`

## What happened

Not a failure yet — a trap we spotted before walking into it. The
`envs/spyder_env.py` docstring already flags it: adding terrain height samples
to the observation "would change the obs space and orphan the flat-world
checkpoint."

The same applies to the velocity/heading command we need for `ROADMAP.md`
Step 1.

## What we learned

The network's first layer has a fixed input width. Add one number to the
observation and **no existing checkpoint loads at all** — not degraded, just
incompatible. It is a one-way door.

There are only three things that force a from-scratch retrain:

1. the reward numbers
2. the observation list
3. the terrain/task set

Everything else — more steps, new terrain difficulty, new hyperparameters — can
be done on the same model.

## What to do next time

**Decide the observation list once, with slack, and fill unused slots with
zeros.** If we know commands and height samples are coming, put them in the obs
now as zeros. Then turning them on later is a config change, not a retrain.

Same idea for the terrain set: list all types up front, leave the unused ones at
zero difficulty.
