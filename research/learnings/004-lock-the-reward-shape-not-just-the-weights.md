---
triggers: [reward_change, long_run]
guard: none — 'is this reward shape final?' is a judgement, not a check
last_confirmed: 2026-07-26
---

# 004 — Lock the reward *shape*, not just the weights

**Date:** 2026-07-25 · **From:** planning `CORE_PLAN.md` step 2

## What happened

Caught before it cost us anything. The plan said "lock the reward numbers"
(`ctrl_cost_weight`, `healthy_reward`) in step 2, and "put commands in the
observation" in step 4.

Then a contradiction surfaced: step 2 tunes the reward so that **standing loses**
to walking. Step 4 makes standing still a legitimate command. Both cannot be true
of one fixed reward.

## What we learned

Step 4 was never just an observation change. Following a command means the
forward term itself has to change:

```
"go fast":        reward += 1.0 · v_x
"do as told":     reward += exp(−|v_actual − v_command|² / σ)
```

That is a change to the reward's **terms**, not its coefficients. Deferring it to
step 4 would have meant another reset-critic-and-relabel cycle a few million
steps in — the exact thing the plan exists to avoid.

Locking a reward's *numbers* is not locking a reward. A term that is going to be
added later is just as breaking as a coefficient that is going to change.

## What to do next time

Before any fresh run, ask: **what will this reward have to express six months
from now?** Write those terms in on day one, even if they start disabled or
narrowly sampled. Widening a distribution is free; adding a term is not.

Bonus: command tracking kills the do-nothing exploit better than tuning weights
does (see [[001-flat-reward-breaks-on-terrain]]). Once commands are sampled, no
single behaviour scores well across all of them — freezing fails every nonzero
command, so there is nothing left to farm.
