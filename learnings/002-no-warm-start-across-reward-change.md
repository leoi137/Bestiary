# 002 — Don't warm-start a critic across a reward change

**Date:** 2026-07-25 · **From:** `runs/spyder_desert_v0`

## What happened

We loaded the flat-ground checkpoint (`spyder_walk_v3`, ~7000 reward per
episode) and continued training on the desert (~500 reward per episode) with a
fresh replay buffer. The gait was destroyed almost immediately:
`rollout/ep_rew_mean` went 6331 → 146 within the first few thousand steps, and
2M more steps never recovered it.

## What we learned

The **actor** can usually survive a reward change. The **critic** cannot.

SAC's critic predicts total future reward, bootstrapped over 1000 steps at
γ=0.99. It had learned "this state is worth ~7000." Reality was now ~500. Every
target was wrong by a large factor, the error compounded through the Bellman
backup, and the actor was dragged down with it.

This is why LLM post-training gets away with changing rewards (RLHF → DPO →
new reward model) and we don't: LLM RL is effectively one prompt, one response,
one terminal reward. DPO has no critic at all, GRPO dropped it. There is no
long-horizon bootstrapped value function to go stale. We have one.

### Third strike: the inherited policy had no exploration left

Same root cause (inheriting a confident policy), different mechanism. SAC's
entropy coefficient α (`train/ent_coef` in TensorBoard, `ent_coef="auto"` by
default) is what pays the policy to stay a bit random. Measured from the run:

| step | α |
|---|---|
| 3,750,341 (start of fine-tune) | 0.136 |
| 3,871,820 | 0.015 |
| 4,534,348 | 0.010 |
| 5,749,650 (end) | 0.008 |

It entered the desert **already annealed** from 3.75M flat steps, then fell 17×
in the first 120k and flatlined. After that the entropy term contributed almost
nothing to the loss — the remaining "exploration" was small noise around the
inherited gait, never a broad re-search for a terrain gait.

A fresh run starts α near 1.0 and explores widely before committing. So even if
the reward had pointed the right way, v0 had no exploration pressure left to
follow it.

## What to do next time

The rule is **not** "never warm-start." It is: **the actor survives a reward
change, the critic doesn't.** The proper recipe is keep the actor, **reset the
critic**, relabel the replay buffer with the new reward, and lower the learning
rate. (Resetting weights while keeping the buffer is a known win in deep RL —
see the primacy-bias literature.)

Two reasons we still went from scratch this time:

1. **The buffer couldn't be relabelled.** `envs/spyder_env.py:289` drops world
   x,y from the observation, so the stored transitions don't contain the
   x-displacement the forward reward needs. Control cost is recoverable from the
   stored actions; forward reward is not. Half a relabel is worse than none.
2. **The old actor was the problem.** Its learned habit was brace-and-creep at
   0.37 m/s — the exact prior we wanted gone. Normally the actor is the part
   worth keeping; here it was poisoned.
