# 0003 — On Isaac Lab, train PPO at high env count rather than SAC at low

**Date:** 2026-07-29 · **Status:** accepted · **Robot:** both

## The decision

Where a task runs on Isaac Lab, it trains with **rsl_rl PPO at 1024+
environments**, not with SAC at 16–64. The MuJoCo track keeps SAC, unchanged.

Confidence in this is deliberately recorded as **low — about 60/40**. It is a
default chosen so work can proceed, not a finding. The experiment that would
settle it is specified below and has not been run.

## Why we asked

Every reward in this repository was tuned for SAC, and `research/learnings/`
is largely a record of SAC behaviour. Isaac Lab's locomotion stack is PPO.
Adopting PPO discards accumulated tuning; keeping SAC discards most of what
GPU parallelism is for. Neither is obviously right, so the question was
whether a middle path — SAC at a modest env count — beats PPO at scale.

## What we actually verified

Measured on this machine, 2026-07-29, ANYmal-C on the Bestiary desert
(`Bestiary-Desert-Coarse-Anymal-C-v0`), from tensorboard `Perf/total_fps`:

| Configuration | steps/s | note |
|---|---|---|
| MuJoCo SAC, 1 env | 110 | `hound_track_rel_s1`, from the ledger |
| Isaac PPO, 64 env | 700–1,400 | three runs, 2 iterations each |
| Isaac PPO, 1024 env | 7,630 | steady state, iteration 0 excluded |

So 1024 envs is **69x** the MuJoCo rate; 64 envs is only **6–13x**. Isaac's
fixed per-step cost — Kit, PhysX dispatch — is paid whether 64 or 1024
environments are stepped, so small env counts throw most of the advantage away.

Peak VRAM at 1024 envs was **4,649 MiB** of 8,192, inside the 6,000 MiB
ceiling, so the env count is not forced down by hardware.

**What was NOT verified, and it is the number that decides this:** the
sample-efficiency ratio between SAC and PPO on our task. Published legged
locomotion suggests PPO needs somewhere between 10x and 100x more samples for
comparable behaviour. At a 69x throughput advantage the two effects are the
same order of magnitude, which is precisely why this decision is 60/40 and not
90/10.

## The trigger to revisit

Reopen when **any** of these becomes true:

1. **The head-to-head is run.** Same task, same reward, same terrain, trained
   to the *same reward threshold* on both stacks, wall-clock to threshold
   compared, three seeds per arm per the seed rule. This is the real answer and
   costs roughly a day. Until then this decision is a coin-flip with a reason.
2. **PPO at 1024 fails to reach a behaviour SAC already reached.** If a reward
   SAC solved on the desert cannot be solved by PPO at scale within ~3x the
   wall clock, sample efficiency is dominating and the middle path is back.
3. **`skrl` is installed and its GPU-resident SAC works at 256+ envs.** The
   architectural objection below is specific to Stable-Baselines3; skrl removes
   most of it, which makes the middle path much cheaper to test.

## What we gave up

The middle path — SAC at 64–256 envs — is genuinely unexplored, and it is the
option where our existing expertise compounds instead of being discarded: the
tuned reward, the 38-assertion oracle, learnings 011 and 012, and every
calibrated prediction about SAC behaviour. Nobody has published a config for
it, which is an argument both ways.

We also gave up, for now, on the possibility that SAC's efficiency makes 64
envs competitive. The throughput table above is the reason: 6–13x is too small
a gain to justify a stack change at all, so if SAC were the right algorithm the
correct conclusion would be to stay on MuJoCo entirely.

## How we would know this was wrong

- Stable-Baselines3 SAC turns out to work acceptably at 1024 Isaac envs,
  contradicting the expectation that host-side numpy transfer and a
  host-resident replay buffer make it impractical. (Isaac observations are
  torch tensors on `cuda`; SB3 copies them to CPU every step, and our
  single-env buffer is already 2.6 GB.)
- The head-to-head shows SAC at 32–64 envs reaching the threshold in less
  wall-clock than PPO at 1024.
- PPO at scale reaches high reward on a metric that a do-nothing policy also
  scores well on — the failure mode `learnings/011` documents. Throughput would
  then have bought faster arrival at the same wrong answer.
