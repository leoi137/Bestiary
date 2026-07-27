# 004 — Why changing the reward poisons the replay buffer

**One sentence:** The replay buffer is a warehouse of past experiences, each
one *labelled* with the reward it earned, and if you change the reward
function the labels do not update — so the robot keeps being trained on prices
that no longer exist.

## The idea

SAC, the algorithm this project trains with, is **off-policy**: it does not
learn only from what the robot is doing right now. Every step it takes gets
filed away as a **transition** — a five-part record of *(what I saw, what I
did, what I was paid, what I saw next, did the episode end)*. That warehouse
is the **replay buffer**.

Learning happens by drawing random transitions back out of the warehouse and
adjusting the networks to predict them better. This is why off-policy methods
are sample-efficient: one hard-won second of robot experience gets studied
hundreds of times instead of once.

Here is the trap. The *"what I was paid"* field is a **number, frozen at the
moment it was stored**. It is not a pointer to the reward function; it is the
output of whatever reward function happened to be running that day. Change the
reward and every one of the million rows already in the warehouse keeps its old
price tag. The robot then spends the next hour of training learning to predict
a reward nobody will ever pay it again — and the network that does that
predicting, the critic, is the thing the whole algorithm steers by.

It is not a slow drift. It is a confident network being confidently taught the
wrong prices, and it does not recover on its own.

## The math

One transition, for the hound:

```
bytes = 2 · n_obs · 8   +   n_act · 4   +   3 · 4
        └ s and s' ┘        └action┘        └ reward, done, timeout ┘
```

`n_obs = 169` observations (float64, 8 bytes each — this is why the file is
2.6 GB and not 1.4), `n_act = 16` motor commands (float32).

```
bytes = 2 · 169 · 8 + 16 · 4 + 12
      = 2704 + 64 + 12
      = 2780 bytes per transition

× 1,000,000 transitions = 2,780,000,000 bytes = 2.589 GiB
```

The real file `runs/hound_pd_desert_s1/ant_buffer.pkl` measured
**2,780,004,759 bytes**. The prediction is off by 4,759 bytes — 1.7 parts per
million, which is the pickle header.

And the reuse rate, which is what makes a stale label expensive:

```
draws = env_steps × batch_size = 1,000,000 × 256 = 256,000,000
reuse = draws / buffer_size = 256,000,000 / 1,000,000 ≈ 256
```

**Physically: every single step the robot takes is replayed into the networks
about 256 times, so one wrong price tag is not one bad lesson — it is 256 of
them.**

## Where it bites here

`research/nulls.jsonl` row 1. The spider's gait was warm-started from
`spyder_walk_v3` into `SpyderDesert-v0`, which pays a different reward.
`ep_rew_mean` fell from **6331 to 146** — a **43.4×** collapse — within a few
thousand steps, and never recovered. That cost about 15 GPU-hours and it is why
`research/learnings/002` exists.

It is now impossible to repeat by accident. `src/bestiary/envs/reward_spec.py`
hashes what a reward pays for, `train.py` pins that hash into the run's
`config.json`, and a resume whose hash moved **raises** instead of warning.
This cycle is the first real test: the tracking reward hashes
`eb046c4c7013310a` against the old `89d9e92c6af6afec`, so
`HoundPDTrackDesert-v0` had to be a new env id trained from scratch — which is
exactly what happened, rather than a resume that would have looked fine for
about twenty minutes.

Assumes [002 — why one seed is not a result](002-why-one-seed-is-not-a-result.md)
only for the habit of distrusting a single number. Arithmetic:
`scripts/004_replay_buffer.py`.

## If you want to go deeper

`research/learnings/002-no-warm-start-across-reward-change.md` — the incident,
with what the critic's loss curve looked like while it happened.
