# 0001 — Defer Isaac Lab; stay on MuJoCo, port to MJX when throughput is the bottleneck

**Date:** 2026-07-25 · **Status:** accepted · **Robot:** both

## The decision

We are not porting to NVIDIA Isaac Lab now. We stay on MuJoCo, and when
single-environment throughput becomes the binding constraint we port to
**MJX** (MuJoCo compiled through JAX) rather than Isaac Lab.

## Why we asked

`hound_desert_v0` ran 3,750,000 steps in 8 h 05 m at 129 steps/s in a single
environment. That is roughly four experiments a week. Production legged-RL
stacks train with thousands of parallel environments on one GPU, on the order
of a thousand times more samples per hour. Experiment *count*, not reward
quality, is what limits this project — see
`../episodes/001-hound-throughput.md`.

## What we actually verified

Checked against NVIDIA's published requirements on 2026-07-25, not from
memory.

| | This machine | Isaac Sim 5.x minimum |
|---|---|---|
| GPU | RTX 2080 (Turing) | RTX 4080 (Ada) |
| VRAM | 8 GB (~7.4 GB free; display holds 570 MB) | 16 GB |
| Driver | 580.173.02 | 580.65.06 ✓ |
| RAM | 31 GB | 32 GB (marginal) |
| OS | Ubuntu 24.04 | 22.04 / 24.04 ✓ |
| Disk | 654 GB free | 50 GB ✓ |
| Python | 3.13 | 3.11 (needs a separate venv) |

So: **half the minimum VRAM**, on an architecture no longer listed as
supported. Two mitigating facts — the hard architectural requirement is RT
cores, which Turing has (that is why the A100 and H100 are *unsupported*
despite being datacenter cards), and the 16 GB figure is justified in the docs
by *rendering* complex scenes, which headless RL training does not do. It
would probably run. It is not certain, and we would be the first on that
configuration.

## Why MJX wins anyway

The install is a weekend. The **port** is the real cost, and it is where the
two options diverge sharply.

Isaac Lab runs on PhysX, a different physics engine:

- The MJCF model would have to be converted to USD, and the hound's driven hub
  wheels will need hand-fixing.
- `envs/hound.py` becomes a `DirectRLEnv` — every reward term rewritten as a
  batched torch operation.
- The procedural desert in `terrain/generate.py` is re-authored against Isaac
  Lab's terrain generator, not ported.
- **The 38-assertion check in `robots/hound/check.py` does not transfer.**
  Contact, friction, and solver behaviour differ, so the tuned reward may not
  either. We would lose our only regression oracle at exactly the moment we
  changed everything else.

MJX keeps the same MJCF file and the same physics semantics. `build.py` and
`check.py` survive intact; only the env layer is rewritten.

## The trigger to revisit

Reopen this decision when **either** of these becomes true:

1. **We reach the computer-vision or sim-to-real stage.** Isaac Sim's camera
   and sensor simulation and its domain randomization are genuinely better
   than MuJoCo's. That is the stage where the rebuild buys capability rather
   than just speed — and it is where this project is headed.
2. **A GPU with ≥16 GB VRAM is available.** Funding, a grant, or a program
   award all count.

Until one fires, do not re-argue this. Re-reading this file is the check.

## What we gave up

Isaac Lab is the industry standard for legged locomotion. There is real
résumé and credibility value in having used it, which matters for xTech and
similar programs. We are trading that for keeping our validation oracle and
not fighting an unsupported GPU. The trigger above is what buys it back.

## How we would know this was wrong

- MJX turns out to be materially slower than expected on a 2080 (say under
  5,000 steps/s for a 16-DoF robot), removing most of the throughput
  argument.
- The MJX port to a batched env turns out to cost as much as the Isaac Lab
  port would have, removing the cost argument.
- Someone demonstrates Isaac Lab training a comparable robot on 8 GB of
  Turing without incident, removing the risk argument.
