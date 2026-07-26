# Lessons

The curriculum. One idea per page, explained from scratch, in plain language,
with the real equation worked on a real number this project actually produced.

Written for someone smart who is learning robotics and reinforcement learning
from the ground up and wants the math, the algorithms, the network
architectures, and the structure — **briefly and simply, but with the real
detail rather than a hand-wave.**

## The standard

- **One idea per lesson.** If the title needs "and", it is two lessons.
- **One page.** ~300–500 words plus the math. Hard limit. Brief-and-simple
  means cutting scope, never cutting rigor — a long lesson is an unfinished
  one.
- **The one-sentence answer comes first**, before any context. A reader who
  stops after the first line should still have learned something true.
- **Every term defined the first time it appears.** No jargon carried in from
  a paper.
- **The equation is written out**, every symbol defined with units, and
  **worked on a real number from this repo** — not a toy example. The
  arithmetic is what turns a symbol into an idea.
- **One plain sentence after the algebra** saying what it means physically.
  That sentence is the lesson; the algebra is the evidence.
- **Ends with where it bites** — the file in this repo where the idea is
  load-bearing, and what broke when we got it wrong.

## How this differs from the two neighbouring folders

| folder | reader | answers |
|---|---|---|
| **`docs/lessons/`** | someone learning the field | *what is this thing?* |
| `docs/theory/` | someone who now needs the depth | *how does it work, exactly?* |
| `../research/learnings/` | someone following this project | *what surprised us, and why?* |

A lesson is the doorway; a theory note is the room behind it. Lessons link
into theory, never the reverse.

## When they get written

**When the project touches the idea** — the same rule the theory notes follow.
Theory learned in the abstract does not stick and does not get used; theory
learned the week it decides something does both. The cycle that tunes an
entropy coefficient is the right time for the entropy lesson.

Files are numbered in the order written. **The reading order is this index**,
which gets re-sorted as the set grows.

## Reading order

1. [001 — What a reward function is, and how ours told the robot to stand still](001-what-a-reward-function-is.md)

### Planned, roughly in the order they will be needed

- What a policy is, and why it is a neural network
- Actor and critic: why one network is not enough
- What the replay buffer holds, and why changing the reward poisons it
- Entropy, and what `ent_coef` collapsing to 0.008 actually meant
- Torque control versus PD position targets
- What an observation is, and why its width is a one-way door
- Discounting: what γ = 0.99 is really saying about the future
- Why parallel environments change everything

Each lands when the project needs it to decide something, not before.
