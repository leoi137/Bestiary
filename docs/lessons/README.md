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
2. [002 — Why one training run is not a result](002-why-one-seed-is-not-a-result.md)
3. [004 — Why changing the reward poisons the replay buffer](004-why-a-reward-change-poisons-the-buffer.md)
4. [003 — Why two rewards should be multiplied, not added](003-add-or-multiply.md)

Note the reading order is not the file order: 004 explains the machinery that
003's reward change had to be built around, so it reads first.

### Planned, roughly in the order they will be needed

- What a policy is, and why it is a neural network
- Actor and critic: why one network is not enough
- Entropy, and what `ent_coef` collapsing to 0.008 actually meant
- Torque control versus PD position targets
- What an observation is, and why its width is a one-way door
- Discounting: what γ = 0.99 is really saying about the future
- Why parallel environments change everything

Each lands when the project needs it to decide something, not before.

**The planned list is down to 7**, from 8. Cycle 006 took *"What the replay
buffer holds, and why changing the reward poisons it"* off it — the first time
the queue has been drawn down rather than added around. It was the right item
by both tests the rule offers: it was next-in-line *and* it was the idea the
cycle was built around, because the tracking reward had to ship as a new env id
trained from scratch precisely so the buffer would not be poisoned.

The title lost its "and" on the way in. The queued phrasing named two things,
and the one-idea rule at the top of this file says that is two lessons; what
the buffer holds is the setup, and the poisoning is the idea.
