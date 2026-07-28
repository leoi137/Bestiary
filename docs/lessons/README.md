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
5. [006 — What γ = 0.99 is really saying about the future](006-what-gamma-is-saying-about-the-future.md)
6. [005 — What `ent_coef` really measures](005-what-ent-coef-really-measures.md)

Note the reading order is not the file order: 004 explains the machinery that
003's reward change had to be built around, so it reads first. 006 reads after
003 because its worked example is a constant of 003's reward — the termination
penalty — and it explains why per-step economics are the only economics the
robot can act on. 005 reads last of the current set because it is the first one
that needs a reward change to already be understood — it compares the entropy
dynamics *across* two rewards.

### Planned, roughly in the order they will be needed

- What a policy is, and why it is a neural network
- Actor and critic: why one network is not enough
- Torque control versus PD position targets
- What an observation is, and why its width is a one-way door
- ~~Discounting: what γ = 0.99 is really saying about the future~~ → [006](006-what-gamma-is-saying-about-the-future.md)
- Why parallel environments change everything

Each lands when the project needs it to decide something, not before.

**The planned list is down to 5**, from 6, from 7, and from 8. Cycle 008 took
*"Discounting: what γ = 0.99 is really saying about the future"* off it on the
now-usual both-tests basis: it was on the list *and* it was the idea the cycle
actually touched, because the cycle was re-deriving reward constants against
measured per-step rates and `TERMINATION_PENALTY = 10.0` is literally
`c/(1−γ)`. Cycle 007 took the entropy lesson off on the same test, and cycle
006 the one before it.

Three consecutive draw-downs. The queue has now shrunk every cycle since the
rule was written down, which is what the rule was for.

006 also corrected its source on the way in, in the same way 005 did. The
brief that commissioned it described `c/(1−γ)` as the *income* death destroys;
the derivation this repo actually shipped
(`docs/theory/command-tracking-reward.md` §4) sets c to the *loss* an early
flailing policy escapes by dying. Same series, opposite sign, and only the
second one explains why the constant exists. The lesson teaches the code.

**On 005 (kept):** the queued title carried a wrong number and a wrong reading,
and both were fixed on the way in. The number: 0.008 appears in no run's log — the
old-reward run's floor is 1.03e−2 and the tracking run's is 1.76e−4
(`research/scripts/entropy_lesson_math.py`). The reading: the queue phrased
the collapse as the thing to explain, but α falling means the policy was
*more* random than the target, not less, so the lesson is what the coefficient
measures rather than what one collapse meant.

Cycle 006's note on titles still applies: the queued phrasing named two things
and the one-idea rule says that is two lessons.
