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

1. [008 — What a policy is, and why it is a neural network](008-what-a-policy-is.md) — a policy is a function from 169 sensed numbers to 16 motor commands; a lookup table for it would be 10⁸⁹ times the atoms in the observable universe.
2. [009 — Why one network is not enough](009-actor-and-critic.md) — the actor picks the move, the critic guesses what follows; the critic's first layer is `Linear(185, 256)` against the actor's `Linear(169, 256)`, and those 16 extra columns are the entire architecture.
3. [001 — What a reward function is, and how ours told the robot to stand still](001-what-a-reward-function-is.md)
4. [002 — Why one training run is not a result](002-why-one-seed-is-not-a-result.md)
5. [004 — Why changing the reward poisons the replay buffer](004-why-a-reward-change-poisons-the-buffer.md)
6. [003 — Why two rewards should be multiplied, not added](003-add-or-multiply.md)
7. [006 — What γ = 0.99 is really saying about the future](006-what-gamma-is-saying-about-the-future.md)
8. [007 — When a tolerance scales with the command, the command cancels](007-a-tolerance-that-cancels-the-command.md) — make the tolerance proportional to the command and a do-nothing machine is paid the same for every command you can give it.
9. [005 — What `ent_coef` really measures](005-what-ent-coef-really-measures.md)

Note the reading order is not the file order: 004 explains the machinery that
003's reward change had to be built around, so it reads first. 006 reads after
003 because its worked example is a constant of 003's reward — the termination
penalty — and it explains why per-step economics are the only economics the
robot can act on. 007 reads after 006 because it is the same failure one level
up — 003 says multiply the two channels, 007 says the *tolerance inside* a
channel can hand back the free lunch multiplying was meant to remove. 005 reads
last of the current set because it is the first one
that needs a reward change to already be understood — it compares the entropy
dynamics *across* two rewards.

### Planned, roughly in the order they will be needed

- ~~What a policy is, and why it is a neural network~~ → [008](008-what-a-policy-is.md)
- ~~Actor and critic: why one network is not enough~~ → [009](009-actor-and-critic.md)
- Torque control versus PD position targets
- What an observation is, and why its width is a one-way door
- ~~Discounting: what γ = 0.99 is really saying about the future~~ → [006](006-what-gamma-is-saying-about-the-future.md)
- Why parallel environments change everything

Each lands when the project needs it to decide something, not before.

**The planned list is down to 3**, from 4, from 5, from 6, from 7, and from 8.
Cycle 011 took *"Actor and critic: why one network is not enough"* off it — off
the top of the queue, and on the **both-tests** basis rather than 009's weaker
queue-order-only one. The cycle spent itself on a single 5,106,759-byte
checkpoint file, and "what is actually inside those bytes" is answered by
`policy.pth`: 227,330 critic parameters, 227,330 more in a target copy, and
117,536 in the actor. The idea the work touched and the head of the list were
the same item.

That lesson also cost its own correction, in the way 005 and 006 did. The draft
said the 100-step horizon was "one second" at 100 Hz; the env runs at 20 Hz
(`frame_skip = 10` × 0.005 s), so it is **five** seconds. Caught by moving the
number out of prose and into
`docs/lessons/scripts/actor_critic_math.py`, which now reads `dt` off the env
rather than trusting a remembered control rate. The number rule earning its
keep, again, on a figure nobody would have questioned.

Cycle
009 took *"What a policy is, and why it is a neural network"* off it — the head
of the queue, and this time the queue is what chose the topic rather than the
week's surprise. **This is a draw-down.**

Cycle 008 took *"Discounting: what γ = 0.99 is really saying about the future"*
off on the both-tests basis: it was on the list *and* it was the idea the cycle
actually touched, because the cycle was re-deriving reward constants against
measured per-step rates and `TERMINATION_PENALTY = 10.0` is literally
`c/(1−γ)`. Cycle 007 took the entropy lesson off on the same test, and cycle
006 the one before it.

007 broke that streak: the idea it teaches — a reward tolerance that cancelled
its own command — was the one the work actually touched, so it was written
instead of the next queued item and the queue did not shrink. The rule permits
that, but it left a debt, recorded here at the time as *"the next lesson owes
the list an item."* 008 (the lesson file, from cycle 009) is that item paid
back: it was taken strictly in queue order, off the top.

One honest caveat on it. The both-tests standard was not met — the policy
network was not the thing this cycle's work turned on, it was simply next. That
is the weaker of the two justifications, and it is the right one to use when
the queue is owed. Where it earns its place regardless is the last section: the
observation width is a one-way door precisely *because* `W₁` is
`Linear(169, 256)`, and that is a fact the reader has been asked to take on
trust in three earlier lessons without ever being shown the matrix.

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
