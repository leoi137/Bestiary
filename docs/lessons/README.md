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
- **Explained quantities read US-first with SI in parentheses** —
  `4.87 lb (2.21 kg)`, `9.0 in (229 mm)`. The math, the code, and every
  number quoted from a log stay SI; angles and dimensionless ratios have no
  US twin and are left alone. (Lessons 001–014 predate this convention and
  read SI-first.)
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
2. [013 — What an observation is, and why its width is a one-way door](013-what-an-observation-is.md) — the array the policy is handed each step *is* its whole world; its width is a column count in `W1`, so 113 → 169 is 14,336 weights that do not exist in the old checkpoint and `SAC.load()` raises rather than degrades.
3. [009 — Why one network is not enough](009-actor-and-critic.md) — the actor picks the move, the critic guesses what follows; the critic's first layer is `Linear(185, 256)` against the actor's `Linear(169, 256)`, and those 16 extra columns are the entire architecture.
4. [001 — What a reward function is, and how ours told the robot to stand still](001-what-a-reward-function-is.md)
5. [002 — Why one training run is not a result](002-why-one-seed-is-not-a-result.md)
6. [004 — Why changing the reward poisons the replay buffer](004-why-a-reward-change-poisons-the-buffer.md)
7. [003 — Why two rewards should be multiplied, not added](003-add-or-multiply.md)
8. [006 — What γ = 0.99 is really saying about the future](006-what-gamma-is-saying-about-the-future.md)
9. [007 — When a tolerance scales with the command, the command cancels](007-a-tolerance-that-cancels-the-command.md) — make the tolerance proportional to the command and a do-nothing machine is paid the same for every command you can give it.
10. [005 — What `ent_coef` really measures](005-what-ent-coef-really-measures.md)
11. [011 — Torque control versus PD position targets](011-torque-versus-pd-position-targets.md) — the same 16 numbers mean *how hard to push* in one env and *what angle to be at* in the other; `tau = 90.0 × 0.41973 = 37.776 N·m` is what the servo does for free, and it bought ~5x fewer samples and a slightly *lower* ceiling.
12. [014 — The anatomy of the Spyder policy, input to output](014-anatomy-of-the-spyder-policy.md) — the whole brain on one page, measured out of `model_1499.pt`: `Linear(235, 512) → 512 → 256 → 128 → 12` with ELU, the 235 broken into its eight named blocks, `tau = 15.0 · a` at zero error, and an episode return that is literally metres travelled.
13. [010 — Why a test can pass without testing anything](010-the-empty-set-says-yes.md) — "every X has property P" is true when there are no X, so 3 of this suite's 111 set-quantified assertions were reporting `PASS` over an empty set; one of them had coverage 0/9 while green.
14. [012 — When an average hides a single winner](012-when-an-average-hides-a-single-winner.md) — one of six cells was 98.8% of the total gap, so a 5.04x headline becomes 1.05x when it is dropped and undefined when a different one is; always compute leave-one-out before believing an aggregate.

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
- ~~Torque control versus PD position targets~~ → [011](011-torque-versus-pd-position-targets.md)
- ~~What an observation is, and why its width is a one-way door~~ → [013](013-what-an-observation-is.md)
- ~~Discounting: what γ = 0.99 is really saying about the future~~ → [006](006-what-gamma-is-saying-about-the-future.md)
- Why parallel environments change everything

Each lands when the project needs it to decide something, not before.

**014 was requested by the operator and is not on the planned list, which
therefore stays at 1.** The one-idea and one-page rules are both stretched: it
is a reference page covering network, observation, action and reward for a
single checkpoint, and it is roughly three pages. The trade it buys is that
those four things are only comprehensible together — 008, 009, 011 and 013 each
had to gesture at the other three — and every dimension in it is read out of
one trained file rather than four. **The next lesson owes the list its
remaining item.**

Its own correction, in the pattern the number rule keeps producing, and this
one cut against the brief twice. First, the brief described the checkpoint as
an rsl-rl `ActorCritic` and asked for that module listing; the file on disk is
rsl-rl 5.x, which stores **two independent `MLPModel`s** under separate
`actor_state_dict` and `critic_state_dict` keys, with the action spread living
in a `GaussianDistribution` submodule rather than a bare `std` tensor. Second,
the brief pointed at `src/bestiary/isaac/rl_cfg.py` for the network dims —
that file declares none at all. It subclasses `AnymalCRoughPPORunnerCfg` and
changes only `experiment_name`, so `[512, 256, 128]` and `elu` exist nowhere in
this repository's source and are recoverable only from the run's own
`params/agent.yaml`. Both caught by
`docs/lessons/scripts/014_spyder_policy_anatomy.py`, which rebuilds the model
from the config and loads the checkpoint into it with `strict=True` — a listing
the real weights refuse to enter is not a measurement.

One rate to keep straight while reading across the set: 008, 009, 011 and 013
describe the MuJoCo envs at 20 Hz, while the Isaac task 014 describes runs at
**50 Hz** (0.005 s physics, `decimation = 4`). Two stacks, two control rates,
and a per-step number carried from one to the other is wrong by 2.5x.

**The planned list is down to 1 after 013, and cycle 012's debt is paid.** 013 was
taken strictly in queue order — off the top of the list, on the weaker
queue-order-only justification rather than the both-tests one. The cycle that
commissioned it was spent on a reward decomposition, not on observations; the
observation list is simply what the queue said was next, and the queue is what
a cycle owes when the previous one skipped it. Where it earns its place anyway
is that three earlier lessons — 008, 009 and 011 — have all leaned on
`Linear(obs, 256)` without the reader ever being shown what `obs` *contains*,
or why a checkpoint dies rather than degrades when it changes.

Its own correction — and this one cuts against the draft rather than against
the repo, which is worth recording as its own kind of result. The lesson was
first drafted claiming `research/CORE_PLAN.md` and `CLAUDE.md` carry a *wrong*
number, because both name **141** while Spyder measures **113**. They do not.
`CORE_PLAN.md` is a proposal — it computes `141 × 256 = 36,096` as the width
Spyder *would* have after adding reserved command and height slots — and
`CLAUDE.md` states in the same sentence that Spyder "is at 113 today" and that
the plan "is not yet applied". Read as a plan, both are exactly right, and a
lesson that accused them would have put a false correction into the record
while sounding like the number rule working.

What the script does establish, and what the prose nowhere says, is where the
reserved-slot design actually landed: not on Spyder at 141 but on the **Hound
at 169**, in all six of its envs including both tracking envs, every one
sharing obs hash `11093686ef09fe13`. And 141 is not merely hypothetical — it
is on disk as a fossil, the `(256, 141)` first layer in
`runs/hound_desert_test150k/` sitting against a live env of 169, which is
precisely why that run is the dead one. All of it from
`docs/lessons/scripts/013_observation_width_math.py` building every env and
reading its declared spec instead of trusting prose.

**The planned list was 2 after 012, and that cycle did not draw it down.**
*When an average hides a single winner* is not on the list and was written
anyway, on the basis 007 and 010 used: it is the idea the cycle's work actually
turned on. The cycle's entire result was a `drive_grid_mean` ratio measured
against a pre-registered bar, and the question of whether that mean meant
anything is the only question the week contained — 98.8% of the gap sat in one
of six cells. Writing *"what an observation is"* off the top of the queue would
have meant teaching a one-way door from the ledger while the week's real idea
went unwritten.

That is the weaker of the two justifications and it leaves a debt, recorded
here as 007's and 010's were: **the next lesson owes the list an item and should
be taken strictly in queue order.**

Its own correction, in the pattern the number rule keeps producing: the gap
column was first added up in prose as +94.94, and
`docs/lessons/scripts/012_leave_one_out.py` — summing the same six cells
straight out of the measurement JSONs — returns **+94.93**. The prose sum had
accumulated a rounding error of exactly the kind the leave-one-out table is
there to expose. The script also refuses to print a ratio whose denominator has
gone negative, which is why the `(-0.3, 0, 0)` row reads UNDEFINED rather than a
plausible-looking number.

**The planned list was down to 2 after 011, and cycle 012's debt is paid.** 011 was taken
strictly in queue order — off the top of the list, on the weaker
queue-order-only justification rather than the both-tests one, which is exactly
the justification a cycle uses when the queue is owed. It also earns its place
on merit: three lessons and two ledger rows have leaned on the PD-versus-torque
comparison without the reader ever being shown what a position target *is*, and
`tau = kp(q_des − q) − kv·qdot` on the real gains is a four-line answer that had
never been written down.

Its own correction, in the pattern 005, 006 and 009 set: the docstring of
`robots/hound/build.py::actuator_xml` said the PD loop runs "five times per
control step". It runs **ten** — 200 Hz physics under a 20 Hz policy. Caught by
`docs/lessons/scripts/pd_vs_torque_math.py` reading `timestep` off the XML and
`frame_skip` off the env instead of trusting the prose, and fixed in the source.
That is the fourth remembered control rate the number rule has caught.

**The planned list was still 3 after cycle 012 — that cycle did not draw it
down, and owed it an item.** 010 is not on the list and was written anyway, on the same basis 007
was: it is the idea the cycle's work actually turned on, and it had become
load-bearing that week rather than merely interesting. The cycle spent itself
auditing what fraction of the guard suite checks nothing, which is a question
about quantifiers over empty sets and about nothing else; writing the head of
the queue instead would have meant teaching torque-versus-PD from the ledger
while the week's real idea went unwritten.

That is the weaker of the two justifications and it leaves a debt, recorded
here as 007's was: **the next lesson owes the list an item and should be taken
strictly in queue order.** Whether *"why a test can pass without testing
anything"* belongs on the planned list retroactively is not a question this
cycle gets to answer in its own favour — it went in the reading order, not the
backlog.

**The planned list was down to 3**, from 4, from 5, from 6, from 7, and from 8.
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
