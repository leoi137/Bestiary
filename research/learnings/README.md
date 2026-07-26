# Learnings

One file per lesson. Written when something surprises us — a run that failed,
a number that was not what we expected, a rule we got wrong.

The point: **the weights are disposable, the learnings are not.** When we
retrain from scratch, this folder is what carries over.

---

## Who these are written for

Two readers, and a learning is only finished when it works for both.

**A human who is learning robotics.** Not someone who already knows the
field — someone building the knowledge as they go. That means plain English,
jargon defined the first time it appears, and the actual mechanism explained
rather than named. "The critic was stale" teaches nothing. "The critic is the
network that predicts future reward; when we changed the reward, every number
it had learned was an estimate of a quantity that no longer existed" teaches
the idea.

**A model resuming with no memory of the run.** That means dates, run names,
exact numbers with units, and file paths — enough to reconstruct the situation
without the conversation that produced it.

Detail is the goal, not brevity. A learning that is too short to teach the
mechanism has failed at its only job. Length is fine; padding is not.

## The math standard

Most of these lessons are physics or optimization, and the math is where the
understanding actually lives. So when a lesson has math in it:

- **Write the equation.** Do not describe it in words and move on.
- **Define every symbol**, including the obvious ones, with units.
- **Work it with the real numbers from the run**, not a toy example. The
  arithmetic is the proof that the explanation matches what happened.
- **Say what the equation means physically** in one sentence, in plain
  English, after the algebra.

Worked example of the standard, from the control-cost lesson:

> SAC maximizes the discounted return
>
>     G = Σₜ γᵗ · rₜ
>
> where `rₜ` is the reward at step `t` (dimensionless), `γ = 0.99` is the
> discount factor (dimensionless, how much a reward one step later is worth),
> and `t` counts control steps at 20 Hz, so one step is 0.05 s.
>
> Our reward per step was
>
>     rₜ = w_fwd · vₓ − w_ctrl · ‖a‖²
>
> with `vₓ` the forward velocity in m/s, `a` the action vector (16 torques,
> normalized to [−1, 1]), `w_fwd = 1.0` and `w_ctrl = 0.5`.
>
> Standing still gives `vₓ = 0` and `a ≈ 0`, so `rₜ ≈ 0`. Moving at 0.3 m/s
> needed roughly `‖a‖² ≈ 1.2`, giving
>
>     r = 1.0 × 0.3 − 0.5 × 1.2 = 0.3 − 0.6 = −0.3
>
> **Moving was worth less than doing nothing.** The optimal policy under that
> reward is to stand still, and the robot found it. That is not a training
> failure; the robot solved exactly the problem we posed.

## How to add one

Number in order, add a line to the index, and use this shape:

```markdown
---
triggers: [warm_start, reward_change, critic_reset]
guard: standing-control          # or: none — see "Make it a guard" below
last_confirmed: 2026-07-25
---

# NNN — Short title

**Date:** YYYY-MM-DD · **From:** <run name, or where it came from>
**Robot:** <hound | spyder | both | n/a>

## What we believed before
The belief this lesson overturned, and **why it was reasonable at the time**.
Not self-flagellation — the shape of the mistake is the part that transfers.
"The reward was gamed" teaches nothing; "we verified walking beat standing on
flat ground and never re-checked when the terrain changed the ratio" does.

## What happened
The observation, with real numbers and units. What we expected, what we got.

## Why it happened
The mechanism, in plain English. This is the section a reader learns from —
give it the most room.

## The math
The equation, every symbol defined with units, worked with the run's real
numbers. Skip this section only if the lesson genuinely has no math in it,
and say so explicitly rather than leaving it out silently.

## What to do next time
The rule this becomes. Concrete enough to act on without re-reading the rest.

## How we would know this is wrong
What observation would overturn this lesson. A learning with no way to be
wrong is a belief, not a finding.
```

### The front matter

**`triggers`** is the whole retrieval mechanism. A lesson is useful in a
specific instant — learning 002 matters in exactly the moment someone proposes
warm-starting a critic — and that is precisely the moment nobody thinks to go
looking. So the lesson declares its own moment, and the intended action is
written in the same vocabulary before acting. This is interception rather than
recall: it works when you do not know what you do not know.

Keep the vocabulary small and shared. Current terms: `warm_start`,
`reward_change`, `critic_reset`, `obs_change`, `new_terrain`, `new_robot`,
`metric_added`, `long_run`, `resume`, `refactor`, `comparison`.

**`last_confirmed`** is the staleness handle. A cycle that touches a lesson's
subject either re-confirms the date or explains why it could not.

### Make it a guard

**If the lesson can be expressed as an assertion, it must also become one** —
`src/bestiary/guards/`. Prose depends on someone reading it at the right
moment; a guard depends on nothing.

Set `guard:` to the guard's name, or to `none` with a one-line reason. "None
because it is a judgement, not a check" is a legitimate answer. "None because
I did not get to it" is a TODO, not an answer.

Link related lessons inline by filename so the folder becomes a graph rather
than a list.

## Retiring one that turned out wrong

The **"How we would know this is wrong"** section is not decoration — it is a
live trigger, the same way a decision's revisit trigger is. When a later run
observes the falsifier, the lesson is retired **by supersession**:

1. Write a **new numbered learning** saying what was observed, why the old
   mechanism was wrong or incomplete, and what replaces it.
2. Add one line to the old file, directly under its title:
   `> **Superseded by [NNN](NNN-slug.md), YYYY-MM-DD.**` Nothing else in the
   old file changes.
3. Mark it superseded in the index below, and leave it there.

Never edit the old lesson to be right, and never delete it. The wrong version
staying readable is the evidence that the method catches its own mistakes — a
folder that quietly self-corrects looks identical to one that was never wrong,
and neither teaches anything.

## The number rule

**No number in this folder was written by hand.** Every figure comes from a
run log, a ledger row, a check output, or a short script committed alongside
the lesson. A learning with invented arithmetic is worse than no learning,
because it reads exactly like a correct one and will be trusted later.

## Index

- [001 — A reward tuned on flat ground breaks on terrain](001-flat-reward-breaks-on-terrain.md)
- [002 — Don't warm-start a critic across a reward change](002-no-warm-start-across-reward-change.md)
- [003 — Changing the observation list throws away every checkpoint](003-obs-list-is-a-one-way-door.md)
- [004 — Lock the reward *shape*, not just the weights](004-lock-the-reward-shape-not-just-the-weights.md)
- [005 — The standing check caught it again, on a different robot, from scratch](005-standing-check-caught-it-on-a-second-robot.md)
- [006 — Our regression oracle covered the robot, not the trainer](006-the-oracle-only-covered-the-robot.md)
- [007 — A peak score hides an unreliable policy](007-peak-score-hides-an-unreliable-policy.md)

> Lessons 001–005 predate the writing standard above. They are correct but
> terse. Rewrite one to the new standard whenever a run touches its subject —
> do not bulk-rewrite them for their own sake.
