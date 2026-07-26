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

Number in order, add a line to the index, and use these sections:

```markdown
# NNN — Short title

**Date:** YYYY-MM-DD · **From:** <run name, or where it came from>
**Robot:** <hound | spyder | both | n/a>

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

Link related lessons inline by filename so the folder becomes a graph rather
than a list.

## Index

- [001 — A reward tuned on flat ground breaks on terrain](001-flat-reward-breaks-on-terrain.md)
- [002 — Don't warm-start a critic across a reward change](002-no-warm-start-across-reward-change.md)
- [003 — Changing the observation list throws away every checkpoint](003-obs-list-is-a-one-way-door.md)
- [004 — Lock the reward *shape*, not just the weights](004-lock-the-reward-shape-not-just-the-weights.md)
- [005 — The standing check caught it again, on a different robot, from scratch](005-standing-check-caught-it-on-a-second-robot.md)
- [006 — Our regression oracle covered the robot, not the trainer](006-the-oracle-only-covered-the-robot.md)

> Lessons 001–005 predate the writing standard above. They are correct but
> terse. Rewrite one to the new standard whenever a run touches its subject —
> do not bulk-rewrite them for their own sake.
