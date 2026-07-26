---
name: write-learning
description: Write one entry in research/learnings/ to the repo's standard — plain English that teaches the mechanism, real math worked with the run's actual numbers, and an explicit way the lesson could be proven wrong. Use whenever a run, check, or investigation produces something surprising that should outlive the weights.
---

# write-learning

A learning is written when something **surprised us**. A run that went as
predicted confirms the model and belongs in `research/ledger.jsonl`, not here.

The premise of the whole folder: *the weights are disposable, the learnings are
not.* When we retrain from scratch — and we will — this is what carries over.

## Two readers, both mandatory

**A human learning robotics.** Not someone who already knows the field.
Jargon gets defined the first time it appears, and the mechanism gets
explained rather than named.

> ✗ "The critic was stale."
>
> ✓ "The critic is the network that predicts future reward. When we changed
> the reward, every number it had learned was an estimate of a quantity that
> no longer existed — so its confident predictions were confidently about the
> wrong thing."

**A model resuming with no memory of the run.** Dates, run names, exact
numbers with units, file paths. Enough to reconstruct the situation without
the conversation that produced it.

A draft that only works for one of these is not finished.

## Length

Detail is the goal. A lesson too short to explain its mechanism has failed at
its only job. Do not pad, but do not compress the explanation out of it
either — the old "four short sections, no essays" rule was wrong and is gone.

## The math standard — not optional

Most of these lessons are physics or optimization, and the math is where the
understanding actually lives.

1. **Write the equation.** Do not describe it in prose and move on.
2. **Define every symbol**, including the obvious ones, **with units**.
3. **Work it with the run's real numbers.** Not a toy example — the arithmetic
   is the proof that the explanation matches what actually happened.
4. **One plain-English sentence after the algebra** saying what it means
   physically.

If a lesson genuinely has no math in it, say so explicitly in the section
rather than dropping the section silently.

The worked example of this standard lives in `research/learnings/README.md`.
Read it before writing your first one.

## Sections

```markdown
# NNN — Short title

**Date:** YYYY-MM-DD · **From:** <run name, or where it came from>
**Robot:** <hound | spyder | both | n/a>

## What happened
## Why it happened          <- the section a reader learns from; give it room
## The math
## What to do next time
## How we would know this is wrong
```

**"How we would know this is wrong" is required.** A learning with no way to
be wrong is a belief, not a finding. Name the observation that would overturn
it.

## Before writing

- **Read the index.** If a lesson already covers this, extend that file rather
  than adding a near-duplicate. Five sharp lessons beat twelve overlapping
  ones.
- **Get the real numbers.** Pull them from the run log, the ledger row, or the
  check output. Never write an approximate number you could look up — a
  learning with invented arithmetic is worse than no learning, because it will
  be trusted later.
- **Check it is a surprise.** If you predicted it, it is a ledger row.

## After writing

- Add the index line in `research/learnings/README.md`.
- Link related lessons inline by filename, so the folder becomes a graph
  rather than a list.
- Commit it **alone**, via `commit-push`. One commit per learning — that is
  the unit the record is read in, and a commit landing three at once destroys
  the ability to see when each was understood.

## Rewriting an old lesson

Lessons 001–005 predate this standard. They are correct but terse. Rewrite one
**when a run touches its subject** — never as a bulk pass, which produces
uniform prose nobody learned anything from writing.
