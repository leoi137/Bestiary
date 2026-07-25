# Learnings

One file per lesson. Written when something surprises us — a run that failed, a
number that was not what we expected, a rule we got wrong.

The point: **the weights are disposable, the learnings are not.** When we retrain
from scratch, this folder is what carries over.

## How to add one

Copy the shape of an existing file. Four short sections, no essays:

```markdown
# NNN — Short title

**Date:** YYYY-MM-DD · **From:** <run name, or where it came from>

## What happened
## What we learned
## What to do next time
```

Number them in order. Add a line to the index below.

## Index

- [001 — A reward tuned on flat ground breaks on terrain](001-flat-reward-breaks-on-terrain.md)
- [002 — Don't warm-start a critic across a reward change](002-no-warm-start-across-reward-change.md)
- [003 — Changing the observation list throws away every checkpoint](003-obs-list-is-a-one-way-door.md)
- [004 — Lock the reward *shape*, not just the weights](004-lock-the-reward-shape-not-just-the-weights.md)
- [005 — The standing check caught it again, on a different robot, from scratch](005-standing-check-caught-it-on-a-second-robot.md)
