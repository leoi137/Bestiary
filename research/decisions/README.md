# Decisions

A choice we made, why we made it, and — the part that matters — **the
observation that would reverse it**.

Without a written trigger, a settled question gets re-argued every few weeks
by whoever has the most recent context. With one, checking whether a decision
still holds takes seconds.

Shape:

```markdown
# NNNN — Short title

**Date:** YYYY-MM-DD · **Status:** accepted | superseded by NNNN · **Robot:** ...

## The decision
## Why we asked
## What we actually verified     (numbers and sources, not recollection)
## The trigger to revisit        (concrete and checkable)
## What we gave up
## How we would know this was wrong
```

A decision is never edited to match how things turned out. If it changes,
write the next one and mark this one `superseded by NNNN`.

## Index

- [0001 — Defer Isaac Lab; stay on MuJoCo, port to MJX](0001-defer-isaac-lab.md)
- [0002 — The backward eval cell is off-distribution; report it separately](0002-off-manifold-eval-cell.md)
- [0003 — On Isaac Lab, PPO at high env count over SAC at low](0003-ppo-at-scale-over-sac-at-small-scale.md)
- [0004 — Inherit the Isaac reward knowingly; re-scope it for Hound](0004-inherit-the-isaac-reward-knowingly.md)
- [0005 — The Isaac Hound stack: what survives an adversarial pass](0005-the-isaac-hound-stack-what-survives.md)
- [0006 — What the reward economics may and may not assume, going into the first evaluated arm](0006-what-the-reward-economics-may-assume.md)
