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
- [0003 — On Isaac Lab, PPO at high env count over SAC at low](0003-ppo-at-scale-over-sac-at-small-scale.md)
