# Episodes

One file per cycle of the research loop.

An episode records what was tried, **what was predicted before the result was
known**, and what actually happened. The prediction is the point. A record
written after the fact always sounds reasonable; a prediction written before
it can be wrong, and being wrong on paper is how the diagnosis improves.

## Rules

- **Never edit an episode to match how things turned out.** If the diagnosis
  changes, write the next episode and say what the previous one got wrong.
- **Write the prediction before starting the run**, not while it is finishing.
- **Make the prediction falsifiable.** "It should do better" is not a
  prediction. "A plateau between 500k and 1.5M steps, with `ent_coef` under
  0.02 by 60k" is.
- One episode per cycle. If a cycle produced nothing, write that — a quiet
  cycle is data about the loop.

## Shape

```markdown
# NNN — Short title

**Date:** YYYY-MM-DD · **Robot:** ... · **Run:** <run name, or none>

## Thesis            what we believe is blocking progress right now
## Diagnosis         the evidence, with numbers
## Ranked actions    what to do, cheapest-useful first
## Prediction        falsifiable, written before the result was known
## Open questions    what the next episode inherits
```

## Index

- [001 — Hound: the pipeline is the bottleneck, not the reward](001-hound-throughput.md)
  — written 2026-07-25 while `hound_desert_v0` was still running. Its primary
  prediction (a plateau, a crouching survivor rather than a walker) was
  largely borne out; see `../ledger.jsonl`.
- [002 — PD position targets: does changing the action space unstick the hound?](002-pd-position-targets.md)
  — prediction committed before the run started. Called 55% that
  `ep_rew_mean` would clear 1096; it finished at 1099.63.
- [003 — PD made the plateau 5× cheaper, not higher](003-pd-result-cheaper-not-higher.md)
  — scores 002. PD reached the same band in 5× fewer samples and held it far
  more reliably, but did not raise the ceiling.
