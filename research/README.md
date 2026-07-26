# Research

The written record. Weights are disposable; this folder is not.

Four kinds of artifact, deliberately kept separate because they rot at
different rates and answer different questions.

```
learnings/     lessons that outlive any single run       — timeless
decisions/     choices made, each with a reversal trigger — until the trigger fires
episodes/      one file per loop cycle                    — a snapshot, never edited
ledger.jsonl   one row per finished run                   — append-only
CORE_PLAN.md   the locked reward and observation spec
```

## learnings/

One file per lesson, written when something surprises us. Written for a human
who is learning robotics *and* for a model resuming with no memory. The
writing standard — plain English, defined jargon, real math worked with the
run's real numbers — is in `learnings/README.md` and is not optional.

## decisions/

A decision, the reasoning behind it, and **the observation that would
reverse it**. The trigger is the whole point: without it, a settled question
gets re-argued every few weeks by whoever has the most recent context. With
it, the loop can check the trigger in seconds and move on.

## episodes/

One file per cycle of the research loop: what was tried, **what was predicted
before the result was known**, and what actually happened.

Episodes are snapshots. Once written, an episode is not edited to match how
things turned out — a prediction that was wrong is more informative than one
quietly corrected. If the diagnosis changes, write the next episode.

## ledger.jsonl

One JSON object per line, one line per finished run. **Append only.** Never
rewrite this file: an unattended process that rewrites can clobber the whole
record on a crash or a race, whereas one that appends cannot lose what is
already there.

Fields: `run`, `date`, `robot`, `env_id`, `algo`, `wrapper`, `seed`, `steps`,
`wall_clock_s`, `fps`, `best_eval_return`, `final_ep_rew_mean`,
`final_ep_len_mean`, `final_ent_coef`, `verdict`, `notes`.

**Required from row 3 onward:** `mean_eval_after_converge` and
`eval_crash_rate`. `best_eval_return` is a maximum over a noisy sequence, so
it rewards instability and grows with run length — comparing two runs on it
alone ranked a policy that scored 1218 once and 390 repeatedly *above* one
that reliably scored 1170. Rows 1 and 2 predate these fields and carry the
numbers in `notes` instead. See
[learning 007](learnings/007-peak-score-hides-an-unreliable-policy.md).

`verdict` is one of `plateau`, `improved`, `regressed`, `crashed`,
`inconclusive` — a coarse judgement so the history can be scanned without
reading every note.

## The episode contract

One cycle of the loop does exactly this, and stops:

1. Read `ledger.jsonl` and the open questions in the latest episode.
2. Run **one** experiment.
3. Append **one** row to the ledger.
4. Write **at most one** learning.
5. Write the next episode, including a falsifiable prediction for what comes
   after.

Bounded on purpose. An unbounded loop generates motion rather than knowledge.
