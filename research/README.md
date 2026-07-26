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

## Two rules that apply to everything in this folder

**The seed rule — no effect is claimed from one run.** A comparison needs ≥3
seeds per arm, reported as mean and spread, with exactly one variable changed
between arms. Between-seed variance in SAC is comparable to most effects worth
claiming, so a single-seed difference is a hypothesis, never a finding. Single
runs are still worth doing as **probes** — they are written up with `seeds: 1`
and the word *provisional*, and they say what would settle the question.

**The number rule — no number enters the record unless code computed it.**
Every figure here comes from a run log, a check script, a measurement command,
or a small script committed alongside it. Arithmetic done in prose is fluent,
checkable, and unchecked; if a calculation is worth recording it is worth four
lines of Python.

## learnings/

One file per lesson, written when something surprises us. Written for a human
who is learning robotics *and* for a model resuming with no memory. The
writing standard — plain English, defined jargon, real math worked with the
run's real numbers — is in `learnings/README.md` and is not optional.

Every learning carries a **"How we would know this is wrong"** section, and
that section is a live trigger: when a later cycle observes the falsifier, the
learning is **superseded by a new numbered learning**. The old file gets one
header line pointing at its successor and is otherwise untouched — never
edited, never deleted. A record that only accumulates becomes wrong in ways
nobody can find; one that edits its mistakes away destroys the evidence that
the method works.

Learnings teach *a mechanism that surprised us*, to a reader following this
project. `../docs/lessons/` teaches *a concept from scratch*, briefly, to
someone learning the field. The same event often deserves one of each; no file
should try to be both.

## decisions/

A decision, the reasoning behind it, and **the observation that would
reverse it**. The trigger is the whole point: without it, a settled question
gets re-argued every few weeks by whoever has the most recent context. With
it, checking whether a decision still holds takes seconds.

## episodes/

One file per cycle of the research loop: what was tried, **what was predicted
before the result was known**, and what actually happened.

Episodes are snapshots. Once written, an episode is not edited to match how
things turned out — a prediction that was wrong is more informative than one
quietly corrected. If the diagnosis changes, write the next episode.

## ledger.jsonl

One JSON object per line, one line per finished run. **Append only.** Never
rewrite this file: a process that rewrites can clobber the whole record on a
crash or a race, whereas one that appends cannot lose what is already there.

Fields: `run`, `date`, `robot`, `env_id`, `algo`, `wrapper`, `seed`, `steps`,
`wall_clock_s`, `fps`, `best_eval_return`, `final_ep_rew_mean`,
`final_ep_len_mean`, `final_ent_coef`, `verdict`, `notes`.

**Required from row 3 onward**, four more fields:

- `mean_eval_after_converge` and `eval_crash_rate`, because
  `best_eval_return` is a maximum over a noisy sequence — it rewards
  instability and grows with run length. Comparing two runs on it alone ranked
  a policy that scored 1218 once and 390 repeatedly *above* one that reliably
  scored 1170. See
  [learning 007](learnings/007-peak-score-hides-an-unreliable-policy.md).
- `seeds` (how many seeds the row summarizes) and `provisional` (true when
  `seeds` is 1, or when more than one variable differed from the run being
  compared against).

Rows 1 and 2 predate all four. They carry the stability numbers in `notes`
instead, and both are `seeds: 1` — so the PD-versus-torque comparison they
support is provisional twice over, since the two runs also differed in step
budget (3.75M vs 1M).

`verdict` is one of `plateau`, `improved`, `regressed`, `crashed`,
`inconclusive` — a coarse judgement so the history can be scanned without
reading every note. A result that failed its refutation pass is
`inconclusive`, whatever the numbers looked like.

## The cycle contract

One research cycle does exactly this, and stops:

1. Read `ledger.jsonl` and the open questions inherited from the last cycle.
2. Choose **one** experiment.
3. Write the falsifiable prediction — **before** running anything.
4. Run it.
5. Have it **refuted** — an independent pass whose job is to kill the
   conclusion by checking for a wrong metric, dead instrumentation, a
   confound, seed noise, or a peak compared against a mean.
6. Append **one** row to the ledger.
7. Write **at most one** learning — only if something surprised us, and only
   if it survived step 5.
8. Write the episode.

Bounded on purpose. An unbounded process generates motion rather than
knowledge.

Step 5 exists because whoever picks the metric, predicts the result, runs it,
and grades it has four chances to be generous and none to be caught.
`eval/mean_idle_legs` logged exactly 0.00 through two complete runs before
anyone noticed it was fed from a key neither run set.
