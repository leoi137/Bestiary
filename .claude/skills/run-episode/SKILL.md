---
name: run-episode
description: Run ONE bounded cycle of the Bestiary research loop — read the ledger and open questions, pick a single experiment, write a falsifiable prediction BEFORE running it, record the result, and write the next episode. Use when advancing the robotics research loop, or as the body of a /loop. Never starts a training run without explicit authorization.
---

# run-episode

One cycle. Bounded on purpose: an unbounded loop generates motion rather than
knowledge.

> **Hard rule — no unattended training.** A training run takes hours and holds
> the only GPU (one RTX 2080). This skill may *propose* a run and prepare
> everything for it, but it must not start one unless the invoking instruction
> explicitly authorizes it. If the cycle's chosen experiment needs GPU time and
> you do not have that authorization, stop at step 3, write the prepared
> experiment into the episode, and hand back.

## The contract

Exactly this, then stop:

1. Read state.
2. Choose **one** experiment.
3. Write the prediction **before** running it.
4. Run it (only if authorized).
5. Append **one** ledger row.
6. Write **at most one** learning.
7. Write the next episode.
8. Commit each of the above as its own commit.

## 1 — Read state

- `research/ledger.jsonl` — tail it. What has already been tried, and what came
  of it. Never re-run an experiment already in the ledger without saying why.
- `research/episodes/` — the highest-numbered file. Its **Open questions**
  section is what this cycle inherits.
- `research/decisions/` — scan the **triggers**. If a trigger has fired, that
  is this cycle's work: supersede the decision. If none has, do not re-argue
  any of them.
- `research/learnings/` — the index. Do not repeat a mistake already written
  down here.

Check the ledger's last row against reality before trusting it. If a run
finished after the row was written, the row is stale.

## 2 — Choose one experiment

One. The highest-leverage open question, not the easiest.

Prefer the experiment that would **change what we do next** regardless of
outcome. An experiment whose every result leads to the same next action is not
worth the GPU time.

State in one sentence what it tests. If that needs the word "and", it is two
experiments — pick one.

## 3 — Write the prediction first

Before running anything, write into the draft episode:

- What you expect, with **numbers and ranges**, not adjectives.
- A confidence, as a percentage.
- The most likely *alternative* outcome, also with numbers.
- What result would mean the current diagnosis is wrong.

"It should do better" is not a prediction. "A plateau between 500k and 1.5M
steps, with `ent_coef` under 0.02 by 60k" is.

This is the step that makes the record evidence rather than narration. Written
after the fact, everything sounds reasonable.

## 4 — Run it

Only with explicit authorization. Follow the commands in `CLAUDE.md`. While it
runs, do work that does not need the GPU — a theory note, a learning rewrite,
a check.

## 5 — Append one ledger row

One JSON object, one line, **appended**. Never rewrite `ledger.jsonl`: a
process that appends cannot lose what is already there; one that rewrites can
lose all of it on a crash or a race.

Fields are listed in `research/README.md`. `verdict` is one of `plateau`,
`improved`, `regressed`, `crashed`, `inconclusive`.

Put the real numbers in. A row without `wall_clock_s` and `fps` cannot be used
to argue about throughput later.

## 6 — Write at most one learning

Only if something **surprised** you. A run that went as predicted confirms the
model and belongs in the ledger, not in `learnings/`.

If there is a learning, use the `write-learning` skill — it enforces the
standard, which includes real math worked with this run's real numbers.

At most one. Two "learnings" from one cycle usually means one of them is a
restatement.

## 7 — Write the next episode

`research/episodes/NNN-short-title.md`, following the shape in that folder's
README: Thesis · Diagnosis · Ranked actions · Prediction · Open questions.

**Never edit a previous episode to match how things turned out.** If the last
episode's prediction was wrong, say so in this one and explain what the
diagnosis missed. That is the most valuable paragraph in the file.

## 8 — Commit

Use the `commit-push` skill. One commit per artifact — the ledger append, the
learning, the episode. These are the units the record is read in, so they are
the units it is committed in. A single commit landing all three destroys the
ability to see when each was understood.

## When to use a subagent

Delegate when the work would flood this context with material you do not need
to keep:

- **Reading a long run log or TensorBoard export** to extract a few numbers.
- **Literature or method search** — use the `robotics-research` skill.
- **Surveying several files** to answer one question.

Ask the subagent for the conclusion and the numbers, not the file contents.
Keep the decision in the main thread.

## Failure modes this skill exists to prevent

- **Drift.** Doing many small things, none of which answers a question. Fixed
  by: one experiment per cycle.
- **Narration.** Writing up results as though they were expected. Fixed by:
  the prediction is written first.
- **Re-litigation.** Re-arguing a settled decision because the context is
  fresh. Fixed by: decisions have triggers; check them, do not reopen them.
- **Silent loss.** Rewriting the ledger and clobbering it. Fixed by:
  append-only.
