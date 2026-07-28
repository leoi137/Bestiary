# 013 — An account that did not add up

**No training run.** The card was cold for this whole episode, and that is the
first thing to say about it.

## Thesis

A published decomposition of a reward is a claim about *where the return came
from*. It is only a claim at all if the parts add up to the whole. Ours did
not, on the environment we are currently running, and nothing in the output
said so.

## Diagnosis

`record/track_eval.py` split a tracking run's return into four named terms:

    reward_track   reward_ctrl   reward_contact   reward_termination

That list was a constant typed into the module. `HoundPDTrackDesert-v0` does
pay exactly those four. `HoundPDTrackRelDesert-v0` — the environment behind
the most recent ledger row — pays **five**: it gained `reward_shaping` when
the potential-based shaping term landed, and the instrument was never told.

So every `drive_grid_reward_*` field it printed was a partial account of the
return, formatted identically to a complete one. The measurement files carry
no field naming which decomposition they are, so a reader has no way to
notice.

The size of the gap, on the measurements actually committed:

| arm | sum of the four reported terms | drive-grid return | residual |
|---|---|---|---|
| zero action | 4.685656 | 3.911829 | **+0.773827** |
| trained (`_best`) | 20.926235 | 19.733243 | **+1.192992** |

Both sides of that table are means over the same drive cells, and the mean is
linear, so the residual is arithmetic rather than an estimate.

This is not a cosmetic defect. The previous ledger row's entire published
conclusion was a decomposition argument — *the control cost is 102.9 % of the
gap to zero action* — and the next two seeds were going to be read the same
way.

## What happened

The constant is gone. Reward terms are now **discovered** from the
environment's own step info — every key prefixed `reward_`, in the order the
environment declares them — and asserted to **sum to the reward that was
actually paid, on every step**. A term the instrument cannot see is now a loud
failure naming the step and the values, instead of a quiet subtraction from a
total nobody checks.

Discovery alone would not have been enough. It would still be silent if an
environment logged a term it does not pay, or paid one it does not log; the
sum check is what makes the class of bug impossible rather than this instance
of it fixed. The two rules also cover each other — a stray non-reward key
matching the prefix would be caught by the sum, and a genuine term that failed
to match the prefix would be caught by it too.

Deriving the list from the reward *specification* was the obvious route and
was rejected: the spec's names are the reward's vocabulary (`track_cmd`,
`pbrs_shaping`) and the info keys are the instrument's (`reward_track`,
`reward_shaping`). Bridging them needs a hand-written mapping table, which is
a second hardcoded list with exactly the failure mode being removed.

Verification, all of it in `research/scripts/decomposition_completeness.py`:

- on the five-term environment the new decomposition closes to **−1.96e−14**,
  against an old four-term residual of **+0.703925** on one zero-action
  episode;
- on the four-term environment the old and new sums are **bit-identical** —
  the same float additions in the same order, checked across 7 cells × 3 seeds
  — so **no previously published four-term number moves**, including the
  102.9 % figure;
- handed a term list missing `reward_shaping`, the assertion raises.

The per-step cost is 0.82 µs against `env.step`'s 938.8 µs — 0.09 %.

## What the refutation took

An independent check of this work refuted one of its six claims, and it was
the one that mattered.

**The fix had no guard.** `discover_terms` and `assert_decomposition_complete`
were exercised only by a research script that nothing runs automatically, so
the guard suite would have stayed green with the bug reinstated. By this
repository's own standard — *a change is not done until the thing that would
catch its regression exists* — the change was not done.

It also found two more hardcoded four-term lists still in the tree, both
consuming exactly the fields the defect is about: one inside a script whose
output is published in a lesson, one inside the demo written to illustrate the
defect. Both are fixed.

Both findings produced `guards/decomposition-completeness`, which asserts the
mechanism on synthetic inputs whose answers are known by construction *and*
asserts that every committed measurement's per-term means sum to its own
aggregate. Three measurement files were produced by the defect and cannot be
corrected without re-running a grid evaluation; they are grandfathered by name
and their residuals are **printed on every run**, so the omission is disclosed
in code rather than silent. A permanently red launch gate gets bypassed, which
is worse than the defect.

The refutation also corrected a number in this episode's own draft. The first
write-up led with *"343.5 % of the return"*. That is a ratio against a return
of −0.205 and it is not a fair summary: across 7 cells × 3 seeds the absolute
residual spans +0.043 to +8.37 and the relative one spans 0.9 % to 1175 %,
depending entirely on which denominator is chosen. The honest figures are the
absolute residuals in the table above.

## How the prediction did

**There was none, and that is a defect in the episode rather than a detail.**
No probabilistic claim was written down in advance, so nothing was appended to
`calibration.jsonl` and nothing could be scored. The six claims this work
made were adversarially graded *after* the fact — five survived, one was
refuted — which is a useful result and is not calibration.

The rule exists precisely for work like this, where the outcome feels certain
while it is being written.

## Also landed

- **Lesson 013 — what an observation is, and why its width is a one-way door**,
  taken off the head of the planned queue, with the arithmetic behind it
  reading every environment's spec live rather than trusting prose.
- `ant_sac_best.txt` now records the **step** a best checkpoint was written
  at, not only its score. The previous cycle had to recover "~1.48M steps"
  from file modification times to discover its own comparison was confounded
  with 520k steps of extra training.
- *Seven runs, plainly* — a dated briefing on the first seven runs, written
  earlier and carried untracked for six cycles — is committed.
