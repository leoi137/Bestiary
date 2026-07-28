# 010 — A green guard that checked nothing

**Date:** 2026-07-28 · **Run:** none launched; `hound_track_rel_s1` inherited mid-flight

## Thesis

A checkpoint this project measures is overwritten in place while it is being
measured. Freeze it to a content-addressed copy first, measure the copy, record
the hash, and guard the record — then a published number can always be traced
back to the exact bytes that produced it.

## Diagnosis

`ant_sac_best.zip` is rewritten every time the eval callback beats its previous
best. That had already cost this record twice:

- a refutation re-measured a *different* checkpoint under the same filename and
  noticed only because someone happened to hold the old hash
- a published ratio of ×1.47 turned out to be a property of one vanished
  checkpoint's crash rate. The measurement JSON was committed at 10:58:13 and
  the file it named was overwritten at **11:07:01**, nine minutes later. Three
  cycles reasoned from that number.

The obvious fix had already been tried and had already failed. Hashing before
and after a measurement proves the artifact was stable *during* it, not that it
is retrievable *afterwards* — and in the actual incident the two hashes agreed
and the file was gone forty minutes later. So immutability has to come from the
copy, not from a check wrapped around a mutable read.

## What happened

The mechanism went in as designed: `record/freeze.py` copies a checkpoint to
`runs/<name>/measured/<sha256>.zip`, so the path *is* the hash and a wrong file
cannot hide behind a right name. Both eval tools now freeze before loading and
load the frozen copy. Re-freezing unchanged bytes is a no-op measured at 19 ms.
A guard, `measurement-provenance`, and a 16-assertion oracle exercising the
failing direction followed.

Every signal was green — 13/13 fast guards, `ruff` clean, 16/16 oracle, the
freeze exercised on a real 5,106,759-byte checkpoint.

**Then an independent refuter was asked to kill the claim, and did.**

The guard was verifying **zero** artifacts. Two of its five assertions were
green because their input sets were empty. Nine measurement JSONs named a
checkpoint and none recorded a hash. The ledger — the file this project treats
as authoritative — carried zero hashes across four rows, because it called the
newly-freezing eval function, then discarded the hash and wrote a bare mutable
filename. And the freeze covered 2 of the 5 code paths that can turn a
checkpoint into a published number; the 3 unwired ones were **the research
scripts that produced the numbers in the original incidents**.

An assertion quantified over an empty set is vacuously true. `V(G) = true` and
coverage `= 0/9` hold simultaneously, with no contradiction. The guard printed
`0 verified, 0 MISMATCHED` on its very first run and it was read as "no problems
found" when it says "nothing was examined".

Three defects were fixed in-cycle: the ledger now records which checkpoint each
eval number came from; the guard now walks nested measurement blocks, without
which it was blind to `greedy_eval`'s *default* output shape; and two regression
assertions were added, taking the oracle to 18. Three holes it could not close
were recorded as anomalies rather than chased.

## How the prediction did

Three were committed before any work started. One resolved this cycle.

**P2 — "no measurement JSON records a checkpoint hash" (p = 0.75): FALSE.** One
already did, under a non-standard truncated field name, and it is a correct
prefix of the hash in a committed sidecar. Being wrong changed the build: the
guard now also asserts that a truncated hash agrees with the full one in the
same file, an assertion that would not exist had the prediction held.

The claim is also a lesson in writing them. It named a *field*, and no file
carries a field by that name — read literally it is **true**, and it is graded
false only on the reading it was written to test. Recorded as ambiguous rather
than resolved quietly, because picking a reading after seeing the answer is
exactly what a calibration record exists to prevent.

**P1** (the live checkpoint is overwritten again before the run ends, p = 0.80)
and **P3** (the run finishes inside its ceiling, p = 0.70) resolve at the next
harvest. At the time of writing the checkpoint's hash is unchanged and the run
is at 863,933 of 2,000,000 steps at 111.9 steps/s.

## What this cost, and what it is worth

The headline claim — that this class of failure is now *impossible* — did not
survive, and should not have. The honest statement is narrower: **the two
primary eval tools and the ledger now carry checkpoint identity; three research
scripts do not; and nothing forces a measurement to be written to disk at all**,
which is the mechanism that actually caused the original incidents.

The cycle's real finding is the one it did not go looking for. This project's
central method is that a lesson becomes a guard, and thirteen guards now gate
every launch. That method has a failure mode nobody had written down: a guard's
input set is emptiest on the day it is committed, which is the only day anyone
inspects it. Green meant "correct on inputs the author chose", and was read as
"the property holds here". Written up as
[learnings/014](../learnings/014-a-green-guard-that-checked-nothing.md),
provisional at one observed instance, with the cheap falsifier named: audit the
other twelve guards' input sets.
