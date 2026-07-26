# The record as a memory system

Whoever picks up this project next — days later, or with no memory of the runs
at all — inherits exactly what is written here. Nothing else survives. That
makes the record not documentation but *working memory*, and it has to be
engineered like one.

The naive failure is assumed to be forgetting. It is not. The record already
holds the facts. The real failures are subtler and all three are fatal:

1. **Retrieval fires too late.** The record is indexed by when it was written,
   so finding the right lesson requires already suspecting it exists. Learning
   002 — *never warm-start a critic across a reward change* — is useful in
   exactly one instant: when someone is about to warm-start a critic. That is
   precisely the instant nobody goes looking.
2. **Nothing measures whether the reasoning is getting better.** Predictions
   were being written and scored one at a time, and the scores went nowhere.
3. **Volume becomes noise.** At two hundred lessons, "read the index" stops
   working, and the record's usefulness starts *declining* while it looks like
   it is growing.

Seven mechanisms, each aimed at one of those.

---

## 1. Guards — lessons the machine enforces

`src/bestiary/guards/` · `python -m bestiary.guards`

A lesson in prose is worth whatever someone's willingness to read it at the
right moment is worth. A lesson as an assertion is enforced whether anyone
remembers it or not.

**Every lesson that can become a check becomes a check.** Each guard names the
learning it enforces, and a failure quotes it:

| guard | enforces | catches |
|---|---|---|
| `ledger-schema` | learnings/007 | a row that hides instability behind a peak, or a single-seed result not marked provisional |
| `checkpoint-width` | learnings/003 | a checkpoint orphaned by an observation-list change |
| `metric-liveness` | episodes/003 | a metric that is silently constant and therefore measuring nothing |
| `standing-control` | learnings/001, 005 | a reward under which doing nothing outscores the trained policy |

This is the strongest of the seven, because it is the only one that does not
depend on anyone reading anything. `--fast` skips the guards that step physics
and runs in under half a second, which makes it usable as a gate in front of
every training launch.

> When these were first run, they immediately surfaced two things nobody knew:
> a hound checkpoint orphaned at 141 observations against a 169-observation
> env, and a run directory pointing at an env id that no longer exists.

## 2. Triggers — the record raises its hand

Every learning and decision carries a machine-checkable condition in its front
matter: `triggers:` — the situations in which it *must* be read.

```yaml
triggers: [warm_start, reward_change, critic_reset]
```

Before acting, the intended action is written down in the same vocabulary, and
matching entries are surfaced. This is **interception, not recall**: it works
precisely when you do not know what you do not know. Recall requires a
question; interception does not.

## 3. Calibration — knowing how often you are wrong

`research/calibration.jsonl` · `python -m bestiary.record.calibration`

Every prediction here is written before the result and carries an explicit
probability, so the entire history is scoreable. One row per resolved
prediction; the tool reports the Brier score and, more usefully, a reliability
table: do claims made at 70% come true about 70% of the time, and in which
direction does the bias run?

A record that stores only conclusions cannot tell you whether the person
writing them is any good. One that stores probabilities can.

## 4. The prior belief — recording *being* wrong

Facts transfer poorly. "The reward was gamed" is a fact and teaches almost
nothing, because the next broken reward will look just as reasonable.

So every learning states **what was believed immediately before, and why that
belief was reasonable**:

> I checked that walking beat standing on flat ground, concluded the reward
> was sound, and never re-checked when the terrain changed the ratio.

That is the transferable part — the shape of the mistake, not its subject.

## 5. Null results — what was already tried

`research/nulls.jsonl`

One line per dead end: what was tried, what happened, what it cost, and the
condition under which it would be worth trying again. Deliberately the cheapest
possible thing to write, because null results never feel like findings and
anything more expensive than one line will not get written.

The most valuable thing a long-running project can know is *I already tried
that*, and it is the least likely thing to be written down.

## 6. Anomalies — what was noticed and never explained

`research/anomalies.jsonl`

Observations that were surprising but not chased: a non-monotonic survival
band, a metric behaving oddly, a number that did not fit. Each carries why it
matters and the cheapest next step.

Without somewhere to put them, anomalies are silently discarded — and in
research they are disproportionately where the real findings come from.

## 7. Sweep and compaction — keeping the record readable

Periodically, a pass whose only job is maintenance:

- **Fired triggers.** Learnings whose falsifier has been observed; decisions
  whose revisit condition has been met.
- **Contradictions.** Two entries that cannot both be true.
- **Staleness.** Claims not re-confirmed in a long time, especially
  provisional ones being quoted as settled.
- **Compaction.** Clusters of related lessons synthesized into one higher-order
  principle. Originals stay, marked subsumed.

Compaction is what keeps retrieval working at scale, and skipping it is how a
record becomes an archive nobody reads.

---

## Retirement, not correction

A learning falsified later is **superseded by a new numbered learning**. The old
file gets one header line pointing at its successor and is otherwise untouched
— never edited, never deleted.

The wrong version staying readable is the evidence that the method catches its
own mistakes. A record that quietly self-corrects is indistinguishable from one
that was never wrong, and neither teaches anything.

## Provenance

Every number carries where it came from — a run log, a check output, a
committed script. This is the number rule, and it exists because prose
arithmetic is fluent, checkable, and unchecked. If a calculation is worth
recording it is worth four lines of Python, and the script ships in the commit.

## The self-test

Before retrieving, predict what the record will say. If the retrieval is
surprising, that gap is itself a finding: it means the record exists but is not
being internalized, which is the failure mode this whole document is built
against — and it is otherwise invisible.

---

## The files

| file | what | shape |
|---|---|---|
| `ledger.jsonl` | one row per finished run | append-only |
| `calibration.jsonl` | one row per resolved prediction | append-only |
| `nulls.jsonl` | one row per dead end | append-only |
| `anomalies.jsonl` | one row per unexplained observation | append-only, `status` may change |
| `learnings/` | one file per lesson, with triggers and a falsifier | superseded, never edited |
| `decisions/` | one file per choice, with a reversal trigger | superseded, never edited |
| `episodes/` | one file per cycle | snapshots, never edited |
| `guards/` (in `src/`) | the lessons that became assertions | code |

Everything append-only is append-only for the same reason: a process that
appends cannot lose what is already there, and one that rewrites can lose all
of it in a single bad write.
