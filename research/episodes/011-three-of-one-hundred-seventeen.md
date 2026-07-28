# 011 — Three of a hundred and seventeen

**Date:** 2026-07-28 · No GPU work. A training run was already in progress under
its declared ceiling and was left alone; this is what happened alongside it.

## Thesis

`learnings/014` claimed that a guard can be green, fully tested, and check
nothing — and generalised it from a single instance. It named its own falsifier:

> A coverage audit of the other twelve guards finds every one has a non-empty
> input set, and always did on its commit day. Then `measurement-provenance` was
> careless rather than symptomatic, and the general claim is overreach.

That audit had never been run, costs nothing, and `014` was about to be cited.
So: **what fraction of the guard suite actually checks something?**

## Diagnosis

You cannot measure this from outside the guards. A `Finding` carried
`(label, ok, detail)` — a binary verdict and a free-text string — so the only
way to recover *how much did this assertion examine* was to parse English out of
the detail string. That is precisely the hand-derived number this repo's number
rule forbids.

So the measurement and the fix are the same work. `Finding` gained `n`: the size
of the input set the assertion actually quantified over. `n = 0` renders as
**VACUOUS**, never `PASS`. `n = None` means *not set-quantified* — "at least 8000
MiB of RAM free" has no set behind it, and inventing one would be a lie.

**The ruling was declared before the count**, because otherwise the number is
unreproducible: where a guard passes while naming an uncheckable remainder
("9 runs predate the spec record"), `n` counts the **checkable set only**. A pass
over 2 verified and 11 grandfathered is `n = 2` — a real check with poor
coverage. A pass over 0 verified is `n = 0`, however honestly the remainder is
named.

## What happened

All 16 guards wired. Measured by
[`research/scripts/guard_vacuity.py`](../scripts/guard_vacuity.py):

| | |
|---|---|
| guards | 16 |
| assertions | **117** |
| set-quantified (vacuity-capable) | 111 |
| scalar thresholds / single named properties | 6 |
| **VACUOUS today** | **3** |
| guards containing them | **1** |
| vacuous outside `measurement-provenance` | **0** |

All three are in `measurement-provenance` — one more than that guard was
believed to have. Its assertion *"every non-grandfathered measurement naming a
checkpoint records its sha256"* reads `0 delinquent` while having verified
nothing, because all nine candidates are grandfathered.

Counting `Finding(` literals in the source would have been the wrong instrument:
most guards emit one finding per input item, so the assertion count is itself a
function of the input set — 95 static literals against 117 live assertions.

### Then an independent refutation killed the conclusion

The obvious reading — *`014` is overreach, `measurement-provenance` was merely
careless* — did not survive review, and the reason is that **the falsifier is a
conjunction**. Its second clause, *"and always did on its commit day"*, was never
tested by the audit. It is also false, on this repository's own timestamps:

| guard | committed | earliest run carrying the data it checks |
|---|---|---|
| `terrain-spec` | 2026-07-27 05:16 | 2026-07-27 08:04 |
| `reward-spec` | 2026-07-27 03:39 | 2026-07-27 04:40 |

Both guards were **vacuous on the day they shipped** and grew inputs afterwards.
Which inverts the result: *"3 vacuous today, all in the newest guard"* is what
`014` predicts once data accumulates, not evidence against it —
`measurement-provenance` is simply the youngest module in the registry. Vacuity
here looks like a function of a guard's **age**, not of its author's care.

An earlier version of the audit script printed `CONFIRMED` off the one clause it
can see. That text is gone; it now states which half it tests and refuses the
verdict.

The same review found two defects in the instrument itself:

1. **The convention split.** A finding whose label names one artifact got `n = 1`
   in some modules and `n = None` in others — the rule was applied three
   different ways. Not cosmetic: it made 26 of `checkpoint-width`'s 27 assertions
   invisible, so "72 set-quantified" was an artifact of the split rather than a
   measurement. Adjudicated to `n = 1` and swept; set-quantified went 72 → 111.
2. **A number that moves.** Total items verified read 6955, then 6956, then 6995
   within one day with no code change between the last two. The mover is the live
   training run writing into `runs/`. The vacuity count held at 3 throughout —
   but only because it is a threshold at zero, and the totals are not.

## How the prediction did

**0 for 3.** All three predictions were committed before the measurement ran.

| | claim | p | outcome |
|---|---|---|---|
| P1 | a vacuous assertion turns up in a module the hand read had cleared | 0.45 | **false** |
| P2 | total vacuous lands in 5–16 | 0.55 | **false** — measured 3 |
| P3 | at least one guard other than `measurement-provenance` is vacuous | 0.75 | **false** — zero |

P1 and P3 failed the same way: both assumed a hand read of the guard source was
roughly right. It was not, and its error ran *opposite* to the one predicted — it
called 10 of 16 modules vacuity-capable and concluded the falsifier looked
already refuted, by conflating *could go empty on a fresh clone* with *is empty
now*. Those are different questions and the script now reports them separately.

P2 missed low, and its own written alternative had named 2–4 — the band was
wrong, the reasoning behind it was not.

Hit rate 52% → 48%, Brier 0.2220 → 0.2291. The 40–60% band now stands at 1 of 7,
worse than answering "50%" at random, and it has been the worst band three
cycles running.

## What this leaves

**No learning was written.** The claim did not survive refutation, so it is an
open question, not an entry. The refutation's own finding — that guards are
vacuous *at birth* — is concrete and timestamped, but it is one pass and
untested, and promoting it now would repeat exactly the over-generalisation that
put `014` under review in the first place.

`014` stands, by a mechanism nobody had proposed: not carelessness, age.

The lasting output is that the question is now cheap to ask again —
`n`, the `VACUOUS` status, and a committed audit that says which half of the
falsifier it can see.
