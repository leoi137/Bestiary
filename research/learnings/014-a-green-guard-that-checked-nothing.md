---
triggers: [metric_added, refactor, comparison]
guard: measurement-provenance — but see below; this learning is partly ABOUT the limits of that guard, and the honest answer is that the general form (a guard reporting its own coverage) is not built
last_confirmed: 2026-07-28
---

# 014 — a guard can be green, fully tested, and check nothing at all

**Date:** 2026-07-28 · **From:** cycle 011, building `measurement-provenance`
**Robot:** n/a — this is about the record, not the machine

## What we believed before

This project's central method is that **a lesson becomes a guard**. Prose
depends on someone reading it at the right moment; an assertion depends on
nothing. `CLAUDE.md` says it, `research/MEMORY.md` says it, and thirteen guards
now gate every training launch. The belief that follows, and that nobody had
written down because it seemed too obvious to write down, is:

> *If the guard for lesson X is green, lesson X is being enforced.*

That was reasonable. It is how the other twelve guards behave. `checkpoint-width`
reads every run's `config.json` and compares hashes; `nulls` scans six runs
against four declared dead ends; `track-length-bias` runs seven assertions
against synthetic episodes it constructs itself. Each has an input set that is
non-empty by construction, so green really does mean checked.

## What happened

Cycle 011 built `measurement-provenance` to enforce `learnings/013` — a number
is only as durable as the artifact it came from. The build went well by every
signal available:

- **13/13 fast guards pass**, the new one included
- **`ruff --select F`** clean
- **18/18 oracle assertions**, deliberately exercising the *failing* direction
- `freeze_checkpoint` exercised on the live run's real 5,106,759-byte
  checkpoint, idempotent re-freeze in 19 ms

Then an independent refuter was asked to kill the claim, and did. Running
`research/scripts/measurement_provenance_coverage.py`:

```
guard verdict                         ALL PASS
assertions                            5/5 pass
assertions green on an EMPTY input    2/5

measurement JSONs (parseable)         12
  ...naming a checkpoint              9
  ...recording a sha256               0
checkpoint hashes ACTUALLY verified   0

real SAC.load CALLS (ast, not grep)   7
  ...that can publish a number        5
  ...of those, freezing first         2

ledger rows                           4
  ...carrying a checkpoint sha256     0
```

**Zero.** The guard whose entire purpose is to verify that committed
measurements still name their bytes was green while verifying the bytes of
nothing at all. Two of its five assertions were green because their input sets
were empty. Nine measurement JSONs named a checkpoint and none recorded a hash,
so there was nothing for the assertion to bite on. The ledger — the file this
project treats as authoritative — carried zero hashes in four rows.

None of that was visible in `All guards passed (13 guards)`.

## Why it happened

A guard is a function from an **input set** to a verdict. Everyone reasons
about the function and nobody reasons about the set.

Universally quantified statements are **vacuously true over an empty set**.
"Every committed measurement still matches its recorded hash" is true when no
measurement records a hash, in exactly the way "every unicorn in this room is
purple" is true. The assertion is correct, the implementation is correct, the
test suite is correct, and the conjunction proves nothing about the repository.

The oracle made this *harder* to see rather than easier. 18 assertions in a
`tempfile.TemporaryDirectory` where the author writes both the JSON and the
frozen bytes prove the function is right on inputs the author chose. They say
nothing about whether the real input set is non-empty — and the more
comprehensive the oracle, the more confident the green tick feels. The signal
that should have raised the alarm was *produced by the guard itself*, in its own
detail string, and read straight past:

```
0 verified, 0 not present locally, 0 MISMATCHED
```

That was printed, in full, on the first run. It was read as "no problems found"
when it says "nothing was examined". Those are opposite statements sharing a
number.

The deeper reason is a mismatch in when the two things become true. A guard is
written *at the moment the fix is written*, when by construction nothing in the
repository has yet adopted the new convention. So a newly written guard's input
set is almost always empty **on the day it is committed** — the day it is
inspected, trusted, and never looked at again. It only becomes load-bearing
later, and nobody is watching then.

## The math

Let $G$ be a guard asserting property $P$ over an input set $S$. What the guard
reports is

$$V(G) \;=\; \bigwedge_{x \in S} P(x)$$

and what a reader takes from $V(G) = \text{true}$ is "$P$ holds where it
matters". Those coincide only when $S$ covers what matters. Define **coverage**

$$c(G) \;=\; \frac{|S|}{|S^{*}|}$$

- $S$ — items the guard actually examined, dimensionless count
- $S^{*}$ — items the property must hold over for the lesson to be enforced
- $c$ — dimensionless, in $[0, 1]$

For `measurement-provenance` on 2026-07-28, taking $S^{*}$ as the JSONs that
name a checkpoint:

$$c \;=\; \frac{0}{9} \;=\; 0.00$$

The empty conjunction is the crux. By definition $\bigwedge_{x \in \emptyset}
P(x) = \text{true}$, so

$$V(G) \;=\; \text{true} \quad\text{and}\quad c(G) \;=\; 0$$

**simultaneously, with no contradiction.** The verdict carries no information
about the property whatsoever.

The second number is coverage of the *mechanism* rather than the check. Of the
7 real `SAC.load` calls, 5 can publish a number, and 2 freeze first:

$$c_{\text{mech}} \;=\; \frac{2}{5} \;=\; 0.40$$

So the claim cycle 011 nearly recorded — that this failure class is now
*impossible* — required $c_{\text{mech}} = 1$ and had $0.40$. Three research
scripts still load a mutable checkpoint, and those three wrote the very JSONs
now grandfathered into the guard. **The covered paths were not the ones with the
track record of causing the problem.**

Plainly: a green guard told us a property held over nine files, having looked at
none of them, while three of the five programs that could break the property
were not wired to it.

## What to do next time

**A guard must report the size of its input set, and a reader must look at it.**
Every `Finding.detail` here now leads with counts (`0 verified`, `9 JSON(s) name
a checkpoint`), which is necessary and — as this cycle proves — not sufficient,
because it was printed and ignored.

**Treat a newly written guard as unverified until its coverage is non-zero.**
The day a guard is committed is the day its input set is emptiest and its green
tick is least meaningful. `measurement-provenance` becomes load-bearing at the
next harvest, when the first hash-carrying JSON lands; until then it is a
promise, not a check.

**When claiming a class of bug is impossible, count the paths, with code.** The
grep said 13 sites; `ast` said 7 real calls, 5 of them publishing. Six of the
grep hits were docstrings *explaining an invariant* — this repo talks about
`SAC.load()` constantly. A text search cannot distinguish an invariant being
explained from one being used, and the difference was 2/13 versus 2/5.

**Say "reduced" unless coverage is 1.** The honest statement is: the two primary
eval tools now freeze; three research scripts do not; nothing forces a
measurement to be written at all (`anomalies` row 30), which is the mechanism
that actually caused rows 20/23/27.

## How we would know this is wrong

This learning claims the empty-input-set failure is **general** to this project's
guard method, not a one-off. It is wrong if:

- **A coverage audit of the other twelve guards finds every one has a non-empty
  input set**, and always did on its commit day. Then `measurement-provenance`
  was careless rather than symptomatic, and the general claim is overreach. This
  is cheap and has not been run — the honest state of the evidence is **one
  observed instance, generalised**. That makes this learning **provisional** in
  the sense the seed rule means it: $n = 1$.
- **`measurement-provenance` reaches coverage 1 and still misses a real
  overwrite.** Then the defect is the assertion, not the input set, and the
  framing here is wrong.
- **A vacuous guard is shown to be harmless** — i.e. every case where coverage
  was 0 was also a case where the property could not have been violated. This
  one is already close to refuted: coverage was 0 *while* three unwired scripts
  could violate the property freely.

The specific sub-claim that the *absence* branch is a hole
(`anomalies` row 31) would be wrong if frozen checkpoints turn out to be
retained reliably in practice. Nothing currently requires them to be; `runs/`
and `*.zip` are gitignored, so no frozen copy ever travels with the repository.
