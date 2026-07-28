# 010 — Why a test can pass without testing anything

**One sentence:** The statement *"every X has property P"* is **true when there
are no X at all**, so a check written that way reports success both when
everything passed and when nothing was looked at — and those two outcomes are
indistinguishable from the outside.

## The idea

Say *"every checkpoint in this folder still matches its recorded hash."* If the
folder holds five checkpoints and all five match, the statement is true. Now
empty the folder. Is it still true?

Yes. And this is not a technicality or a quirk of any programming language — it
is what the word *every* means. **"Every X has property P" is shorthand for
"there is no X that violates P".** An empty collection cannot supply a
violator, so there is nothing to make the claim false, so it is true. Logicians
call this **vacuous truth**: true because empty, not because verified.

The trap is that a normal, verified success and a vacuous one produce the *same
word*. A check prints `PASS` either way. A reader sees `PASS` and concludes
*this was examined and it was fine*, when the honest reading may be *this was
not examined*.

Two different questions are being confused, and it is worth naming them
separately:

- the **verdict** — did anything violate the rule? (`PASS` / `FAIL`)
- the **coverage** — how many things did it actually look at?

A check reports the first. Until this repo's cycle 012 it did not report the
second at all, and a verdict without a coverage number cannot distinguish
*thorough* from *blind*.

## The math

Write the assertion as a quantifier over a set S with property P:

    for all x in S: P(x)     is equivalent to     there is no x in S: not P(x)

When S is empty there is no x to choose, so the right-hand side is trivially
satisfied and the whole statement is **true regardless of what P says** — even
for a P that nothing could ever satisfy.

Define **coverage** as the fraction of the things the assertion *speaks about*
that it actually read:

    c = verified / claimed

Now the real case. On 2026-07-28 this repo's `measurement-provenance` guard
asserted *"every frozen checkpoint still hashes to what its measurement
recorded."* Nine measurement files named a checkpoint. Zero of them had
recorded a hash, so zero could be checked:

    size of S  = 0        checkpoints with a recorded hash to compare
    claimed    = 9        measurement files the assertion speaks about
    verified   = 0
    c          = 0 / 9    = 0.00
    verdict               = PASS

Both lines are correct at the same time. The verdict is a true statement about
an empty set; the coverage is zero. There is no contradiction to notice, which
is exactly why nobody noticed. The guard even printed its own count —
`0 verified, 0 MISMATCHED` — and that was read as *no problems found* when it
says *nothing was examined*.

Measured across the whole suite by
[`scripts/010_vacuous_truth.py`](scripts/010_vacuous_truth.py): **117
assertions, 111 of them set-quantified, 3 vacuous — 2.7%**, all three in that
one guard.

**What it means:** a green check is a claim about the things it looked at, and
says nothing whatever about the things it did not. If it looked at none, green
is the correct output and it carries no information.

## Where it bites here

`src/bestiary/guards/__init__.py`. Every assertion now declares `n`, the number
of things it examined, and the runner renders `n = 0` as **VACUOUS** rather than
`PASS`. `n = None` means *not set-quantified* — a threshold like "at least 8000
MiB of RAM free" has no set behind it, and claiming one would be a lie.

Vacuous is deliberately **not** a failure: `runs/` is gitignored, so a fresh
clone genuinely has nothing to check, and failing there would make the suite
useless to anyone who cloned this repo. It is a third status, and the point is
only that it can no longer hide inside the word `PASS`.

What it cost to learn: the guard above shipped green, with its own 18-assertion
test suite also green, having verified nothing — and the failure it was written
to prevent had already put a wrong number into a published write-up that three
later cycles reasoned from.

## If you want to go deeper

[`research/learnings/014`](../../research/learnings/014-a-green-guard-that-checked-nothing.md)
— the incident, and the still-open question of whether a guard is simply vacuous
on the day it is written, before any data exists for it to check.
