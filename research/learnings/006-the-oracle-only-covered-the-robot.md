---
triggers: [refactor, metric_added]
guard: none — coverage of an entry point is proven by running it, not by asserting on it
last_confirmed: 2026-07-26
---

# 006 — Our regression oracle covered the robot, not the trainer

**Date:** 2026-07-25 · **From:** the PD-position-targets refactor, cycle 002
**Robot:** n/a — this is about the test suite, not the machine

## What happened

The repository was restructured into an installable package: 23 files moved,
every import rewritten, and all filesystem paths centralised into a new module
called `bestiary.paths`.

The refactor was verified carefully, and everything checked passed:

- `robots/hound/check.py` — **38/38 assertions**
- `robots/spyder/check.py` — physics **bit-for-bit identical** over 2,000 steps
- Both Spyder rollout hashes unchanged (`ebd1224c`, `cd8f2b2d`)
- `build.py` regenerated both model XMLs **byte-for-byte**
- Every module imported cleanly, from a directory outside the repo
- `python -m bestiary.train.train --help` printed its usage text

Then the first actual training launch died immediately:

```
File "src/bestiary/train/train.py", line 250, in main
    run_dir = paths.RUNS / args.run_name
              ^^^^^
UnboundLocalError: cannot access local variable 'paths'
    where it is not associated with a value
```

**Training could not start at all.** Every check above passed anyway.

## Why it happened

Two separate causes, and the second one is the lesson.

### The immediate cause: a name collision

`main()` already contained this, from long before the refactor:

```python
run_dir = paths.RUNS / args.run_name      # line 250 — reads the MODULE
run_paths = _run_paths(run_dir)           # line 252 — was: paths = ...
```

The codebase already used `paths` as the name for a dict of per-run artifact
paths. The refactor introduced a *module* with the same name.

Python decides whether a name is local or global **at compile time, for the
whole function body**. Because `paths` was assigned anywhere inside `main()`,
every mention of `paths` in `main()` became local — including the one on the
line *before* the assignment. The module-level import was invisible from
inside that function.

This is why the error is `UnboundLocalError` and not `NameError`. Python knew
exactly which variable was meant; it just had not been given a value yet.

### The real cause: nothing we ran executed that line

This is the part worth carrying forward.

`robots/hound/check.py` is an excellent oracle — 38 assertions, real
measurements, rollout hashes that catch a physics change to the bit. It gave
a strong, and entirely accurate, signal: **the robot was unchanged.**

It says nothing about the trainer, because nothing in it imports `train.py`.
The suite covers `build.py`, `envs/`, and `terrain/`. The two *entry points* a
person actually types — `train` and `watch` — had no coverage at all.

And `--help` is worse than no test, because it *looks* like a test. `argparse`
calls `sys.exit()` when it prints usage, which happens **before** line 250.
The smoke test exercised argument parsing and then quit, one line short of the
bug, while reporting success.

## The math

No physics here. But there is a precise rule, and it is worth stating exactly
because it is counter-intuitive:

> For a function body `B`, Python computes the set of local names
>
>     L(B) = { n : n is the target of an assignment, `for`, `with ... as`,
>                  `import`, `def`, or `class` anywhere in B }
>
> Every occurrence of `n ∈ L(B)` inside `B` resolves to the local binding,
> **regardless of position**. Enclosing and module scopes are not consulted
> for those names at all.

So visibility is decided by the *presence* of an assignment, not by its
*position*. Reading a global on line 250 and assigning the same name on line
252 is not "read then shadow" — it is one local variable, read too early.

Worked on the real code, with `B = main()`:

```
L(main) ∋ paths        (because line 252 assigned it)
line 250: paths.RUNS   -> resolves to LOCAL paths -> unbound -> raises
```

Moving the assignment above the read would have "fixed" it while leaving the
collision in place. Renaming the local removes the collision itself.

## What to do next time

**1. A linter catches this class of bug for free — so run one.**

`ruff check --select F` flags it in milliseconds:

```
F823 Local variable `paths` referenced before assignment
```

Ruff is now a `dev` extra and `[tool.ruff.lint] select = ["F"]` is in
`pyproject.toml`. Style rules (`E`/`W`) are deliberately left off: the
comments in this codebase are load-bearing prose and reflowing them to satisfy
a linter would damage them. `F` is pyflakes — it finds real bugs, not opinions.

Run before committing any refactor:

```bash
python -m ruff check --select F src/ concepts/
```

**2. "It imports" and "`--help` works" are not coverage of an entry point.**

Both stop before the code that matters. The cheapest honest check on a
trainer is to *actually train*, for a couple of hundred steps. That runs
`main()` end to end and would have caught this in about sixteen seconds.

**3. When a refactor introduces a new module name, grep for it as a local
variable first.**

`grep -rn "\bpaths\s*="` across the codebase, before naming the module
`paths`, would have shown the collision immediately. Generic module names —
`paths`, `config`, `utils`, `types`, `data` — are exactly the words already in
use as local variables.

**4. Know what the oracle actually covers.** Ours is a *robot* oracle and is
very good at that. It was quietly being treated as a *repository* oracle. A
green suite means "the thing the suite tests is fine", never "the change is
fine" — and the gap between those is where a long run fails overnight, having
reported success on the way in.

## How we would know this is wrong

- If `ruff --select F` starts producing enough false positives on this
  codebase that it gets ignored, the rule is not carrying its weight and the
  argument for it collapses.
- If a short real training run turns out to be too slow or too fragile to use
  as a routine check, then point 2 is wrong and the right answer is a proper
  unit test around `main()`'s setup rather than a live run.
- If a future entry-point bug slips through *with* ruff and a smoke run both
  in place, the diagnosis here was too narrow — the problem would be coverage
  in general, not these two specific gaps.

## See also

- `research/episodes/002-pd-position-targets.md` — the cycle this happened in
- [003 — Changing the observation list throws away every checkpoint](003-obs-list-is-a-one-way-door.md)
  — the other lesson about a change whose blast radius was wider than it looked
