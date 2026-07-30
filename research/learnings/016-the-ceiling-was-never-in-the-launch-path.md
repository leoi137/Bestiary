---
triggers: [new_machine, refactor, long_run]
guard: none — proposed in "What to do next time" and NOT YET WRITTEN. This is a
       TODO, not an answer, and it is recorded as one deliberately.
last_confirmed: 2026-07-30
---

# 016 — The thread ceiling was real, and it was never in the launch path

**Date:** 2026-07-30 · **From:** the first attempt to run
`Bestiary-Desert-Coarse-Anymal-C-v0` on a second machine, a host with 48 CPU
cores
**Robot:** n/a — this is a launch-path defect, not a physics one

Every local number below is printed by
`research/scripts/016_openblas_fork_math.py`, which measures this venv rather
than quoting documentation. The numbers from the other machine are labelled as
reported, because that is what they are: one observation, not a sweep.

## What we believed before

That the project's CPU ceiling was covered. `CLAUDE.md:28` sets it as part of
hard rule 3 — *≤6 torch threads (`OMP_NUM_THREADS=6`)* — the workspace's
resource-ceiling document repeats it in its table, and the MuJoCo launch line
written down for a training run carries it literally:

    OMP_NUM_THREADS=6 nohup python -m bestiary.train.train ...

That is not a ceiling somebody forgot to write down. It is one of the
most-repeated numbers in the project, repeated because hard rule 4 says the
machine is shared and overloading it is a correctness problem rather than a
performance one.

The belief was reasonable and it was also, on the MuJoCo side, true. What made
it false is that a **second launch path** was added — the Isaac Lab entry point,
`src/bestiary/isaac/train_desert.py` — and it was written without that prefix,
because nothing in the repository is capable of noticing. Grepping both
repositories for `OMP_NUM_THREADS` returns **five documents and zero lines of
code**: one of them is this repository's `CLAUDE.md:28`, and the other four are
prose or a copy-pasteable launch line. There is no `os.environ` assignment, no
launch wrapper, and no assertion anywhere in `src/` — grep it and see.

**The ceiling lived only in prose, and prose does not run.** That is the shape of
the mistake, and it is the transferable part: a resource ceiling that is
documented in five places and executed in none is not a ceiling, it is a wish.

## What happened

Reported from the second machine, 2026-07-29. Training died with **SIGSEGV,
exit status 139**, roughly 3 seconds into startup — before the first iteration,
before the terrain mesh was built, before anything the run was for. It died
identically at **64 environments and at 4096**, which is what ruled out every
scaling explanation in one step: a memory or parallelism problem that is
indifferent to a 64× change in problem size is not a memory or parallelism
problem.

Exit 139 is `128 + 11`, the shell's encoding of a process killed by signal 11,
`SIGSEGV`.

The crash backtrace, from Kit's own breakpad handler, innermost frame first:

    pthread_join
    libscipy_openblas...!blas_thread_shutdown_
    __register_atfork
    __libc_fork
    libomni.platforminfo.plugin.so

Read outward, that is: an Omniverse Kit plugin called `fork()`; glibc's `fork`
ran the handlers that had been registered with `__register_atfork`; one of those
handlers belonged to OpenBLAS and its job is to tear down the BLAS thread pool;
and that teardown called `pthread_join`, where the process died.

**The fix was one environment variable.** With `OPENBLAS_NUM_THREADS=1`
prepended and nothing else changed, the identical command reported **exit 0**,
ran **30 iterations**, and printed a mean reward.

Three of the four frames in that backtrace are verifiable from a file sitting in
this venv right now, on this machine, with no Isaac Lab involved at all:

    /home/.../venv/lib/python3.13/site-packages/numpy.libs/libscipy_openblas64_-32a4b2a6.so
      defines  blas_thread_shutdown_    True
      imports  __register_atfork        True
      imports  pthread_create           True
      imports  pthread_join             True

## Why it happened

Three separate facts have to be lined up, and the middle one is the one that
misleads.

**One — `fork()` from a multithreaded process gives the child exactly one
thread.** This is POSIX, not a bug. The child is a copy of the calling thread
only; every other thread in the parent simply does not exist on the other side
of the call. Any mutex those threads were holding stays locked forever, and any
bookkeeping that says "there are N worker threads" is now a lie. This is why
`pthread_atfork` handlers exist: a library gets a chance to fix its own state in
the child. OpenBLAS registers one, and its handler tears down the thread pool so
the child does not inherit references to threads that are gone.

The failure here is that the handler is doing the teardown by **joining** the
pool, and in the child there is nothing to join. `pthread_join` on a thread ID
that does not exist in this process is undefined behaviour; on this host the
undefined behaviour was a segmentation fault. (The same shape can also hang
instead — this project observed the crash, and should not claim the hang it did
not see.)

**Two — the library named in the backtrace is shipped by numpy, not by scipy.**
This is where an hour goes if you take the filename at face value. Modern numpy
wheels are built against the `scipy-openblas` distribution and vendor the result
*inside the numpy wheel*, at `numpy.libs/libscipy_openblas64_-<hash>.so`. So a
frame reading `libscipy_openblas...` is evidence that **numpy** was imported and
is no evidence at all about scipy. Confirmed by the path printed above: the
filename says scipy, the directory says numpy.

**Three — the pool is sized to the host, so the same code is a different hazard
on a different machine.** With none of the thread variables set, OpenBLAS sizes
its pool to the processor count. Measured here, four fresh interpreters:

| environment | pool |
|---|---|
| nothing set — *this is the Isaac launch path* | **16** |
| `OMP_NUM_THREADS=6` | **6** |
| `GOTO_NUM_THREADS=3` over `OMP_NUM_THREADS=6` | **3** |
| `OPENBLAS_NUM_THREADS=1` over `OMP_NUM_THREADS=6` | **1** |

`os.cpu_count()` here is 16 and the unset pool is 16. On a 48-core host the
unset pool is 48. So the number of threads the `atfork` handler must dispose of
in a child that has none of them goes from 15 to 47 — the same source code, a
**3.13×** larger hazard, and the crash appeared only at the larger one.

That makes this a portability defect **in our code**, not a quirk of the other
box. Nothing about the second machine is unusual; it simply has more cores, and
more cores is the direction hardware moves.

### The part of the diagnosis that was wrong, and stayed wrong for several probes

The first instrument reached for was `py-spy`, which samples a running process's
Python stacks. It reported the process sitting in `set_materialx_paths` inside
`omni.usd.config`, and that sent the investigation after a graphics-driver
problem — Vulkan, display, renderer configuration — for several probes. There
was no graphics problem. **py-spy had sampled an idle thread**, and an idle
thread's stack frame is a perfectly plausible-looking answer to a question
nobody asked it.

The real cause came from the breakpad backtrace, which reports the thread that
*died* rather than a thread that happened to be sampled.

This is the same trap `learnings/011` records in a different costume. There, a
plausible failure mode (the machine is falling over) was visible, correlated
with a bad number, and accounted for 0.9% of it. Here, a plausible stack frame
was visible, came from a real profiler, and had nothing to do with the crash. In
both cases the error was accepting the most legible signal instead of the one
that could be tied to the outcome by a mechanism.

Recorded separately as `anomalies.jsonl` row 58, so that it is findable by
somebody grepping for `py-spy` rather than only by somebody reading this lesson.
This project's anomaly file already carries a family of instrument caveats —
rows 36, 41, 43 and 55 — and "py-spy is not a trustworthy first instrument on a
multithreaded Kit process" belongs with them.

### The claim this lesson does NOT make

**Why our task and not Isaac Lab's own is established at the level of import
ordering, and no further.** `Isaac-Velocity-Rough-Anymal-C-v0` was reported to
run on the same host without the pin. The ordering that explains it is real and
checkable in the installed tree: `train_rsl_rl.run()` calls
`resolve_task_config(...)` at line 116 and only enters
`with launch_simulation(env_cfg, args_cli)` at line 118. Configuration is
resolved **before Kit starts**, so the entry-point module of our task is
imported before anything can fork. And ours reaches further than theirs:
`src/bestiary/isaac/anymal_desert_env_cfg.py:50` imports
`bestiary.terrain.isaac_hf`, which at line 61 does `from scipy import
interpolate` — a second vendored OpenBLAS, in a path Isaac Lab's own ANYmal
config never touches.

What is **not** established is that importing those modules is what *started the
pool threads*. OpenBLAS creates workers lazily, on a BLAS call above a size
threshold, and no measurement here shows which call did it — or whether numpy
alone, which both paths import, would have been enough on a 48-core host. The
cheapest discriminating test costs one run and no GPU-hours: launch **Isaac
Lab's own task through our entry point**, `train_desert.py --task
Isaac-Velocity-Rough-Anymal-C-v0`, without the pin. If it crashes, the variable
is our entry point and not the terrain bridge.

## The math

There is arithmetic here and it is deliberately small, because the whole point
is that the failure has no scale to it.

**Exit status.** A shell reports a signal death as

    status = 128 + signal

    139 = 128 + 11,    signal 11 = SIGSEGV

**Pool size.** With none of the three variables set, OpenBLAS sizes its pool to
the processor count `P`:

    pool = P                        (measured: P = 16 here, pool = 16)

With a variable set, the measured precedence is

    OPENBLAS_NUM_THREADS  >  GOTO_NUM_THREADS  >  OMP_NUM_THREADS

**Threads the `atfork` handler must dispose of in the child.** The child has
exactly one thread — the one that called `fork()`. Everything else in the pool
is a thread ID the child can name and cannot join:

    orphans = pool − 1

    this host, unset      : 16 − 1 = 15
    48-core host, unset   : 48 − 1 = 47        (47 / 15 = 3.13×)
    either host, pinned   :  1 − 1 =  0

**In plain English: pinning the pool to one thread does not make the fork safe.
It makes the teardown have nothing to tear down.** The unsafe operation is still
there, and any future Kit plugin that forks at a moment when some *other*
library has a populated thread pool will find the same hole. `OPENBLAS_NUM_THREADS=1`
removes this instance of it, not the class.

The `pool − 1` count also explains why the local machine looked fine and why
that is not reassuring. Fifteen orphaned thread IDs are undefined behaviour just
as forty-seven are. The local run did not crash; nothing measured says it was
safe rather than lucky.

## What to do next time

**A resource ceiling belongs in the launch path, and the launch path belongs in
one place.** Five documents and zero assignments is the defect. Every entry
point that can start a long process must set the ceiling itself, or read it from
a single module that does.

**The guard, stated precisely — and it does not exist yet.** The front matter
says `none` and that is a TODO being recorded honestly, not a judgement that no
check is possible. The check that would have caught this:

- Assert on the **process environment**, not on a document: that
  `OPENBLAS_NUM_THREADS` is `"1"` and `OMP_NUM_THREADS` is at or under the
  hard-rule-3 ceiling of 6, read from `os.environ`.
- Run it **at the top of every Isaac entry point** —
  `train_desert.py`, `check_hound.py`, `check_desert_terrain.py`,
  `view_desert.py` — before the first `import isaaclab`, because OpenBLAS reads
  these variables once at library initialisation and a value set afterwards
  changes nothing.
- Better still, have the entry point **set** the variables and then assert they
  took, by reading the pool size back out of OpenBLAS the way
  `research/scripts/016_openblas_fork_math.py` does. A guard that only checks
  the environment can be satisfied by a launch line; a guard that checks the
  pool cannot.

Note the sequencing constraint that makes the ordinary `guards --fast` gate
insufficient here: the process that must be checked is the one about to import
numpy, so the assertion has to be inside it. A separate green preflight in a
separate process proves nothing about the process that crashes.

**Pin it locally too.** The local pool is 16, the local hazard is 15 orphans,
and the only evidence that this is survivable is that it has not yet been
observed to fail. Set it on both machines and the question stops mattering.

**When a profiler and a crash handler disagree, the crash handler wins.**
`py-spy` samples; breakpad reports the thread that died. Before spending a probe
on a stack frame, ask whether the instrument had any way to know which thread
mattered.

**Read the directory, not the filename, when a shared object is in a
backtrace.** `numpy.libs/libscipy_openblas64_` is numpy's.

## How we would know this is wrong

**One observation, one host, no sweep.** The crash was seen on one machine and
the fix was confirmed by one rerun of one command. Under this repository's seed
rule that makes the *causal* claim a probe. The parts that are measured on this
machine — pool sizing, variable precedence, symbol table, import ordering — are
not probes; the mechanism connecting them to that particular SIGSEGV is.

This learning is wrong if any of these is observed:

- **`OMP_NUM_THREADS=6` alone prevents the crash on the 48-core host.** It caps
  the pool to 6 here, measured. If it also fixes the crash there, then the
  correct lesson is narrower than "the ceiling was never in the launch path" —
  it becomes "the ceiling was correct and simply absent," with no need to prefer
  `OPENBLAS_NUM_THREADS` at all, and the recommended guard changes accordingly.
- **The pin does not survive a longer run.** Thirty iterations is minutes, not
  hours; the wall clock of that particular run is **UNVERIFIED** and deliberately
  not converted here, because the only iteration time in the record belongs to a
  different robot at a different environment count and mixing the two is the
  error `learnings/017` is about. `fork()` from a Kit plugin at startup is one
  occasion; if a later fork — a video encoder, a USD subprocess, a checkpoint
  writer — reintroduces the same crash with the pin in place, then
  `OPENBLAS_NUM_THREADS=1` is a workaround for one call site and the real fix is
  elsewhere.
- **Isaac Lab's own task crashes through our entry point.** Then the terrain
  bridge's scipy import is not the differential, our entry point's pre-app
  imports are, and the "our code reaches further than theirs" paragraph above is
  wrong about which reach mattered.
- **The local 16-thread pool is shown to be safe by construction.** If glibc or
  OpenBLAS is shown to make `pthread_join` in the child benign below some pool
  size, then "the local run was lucky" is false and the local pin is
  unnecessary. Nothing measured here supports that, and it would need a primary
  source, not an absence of crashes.
- **The crash reproduces with `OPENBLAS_NUM_THREADS=1` at a larger environment
  count.** The failure was shown to be scale-independent by crashing at both 64
  and 4096 environments. The **fix was not**: the environment count of the
  passing run is **UNVERIFIED**. If it was confirmed only at a small size, then
  the scale-independence argument covers the failure and not the fix, and a
  4096-environment run with the pin is the missing observation.
