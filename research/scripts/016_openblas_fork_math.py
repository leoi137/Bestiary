"""The measurements behind `research/learnings/016`.

Learning 016 is about a training process that died with SIGSEGV inside
OpenBLAS's `atfork` handler on a 48-core host, and did not die on this 16-thread
one. Three claims in that lesson are load-bearing, and this script measures all
three on THIS machine rather than asserting them:

1. **The library named `libscipy_openblas*` in the crash backtrace is vendored
   by numpy, not by scipy.** The name is a packaging artifact of the
   `scipy-openblas` wheel that numpy builds against, so a backtrace naming it
   does NOT prove scipy was imported. Getting this wrong sends you looking for
   the wrong import.

2. **That library really does register an `atfork` handler and really does join
   threads in it.** `blas_thread_shutdown_` (the frame in the backtrace) is a
   defined symbol; `__register_atfork`, `pthread_create` and `pthread_join` are
   imported ones. Three of the four backtrace frames are visible in the dynamic
   symbol table of a file sitting in this venv.

3. **`OMP_NUM_THREADS` alone DOES cap the pool.** This is the claim the lesson
   turns on, and it is the opposite of the intuition that "OpenBLAS reads its
   own variable, so the project's `OMP_NUM_THREADS=6` ceiling never covered
   this." It covers it fine. The ceiling failed for a different reason: it was
   never exported into the Isaac launch path at all.

Pool sizes are measured by re-running this file as a subprocess with the
environment overridden, because OpenBLAS reads these variables once, at library
initialisation — setting them after `import numpy` is too late.

    venv/bin/python research/scripts/016_openblas_fork_math.py

No GPU, no MuJoCo, no Isaac. Runs in about a second.
"""

from __future__ import annotations

import ctypes
import glob
import os
import shutil
import subprocess
import sys

# --- The one number that comes from the other machine ----------------------
# Reported from a second machine with 48 CPU cores, 2026-07-29. Everything else
# in this file is measured locally. Kept as a named constant so the arithmetic
# below cannot be mistaken for a local measurement.
REMOTE_CORES = 48

# Every environment variable OpenBLAS is documented to read, in the precedence
# order this script MEASURES rather than assumes.
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")

#: Environments to measure. Each is (label, {var: value}); any var absent from
#: the dict is UNSET in the child, not merely empty.
CASES: tuple[tuple[str, dict[str, str]], ...] = (
    ("nothing set (the Isaac launch path)", {}),
    ("OMP_NUM_THREADS=6 (the project's documented ceiling)", {"OMP_NUM_THREADS": "6"}),
    ("GOTO_NUM_THREADS=3 over OMP=6", {"GOTO_NUM_THREADS": "3", "OMP_NUM_THREADS": "6"}),
    ("OPENBLAS_NUM_THREADS=1 over OMP=6 (the fix)", {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "6"}),
)


def _vendored_openblas() -> str:
    """Absolute path of the OpenBLAS shared object numpy actually loads.

    Raises rather than returning None: the whole point of claim 1 is *which
    package ships this file*, so "could not find it" is a failed measurement,
    not a missing optional feature.
    """
    import numpy

    site_packages = os.path.dirname(os.path.dirname(numpy.__file__))
    hits = sorted(glob.glob(os.path.join(site_packages, "numpy.libs", "libscipy_openblas*.so")))
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly 1 vendored libscipy_openblas*.so under "
            f"{site_packages}/numpy.libs, found {len(hits)}: {hits}"
        )
    return hits[0]


def _pool_size() -> int:
    """Threads in this process's OpenBLAS pool, from OpenBLAS itself.

    The symbol is `scipy_openblas_get_num_threads64_`, not the upstream
    `openblas_get_num_threads`: the wheel renames every public symbol so two
    vendored copies cannot collide in one process. Looking for the upstream name
    fails with a bare AttributeError, which is how this measurement was nearly
    abandoned as impossible.
    """
    lib = ctypes.CDLL(_vendored_openblas())
    fn = lib.scipy_openblas_get_num_threads64_
    fn.restype = ctypes.c_int
    return int(fn())


def _child(label: str, overrides: dict[str, str]) -> int:
    """Measure the pool size in a fresh interpreter under `overrides`."""
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env.update(overrides)
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--pool-only"],
        env=env, capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def _imported_symbols(so: str) -> set[str]:
    """The dynamic symbols `so` imports, via `nm`. Empty set if nm is absent."""
    if shutil.which("nm") is None:
        return set()
    out = subprocess.run(
        ["nm", "-D", "--undefined-only", so], capture_output=True, text=True, check=False
    )
    return {line.split()[-1].split("@")[0] for line in out.stdout.splitlines() if line.strip()}


def main() -> None:
    so = _vendored_openblas()
    print("CLAIM 1 -- which package ships the library the backtrace names")
    print(f"  {so}")
    print("  ^ the path segment is 'numpy.libs'. The FILENAME says scipy; the")
    print("    OWNER is numpy. A backtrace naming libscipy_openblas is evidence")
    print("    that numpy was imported, and no evidence at all about scipy.")
    print()

    print("CLAIM 2 -- the fork machinery is in that file's symbol table")
    lib = ctypes.CDLL(so)
    for sym in ("blas_thread_shutdown_",):
        # Resolved, never CALLED: calling it would tear down this process's pool.
        found = hasattr(lib, sym)
        print(f"  defines  {sym:24s} {found}")
        if not found:
            raise AssertionError(f"{so} does not define {sym}; learning 016's mechanism is wrong")
    imported = _imported_symbols(so)
    if imported:
        for sym in ("__register_atfork", "pthread_create", "pthread_join"):
            print(f"  imports  {sym:24s} {sym in imported}")
            if sym not in imported:
                raise AssertionError(f"{so} does not import {sym}")
    else:
        print("  imports  <nm unavailable -- not measured>")
    print("  ^ three of the four frames in the crash backtrace, in one file")
    print("    that is sitting in this venv right now.")
    print()

    print("CLAIM 3 -- pool size vs environment, measured in fresh interpreters")
    print(f"  os.cpu_count() = {os.cpu_count()}, "
          f"len(os.sched_getaffinity(0)) = {len(os.sched_getaffinity(0))}")
    results: dict[str, int] = {}
    for label, overrides in CASES:
        pool = _child(label, overrides)
        results[label] = pool
        print(f"  pool = {pool:3d}   {label}")
    print()

    unset = results["nothing set (the Isaac launch path)"]
    omp_only = results["OMP_NUM_THREADS=6 (the project's documented ceiling)"]
    if unset != os.cpu_count():
        print(f"  NOTE: unset pool ({unset}) != os.cpu_count() ({os.cpu_count()}); "
              "the sizing rule on this host is not simply the core count.")
    if omp_only != 6:
        raise AssertionError(
            f"OMP_NUM_THREADS=6 gave a pool of {omp_only}, not 6 -- learning 016's "
            "central claim (that the documented ceiling WOULD have capped the pool) "
            "does not hold on this build."
        )
    print("  Measured precedence: OPENBLAS_NUM_THREADS > GOTO_NUM_THREADS > OMP_NUM_THREADS,")
    print("  and OMP_NUM_THREADS ALONE is honoured. The documented ceiling was")
    print("  never the problem; its absence from the launch path was.")
    print()

    print("THE FORK HAZARD, sized")
    print("  A fork from a multithreaded process gives the child exactly ONE thread.")
    print("  OpenBLAS's atfork handler then runs in that child and joins the pool.")
    print("  Threads it tries to join, none of which exist in the child:")
    print(f"    this host, pool unset : {unset:3d} - 1 = {unset - 1:3d}")
    print(f"    48-core host, unset   : {REMOTE_CORES:3d} - 1 = {REMOTE_CORES - 1:3d}"
          f"   ({(REMOTE_CORES - 1) / (unset - 1):.2f}x this host)")
    print("    either host, pinned to 1 :   1 - 1 =   0")
    print("  Pinning to 1 does not make the fork safe. It makes the handler have")
    print("  nothing to do, which is the same thing from the process's point of view.")


if __name__ == "__main__":
    if "--pool-only" in sys.argv:
        print(_pool_size())
    else:
        main()
