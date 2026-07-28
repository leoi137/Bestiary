"""Guard: run artifacts stay inside their ceiling, and finished runs are pruned.

Enforces the retention policy in `Scriptorium/SYSTEM.md`.

Replay buffers are 94% of everything under `runs/` and exist only to resume an
interrupted run. A finished run is not going to be resumed — and per
`CORE_PLAN.md` a reward change makes its stored transitions worthless anyway —
so the buffer is pure cost from the moment the run ends.

A process that never deletes will eventually fill the disk, and it will do so
at 3am in the middle of the longest run it has ever attempted. This guard makes
that arrive as a failed check instead.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from bestiary import paths
from bestiary.guards import Finding

CEILING_GB = 60.0  # SYSTEM.md; this loop's share of a shared machine
FREE_FLOOR_GB = 100.0  # below this on the filesystem, stop starting runs
BUFFER_NAME = "ant_buffer.pkl"


def _dir_size_gb(path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def _live_run_names() -> set[str]:
    """Run names of training processes currently alive, from their argv.

    Matching on the bare module path is not enough: any shell command that
    merely *mentions* `bestiary.train.train` — including the check that calls
    this function — appears in pgrep output, and a `--run-name` appearing
    anywhere in such a line would wrongly mark a run as live. So the process
    must actually look like a python interpreter running the module, and
    shell wrappers are excluded explicitly.
    """
    out = subprocess.run(
        ["pgrep", "-af", r"python.*-m\s+bestiary\.train\.train"],
        capture_output=True,
        text=True,
    )
    names: set[str] = set()
    for line in out.stdout.splitlines():
        _, _, cmdline = line.partition(" ")
        parts = cmdline.split()
        if not parts:
            continue
        # `bash -c '... python -m bestiary.train.train ...'` is a shell holding
        # the text, not a trainer. The real process has python as argv[0].
        if "python" not in parts[0]:
            continue
        if "--run-name" in parts:
            names.add(parts[parts.index("--run-name") + 1])
    return names


def _finished_run_dirs() -> list[Path]:
    """Run directories not currently being trained into.

    This is the set the buffer assertion quantifies over: a live run's buffer
    is in use, so it is not a retention violation and was never examined.
    """
    if not paths.RUNS.exists():
        return []
    live = _live_run_names()
    return [
        p for p in sorted(paths.RUNS.iterdir())
        if p.is_dir() and p.name not in live
    ]


def _buffers(run_dirs: list[Path]) -> list[tuple[str, float]]:
    """(run name, buffer size GB) for those dirs still holding a buffer."""
    out = []
    for run_dir in run_dirs:
        buf = run_dir / BUFFER_NAME
        if buf.exists():
            out.append((run_dir.name, buf.stat().st_size / 1024**3))
    return out


def prunable() -> list[tuple[str, float]]:
    """(run name, buffer size GB) for finished runs still holding a buffer."""
    return _buffers(_finished_run_dirs())


def run() -> list[Finding]:
    findings: list[Finding] = []

    if not paths.RUNS.exists():
        return [Finding("runs/ exists", True, "no runs yet")]

    # The next two are scalar thresholds against a single total, not
    # quantifiers over a set, so they leave n unset rather than claim one.
    used = _dir_size_gb(paths.RUNS)
    findings.append(
        Finding(
            f"runs/ is under the {CEILING_GB:.0f} GB ceiling",
            used <= CEILING_GB,
            f"{used:.1f} GB used",
        )
    )

    free = shutil.disk_usage(paths.REPO_ROOT).free / 1024**3
    findings.append(
        Finding(
            f"at least {FREE_FLOOR_GB:.0f} GB free on the filesystem",
            free >= FREE_FLOOR_GB,
            f"{free:.0f} GB free",
        )
    )

    # n is the number of finished runs actually inspected for a buffer, not
    # the number found holding one.
    finished = _finished_run_dirs()
    stale = _buffers(finished)
    total = sum(size for _, size in stale)
    findings.append(
        Finding(
            "no finished run is still holding a replay buffer",
            not stale,
            (
                f"{len(stale)} run(s), {total:.1f} GB recoverable: "
                f"{', '.join(n for n, _ in stale)}\n"
                "         python -m bestiary.record.retention --apply"
            )
            if stale
            else "nothing to prune",
            n=len(finished),
        )
    )
    return findings
