"""Apply the run-retention policy from `Scriptorium/SYSTEM.md`.

    python -m bestiary.record.retention            # dry run — always the default
    python -m bestiary.record.retention --apply    # actually delete

Replay buffers are resume-only and are 94% of everything under `runs/`. Once a
run has ended, its buffer guards nothing: the two `.zip` checkpoints,
`ant_tb/`, and `config.json` are the run, and together they are about 9 MB.

Deleting is irreversible, so the safeguards are not optional:

- **Dry run by default.** `--apply` is required to remove anything.
- **Never a live run.** Any run whose name appears in a running
  `bestiary.train.train` argv is skipped, and so is any whose buffer was
  written in the last `--min-age-minutes` (default 30) — a run can be alive
  between process checks.
- **Never a checkpoint.** Only `ant_buffer.pkl` is ever removed. There is no
  flag to delete anything else.
"""
from __future__ import annotations

import argparse
import sys
import time

from bestiary import paths
from bestiary.guards.disk import BUFFER_NAME, _live_run_names

DEFAULT_MIN_AGE_MINUTES = 30


def candidates(min_age_minutes: int) -> tuple[list[tuple[str, float]], list[tuple[str, str]]]:
    """(deletable, skipped-with-reason)."""
    live = _live_run_names()
    now = time.time()
    deletable: list[tuple[str, float]] = []
    skipped: list[tuple[str, str]] = []

    if not paths.RUNS.exists():
        return deletable, skipped

    for run_dir in sorted(p for p in paths.RUNS.iterdir() if p.is_dir()):
        buf = run_dir / BUFFER_NAME
        if not buf.exists():
            continue
        if run_dir.name in live:
            skipped.append((run_dir.name, "training process is alive"))
            continue
        age_minutes = (now - buf.stat().st_mtime) / 60
        if age_minutes < min_age_minutes:
            skipped.append((run_dir.name, f"buffer written {age_minutes:.0f} min ago"))
            continue
        deletable.append((run_dir.name, buf.stat().st_size / 1024**3))
    return deletable, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="actually delete (default is a dry run)")
    parser.add_argument("--min-age-minutes", type=int, default=DEFAULT_MIN_AGE_MINUTES)
    args = parser.parse_args()

    deletable, skipped = candidates(args.min_age_minutes)

    for name, reason in skipped:
        print(f"  SKIP    {name:<28} {reason}")

    if not deletable:
        print("Nothing to prune.")
        return 0

    total = sum(size for _, size in deletable)
    verb = "DELETE" if args.apply else "would delete"
    for name, size in deletable:
        print(f"  {verb:<7} {name:<28} {size:.2f} GB")
    print(f"\n{len(deletable)} buffer(s), {total:.2f} GB")

    if not args.apply:
        print("Dry run. Re-run with --apply to delete.")
        return 0

    freed = 0.0
    for name, size in deletable:
        (paths.RUNS / name / BUFFER_NAME).unlink()
        freed += size
    print(f"Freed {freed:.2f} GB. Checkpoints, tensorboard and config are untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
