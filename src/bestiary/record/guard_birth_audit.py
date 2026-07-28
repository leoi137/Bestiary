"""Was each guard *vacuous on the day it shipped*?

Cycle 012 proposed the **birth-vacuity model**: an assertion is vacuous as a
function of the guard's AGE, not its author's care. A guard that quantifies
over a set the project has not started producing yet is born examining nothing,
and stops being vacuous when the first member of that set lands — not when
anyone improves the guard.

The model is only worth anything if it makes a number. So this measures one,
per guard:

    gap_hours = (first moment the input set was non-empty) - (guard's first commit)

    gap > 0   born VACUOUS, and vacuous for that many hours
    gap <= 0  born covering something — the data was already there

Every date comes from evidence on this machine, named per guard in `EVIDENCE`
so a reader can re-derive it by hand:

  * `git`      — first commit of a path (`--follow --diff-filter=A`), used for
                 tracked files and for source-quantified sets.
  * `mtime`    — `runs/` is gitignored, so file modification time is the only
                 record a run ever existed. Weaker than git and marked so.
  * `ledger`   — the commit that first added a `ledger.jsonl` line carrying the
                 field the assertion needs.
  * `code`     — the set is constructed by the guard itself (synthetic
                 episodes, a registry walk, a terrain it synthesizes). Such a
                 set is non-empty the instant the guard exists, so its
                 first-non-empty IS its first commit and its gap is 0 by
                 construction. Flagged, because this is the model's blind spot:
                 for these guards "born vacuous" is not even definable, and
                 counting them as evidence FOR the model would be circular.
  * `none`     — no set-quantified assertion at all (every Finding carries
                 `n=None`). Cannot be vacuous. Excluded from the statistics.

Run it:

    venv/bin/python -m bestiary.record.guard_birth_audit
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[3]
GUARDS = REPO / "src" / "bestiary" / "guards"
RUNS = REPO / "runs"
LEDGER = REPO / "research" / "ledger.jsonl"


def _git_first_commit(path: Path) -> datetime | None:
    """Author date of the commit that first ADDED `path`, following renames."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "--follow", "--diff-filter=A",
         "--format=%aI", "--", str(path.relative_to(REPO))],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return datetime.fromisoformat(out[-1]) if out else None


def _git_first_commit_touching(prefix: str) -> datetime | None:
    """Author date of the earliest commit that added anything under `prefix`."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "log", "--diff-filter=A", "--format=%aI",
         "--", prefix],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return datetime.fromisoformat(out[-1]) if out else None


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def _earliest_run_artifact() -> datetime:
    """Oldest file anywhere under runs/. runs/ is gitignored; mtime is all there is."""
    files = [p for p in RUNS.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no files under {RUNS}; cannot date runs/")
    return _mtime(min(files, key=lambda p: p.stat().st_mtime))


def _earliest_config_with(key: str) -> datetime:
    """Oldest runs/*/config.json that carries `key` at top level.

    This is the set the *-spec guards actually quantify over: not runs, but
    runs that DECLARE. A run without the key is named in the detail and
    explicitly excluded from n, so it is not coverage.
    """
    dated = []
    for cfg in RUNS.glob("*/config.json"):
        try:
            if key in json.loads(cfg.read_text()):
                dated.append(_mtime(cfg))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{cfg} unreadable while dating key {key!r}: {exc}") from exc
    if not dated:
        raise LookupError(f"no runs/*/config.json carries {key!r}; set is still empty")
    return min(dated)


def _first_ledger_row_with(field: str) -> datetime:
    """Commit date of the earliest ledger.jsonl revision carrying `field`.

    Uses git rather than the row's own `date` string: the row's date says when
    the run happened, and what the guard needs is when the row became visible
    to it.
    """
    revs = subprocess.run(
        ["git", "-C", str(REPO), "log", "--format=%H %aI", "--reverse", "--",
         "research/ledger.jsonl"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    for line in revs:
        sha, iso = line.split()
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{sha}:research/ledger.jsonl"],
            capture_output=True, text=True, check=True,
        ).stdout
        for raw in blob.splitlines():
            if raw.strip() and field in json.loads(raw):
                return datetime.fromisoformat(iso)
    raise LookupError(f"no ledger row has ever carried {field!r}")


@dataclass(frozen=True, slots=True)
class Row:
    guard: str
    module: str
    quantifies_over: str
    kind: str          # git | mtime | ledger | code | none | open
    confidence: str
    resolve: object    # () -> datetime | None ; None = set is STILL empty


def _table() -> list[Row]:
    return [
        Row("privacy", "privacy.py", "tracked source files (git ls-files)",
            "git", "high", lambda: _git_first_commit_touching("README.md")),
        Row("disk", "disk.py", "finished run directories on disk",
            "mtime", "medium", _earliest_run_artifact),
        Row("memory", "memory.py", "nothing — scalar thresholds, every n=None",
            "none", "high", lambda: None),
        Row("spawn-pad", "spawn_pad.py", "cells of a terrain the guard synthesizes",
            "code", "high",
            lambda: _git_first_commit(REPO / "src/bestiary/terrain/generate.py")),
        Row("ledger-schema", "ledger_schema.py", "research/ledger.jsonl rows",
            "git", "high", lambda: _git_first_commit(LEDGER)),
        Row("checkpoint-width", "checkpoint_width.py",
            "runs/*/config.json carrying obs_spec (the 'pinned' set)",
            "mtime", "medium", lambda: _earliest_config_with("obs_spec")),
        Row("reward-spec", "reward_spec.py",
            "runs/*/config.json carrying reward_spec",
            "mtime", "medium", lambda: _earliest_config_with("reward_spec")),
        Row("measurement-provenance", "measurement_provenance.py",
            "measurements naming a checkpoint and NOT grandfathered (pre-07-28)",
            "open", "high", lambda: None),
        Row("terrain-spec", "terrain_spec.py",
            "runs/*/config.json carrying terrain_spec",
            "mtime", "medium", lambda: _earliest_config_with("terrain_spec")),
        Row("tracking-frame", "tracking_frame.py",
            "constructed states the guard builds itself (n=1 each)",
            "code", "high", lambda: _git_first_commit(GUARDS / "tracking_frame.py")),
        Row("eval-sampling", "eval_sampling.py",
            "ledger rows carrying best_eval_episodes",
            "ledger", "high", lambda: _first_ledger_row_with("best_eval_episodes")),
        Row("parked-detector", "parked_detector.py",
            "grid cells in measurements/tracking_baseline_zero_action.json",
            "git", "high", lambda: _git_first_commit(
                REPO / "research/measurements/tracking_baseline_zero_action.json")),
        Row("nulls", "nulls.py", "research/nulls.jsonl rows + run configs",
            "git", "high", lambda: _git_first_commit(REPO / "research/nulls.jsonl")),
        Row("track-length-bias", "track_length_bias.py",
            "nothing — synthetic episodes, every n=None",
            "none", "high", lambda: None),
        Row("metric-liveness", "metric_liveness.py",
            "tensorboard scalars under runs/*/ant_tb",
            "mtime", "medium", _earliest_run_artifact),
        Row("standing-control", "standing.py",
            "EPISODES zero-action episodes the guard rolls out itself",
            "code", "high", lambda: _git_first_commit(GUARDS / "standing.py")),
    ]


def main() -> int:
    now = datetime.now().astimezone()
    results = []
    for row in _table():
        born = _git_first_commit(GUARDS / row.module)
        if born is None:
            raise RuntimeError(f"{row.module} has no add-commit; guard list is stale")
        try:
            first = row.resolve()
        except LookupError:
            first = None
        if row.kind == "none":
            gap = None
        elif first is None:
            gap = (now - born).total_seconds() / 3600.0   # still empty: lower bound
        else:
            gap = (first - born).total_seconds() / 3600.0
        results.append((row, born, first, gap))

    w = max(len(r.guard) for r, *_ in results)
    print(f"{'guard':<{w}}  {'first commit':<25} {'first non-empty':<25} "
          f"{'gap h':>9}  {'evidence':<8} conf")
    print("-" * (w + 82))
    for row, born, first, gap in results:
        f = first.isoformat(timespec="seconds") if first else (
            "STILL EMPTY" if row.kind != "none" else "n/a (not set-quantified)")
        g = "n/a" if gap is None else (f"{gap:+.2f}" + ("+" if first is None else ""))
        print(f"{row.guard:<{w}}  {born.isoformat(timespec='seconds'):<25} {f:<25} "
              f"{g:>9}  {row.kind:<8} {row.confidence}")

    scored = [(r, g) for r, _, _, g in results if g is not None]
    structural = [g for r, g in scored if r.kind == "code"]
    empirical = [(r, g) for r, g in scored if r.kind != "code"]
    pos = sorted(g for _, g in empirical if g > 0)
    nonpos = [g for _, g in empirical if g <= 0]

    print()
    print(f"16 guards: {len(scored)} set-quantified, "
          f"{len(results) - len(scored)} not set-quantified (excluded).")
    print(f"  {len(structural)} quantify over sets they BUILD THEMSELVES "
          f"(gap undefined by construction; excluded from stats).")
    print(f"  {len(empirical)} quantify over accumulating project data — the "
          f"only ones the model can be tested on.")
    print(f"    born VACUOUS  (gap > 0): {len(pos)}   {[f'{g:+.1f}' for g in pos]}")
    print(f"    born covering (gap <= 0): {len(nonpos)}  "
          f"{[f'{g:+.1f}' for g in sorted(nonpos)]}")
    if pos:
        print(f"    positive gaps: median {median(pos):.2f} h, "
              f"range {min(pos):.2f} .. {max(pos):.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
