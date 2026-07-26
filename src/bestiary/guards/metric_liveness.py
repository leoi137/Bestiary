"""Guard: no logged metric is silently constant.

Enforces `research/episodes/003` — `eval/mean_idle_legs` read exactly 0.00 for
every step of two complete runs, because it is populated from an `info` key
that only a shaping wrapper sets and neither run used one. It was not measuring
a stuck robot. It was not measuring anything.

A constant metric is worse than a missing one: a missing metric is obviously
absent, while a constant one renders on the dashboard, appears in the write-up,
and answers a question it never had access to.

Reads TensorBoard event files directly. Marked `slow` because the accumulator
walks the whole event file.
"""
from __future__ import annotations

from pathlib import Path

from bestiary import paths
from bestiary.guards import Finding

# Below this many points a flat line is not yet evidence of anything.
MIN_POINTS = 5

# Metrics that are legitimately constant. Every entry needs a reason, because
# an unexplained exclusion is how a genuinely dead metric gets silenced.
EXPECTED_CONSTANT = frozenset(
    {
        # SAC runs at a fixed learning rate here — no schedule is configured,
        # so a flat line is the correct reading rather than a broken one.
        "train/learning_rate",
    }
)


def _scalars(event_dir: Path) -> dict[str, list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [e.value for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def _event_dirs(run_dir: Path) -> list[Path]:
    """Directories holding tfevents files, one per SB3 logger sub-run."""
    return sorted({p.parent for p in run_dir.rglob("events.out.tfevents.*")})


def run() -> list[Finding]:
    if not paths.RUNS.exists():
        return [Finding("runs/ exists", True, "no runs yet — nothing to check")]

    findings: list[Finding] = []
    for run_dir in sorted(p for p in paths.RUNS.iterdir() if p.is_dir()):
        for event_dir in _event_dirs(run_dir):
            try:
                scalars = _scalars(event_dir)
            except Exception as exc:
                findings.append(
                    Finding(f"{run_dir.name}: events readable", False,
                            f"{type(exc).__name__}: {exc}")
                )
                continue

            dead = sorted(
                tag
                for tag, values in scalars.items()
                if tag not in EXPECTED_CONSTANT
                and len(values) >= MIN_POINTS
                and min(values) == max(values)
            )
            findings.append(
                Finding(
                    f"{run_dir.name}: every logged metric varies",
                    not dead,
                    f"constant: {dead}" if dead else f"{len(scalars)} metrics checked",
                )
            )
    if not findings:
        findings.append(Finding("event files found", True, "none on disk"))
    return findings
