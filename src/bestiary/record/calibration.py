"""Score this project's forecasting against what actually happened.

    python -m bestiary.record.calibration

Every prediction in this repo is written before the result is known and states
an explicit probability, which means the whole history is scoreable. That is
the point: a record that only stores conclusions cannot tell you whether the
person writing them is any good. A record that stores *probabilities* can.

The metric is the Brier score, mean squared error between the stated
probability and the outcome:

    B = (1/N) * sum (p_i - o_i)^2      o_i in {0, 1}

Lower is better. 0.0 is perfect, 0.25 is what you get by saying 50% to
everything, and anything above 0.25 means the forecasts are worse than an
honest shrug. The reliability table underneath it is the more useful half: it
shows whether claims made at 70% actually come true about 70% of the time, and
in which direction the bias runs.

Reads `research/calibration.jsonl`, one row per resolved prediction:

    {"cycle": "002", "date": "2026-07-25", "claim": "...", "p": 0.55,
     "outcome": true, "resolved_in": "episodes/003-...", "run": "..."}

Unresolved predictions are appended with `"outcome": null` and skipped here
until the result lands.
"""
from __future__ import annotations

import json
import sys

from bestiary import paths

CALIBRATION = paths.RESEARCH / "calibration.jsonl"

# Coarse enough that a handful of rows still populate them.
BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))

# Below this, the Brier score is noise and reporting it invites over-reading.
MIN_ROWS_FOR_SCORE = 10


def load() -> list[dict]:
    if not CALIBRATION.exists():
        return []
    return [json.loads(ln) for ln in CALIBRATION.read_text().splitlines() if ln.strip()]


def resolved(rows: list[dict]) -> list[dict]:
    return [r for r in rows if isinstance(r.get("outcome"), bool)]


def brier(rows: list[dict]) -> float:
    return sum((float(r["p"]) - float(r["outcome"])) ** 2 for r in rows) / len(rows)


def reliability(rows: list[dict]) -> list[tuple[str, int, float, float]]:
    """(bucket label, n, mean stated probability, observed frequency)."""
    table = []
    for lo, hi in BUCKETS:
        inside = [r for r in rows if lo <= float(r["p"]) < hi or (hi == 1.0 and float(r["p"]) == 1.0)]
        if not inside:
            continue
        stated = sum(float(r["p"]) for r in inside) / len(inside)
        observed = sum(float(r["outcome"]) for r in inside) / len(inside)
        table.append((f"{lo:.0%}-{hi:.0%}", len(inside), stated, observed))
    return table


def main() -> int:
    rows = load()
    done = resolved(rows)
    pending = len(rows) - len(done)

    if not done:
        print("No resolved predictions yet.")
        print(f"({pending} awaiting an outcome in {CALIBRATION})")
        return 0

    print(f"Resolved predictions: {len(done)}   (pending: {pending})")
    print(f"Hit rate: {sum(float(r['outcome']) for r in done) / len(done):.0%}")

    score = brier(done)
    print(f"Brier score: {score:.4f}   (0 perfect, 0.25 = always saying 50%)")
    if len(done) < MIN_ROWS_FOR_SCORE:
        print(f"  ^ {len(done)} rows is too few to read as a skill estimate. "
              f"Treat it as a placeholder until {MIN_ROWS_FOR_SCORE}.")

    print("\nReliability — what was claimed vs what happened")
    print(f"  {'band':<10} {'n':>3}  {'stated':>7}  {'actual':>7}   bias")
    for label, n, stated, observed in reliability(done):
        gap = observed - stated
        direction = "over-confident" if gap < -0.05 else "under-confident" if gap > 0.05 else "calibrated"
        print(f"  {label:<10} {n:>3}  {stated:>6.0%}  {observed:>6.0%}   {direction}")

    print("\nRead this before writing the next prediction. It is the only")
    print("mechanism here that can tell you your confidence is drifting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
