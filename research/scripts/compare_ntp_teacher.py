"""The numbers behind NTP_STAGE1_METHOD.md's result table.

    venv/bin/python research/scripts/compare_ntp_teacher.py [run_name]

Reads runs/<run>/eval_teacher/results.jsonl and eval_ntp/results.jsonl (both
written by bestiary.isaac.play_ntp on matched holdout scripts) and prints the
survival, distance, and tracking comparison. Refuses silently mismatched
arms: the seed lists must be identical, or the comparison means nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bestiary.paths import RUNS  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    run = RUNS / (sys.argv[1] if len(sys.argv) > 1 else "ntp_spot_s0")
    teacher = rows(run / "eval_teacher" / "results.jsonl")
    ntp = rows(run / "eval_ntp" / "results.jsonl")
    if [r["seed"] for r in teacher] != [r["seed"] for r in ntp]:
        raise SystemExit(
            f"seed mismatch: teacher {[r['seed'] for r in teacher]} vs ntp {[r['seed'] for r in ntp]}"
        )
    n = len(teacher)
    td = [r["distance_m"] for r in teacher]
    nd = [r["distance_m"] for r in ntp]
    diff = [abs(a - b) for a, b in zip(td, nd)]
    err = [r["mean_abs_vx_err"] for r in ntp if r["mean_abs_vx_err"] is not None]
    print(f"episodes: {n} matched holdout scripts")
    print(f"survived: teacher {sum(not r['fell'] for r in teacher)}/{n}, ntp {sum(not r['fell'] for r in ntp)}/{n}")
    print(f"mean distance: teacher {sum(td) / n:.3f} m, ntp {sum(nd) / n:.3f} m")
    print(f"per-episode |ddistance|: mean {sum(diff) / n:.3f} m, max {max(diff):.3f} m")
    print(f"ntp mean |v_x - cmd|: {sum(err) / len(err):.3f} m/s over {len(err)} episodes")


if __name__ == "__main__":
    main()
