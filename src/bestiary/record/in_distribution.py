"""Split a track_eval drive grid into its in-distribution and extrapolation parts.

`decisions/0002` requires that every cycle quoting a `drive_grid_mean` quote it
twice — over all six drive cells, and over the five the env actually samples —
because one cell asks for a command the policy was never trained on and a
do-nothing control collects most of the bar there.

`track_eval` is deliberately NOT changed to emit this itself (0002 again: the
instrument must not move between a pre-registered prediction and its reading),
so this module recomputes the split from a measurement JSON already on disk.
It reads artifacts and never runs a policy, so it is exact and repeatable and
costs no GPU.

Which cell is off-distribution is DERIVED from the env's own sampling floors
rather than hardcoded: a cell is in-distribution when its commanded speeds lie
inside the ranges `hound_track` samples during training. If someone lowers
`VX_MIN_BACKWARD` to 0.3, this file starts reporting five-cell == six-cell on
its own and 0002's first reversal trigger has fired visibly, instead of a
frozen literal quietly asserting something that stopped being true.

Usage:

    python -m bestiary.record.in_distribution research/measurements/foo.json
    python -m bestiary.record.in_distribution foo.json --json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bestiary.envs import hound_track as _ht
from bestiary.record.track_eval import EVAL_GRID, STOP_CELL


def cell_is_in_distribution(cell: tuple[float, float, float]) -> bool:
    """True when the env's command sampler can actually produce this command.

    Mirrors `_resample_command` in `hound_track.py`, which has three branches:
    STOP (the zero command), TURN in place (vx = 0, |wz| in [W_TURN_MIN,
    W_TURN_MAX]) and DRIVE (|vx| in [VX_MIN, VX_MAX] forward or
    [VX_MIN_BACKWARD, VX_MAX] backward, |wz| <= W_DRIVE_MAX).
    """
    vx, _vy, wz = cell
    tol = 1e-9
    if (vx, _vy, wz) == STOP_CELL:
        return True
    if vx == 0.0:
        # TURN in place. Note the floor: a turn slower than W_TURN_MIN is never
        # commanded, because the unsteered drift would match it by accident.
        return _ht.W_TURN_MIN - tol <= abs(wz) <= _ht.W_TURN_MAX + tol
    if abs(wz) > _ht.W_DRIVE_MAX + tol:
        return False
    lo = _ht.VX_MIN if vx > 0 else _ht.VX_MIN_BACKWARD
    return lo - tol <= abs(vx) <= _ht.VX_MAX + tol


def drive_cells() -> tuple[tuple[float, float, float], ...]:
    """The six non-stop cells of the eval grid, in grid order."""
    return tuple(c for c in EVAL_GRID if c != STOP_CELL)


def split_arm(arm: dict) -> dict:
    """Recompute the headline over both cell groupings for one arm.

    `arm` is a `zero_action` or `trained` block from a track_eval JSON. Every
    number here is a mean of per-cell means, exactly as `drive_grid_mean` is
    defined, so the six-cell value reproduces the arm's own `drive_grid_mean`
    and disagreement is a bug rather than a rounding difference.
    """
    cells = arm["cells"]
    out: dict = {"in_dist_cells": [], "extrapolation_cells": []}

    def mean_of(keys: list[str], field: str) -> float | None:
        vals = [cells[k][field] for k in keys if field in cells[k]]
        return sum(vals) / len(vals) if vals else None

    six, five = [], []
    for cell in drive_cells():
        key = str(tuple(float(x) for x in cell))
        if key not in cells:
            raise KeyError(
                f"measurement is missing drive cell {key}; it has "
                f"{sorted(cells)} — this JSON was not produced by this grid"
            )
        six.append(key)
        if cell_is_in_distribution(cell):
            five.append(key)
            out["in_dist_cells"].append(key)
        else:
            out["extrapolation_cells"].append(key)

    if not out["extrapolation_cells"]:
        out["note"] = (
            "every drive cell is in-distribution — decision 0002's first "
            "reversal trigger has fired; the split reporting is now noise"
        )

    for field, name in (("mean", "drive_grid_mean"),
                        ("mean_track", "drive_grid_track")):
        out[f"{name}_six_cell"] = mean_of(six, field)
        out[f"{name}_in_dist"] = mean_of(five, field)
        out[f"{name}_extrapolation"] = mean_of(out["extrapolation_cells"], field)

    reported = arm.get("drive_grid_mean")
    recomputed = out["drive_grid_mean_six_cell"]
    if reported is not None and recomputed is not None:
        drift = abs(reported - recomputed)
        if drift > 1e-6:
            raise ValueError(
                f"recomputed six-cell drive_grid_mean {recomputed!r} does not "
                f"match the arm's reported {reported!r} (differ by {drift:.3e}); "
                "the aggregate is not a plain mean of per-cell means, so the "
                "five-cell number computed the same way would also be wrong"
            )
    out["six_cell_matches_reported"] = reported is not None
    return out


def split_measurement(doc: dict) -> dict:
    """Split every arm present in a track_eval measurement document."""
    arms = {k: v for k, v in doc.items()
            if isinstance(v, dict) and "cells" in v}
    if not arms:
        raise ValueError(
            f"no arm with a 'cells' block in this JSON; top-level keys are "
            f"{sorted(doc)} — was it written with --json?"
        )
    return {
        "env": doc.get("env"),
        "episodes_per_cell": doc.get("episodes_per_cell"),
        "seed0": doc.get("seed0"),
        "run": doc.get("run"),
        "arms": {name: split_arm(arm) for name, arm in arms.items()},
    }


def _fmt(x: float | None) -> str:
    return f"{'n/a':>10}" if x is None else f"{x:10.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("measurement", type=Path,
                    help="a track_eval --json measurement file")
    ap.add_argument("--json", action="store_true",
                    help="print the full structure instead of a table")
    args = ap.parse_args()

    if not args.measurement.exists():
        raise SystemExit(f"no such measurement: {args.measurement}")
    if args.measurement.stat().st_size == 0:
        raise SystemExit(
            f"{args.measurement} is 0 bytes — cycle 010 committed an empty "
            "measurement and quoted a number out of it. Re-measure."
        )
    doc = json.loads(args.measurement.read_text())
    result = split_measurement(doc)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"{result['env']}   {result['episodes_per_cell']} eps/cell   "
          f"decisions/0002 split")
    ex = next(iter(result["arms"].values()))["extrapolation_cells"]
    print(f"extrapolation cells (excluded from the in-dist headline): "
          f"{ex or 'none'}")
    print(f"{'arm':<14}{'six-cell':>10}{'in-dist':>10}{'extrap':>10}"
          f"{'six-trk':>10}{'in-trk':>10}")
    for name, arm in result["arms"].items():
        print(f"{name:<14}"
              f"{_fmt(arm['drive_grid_mean_six_cell'])}"
              f"{_fmt(arm['drive_grid_mean_in_dist'])}"
              f"{_fmt(arm['drive_grid_mean_extrapolation'])}"
              f"{_fmt(arm['drive_grid_track_six_cell'])}"
              f"{_fmt(arm['drive_grid_track_in_dist'])}")
    print("\nBoth headline columns are required by decisions/0002. Neither "
          "may be quoted alone.")


if __name__ == "__main__":
    main()
