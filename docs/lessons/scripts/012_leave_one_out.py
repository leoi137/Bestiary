"""Arithmetic for lesson 012 — when a mean over cells hides a single winner.

Everything here is recomputed from the two committed measurement files:

    research/measurements/track_rel_s1_best.json      the trained policy
    research/measurements/track_rel_zero_action.json  the do-nothing control

Nothing is transcribed. The lesson quotes only what this script prints.

    venv/bin/python docs/lessons/scripts/012_leave_one_out.py
"""
from __future__ import annotations

import json
from typing import Any

from bestiary.envs.hound_track import (
    P_DRIVE_STRAIGHT,
    P_FORWARD,
    P_STOP,
    P_TURN,
)
from bestiary.paths import RESEARCH

MEASUREMENTS = RESEARCH / "measurements"

# The six DRIVE-GRID cells, in the order the evaluation reports them. The
# seventh measured cell -- the stop command (0,0,0) -- is deliberately NOT part
# of this mean: it is scored separately as `stop_cell_mean`, because standing
# still earns ~800 either way and would swamp everything else.
GRID_ORDER = [
    "(0.5, 0.0, 0.0)",
    "(0.8, 0.0, 0.0)",
    "(-0.3, 0.0, 0.0)",
    "(0.5, 0.0, 0.4)",
    "(0.5, 0.0, -0.4)",
    "(0.0, 0.0, 0.45)",
]
STOP_CELL = "(0.0, 0.0, 0.0)"


def load() -> tuple[dict[str, Any], dict[str, Any]]:
    """Trained arm and control arm, each as {cell: mean episode return}."""
    best = json.loads((MEASUREMENTS / "track_rel_s1_best.json").read_text())
    zero = json.loads((MEASUREMENTS / "track_rel_zero_action.json").read_text())

    trained = {k: v["mean"] for k, v in best["trained"]["cells"].items()}
    control = {k: v["mean"] for k, v in zero["zero_action"]["cells"].items()}

    # The best-checkpoint file carries its own copy of the control arm. If the
    # two ever disagree the comparison is between different measurements, so
    # this is checked rather than assumed.
    embedded = {k: v["mean"] for k, v in best["zero_action"]["cells"].items()}
    for cell in GRID_ORDER:
        a, b = embedded[cell], control[cell]
        assert abs(a - b) < 1e-9, f"control arm differs for {cell}: {a} vs {b}"

    missing = [c for c in GRID_ORDER if c not in trained or c not in control]
    assert not missing, f"cells missing from a measurement file: {missing}"
    return trained, control


def mean(values: list[float]) -> float:
    assert values, "mean of no cells is not a number"
    return sum(values) / len(values)


def main() -> None:
    trained, control = load()
    gaps = {c: trained[c] - control[c] for c in GRID_ORDER}

    print("1. The per-cell table the headline replaced\n")
    print(f"  {'command':<18}{'policy':>10}{'control':>10}{'gap':>10}")
    for c in GRID_ORDER:
        print(f"  {c:<18}{trained[c]:>10.2f}{control[c]:>10.2f}{gaps[c]:>+10.2f}")
    total = sum(gaps.values())
    print(f"  {'sum':<18}{'':>10}{'':>10}{total:>+10.2f}")

    won = [c for c in GRID_ORDER if gaps[c] > 0]
    print(f"\n  cells won by the policy            {len(won)}/{len(GRID_ORDER)}")

    mt, mc = mean([trained[c] for c in GRID_ORDER]), mean([control[c] for c in GRID_ORDER])
    print(f"  mean policy                        {mt:.2f}")
    print(f"  mean control                       {mc:.2f}")
    print(f"  ratio                              {mt / mc:.2f}x")
    print(f"  mean gap  = {total:+.2f}/{len(GRID_ORDER)} = {total / len(GRID_ORDER):+.2f}")

    top = max(GRID_ORDER, key=lambda c: gaps[c])
    print("\n2. Concentration\n")
    print(f"  largest single gap  {top}  {gaps[top]:+.2f}")
    print(f"  share of the total  {gaps[top]:.2f}/{total:.2f} = "
          f"{100 * gaps[top] / total:.1f}%")

    print("\n3. Leave-one-out: drop each cell, recompute the headline\n")
    print(f"  {'dropped':<18}{'policy':>10}{'control':>10}{'ratio':>12}")
    for c in GRID_ORDER:
        rest = [x for x in GRID_ORDER if x != c]
        pt, pc = mean([trained[x] for x in rest]), mean([control[x] for x in rest])
        # A ratio whose denominator changes sign has no value, not a large one.
        ratio = f"{pt / pc:.2f}x" if pc > 0 else "UNDEFINED"
        print(f"  {c:<18}{pt:>10.2f}{pc:>10.2f}{ratio:>12}")

    print("\n4. Weighting cells by how often the command actually occurs\n")
    # The training command sampler (envs/hound_track.py::_resample_command)
    # draws STOP with P_STOP, turn-in-place with P_TURN, and otherwise a DRIVE
    # command: forward with P_FORWARD, straight with P_DRIVE_STRAIGHT. Each
    # measured cell inherits the probability of the branch it sits in, split
    # evenly between cells sharing a branch.
    p_drive = 1.0 - P_STOP - P_TURN
    fwd_straight = p_drive * P_FORWARD * P_DRIVE_STRAIGHT
    bwd_straight = p_drive * (1 - P_FORWARD) * P_DRIVE_STRAIGHT
    fwd_turning = p_drive * P_FORWARD * (1 - P_DRIVE_STRAIGHT)
    weights = {
        STOP_CELL: P_STOP,
        "(0.0, 0.0, 0.45)": P_TURN,
        "(0.5, 0.0, 0.0)": fwd_straight / 2,
        "(0.8, 0.0, 0.0)": fwd_straight / 2,
        "(-0.3, 0.0, 0.0)": bwd_straight,
        "(0.5, 0.0, 0.4)": fwd_turning / 2,
        "(0.5, 0.0, -0.4)": fwd_turning / 2,
    }
    # Backward-turning commands are sampled but were never measured, so their
    # mass is dropped and the rest renormalised. Stated, not hidden.
    covered = sum(weights.values())
    wt = sum(weights[c] * trained[c] for c in weights) / covered
    wc = sum(weights[c] * control[c] for c in weights) / covered
    for c, w in weights.items():
        print(f"  {c:<18}p = {w:.3f}")
    print(f"  unmeasured (backward turning)  p = {1 - covered:.3f}  dropped")
    print(f"\n  weighted policy   {wt:.2f}")
    print(f"  weighted control  {wc:.2f}")
    print(f"  ratio             {wt / wc:.4f}x   ({100 * (wt / wc - 1):+.1f}%)")


if __name__ == "__main__":
    main()
