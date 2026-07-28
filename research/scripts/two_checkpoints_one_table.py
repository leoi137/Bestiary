"""The Phi_v/Phi_w ratio table in the record mixes two checkpoints.

`learnings/011`, `research/episodes/007-*.md` and ledger row 4 all present the
same story: the policy's Phi_v goes x1.46 while its Phi_w goes x0.48, so speed
bought with heading cancels, and the control cost then decides the sign.

The mechanism is real. The two multipliers are not from the checkpoint the rest
of that paragraph describes.

The four-term decomposition (ctrl_cost = 105.5% of the gap) is measured on the
FINAL unselected `ant_sac.zip` at 1.5M steps. The x1.46 / x0.48 pair reproduces
only from the MID-RUN `ant_sac_best.zip` at 950k. On the final checkpoint the
same six-cell means are x1.195 and x0.606.

That would be a citation slip and nothing more, except for `anomalies.jsonl`
row 19: `ant_sac_best.zip` is overwritten in place whenever an eval beats the
prior best, so the 950k artifact these two numbers came from NO LONGER EXISTS.
They cannot be re-derived from anything on disk. This script is now the only
thing that can say where they came from.

Run:  venv/bin/python research/scripts/012_two_checkpoints_one_table.py
"""
from __future__ import annotations

import json

import numpy as np

from bestiary import paths

STOP_CELL = "(0.0, 0.0, 0.0)"

MEASUREMENTS = (
    ("final  1.5M", "hound_track_desert_s0_final_sac.json"),
    ("mid-run 950k", "hound_track_desert_s0_midrun_950k.json"),
)


def drive_cell_means(arm: dict) -> tuple[float, float, float]:
    """Mean Phi_v, Phi_w and track over the six DRIVE cells.

    The stop cell is excluded because `track_eval`'s own headline excludes it:
    a standing machine scores ~0.93 there for correctly not moving, which is
    the reading that broke the parked detector (`learnings/011`).
    """
    cells = arm["cells"]
    drive = [k for k in cells if k != STOP_CELL]
    return (
        float(np.mean([cells[k]["mean_phi_v"] for k in drive])),
        float(np.mean([cells[k]["mean_phi_w"] for k in drive])),
        float(np.mean([cells[k]["mean_track"] for k in drive])),
    )


def main() -> None:
    rows = []
    for label, fname in MEASUREMENTS:
        data = json.loads((paths.RESEARCH / "measurements" / fname).read_text())
        pol_v, pol_w, pol_t = drive_cell_means(data["trained"])
        zero_v, zero_w, zero_t = drive_cell_means(data["zero_action"])
        rows.append((label, data["checkpoint"], pol_v, pol_w, pol_t,
                     zero_v, zero_w, zero_t))

    print("Six-cell drive-grid means, both arms, both checkpoints")
    print(f"{'measurement':13s} {'checkpoint':18s} "
          f"{'phi_v':>8s} {'phi_w':>8s} {'track':>9s}")
    for label, ckpt, pv, pw, pt, zv, zw, zt in rows:
        print(f"{label:13s} {ckpt:18s} {pv:8.4f} {pw:8.4f} {pt:9.5f}   policy")
        print(f"{'':13s} {'(zero action)':18s} {zv:8.4f} {zw:8.4f} {zt:9.5f}   control")

    print("\nRatios policy/zero -- the pair the record quotes is x1.46 / x0.48")
    for label, ckpt, pv, pw, pt, zv, zw, zt in rows:
        print(f"  {label:13s} ({ckpt:18s})  "
              f"phi_v x{pv / zv:.3f}   phi_w x{pw / zw:.3f}   track x{pt / zt:.3f}")

    final, midrun = rows[0], rows[1]
    print("\nWHICH CHECKPOINT PRODUCES THE QUOTED PAIR")
    for label, ckpt, pv, pw, pt, zv, zw, zt in rows:
        match_v = abs(pv / zv - 1.46) < 0.01
        match_w = abs(pw / zw - 0.48) < 0.01
        print(f"  {label:13s}: x1.46 {'MATCHES' if match_v else 'does NOT match'}"
              f" ({pv / zv:.3f}),  x0.48 "
              f"{'MATCHES' if match_w else 'does NOT match'} ({pw / zw:.3f})")

    print("\nAND THE PAIR DOES NOT COMPOSE INTO THE TRACK RATIO")
    print("  'x1.46 gained is cancelled by x0.48 lost' reads as a product, but")
    print("  drive_grid_track is a mean of per-cell mean(Phi_v*Phi_w), and")
    print("  mean-of-products != product-of-means:")
    for label, ckpt, pv, pw, pt, zv, zw, zt in rows:
        print(f"    {label:13s}  mean-of-products {pt:.5f}  vs  "
              f"product-of-means {pv * pw:.5f}   "
              f"(off by {100 * (pv * pw - pt) / pt:+.1f}%)")
    lv, lw = midrun[2] / midrun[5], midrun[3] / midrun[6]
    print(f"    ratio arithmetic: {lv:.3f} x {lw:.3f} = {lv * lw:.3f}, "
          f"yet the measured track ratio at that same checkpoint is "
          f"x{midrun[4] / midrun[7]:.3f}")

    print("\nWHAT SURVIVES: the per-cell Delta-track per step, which is what")
    print("the release condition in nulls.jsonl row 4 is actually written over.")
    print(f"  final checkpoint: policy {final[4]:.5f}/step vs "
          f"zero {final[7]:.5f}/step  ->  gain {final[4] - final[7]:+.5f}/step")


if __name__ == "__main__":
    main()
