"""The arithmetic behind cycle 010's length-bias measurement.

Every number quoted in `research/episodes/` or `research/anomalies.jsonl` about
the `mean_track` length bias is computed here, from the committed measurement
`research/measurements/track_length_bias_s0.json`. Nothing is asserted in prose.

    venv/bin/python -m research.scripts.track_length_bias_math
    venv/bin/python research/scripts/track_length_bias_math.py

WHAT THIS IS CHECKING

`anomalies.jsonl` row 20 recorded the bias with two numbers: raw `mean_track`
0.0955 for the policy against zero action's 0.0650 (x1.47), collapsing to
0.0644 vs 0.0650 (x0.992) once normalised to a common 1000-step horizon. That
was the finding that made the bias look decisive -- it reverses the sign of the
tracking verdict.

This script re-derives the same quantities from a fresh measurement of the same
run under the same protocol (n=20/cell, seeds 1000-1019, `ant_sac.zip`) and
reports the gap between what row 20 recorded and what reproduces today.
"""
from __future__ import annotations

import json

from bestiary import paths

MEASUREMENT = paths.RESEARCH / "measurements" / "track_length_bias_s0.json"

# anomalies.jsonl row 20, 2026-07-27. Quoted here to be CHECKED, not trusted.
ROW20_RAW_POLICY = 0.0955
ROW20_RAW_ZERO = 0.0650
ROW20_NORM_POLICY = 0.0644
ROW20_NORM_ZERO = 0.0650

STOP_CELL = (0.0, 0.0, 0.0)


def main() -> None:
    d = json.loads(MEASUREMENT.read_text())
    zero, pol = d["zero_action"], d["trained"]

    print(f"measurement: {MEASUREMENT.name}")
    print(f"env {d['env']}  run {d['run']}  checkpoint {d['checkpoint']}")
    print(f"n = {d['episodes_per_cell']}/cell, seed0 = {d['seed0']}, "
          f"deterministic = {d['deterministic']}\n")

    # --- the three aggregates, both arms -------------------------------------
    rows = [
        ("mean_track (per-episode mean of means)", "drive_grid_track"),
        ("mean_track_stepw (per step taken)", "drive_grid_track_stepw"),
        ("track_per_horizon (banked per horizon)", "drive_grid_track_per_horizon"),
    ]
    print(f"{'aggregate':42s} {'zero':>9s} {'policy':>9s} {'ratio':>8s}")
    ratios = {}
    for label, key in rows:
        z, p = zero[key], pol[key]
        ratios[key] = p / z
        print(f"{label:42s} {z:9.6f} {p:9.6f} {ratios[key]:8.4f}")

    # --- survival, which is the whole mechanism ------------------------------
    print(f"\nmean episode length: zero {zero['drive_grid_steps']:.2f}, "
          f"policy {pol['drive_grid_steps']:.2f} "
          f"({pol['drive_grid_steps'] / zero['drive_grid_steps']:.4f} of it)")
    print(f"crashes over the whole grid: zero {zero['crashes']}, "
          f"policy {pol['crashes']}")

    # The bias is exactly the survival shortfall: banking a rate for fewer steps
    # scales the income by the fraction of the horizon actually lived.
    bias = pol["drive_grid_track"] / pol["drive_grid_track_per_horizon"]
    print(f"\nlength bias on the policy arm: "
          f"mean_track / track_per_horizon = {bias:.4f} "
          f"({(bias - 1) * 100:+.1f}%)")
    print(f"length bias on the zero arm:   "
          f"{zero['drive_grid_track'] / zero['drive_grid_track_per_horizon']:.4f} "
          f"(must be exactly 1.0000 -- it never crashes)")

    # --- versus what row 20 recorded ----------------------------------------
    print("\n--- anomalies.jsonl row 20, as recorded vs as reproduced ---")
    print(f"{'quantity':28s} {'row 20':>9s} {'today':>9s} {'delta':>9s}")
    for label, recorded, today in (
        ("raw policy", ROW20_RAW_POLICY, pol["drive_grid_track"]),
        ("raw zero", ROW20_RAW_ZERO, zero["drive_grid_track"]),
        ("normalised policy", ROW20_NORM_POLICY, pol["drive_grid_track_per_horizon"]),
        ("normalised zero", ROW20_NORM_ZERO, zero["drive_grid_track_per_horizon"]),
    ):
        print(f"{label:28s} {recorded:9.4f} {today:9.4f} {today - recorded:+9.4f}")

    print(f"\nraw ratio:        row 20 {ROW20_RAW_POLICY / ROW20_RAW_ZERO:.4f}   "
          f"today {ratios['drive_grid_track']:.4f}")
    print(f"normalised ratio: row 20 {ROW20_NORM_POLICY / ROW20_NORM_ZERO:.4f}   "
          f"today {ratios['drive_grid_track_per_horizon']:.4f}")

    # --- the two claims this cycle pre-registered ---------------------------
    print("\n--- cycle 010's predictions, resolved ---")
    p1 = abs(zero["drive_grid_track"] - zero["drive_grid_track_per_horizon"]) < 0.0005
    print(f"[{'TRUE ' if p1 else 'FALSE'}] p=0.85  zero action's two aggregates agree "
          f"within 0.0005 (delta "
          f"{abs(zero['drive_grid_track'] - zero['drive_grid_track_per_horizon']):.6f})")

    p2 = 0.060 <= pol["drive_grid_track_per_horizon"] <= 0.069
    print(f"[{'TRUE ' if p2 else 'FALSE'}] p=0.70  policy's track_per_horizon in "
          f"[0.060, 0.069] (got {pol['drive_grid_track_per_horizon']:.6f})")

    raw_r, norm_r = ratios["drive_grid_track"], ratios["drive_grid_track_per_horizon"]
    p3 = raw_r > 1.4 and norm_r < 1.05
    print(f"[{'TRUE ' if p3 else 'FALSE'}] p=0.70  raw ratio > 1.4 AND normalised "
          f"< 1.05 (got raw {raw_r:.4f}, normalised {norm_r:.4f})")
    if not p3:
        print("         ^ the normalised half held; the RAW half did not. The raw "
              "ratio is 1.10, not the 1.47 row 20 recorded, so there is far less "
              "bias to remove than the anomaly implies -- see the episode.")

    # --- per-cell, because the grid mean hides where the crashes are ---------
    print("\n--- per drive cell ---")
    print(f"{'command':18s} {'steps':>8s} {'crash':>6s} {'mean_track':>11s} "
          f"{'per_horizon':>12s}")
    for key, c in pol["cells"].items():
        if tuple(c["command"]) == STOP_CELL:
            continue
        print(f"{key:18s} {c['mean_steps']:8.1f} {c['crashes']:6d} "
              f"{c['mean_track']:11.5f} {c['track_per_horizon']:12.5f}")

    # --- both checkpoints, and the corrections an adversarial review forced --
    #
    # Three things this section exists to keep honest, all of them raised by the
    # refutation rather than by the analysis:
    #
    #   1. The bias is quoted as a BIAS on each checkpoint (mean_track /
    #      track_per_horizon), not as a ratio of ratios. The two coincide only
    #      because the zero arm's bias is exactly 1.0, which is a fact about
    #      zero action never crashing and not something to lean on silently.
    #   2. WHICH cell carries the bias differs by checkpoint. Naming (0.5,0,0.4)
    #      as "the" crashing cell is true of ant_sac.zip and false of
    #      ant_sac_best.zip.
    #   3. Dropping the crashing cells drives the bias to exactly 1.0, which is
    #      the real shape of the finding: this is not a diffuse length effect,
    #      it is one cell per checkpoint.
    print("\n--- both checkpoints, with the crashing cells identified ---")
    for path in (MEASUREMENT,
                 MEASUREMENT.with_name("track_length_bias_s0_best.json")):
        if not path.exists():
            print(f"{path.name}: MISSING")
            continue
        dd = json.loads(path.read_text())
        p = dd["trained"]
        drive = [c for c in p["cells"].values()
                 if tuple(c["command"]) != STOP_CELL]
        crashing = [c for c in drive if c["crashes"] > 0]
        clean = [c for c in drive if c["crashes"] == 0]
        mean = lambda cs, k: sum(c[k] for c in cs) / len(cs)  # noqa: E731

        bias = p["drive_grid_track"] / p["drive_grid_track_per_horizon"]
        print(f"\n{dd['checkpoint']}")
        print(f"  bias (mean_track / track_per_horizon) = {bias:.4f} "
              f"({(bias - 1) * 100:+.1f}%)")
        print("  crashing cell(s): "
              + ", ".join(f"{tuple(c['command'])} "
                          f"({c['crashes']}/{c['episodes']} crashes, "
                          f"{c['mean_steps']:.1f} steps)" for c in crashing))
        if clean:
            clean_bias = mean(clean, "mean_track") / mean(clean, "track_per_horizon")
            print(f"  bias over the {len(clean)} NON-crashing cells only = "
                  f"{clean_bias:.6f}  <- must be exactly 1.0: every episode "
                  f"runs the full horizon, so there is no bias to have")


    # --- the naive identity, and where it fails ------------------------------
    #
    # It is tempting to say bias = 1 / (fraction of the horizon lived). That is
    # true of the GRID aggregate only by coincidence of weighting; at cell level
    # it is false, because the episodes that crash also track at a different
    # RATE than the ones that survive. Printed so nobody has to take it on faith.
    print("\n--- bias vs 1/(fraction of horizon lived), per crashing cell ---")
    print(f"{'checkpoint':18s} {'command':18s} {'bias':>8s} {'1/frac':>8s}")
    for path in (MEASUREMENT,
                 MEASUREMENT.with_name("track_length_bias_s0_best.json")):
        if not path.exists():
            continue
        dd = json.loads(path.read_text())
        for key, c in dd["trained"]["cells"].items():
            if tuple(c["command"]) == STOP_CELL or c["crashes"] == 0:
                continue
            cell_bias = c["mean_track"] / c["track_per_horizon"]
            inv_frac = c["horizon"] / c["mean_steps"]
            print(f"{dd['checkpoint']:18s} {key:18s} {cell_bias:8.4f} "
                  f"{inv_frac:8.4f}")

    # --- replication on an independent seed block ----------------------------
    rep = MEASUREMENT.with_name("track_length_bias_s0_seed5000.json")
    if rep.exists():
        rd = json.loads(rep.read_text())
        rz, rp = rd["zero_action"], rd["trained"]
        print(f"\n--- replication: {rep.name} "
              f"(n={rd['episodes_per_cell']}/cell, seed0={rd['seed0']}, "
              f"{rd['checkpoint']}) ---")
        print(f"raw ratio        {rp['drive_grid_track'] / rz['drive_grid_track']:.4f}")
        print("normalised ratio "
              f"{rp['drive_grid_track_per_horizon'] / rz['drive_grid_track_per_horizon']:.4f}")
        print(f"policy crashes over the grid: {rp['crashes']}, "
              f"mean steps {rp['drive_grid_steps']:.1f}")
    else:
        print(f"\n{rep.name}: MISSING")


if __name__ == "__main__":
    main()
