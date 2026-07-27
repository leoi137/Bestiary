"""Arithmetic for research/learnings/010 — how much of a "seed spread" is checkpoint choice.

Cycle 006 compared the PD arm's two training seeds through their
`ant_sac_best.zip` files and read the resulting 91.55-point gap as a property
of the seed. `learnings/008` had already established that `*_best.zip` is
selected by argmax over ONE-episode evaluations, i.e. it is the luckiest
episode rather than the better policy.

This script runs the control that cycle should have run itself: the same
instrument, the same 60 seeds, on the checkpoints that were NOT selected by
argmax (`ant_sac.zip`, the final weights), and reports how much of the gap was
the seed and how much was the selection.

Inputs are the raw JSON emitted by `record/greedy_eval.py`:

    venv/bin/python -m bestiary.record.greedy_eval --run hound_pd_desert_v0 --episodes 60 --json
    venv/bin/python -m bestiary.record.greedy_eval --run hound_pd_desert_v0 --episodes 60 --latest --json
    (and the same two for hound_pd_desert_s1)

    venv/bin/python research/scripts/checkpoint_selection_spread.py \
        --best-s0 /tmp/geval_s0.json --best-s1 /tmp/geval_s1.json \
        --latest-s0 /tmp/lat_s0.json --latest-s1 /tmp/lat_s1.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

THRESHOLD = 1.18   # guards/standing.py's advisory margin; unsourced, see anomalies


def load(path: str) -> dict:
    d = json.loads(open(path).read())
    t = d["trained"]
    return {
        "checkpoint": d["checkpoint"],
        "run": d["run"],
        "returns": np.array(t["returns"], dtype=float),
        "crashes": int(t["crashes"]),
        "mean": float(t["mean"]),
        "sd": float(t["std"]),
        "ratio": float(d["ratio_mean"]),
        "standing": float(d["standing"]["mean"]),
        "episodes": int(d["episodes"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("best-s0", "best-s1", "latest-s0", "latest-s1"):
        ap.add_argument(f"--{k}", required=True)
    args = vars(ap.parse_args())

    b0, b1 = load(args["best_s0"]), load(args["best_s1"])
    l0, l1 = load(args["latest_s0"]), load(args["latest_s1"])

    print("ONE INSTRUMENT, ONE SEED SET (0-59), TWO CHECKPOINTS")
    print(f"  {'':22} {'mean':>9} {'sd':>8} {'median':>9} {'crashes':>8} {'ratio':>7}")
    for tag, a in (("seed 0  best", b0), ("seed 1  best", b1),
                   ("seed 0  latest", l0), ("seed 1  latest", l1)):
        print(f"  {tag:22} {a['mean']:9.2f} {a['sd']:8.2f} "
              f"{np.median(a['returns']):9.2f} {a['crashes']:8d} {a['ratio']:7.4f}")

    spread_best = b1["mean"] - b0["mean"]
    spread_latest = l1["mean"] - l0["mean"]
    print()
    print("THE SPREAD ATTRIBUTED TO THE SEED")
    print(f"  through *_best.zip     {spread_best:+9.2f} points")
    print(f"  through *_sac.zip      {spread_latest:+9.2f} points")
    print(f"  moved by checkpoint choice alone {abs(spread_best - spread_latest):8.2f} points")

    # Paired test on the latest checkpoints: same seeds, so pair episode-wise.
    d = l1["returns"] - l0["returns"]
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    print()
    print("PAIRED COMPARISON ON THE UNSELECTED CHECKPOINTS (n=%d, same seeds)" % n)
    print(f"  mean difference        {d.mean():+9.2f}")
    print(f"  sd of differences      {d.std(ddof=1):9.2f}")
    print(f"  standard error         {se:9.2f}")
    print(f"  t                      {d.mean() / se:9.2f}")
    print(f"  95% CI                 [{d.mean() - 1.96 * se:+.2f}, {d.mean() + 1.96 * se:+.2f}]")
    print("  -> the CI contains zero" if abs(d.mean()) < 1.96 * se
          else "  -> the CI excludes zero")

    print()
    print(f"DOES EITHER SEED CLEAR THE {THRESHOLD} MARGIN?")
    for tag, a in (("seed 0  best", b0), ("seed 1  best", b1),
                   ("seed 0  latest", l0), ("seed 1  latest", l1)):
        print(f"  {tag:22} ratio {a['ratio']:.4f}  "
              f"{'CLEARS' if a['ratio'] > THRESHOLD else 'does not clear'}")

    # The mean is a crash-rate proxy: show what one crash is worth.
    good = b1["returns"][b1["returns"] > 900]
    bad = b1["returns"][b1["returns"] <= 900]
    if len(bad):
        per_crash = (good.mean() - bad.mean()) / b1["episodes"]
        print()
        print("WHY THE MEAN IS MOSTLY A CRASH COUNT (seed 1, best checkpoint)")
        print(f"  non-crashed episodes   n={len(good):3d}  mean {good.mean():8.2f}")
        print(f"  crashed episodes       n={len(bad):3d}  mean {bad.mean():8.2f}")
        print(f"  one extra crash moves the n={b1['episodes']} mean by "
              f"{per_crash:.2f} points")
        print(f"  the {abs(spread_best):.2f}-point 'seed spread' is "
              f"{abs(spread_best) / per_crash:.1f} crashes' worth")


if __name__ == "__main__":
    main()
