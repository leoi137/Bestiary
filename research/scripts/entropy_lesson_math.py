"""Numbers for docs/lessons/005 — SAC's entropy coefficient.

    venv/bin/python research/scripts/entropy_lesson_math.py

The number rule: a lesson's arithmetic comes from a committed script, never
from prose. This one does two jobs.

1. Pull the actual `ent_coef` trajectory out of both runs' training logs, so
   the comparison between the old forward-velocity reward and the new
   command-tracking one is read rather than remembered.
2. Work the target-entropy arithmetic. SAC's default target is
   H_target = -dim(A), which for the 16-actuator hound is -16 nats. Converting
   that to a per-actuator standard deviation is what turns "-16 nats" from a
   symbol into something you can picture.
"""
from __future__ import annotations

import math
import re

from bestiary import paths

RUNS = {
    "hound_pd_desert_s1": "forward-velocity reward (the old one)",
    "hound_track_desert_s0": "command-tracking reward (the new one)",
}
# Sampled rather than dumped: the logs hold thousands of blocks each.
AT_STEPS = (50_000, 100_000, 215_000, 400_000, 700_000, 1_000_000, 1_400_000)

ACT_DIM = 16  # hound16: 4 wheels + 12 leg joints. runs/*/config.json obs_spec.


def trajectory(run: str) -> list[tuple[int, float]]:
    """(total_timesteps, ent_coef) for every logged block, in order."""
    log = paths.RUNS / run / "train.log"
    if not log.exists():
        return []
    text = log.read_text()
    # SB3 prints a fixed-width table; pair each ent_coef with the
    # total_timesteps printed in the same block.
    blocks = text.split("---------------------------------")
    out: list[tuple[int, float]] = []
    for b in blocks:
        m_s = re.search(r"total_timesteps\s*\|\s*([0-9.e+]+)", b)
        m_e = re.search(r"ent_coef\s*\|\s*([0-9.e+-]+)", b)
        if m_s and m_e:
            try:
                out.append((int(float(m_s.group(1))), float(m_e.group(1))))
            except ValueError:
                continue
    return out


def nearest(traj: list[tuple[int, float]], step: int) -> tuple[int, float] | None:
    later = [p for p in traj if p[0] >= step]
    return later[0] if later else None


def sigma_for_entropy(h_total: float, dim: int) -> float:
    """Per-dimension std of a diagonal Gaussian with total differential entropy h_total.

    H = sum_i 0.5 * ln(2*pi*e*sigma_i^2).  With all sigma equal:
        H = dim * 0.5 * ln(2*pi*e*sigma^2)
    =>  sigma = sqrt( exp(2*H/dim) / (2*pi*e) )
    """
    return math.sqrt(math.exp(2.0 * h_total / dim) / (2.0 * math.pi * math.e))


def main() -> None:
    print(f"Target entropy H_target = -dim(A) = -{ACT_DIM} nats  (SAC default)")
    sigma = sigma_for_entropy(-float(ACT_DIM), ACT_DIM)
    print(f"  a diagonal Gaussian with H = -{ACT_DIM} nats over {ACT_DIM} dims has")
    print(f"  per-actuator sigma = {sigma:.6f}")
    print(f"  check: H = {ACT_DIM} * 0.5 * ln(2*pi*e*{sigma:.6f}^2) = "
          f"{ACT_DIM * 0.5 * math.log(2 * math.pi * math.e * sigma ** 2):.4f} nats")
    print()

    for run, label in RUNS.items():
        traj = trajectory(run)
        if not traj:
            print(f"{run}: no log found")
            continue
        print(f"{run} — {label}")
        print(f"  {len(traj)} logged blocks, {traj[0][0]} .. {traj[-1][0]} steps")
        for s in AT_STEPS:
            p = nearest(traj, s)
            if p:
                print(f"    at >= {s:>9,}: step {p[0]:>9,}  ent_coef = {p[1]:.3e}")
        lo = min(traj, key=lambda p: p[1])
        print(f"    minimum:           step {lo[0]:>9,}  ent_coef = {lo[1]:.3e}")
        print(f"    final:             step {traj[-1][0]:>9,}  ent_coef = {traj[-1][1]:.3e}")
        print()

    a = nearest(trajectory("hound_pd_desert_s1"), 1_000_000)
    b = min(trajectory("hound_track_desert_s0"), key=lambda p: p[1], default=None)
    if a and b:
        print(f"ratio at the tracking run's minimum: {a[1]:.3e} / {b[1]:.3e} = "
              f"{a[1] / b[1]:.1f}x")


if __name__ == "__main__":
    main()
