"""The zero-action baseline under the command-tracking reward, measured.

WHY THIS EXISTS

`docs/theory/command-tracking-reward.md` Section 5 PREDICTS what a do-nothing
policy scores under the new reward: ~0.132/step on the command mixture, ~87 per
episode, and ~0 net on a drive-only eval grid. Those are the numbers the whole
window's binary question is asked against, and the note is explicit that they
are predictions computed from the kernel, not measurements of anything.

Section 5 also says the ONLY number that carries over from the old reward is
"the zero-action baseline re-measured under the new reward and a stated command
grid". This script is that re-measurement. It must run BEFORE the trained
policy is harvested, or the comparison has no denominator.

The old 955-1078 figures are dead as comparators and this script does not
compute them: the objective changed, and the old policies trained with
zero-filled command slots, so evaluating them under nonzero commands feeds them
observations off their training manifold.

TWO PROTOCOLS, AND THE DIFFERENCE MATTERS

`mixture`  -- commands drawn exactly as training draws them. This is what a
              training-time `ep_rew_mean` is comparable to. Its episode-to-
              episode variance is LARGE and not from the physics: only ~4
              commands are drawn per 1000-step episode, so a lucky episode that
              draws two STOPs scores far above one that draws none. Never read
              a single episode of this.

`grid`     -- the fixed eval grid of Section 3, run one command per episode.
              This is the headline protocol, because "does the policy beat
              zero-action?" is undefined over a mixture containing STOP, where
              zero-action is near-optimal BY DESIGN. The (0,0,0) cell is
              reported separately as a stop-competence check, where a good
              policy should MATCH zero-action rather than beat it.

Usage:
    venv/bin/python -m research.scripts.tracking_baseline --episodes 60
    venv/bin/python research/scripts/tracking_baseline.py --episodes 60 --json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# The Section 3 eval grid. The stop cell is LAST and is excluded from the
# headline ratio; see the module docstring.
EVAL_GRID = [
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.5, 0.0, 0.4),
    (0.5, 0.0, -0.4),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 0.0),      # stop-competence cell, reported separately
]
STOP_CELL = (0.0, 0.0, 0.0)


def _rollout(env, seed: int, forced_cmd=None) -> dict:
    """One zero-action episode. `forced_cmd` pins the command for the whole run."""
    env.reset(seed=seed)
    if forced_cmd is not None:
        env.unwrapped._cmd = np.array(forced_cmd, dtype=float)
        # Push the resample past the horizon so the command is genuinely held.
        env.unwrapped._steps_until_resample = 10**9
    action = np.zeros(env.action_space.shape[0])
    total = 0.0
    phi_v, phi_w, contact, ctrl = [], [], [], []
    steps = 0
    terminated = False
    while True:
        _, r, term, trunc, info = env.step(action)
        total += r
        phi_v.append(info["track_phi_v"])
        phi_w.append(info["track_phi_w"])
        contact.append(-info["reward_contact"])
        ctrl.append(-info["reward_ctrl"])
        steps += 1
        if term or trunc:
            terminated = term
            break
    return {
        "return": total,
        "steps": steps,
        "terminated": terminated,
        "mean_phi_v": float(np.mean(phi_v)),
        "mean_phi_w": float(np.mean(phi_w)),
        "mean_track": float(np.mean(np.array(phi_v) * np.array(phi_w))),
        "mean_contact_cost": float(np.mean(contact)),
        "mean_ctrl_cost": float(np.mean(ctrl)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=60,
                    help="episodes for the mixture protocol, and per grid cell")
    ap.add_argument("--grid-episodes", type=int, default=20)
    ap.add_argument("--env", default="HoundPDTrackDesert-v0")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym

    import bestiary.envs  # noqa: F401

    env = gym.make(args.env)

    mixture = [_rollout(env, seed=s) for s in range(args.episodes)]
    mix_returns = np.array([e["return"] for e in mixture])

    grid = {}
    for cell in EVAL_GRID:
        eps = [_rollout(env, seed=1000 + i, forced_cmd=cell)
               for i in range(args.grid_episodes)]
        rets = np.array([e["return"] for e in eps])
        grid[str(cell)] = {
            "command": list(cell),
            "mean": float(rets.mean()),
            "sd": float(rets.std(ddof=1)),
            "mean_track": float(np.mean([e["mean_track"] for e in eps])),
            "mean_phi_v": float(np.mean([e["mean_phi_v"] for e in eps])),
            "mean_phi_w": float(np.mean([e["mean_phi_w"] for e in eps])),
            "crashes": int(sum(e["terminated"] for e in eps)),
            "episodes": len(eps),
        }

    drive_cells = [v for k, v in grid.items() if tuple(v["command"]) != STOP_CELL]
    result = {
        "env": args.env,
        "policy": "zero-action",
        "mixture": {
            "episodes": args.episodes,
            "mean": float(mix_returns.mean()),
            "sd": float(mix_returns.std(ddof=1)),
            "min": float(mix_returns.min()),
            "max": float(mix_returns.max()),
            "mean_track_per_step": float(np.mean([e["mean_track"] for e in mixture])),
            "mean_contact_per_step": float(np.mean([e["mean_contact_cost"] for e in mixture])),
            "crashes": int(sum(e["terminated"] for e in mixture)),
        },
        "grid": grid,
        # THE headline denominator: the drive-grid mean excluding the stop cell.
        "drive_grid_mean": float(np.mean([c["mean"] for c in drive_cells])),
        "stop_cell_mean": grid[str(STOP_CELL)]["mean"],
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    m = result["mixture"]
    print(f"zero-action under {args.env}")
    print(f"\nMIXTURE  ({m['episodes']} episodes, commands drawn as in training)")
    print(f"  return   {m['mean']:8.2f} +/- {m['sd']:.2f}   "
          f"range [{m['min']:.1f}, {m['max']:.1f}]")
    print(f"  track    {m['mean_track_per_step']:8.4f}/step   "
          f"contact {m['mean_contact_per_step']:.4f}/step   crashes {m['crashes']}")
    print(f"\nEVAL GRID  ({args.grid_episodes} episodes per cell, command held)")
    print(f"  {'command':>20}  {'return':>9}  {'sd':>7}  {'track':>7}  "
          f"{'phi_v':>6}  {'phi_w':>6}")
    for k, v in grid.items():
        tag = "  <- stop cell" if tuple(v["command"]) == STOP_CELL else ""
        print(f"  {k:>20}  {v['mean']:9.2f}  {v['sd']:7.2f}  {v['mean_track']:7.4f}  "
              f"{v['mean_phi_v']:6.3f}  {v['mean_phi_w']:6.3f}{tag}")
    print(f"\n  DRIVE-GRID MEAN (stop cell excluded): {result['drive_grid_mean']:.2f}")
    print(f"  stop cell:                            {result['stop_cell_mean']:.2f}")


if __name__ == "__main__":
    main()
