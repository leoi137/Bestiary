"""Where a command-tracking return actually goes: the four reward terms, summed.

    venv/bin/python research/scripts/track_return_decomposition.py --run <name>
    venv/bin/python research/scripts/track_return_decomposition.py --run <name> --latest

`track_eval` answers *how much* a policy scores. It does not answer *why*, and
on 2026-07-27 that gap produced a wrong diagnosis that survived until an
independent refutation: cycle 007 saw a drive-grid return of -4.19 next to 71
crashes in 120 episodes and concluded the crashes were the cause. They are not.
The termination penalty is one term of four, and this script is what settles it
rather than an argument about it.

`HoundPDTrackDesert-v0` already exposes every component in its `info` dict
(`envs/hound_track.py:328-331`), so nothing here re-derives the reward -- it
sums what the env itself reports, which is the only version that cannot drift
from what the policy was actually trained on.

Reported per episode and averaged over the six DRIVE cells (the stop cell is
excluded exactly as `track_eval` excludes it, and reported separately), so the
numbers line up with `drive_grid_mean` term for term:

    return = track - ctrl - contact - termination
"""
from __future__ import annotations

import argparse
import json

import gymnasium as gym
import numpy as np

import bestiary.envs  # noqa: F401 -- registers the env ids
from bestiary import paths
from bestiary.record.track_eval import (
    EVAL_GRID,
    STOP_CELL,
    TRACK_ENV,
    assert_decomposition_complete,
    discover_terms,
)

# This module used to carry its OWN copy of the hardcoded four-term tuple --
# a second instance of anomalies.jsonl row 39, in the script whose entire
# output is a decomposition. Terms are now discovered from the env's step info
# and checked to sum to the reward, exactly as in `record/track_eval.py`.


def episode(env, seed: int, policy, cmd) -> dict:
    """One episode under a held command; sum each reward term over its steps."""
    obs, _ = env.reset(seed=seed)
    env.unwrapped._cmd = np.array(cmd, dtype=float)
    env.unwrapped._steps_until_resample = 10**9
    obs = env.unwrapped._get_obs()

    zero = np.zeros(env.action_space.shape[0])
    terms: tuple[str, ...] | None = None
    sums: dict[str, float] = {}
    total, steps = 0.0, 0
    terminated = False
    while True:
        action = zero if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(action)
        total += r
        if terms is None:
            terms = discover_terms(info)
            sums = dict.fromkeys(terms, 0.0)
        assert_decomposition_complete(terms, info, r, steps)
        for t in terms:
            sums[t] += float(info[t])
        steps += 1
        if term or trunc:
            terminated = term
            break
    return {"return": total, "steps": steps, "terminated": terminated,
            "terms": list(terms), **sums}


def arm(env, policy, episodes: int, seed0: int) -> dict:
    out = {}
    seen: set[tuple[str, ...]] = set()
    for cell in EVAL_GRID:
        eps = [episode(env, seed0 + i, policy, cell) for i in range(episodes)]
        seen.update(tuple(e["terms"]) for e in eps)
        if len(seen) != 1:
            raise SystemExit(f"episodes disagree on reward terms: {sorted(seen)}")
        terms = tuple(next(iter(seen)))
        out[str(cell)] = {
            "command": list(cell),
            "return": float(np.mean([e["return"] for e in eps])),
            "steps": float(np.mean([e["steps"] for e in eps])),
            "crashes": int(sum(e["terminated"] for e in eps)),
            **{t: float(np.mean([e[t] for e in eps])) for t in terms},
        }
    drive = [c for c in out.values() if tuple(c["command"]) != STOP_CELL]
    agg = {
        "drive_grid_return": float(np.mean([c["return"] for c in drive])),
        "drive_grid_steps": float(np.mean([c["steps"] for c in drive])),
        "drive_grid_crashes": int(sum(c["crashes"] for c in drive)),
        "terms": list(terms),
        **{f"drive_grid_{t}": float(np.mean([c[t] for c in drive])) for t in terms},
    }
    return {"cells": out, **agg}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=20, help="episodes PER GRID CELL")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--latest", action="store_true",
                    help="ant_sac.zip (unselected) instead of ant_sac_best.zip")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from stable_baselines3 import SAC

    ckpt = paths.RUNS / args.run / ("ant_sac.zip" if args.latest else "ant_sac_best.zip")
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")

    env = gym.make(TRACK_ENV)
    try:
        policy = SAC.load(ckpt, device="cpu")
        trained = arm(env, policy, args.episodes, args.seed0)
        zero = arm(env, None, args.episodes, args.seed0)
    finally:
        env.close()

    gap = trained["drive_grid_return"] - zero["drive_grid_return"]
    # Signed contribution of each term to the gap. track is earned (+ is good);
    # the three costs are already negative in `info`, so a more-negative cost
    # shows up as a negative contribution. They sum to the gap by construction.
    # "By construction" is now true. It was not before: the term list was a
    # hardcoded 4-tuple and this env family has envs paying five, so
    # `residual_check` below was the only field that knew, and it printed a
    # small number next to a table that looked complete.
    if list(trained["terms"]) != list(zero["terms"]):
        raise SystemExit(
            f"arms disagree on reward terms: {trained['terms']} vs {zero['terms']}"
        )
    terms = list(trained["terms"])
    contrib = {t: trained[f"drive_grid_{t}"] - zero[f"drive_grid_{t}"] for t in terms}

    result = {
        "run": args.run,
        "checkpoint": ckpt.name,
        "episodes_per_cell": args.episodes,
        "seed0": args.seed0,
        "trained": trained,
        "zero_action": zero,
        "gap_to_zero_action": gap,
        "gap_contribution": contrib,
        "gap_contribution_share": {
            t: (v / gap if gap else float("nan")) for t, v in contrib.items()
        },
        "residual_check": gap - sum(contrib.values()),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"{args.run}  [{ckpt.name}]  {args.episodes} episodes/cell, "
          f"drive grid = {len(EVAL_GRID) - 1} cells\n")
    print(f"{'':22} {'policy':>12} {'zero action':>12} {'contribution':>14}")
    for t in terms:
        print(f"  {t:20} {trained[f'drive_grid_{t}']:12.2f} "
              f"{zero[f'drive_grid_{t}']:12.2f} {contrib[t]:14.2f}")
    print(f"  {'-' * 60}")
    print(f"  {'return':20} {trained['drive_grid_return']:12.2f} "
          f"{zero['drive_grid_return']:12.2f} {gap:14.2f}")
    print(f"  {'mean episode steps':20} {trained['drive_grid_steps']:12.1f} "
          f"{zero['drive_grid_steps']:12.1f}")
    print(f"  {'crashes':20} {trained['drive_grid_crashes']:12d} "
          f"{zero['drive_grid_crashes']:12d}"
          f"   of {args.episodes * (len(EVAL_GRID) - 1)}")
    print(f"\n  residual (must be ~0): {result['residual_check']:.2e}\n")
    print("  share of the gap explained by each term:")
    for t, s in result["gap_contribution_share"].items():
        print(f"    {t:22} {s * 100:7.1f}%")


if __name__ == "__main__":
    main()
