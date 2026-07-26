"""Measure a trained policy the same way the do-nothing control is measured.

`guards/standing.py` asks the project's cheapest real question — does the
trained policy beat doing nothing? — but it answers it with two different kinds
of number. The numerator is `final_ep_rew_mean` read out of the ledger: a
*training* rollout mean, collected with SAC's exploration noise still on, over
episodes that averaged 799 steps. The denominator is a *deterministic*
zero-action rollout over 1000 steps, measured live. The comparison is between a
noisy past statistic and a clean present one.

That is not a small discrepancy. On `hound_desert_v0` the ledger numerator gives
1010.0 / 960.6 = x1.05; the greedy numerator gives 1218.3 / 960.6 = x1.27. The
guard's verdict at its 1.18 advisory margin flips on which one is used, and the
guard never states that it chose.

This module measures both arms under one protocol so the ratio means something:
same env, same episode count, same seed sequence, deterministic actions on both
sides. It deliberately does **not** change `standing.py` — correcting that
guard's numerator would change published verdicts, and `loop/README.md` requires
an operator for any edit that alters what a check asserts. This is an
instrument, not a relaxation.

    python -m bestiary.record.greedy_eval --run spyder_desert_v0
    python -m bestiary.record.greedy_eval --run hound_pd_desert_v0 --episodes 10 --json

Why greedy rather than stochastic: `learnings/007` records that a peak eval
score hides an unreliable policy, so the honest summary of a policy is the mean
of many deterministic episodes plus their spread — never a best-of. Both are
reported; the spread is the part that matters.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bestiary import paths

# Matches guards/standing.py so the two are comparable by construction. Episode
# i uses seed SEED0 + i on both arms, which is what makes the paired comparison
# legitimate rather than two independent samples of different terrain draws.
EPISODES = 5
SEED0 = 0


@dataclass(frozen=True, slots=True)
class Arm:
    """One policy's measured return distribution."""

    label: str
    returns: list[float]
    lengths: list[int]

    @property
    def mean(self) -> float:
        return float(np.mean(self.returns))

    @property
    def std(self) -> float:
        # ddof=1: these are a sample of episodes, not the population.
        return float(np.std(self.returns, ddof=1)) if len(self.returns) > 1 else 0.0

    @property
    def worst(self) -> float:
        return float(np.min(self.returns))

    @property
    def best(self) -> float:
        return float(np.max(self.returns))


def _run_dir(run: str) -> Path:
    d = paths.RUNS / run
    if not d.is_dir():
        raise FileNotFoundError(f"no run directory at {d}")
    return d


def _env_id(run_dir: Path) -> str:
    config = run_dir / "config.json"
    if not config.exists():
        raise FileNotFoundError(
            f"{config} is missing; the env a run trained on is only recorded there"
        )
    env_id = json.loads(config.read_text()).get("env_id")
    if not env_id:
        raise ValueError(f"{config} records no env_id")
    return str(env_id)


def _rollout(env_id: str, model, episodes: int, seed0: int) -> Arm:
    """Roll `episodes` deterministic episodes. `model=None` means zero action."""
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — importing registers the env ids

    env = gym.make(env_id)
    try:
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        returns: list[float] = []
        lengths: list[int] = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed0 + episode)
            total, steps = 0.0, 0
            while True:
                if model is None:
                    action = zero
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(steps)
        return Arm("zero-action" if model is None else "greedy", returns, lengths)
    finally:
        env.close()


def compare(run: str, episodes: int = EPISODES, seed0: int = SEED0,
            latest: bool = False) -> dict:
    """Greedy policy vs zero action on the run's own env, one protocol."""
    from stable_baselines3 import SAC

    run_dir = _run_dir(run)
    env_id = _env_id(run_dir)

    name = "ant_sac.zip" if latest else "ant_sac_best.zip"
    checkpoint = run_dir / name
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} does not exist")

    # CPU: inference on one env is not worth a CUDA context, and this must be
    # runnable while the GPU is held by a training run.
    model = SAC.load(checkpoint, device="cpu")

    trained = _rollout(env_id, model, episodes, seed0)
    standing = _rollout(env_id, None, episodes, seed0)
    ratio = trained.mean / standing.mean if standing.mean else float("inf")

    return {
        "run": run,
        "env_id": env_id,
        "checkpoint": checkpoint.name,
        "episodes": episodes,
        "seeds": list(range(seed0, seed0 + episodes)),
        "trained": asdict(trained) | {
            "mean": trained.mean, "std": trained.std,
            "worst": trained.worst, "best": trained.best,
        },
        "standing": asdict(standing) | {
            "mean": standing.mean, "std": standing.std,
            "worst": standing.worst, "best": standing.best,
        },
        "ratio_mean": ratio,
        "ratio_worst_case": (trained.worst / standing.mean) if standing.mean else float("inf"),
        "episodes_where_standing_wins": sum(
            1 for r in trained.returns if r < standing.mean
        ),
    }


def _format(result: dict) -> str:
    t, s = result["trained"], result["standing"]
    lines = [
        f"{result['run']}  ({result['env_id']}, {result['checkpoint']}, "
        f"{result['episodes']} episodes, seeds {result['seeds'][0]}-{result['seeds'][-1]})",
        f"  greedy      mean {t['mean']:8.1f}  sd {t['std']:7.1f}  "
        f"worst {t['worst']:8.1f}  best {t['best']:8.1f}",
        f"  zero-action mean {s['mean']:8.1f}  sd {s['std']:7.1f}  "
        f"worst {s['worst']:8.1f}  best {s['best']:8.1f}",
        f"  ratio (mean/mean)        x{result['ratio_mean']:.3f}",
        f"  ratio (worst greedy)     x{result['ratio_worst_case']:.3f}",
        f"  episodes below standing  {result['episodes_where_standing_wins']}"
        f" of {result['episodes']}",
        f"  per-episode greedy       {[round(v, 1) for v in t['returns']]}",
        f"  per-episode lengths      {t['lengths']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name under runs/")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed0", type=int, default=SEED0)
    parser.add_argument("--latest", action="store_true",
                        help="use ant_sac.zip instead of ant_sac_best.zip")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = compare(args.run, args.episodes, args.seed0, args.latest)
    print(json.dumps(result, indent=2) if args.json else _format(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
