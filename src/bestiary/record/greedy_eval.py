"""Measure a trained policy the same way the do-nothing control is measured.

`guards/standing.py` asks the project's cheapest real question — does the
trained policy beat doing nothing? — but it answers it with two different kinds
of number. The numerator is `final_ep_rew_mean` read out of the ledger: a
*training* rollout mean, collected with SAC's exploration noise still on, over
episodes that averaged 799 steps. The denominator is a *deterministic*
zero-action rollout over 1000 steps, measured live. The comparison is between a
noisy past statistic and a clean present one.

That discrepancy is real but it is smaller than it first looks, and the first
version of this docstring got it wrong in an instructive way. It argued that
"the greedy numerator gives 1218.3 / 960.6 = x1.27" — using `best_eval_return`,
which is a maximum over a noisy sequence and is exactly the statistic
`learnings/007` exists to condemn. Measured properly at n=60, the honest greedy
ratio for `hound_desert_v0` is **x1.042**, essentially identical to the ledger
numerator this module was written to replace. A tool built to remove a bias was
motivated by a number exhibiting that bias.

The real defect in `standing.py` is not the size of the gap. It is that the
comparison is unstated and the sample is tiny.

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

This is also where the ledger's `eval_crash_rate` is computed. `record/ledger.py`
reads `crash_rate` off the trained arm rather than re-deriving it, so the
definition of a crash lives in exactly one place: `Arm.crashes`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bestiary import paths

# Episode i uses seed SEED0 + i on BOTH arms, which is what makes this a paired
# comparison rather than two independent samples of different terrain draws.
#
# 20, not 5, and not standing.py's 3. These policies are bimodal: they either
# complete the episode or fail early, at measured rates of 26.7% (torque hound,
# 16/60) and 10.0% (PD hound, 6/60). At a 26.7% failure rate, five episodes
# return a clean sweep 0.733**5 = 21% of the time, so n=5 routinely produces a
# confident picture of a policy that fails one run in four. That is not
# hypothetical: an n=5 draw in this repo once showed the torque hound at x1.265
# with 0/5 failures, against its true x1.042 with 16/60 — and it was reported as
# overturning a published result before an independent check killed it.
#
# A crash rate is a proportion, and proportions need samples. Anything claiming
# reliability should raise this further and say what n it used.
EPISODES = 20
SEED0 = 0


@dataclass(frozen=True, slots=True)
class Arm:
    """One policy's measured return distribution."""

    label: str
    returns: list[float]
    lengths: list[int]
    # Per-episode `terminated` exactly as the env reported it. NOT inferred
    # from `lengths` -- see `crashes` for why that distinction is the whole
    # point of this field.
    terminated: list[bool]
    # The truncation horizon the episodes were rolled under, carried so a
    # consumer never has to look it up a second way and get a different answer.
    max_episode_steps: int

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

    @property
    def crashes(self) -> int:
        """Episodes the env ended early because the robot became unhealthy.

        Counted from the env's own `terminated` flag, never from
        `length < max_episode_steps`. The two agree today, and that agreement
        is a property of the current envs rather than of the metric: every env
        here returns `truncated=False` from `step` and lets the `TimeLimit`
        wrapper supply the horizon, so going unhealthy is the only way to stop
        short. `_rollout` asserts that rather than assuming it.

        The proxy is wrong in both directions the moment that changes:

        * An env that truncates on its own -- a goal reached, a boundary
          crossed, a curriculum stage passed -- produces episodes shorter than
          the cap that are not crashes. The proxy publishes them as crashes.
        * A robot that goes unhealthy on the *last* step is reported by
          `TimeLimit` as `terminated=True` **and** `truncated=True`, with a
          length exactly equal to the cap. That is a crash, and the proxy
          scores it as a clean episode.

        The second one is live today, not hypothetical: `TimeLimit.step` sets
        `truncated` without clearing `terminated`, so the flags genuinely
        co-occur. It is rare, and a rate that is quietly wrong only rarely is
        the kind that survives into the record.
        """
        return sum(1 for t in self.terminated if t)

    @property
    def crash_rate(self) -> float:
        """`crashes` as a proportion of episodes rolled."""
        return self.crashes / len(self.terminated) if self.terminated else 0.0


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
    cap = env.spec.max_episode_steps if env.spec else None
    if not cap:
        raise ValueError(
            f"{env_id} declares no max_episode_steps, so an episode that stopped "
            f"short cannot be distinguished from one that ran to the horizon and "
            f"a crash rate cannot be defined. Register the env with one."
        )
    try:
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        returns: list[float] = []
        lengths: list[int] = []
        terminations: list[bool] = []
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

            # The assumption the crash rate used to rest on, now checked rather
            # than trusted. `lengths` is read as a crash proxy in this repo's
            # notes and episodes, and it is only a valid one while termination
            # is the sole way to stop short of the cap. An env that starts
            # truncating on its own would turn every such episode into a
            # phantom crash in a published ledger row, and nothing would say so.
            if not terminated and steps < cap:
                raise RuntimeError(
                    f"{env_id} episode {episode} ended at step {steps} of {cap} "
                    f"with terminated=False, truncated={truncated}: the env "
                    f"truncated on its own. crash_rate counts terminations and "
                    f"is still right, but episode LENGTH is no longer a crash "
                    f"proxy -- decide what this truncation means before any "
                    f"number derived from lengths enters the record."
                )

            returns.append(total)
            lengths.append(steps)
            terminations.append(bool(terminated))
        return Arm(
            "zero-action" if model is None else "greedy",
            returns, lengths, terminations, int(cap),
        )
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
            "crashes": trained.crashes, "crash_rate": trained.crash_rate,
        },
        "standing": asdict(standing) | {
            "mean": standing.mean, "std": standing.std,
            "worst": standing.worst, "best": standing.best,
            "crashes": standing.crashes, "crash_rate": standing.crash_rate,
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
        # Terminations, not short lengths -- see Arm.crashes.
        f"  greedy crashes           {t['crashes']} of {result['episodes']}"
        f"  (rate {t['crash_rate']:.4f}, cap {t['max_episode_steps']} steps)",
        f"  per-episode greedy       {[round(v, 1) for v in t['returns']]}",
        f"  per-episode lengths      {t['lengths']}",
        f"  per-episode terminated   {[int(x) for x in t['terminated']]}",
    ]
    return "\n".join(lines)


def _format_both(best: dict, latest: dict) -> str:
    """Both checkpoints side by side, plus what choosing between them is worth.

    This exists because `research/learnings/010` happened: two training seeds
    were compared through their `*_best.zip` files, the 91.55-point gap was
    read as a property of the seed, and on the unselected `*_sac.zip` the same
    instrument on the same 60 seeds gave -6.87 with the other seed ahead.
    `learnings/008` had already established that `*_best.zip` is an argmax over
    ONE-episode evaluations. Prose did not stop it, so the default output
    changed instead.
    """
    delta = latest["trained"]["mean"] - best["trained"]["mean"]
    return "\n".join([
        _format(best),
        "",
        _format(latest),
        "",
        "  SELECTION DELTA (latest - best)",
        f"    mean          {delta:+8.1f}",
        f"    ratio         {latest['ratio_mean'] - best['ratio_mean']:+8.3f}"
        f"   ({best['ratio_mean']:.3f} -> {latest['ratio_mean']:.3f})",
        f"    crashes       {best['trained']['crashes']} -> "
        f"{latest['trained']['crashes']} of {best['episodes']}",
        "  Quote BOTH or neither. *_best.zip is selected by argmax over "
        "one-episode",
        "  evaluations (learnings/008), so a comparison through it alone is a "
        "comparison",
        "  of two lucky draws (learnings/010).",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name under runs/")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed0", type=int, default=SEED0)
    parser.add_argument("--latest", action="store_true",
                        help="measure ONLY ant_sac.zip instead of both")
    # Both checkpoints are measured by DEFAULT, and opting out is explicit.
    # The cost is one extra pass; the alternative cost is a published
    # comparison between two argmax-selected artifacts, which is what
    # learnings/010 records happening.
    parser.add_argument("--best-only", action="store_true",
                        help="measure ONLY ant_sac_best.zip; you are asking "
                             "for a number learnings/008 says is a lucky draw")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.latest or args.best_only:
        result = compare(args.run, args.episodes, args.seed0, args.latest)
        print(json.dumps(result, indent=2) if args.json else _format(result))
        return 0

    best = compare(args.run, args.episodes, args.seed0, latest=False)
    latest = compare(args.run, args.episodes, args.seed0, latest=True)
    if args.json:
        print(json.dumps({"best": best, "latest": latest,
                          "selection_delta_mean":
                              latest["trained"]["mean"] - best["trained"]["mean"]},
                         indent=2))
    else:
        print(_format_both(best, latest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
