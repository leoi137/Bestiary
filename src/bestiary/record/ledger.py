"""Write a finished run into the ledger, with numbers code computed.

The ledger is the one artifact every later argument is built on, and until now
nothing could write a row. `ledger-schema` requires `mean_eval_after_converge`,
`eval_crash_rate`, `seeds` and `provisional` from row 3 onward, and a grep for
the first two found them *only inside the guard demanding them*. So the next
finished run would have needed four numbers no code produced, discovered after
the GPU time was already spent, with the number rule forbidding the obvious
escape of working them out in prose.

This closes that. Everything here is read from the run's own TensorBoard event
files or measured by rolling its checkpoints — nothing is estimated, and the
module refuses to write a row it cannot fill.

    venv/bin/python -m bestiary.record.ledger --run hound_v2 --verdict improved \\
        --notes "..." --dry-run
    venv/bin/python -m bestiary.record.ledger --run hound_v2 --verdict improved \\
        --notes "..." --append

## One field changes meaning at row 3, deliberately

`best_eval_return` in rows 1-2 is the maximum of `eval/mean_reward` over
training. `learnings/008` established that this is a maximum over **one-episode**
draws: `VideoEvalCallback` rolls a single episode per evaluation and saves
`ant_sac_best.zip` whenever that draw beats the record, so at 14 evaluations of
a policy that fails 26.7% of episodes, the probability the saved checkpoint is a
good-mode snapshot is 1 - 0.267**14, i.e. one. It is not a policy score, it
cannot rank two runs, and it grows with the number of evaluations.

From row 3 `best_eval_return` is instead the **mean of N deterministic episodes**
on the best checkpoint, with `best_eval_episodes` recording N — which is what the
`eval-sampling` guard requires. The old-style number is preserved under
`peak_single_episode_eval` so nothing is lost and the two are never confused.

A field quietly changing meaning is exactly the drift this project exists to
prevent, so it is stated here, in the row's own notes, and enforced by a guard
rather than remembered.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from bestiary import paths
from bestiary.guards.ledger_schema import BASE_FIELDS, FIELDS_FROM_ROW_3, VERDICTS

# Episodes rolled to characterise the finished policy. 20, matching
# record/greedy_eval.py: these policies are bimodal and a crash rate is a
# proportion, so five episodes of a 26.7%-failure policy come back clean 21% of
# the time (research/scripts/learning_008_math.py).
EVAL_EPISODES = 20

# "After converge" means the last 60% of training. Taken from ledger row 2,
# which reported "mean eval after 400k" on a 1M-step run. Not a principled
# convergence test -- it is the convention already in the record, made explicit
# and applied identically to every row so the numbers stay comparable.
CONVERGE_FRACTION = 0.4

# env_id prefix -> robot, so the field is derived rather than typed by hand.
ROBOTS = (
    ("HoundPD", "hound"),
    ("Hound", "hound"),
    ("Spyder", "spyder"),
    ("Ant", "ant"),
    ("Humanoid", "humanoid"),
    ("Walker", "walker"),
)


def _robot(env_id: str) -> str:
    for prefix, name in ROBOTS:
        if env_id.startswith(prefix):
            return name
    raise ValueError(
        f"cannot derive a robot from env_id {env_id!r}; add it to ROBOTS "
        f"rather than guessing at write time"
    )


def _scalars(run_dir: Path) -> dict[str, list]:
    """Every scalar series in the run, newest sub-run wins on conflict."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_dirs = sorted({p.parent for p in run_dir.rglob("events.out.tfevents.*")})
    if not event_dirs:
        raise FileNotFoundError(f"no TensorBoard event files under {run_dir}")

    merged: dict[str, list] = {}
    for d in event_dirs:
        acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            merged.setdefault(tag, []).extend(acc.Scalars(tag))
    for tag in merged:
        merged[tag].sort(key=lambda e: e.step)
    return merged


def _require(scalars: dict[str, list], tag: str) -> list:
    if tag not in scalars or not scalars[tag]:
        raise KeyError(
            f"the run logged no {tag!r}; a ledger row cannot be filled from it. "
            f"Available: {sorted(scalars)}"
        )
    return scalars[tag]


def summarize(run: str, episodes: int = EVAL_EPISODES,
              converge_fraction: float = CONVERGE_FRACTION) -> dict:
    """Every ledger field for a finished run, computed from its own artifacts."""
    from bestiary.record import greedy_eval

    run_dir = paths.RUNS / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run directory at {run_dir}")

    config = json.loads((run_dir / "config.json").read_text())
    env_id = str(config["env_id"])

    scalars = _scalars(run_dir)
    rollout_rew = _require(scalars, "rollout/ep_rew_mean")
    rollout_len = _require(scalars, "rollout/ep_len_mean")
    evals = _require(scalars, "eval/mean_reward")

    steps = int(rollout_rew[-1].step)
    wall_s = int(round(rollout_rew[-1].wall_time - rollout_rew[0].wall_time))

    # Measured, not read off time/fps: that tag is SB3's instantaneous rate and
    # a run stopped and resumed would report the last window rather than the run.
    fps = int(round(steps / wall_s)) if wall_s else 0

    cutoff = converge_fraction * steps
    after = [e.value for e in evals if e.step >= cutoff]
    if not after:
        raise ValueError(
            f"no eval points after step {cutoff:.0f} ({converge_fraction:.0%} of "
            f"{steps}); the run is too short to summarise honestly"
        )

    # The finished policy, measured the way learnings/008 requires: N
    # deterministic episodes, reported as a mean with its n, never as a peak.
    best = greedy_eval.compare(run, episodes=episodes)
    latest = greedy_eval.compare(run, episodes=episodes, latest=True)

    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids

    max_steps = gym.spec(env_id).max_episode_steps
    if not max_steps:
        raise ValueError(f"{env_id} declares no max_episode_steps; cannot define a crash")
    crashes = sum(1 for n in latest["trained"]["lengths"] if n < max_steps)

    seeds = 1  # one run, one seed. A multi-seed row is written by hand from several.
    row = {
        "run": run,
        "date": date.today().isoformat(),
        "robot": _robot(env_id),
        "env_id": env_id,
        "algo": str(config.get("algo", "SAC")),
        "wrapper": config.get("wrapper"),
        "seed": config.get("seed"),
        "steps": steps,
        "wall_clock_s": wall_s,
        "fps": fps,

        # See the module docstring: this is a MEAN over `best_eval_episodes`
        # deterministic episodes, not the old max-over-single-draws.
        "best_eval_return": round(float(best["trained"]["mean"]), 2),
        "best_eval_episodes": episodes,
        "best_eval_sd": round(float(best["trained"]["std"]), 2),
        "peak_single_episode_eval": round(max(e.value for e in evals), 2),

        "final_ep_rew_mean": round(float(rollout_rew[-1].value), 2),
        "final_ep_len_mean": round(float(rollout_len[-1].value), 2),
        "final_ent_coef": (round(float(scalars["train/ent_coef"][-1].value), 4)
                           if "train/ent_coef" in scalars else None),

        "mean_eval_after_converge": round(float(np.mean(after)), 2),
        "eval_crash_rate": round(crashes / episodes, 4),
        "seeds": seeds,
        "provisional": seeds < 3,

        "obs_spec_hash": (config.get("obs_spec") or {}).get("hash"),

        # What this row's numbers were paid for. `reward_shape_hash` is the one
        # that decides whether two rows may be compared at all: it is invariant
        # to retuning and moves the moment the reward starts paying for a
        # different thing. Rows 1-2 carry null for both, which is the honest
        # value -- nothing recorded their reward, and back-filling it from
        # today's code would invent provenance (see envs/reward_spec.py).
        "reward_spec_hash": (config.get("reward_spec") or {}).get("hash"),
        "reward_shape_hash": (config.get("reward_spec") or {}).get("shape_hash"),

        "verdict": None,   # filled by the caller — a judgement, not a measurement
        "notes": "",
    }
    row["_measured"] = {
        "latest_mean": round(float(latest["trained"]["mean"]), 2),
        "latest_sd": round(float(latest["trained"]["std"]), 2),
        "standing_mean": round(float(best["standing"]["mean"]), 2),
        "ratio_vs_standing": round(float(best["ratio_mean"]), 3),
        "eval_points_after_converge": len(after),
        "max_episode_steps": max_steps,
    }
    return row


def validate(row: dict) -> None:
    """Refuse to write a row the schema guard would reject. Loud, and early."""
    missing = [f for f in (*BASE_FIELDS, *FIELDS_FROM_ROW_3) if f not in row]
    if missing:
        raise ValueError(f"row is missing required fields: {missing}")
    empty = [f for f in (*BASE_FIELDS, *FIELDS_FROM_ROW_3)
             if row[f] is None and f not in ("wrapper", "seed", "final_ent_coef")]
    if empty:
        raise ValueError(f"row has unset required fields: {empty}")
    if row["verdict"] not in VERDICTS:
        raise ValueError(f"verdict {row['verdict']!r} not in {sorted(VERDICTS)}")
    if row["seeds"] == 1 and row["provisional"] is not True:
        raise ValueError("a single-seed row must be marked provisional (the seed rule)")

    existing = {json.loads(ln)["run"]
                for ln in paths.LEDGER.read_text().splitlines() if ln.strip()}
    if row["run"] in existing:
        raise ValueError(
            f"{row['run']!r} already has a ledger row. The ledger is append-only "
            f"and run names are unique -- correct the record with a new row and a "
            f"note, never by rewriting this one."
        )


def append(row: dict) -> None:
    validate(row)
    payload = {k: v for k, v in row.items() if not k.startswith("_")}
    with paths.LEDGER.open("a") as fh:          # append-only, never rewrite
        fh.write(json.dumps(payload) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--verdict", choices=sorted(VERDICTS),
                        help="required with --append; a judgement, not a measurement")
    parser.add_argument("--notes", default="")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--append", action="store_true",
                        help="write the row; without this, print it and stop")
    args = parser.parse_args()

    row = summarize(args.run, episodes=args.episodes)
    row["verdict"] = args.verdict
    row["notes"] = args.notes

    if not args.append:
        print(json.dumps(row, indent=2))
        print("\n-- dry run; pass --append to write, and --verdict to set one --")
        return 0

    if not args.verdict:
        raise SystemExit("--verdict is required with --append")
    append(row)
    print(f"appended {args.run} to {paths.LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
