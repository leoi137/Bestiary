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

## The two stability fields, defined

`learnings/007` asked for these two and did not define them tightly, so both
definitions are pinned here and both are recorded on the row alongside the
window or sample that produced them.

**`mean_eval_after_converge`** — the mean of `eval/mean_reward` from step
400,000 onward, an absolute cutoff, identical for every run. Verified against
the record: it reproduces both numbers `learnings/007` published. See
`CONVERGE_AFTER_STEPS`, which shows what the fractional reading returns instead
and why that reading is biased in the wrong direction.

**`eval_crash_rate`** — the fraction of N deterministic episodes on the run's
**final** checkpoint that the env `terminated` early, i.e. the robot became
unhealthy. Two details do real work. It is counted from the env's `terminated`
flag rather than from `length < 1000`, because those differ (`greedy_eval`'s
`Arm.crashes` sets out how); and it is measured on `ant_sac.zip` rather than
`ant_sac_best.zip`, because the best checkpoint is selected partly for not
having crashed and so cannot honestly report a crash rate.

The oracle for both is `record/check_ledger_fields.py`.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from bestiary import paths
from bestiary.guards.eval_sampling import MIN_EVAL_EPISODES, N_FIELD
from bestiary.guards.ledger_schema import BASE_FIELDS, FIELDS_FROM_ROW_3, VERDICTS

# Episodes rolled to characterise the finished policy. 20, matching
# record/greedy_eval.py: these policies are bimodal and a crash rate is a
# proportion, so five episodes of a 26.7%-failure policy come back clean 21% of
# the time (research/scripts/learning_008_math.py).
EVAL_EPISODES = 20

# "After converge" means "from step 400,000 onward", an ABSOLUTE cutoff.
#
# This constant is checked against the record rather than chosen: it is the
# cutoff that reproduces both published numbers in `learnings/007`, which is
# where the field comes from. That learning's table reports "mean eval after
# 400k" as 887.5 for hound_desert_v0 (3.75M steps) and 1113.1 for
# hound_pd_desert_v0 (1.0M steps). Recomputed from the runs' own event files:
#
#     cutoff = 400,000 absolute -> 887.53 and 1113.14   <- matches the record
#     cutoff = 40% of run steps -> 1013.94 and 1113.14
#
# An earlier version of this module read the convention as a *fraction* (0.4),
# which is right only for the 1M-step run, where 40% happens to land on 400k.
# On the 3.75M-step torque run the fractional window starts at 1.5M and returns
# 1013.94 -- 126 points above the published 887.5. That error is not neutral:
# it discards the run's early, more frequent crashes and so flatters precisely
# the unstable policy that `learnings/007` exists to stop the ledger from
# rewarding. A fraction also makes the window a different absolute region of
# training for every run length, which is the comparability failure that same
# learning names in "never compare peaks across runs of different length".
#
# So: absolute, identical for every row, and recorded ON the row as
# `mean_eval_after_converge_from_step` so a later reader never has to guess
# which window produced the number.
#
# It is a convention, not a convergence test. Nothing detects convergence here;
# 400k is where the record drew the line, and the honest thing is to draw it in
# the same place every time and say where it is.
CONVERGE_AFTER_STEPS = 400_000

# Below this, a mean over the converged window is a mean over too few draws to
# be a summary of anything. `eval/mean_reward` is a ONE-episode draw
# (learnings/008), and these policies are bimodal, so a handful of points can
# miss the failure mode entirely -- the same argument as EVAL_EPISODES above.
MIN_CONVERGED_EVAL_POINTS = 5

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


def after_converge(evals: list, after_step: int = CONVERGE_AFTER_STEPS) -> list[float]:
    """The `eval/mean_reward` values from `after_step` onward.

    Split out and named so the one judgement call in this module -- where the
    converged window starts -- is a function with an oracle rather than three
    lines buried in `summarize`. See `CONVERGE_AFTER_STEPS`.

    Each value is a single-episode draw (learnings/008). That is fine here and
    fatal for a maximum: averaging many one-episode draws is an unbiased
    estimate of the policy's mixture, crashes included, whereas taking the max
    over them estimates only its best mode. This field exists to be the honest
    counterpart to `best_eval_return`, so the small sample per point is the
    point, not a defect -- provided there are enough points.
    """
    values = [e.value for e in evals if e.step >= after_step]
    if len(values) < MIN_CONVERGED_EVAL_POINTS:
        raise ValueError(
            f"only {len(values)} eval point(s) at or after step {after_step:,} "
            f"(need {MIN_CONVERGED_EVAL_POINTS}); the run is too short, or "
            f"evaluated too rarely, to summarise honestly. Last eval step was "
            f"{int(evals[-1].step) if evals else 'n/a'}."
        )
    return values


def summarize(run: str, episodes: int = EVAL_EPISODES,
              converge_after_steps: int = CONVERGE_AFTER_STEPS) -> dict:
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

    after = after_converge(evals, converge_after_steps)

    # The finished policy, measured the way learnings/008 requires: N
    # deterministic episodes, reported as a mean with its n, never as a peak.
    best = greedy_eval.compare(run, episodes=episodes)
    latest = greedy_eval.compare(run, episodes=episodes, latest=True)

    # `eval_crash_rate` describes the policy the run ENDED with, so it is
    # measured on ant_sac.zip (`latest`), not on ant_sac_best.zip. Measuring it
    # on the best checkpoint would be circular: learnings/008 shows that
    # checkpoint is selected as the argmax of single-episode draws, so it is
    # chosen partly *for* not having crashed. Its crash rate is biased low by
    # construction, and a reliability number that flatters unreliable policies
    # is the exact defect learnings/007 was written about.
    #
    # Not recomputed here -- `greedy_eval.Arm.crashes` owns the definition of a
    # crash, and counts terminations rather than short episodes.
    crash_rate = latest["trained"]["crash_rate"]

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
        # The window and the sample size that produced the number above, on the
        # row itself. Same reason eval-sampling makes `best_eval_return` carry
        # `best_eval_episodes`: a summary statistic whose window is implicit
        # cannot be compared against anything later.
        "mean_eval_after_converge_from_step": converge_after_steps,
        "mean_eval_after_converge_points": len(after),
        "mean_eval_after_converge_sd": round(float(np.std(after, ddof=1)), 2)
                                       if len(after) > 1 else 0.0,

        "eval_crash_rate": round(float(crash_rate), 4),
        "eval_crash_rate_episodes": episodes,
        "eval_crash_rate_checkpoint": "ant_sac.zip",
        # ...and WHICH ant_sac.zip. A bare filename is what put a number into
        # the record naming a checkpoint that was overwritten nine minutes
        # later (anomalies 19/20/23/27, learnings/013). compare() freezes both
        # checkpoints to content-addressed paths, so the identity is already in
        # hand here -- discarding it and writing only the filename would
        # reproduce the original defect in the append-only record itself.
        "eval_crash_rate_checkpoint_sha256": latest.get("checkpoint_sha256"),
        "best_eval_checkpoint_sha256": best.get("checkpoint_sha256"),

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
        "latest_crashes": latest["trained"]["crashes"],
        "best_ckpt_crash_rate": round(float(best["trained"]["crash_rate"]), 4),
        "standing_mean": round(float(best["standing"]["mean"]), 2),
        "ratio_vs_standing": round(float(best["ratio_mean"]), 3),
        "max_episode_steps": latest["trained"]["max_episode_steps"],
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

    # Refuse a row the eval-sampling guard would reject, for the same reason
    # the schema fields are checked above -- but this one is urgent rather than
    # tidy. `guards --fast` gates EVERY training launch, and it reads the whole
    # ledger, so appending one under-sampled row does not just record a weak
    # number: it turns the launch gate red permanently, and the ledger is
    # append-only, so the row cannot be taken back out. Catching it here costs
    # two lines; catching it afterwards costs the ability to train.
    n = row.get(N_FIELD)
    if not isinstance(n, int) or n < MIN_EVAL_EPISODES:
        raise ValueError(
            f"{N_FIELD}={n!r}: a ledger row needs at least {MIN_EVAL_EPISODES} "
            f"evaluation episodes (learnings/008 -- a crash rate over fewer than "
            f"that is not a rate). Re-run without a reduced --episodes."
        )
    rate = row["eval_crash_rate"]
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"eval_crash_rate={rate!r} is not a proportion in [0, 1]")

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
