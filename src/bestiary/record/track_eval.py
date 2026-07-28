"""Measure a command-tracking policy against zero action, on a stated command grid.

`record/greedy_eval.py` is the instrument for the OLD reward and it cannot
measure this one. Two reasons, and the second is the one that matters:

1. It has no notion of a command. It rolls episodes out and sums the reward,
   which under `HoundPDTrackDesert-v0` means summing over whatever commands the
   sampler happened to draw.
2. **"Does the policy beat zero action?" is undefined over a mixture containing
   STOP.** Zero action is near-optimal on STOP *by design* -- it scores 909.35
   there against a theoretical 1000 -- so a policy that never moves collects a
   tenth of the mixture's value legitimately, and a ratio computed over the
   mixture silently rewards standing again. The exploit metric is dead only
   relative to a stated protocol.

So the protocol is the instrument. `docs/theory/command-tracking-reward.md`
section 3 fixes a grid of commands, requires both arms be run on identical
command sequences paired by seed, and requires the ratio be reported
**excluding the (0,0,0) cell**, which is reported separately as a
stop-competence check where the policy should MATCH zero action rather than
beat it. Publish the grid with the number, always.

    venv/bin/python -m bestiary.record.track_eval --run hound_track_desert_s0
    venv/bin/python -m bestiary.record.track_eval --run hound_track_desert_s0 \
        --episodes 20 --json

With no `--run`, it measures the zero-action arm alone -- which is how the
committed denominator in `research/measurements/tracking_baseline_zero_action.json`
was produced.

WHAT TO READ, IN ORDER

`drive_grid_ratio` is the headline: the policy's mean over the six driving
cells divided by zero action's. Section 5 predicts >= 5x for a successful run,
against 1.128x for the reward this replaces.

`mean_track` per cell is the unitless one, E[Phi_v * Phi_w] in [0,1]. It
survives cost-coefficient retuning, so it is the number that will still mean
something after the contact cost is fixed.

⚠️ **`mean_track` does NOT survive episode-length changes, and this docstring
claimed it did until 2026-07-28.** It is a mean over episodes of each episode's
own per-step mean, so every episode carries equal weight regardless of how long
it lived. That is the RATE a policy tracks at *while it is still up*, and it is
a real quantity — but it is not what the reward pays, because the reward is a
per-step integral over a fixed horizon. A policy that crashes at step 300 at
rate 0.30 and one that survives 1000 steps at rate 0.30 read identically here
and bank income in a 1:3.33 ratio. Three numbers are now reported and they
answer three different questions:

| field | question it answers | length-biased? |
|---|---|---|
| `mean_track` | while up, how well does it track? | yes, by design |
| `mean_track_stepw` | per step actually taken, how well? | no |
| `track_per_horizon` | how much did it BANK, per unit of horizon? | no — this is the one comparable to `mean` |

`track_per_horizon` is the tracking analogue of the return: total Phi integral
divided by the env's own `max_episode_steps`, so a crash costs it exactly the
steps it forfeited. **When `mean_track` and `track_per_horizon` disagree in
direction, the disagreement IS the finding** — it says the arms differ in
survival, not in competence, and the ledger's return-based numbers are tracking
survival.

`command_gain` is the cheapest lie-detector in the whole design. It regresses
achieved forward velocity on commanded forward velocity across the drive
cells. A slope near 1 is tracking; a slope near 0 means the policy is ignoring
the three command slots entirely and has found some command-independent gait
that the mixture happens to pay for. A policy can post a respectable return
with a dead command input, and this is the only number here that notices.

DO NOT READ `drive_grid_ratio` WHEN EITHER ARM IS NEGATIVE

Returns here are not bounded below by zero: the costs are paid every step and
the tracking term can be ~0, so zero action itself scores **-4.72 and -4.69**
on the two (0.5, 0, +-0.4) cells. A ratio of two negative numbers is not a
performance ordering, and a ratio that crosses zero is not even monotone.
Measured against the live run at 82k steps -- 5% trained, crashing 21 of 21
episodes, unambiguously worse than doing nothing -- this reported a *ratio of
3.328* on the (0.5, 0, 0.4) cell, because both arms were negative and the
policy's was less so.

`drive_grid_track` does not have this failure: it is E[Phi_v * Phi_w], bounded
in [0, 1] by construction, and it read 0.0115 against zero action's 0.0652 on
that same measurement -- correctly saying "far worse". Section 5 already
recommends it as the metric that survives cost-coefficient retuning; it also
happens to be the only one of the two that is safe to quote unconditionally.

So: read the track score first, and quote the ratio only once both arms are
comfortably positive. Both are reported because the section 5 prediction
(">= 5x") is stated as a ratio and must be resolvable in its own terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bestiary import paths

# Section 3's eval grid. The stop cell is LAST and is excluded from the
# headline ratio.
EVAL_GRID: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.3, 0.0, 0.0),
    (0.5, 0.0, 0.4),
    (0.5, 0.0, -0.4),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 0.0),      # stop-competence cell, reported separately
)
STOP_CELL = (0.0, 0.0, 0.0)
TRACK_ENV = "HoundPDTrackDesert-v0"

# Every env this protocol is defined for. The grid, the paired seeds and the
# excluded stop cell are properties of the COMMAND SPACE, which these envs
# share; only the reward that scores a cell differs. So the protocol transfers
# and the numbers do not — a `drive_grid_mean` is meaningful only against
# another one measured under the same reward spec, which is why `--env` is
# recorded in the JSON beside every result rather than assumed by the reader.
TRACK_ENVS = ("HoundPDTrackDesert-v0", "HoundPDTrackRelDesert-v0")


TERMS = ("reward_track", "reward_ctrl", "reward_contact", "reward_termination")


def resolve_horizon(env) -> int:
    """The episode horizon `track_per_horizon` divides by. Raises if there is none.

    Asked of the env, never assumed: a hardcoded 1000 would silently mis-scale
    the moment a TimeLimit moves.

    It RAISES rather than falling back to the episode's own length, which is
    what an earlier version did. That fallback is worse than a wrong number --
    it silently degenerates `track_per_horizon` into `mean_track_stepw`, leaves
    every assertion in `guards/track_length_bias.py` green, and records nothing
    in the emitted JSON about which of the two a reader is looking at. "Per
    horizon" is undefined without a horizon, so the honest move is to refuse.

    Separate from `rollout` so the guard can assert this behaviour without
    having to build a whole env to do it.
    """
    horizon = getattr(getattr(env, "spec", None), "max_episode_steps", None)
    if not horizon:
        raise ValueError(
            f"{getattr(getattr(env, 'spec', None), 'id', env)!r} reports "
            f"max_episode_steps={horizon!r}, so track_per_horizon has no "
            f"denominator. This protocol is defined for fixed-horizon episodes; "
            f"an env without a TimeLimit needs a stated horizon, not a guess."
        )
    return int(horizon)


def rollout(env, seed: int, policy=None, forced_cmd=None, *,
            deterministic: bool = True, action_seed: int | None = None,
            on_step=None) -> dict:
    """One episode. `policy=None` means zero action; `forced_cmd` pins the command.

    Deterministic on both arms by default: zero action is deterministic by
    definition and the policy is queried with `deterministic=True`, so the two
    are compared like with like. `learnings/007` is the record of what happens
    when a noisy statistic is compared against a clean one.

    `deterministic=False` samples from the actor's squashed Gaussian instead --
    the policy SAC actually optimised, entropy bonus included. It is a strictly
    noisier statistic, so it is only meaningful with `action_seed` set (the
    episode is then reproducible) and reported with its spread. The control cost
    is quadratic, so a sampled action pays E[sum a^2] = sum (E a)^2 + sum Var(a)
    -- strictly more than its own MEAN action. It is not strictly more than the
    DETERMINISTIC action, because tanh(mu) is the median of a squashed Gaussian
    and not its mean; the two effects are measured separately in
    `research/scripts/deterministic_vs_stochastic.py`.

    `on_step(step_index, obs_before, action, info)` is an optional observer, so a
    caller that needs the visited states or per-step terms does not have to
    reimplement the protocol.
    """
    if action_seed is not None:
        import torch  # local: only the stochastic arm needs a seeded sampler
        torch.manual_seed(action_seed)

    obs, _ = env.reset(seed=seed)
    if forced_cmd is not None:
        env.unwrapped._cmd = np.array(forced_cmd, dtype=float)
        # Push the resample past the horizon so the command is genuinely held
        # for the whole episode, then refresh the observation so the policy is
        # actually SHOWN the command it is being scored against.
        env.unwrapped._steps_until_resample = 10**9
        obs = env.unwrapped._get_obs()

    zero = np.zeros(env.action_space.shape[0])
    total = 0.0
    term_sums = dict.fromkeys(TERMS, 0.0)
    phi_v, phi_w, achieved, commanded = [], [], [], []
    steps, terminated = 0, False
    while True:
        if policy is None:
            action = zero
        else:
            action = policy.predict(obs, deterministic=deterministic)[0]
        obs_before = obs
        obs, r, term, trunc, info = env.step(action)
        total += r
        for t in TERMS:
            term_sums[t] += float(info[t])
        phi_v.append(info["track_phi_v"])
        phi_w.append(info["track_phi_w"])
        achieved.append(info["achieved_vx"])
        commanded.append(info["cmd_vx"])
        if on_step is not None:
            on_step(steps, obs_before, action, info)
        steps += 1
        if term or trunc:
            terminated = term
            break
    horizon = resolve_horizon(env)
    # The banked integral is taken from the env's OWN paid term, not recomputed
    # as phi_v*phi_w. They differ on exactly one step: the env pays
    # `phi_v*phi_w if healthy else 0.0`, so on a crashing episode the terminal
    # unhealthy step earns nothing while the naive product still scores it.
    # Sourcing it here makes track_per_horizon exactly the reward's tracking
    # integral rather than an approximation good to ~4 significant figures.
    track = np.asarray(phi_v) * np.asarray(phi_w)
    return {
        "return": total,
        "steps": steps,
        "horizon": int(horizon),
        "terminated": terminated,
        "mean_phi_v": float(np.mean(phi_v)),
        "mean_phi_w": float(np.mean(phi_w)),
        "mean_track": float(track.mean()),
        # The un-normalised integral, taken from the env's own paid term. Every
        # length-aware aggregate downstream is built from this and a step count,
        # never from a mean of means.
        "track_income": float(term_sums["reward_track"]),
        "achieved_vx": float(np.mean(achieved)),
        "commanded_vx": float(np.mean(commanded)),
        **term_sums,
    }


def aggregate_cell(cell, eps: list[dict]) -> dict:
    """Aggregate one grid cell's episodes. Pure — no env, no physics, no policy.

    Split out of `_arm` so `guards/track_length_bias.py` can assert the real
    aggregation rather than a private reimplementation of it. A guard that
    recomputes the arithmetic it is checking bounds nothing; `tracking-frame`
    learned that about constants and it is just as true about formulas.
    """
    rets = np.array([e["return"] for e in eps])
    steps = np.array([float(e["steps"]) for e in eps])
    return {
        "command": list(cell),
        "mean": float(rets.mean()),
        "sd": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "mean_track": float(np.mean([e["mean_track"] for e in eps])),
        "mean_phi_v": float(np.mean([e["mean_phi_v"] for e in eps])),
        "mean_phi_w": float(np.mean([e["mean_phi_w"] for e in eps])),
        "achieved_vx": float(np.mean([e["achieved_vx"] for e in eps])),
        "commanded_vx": float(cell[0]),
        "crashes": int(sum(e["terminated"] for e in eps)),
        "episodes": len(eps),
        # anomalies.jsonl row 20: mean_track is a per-step mean over
        # episodes of unequal length. Report the lengths next to it so the
        # bias is visible rather than latent.
        "mean_steps": float(steps.mean()),
        "sd_steps": float(steps.std(ddof=1)) if len(eps) > 1 else 0.0,
        # The divisor, written down. Without it a reader cannot check
        # track_per_horizon against mean_steps and mean_track_stepw, and has to
        # trust that the code found the right TimeLimit.
        "horizon": int(eps[0]["horizon"]),
        # The two length-aware aggregates. Both are built from the summed
        # integral and a step count -- NEVER from a mean of per-episode
        # means, which is the bias being corrected here.
        "mean_track_stepw": float(
            sum(e["track_income"] for e in eps) / steps.sum()),
        "track_per_horizon": float(
            np.mean([e["track_income"] / e["horizon"] for e in eps])),
        **{t: float(np.mean([e[t] for e in eps])) for t in TERMS},
    }


def _arm(env, policy, episodes: int, seed0: int, *,
         deterministic: bool = True, action_seed0: int | None = None,
         on_step=None) -> dict:
    """Run one arm over the whole grid. Both arms get the same seeds.

    `action_seed0` seeds the action sampler per episode (cell index and episode
    index both enter it), so a `deterministic=False` arm is reproducible.
    `on_step` is forwarded to `rollout`, so a caller can keep the per-step trace
    of the very same rollouts these aggregates are computed from.
    """
    cells = {}
    for ci, cell in enumerate(EVAL_GRID):
        eps = [rollout(env, seed=seed0 + i, policy=policy, forced_cmd=cell,
                       deterministic=deterministic,
                       action_seed=(None if action_seed0 is None
                                    else action_seed0 + 1000 * ci + i),
                       on_step=on_step)
               for i in range(episodes)]
        cells[str(cell)] = aggregate_cell(cell, eps)
    drive = [c for c in cells.values() if tuple(c["command"]) != STOP_CELL]
    # Command gain: OLS slope of achieved on commanded forward velocity across
    # the drive cells. Failure mode 3 -- a dead command input reads ~0.
    x = np.array([c["commanded_vx"] for c in drive])
    y = np.array([c["achieved_vx"] for c in drive])
    slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else float("nan")
    return {
        "cells": cells,
        "drive_grid_mean": float(np.mean([c["mean"] for c in drive])),
        "drive_grid_track": float(np.mean([c["mean_track"] for c in drive])),
        "drive_grid_track_stepw": float(
            np.mean([c["mean_track_stepw"] for c in drive])),
        "drive_grid_track_per_horizon": float(
            np.mean([c["track_per_horizon"] for c in drive])),
        "drive_grid_steps": float(np.mean([c["mean_steps"] for c in drive])),
        "stop_cell_mean": cells[str(STOP_CELL)]["mean"],
        "command_gain": slope,
        "crashes": int(sum(c["crashes"] for c in cells.values())),
        "deterministic": deterministic,
        **{f"drive_grid_{t}": float(np.mean([c[t] for c in drive]))
           for t in TERMS},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="run name under runs/; omit to measure zero action alone")
    ap.add_argument("--episodes", type=int, default=20, help="episodes PER GRID CELL")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--latest", action="store_true",
                    help="use ant_sac.zip instead of ant_sac_best.zip")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the actor instead of taking its mean action; "
                         "noisier, so --action-seed pins it and the spread is "
                         "what to read")
    ap.add_argument("--action-seed", type=int, default=7000,
                    help="seed for the action sampler under --stochastic")
    ap.add_argument("--env", default=TRACK_ENV, choices=TRACK_ENVS,
                    help="which command-tracking env to score under; the "
                         "default is the original one, so every number already "
                         "in the record keeps its meaning")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    track_env = args.env

    import gymnasium as gym

    import bestiary.envs  # noqa: F401

    env = gym.make(track_env)

    policy, checkpoint, frozen = None, None, None
    if args.run:
        from stable_baselines3 import SAC

        from bestiary.record.freeze import freeze_checkpoint
        run_dir: Path = paths.RUNS / args.run
        checkpoint = "ant_sac.zip" if args.latest else "ant_sac_best.zip"
        ckpt = run_dir / checkpoint
        if not ckpt.exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        # Freeze BEFORE loading, and load from the frozen copy. ant_sac_best.zip
        # is rewritten in place mid-run, so a number measured off that filename
        # names a file that may not exist by the time anyone checks it. This has
        # already cost the record a published conclusion -- anomalies 19/20/27,
        # learnings/013. freeze.py's docstring has the whole argument.
        frozen = freeze_checkpoint(ckpt, run_dir=run_dir)
        # Refuse to compare across a moved reward. The whole point of the
        # spec hash is that a number measured under one objective must not be
        # quoted against another.
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("env_id") != track_env:
            raise SystemExit(
                f"{args.run} trained on {cfg.get('env_id')}, not {track_env}. "
                "Its policy was trained with zero-filled command slots, so "
                "evaluating it under nonzero commands feeds it observations off "
                "its training manifold -- undefined behaviour, not a baseline. "
                "See docs/theory/command-tracking-reward.md section 5."
            )
        policy = SAC.load(frozen.frozen, device="cpu")

    zero = _arm(env, None, args.episodes, args.seed0)
    result = {
        "env": track_env,
        "episodes_per_cell": args.episodes,
        "seed0": args.seed0,
        "deterministic": not args.stochastic,
        "action_seed": args.action_seed if args.stochastic else None,
        "zero_action": zero,
    }
    if policy is not None:
        trained = _arm(env, policy, args.episodes, args.seed0,
                       deterministic=not args.stochastic,
                       action_seed0=args.action_seed if args.stochastic else None)
        result["run"] = args.run
        result.update(frozen.as_json_fields())
        result["trained"] = trained
        result["drive_grid_ratio"] = (
            trained["drive_grid_mean"] / zero["drive_grid_mean"]
            if zero["drive_grid_mean"] else float("nan")
        )
        result["stop_cell_ratio"] = (
            trained["stop_cell_mean"] / zero["stop_cell_mean"]
            if zero["stop_cell_mean"] else float("nan")
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"{track_env}   {args.episodes} episodes per cell, seeds "
          f"{args.seed0}-{args.seed0 + args.episodes - 1}, both arms identical")
    if args.stochastic:
        print(f"  policy arm SAMPLED (deterministic=False), "
              f"action seed {args.action_seed}")
    header = f"  {'command':>18}  {'zero':>9}"
    if policy is not None:
        header += f"  {'policy':>9}  {'ratio':>7}  {'track':>7}"
    print(header)
    for k in zero["cells"]:
        z = zero["cells"][k]
        line = f"  {k:>18}  {z['mean']:9.2f}"
        if policy is not None:
            t = result["trained"]["cells"][k]
            ratio = t["mean"] / z["mean"] if z["mean"] else float("nan")
            line += f"  {t['mean']:9.2f}  {ratio:7.3f}  {t['mean_track']:7.4f}"
        if tuple(z["command"]) == STOP_CELL:
            line += "   <- stop cell, EXCLUDED from the headline"
        print(line)

    print(f"\n  zero-action drive-grid mean   {zero['drive_grid_mean']:9.2f}")
    if policy is not None:
        t = result["trained"]
        print(f"  policy     drive-grid mean   {t['drive_grid_mean']:9.2f}")
        print(f"  DRIVE-GRID RATIO             {result['drive_grid_ratio']:9.3f}"
              "    (section 5 predicts >= 5x for a successful run)")
        print(f"  drive-grid track score       {t['drive_grid_track']:9.4f}"
              f"    (zero action {zero['drive_grid_track']:.4f}, max 1.0)")
        print(f"  command gain (slope)         {t['command_gain']:9.3f}"
              "    (~1 tracks, ~0 = dead command input)")
        print(f"  stop cell                    {t['stop_cell_mean']:9.2f}"
              f"    (zero action {zero['stop_cell_mean']:.2f} — MATCH, do not beat)")
        print(f"  crashes                      {t['crashes']:9d}"
              f" of {args.episodes * len(EVAL_GRID)}")


if __name__ == "__main__":
    main()
