"""Train SAC on Ant-v5 inside a per-run directory.

Each invocation lives under `runs/<run_name>/` so different reward-shaping
experiments don't clobber each other. The directory holds the model, the
replay buffer, the TensorBoard logs, eval videos, and a config.json
recording what produced it.

Examples:
    # Baseline (no wrapper, default reward), 750k steps:
    python -m bestiary.train.train --run-name baseline_seed0 --seed 0

    # Foot-contact shaping experiment from scratch:
    python -m bestiary.train.train --run-name foot_contact_v1 \\
                    --wrapper foot_contact \\
                    --wrapper-kwargs '{"penalty": 1.0, "window": 50}' \\
                    --seed 0 --steps 1_500_000

    # Resume the same run (auto-detected by re-using the same --run-name):
    python -m bestiary.train.train --run-name foot_contact_v1 --steps 500_000

Re-invoking with the same --run-name resumes from disk; a new --run-name
starts fresh. The wrapper/seed args are only consulted on the *first*
invocation (when config.json is written); on resume they're loaded from
config.json so the run stays internally consistent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from bestiary import paths

import bestiary.envs  # noqa: F401 -- registers Spyder-v0 with Gymnasium
from bestiary.rewards import WRAPPERS
from bestiary.terrain import TerrainSpec

DEFAULT_ENV = "Ant-v5"

# `config.get("terrain_spec")` returns None both for a key that is absent and
# for one recorded as null, and those are different facts: "nothing is known
# about this run's ground" versus "this run is verified to be on a flat world".
# A sentinel is the cheapest way to keep them apart. See
# `_record_or_verify_terrain_spec`.
_UNRECORDED = object()

# --- Hyperparameters (the spec) --------------------------------------------
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 256
DEVICE = "cuda"


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "model": run_dir / "ant_sac.zip",
        "buffer": run_dir / "ant_buffer.pkl",
        "best_model": run_dir / "ant_sac_best.zip",
        "best_reward": run_dir / "ant_sac_best.txt",
        "tb": run_dir / "ant_tb",
        "videos": run_dir / "videos",
        "config": run_dir / "config.json",
    }


def _make_env(env_id: str, wrapper_name: str | None, wrapper_kwargs: dict[str, Any],
              seed: int | None, render_mode: str | None = None) -> gym.Env:
    """Build a MuJoCo env by id, optionally wrapped, optionally seeded."""
    env = gym.make(env_id, render_mode=render_mode)
    if wrapper_name is not None:
        if wrapper_name not in WRAPPERS:
            raise ValueError(f"Unknown wrapper {wrapper_name!r}. "
                             f"Available: {list(WRAPPERS)}")
        env = WRAPPERS[wrapper_name](env, **wrapper_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


class VideoEvalCallback(BaseCallback):
    """Every `record_every` env-steps, roll one greedy eval episode + save MP4.

    Logs three reward streams to TensorBoard:
      eval/mean_reward       -- what the policy actually optimizes (shaped)
      eval/base_reward       -- the unshaped Ant-v5 reward, for cross-run
                                comparison (computed as shaped - shaping)
      eval/mean_idle_legs    -- diagnostic: avg # of legs with no recent
                                ground contact (low = good quadruped gait)

    The last two are only meaningful when a shaping wrapper is active; with
    no wrapper they degenerate to base_reward == mean_reward and
    mean_idle_legs == 0.
    """

    def __init__(self, eval_env: gym.Env, record_every: int, video_dir: Path,
                 best_model_path: Path, best_reward_path: Path, fps: int = 30):
        super().__init__()
        self.eval_env = eval_env
        self.record_every = record_every
        self.video_dir = video_dir
        self.best_model_path = best_model_path
        self.best_reward_path = best_reward_path
        self.fps = fps
        self.best_eval_reward = float("-inf")

    def _on_training_start(self) -> None:
        self.video_dir.mkdir(parents=True, exist_ok=True)
        if self.best_reward_path.exists():
            self.best_eval_reward = float(self.best_reward_path.read_text().strip())
            print(f"[best] previous best eval reward: {self.best_eval_reward:.1f}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.record_every == 0:
            self._record_one_episode()
        return True

    def _record_one_episode(self) -> None:
        frames: list[np.ndarray] = []
        obs, _ = self.eval_env.reset()
        total_reward = 0.0
        total_shaping = 0.0
        total_idle = 0.0
        n_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            frames.append(self.eval_env.render())
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += float(reward)
            total_shaping += float(info.get("shaping/foot_contact", 0.0))
            total_idle += float(info.get("shaping/idle_legs", 0.0))
            n_steps += 1

        path = self.video_dir / f"eval_step_{self.num_timesteps:09d}.mp4"
        imageio.mimsave(path, frames, fps=self.fps, codec="libx264")

        base_reward = total_reward - total_shaping
        mean_idle = total_idle / max(n_steps, 1)

        self.logger.record("eval/mean_reward", total_reward)
        self.logger.record("eval/base_reward", base_reward)
        self.logger.record("eval/mean_idle_legs", mean_idle)
        self.logger.record("eval/episode_length", n_steps)

        if self.verbose:
            print(f"[video] step={self.num_timesteps:,}  "
                  f"shaped={total_reward:.1f}  base={base_reward:.1f}  "
                  f"idle_legs={mean_idle:.2f}  ->  {path}")

        if total_reward > self.best_eval_reward:
            self.best_eval_reward = total_reward
            self.model.save(self.best_model_path)
            self.best_reward_path.write_text(f"{total_reward:.6f}")
            if self.verbose:
                print(f"[best]  new best eval reward: {total_reward:.1f}  "
                      f"->  {self.best_model_path}")
        self.logger.record("eval/best_mean_reward", self.best_eval_reward)


def _load_or_init_config(run_paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    """First invocation: write config.json from CLI args. Resume: read it back.

    Keeping wrapper/seed pinned in config.json (not re-read from CLI on resume)
    prevents an accidental --wrapper change mid-run from contaminating a buffer
    that was filled under different reward semantics.
    """
    if run_paths["config"].exists():
        config = json.loads(run_paths["config"].read_text())
        print(f"[config] loaded existing config from {run_paths['config']}")
        if (args.wrapper is not None or args.wrapper_kwargs != "{}"
                or args.seed is not None or args.env != DEFAULT_ENV):
            print("[config] note: --env/--wrapper/--seed args ignored on resume "
                  "(config.json is the source of truth)")
        return config

    wrapper_kwargs = json.loads(args.wrapper_kwargs)
    config = {
        "env_id": args.env,
        "algo": "SAC",
        "wrapper": args.wrapper,
        "wrapper_kwargs": wrapper_kwargs,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "buffer_size": BUFFER_SIZE,
            "batch_size": BATCH_SIZE,
            "policy": "MlpPolicy",
        },
        "env_kwargs": {},
        "seed": args.seed,
        "notes": "",
    }
    run_paths["config"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["config"].write_text(json.dumps(config, indent=2) + "\n")
    print(f"[config] wrote new config to {run_paths['config']}")
    return config


def _record_or_verify_obs_spec(config: dict[str, Any], run_paths: dict[str, Path],
                               env: gym.Env) -> None:
    """Pin what this run trained against — or refuse to resume if it moved.

    `learnings/003`: the observation list is the one truly one-way door here.
    Until now no run recorded which observation it was trained against, so the
    only evidence was the pickled space width inside the checkpoint, and the
    only way to detect a change was to attempt a load and catch the exception.
    That is an autopsy. `hound_desert_test150k` was orphaned this way and it
    took git archaeology to work out that the env had never been committed at
    the width its checkpoint carries.

    A width change would at least fail loudly at `SAC.load()`. The dangerous
    case is the one that does NOT: reordering two terms, or redefining what a
    term means, keeps the width identical, loads cleanly, and silently feeds
    the policy a permuted world. The hash catches that; nothing else here can.

    Fresh run: record it. Resume: compare and raise. Legacy runs (started
    before this existed) carry no spec — they are recorded on their next
    resume rather than guessed at, because back-filling a spec from today's
    code is exactly the false provenance this is meant to prevent.
    """
    spec = env.unwrapped._obs_spec
    recorded = config.get("obs_spec")

    if recorded is None:
        config["obs_spec"] = spec.to_record()
        run_paths["config"].write_text(json.dumps(config, indent=2) + "\n")
        print(f"[config] pinned obs spec {spec.hash} ({spec.width} values)")
        return

    if recorded.get("hash") != spec.hash:
        raise RuntimeError(
            f"observation spec changed since this run started.\n"
            f"  recorded: {recorded.get('hash')} "
            f"width {recorded.get('width')} {recorded.get('terms')}\n"
            f"  current:  {spec.hash} width {spec.width} "
            f"{[(t.name, t.size) for t in spec.terms]}\n"
            f"Resuming would train a policy against a different observation "
            f"than the one in its checkpoint and replay buffer. Either restore "
            f"the observation list, or start a NEW run name — do not resume "
            f"this one (learnings/003)."
        )
    print(f"[config] obs spec {spec.hash} matches the recorded spec")


def _record_or_verify_reward_spec(config: dict[str, Any], run_paths: dict[str, Path],
                                  env: gym.Env) -> None:
    """Pin the reward this run trained under — or refuse to resume if it moved.

    The observation half of this got written first because it fails loudly:
    change the width and `SAC.load()` raises. The reward half fails silently,
    which is worse. `research/anomalies.jsonl` records the damage already done
    — three hound runs at two different `ctrl_cost_weight` values, two of them
    compared against each other in the record, with nothing anywhere saying so.

    Resuming across a reward change is `learnings/002` exactly: the replay
    buffer is full of transitions labelled with rewards that will never be paid
    again, and the critic has been fitted to them. That run destroyed a working
    gait in a few thousand steps and never recovered.

    Envs that do not declare a reward spec are skipped rather than guessed at,
    for the same reason legacy runs carry no obs spec: back-filling provenance
    from today's code is the false-provenance failure this exists to prevent.
    """
    spec = getattr(env.unwrapped, "_reward_spec", None)
    if spec is None:
        return

    recorded = config.get("reward_spec")

    if recorded is None:
        config["reward_spec"] = spec.to_record()
        run_paths["config"].write_text(json.dumps(config, indent=2) + "\n")
        print(f"[config] pinned reward spec {spec.hash} "
              f"(shape {spec.shape_hash}, {len(spec.terms)} terms)\n"
              f"{spec.describe()}")
        return

    if recorded.get("hash") == spec.hash:
        print(f"[config] reward spec {spec.hash} matches the recorded spec")
        return

    # Both cases raise. They are separated because the remedies differ and a
    # single "the reward changed" message sends people looking in the wrong
    # place -- learnings/004 is precisely that these are different failures.
    shape_moved = recorded.get("shape_hash") != spec.shape_hash
    if shape_moved:
        detail = (
            f"The reward's TERMS changed, not just its weights.\n"
            f"  recorded shape: {recorded.get('shape_hash')} "
            f"{[t['name'] for t in recorded.get('terms', [])]}\n"
            f"  current shape:  {spec.shape_hash} {[t.name for t in spec.terms]}\n"
            f"This reward pays for a different thing than the one in the replay "
            f"buffer. The two runs are not comparable in kind and no relabelling "
            f"fixes it (learnings/004). Start a NEW run name."
        )
    else:
        detail = (
            f"The reward's WEIGHTS changed; its terms did not.\n"
            f"  recorded: {recorded.get('hash')} {recorded.get('terms')}\n"
            f"  current:  {spec.hash} "
            f"{[(t.name, t.weight) for t in spec.terms]}\n"
            f"Every transition in the replay buffer is labelled with a reward "
            f"that will never be paid again, and the critic is fitted to them. "
            f"Start a NEW run name."
        )

    raise RuntimeError(
        f"reward spec changed since this run started.\n{detail}\n"
        f"Warm-starting across a reward change destroyed a working gait in a "
        f"few thousand steps once already (learnings/002, nulls.jsonl row 1)."
    )


def _record_or_verify_terrain_spec(config: dict[str, Any], run_paths: dict[str, Path],
                                   env: gym.Env) -> None:
    """Pin the ground this run trained on — or refuse to resume if it moved.

    The third input to a run's dynamics, and the last one to get this
    treatment. `research/anomalies.jsonl` (2026-07-27): the heightfield had no
    hash, no `config.json` field and no guard, while the observation and the
    reward had all three.

    A terrain swap is the quietest of the three failures. Changing the
    observation makes `SAC.load()` raise; changing the reward at least leaves a
    weight to notice in the source. Changing the ground leaves nothing: the
    checkpoint loads, the widths match, the reward is untouched, and every
    number the run produces is measured against a world the previous numbers
    never saw. `research/scripts/compare_terrain_grids.py` put a figure on it —
    the GRID=2048 regen that was on the table correlates with the committed
    terrain at **+0.0610**, because `generate.py` indexes its `(n, n)` phase
    array by FFT bin, so changing `n` hands every phase to a different
    wavelength even though the RNG stream is bit-identical.

    THREE STATES, NOT TWO

    The key is written even when there is no terrain, and `null` is a real
    answer meaning "this model has no heightfield":

        absent   this run predates the terrain record (or its env is not
                 MuJoCo), so nothing can be said about its ground
        null     determined: flat world, no heightfield
        object   determined: this heightfield

    A flat env is therefore not merely exempt, it is *asserted flat* — so
    repointing `Hound-v0` at a desert XML mid-run raises rather than passing
    for want of anything to compare. Recording only the terrain case would have
    made "flat" and "unknown" the same value, and they are not.

    Legacy runs stay legacy: back-filling a hash from today's asset would state
    that `hound_pd_desert_s1` trained on the terrain that happens to be checked
    out right now, which is the false provenance this whole mechanism exists to
    prevent.
    """
    model = getattr(env.unwrapped, "model", None)
    if model is None:
        # Not MuJoCo, so there is no compiled ground to measure. Say so rather
        # than writing `null`, which would claim the world is flat.
        print(f"[config] {type(env.unwrapped).__name__} exposes no MuJoCo model; "
              f"terrain is not recorded for this run")
        return

    spec = TerrainSpec.from_model(model)
    # `.get()` cannot tell an absent key from a recorded `null`, and that is the
    # whole distinction between "predates the record" and "verified flat".
    recorded = config["terrain_spec"] if "terrain_spec" in config else _UNRECORDED

    if recorded is _UNRECORDED:
        config["terrain_spec"] = None if spec is None else spec.to_record()
        run_paths["config"].write_text(json.dumps(config, indent=2) + "\n")
        if spec is None:
            print("[config] pinned terrain: none (flat world, no heightfield)")
        else:
            print(f"[config] pinned terrain {spec.hash} "
                  f"(field {spec.field_hash})\n{spec.describe()}")
        return

    if recorded is None and spec is None:
        print("[config] terrain: still a flat world, as recorded")
        return

    if recorded is None:
        raise RuntimeError(
            f"this run was started on a FLAT world and the env now has terrain.\n"
            f"  recorded: no heightfield\n"
            f"  current:  {spec.hash} (field {spec.field_hash})\n{spec.describe()}\n"
            f"Every transition in the replay buffer was collected on level "
            f"ground, and the critic is fitted to them. `learnings/001` is "
            f"this move going wrong: the same reward that paid 7.05 per step "
            f"for forward progress on the flat world paid 0.29 on the desert, "
            f"flipping the payoff-to-effort ratio from 8.7:1 to 0.51:1 and "
            f"making standing still the better policy. Start a NEW run name."
        )

    if spec is None:
        raise RuntimeError(
            f"this run was started on terrain and the env is now a FLAT world.\n"
            f"  recorded: {recorded.get('hash')} (field {recorded.get('field_hash')}) "
            f"{recorded.get('nrow')}x{recorded.get('ncol')} from "
            f"{recorded.get('source')!r}\n"
            f"  current:  no heightfield\n"
            f"The model lost its floor — most likely a `*_desert.xml` was "
            f"regenerated without its <hfield>, or --env resolved to the flat "
            f"twin. Restore the world; do not resume onto different ground."
        )

    if recorded.get("hash") == spec.hash:
        print(f"[config] terrain {spec.hash} matches the recorded ground")
        return

    # Both cases raise. They are separated because the remedies point at
    # different files, and one message saying "the terrain changed" sends
    # people to regenerate an asset when the actual edit was one number in an
    # XML — the same reasoning that splits reward hash from reward shape_hash.
    if recorded.get("field_hash") != spec.field_hash:
        detail = (
            f"The height SAMPLES changed. This is a different terrain.\n"
            f"  recorded: field {recorded.get('field_hash')} "
            f"{recorded.get('nrow')}x{recorded.get('ncol')} "
            f"sha256 {recorded.get('data_sha256')}\n"
            f"  current:  field {spec.field_hash} {spec.nrow}x{spec.ncol} "
            f"sha256 {spec.data_sha256}\n"
            f"`assets/terrain/desert_hfield.bin` is not the file this run "
            f"trained on. Another seed changes the world, and so does another "
            f"GRID: research/scripts/compare_terrain_grids.py measured the "
            f"1024 -> 2048 regen at correlation +0.0610 with the committed "
            f"terrain — a different world with the same statistics. Restore "
            f"the asset (`git checkout`), or start a NEW run name and "
            f"re-measure every number a MOVING policy produced against it."
        )
    else:
        detail = (
            f"The same height samples, placed differently in the world.\n"
            f"  recorded: {recorded.get('hash')} extent "
            f"{recorded.get('x_half_extent_m')} x {recorded.get('y_half_extent_m')} m, "
            f"z_span {recorded.get('z_span_m')} m, base {recorded.get('z_base_m')}, "
            f"pos {recorded.get('pos_m')}, quat {recorded.get('quat')}\n"
            f"  current:  {spec.hash} extent "
            f"{spec.x_half_extent_m} x {spec.y_half_extent_m} m, "
            f"z_span {spec.z_span_m} m, base {spec.z_base_m}, "
            f"pos {list(spec.pos_m)}, quat {list(spec.quat)}\n"
            f"A `<hfield size=...>` or floor-geom attribute was edited in a "
            f"`*_desert.xml`. The elevation span multiplies every slope in the "
            f"world, so this is a dynamics change even though the .bin is "
            f"untouched. It is a one-line revert."
        )

    raise RuntimeError(
        f"the terrain changed since this run started.\n{detail}\n"
        f"Resuming would train against ground the replay buffer has never "
        f"seen: every stored transition, and the critic fitted to them, "
        f"describes a world that no longer exists. The run would not be "
        f"comparable in kind to itself, let alone to any ledger row beside it."
    )


def _build_model(env: gym.Env, run_paths: dict[str, Path], seed: int | None) -> tuple[SAC, bool]:
    """Resume from runs/<name>/ant_sac.zip if present, else fresh agent."""
    if run_paths["model"].exists():
        print(f"Resuming from {run_paths['model']}")
        model = SAC.load(run_paths["model"], env=env, device=DEVICE,
                         tensorboard_log=str(run_paths["tb"]))
        if run_paths["buffer"].exists():
            print(f"  loading replay buffer from {run_paths['buffer']}")
            model.load_replay_buffer(run_paths["buffer"])
        else:
            print("  no replay buffer file -- starting with an empty buffer")
        return model, True

    print(f"Starting fresh SAC run -> {run_paths['model']}")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        tensorboard_log=str(run_paths["tb"]),
        seed=seed,
        verbose=1,
    )
    return model, False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", type=str, required=True,
                   help="subdirectory under runs/ to write artifacts into")
    p.add_argument("--env", type=str, default=DEFAULT_ENV,
                   help=f"Gymnasium env id, set once at run creation "
                        f"(default: {DEFAULT_ENV}); pinned in config.json on resume")
    p.add_argument("--steps", type=int, default=750_000,
                   help="env-steps to run THIS invocation (default: 750_000)")
    p.add_argument("--video-every", type=int, default=50_000,
                   help="record an eval MP4 every N env-steps; 0 disables")
    p.add_argument("--wrapper", type=str, default=None, choices=list(WRAPPERS.keys()),
                   help="reward-shaping wrapper to apply (omit for none)")
    p.add_argument("--wrapper-kwargs", type=str, default="{}",
                   help='JSON dict of kwargs for the wrapper, e.g. \'{"penalty": 1.0}\'')
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for SAC + env (default: None = nondeterministic)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = paths.RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_paths = _run_paths(run_dir)

    config = _load_or_init_config(run_paths, args)
    env_id = config["env_id"]
    wrapper_name = config["wrapper"]
    wrapper_kwargs = config["wrapper_kwargs"]
    seed = config["seed"]

    train_env = _make_env(env_id, wrapper_name, wrapper_kwargs, seed)
    # Before a single step is taken: pin what this run trains against on a
    # fresh run, and on a resume refuse to continue if any of the three has
    # moved. The observation is the loud failure (SAC.load raises), the reward
    # is the silent one, and the terrain is the silent one that leaves no trace
    # in the source at all.
    _record_or_verify_obs_spec(config, run_paths, train_env)
    _record_or_verify_reward_spec(config, run_paths, train_env)
    _record_or_verify_terrain_spec(config, run_paths, train_env)

    model, is_resume = _build_model(train_env, run_paths, seed)

    callbacks = []
    if args.video_every > 0:
        eval_env = _make_env(env_id, wrapper_name, wrapper_kwargs, seed, render_mode="rgb_array")
        callbacks.append(VideoEvalCallback(
            eval_env=eval_env,
            record_every=args.video_every,
            video_dir=run_paths["videos"],
            best_model_path=run_paths["best_model"],
            best_reward_path=run_paths["best_reward"],
        ))

    try:
        model.learn(
            total_timesteps=args.steps,
            reset_num_timesteps=not is_resume,
            callback=callbacks or None,
            progress_bar=True,
        )
    finally:
        print(f"\nSaving model to {run_paths['model']}")
        model.save(run_paths["model"])
        print(f"Saving replay buffer to {run_paths['buffer']}")
        model.save_replay_buffer(run_paths["buffer"])
        train_env.close()


if __name__ == "__main__":
    main()
