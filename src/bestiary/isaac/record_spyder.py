"""Record camera-labeled rollouts from a trained Spyder checkpoint.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.record_spyder \\
        --checkpoint runs/spyder_gentle_s1/box_logs/2026-08-06_07-53-39/model_1499.pt \\
        --episodes 2 --out runs/spyder_tapes

The stage-2 data factory: the height-scan teacher walks the gentle mix under
seeded one-command scripts while a robot-mounted camera films, and every
policy step is logged as the exact (obs, action, command) the network consumed
and produced, plus 10 Hz first-person frames and a mechanical language caption.
Format contract: `research/SPYDER_TAPES_SPEC.md` — this file and that spec
change together or not at all. Schedules come from `spyder_tape_commands`, the
pure module the closed-loop eval will replay byte-identically.

WHY THE TEACHER'S SCAN MAKES THE CAMERA LEARNABLE. The teacher is privileged:
its 187-ray height scan senses the ground its foot placement responds to. The
camera films that same ground, so a student distilled from (frames -> actions)
is forced to recover the scan's information from pixels. Tapes from a
proprioception-only teacher would contain nothing for the camera to explain —
the failure mode this recorder exists to avoid.

WHAT A RECORDING IS NOT (inherited verbatim from the Spot recorder): nothing
is normalized, reordered, or converted. Raw SI tensors at the policy's own
rate, raw uint8 frames at the camera's own rate; normalization statistics
belong to training code and are computed from the train split only. Episodes
that fall are quarantined in `fallen/`, never deleted; holdout is decided
here, at record time, by seed (`seed % 10 == 0`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from bestiary import paths
from bestiary.isaac.spyder_tape_commands import (
    DRIVE_S,
    SETTLE_S,
    STAND_S,
    command_text,
    sample_command,
)

#: Interface contract, asserted against the live env at startup — a mismatch
#: means the task moved under this recorder and the tape would be garbage.
#: 235 = 3 lin vel + 3 ang vel + 3 gravity + 3 command + 12 joint pos +
#: 12 joint vel + 12 prev action + 187 height-scan rays
#: (`spyder_gentle_env_cfg`: same 17x11 grid as upstream, 0.16 m spacing).
EXPECTED_OBS_DIM = 235
EXPECTED_ACT_DIM = 12
EXPECTED_STEP_DT_S = 0.02  # 200 Hz physics, decimation 4 -> policy at 50 Hz

#: Camera. 224 px because that is PaliGemma's input resolution (SigLIP-So400m
#: /14 -> 16x16 patches; the PaliGemma paper, arXiv:2407.07726) — taping at
#: the student's native resolution avoids a lossy resize at training time.
#: 10 Hz frames against 50 Hz actions is the VLA norm (frames condition
#: action CHUNKS, not single steps); STRIDE below derives from it.
CAM_RES_PX = 224
FRAME_HZ = 10.0
#: Mount: torso frame, forward of and above the torso centre, pitched down so
#: the view covers the ground the height scan prices — the scan reaches
#: 1.28 m ahead (`spyder_gentle_env_cfg` footprint). First smoke (2026-08-08,
#: z=0.12, pitch 20) filmed the torso shell over the bottom ~40% of frame —
#: the near ground was hidden — so the mount rose to 0.22 m and 25 deg
#: (operator decision on the smoke stills): lens ~0.45 m over ground at
#: stance, centre ray hits ~0.97 m ahead, 82 deg horizontal FOV (12 mm focal,
#: 20.955 mm aperture) spans near-feet to horizon over the shell's front edge.
CAM_POS_M = (0.18, 0.0, 0.22)
CAM_PITCH_DEG = 25.0
CAM_FOCAL_MM = 12.0

parser = argparse.ArgumentParser(description="Record Spyder camera tapes from a trained checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="model_*.pt to tape (the teacher)")
parser.add_argument("--episodes", type=int, default=16, help="episodes to record this invocation")
parser.add_argument("--seed0", type=int, default=0, help="first episode seed; episode k uses seed0+k")
parser.add_argument("--out", type=Path, default=paths.RUNS / "spyder_tapes",
                    help="output root; train/ holdout/ fallen/ are created under it")
parser.add_argument("--task", type=str, default="Bestiary-Gentle-Spyder-Play-v0")
parser.add_argument("--num_envs", type=int, default=16, help="parallel tape decks (one episode each per batch)")
parser.add_argument("--rows", type=int, default=4, help="terrain grid rows (Play default is 3)")
parser.add_argument("--cols", type=int, default=4, help="terrain grid cols")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Headless data factory with RTX sensors: cameras on, no viewer, no window.
args.headless = True
args.enable_cameras = True
_extra = "--/log/channels/omni.physx.fabric.plugin=error"
args.kit_args = f"{args.kit_args} {_extra}" if getattr(args, "kit_args", None) else _extra

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402  (kit boots before imports)
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from bestiary.isaac import tasks  # noqa: E402
from bestiary.isaac.spyder_cfg import TORSO_PRIM_SUBPATH  # noqa: E402

#: Sampler exile — nothing may overwrite an injected command. Same constant
#: and reason as `play_spyder` / `eval_hound`.
NO_RESAMPLE_S = 1_000_000.0


def _cam_quat_wxyz(pitch_deg: float) -> tuple[float, float, float, float]:
    """ROS-optical camera looking along body +x, pitched down `pitch_deg`.

    A level ROS camera (+z out the lens, +x right, +y down) forward-mounted in
    a +x-forward/+z-up body frame is q0 = (0.5, -0.5, 0.5, -0.5) — its matrix
    maps cam x->-y_body, cam y->-z_body, cam z->+x_body. Pitch-down is a
    rotation about body +y (takes +x toward -z), composed as the Hamilton
    product q_y(pitch) * q0. Computed, not hardcoded, so the pitch is one
    number instead of four opaque ones.
    """
    h = math.radians(pitch_deg) / 2.0
    c, s = math.cos(h), math.sin(h)
    w0, x0, y0, z0 = 0.5, -0.5, 0.5, -0.5
    return (c * w0 - s * y0, c * x0 + s * z0, c * y0 + s * w0, c * z0 - s * x0)


def _rgb(cam) -> np.ndarray:
    """Latest frames, (N, H, W, 3) uint8, whatever the annotator's dtype."""
    arr = cam.data.output["rgb"].detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    """Encode one episode's frames; H.264 CRF 18, the player-proven pipe."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", f"{fps:g}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path} — tape incomplete, not written")


def main() -> int:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not on PATH — frames could not be encoded; install it first")

    tasks.register()
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = args.seed0
    # No timeout teleports inside a batch; the batch loop owns episode length.
    env_cfg.episode_length_s = 3600.0
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.resampling_time_range = (NO_RESAMPLE_S, NO_RESAMPLE_S)
    cmd_cfg.rel_standing_envs = 0.0  # a standing-flagged env zeroes injected commands
    cmd_cfg.debug_vis = False  # command arrows must not appear in taped frames
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    # Wider grid than Play's 3x3: one tile per deck is the diversity floor.
    env_cfg.scene.terrain.terrain_generator.num_rows = args.rows
    env_cfg.scene.terrain.terrain_generator.num_cols = args.cols

    # The trained command envelope, read off the live config so the sampler
    # cannot disagree with the task (play_spyder's rule).
    vx_range = tuple(cmd_cfg.ranges.lin_vel_x)
    vy_range = tuple(cmd_cfg.ranges.lin_vel_y)
    wz_range = tuple(cmd_cfg.ranges.ang_vel_z)
    vx_min = float(getattr(cmd_cfg, "min_lin_vel_x", 0.0))
    wz_min = float(getattr(cmd_cfg, "min_ang_vel_z", 0.0))

    # The camera, attached to the torso link inside each env's robot prim.
    # A dynamically-added scene attribute is how Isaac Lab's InteractiveScene
    # discovers sensors: it iterates the cfg's attributes at construction.
    stride = round(1.0 / (EXPECTED_STEP_DT_S * FRAME_HZ))
    if not math.isclose(stride * EXPECTED_STEP_DT_S * FRAME_HZ, 1.0, rel_tol=1e-9):
        raise SystemExit(f"FRAME_HZ {FRAME_HZ} does not divide the {1/EXPECTED_STEP_DT_S:g} Hz policy rate")
    env_cfg.scene.front_camera = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{TORSO_PRIM_SUBPATH}/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS_M, rot=_cam_quat_wxyz(CAM_PITCH_DEG), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAM_FOCAL_MM, clipping_range=(0.05, 60.0)),
        width=CAM_RES_PX,
        height=CAM_RES_PX,
        data_types=["rgb"],
        update_period=1.0 / FRAME_HZ,
    )
    # Render only when a frame is due: physics steps per render = decimation
    # (physics per policy step) x stride (policy steps per frame). Frames are
    # therefore at most one policy step older than the obs row they pair with.
    env_cfg.sim.render_interval = int(env_cfg.decimation) * stride

    env = gym.make(args.task, cfg=env_cfg)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:  # noqa: BLE001 -- shim, absent in some versions; runner.load is the real check
        pass
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"[bestiary] teacher loaded from {ckpt}", flush=True)

    scene = env.unwrapped.scene
    cam = scene["front_camera"]
    term = env.unwrapped.command_manager.get_term("base_velocity")
    step_dt = float(env.unwrapped.step_dt)
    if not math.isclose(step_dt, EXPECTED_STEP_DT_S, rel_tol=1e-6):
        raise RuntimeError(
            f"policy rate moved: step_dt={step_dt}, recorder and spec were written "
            f"for {EXPECTED_STEP_DT_S} (50 Hz). Re-verify the contract before taping."
        )
    dof_names = list(scene["robot"].data.joint_names)
    terrain_md5 = hashlib.md5(paths.GENTLE_HFIELD.read_bytes()).hexdigest()

    def _obs():
        got = wrapper.get_observations()
        return got[0] if isinstance(got, tuple) else got

    def _pol(o) -> torch.Tensor:
        """The policy-group tensor out of whatever the wrapper returned.

        rsl-rl 5.x hands back a TensorDict whose batch shape is (num_envs,) —
        `.shape[1]` does not exist on it even though the policy consumes it
        directly (measured on the 2026-08-08 smoke run). The tape stores the
        actual network input, so it is extracted here; older wrappers return
        the raw tensor and pass through untouched.
        """
        if torch.is_tensor(o):
            return o
        if hasattr(o, "keys") and "policy" in o.keys():
            return o["policy"]
        raise RuntimeError(f"cannot find the policy obs tensor in {type(o).__name__}")

    n = args.num_envs
    settle_steps = int(SETTLE_S / step_dt)
    stand_steps = int(STAND_S / step_dt)
    tape_steps = stand_steps + int(DRIVE_S / step_dt)

    for sub in ("train", "holdout", "fallen"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    written = 0
    batches = math.ceil(args.episodes / n)
    t_wall0 = time.monotonic()
    for b in range(batches):
        seeds = [args.seed0 + b * n + i for i in range(n)]
        cmds = [
            sample_command(np.random.default_rng(s), vx_range, vx_min, vy_range, wz_range, wz_min)
            for s in seeds
        ]
        cmd_tensor = torch.tensor(cmds, device="cuda:0", dtype=torch.float32)

        env.unwrapped.reset()
        obs = _obs()
        obs_w = int(_pol(obs).shape[-1])
        if obs_w != EXPECTED_OBS_DIM:
            raise RuntimeError(
                f"obs width {obs_w} != expected {EXPECTED_OBS_DIM} — the task's "
                "observation layout moved; recorder and spec must be re-verified."
            )

        # Settle, untaped: zero commands while the reset scatter rings down.
        for _ in range(settle_steps):
            term.vel_command_b[:, :] = 0.0
            with torch.inference_mode():
                obs, _, _, _ = wrapper.step(policy(obs))

        obs_log: list[np.ndarray] = []
        act_log: list[np.ndarray] = []
        cmd_log: list[np.ndarray] = []
        frame_log: list[np.ndarray] = []
        active_until = [tape_steps] * n
        fell = [False] * n

        for t in range(tape_steps):
            term.vel_command_b[:, :] = 0.0 if t < stand_steps else cmd_tensor
            if t % stride == 0:
                frames = _rgb(cam)
                if b == 0 and t == 0:
                    stds = frames.reshape(n, -1).std(axis=1)
                    if float(stds.max()) < 2.0:
                        raise RuntimeError(
                            f"camera frames are blank (max per-env std {stds.max():.2f}) — "
                            "the render-anomaly class STATE documents (frozen boot camera / "
                            "sky-only offscreen). Fix the camera before taping."
                        )
                frame_log.append(frames)
            with torch.inference_mode():
                act = policy(obs)
                obs_log.append(_pol(obs).detach().cpu().numpy().copy())
                act_log.append(act.detach().cpu().numpy().copy())
                cmd_log.append(term.vel_command_b.detach().cpu().numpy().copy())
                obs, _, dones, _ = wrapper.step(act)
            done_idx = torch.nonzero(dones, as_tuple=False).flatten().tolist()
            for i in done_idx:
                if active_until[i] == tape_steps and t + 1 < tape_steps:
                    active_until[i] = t + 1
                    fell[i] = True

        obs_arr = np.stack(obs_log)      # (T, N, obs)
        act_arr = np.stack(act_log)      # (T, N, 12)
        cmd_arr = np.stack(cmd_log)      # (T, N, 3)
        frame_arr = np.stack(frame_log)  # (F, N, H, W, 3)

        # Staleness guard, once: a driving survivor's frames must change.
        if b == 0:
            for i in range(n):
                if not fell[i] and cmds[i] != (0.0, 0.0, 0.0):
                    diff = np.abs(np.diff(frame_arr[:, i].astype(np.int16), axis=0)).mean()
                    if diff < 0.05:
                        raise RuntimeError(
                            f"frames of driving env {i} are static (mean |diff| {diff:.4f}) — "
                            "render_interval/update_period mismatch is serving stale frames."
                        )
                    break

        for i in range(n):
            if written >= args.episodes:
                break
            seed, (vx, vy, wz) = seeds[i], cmds[i]
            t_end = active_until[i]
            f_end = math.ceil(t_end / stride)
            sub = "fallen" if fell[i] else ("holdout" if seed % 10 == 0 else "train")
            stem = args.out / sub / f"ep_{seed:05d}"
            _write_mp4(stem.with_suffix(".mp4"), frame_arr[:f_end, i], FRAME_HZ)
            np.savez_compressed(
                stem.with_suffix(".npz"),
                obs=obs_arr[:t_end, i],
                act=act_arr[:t_end, i],
                cmd=cmd_arr[:t_end, i],
                meta=json.dumps({
                    "seed": seed,
                    "text": command_text(vx, vy, wz),
                    "command": [vx, vy, wz],
                    "fell": fell[i],
                    "steps": t_end,
                    "frames": f_end,
                    "dt_s": step_dt,
                    "frame_hz": FRAME_HZ,
                    "frame_stride": stride,
                    "stand_steps": stand_steps,
                    "obs_dim": EXPECTED_OBS_DIM,
                    "act_dim": EXPECTED_ACT_DIM,
                    "dof_names": dof_names,
                    "checkpoint": str(ckpt),
                    "task": args.task,
                    "terrain_rows_cols": [args.rows, args.cols],
                    "terrain_hfield": paths.GENTLE_HFIELD.name,
                    "terrain_hfield_md5": terrain_md5,
                    "cam": {
                        "res_px": CAM_RES_PX,
                        "pos_m": list(CAM_POS_M),
                        "pitch_deg": CAM_PITCH_DEG,
                        "focal_mm": CAM_FOCAL_MM,
                        "convention": "ros",
                    },
                    "spec": "research/SPYDER_TAPES_SPEC.md",
                }),
            )
            written += 1
            print(
                f"[{written}/{args.episodes}] seed={seed} '{command_text(vx, vy, wz)}' "
                f"steps={t_end} frames={f_end} fell={fell[i]} -> {stem}.npz",
                flush=True,
            )

        done_frac = written / args.episodes
        elapsed = time.monotonic() - t_wall0
        eta = elapsed / done_frac - elapsed if done_frac else float("inf")
        rate = (b + 1) * (settle_steps + tape_steps) * n / elapsed
        print(
            f"[batch {b + 1}/{batches}] {rate:,.0f} env-steps/s wall, "
            f"elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min",
            flush=True,
        )

    del frame_arr, obs_arr, act_arr, cmd_arr
    print(f"[bestiary] taping done: {written} episodes under {args.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    # Kit teardown drops stdout and close() exits 0 unconditionally — same two
    # traps, same flush-and-_exit answer as every entry point in this package.
    import os
    import sys
    import traceback

    try:
        code = main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
