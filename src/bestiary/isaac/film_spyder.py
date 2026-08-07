"""Film a trained Spyder from a CAMERA SENSOR, not the viewport.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.film_spyder \\
        --checkpoint runs/spyder_forward_s1/box_logs/2026-08-06_22-59-19/model_1499.pt \\
        --out runs/spyder_forward_s1/forward_rough.mp4 --seconds 18

WHY THIS FILE EXISTS. `play_spyder --video` renders through the VIEWPORT, and
on this project's installs the viewport camera is frozen at boot pointing at
the sky (`research/anomalies.jsonl` id 62). Four mechanisms were measured as
silent no-ops on the recorded frames — `viewer.origin_type = "asset_root"`,
`sim.set_camera_view`, a USD transform write on every `UsdGeom.Camera` on the
stage, and a USDRT/Fabric mirror of the same pose. Every local take since
2026-08-06 has come out as 100% sky; the rented box rendered terrain but with
the same immovable camera, so a policy holding 5 m/s left the frame in
seconds and `--follow` changed nothing.

The fix is not a fifth way to push a pose at the viewport's camera. It is to
stop using the viewport: an `isaaclab.sensors.Camera` owns ITS OWN render
product bound to ITS OWN prim, so there is no ambiguity about which camera
produced a frame, and `set_world_poses_from_view` is the supported way to move
it. That makes a real trailing shot possible for the first time.

WHAT THE GUARD IS FOR. Every failure this file exists to fix produced a large,
valid, perfectly encoded MP4 — of nothing. Size and exit code cannot tell a
good take from sky, so they are not the check. `_MotionGuard` samples the
frames actually written and requires the take to MOVE: a static camera aimed
at empty sky, an empty-terrain take, and a dead render product all collapse to
near-zero inter-frame difference, while any take containing a walking robot
does not. It raises with the measured number, so a failure says how far off it
was rather than merely that it failed.

The floor is 1.2/255 mean absolute inter-frame difference. Measured
2026-08-07 on this install: the sky-only takes scored 0.02-0.05 and the first
good box take scored 4.9, so the floor sits ~25x above the failures and ~4x
below the passes — chosen inside that gap rather than at either edge.

Nothing filmed here enters the record; this is the eyes, and the per-cell
grid remains the instrument.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

#: Nine robots on the stage, eight hidden from the renderer. Inherited from
#: `play_spyder`, where `num_envs = 1` was measured to draw terrain and NO
#: robot across three camera setups on this install; the working count is kept
#: and the twins are made invisible rather than deleted.
NUM_ENVS = 9

#: Camera pose in the robot's YAW frame, metres: 7.0 m behind, 2.5 m to the
#: side, 3.4 m up. Behind-and-above-the-shoulder — far enough back that a
#: 0.5 m machine bounding at 5 m/s stays framed through a stride, high enough
#: to look DOWN onto the terrain it is crossing rather than across it, which
#: is the whole point of filming on rough ground.
CAM_OFFSET_B = (-7.0, 2.5, 3.4)

#: Side-tracking offset, WORLD frame, metres: 11 m out on -y and 3.6 m up,
#: level with the strip. Parked on -y and looking back at the machine, the
#: camera's right-hand direction is world +x, so travel up the demo ramp reads
#: as left-to-right across the frame.
CAM_OFFSET_SIDE_W = (0.0, -11.0, 3.6)

#: Aim point above the robot's origin, metres. The torso rides ~0.25 m up in a
#: bound; aiming slightly high puts the machine on the centre line rather than
#: the bottom edge.
CAM_LOOKAT_DZ = 0.4

#: Mean absolute inter-frame difference, 0-255, below which a take is declared
#: blank. See the module docstring for the two measurements that bracket it.
MOTION_FLOOR = 1.2

#: Frames sampled for the guard. The check is a sanity floor, not a metric —
#: 60 samples spread across a take is plenty to separate 0.03 from 4.9, and
#: holding every frame of a 20 s 720p take in memory is not.
MOTION_SAMPLES = 60


class _MotionGuard:
    """Accumulates a cheap "did anything move?" statistic over a take.

    Deliberately measured on the frames HANDED TO THE ENCODER, not on the
    simulation state: the whole class of bug being killed here is one where
    the physics is perfect and the pixels are of the sky.
    """

    def __init__(self, samples: int = MOTION_SAMPLES):
        self._samples = samples
        self._prev: np.ndarray | None = None
        self._diffs: list[float] = []
        self._n = 0

    def observe(self, frame: np.ndarray, total_expected: int) -> None:
        # Sample evenly rather than taking the first N: a take can begin on a
        # reset frame and the interesting part is the middle.
        stride = max(1, total_expected // self._samples)
        self._n += 1
        if self._n % stride:
            return
        # Downsample 8x and go grayscale: the statistic is "did the picture
        # change", and full resolution costs 30x the arithmetic to answer it.
        small = frame[::8, ::8, :3].mean(axis=2).astype(np.float32)
        if self._prev is not None:
            self._diffs.append(float(np.abs(small - self._prev).mean()))
        self._prev = small

    @property
    def score(self) -> float:
        return float(np.median(self._diffs)) if self._diffs else 0.0

    def assert_alive(self, path: Path) -> None:
        score = self.score
        if score < MOTION_FLOOR:
            raise SystemExit(
                f"BLANK TAKE: {path} has a median inter-frame difference of "
                f"{score:.3f}/255, below the {MOTION_FLOOR} floor over "
                f"{len(self._diffs)} samples. The encoder ran and the file is "
                f"valid, so this is the anomalies.jsonl id 62 failure mode — "
                f"the camera rendered something that does not move (sky, empty "
                f"terrain, or a dead render product). The file is kept for "
                f"inspection; it is not a usable take."
            )
        print(f"[bestiary] motion guard: {score:.2f}/255 (floor {MOTION_FLOOR}) — take is live", flush=True)


class _Enc:
    """Raw RGB -> ffmpeg. Same pipe as `play_spyder`; gym's RecordVideo froze
    on the Hound path and this owns nothing clever."""

    def __init__(self, path: Path, fps: int):
        self.path, self._fps, self._proc = Path(path), int(fps), None

    def add(self, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr[..., :3])
        if self._proc is None:
            h, w = arr.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self._fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(self.path)],
                stdin=subprocess.PIPE,
            )
        self._proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="Bestiary-Forward-Spyder-Play-v0")
    parser.add_argument("--out", type=str, required=True, help="output MP4 path")
    parser.add_argument("--seconds", type=float, default=18.0, help="sim seconds to film")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--tiles", type=int, default=0,
        help="0 (default) leaves the task's own terrain alone — correct for "
        "Bestiary-Demo-Spyder-Play-v0, whose single tile IS the strip. A "
        "positive N reshapes a TILED task's grid to N x N of 8 m cells; the "
        "play default of 3 gives 24 m, which a 5 m/s bounder clears in about "
        "7 s and then falls off the edge (measured 2026-08-07, three takes).",
    )
    parser.add_argument(
        "--rough-only", action="store_true",
        help="tiled tasks only: drop the pyramid sub-terrain, whose 2 m "
        "platform is the flat plateau the forward policy kept being filmed on",
    )
    parser.add_argument(
        "--center-spawn", action="store_true",
        help="tiled tasks only: start mid-grid instead of on the seed's random "
        "tile, which can be a grid edge two seconds from the flat border. The "
        "demo task pins its own spawn in the env cfg and needs neither.",
    )
    parser.add_argument(
        "--strip-m", type=float, default=0.0,
        help="demo task only: override the strip's length along +x, and move "
        "the spawn to stay mid-pad. THE LENGTH MUST MATCH THE SUBJECT'S SPEED "
        "— the strip exists to be crossed on camera, and a policy commanded at "
        "0.6 m/s needs 130 s to cross the 78 m default while the forward "
        "diagnostic's 5 m/s crosses it in 16.",
    )
    parser.add_argument(
        "--cam-mode", choices=("chase", "side"), default="chase",
        help="chase: behind-and-above, looking the way the machine is going. "
        "side: a level tracking shot from -y, in which +x travel reads as "
        "left-to-right across frame.",
    )
    parser.add_argument(
        "--vx", type=float, default=0.4,
        help="commanded forward speed. The forward-diagnostic policy does not "
        "read the command at all (its reward is base-frame v_x alone), so this "
        "matters only when filming a command-following checkpoint.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # A camera sensor IS the render path here, so cameras are never optional
    # and a viewport is never wanted.
    args.enable_cameras = True
    args.visualizer = ["none"]
    # Only the log-noise flag, matching play_spyder: the fabric scene delegate
    # must stay ON or this robot is not drawn at all (measured on the Hound).
    _extra = "--/log/channels/omni.physx.fabric.plugin=error"
    args.kit_args = f"{args.kit_args} {_extra}" if getattr(args, "kit_args", None) else _extra
    return args


def main(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import quat_apply, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    from bestiary.isaac import tasks

    tasks.register()

    # The cfg sequence below is `play_spyder`'s, deliberately step for step:
    # it is the one boot measured to work on this install, and every deviation
    # tried here cost a take. The ONLY additions are the camera sensor and the
    # wider terrain.
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = args.seed
    # Filming should end when the take ends or the machine falls, never on a
    # mid-shot timeout teleport.
    env_cfg.episode_length_s = 3600.0
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.rel_standing_envs = 0.0  # a standing-flagged env has its command zeroed every step
    # One robot on a stable patch: no terrain-level shuffling between respawns.
    # Load-bearing here, not hygiene — left on over the resized grid below it
    # took the run down with a device-side assert in GpuArticulationView
    # (measured 2026-08-07, this file's first boot).
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None

    # -- Room to run. The play grid is 3x3 tiles of 8 m; at the forward
    # policy's measured 4-5.4 m/s that is ~7 s before the south edge and a
    # 20 m fall. Widening the generator is the whole fix: same sub-terrains,
    # same difficulty sampling, more of it.
    if args.strip_m > 0.0:
        gen = env_cfg.scene.terrain.terrain_generator
        if "demo_ramp" not in gen.sub_terrains:
            raise SystemExit(
                f"--strip-m applies to the demo task's single ramp tile, but "
                f"{args.task} has sub-terrains {sorted(gen.sub_terrains)}. "
                "Use --tiles for a tiled task."
            )
        gen.size = (args.strip_m, gen.size[1])
        # Keep the spawn mid-pad: the pad is the first `pad_frac` of the strip
        # measured from -x, so its middle sits at half that from the near end.
        pad = gen.sub_terrains["demo_ramp"].pad_frac
        spawn_x = -args.strip_m / 2.0 + args.strip_m * pad / 2.0
        env_cfg.events.reset_base.params["pose_range"]["x"] = (spawn_x, spawn_x)
        print(f"[bestiary] strip {args.strip_m:.1f} m, spawn x {spawn_x:+.1f} m", flush=True)

    if args.tiles > 0:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.num_rows = args.tiles
        gen.num_cols = args.tiles
        if args.rough_only and "isaac_slope" in gen.sub_terrains:
            del gen.sub_terrains["isaac_slope"]
            # Proportions are normalised by the generator, so the remaining two
            # simply split the grid; no renormalisation needed here.

    # -- The camera. A sensor, not the viewport: its render product is bound
    # to this prim and nothing else on the stage can be confused for it.
    # PER-ENV, not a single `/World/film_cam`. A sensor living in the
    # InteractiveScene is reset with the scene's env ids, so a one-instance
    # camera is indexed 0..8 against a length-1 timestamp buffer and the run
    # dies in an ATen IndexKernel device-side assert before the first frame
    # (measured 2026-08-07: threads 1-8 assert, thread 0 does not — the
    # signature of exactly this off-by-num_envs). Nine render products cost
    # more than one; a boot that never reaches a frame costs the whole take.
    env_cfg.scene.film_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/film_cam",
        update_period=0.0,
        width=args.width,
        height=args.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 24 mm on a 20.955 mm aperture ~ 47 deg horizontal: wide enough to
            # hold the machine and the ground it is crossing, narrow enough
            # that the machine is not a speck. Isaac Lab's own default optics.
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:  # noqa: BLE001 -- shim; runner.load is the real check
        pass

    device = str(env.unwrapped.device)
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=device)
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=device)
    print(f"[bestiary] policy loaded from {ckpt.name}", flush=True)

    # -- One robot on screen; see NUM_ENVS. USD visibility is render-side, so
    # PhysX never notices and all nine still step identically.
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for i in range(1, env_cfg.scene.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[bestiary] solo view: {hidden} twin robots hidden from the renderer", flush=True)

    # -- Spawn in the MIDDLE of the grid. Tile assignment is random, and a
    # south-edge tile put the machine on the flat 20 m safety border within
    # two seconds — the exact "this is filmed on flat ground" failure this
    # session is fixing (measured 2026-08-07: spawn y = -60.0 on a grid whose
    # rough ground ends at -64). Centring costs nothing and makes the take
    # independent of which tile the seed happened to draw.
    if args.center_spawn:
        origins = env.unwrapped.scene.terrain.env_origins
        centre = origins.mean(dim=0)
        # Snap to the nearest real tile origin rather than the mean of them:
        # the mean can land between tiles, and tile origins are the poses the
        # terrain guarantees are on ground.
        nearest = int(torch.argmin(torch.linalg.norm(origins - centre, dim=1)))
        origins[0] = origins[nearest].clone()
        env.unwrapped.reset()
        print(
            f"[bestiary] centre spawn: env 0 origin -> "
            f"({float(origins[0, 0]):+.1f}, {float(origins[0, 1]):+.1f})",
            flush=True,
        )

    camera = env.unwrapped.scene["film_cam"]
    robot = env.unwrapped.scene["robot"]
    term = env.unwrapped.command_manager.get_term("base_velocity")
    dt = env.unwrapped.step_dt
    fps = round(1.0 / dt)
    total = int(args.seconds / dt)

    offset_b = torch.tensor([CAM_OFFSET_B], device=device, dtype=torch.float32)
    side_offset_w = torch.tensor([CAM_OFFSET_SIDE_W], device=device, dtype=torch.float32)
    enc = _Enc(Path(args.out), fps)
    guard = _MotionGuard()

    def _obs():
        got = wrapper.get_observations()
        return got[0] if isinstance(got, tuple) else got

    obs = _obs()
    sim_t, next_report, written = 0.0, 0.0, 0
    try:
        for _ in range(total):
            # Aim BEFORE stepping: the sensor renders inside the step, so a
            # pose written after it would be one frame stale in the file.
            pos = robot.data.root_pos_w[:1]
            if args.cam_mode == "chase":
                # YAW ONLY. Rotating the offset by the full orientation ties
                # the camera to the torso's roll and pitch, and this policy
                # bounds: the first take with `quat_apply(root_quat_w, ...)`
                # tumbled the shot to ground level behind a rise, filming a
                # ridge (measured 2026-08-07). Yaw keeps the camera trailing
                # the heading while staying level and above.
                eye = pos + quat_apply(yaw_quat(robot.data.root_quat_w[:1]), offset_b)
            else:
                # WORLD frame, not the robot's: a tracking shot holds its
                # bearing while the subject crosses it. Parked on -y looking
                # +y, so world +x runs left-to-right across the frame — which
                # is the direction the demo strip gets harder.
                eye = pos + side_offset_w
            target = pos.clone()
            target[:, 2] += CAM_LOOKAT_DZ
            # env_ids=[0] because only env 0's camera is ever read; the other
            # eight exist so the sensor's instance count matches the scene's.
            camera.set_world_poses_from_view(eyes=eye, targets=target, env_ids=[0])

            term.vel_command_b[:, 0] = args.vx
            term.vel_command_b[:, 1] = 0.0
            term.vel_command_b[:, 2] = 0.0
            with torch.inference_mode():
                obs, _, _, _ = wrapper.step(policy(obs))

            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.numel():
                frame = rgb[0, ..., :3].detach().cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                enc.add(frame)
                guard.observe(frame, total)
                written += 1

            sim_t += dt
            if sim_t >= next_report:
                v = robot.data.root_lin_vel_b
                p = robot.data.root_pos_w
                print(
                    f"  t={sim_t:6.1f}s  vx={float(v[0, 0]):+.2f} m/s  "
                    f"pos({float(p[0, 0]):+.1f}, {float(p[0, 1]):+.1f}, {float(p[0, 2]):+.1f})",
                    flush=True,
                )
                next_report = sim_t + 1.0
    finally:
        enc.close()

    print(f"[bestiary] wrote {written} frames at {fps} fps -> {args.out}", flush=True)
    guard.assert_alive(Path(args.out))
    return 0


if __name__ == "__main__":
    # Kit teardown drops stdout and close() exits 0 unconditionally — the same
    # two traps every entry point in this package answers with flush-and-_exit.
    import os
    import sys

    _args = parse_args()
    _app = AppLauncher(_args).app
    try:
        _code = main(_args)
    finally:
        _app.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
