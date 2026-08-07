"""Drive ONE trained Spyder with the keyboard, FPS layout: W/S forward/back,
A/D side-step, Q/E turn, space to stop, x to quit.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.play_spyder \\
        --checkpoint runs/spyder_gentle_s1/box_logs/2026-08-06_07-53-39/model_1499.pt

KEYS (type in the terminal that launched this; every accepted key echoes):

    w / s / UP / DOWN      forward speed  +-0.1 m/s per press
    a / d / LEFT / RIGHT   side-step      +-0.1 m/s per press (a = left)
    q / e                  turn rate      +-0.2 rad/s per press (q = left)
    SPACE                  full stop (0, 0, 0)
    r                      respawn (a flipped machine needs this, not throttle)
    x                      quit

This is the WASD interface the whole Spyder track was aimed at: the policy was
trained to make its body velocity equal a 3-number command, and this file does
nothing but let the keyboard write those numbers into the command buffer every
step. The policy cannot tell this from the training sampler.

WHAT THE COMMAND INJECTION IS, PRECISELY. The env's own sampler is pinned out
of the way (resampling 1e6 s, standing fraction 0), and each step this file
writes (vx, vy, wz) into `command_manager.get_term("base_velocity")`'s buffer.
Injection, never `reset()`: a reset teleports the robot to spawn, which is
exactly wrong for driving. `rel_standing_envs = 0` is load-bearing, not
hygiene — the Play cfg inherits 0.1, and a standing-flagged env has its
command ZEROED by `_update_command` every step, which would silently eat every
keystroke with probability 0.1 per session.

HOW TO SEE IT, on this machine, measured by the Hound sessions
--------------------------------------------------------------
  * LIVE WINDOW: the default here is `--visualizer kit` — the full viewport.
    Heavy on this desktop but it animates. The lightweight Newton viewer does
    NOT animate policy playback on this install (`hound_visual.py`'s
    measurement) and is therefore not the default despite being the default
    everywhere else in this package.
  * VIDEO (`--video`): offscreen RTX render to MP4, no window involved —
    the path that demonstrably produced seed 1's first watchable clip.
    Mutually exclusive with the live viewer (measured on the Hound: Newton
    fails to configure under a headless camera boot and takes the run down).
  * `--script "w:3,a:2,space:1,s:3"` replaces the keyboard with a timed key
    sequence — the no-tty smoke test, and the way a repeatable demo clip is
    recorded: `--script ... --video`.

Nothing seen here enters the record. The per-cell grid (eval battery) is the
instrument; this is the eyes and hands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

#: Nine robots, matching the first clip that demonstrably RENDERED on this
#: install. Every env receives the same injected command — nine machines
#: obeying one keyboard — and telemetry reports env 0. (With num_envs = 1 the
#: offscreen renderer drew terrain and no robot across three camera setups;
#: unexplained, recorded in STATE, sidestepped by using the proven count.)
NUM_ENVS = 9

#: Command limits for the keyboard, from the TRAINED distribution's edges
#: (spyder_gentle_env_cfg: |v_x| <= 0.6, |w_z| <= 0.8). The keyboard may ask
#: for anything inside the trained envelope; outside it the policy is being
#: asked a question it never trained on.
VX_LIMITS = (-0.6, 0.6)
WZ_LIMITS = (-0.8, 0.8)
#: Strafe envelope. Zero for the gentle task (it never commanded v_y); the
#: ladder and overnight tasks trained on +/-0.4 m/s (spyder_ladder_env_cfg).
#: Asking a gentle-era checkpoint to strafe is asking outside its training.
VY_LIMITS = (-0.4, 0.4)

#: Sampler exile: with resampling pinned this far out, nothing overwrites the
#: injected command. Same constant eval_hound uses, same reason.
NO_RESAMPLE_S = 1_000_000.0


class KeyboardDriver:
    """Terminal keys -> a live velocity command.

    cbreak mode, zero-timeout select, restored on exit. Same contract as the
    Hound's driver (another session's uncommitted file — the idea is shared,
    the code is deliberately self-contained so this file survives a clone).
    """

    VX_STEP = 0.1
    VY_STEP = 0.1
    WZ_STEP = 0.2

    def __init__(self, vx_limits: tuple, vy_limits: tuple, wz_limits: tuple):
        import sys

        if not sys.stdin.isatty():
            raise SystemExit(
                "driving needs an interactive terminal (stdin is not a tty). "
                "For a headless run use --script."
            )
        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self.quit = False
        self.reset_requested = False
        self._vx_lo, self._vx_hi = vx_limits
        self._vy_lo, self._vy_hi = vy_limits
        self._wz_lo, self._wz_hi = wz_limits
        self._fd = sys.stdin.fileno()

        import termios
        import tty

        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self) -> None:
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self) -> None:
        import os
        import select

        changed = False
        while select.select([self._fd], [], [], 0)[0]:
            ch = os.read(self._fd, 1)
            if ch == b"\x1b":  # arrows arrive as ESC [ A/B/C/D
                seq = os.read(self._fd, 2) if select.select([self._fd], [], [], 0)[0] else b""
                ch = {b"[A": b"w", b"[B": b"s", b"[D": b"a", b"[C": b"d"}.get(seq, b"")
            if ch == b"w":
                self.vx = min(self._vx_hi, round(self.vx + self.VX_STEP, 3))
                changed = True
            elif ch == b"s":
                self.vx = max(self._vx_lo, round(self.vx - self.VX_STEP, 3))
                changed = True
            # FPS layout: A/D side-step (+y is LEFT in the base frame, so A
            # increments), Q/E turn. Quit moved q -> x when Q became turn-left.
            elif ch == b"a":
                self.vy = min(self._vy_hi, round(self.vy + self.VY_STEP, 3))
                changed = True
            elif ch == b"d":
                self.vy = max(self._vy_lo, round(self.vy - self.VY_STEP, 3))
                changed = True
            elif ch == b"q":
                self.wz = min(self._wz_hi, round(self.wz + self.WZ_STEP, 3))
                changed = True
            elif ch == b"e":
                self.wz = max(self._wz_lo, round(self.wz - self.WZ_STEP, 3))
                changed = True
            elif ch == b" ":
                self.vx = self.vy = self.wz = 0.0
                changed = True
            elif ch == b"r":
                self.reset_requested = True
            elif ch == b"x":
                self.quit = True
        if changed:
            print(
                f"  >>> command  vx={self.vx:+.1f} m/s  side={self.vy:+.1f} m/s  turn={self.wz:+.1f} rad/s",
                flush=True,
            )


class ScriptDriver:
    """`--script "w:3,a:2,space:1"` -> the same interface, no tty needed.

    Each entry is key:seconds — the key is pressed once, then held for that
    long. This is the smoke test that proves env + checkpoint + injection end
    to end on a headless box, and the way a repeatable demo video is made.
    """

    def __init__(self, script: str, vx_limits: tuple, vy_limits: tuple, wz_limits: tuple):
        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self.quit = False
        self.reset_requested = False
        self._vx_lo, self._vx_hi = vx_limits
        self._vy_lo, self._vy_hi = vy_limits
        self._wz_lo, self._wz_hi = wz_limits
        self._steps_left = 0.0
        self._queue = []
        for item in script.split(","):
            key, _, dur = item.strip().partition(":")
            self._queue.append((key.strip(), float(dur or 1.0)))

    def restore(self) -> None:
        pass

    def _press(self, key: str) -> None:
        if key == "w":
            self.vx = min(self._vx_hi, round(self.vx + KeyboardDriver.VX_STEP, 3))
        elif key == "s":
            self.vx = max(self._vx_lo, round(self.vx - KeyboardDriver.VX_STEP, 3))
        elif key == "a":
            self.vy = min(self._vy_hi, round(self.vy + KeyboardDriver.VY_STEP, 3))
        elif key == "d":
            self.vy = max(self._vy_lo, round(self.vy - KeyboardDriver.VY_STEP, 3))
        elif key == "q":
            self.wz = min(self._wz_hi, round(self.wz + KeyboardDriver.WZ_STEP, 3))
        elif key == "e":
            self.wz = max(self._wz_lo, round(self.wz - KeyboardDriver.WZ_STEP, 3))
        elif key == "space":
            self.vx = self.vy = self.wz = 0.0
        elif key == "r":
            self.reset_requested = True
        else:
            raise SystemExit(
                f"--script key {key!r} is not one of w/s (drive), a/d (side-step), "
                "q/e (turn), space, r. NOTE: a/d changed from turn to side-step and "
                "q/e took over turning when the strafe channel went live — old "
                "scripts written for the turn-on-a/d layout mean something else now."
            )
        print(
            f"  >>> script   {key:<5} -> vx={self.vx:+.1f} m/s  side={self.vy:+.1f} m/s  turn={self.wz:+.1f} rad/s",
            flush=True,
        )

    def tick(self, dt: float) -> None:
        """Advance the script clock by one policy step."""
        self._steps_left -= dt
        if self._steps_left <= 0.0:
            if not self._queue:
                self.quit = True
                return
            key, dur = self._queue.pop(0)
            self._press(key)
            self._steps_left = dur

    def poll(self) -> None:  # same surface as KeyboardDriver
        pass


def _aim_camera(env, eye, target) -> None:
    """Point the render camera at `target` from `eye`, world frame, +Z up.

    Three mechanisms exist on this install and two of them are silent no-ops
    on the offscreen --video path (both measured, 2026-08-06, each costing a
    take of empty terrain): `viewer.origin_type = "asset_root"` never updates
    without a GUI, and `sim.set_camera_view` moves a viewport camera the
    offscreen render product does not read. Writing the camera prim's USD
    transform is the one that works everywhere, so the working paths are
    tried in order: viewport controller (live viewer), then the prim write.
    pxr import stays inside the function — module import must remain pre-app
    safe (see commands.py).
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if not hasattr(_aim_camera, "_cams"):
        # Every camera prim on the stage, because the offscreen render
        # product's camera is none of the documented ones: viewer cfg,
        # set_camera_view and /OmniverseKit_Persp writes were each measured
        # as no-ops on the recorded frames (2026-08-06, three takes).
        _aim_camera._cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
        print(
            f"[bestiary] follow: driving {len(_aim_camera._cams)} stage cameras: "
            f"{[str(p.GetPath()) for p in _aim_camera._cams]}",
            flush=True,
        )
    view = Gf.Matrix4d()
    view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    cam_to_world = view.GetInverse()
    for prim in _aim_camera._cams:
        if not prim.IsValid():
            continue
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
            op = ops[0]
        else:
            xf.ClearXformOpOrder()
            op = xf.AddTransformOp()
        op.Set(cam_to_world)
    # The USD write above is invisible to the RTX pass when the fabric scene
    # delegate owns the render transforms (this robot needs fabric ON to be
    # drawn at all — see the boot comment). Mirror the pose into Fabric via
    # USDRT so the renderer sees it too.
    try:
        import usdrt
        from usdrt import Rt

        rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        rot = cam_to_world.ExtractRotationQuat()
        pos = usdrt.Gf.Vec3d(*eye)
        quat = usdrt.Gf.Quatf(
            float(rot.GetReal()), *[float(v) for v in rot.GetImaginary()]
        )
        for prim in _aim_camera._cams:
            rt_prim = rt_stage.GetPrimAtPath(str(prim.GetPath()))
            if not rt_prim.IsValid():
                continue
            rt_xf = Rt.Xformable(rt_prim)
            rt_xf.GetWorldPositionAttr().Set(pos)
            rt_xf.GetWorldOrientationAttr().Set(quat)
    except Exception as exc:  # surfaced once, not per frame — 50 Hz spam
        if not getattr(_aim_camera, "_rt_warned", False):
            _aim_camera._rt_warned = True
            print(f"[bestiary] follow: USDRT mirror failed: {exc!r}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to a model_*.pt, e.g. runs/spyder_gentle_s1/box_logs/<run>/model_1499.pt",
    )
    parser.add_argument("--task", type=str, default="Bestiary-Gentle-Spyder-Play-v0")
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument(
        "--show_all", action="store_true",
        help="render every env's robot; default hides all but env 0 so the "
        "screen shows ONE machine (the physics twins keep running unseen)",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--script",
        type=str,
        default="",
        help='timed key sequence replacing the keyboard, e.g. "w:3,q:2,a:2,space:1,s:3" (w/s drive, a/d side-step, q/e turn)',
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="record to this MP4 path (offscreen RTX; disables the live viewer)",
    )
    parser.add_argument(
        "--max_seconds", type=float, default=0.0,
        help="stop after this many sim seconds (0 = run until q / script end)",
    )
    parser.add_argument(
        "--follow", action="store_true",
        help="camera tracks the robot instead of the fixed world view — "
        "required for policies that cover ground (a 3 m/s runner exits the "
        "fixed frame in ~5 s)",
    )
    parser.add_argument(
        "--cam",
        type=str,
        default="",
        help="fixed camera override 'ex,ey,ez,lx,ly,lz' (world frame, m). "
        "The boot-time viewer pose is the ONE camera control the offscreen "
        "renderer honours (asset_root, set_camera_view, USD and USDRT prim "
        "writes are all measured no-ops there), so filming a fast policy "
        "means parking this beside its known path.",
    )
    AppLauncher.add_app_launcher_args(parser)
    # A live window is the point, and on this machine that means the KIT
    # viewport: the lightweight Newton viewer does not animate policy playback
    # here (hound_visual.py's measurement), so defaulting to it would produce
    # a clean-looking run and an empty screen — tonight's exact failure.
    parser.set_defaults(visualizer=["kit"])
    args = parser.parse_args()
    if args.video:
        # Offscreen recording and the live viewer are alternatives, measured
        # on the Hound: a headless camera boot breaks the viewer's configure.
        args.visualizer = ["none"]
        args.enable_cameras = True
    # DEFAULT BOOT, deliberately — the opposite of play_hound's. That file
    # disables the fabric scene delegate and animates a physics-free ghost,
    # because with fabric ON its captures froze and with fabric OFF the
    # renderer never sees robot poses. The Spyder measured differently on
    # this same install: the 9-env RecordVideo clip (default boot) animated
    # correctly, and this file's first recording with the fsd flag copied
    # from play_hound rendered terrain with NO robot at all. Only the
    # log-noise flag is kept.
    _extra = "--/log/channels/omni.physx.fabric.plugin=error"
    args.kit_args = f"{args.kit_args} {_extra}" if getattr(args, "kit_args", None) else _extra
    return args


def main(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    from bestiary.isaac import tasks

    tasks.register()

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = args.seed
    # An hour per episode: driving should end when the driver quits or the
    # machine falls (base contact), never on a mid-drive timeout teleport.
    env_cfg.episode_length_s = 3600.0
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (NO_RESAMPLE_S, NO_RESAMPLE_S)
    cmd.rel_standing_envs = 0.0  # see the module docstring — this eats keystrokes otherwise
    # One robot on a stable patch: no terrain-level shuffling between respawns.
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None

    # PIN THE CAMERA, every run. Kit persists the viewport pose in its user
    # config, so one bad boot (a failed 1-env probe saved a sky-pointing
    # camera) poisons every later recording that trusts the default. World
    # frame, high enough to hold the 3x3 play grid, aimed at its centre.
    # --follow re-aims the camera at the robot EVERY FRAME inside the loop
    # below, via sim.set_camera_view. The declarative route
    # (viewer.origin_type = "asset_root") was measured a no-op on the
    # offscreen --video path on this install — the viewport controller that
    # honours it never updates without a GUI, and the take came out as 20 s
    # of empty terrain. A policy holding >1 m/s leaves the fixed world view
    # inside 5 s, so fast policies need the flag.
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (9.0, 9.0, 5.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
    if args.cam:
        vals = [float(x) for x in args.cam.split(",")]
        if len(vals) != 6:
            raise SystemExit(f"--cam needs 6 comma-separated floats, got {len(vals)}: {args.cam!r}")
        env_cfg.viewer.eye = tuple(vals[:3])
        env_cfg.viewer.lookat = tuple(vals[3:])

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)

    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        from importlib import metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:  # noqa: BLE001 -- shim, absent in some versions; runner.load is the real check
        pass
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=str(env.unwrapped.device))
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=str(env.unwrapped.device))
    print(f"[bestiary] policy loaded from {ckpt.name}", flush=True)

    recorder = None
    if args.video:
        import subprocess

        import numpy as np

        class _Rec:
            """env.render() frames -> ffmpeg. gym's RecordVideo froze on the
            Hound path; this pipe was the fix and it owns nothing clever."""

            def __init__(self, path, fps):
                self.path, self._fps, self._proc = path, int(fps), None

            def add(self, frame):
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
                arr = np.ascontiguousarray(arr[..., :3])
                if self._proc is None:
                    h, w = arr.shape[:2]
                    self._proc = subprocess.Popen(
                        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{w}x{h}", "-r", str(self._fps), "-i", "-",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(self.path)],
                        stdin=subprocess.PIPE,
                    )
                self._proc.stdin.write(arr.tobytes())

            def close(self):
                if self._proc is not None:
                    self._proc.stdin.close()
                    self._proc.wait()

        recorder = _Rec(args.video, round(1.0 / env.unwrapped.step_dt))

    # ONE robot on screen. num_envs=1 renders no robot at all on this install
    # (terrain only — three camera setups and a replicate_physics probe all
    # failed), so the working configuration is kept and the extra eight are
    # made invisible to the RENDERER only: USD visibility is a render-side
    # attribute, not an xformOp, so PhysX never notices. All nine still obey
    # the injected command; telemetry reports the one you can see.
    if not args.show_all:
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

    driver = (
        ScriptDriver(args.script, VX_LIMITS, VY_LIMITS, WZ_LIMITS)
        if args.script
        else KeyboardDriver(VX_LIMITS, VY_LIMITS, WZ_LIMITS)
    )
    term = env.unwrapped.command_manager.get_term("base_velocity")
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt

    if not args.script:
        print(__doc__.split("KEYS", 1)[1].split("WHAT THE", 1)[0], flush=True)

    def _obs():
        # rsl-rl 5.x returns the obs alone; older wrappers returned (obs, extras).
        got = wrapper.get_observations()
        return got[0] if isinstance(got, tuple) else got

    obs = _obs()
    sim_t, next_report = 0.0, 0.0
    try:
        while not driver.quit:
            driver.poll()
            if isinstance(driver, ScriptDriver):
                driver.tick(dt)
            if driver.reset_requested:
                driver.reset_requested = False
                env.unwrapped.reset()
                obs = _obs()
                print("  >>> respawned", flush=True)
            # The injection. vy went live with the ladder/overnight tasks
            # (trained on +/-0.4 m/s); a gentle-era checkpoint fed a nonzero
            # vy is being asked outside its training and will half-ignore it.
            term.vel_command_b[:, 0] = driver.vx
            term.vel_command_b[:, 1] = driver.vy
            term.vel_command_b[:, 2] = driver.wz
            with torch.inference_mode():
                obs, _, _, _ = wrapper.step(policy(obs))
            if recorder is not None:
                if args.follow:
                    p = robot.data.root_pos_w.torch if hasattr(robot.data.root_pos_w, "torch") else robot.data.root_pos_w
                    px, py, pz = float(p[0, 0]), float(p[0, 1]), float(p[0, 2])
                    _aim_camera(env.unwrapped, (px + 2.6, py + 2.6, pz + 1.6), (px, py, pz + 0.2))
                frame = env.render()
                if frame is not None:
                    recorder.add(frame)
            sim_t += dt
            if sim_t >= next_report:
                v = robot.data.root_lin_vel_b.torch if hasattr(robot.data.root_lin_vel_b, "torch") else robot.data.root_lin_vel_b
                w = robot.data.root_ang_vel_b.torch if hasattr(robot.data.root_ang_vel_b, "torch") else robot.data.root_ang_vel_b
                pw = robot.data.root_pos_w.torch if hasattr(robot.data.root_pos_w, "torch") else robot.data.root_pos_w
                print(
                    f"  t={sim_t:6.1f}s  cmd(vx={driver.vx:+.2f}, wz={driver.wz:+.2f})  "
                    f"achieved(vx={float(v[0, 0]):+.2f} m/s, wz={float(w[0, 2]):+.2f} rad/s)  "
                    f"pos({float(pw[0, 0]):+.1f}, {float(pw[0, 1]):+.1f}, {float(pw[0, 2]):+.1f})",
                    flush=True,
                )
                next_report = sim_t + 1.0
            if args.max_seconds and sim_t >= args.max_seconds:
                break
    finally:
        driver.restore()
        if recorder is not None:
            recorder.close()
            print(f"[bestiary] video written: {args.video}", flush=True)
    return 0


if __name__ == "__main__":
    # Kit teardown drops stdout and close() exits 0 unconditionally — same two
    # traps, same flush-and-_exit answer as every entry point in this package.
    import os
    import sys
    import traceback

    _args = parse_args()
    AppLauncher(_args)
    try:
        code = main(_args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
