"""Close the loop: the trained imitator drives Spot on the holdout scripts.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.play_ntp \\
        --run ntp_spot_s0 --controller ntp --episodes 12 --video-episodes 1

    # the baseline arm: the TEACHER on the byte-identical scripts
    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.play_ntp \\
        --controller teacher --episodes 12

This is checklist item 5 and the instrument that resolves prediction P16:
survival and command tracking, transformer vs teacher, on the holdout
episodes' command scripts (seeds ≡ 0 mod 10 — recorded but never trained on,
regenerated here from `spot_commands.phase_schedule`, the same function the
recorder drew from, which is what makes "identical scripts" a fact rather
than a claim).

HOW THE TRANSFORMER DRIVES. Same tap as the recorder, reversed:
`_compute_action(obs)` receives the deployed 48-dim observation; instead of
the teacher's TorchScript we normalize with the run's own `stats.json`,
append to a rolling (obs, act) history, ask the causal transformer for the
next action token, de-normalize, and hand it back. The base class applies
`default_pose + 0.2 × action` exactly as it would for the teacher — the
swap is invisible to the physics.

VIDEO. `--video-episodes N` renders the last N episodes through a chase
camera parented to Spot's body and writes PNG frames at 25 fps
(`<out>/frames_ep<seed>/`); the driver prints the ffmpeg line that turns
them into an mp4. Non-video episodes render at 5 Hz for speed (the
recorder's measured lesson).

Metrics per episode -> <out>/results.jsonl: fell, steps survived, distance,
mean |v_x - cmd_x| over driving phases (body frame, the policy's own
tracking definition). Nothing here writes to the record; the ledger row is
authored from these files after the refute pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Closed-loop eval: transformer or teacher drives Spot.")
parser.add_argument("--run", default="ntp_spot_s0", help="run dir under runs/ holding checkpoint + stats.json")
parser.add_argument("--checkpoint", default="ntp_best.pt")
parser.add_argument("--controller", choices=["ntp", "teacher"], default="ntp")
parser.add_argument("--episodes", type=int, default=12, help="holdout scripts to replay (seeds 0,10,20,...)")
parser.add_argument("--video-episodes", type=int, default=0, help="render the first N episodes to PNG frames")
parser.add_argument("--out", type=Path, default=None, help="default runs/<run>/eval_<controller>")
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
parser.add_argument("--cam-dist", type=float, default=6.0, help="chase camera distance behind the body (m)")
parser.add_argument("--cam-height", type=float, default=2.5, help="chase camera height above ground (m)")
parser.add_argument(
    "--style",
    choices=["none", "studio"],
    default="none",
    help="filming look: 'studio' = teal robot, charcoal floor, cool dome light. "
    "Pure USD material/light overrides — the physics scene is untouched",
)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import carb  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.deprecation_manager import import_module  # noqa: E402
from isaacsim.core.experimental.utils.stage import define_prim  # noqa: E402
from isaacsim.core.rendering_manager import RenderingManager  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

torch = import_module("torch")
np = import_module("numpy")

from bestiary.paths import RUNS  # noqa: E402

from .spot_commands import FALL_HEIGHT_M, STAND_S, phase_schedule  # noqa: E402

#: Video cadence. 25 fps is watchable and half the policy rate; frames are
#: captured on render ticks, so video episodes set render dt to this.
VIDEO_DT_S = 0.04
FAST_DT_S = 0.2  # non-video render tick — the recorder's measured 5 Hz lesson


class NTPController(SpotFlatTerrainPolicy):
    """SpotFlatTerrainPolicy with the brain swapped at the _compute_action tap."""

    def __init__(self, run_dir: Path, checkpoint: str, mode: str, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.mode = mode
        if mode == "ntp":
            from bestiary.ntp.model import NTPConfig, NTPModel

            payload = torch.load(run_dir / checkpoint, map_location=args.device, weights_only=False)
            self.net = NTPModel(NTPConfig(**payload["config"])).to(args.device).eval()
            self.net.load_state_dict(payload["model"])
            stats = json.loads((run_dir / "stats.json").read_text())
            as_t = lambda k: torch.tensor(stats[k], dtype=torch.float32, device=args.device)  # noqa: E731
            self.obs_mean, self.obs_std = as_t("obs_mean"), as_t("obs_std")
            self.act_mean, self.act_std = as_t("act_mean"), as_t("act_std")
            self.k = self.net.cfg.context
            self.hist_obs: list = []
            self.hist_act: list = []
            print(
                f"ntp controller: {checkpoint} step {payload['step']}, val {payload['val_loss']:.4f}",
                flush=True,
            )

    def reset_history(self) -> None:
        if self.mode == "ntp":
            self.hist_obs, self.hist_act = [], []

    def _compute_action(self, obs):
        if self.mode == "teacher":
            return super()._compute_action(obs)
        with torch.no_grad():
            self.hist_obs.append((obs.to(args.device) - self.obs_mean) / self.obs_std)
            self.hist_obs = self.hist_obs[-self.k :]
            self.hist_act = self.hist_act[-(self.k - 1) :]
            obs_h = torch.stack(self.hist_obs)[None]
            act_h = (
                torch.stack(self.hist_act)[None]
                if self.hist_act
                else torch.zeros(1, 0, 12, device=args.device)
            )
            a_norm = self.net.predict_next_action(obs_h, act_h)[0]
            self.hist_act.append(a_norm)
            return a_norm * self.act_std + self.act_mean


def main() -> None:
    run_dir = RUNS / args.run
    out = args.out or run_dir / f"eval_{args.controller}"
    out.mkdir(parents=True, exist_ok=True)

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        raise SystemExit(1)
    prim = define_prim("/World/Ground", "Xform")
    prim.GetReferences().AddReference(assets_root_path + "/Isaac/Environments/Grid/default_environment.usd")
    define_prim("/World/PhysicsScene", "PhysicsScene")
    SimulationManager.set_physics_sim_device(args.device)

    spot = NTPController(run_dir, args.checkpoint, args.controller, prim_path="/World/Spot", position=[0, 0, 0.8])

    if args.style == "studio":
        # Filming look only: material binds stronger-than-descendants override
        # the referenced assets' looks without touching a single physics prim.
        from pxr import Gf, Sdf, UsdLux, UsdShade

        stage = omni.usd.get_context().get_stage()

        def make_material(path: str, color: tuple, roughness: float, metallic: float):
            mat = UsdShade.Material.Define(stage, path)
            sh = UsdShade.Shader.Define(stage, path + "/shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            return mat

        from pxr import Usd

        teal = make_material("/World/Looks/teal_body", (0.05, 0.62, 0.68), roughness=0.35, metallic=0.3)
        blue = make_material("/World/Looks/blue_legs", (0.55, 0.78, 0.88), roughness=0.45, metallic=0.15)
        floor = make_material("/World/Looks/studio_floor", (0.07, 0.09, 0.13), roughness=0.85, metallic=0.0)

        api = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Ground"))
        api.Bind(floor, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        # The Spot asset instances its meshes, and bindings cannot reach
        # inside instances (two measured failures: a root-level bind tinted
        # only the un-instanced shins; per-mesh binds tinted nothing). So:
        # de-instance the whole robot subtree first, then bind per mesh.
        spot_root = stage.GetPrimAtPath("/World/Spot")
        for prim in Usd.PrimRange(spot_root):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(spot_root):
            if prim.GetTypeName() not in ("Mesh", "GeomSubset"):
                continue
            leggy = any(t in str(prim.GetPath()).lower() for t in ("uleg", "lleg", "hip", "leg"))
            api = UsdShade.MaterialBindingAPI.Apply(prim)
            api.Bind(blue if leggy else teal, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        dome = UsdLux.DomeLight.Define(stage, "/World/StyleDome")
        dome.CreateIntensityAttr(600.0)
        dome.CreateColorAttr(Gf.Vec3f(0.85, 0.95, 1.0))

    dt = float(spot._dt)
    SimulationManager.set_physics_dt(dt)
    RenderingManager.set_dt(FAST_DT_S)

    state = {"first_step": True, "cmd": torch.zeros(3, device=args.device)}

    def on_physics_step(step_size: float, context: object) -> None:
        if state["first_step"]:
            spot.initialize()
            state["first_step"] = False
        else:
            spot.forward(step_size, state["cmd"])

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    def pose():
        p, q = spot.robot.get_world_poses()
        p, q = p.numpy()[0], q.numpy()[0]
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        return float(p[0]), float(p[1]), float(p[2]), yaw

    # Filming is a DEDICATED run: the frame writer must attach BEFORE the
    # timeline ever plays — attaching (or detaching) while the simulation is
    # live invalidates the physics view, whatever the ordering against
    # initialize(): three variants all died with "Failed to get DOF
    # stiffnesses" / dead get_velocities on this stack (box, 2026-08-07).
    capture = {"track": None}
    if args.video_episodes:
        if args.video_episodes != args.episodes:
            raise SystemExit(
                f"--video-episodes {args.video_episodes} != --episodes {args.episodes}: "
                "filming is all-or-nothing per invocation; run metrics separately"
            )
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        # Chase camera at TOP LEVEL (never inside the robot's subtree),
        # repositioned every tick to follow the body.
        stage = omni.usd.get_context().get_stage()
        cam = UsdGeom.Camera.Define(stage, "/World/chase_cam")
        xf = UsdGeom.Xformable(cam.GetPrim())
        t_op = xf.AddTranslateOp()
        r_op = xf.AddRotateXYZOp()

        # Pitch keeps the body centered: down from horizontal by the angle
        # the body (standing ~0.55 m) subtends at this distance and height.
        cam_pitch = 90.0 - math.degrees(math.atan2(args.cam_height - 0.55, args.cam_dist))

        def track_camera() -> None:
            x, y, z, yaw = pose()
            t_op.Set(
                Gf.Vec3d(
                    x - args.cam_dist * math.cos(yaw),
                    y - args.cam_dist * math.sin(yaw),
                    args.cam_height,
                )
            )
            r_op.Set(Gf.Vec3f(cam_pitch, 0.0, math.degrees(yaw) - 90.0))

        capture["track"] = track_camera
        rp = rep.create.render_product("/World/chase_cam", (1280, 720))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(out / "frames"), rgb=True)
        writer.attach([rp])
        RenderingManager.set_dt(VIDEO_DT_S)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    results_path = out / "results.jsonl"
    results = open(results_path, "a")
    survived = 0
    for ep in range(args.episodes):
        seed = ep * 10  # holdout seeds: 0, 10, 20, ... — recorded, never trained on
        rng = np.random.default_rng(seed)
        video = capture["track"] is not None

        yaw = float(rng.uniform(-math.pi, math.pi))
        spot.robot.set_world_poses(
            positions=[[0.0, 0.0, 0.8]],
            orientations=[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        )
        spot._previous_action = None
        spot._policy_counter = 0
        spot.reset_history()
        state["first_step"] = True
        state["cmd"] = torch.zeros(3, device=args.device)

        settle_target = spot._policy_counter + int(STAND_S / dt)
        while spot._policy_counter < settle_target:
            simulation_app.update()
            if video:
                capture["track"]()
        x0, y0, _, _ = pose()

        fell = False
        err_sum, err_n = 0.0, 0
        for dur, cmd in phase_schedule(rng):
            state["cmd"] = torch.tensor(cmd, device=args.device, dtype=torch.float32)
            steps = int(dur / dt)
            t_target = spot._policy_counter + steps
            while spot._policy_counter < t_target:
                simulation_app.update()
                if video:
                    capture["track"]()
                _, _, z, _ = pose()
                if abs(cmd[0]) > 1e-6 and args.controller == "ntp" and spot.hist_obs:
                    # obs[0] is body-frame v_x, normalized; de-normalize to SI
                    vx = float(spot.hist_obs[-1][0] * spot.obs_std[0] + spot.obs_mean[0])
                    err_sum, err_n = err_sum + abs(vx - cmd[0]), err_n + 1
                if z < FALL_HEIGHT_M:
                    fell = True
                    break
            if fell:
                break

        x1, y1, _, _ = pose()
        survived += not fell
        row = {
            "controller": args.controller,
            "seed": seed,
            "fell": fell,
            "policy_steps": int(spot._policy_counter // spot._decimation),
            "distance_m": round(math.hypot(x1 - x0, y1 - y0), 3),
            "mean_abs_vx_err": round(err_sum / err_n, 4) if err_n else None,
        }
        results.write(json.dumps(row) + "\n")
        results.flush()
        print(f"[{ep + 1}/{args.episodes}] {row}", flush=True)

    results.close()
    print(f"{args.controller}: {survived}/{args.episodes} survived -> {results_path}", flush=True)
    if capture["track"] is not None:
        print(
            f"video frames: {out}/frames\n  ffmpeg -framerate 25 -pattern_type glob "
            f"-i '{out}/frames/rgb_*.png' -c:v libx264 -pix_fmt yuv420p {out}/drive.mp4",
            flush=True,
        )
    simulation_app.close()


if __name__ == "__main__":
    main()
