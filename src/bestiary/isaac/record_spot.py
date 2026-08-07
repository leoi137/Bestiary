"""Record (observation, action) rollouts from the Spot flat-terrain policy.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.record_spot \\
        --episodes 20 --out runs/spot_rollouts

The stage-1 data factory: NVIDIA's pretrained Spot quadruped policy walks a
flat grid world under a scripted command schedule, and every policy step is
logged as the exact (obs, action) pair the network consumed and produced. The
output is the supervised dataset for next-token imitation — see
`research/SPOT_ROLLOUTS_SPEC.md` for the format contract; this file and that
spec must change together.

WHERE THE TAP IS. `PolicyController._compute_action(obs)` receives the
already-assembled 48-dim observation and returns the raw 12-dim TorchScript
output (verified raw: no clip, no scale, `policy_controller.py` in the
isaac-sim/IsaacSim GitHub tree, retrieved 2026-08-06). Subclassing that one
method records the true interface tensors without duplicating any upstream
logic — if NVIDIA refactors `forward()`, this tap still sees exactly what the
net sees.

WHAT A RECORDING IS NOT. Nothing here is normalized, reordered, or converted.
Raw SI tensors at the policy's own rate, in the robot's own joint order
(`dof_names` is saved into every episode). Normalization statistics belong to
the *training* code and are computed from the train split only — computing
them here would leak the holdout.

EPISODES THAT FALL ARE QUARANTINED, NOT DELETED: `fell=True` episodes land in
`fallen/` — imitating a teacher's fall teaches falling, but the footage is
evidence for debugging. Holdout split is decided here, at record time, by
seed: `seed % 10 == 0` → `holdout/`, never trained on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Record Spot flat-terrain policy rollouts.")
parser.add_argument("--episodes", type=int, default=20, help="episodes to record this invocation")
parser.add_argument("--seed0", type=int, default=0, help="first episode seed; episode k uses seed0+k")
parser.add_argument(
    "--out",
    type=Path,
    default=Path("runs/spot_rollouts"),
    help="output root; train/ holdout/ fallen/ are created under it",
)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
parser.add_argument(
    "--policy-path",
    type=str,
    default=None,
    help="TorchScript policy to record from; default is the sample-asset teacher "
    "named in research/SPOT_ROLLOUTS_SPEC.md, and the path used is saved into "
    "every episode's metadata either way",
)
parser.add_argument("--env-config", type=str, default=None, help="env yaml matching --policy-path")
args, _unknown = parser.parse_known_args()

# Kit must boot before any isaacsim.* import below; headless — this is a data
# factory, not a viewer (play_spyder is the eyes; this is the tape recorder).
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

#: The trained command distribution's edges, verbatim from the policy's own
#: env config (spot_env.yaml on the NVIDIA asset server, fetched 2026-08-06:
#: lin_vel_x [-2.0, 3.0] m/s, lin_vel_y [-1.5, 1.5] m/s, ang_vel_z
#: [-2.0, 2.0] rad/s). Commands are sampled INSIDE these; outside them the
#: policy is being asked a question it never trained on (play_spyder's rule).
VX_RANGE = (-2.0, 3.0)
VY_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)

#: Interface contract, from the SpotFlatTerrainPolicy source (GitHub,
#: retrieved 2026-08-06). Asserted against the live objects at startup —
#: a mismatch means NVIDIA shipped a different policy than the one this
#: recorder and the spec were written against, and the tape would be garbage.
OBS_DIM = 48
ACT_DIM = 12
EXPECTED_DT_S = 0.002       # spot_env.yaml sim dt
EXPECTED_DECIMATION = 10    # spot_env.yaml → policy steps at 50 Hz

#: A Spot that has fallen: torso below this height. Default stand is ~0.55 m
#: (spawn at 0.8 m settles to stance); 0.3 m is unambiguous collapse.
FALL_HEIGHT_M = 0.3

#: Command schedule shape per episode (seeded, reproducible): a stand phase,
#: then 2-4 driving phases of 2-4 s each, then a stop phase. One phase in
#: every episode is forced pure-forward — stage 1's target behaviour must
#: appear in every tape, not merely with sampling luck.
STAND_S = 1.0
PHASES = (2, 4)
PHASE_S = (2.0, 4.0)
STOP_S = 1.0


def phase_schedule(rng: "np.random.Generator") -> list[tuple[float, tuple[float, float, float]]]:
    """The episode's command script: [(duration_s, (vx, vy, wz)), ...]."""
    phases: list[tuple[float, tuple[float, float, float]]] = [(STAND_S, (0.0, 0.0, 0.0))]
    n = int(rng.integers(PHASES[0], PHASES[1] + 1))
    forced_forward = int(rng.integers(0, n))  # which driving phase is pure +vx
    for k in range(n):
        dur = float(rng.uniform(*PHASE_S))
        if k == forced_forward:
            cmd = (float(rng.uniform(0.5, VX_RANGE[1])), 0.0, 0.0)
        else:
            cmd = (
                float(rng.uniform(*VX_RANGE)),
                float(rng.uniform(*VY_RANGE)),
                float(rng.uniform(*WZ_RANGE)),
            )
        phases.append((dur, cmd))
    phases.append((STOP_S, (0.0, 0.0, 0.0)))
    return phases


class RecordingSpotPolicy(SpotFlatTerrainPolicy):
    """The tap: record the exact tensors crossing the policy interface."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.tape_obs: list = []
        self.tape_act: list = []
        self.tape_cmd: list = []
        self.live_command = torch.zeros(3, device=args.device)

    def _compute_action(self, obs):
        action = super()._compute_action(obs)
        self.tape_obs.append(obs.detach().cpu().numpy().copy())
        self.tape_act.append(action.detach().cpu().numpy().copy())
        self.tape_cmd.append(self.live_command.detach().cpu().numpy().copy())
        return action

    def clear_tape(self) -> None:
        self.tape_obs, self.tape_act, self.tape_cmd = [], [], []


def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        sys.exit(1)

    # Flat grid world — the terrain the policy was trained for. Same scene as
    # NVIDIA's spot_standalone.py example.
    prim = define_prim("/World/Ground", "Xform")
    prim.GetReferences().AddReference(assets_root_path + "/Isaac/Environments/Grid/default_environment.usd")
    define_prim("/World/PhysicsScene", "PhysicsScene")

    SimulationManager.set_physics_sim_device(args.device)

    spot = RecordingSpotPolicy(
        prim_path="/World/Spot",
        position=[0, 0, 0.8],
        policy_path=args.policy_path,
        env_config_path=args.env_config,
    )

    # The policy's env config is the authority on rates; the sim must run at
    # ITS dt, not a convenient one. (NVIDIA's own standalone example boots
    # 200 Hz against a 500 Hz-trained policy; we do not inherit that.)
    dt = float(spot._dt)
    decimation = int(spot._decimation)
    if not math.isclose(dt, EXPECTED_DT_S, rel_tol=1e-6) or decimation != EXPECTED_DECIMATION:
        raise RuntimeError(
            f"spot_env.yaml rates moved: dt={dt}, decimation={decimation}; "
            f"recorder and spec were written for dt={EXPECTED_DT_S}, decimation={EXPECTED_DECIMATION}. "
            "Re-verify the obs/action contract before recording."
        )
    SimulationManager.set_physics_dt(dt)
    # Rendering is pure overhead here — nothing visual is recorded — but kit
    # still needs render ticks to pump the update loop. At render == policy
    # rate (50 Hz) the 2080 recorded ~49 s of wall per ~11 s episode (measured
    # 2026-08-07, aborted first run); 5 Hz renders are the cheap fix. Episode
    # timing below counts PHYSICS steps, so this knob cannot skew durations.
    RenderingManager.set_dt(0.2)

    state = {"first_step": True, "cmd": torch.zeros(3, device=args.device)}

    def on_physics_step(step_size: float, context: object) -> None:
        if state["first_step"]:
            spot.initialize()
            state["first_step"] = False
        else:
            spot.live_command = state["cmd"]
            spot.forward(step_size, state["cmd"])

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    def advance(phys_steps: int) -> bool:
        """Advance exactly phys_steps physics steps; True if Spot fell.

        `_policy_counter` increments once per physics step inside
        `forward()`, so it is the sim clock — update() batches whatever the
        render cadence dictates and this stays exact either way.
        """
        target = spot._policy_counter + phys_steps
        while spot._policy_counter < target:
            simulation_app.update()
            # warp arrays, shape (1, 3) — measured in the 2026-08-06 smoke run
            height = float(spot.robot.get_world_poses()[0].numpy()[0][2])
            if height < FALL_HEIGHT_M:
                return True
        return False

    for sub in ("train", "holdout", "fallen"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        rng = np.random.default_rng(seed)

        # Reset: new pose, default joints via initialize() on the next physics
        # step, blank action memory, blank tape.
        yaw = float(rng.uniform(-math.pi, math.pi))
        spot.robot.set_world_poses(
            positions=[[0.0, 0.0, 0.8]],
            orientations=[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
        )
        spot._previous_action = None
        spot._policy_counter = 0
        state["first_step"] = True
        state["cmd"] = torch.zeros(3, device=args.device)
        advance(int(STAND_S / dt))
        spot.clear_tape()

        fell = False
        for dur, cmd in phase_schedule(rng):
            state["cmd"] = torch.tensor(cmd, device=args.device, dtype=torch.float32)
            if advance(int(dur / dt)):
                fell = True
                break

        obs = np.asarray(spot.tape_obs, dtype=np.float32)
        act = np.asarray(spot.tape_act, dtype=np.float32)
        cmd_log = np.asarray(spot.tape_cmd, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM or act.shape[1] != ACT_DIM:
            raise RuntimeError(
                f"tape shape wrong: obs {obs.shape}, act {act.shape}; "
                f"expected (*, {OBS_DIM}) and (*, {ACT_DIM}) — the policy interface moved."
            )

        sub = "fallen" if fell else ("holdout" if seed % 10 == 0 else "train")
        path = args.out / sub / f"ep_{seed:05d}.npz"
        np.savez_compressed(
            path,
            obs=obs,
            act=act,
            cmd=cmd_log,
            meta=json.dumps(
                {
                    "seed": seed,
                    "fell": fell,
                    "dt_s": dt,
                    "decimation": decimation,
                    "policy_rate_hz": 1.0 / (dt * decimation),
                    "action_scale": float(spot._action_scale),
                    "dof_names": list(spot.robot.dof_names),
                    "engine": SimulationManager.get_active_physics_engine(),
                    "policy_path": args.policy_path
                    or assets_root_path
                    + "/Isaac/Samples/Policies/Spot_Policies/"
                    + (
                        "newton_policy.pt"
                        if SimulationManager.get_active_physics_engine() == "newton"
                        else "spot_policy.pt"
                    ),
                    "spec": "research/SPOT_ROLLOUTS_SPEC.md",
                }
            ),
        )
        print(f"[{ep + 1}/{args.episodes}] seed={seed} steps={len(obs)} fell={fell} -> {path}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
