"""Is the heading term hard to satisfy on this ground, or impossible?

    venv/bin/python -m research.scripts.heading_ceiling --run hound_track_desert_s0
    venv/bin/python -m research.scripts.heading_ceiling --run ... --episodes 12 --json

WHY THIS EXISTS

`learnings/011` established that `hound_track_desert_s0` loses to standing on
the control cost, not on crashing, and that the mechanism is the heading factor:
the reward is `Phi(u_v) * Phi(u_w)`, the policy's speed factor improves and its
heading factor collapses, and the product cancels. Standing holds heading for
free because a stationary body has yaw_rate ~ 0; driving on a desert does not.

That leaves one question, and the whole next decision turns on it:

    Is sigma_w = 0.10 rad/s (5.7 deg/s) ACHIEVABLE while driving on this
    terrain, or is it asking for something the ground will not permit?

The two answers point opposite ways and cost very differently:

  * ACHIEVABLE -> the objective is fine and the policy is undertrained.
    `ep_rew_mean` was 43.24 and still climbing at 1.5M steps, so this is live.
    The fix is more steps / better exploration, and retuning coefficients would
    be tuning away a real signal.

  * NOT ACHIEVABLE -> Phi_w is capped near 0.5 no matter what, the product
    caps with it, and no reward rebalance lifts it. The fix is structural:
    widen sigma_w, or stop multiplying the two factors so a heading failure
    stops annihilating a perfect speed match.

Choosing coefficients before knowing which one is true is guessing, and it is
the kind of guess that costs a whole training window to discover.

THE MEASUREMENT

Terrain roughness is the only variable. The model, the robot, the reward, the
policy and the command are identical across arms; the heightfield's ELEVATION
SCALE (`model.hfield_size[hid, 2]`) is multiplied by alpha in [0, 1].

Scaling the elevation rather than swapping in the flat XML is deliberate. The
observation includes terrain samples and its width is derived from whether the
model has a heightfield at all, so `hound16pd.xml` would change the observation
width and the checkpoint would not load -- `learnings/003`, the one-way door.
Scaling keeps the observation identical in shape and honest in content: at
alpha = 0.25 the policy genuinely sees a gentler desert.

alpha = 0 is the ceiling arm: same everything, flat ground. Whatever Phi_w the
policy reaches there is what the CONTROLLER can do when the terrain contributes
nothing, which is the upper bound the objective is being asked for.

READ THE CAVEAT BEFORE READING THE NUMBERS

The policy was trained at alpha = 1. Every other arm is off-distribution, so a
drop at low alpha is ambiguous (distribution shift, not terrain) while a RISE
is not -- a policy doing BETTER on ground it never saw cannot be explained by
familiarity. So this design can cleanly demonstrate "the terrain was the
binding constraint" and cannot cleanly demonstrate the opposite. That asymmetry
is why the zero-action control is measured at every alpha too: it is
policy-free, so its Phi_w curve isolates what the GROUND does to a body that is
not trying to do anything.
"""
from __future__ import annotations

import argparse
import json
import statistics

import gymnasium as gym
import numpy as np

import bestiary.envs  # noqa: F401  -- registers the env ids
from bestiary import paths
from bestiary.record.track_eval import rollout
from bestiary.terrain.field import HeightField

ENV_ID = "HoundPDTrackDesert-v0"

# Straight drives first: with w_cmd = 0 the heading factor is measuring pure
# disturbance rejection, which is exactly the quantity in question. The turning
# cell is included because a body that must yaw ON PURPOSE may find the
# tolerance easier or harder, and that difference is itself informative.
COMMANDS = {
    "drive_slow    (vx=0.30, w=0)": (0.30, 0.0, 0.0),
    "drive_mid     (vx=0.55, w=0)": (0.55, 0.0, 0.0),
    "drive_fast    (vx=0.80, w=0)": (0.80, 0.0, 0.0),
    "turn          (vx=0.55, w=0.30)": (0.55, 0.0, 0.30),
}

ALPHAS = (0.0, 0.25, 0.50, 1.00)


def scale_terrain(env, alpha: float, original_z: float, original_pos_z: float,
                  spawn_xy: tuple[float, float]) -> tuple[float, float]:
    """Scale the terrain's RELIEF while holding the ground under the spawn fixed.

    Returns (elevation span in m, ground height at spawn in m).

    The second half is not a refinement, it is the whole validity of the
    experiment. `height_at = geom_z + v * zspan`, and the terrain carries a
    flattened spawn pad whose height therefore scales with `zspan` too. The
    robot spawns at a FIXED z from `init_qpos` -- nothing in `reset_model`
    consults the ground -- so scaling alone drops the pad out from under it. The
    first version of this script did exactly that: at alpha = 0 the pad fell
    ~5 m and the hound free-fell onto flat ground, crashing 2 of 2 episodes on
    the EASIEST terrain in the sweep. The numbers looked like a finding and were
    an artefact of a spawn height nobody had held constant.

    Shifting the geom's z by what the scaling removed restores the contact
    exactly, so the only thing that varies across arms is the relief the robot
    has to drive over. Asserted below rather than trusted.
    """
    model = env.unwrapped.model
    hf = HeightField.from_model(model)
    hid, gid = hf.hid, hf.geom_id

    model.geom_pos[gid, 2] = original_pos_z
    model.hfield_size[hid, 2] = alpha * original_z

    scaled = HeightField.from_model(model)
    drift = scaled.height_at(*spawn_xy) - TARGET_SPAWN_GROUND[0]
    model.geom_pos[gid, 2] -= drift

    corrected = HeightField.from_model(model)
    ground = corrected.height_at(*spawn_xy)
    if abs(ground - TARGET_SPAWN_GROUND[0]) > 1e-9:
        raise AssertionError(
            f"spawn ground not held: wanted {TARGET_SPAWN_GROUND[0]:.9f} m, "
            f"got {ground:.9f} m at alpha={alpha}"
        )
    return float(model.hfield_size[hid, 2]), float(ground)


# Filled once from the committed terrain, before any scaling. A module-level
# one-element list rather than a global rebind so the assertion above reads
# against a value that provably came from the unmodified model.
TARGET_SPAWN_GROUND: list[float] = [0.0]


def arm(env, policy, cmd, episodes: int, seed0: int) -> dict:
    """`episodes` rollouts under one held command. Same seeds on every arm."""
    eps = [rollout(env, seed0 + i, policy=policy, forced_cmd=cmd) for i in range(episodes)]
    phi_w = [e["mean_phi_w"] for e in eps]
    phi_v = [e["mean_phi_v"] for e in eps]
    return {
        # Episode length is not decoration: Phi is averaged over the steps that
        # HAPPENED, so an arm whose episodes die early is averaged over a
        # different, earlier, and generally easier window than one that runs the
        # full horizon. Two arms are only comparable when this matches. On the
        # first full sweep the alpha=0 arm terminated 10/10 at a mean of 367-683
        # steps while alpha>=0.5 ran 1000/1000, which makes those rows
        # cross-readable only with this column in view.
        "steps_mean": float(np.mean([e["steps"] for e in eps])),
        "steps_min": int(min(e["steps"] for e in eps)),
        "phi_w": float(np.mean(phi_w)),
        "phi_w_sd": float(statistics.pstdev(phi_w)) if len(phi_w) > 1 else 0.0,
        "phi_v": float(np.mean(phi_v)),
        "track": float(np.mean([e["mean_track"] for e in eps])),
        "crashes": sum(1 for e in eps if e["terminated"]),
        "n": len(eps),
    }


def yaw_error_from_phi(phi: float, sigma: float) -> float:
    """Invert the Cauchy kernel: Phi = 1/(1+u^2)  ->  |error| = sigma*sqrt(1/Phi - 1).

    Reported because a tolerance kernel value is not physically legible, and
    the decision here is about whether a number of rad/s is attainable.
    """
    if phi <= 0.0:
        return float("inf")
    return sigma * float(np.sqrt(max(0.0, 1.0 / phi - 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run-name under runs/")
    ap.add_argument("--episodes", type=int, default=8, help="episodes per (alpha, command)")
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--latest", action="store_true",
                    help="use ant_sac.zip (the measured checkpoint) not ant_sac_best.zip")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from stable_baselines3 import SAC

    run_dir = paths.RUNS / args.run
    ckpt = run_dir / ("ant_sac.zip" if args.latest else "ant_sac_best.zip")
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    policy = SAC.load(str(ckpt), device="cpu")

    env = gym.make(ENV_ID)
    hf = HeightField.from_model(env.unwrapped.model)
    if hf is None:
        raise SystemExit(f"{ENV_ID} reports no heightfield; this experiment is meaningless")
    # Where the robot actually starts, read from the model rather than assumed
    # to be the origin, and the ground under it on the COMMITTED terrain. Every
    # arm is then held to this exact contact height.
    spawn_xy = (float(env.unwrapped.init_qpos[0]), float(env.unwrapped.init_qpos[1]))
    original_z = float(env.unwrapped.model.hfield_size[0, 2])
    original_pos_z = float(env.unwrapped.model.geom_pos[hf.geom_id, 2])
    TARGET_SPAWN_GROUND[0] = hf.height_at(*spawn_xy)
    spawn_z = float(env.unwrapped.init_qpos[2])

    sigma_w = float(env.unwrapped._sigma_w)
    sigma_v = float(env.unwrapped._sigma_v)

    results: dict = {
        "run": args.run,
        "checkpoint": ckpt.name,
        "episodes_per_cell": args.episodes,
        "seed0": args.seed0,
        "sigma_w": sigma_w,
        "sigma_v": sigma_v,
        "original_elevation_m": original_z,
        "spawn_xy": list(spawn_xy),
        "spawn_z_m": spawn_z,
        "ground_at_spawn_m": TARGET_SPAWN_GROUND[0],
        "clearance_at_spawn_m": spawn_z - TARGET_SPAWN_GROUND[0],
        "arms": [],
    }

    for alpha in ALPHAS:
        elev, ground = scale_terrain(env, alpha, original_z, original_pos_z, spawn_xy)
        for label, cmd in COMMANDS.items():
            pol = arm(env, policy, cmd, args.episodes, args.seed0)
            zero = arm(env, None, cmd, args.episodes, args.seed0)
            results["arms"].append({
                "alpha": alpha,
                "elevation_m": elev,
                "ground_at_spawn_m": ground,
                "command": label,
                "policy": pol,
                "zero_action": zero,
                "policy_yaw_err_rad_s": yaw_error_from_phi(pol["phi_w"], sigma_w),
                "zero_yaw_err_rad_s": yaw_error_from_phi(zero["phi_w"], sigma_w),
            })

    scale_terrain(env, 1.0, original_z, original_pos_z, spawn_xy)  # restore
    env.close()

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"\nheading ceiling — {args.run} / {ckpt.name}, "
          f"n={args.episodes} per cell, seeds {args.seed0}..{args.seed0 + args.episodes - 1}")
    print(f"sigma_w = {sigma_w} rad/s ({np.degrees(sigma_w):.1f} deg/s)   "
          f"committed terrain elevation = {original_z:.3f} m")
    print(f"spawn ({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f}) z={spawn_z:.3f} m, "
          f"ground {TARGET_SPAWN_GROUND[0]:.3f} m -> clearance "
          f"{spawn_z - TARGET_SPAWN_GROUND[0]:.3f} m, HELD CONSTANT across arms\n")
    hdr = (f"{'alpha':>6} {'elev(m)':>8}  {'command':<32} "
           f"{'Phi_w':>7} {'yaw err':>9} {'Phi_v':>7} {'track':>7} {'steps':>7} {'crash':>6}")
    print(hdr)
    print("-" * len(hdr))
    for a in results["arms"]:
        p = a["policy"]
        print(f"{a['alpha']:>6.2f} {a['elevation_m']:>8.3f}  {a['command']:<32} "
              f"{p['phi_w']:>7.4f} {a['policy_yaw_err_rad_s']:>8.3f}  "
              f"{p['phi_v']:>7.4f} {p['track']:>7.4f} {p['steps_mean']:>7.0f} "
              f"{p['crashes']:>3}/{p['n']}")

    print("\nzero action (policy-free — what the GROUND alone does to Phi_w):")
    print(f"{'alpha':>6}  {'command':<32} {'Phi_w':>7} {'yaw err':>9}")
    print("-" * 60)
    for a in results["arms"]:
        z = a["zero_action"]
        print(f"{a['alpha']:>6.2f}  {a['command']:<32} "
              f"{z['phi_w']:>7.4f} {a['zero_yaw_err_rad_s']:>8.3f}")


if __name__ == "__main__":
    main()
