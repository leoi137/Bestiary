"""Measure the velocity noise floor a policy cannot remove on this terrain.

The command-tracking reward's tolerance constants are both defined as 3x a
noise floor: sigma_v = 3 * (linear ripple), sigma_omega = 3 * (yaw ripple).
The linear figure was already in the record (~0.05 m/s of passive backward
creep, `envs/hound.py`). The yaw figure was NOT — the design that needs it
carried an estimate of 0.1 rad/s and said so, flagging it as its weakest
constant.

The number rule forbids shipping that estimate. This measures it.

METHOD. Roll zero-action episodes on the run's own env and record the trunk's
free-joint velocities every step. Zero action on a PD env commands the standing
stance, so whatever velocity remains is what the heightfield injects into a
machine actively trying to hold still — which is exactly the error no tracking
policy can be expected to null.

    qvel[0:3]  trunk linear velocity   (world frame)
    qvel[3:6]  trunk angular velocity  -> qvel[5] is yaw rate

WHY 3x AND NOT 2x OR 4x. With the Cauchy kernel Phi(u) = 1/(1+u^2), an error at
the noise floor scores Phi(1/3) = 1/(1+1/9) = 0.900. So "sigma = 3 * floor"
is not a convention, it is the statement *unfixable noise costs the policy 10%
of the tracking term and no more*. Any other multiple is a different sentence
about how much the terrain is allowed to cost.

Usage:
    venv/bin/python -m bestiary.research.scripts.measure_tracking_noise   # not importable; run by path
    venv/bin/python research/scripts/measure_tracking_noise.py --episodes 20
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# Phi(1/3) — what an error exactly at the noise floor scores under the Cauchy
# kernel. Computed, not asserted, because it is the justification for the 3x.
_FLOOR_SCORE = 1.0 / (1.0 + (1.0 / 3.0) ** 2)


# Action slots that drive a wheel, in the per-leg blocks of four the env
# documents: [abduct, hip, knee, WHEEL] x (FL, FR, RL, RR).
_WHEEL_SLOTS = (3, 7, 11, 15)


def measure(env_id: str, episodes: int, seed0: int, wheel: float = 0.0) -> dict:
    """Roll episodes at a fixed wheel command and report the residual velocity.

    `wheel = 0.0` is the standing case: whatever velocity survives a machine
    actively holding its stance is what the terrain injects and no tracking
    policy can null.

    `wheel > 0` matters because the standing number is only a LOWER bound on
    the noise a MOVING policy faces — a machine crossing 7.82 cm cells at speed
    is shaken far harder than one standing on them, and a tolerance derived
    from the standing floor alone would be tighter than any moving policy could
    ever meet. 0.3 is the one open-loop wheel command the record says survives
    (`research/anomalies.jsonl`: the survivable band is non-monotonic, and both
    0.2 and 0.5 crash).
    """
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  -- registers the env ids

    env = gym.make(env_id)
    try:
        zero = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for slot in _WHEEL_SLOTS:
            zero[slot] = wheel
        vx, vy, yaw, lengths = [], [], [], []
        for episode in range(episodes):
            env.reset(seed=seed0 + episode)
            steps = 0
            while True:
                _, _, terminated, truncated, _ = env.step(zero)
                qvel = env.unwrapped.data.qvel
                vx.append(float(qvel[0]))
                vy.append(float(qvel[1]))
                yaw.append(float(qvel[5]))
                steps += 1
                if terminated or truncated:
                    break
            lengths.append(steps)
    finally:
        env.close()

    vx_a, vy_a, yaw_a = np.array(vx), np.array(vy), np.array(yaw)
    # The planar speed error a stationary machine cannot remove: the magnitude
    # of its residual xy velocity, not the std of one axis. That is the
    # quantity the reward's ||v_xy - v_cmd|| actually sees at v_cmd = 0.
    planar = np.hypot(vx_a, vy_a)

    return {
        "env_id": env_id,
        "wheel_command": wheel,
        "episodes": episodes,
        "seeds": [seed0 + i for i in range(episodes)],
        "steps_total": int(len(vx)),
        "episode_lengths": lengths,
        "linear": {
            "mean_vx": round(float(vx_a.mean()), 5),
            "std_vx": round(float(vx_a.std(ddof=1)), 5),
            "rms_planar_speed": round(float(np.sqrt((planar ** 2).mean())), 5),
            "p95_planar_speed": round(float(np.percentile(planar, 95)), 5),
        },
        "yaw": {
            "mean": round(float(yaw_a.mean()), 5),
            "std": round(float(yaw_a.std(ddof=1)), 5),
            "rms": round(float(np.sqrt((yaw_a ** 2).mean())), 5),
            "p95_abs": round(float(np.percentile(np.abs(yaw_a), 95)), 5),
        },
        "derived": {
            "floor_score_under_cauchy": round(_FLOOR_SCORE, 4),
            "sigma_v_from_rms": round(3.0 * float(np.sqrt((planar ** 2).mean())), 4),
            "sigma_omega_from_std": round(3.0 * float(yaw_a.std(ddof=1)), 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HoundPDDesert-v0")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--wheel", type=float, default=0.0,
                        help="constant wheel command applied to all four wheels")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = measure(args.env, args.episodes, args.seed0, args.wheel)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    lin, yaw, der = result["linear"], result["yaw"], result["derived"]
    print(f"{result['env_id']}: {result['episodes']} episodes at wheel="
          f"{result['wheel_command']}, {result['steps_total']} steps, "
          f"mean length {np.mean(result['episode_lengths']):.0f}")
    print(f"  planar speed   rms {lin['rms_planar_speed']:.4f} m/s   "
          f"p95 {lin['p95_planar_speed']:.4f}   mean vx {lin['mean_vx']:+.4f}")
    print(f"  yaw rate       std {yaw['std']:.4f} rad/s   "
          f"rms {yaw['rms']:.4f}   p95|.| {yaw['p95_abs']:.4f}")
    print(f"  -> sigma_v     {der['sigma_v_from_rms']:.4f} m/s    (3x planar rms)")
    print(f"  -> sigma_omega {der['sigma_omega_from_std']:.4f} rad/s  (3x yaw std)")
    print(f"  an error at the floor scores {der['floor_score_under_cauchy']:.3f} "
          f"under Phi(u)=1/(1+u^2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
