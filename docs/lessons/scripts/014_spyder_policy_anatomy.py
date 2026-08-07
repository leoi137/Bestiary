"""Arithmetic for lesson 014 — the anatomy of the Spyder PPO policy.

Every number in the lesson is printed by this script, and every number is read
off the trained checkpoint or the run's own config dump, never transcribed:

  * `model_1499.pt` is opened and each parameter tensor's shape is listed, so
    the layer widths in the lesson are the widths the optimiser actually
    updated;
  * the rsl-rl model is rebuilt from `params/agent.yaml` and the checkpoint is
    loaded into it with `strict=True` — a `print(model)` that did not accept
    the real weights would be a drawing, not a measurement;
  * the observation blocks are summed and the total is ASSERTED equal to the
    first layer's `in_features`, so a block table that does not add up fails
    here instead of misleading a reader;
  * the action scale, the PD gains and the effort ceiling come from the run's
    `params/env.yaml`, which is what the simulator was handed.

The model listing needs rsl-rl, which lives in the Isaac Lab virtualenv rather
than this repo's `venv`. Without it the script still prints every shape and
every check; it only skips the two `print(model)` blocks.

No GPU, no simulation app, no training, writes nothing. `PYTHONPATH=src`
because `bestiary` is installed into this repo's `venv`, not into the Isaac
Lab one that carries rsl-rl.

    PYTHONPATH=src ~/isaaclab-env/bin/python \
        docs/lessons/scripts/014_spyder_policy_anatomy.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from bestiary.paths import RUNS

RUN_DIR = RUNS / "spyder_gentle_s1" / "box_logs" / "2026-08-06_07-53-39"
CHECKPOINT = RUN_DIR / "model_1499.pt"
ENV_YAML = RUN_DIR / "params" / "env.yaml"
AGENT_YAML = RUN_DIR / "params" / "agent.yaml"


class CfgLoader(yaml.SafeLoader):
    """SafeLoader that tolerates Isaac Lab's python-tagged config dumps.

    `env.yaml` is written with `yaml.dump` on live config objects, so it
    carries `!!python/tuple` and `!!python/object/apply:builtins.slice` tags
    that `safe_load` refuses and `unsafe_load` would execute. Tuples become
    lists (every field this script reads is indexed, not identity-compared)
    and anything else python-tagged becomes None, which is enough: nothing
    here reads a slice.
    """


CfgLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node, deep=True),
)
for _prefix in ("tag:yaml.org,2002:python/object/apply:", "tag:yaml.org,2002:python/name:"):
    CfgLoader.add_multi_constructor(_prefix, lambda loader, suffix, node: None)


def load_cfg(path: Path) -> dict:
    with path.open() as fh:
        return yaml.load(fh, Loader=CfgLoader)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    env = load_cfg(ENV_YAML)
    agent = load_cfg(AGENT_YAML)

    print(f"checkpoint  {CHECKPOINT}")
    print(f"            {CHECKPOINT.stat().st_size:,} bytes, iteration {ckpt['iter']}")
    print(f"top-level keys: {sorted(ckpt)}")

    # -- every tensor in the checkpoint ------------------------------------
    counts = {}
    for group in ("actor_state_dict", "critic_state_dict"):
        rule(group)
        total = 0
        for key, tensor in ckpt[group].items():
            total += tensor.numel()
            print(f"  {key:<28} {str(tuple(tensor.shape)):<14} {tensor.numel():>9,}")
        counts[group] = total
        print(f"  {'TOTAL':<28} {'':<14} {total:>9,}")

    total_params = counts["actor_state_dict"] + counts["critic_state_dict"]
    print(f"\n  actor + critic = {total_params:,} parameters")

    obs_width = ckpt["actor_state_dict"]["mlp.0.weight"].shape[1]
    act_dim = ckpt["actor_state_dict"]["mlp.6.weight"].shape[0]
    critic_out = ckpt["critic_state_dict"]["mlp.6.weight"].shape[0]
    print(f"  measured obs width {obs_width}, action dim {act_dim}, critic output {critic_out}")

    # -- the agent config the shapes must agree with -----------------------
    rule("agent.yaml — what those shapes were asked for")
    for role in ("actor", "critic"):
        spec = agent[role]
        print(
            f"  {role:<7} {spec['class_name']} hidden_dims={spec['hidden_dims']} "
            f"activation={spec['activation']} obs_normalization={spec['obs_normalization']}"
        )
    dist = agent["actor"]["distribution_cfg"]
    print(f"  actor distribution: {dist}")
    alg = agent["algorithm"]
    print(
        f"  PPO: gamma={alg['gamma']} lam={alg['lam']} clip={alg['clip_param']} "
        f"entropy_coef={alg['entropy_coef']} lr={alg['learning_rate']} ({alg['schedule']})"
    )
    print(
        f"  rollout: {env['scene']['num_envs']} envs x {agent['num_steps_per_env']} steps "
        f"= {env['scene']['num_envs'] * agent['num_steps_per_env']:,} samples per update, "
        f"{agent['max_iterations']} iterations"
    )

    # -- the observation, block by block -----------------------------------
    rule("observation blocks (policy group)")
    scan = env["scene"]["height_scanner"]["pattern_cfg"]
    nx = int(round(scan["size"][0] / scan["resolution"])) + 1
    ny = int(round(scan["size"][1] / scan["resolution"])) + 1
    n_rays = nx * ny
    cmd_ranges = env["commands"]["base_velocity"]["ranges"]
    n_cmd = sum(1 for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z") if cmd_ranges[key] is not None)

    blocks = [
        ("base_lin_vel", 3, "body-frame velocity of the torso, m/s"),
        ("base_ang_vel", 3, "body-frame turn rate of the torso, rad/s"),
        ("projected_gravity", 3, "gravity unit vector in the torso frame"),
        ("velocity_commands", n_cmd, "the joystick: vx, vy, wz"),
        ("joint_pos", act_dim, "joint angle minus its default, rad"),
        ("joint_vel", act_dim, "joint angular velocity, rad/s"),
        ("actions", act_dim, "the previous step's raw action"),
        ("height_scan", n_rays, f"{nx} x {ny} downward rays, m, clipped"),
    ]

    declared = list(env["observations"]["policy"])
    named = [name for name, _, _ in blocks]
    missing = [n for n in declared if n not in named and not n.startswith(("concatenate", "enable", "history", "flatten"))]
    if missing:
        raise AssertionError(f"env.yaml declares observation terms this table does not name: {missing}")

    start = 0
    for name, size, meaning in blocks:
        noise = env["observations"]["policy"][name]["noise"]
        amp = f"+-{noise['n_max']}" if noise else "none"
        print(f"  [{start:>3}:{start + size:>3})  {name:<19} {size:>3}   noise {amp:<8} {meaning}")
        start += size

    print(f"  {'':>11}  {'SUM':<19} {start:>3}")
    if start != obs_width:
        raise AssertionError(
            f"the observation blocks sum to {start} but the actor's first layer is "
            f"Linear({obs_width}, ...) — the table is wrong, or the env moved."
        )
    print(f"  matches mlp.0.weight columns ({obs_width}). OK")

    # -- the action, and what the simulator does with it -------------------
    rule("action -> joint target -> torque")
    act = env["actions"]["joint_pos"]
    legs = env["scene"]["robot"]["actuators"]["legs"]
    kp = legs["stiffness"][".*"]
    kd = legs["damping"][".*"]
    effort = legs["effort_limit_sim"]
    inertia = 1.04  # kg.m^2, the value KD was derived against (spyder_cfg.py)
    zeta_check = kd / (2.0 * math.sqrt(kp * inertia))
    print(f"  JointPositionAction  scale={act['scale']}  offset={act['offset']}  "
          f"use_default_offset={act['use_default_offset']}  clip={act['clip']}")
    print(f"  default joint pos    {env['scene']['robot']['init_state']['joint_pos']}")
    print(f"  KP={kp} N.m/rad   KD={kd} N.m.s/rad   effort ceiling={effort} N.m")
    print(f"  KD re-derived: 2*zeta*sqrt(KP*I) with I={inertia} gives zeta={zeta_check:.4f}")
    torque_per_action = kp * act["scale"]
    print(f"  at zero joint error and zero joint speed: tau = KP*scale*a = {torque_per_action} * a")
    print(f"  so |a| >= {effort / torque_per_action:.4f} already saturates the {effort} N.m ceiling")

    dt = env["sim"]["dt"] * env["decimation"]
    print(f"  control step {dt} s ({1.0 / dt:.0f} Hz policy, {1.0 / env['sim']['dt']:.0f} Hz physics), "
          f"episode {env['episode_length_s']} s = {round(env['episode_length_s'] / dt)} steps")

    # -- the learned exploration noise -------------------------------------
    rule("learned action noise (GaussianDistribution, std_type=scalar)")
    std = ckpt["actor_state_dict"]["distribution.std_param"]
    std_dim = std.numel()
    print(f"  std per joint: {[round(v, 4) for v in std.tolist()]}")
    print(f"  mean {std.mean():.4f}, min {std.min():.4f}, max {std.max():.4f} (action units)")
    print(f"  min std {std.min():.4f} -> {std.min() * act['scale']:.4f} rad of target jitter "
          f"-> {std.min() * torque_per_action:.2f} N.m of torque jitter at zero error")

    # -- the reward table --------------------------------------------------
    rule("gentle-task reward terms (weight x step_dt applied per step)")
    for name, term in env["rewards"].items():
        if term is None:
            continue
        extra = {k: v for k, v in (term.get("params") or {}).items() if k in ("std", "threshold")}
        print(f"  {name:<22} weight {term['weight']:>10}   {extra if extra else ''}")
    print(f"  {len(env['rewards'])} terms declared")

    rule("forward-only diagnostic: reward = v_x")
    v_x = 0.37  # m/s, the SAC Spyder-12 desert walk of research/learnings/001
    per_step = v_x * dt
    steps = round(env["episode_length_s"] / dt)
    print(f"  weight 1.0 x step_dt {dt} s: per-step reward = v_x * {dt} = metres travelled")
    print(f"  at learnings/001's {v_x} m/s: {v_x} * {dt} = {per_step:.4f} per step")
    print(f"  over a full {env['episode_length_s']} s episode ({steps} steps): "
          f"return = {per_step * steps:.2f} = metres of forward travel")

    # -- what actually ships -----------------------------------------------
    rule("exported/policy.pt — the deployed half")
    exported = torch.jit.load(str(RUN_DIR / "exported" / "policy.pt"), map_location="cpu")
    shipped = sum(t.numel() for t in exported.state_dict().values())
    print(f"  modules: {[name for name, _ in exported.named_children()]}")
    print(f"  parameters: {shipped:,}")
    print(f"  = actor MLP ({counts['actor_state_dict'] - std_dim:,}) with no critic and no std")
    print(f"  critic share of the trained weights: "
          f"{counts['critic_state_dict'] / total_params:.1%}, none of it exported")

    # -- print(model), if rsl-rl is importable -----------------------------
    rule("print(model)")
    try:
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict
    except ImportError as exc:  # pragma: no cover - depends on which venv runs this
        print(f"  rsl-rl not importable here ({exc}); shapes above are unaffected.")
        print("  Re-run with ~/isaaclab-env/bin/python for the module listing.")
        return

    dummy = TensorDict({"policy": torch.zeros(1, obs_width)}, batch_size=[1])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    models = {
        "actor": MLPModel(
            dummy, groups, "actor", act_dim,
            hidden_dims=agent["actor"]["hidden_dims"],
            activation=agent["actor"]["activation"],
            obs_normalization=agent["actor"]["obs_normalization"],
            distribution_cfg=dict(agent["actor"]["distribution_cfg"]),
        ),
        "critic": MLPModel(
            dummy, groups, "critic", critic_out,
            hidden_dims=agent["critic"]["hidden_dims"],
            activation=agent["critic"]["activation"],
            obs_normalization=agent["critic"]["obs_normalization"],
            distribution_cfg=None,
        ),
    }
    for role, model in models.items():
        # strict=True: the listing below is only evidence if the real weights fit it.
        model.load_state_dict(ckpt[f"{role}_state_dict"], strict=True)
        n = sum(p.numel() for p in model.parameters())
        print(f"\n  # {role} — {n:,} parameters, loaded strict=True")
        print(model)


if __name__ == "__main__":
    main()
