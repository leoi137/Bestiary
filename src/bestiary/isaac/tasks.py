"""Gymnasium registration for the Bestiary Isaac Lab tasks.

DELIBERATELY THE LIGHTEST MODULE IN THE PACKAGE

This imports `gymnasium` and nothing else. That is not tidiness, it is the
mechanism that makes the whole approach work.

Isaac Lab's own task registrations pass their configs as *strings* --
``"module.path:ClassName"`` -- which gymnasium resolves lazily at env-creation
time rather than at registration time. So registration can happen before Isaac
Sim is launched, and the config modules (which do import `isaaclab`) are only
imported later, once the simulation app exists.

That ordering is what lets `train_desert.py` register our tasks and then hand
straight over to Isaac Lab's own `train_rsl_rl.run()`, instead of us
reimplementing a PPO training loop. Import anything heavy here and that breaks:
the config would be imported before the app is up.

The ANYmal and Hound tasks reuse Isaac Lab's anymal_c agent config verbatim.
For the ANYmal tasks that is exactly right: they differ from theirs in the
terrain and nothing else, so the PPO hyperparameters should be identical --
otherwise a throughput or reward comparison against their rough task measures
two changes at once. For the Hound it is a known compromise that also mis-files
the runs: arms 1 and 2 trained under ANYmal's `experiment_name`, so their logs
landed in `logs/rsl_rl/anymal_c_rough/` (STATE's handoff carries the repair).

The SPYDER tasks are the repair, half-applied deliberately: they resolve
`bestiary.isaac.rl_cfg:SpyderGentlePPORunnerCfg`, which changes the
`experiment_name` to `spyder_gentle` and NOTHING else -- the hyperparameters
stay ANYmal's, as a control, per `rl_cfg.py`'s docstring.

Each row of the spec table below names its OWN agent config. It used to be
sniffed from the task id (`"Spyder" in task_id`), which was one substring away
from filing a new Spyder variant's runs into an existing variant's directory --
exactly the `anymal_c_rough` mis-filing this file already carries a paragraph
about. A task and its agent config are now paired in one place, per row, so the
pairing cannot be got wrong by naming a task carelessly.
"""

from __future__ import annotations

import gymnasium as gym

#: Isaac Lab's own rsl_rl PPO config for ANYmal-C rough. Reused, not copied, so
#: our runs stay comparable to Isaac-Velocity-Rough-Anymal-C-v0.
_RSL_RL_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg"
    ":AnymalCRoughPPORunnerCfg"
)

_ANYMAL_CFG_MODULE = "bestiary.isaac.anymal_desert_env_cfg"
_HOUND_CFG_MODULE = "bestiary.isaac.hound_desert_env_cfg"
_SPYDER_CFG_MODULE = "bestiary.isaac.spyder_gentle_env_cfg"
_SPYDER_FWD_CFG_MODULE = "bestiary.isaac.spyder_forward_env_cfg"
_SPYDER_FWD_V5_CFG_MODULE = "bestiary.isaac.spyder_forward_v5_env_cfg"
_HOUND_FWD_V5_CFG_MODULE = "bestiary.isaac.hound_forward_v5_env_cfg"
_HOUND_OVERNIGHT_CFG_MODULE = "bestiary.isaac.hound_overnight_env_cfg"
_SPYDER_LADDER_CFG_MODULE = "bestiary.isaac.spyder_ladder_env_cfg"
_SPYDER_OVERNIGHT_CFG_MODULE = "bestiary.isaac.spyder_overnight_env_cfg"
_SPYDER_FAST_CFG_MODULE = "bestiary.isaac.spyder_fast_env_cfg"
_SPYDER_DEMO_CFG_MODULE = "bestiary.isaac.spyder_demo_env_cfg"

#: Spyder gets its own runner cfg — ANYmal's agent with its own
#: `experiment_name`, so runs stop filing themselves under `anymal_c_rough`
#: (the Hound did exactly that on the rented box; STATE's handoff carries the
#: repair). Same lazy-string mechanism as everything else here.
_SPYDER_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderGentlePPORunnerCfg"

#: The forward-velocity-only diagnostic's runner cfg. Identical PPO
#: hyperparameters (it subclasses the one above), `experiment_name` =
#: `spyder_forward` so its checkpoints never land in `spyder_gentle/` beside
#: seed 1's — two rewards in one run directory is unrecoverable bookkeeping.
_SPYDER_FWD_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderForwardPPORunnerCfg"

#: The same diagnostic on the v5 terrain (`research/decisions/0007`), and on the
#: Hound. One `experiment_name` each — `spyder_forward_v5` and `hound_forward_v5`
#: — because the ground is what changed: two runs of the SAME reward on
#: DIFFERENT terrain in one log tree is a directory whose rows cannot be
#: compared and whose difference nothing records.
_SPYDER_FWD_V5_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderForwardV5PPORunnerCfg"
_HOUND_FWD_V5_RSL_RL_CFG = "bestiary.isaac.rl_cfg:HoundForwardV5PPORunnerCfg"

#: The commanded Hound long run. `experiment_name` = `hound_overnight`, and the
#: separation from `hound_forward_v5/` is load-bearing rather than tidy: both
#: tasks are the same body on the same v5 ground at the same env count, so one
#: shared log tree would read as a single lineage while holding two rewards —
#: one that pays `v_x` and cannot be driven, one that pays command tracking.
_HOUND_OVERNIGHT_RSL_RL_CFG = "bestiary.isaac.rl_cfg:HoundOvernightPPORunnerCfg"

#: The reward-ladder rungs' runner cfgs. One `experiment_name` each —
#: `spyder_ladder_bare` / `_actionrate` / `_tilt` — so the three arms of a
#: three-arm comparison cannot land in one log tree. Same PPO hyperparameters
#: as every other Spyder task (they all descend from `SpyderGentlePPORunnerCfg`).
_SPYDER_LADDER_BARE_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderLadderBarePPORunnerCfg"
_SPYDER_LADDER_ACTIONRATE_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderLadderActionRatePPORunnerCfg"
_SPYDER_LADDER_TILT_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderLadderTiltPPORunnerCfg"

#: The long run's runner cfg. `experiment_name` = `spyder_overnight`, so its
#: checkpoints never land beside a ladder arm's — the ladder's three arms and
#: this run share every PPO hyperparameter and differ in the reward table, which
#: is exactly the pair of facts that makes one shared log directory
#: unrecoverable bookkeeping.
_SPYDER_OVERNIGHT_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderOvernightPPORunnerCfg"

#: The fine-tune's runner cfg. `experiment_name` = `spyder_fast`, and here the
#: separation is load-bearing beyond bookkeeping: this task RESUMES from the
#: overnight run's `model_14999.pt`, so leaving it under `spyder_overnight`
#: would write its new checkpoints into the very directory it reads from, and
#: the numbering would read as one continuous 21,000-iteration curve that never
#: happened.
_SPYDER_FAST_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderFastPPORunnerCfg"


def register() -> None:
    """Register the Bestiary tasks. Idempotent -- safe to call more than once."""
    #: (task id, env-cfg entry point, rsl_rl agent-cfg entry point). Both entry
    #: points are LAZY STRINGS: gymnasium resolves them at env-creation time,
    #: which is what lets this module be imported before Isaac Sim exists.
    specs = (
        ("Bestiary-Desert-Anymal-C-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg", _RSL_RL_CFG),
        (
            "Bestiary-Desert-Coarse-Anymal-C-v0",
            f"{_ANYMAL_CFG_MODULE}:AnymalCDesertCoarseEnvCfg",
            _RSL_RL_CFG,
        ),
        (
            "Bestiary-Desert-Anymal-C-Play-v0",
            f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg_PLAY",
            _RSL_RL_CFG,
        ),
        # Hound. The reward is 0004 Part B's re-scoping of ANYmal-C's table
        # (contact-timing terms deleted, joint penalties split by group);
        # trained as arms 1 and 2 on the rented box, 2026-07-30. What the
        # reward is still KNOWN to get wrong is in STATE and the env cfg's
        # module docstring — point-and-park is measured, not hypothetical.
        ("Bestiary-Desert-Hound-v0", f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg", _RSL_RL_CFG),
        (
            "Bestiary-Desert-Hound-Play-v0",
            f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg_PLAY",
            _RSL_RL_CFG,
        ),
        # Spyder on the gentle terrain: the first Bestiary robot registered
        # here as READY to train — commands dead-zoned, heading mode off,
        # reward retargeted term by term. `spyder_gentle_env_cfg.py`'s module
        # docstring carries the design; `check_spyder.py` is the oracle.
        (
            "Bestiary-Gentle-Spyder-v0",
            f"{_SPYDER_CFG_MODULE}:SpyderGentleEnvCfg",
            _SPYDER_RSL_RL_CFG,
        ),
        (
            "Bestiary-Gentle-Spyder-Play-v0",
            f"{_SPYDER_CFG_MODULE}:SpyderGentleEnvCfg_PLAY",
            _SPYDER_RSL_RL_CFG,
        ),
        # The DIAGNOSTIC arm: same everything, reward replaced by base-frame
        # v_x alone. Not a proposal for how to train a command-following
        # walker — it cannot be, it does not read the command — but the one
        # run that can tell "the reward table is wrong" apart from "the port
        # is wrong". `spyder_forward_env_cfg.py` carries the argument.
        (
            "Bestiary-Forward-Spyder-v0",
            f"{_SPYDER_FWD_CFG_MODULE}:SpyderForwardEnvCfg",
            _SPYDER_FWD_RSL_RL_CFG,
        ),
        (
            "Bestiary-Forward-Spyder-Play-v0",
            f"{_SPYDER_FWD_CFG_MODULE}:SpyderForwardEnvCfg_PLAY",
            _SPYDER_FWD_RSL_RL_CFG,
        ),
        # The same diagnostic on the v5 ground. `research/decisions/0007` makes
        # v5 mandatory for every NEW arm and leaves v4 committed and untouched
        # for the lineages that trained on it, so this is a task BESIDE
        # `Bestiary-Forward-Spyder-v0`, never a repoint of it. One variable
        # against that task, asserted: the bestiary tile's hfield path.
        (
            "Bestiary-ForwardV5-Spyder-v0",
            f"{_SPYDER_FWD_V5_CFG_MODULE}:SpyderForwardV5EnvCfg",
            _SPYDER_FWD_V5_RSL_RL_CFG,
        ),
        (
            "Bestiary-ForwardV5-Spyder-Play-v0",
            f"{_SPYDER_FWD_V5_CFG_MODULE}:SpyderForwardV5EnvCfg_PLAY",
            _SPYDER_FWD_V5_RSL_RL_CFG,
        ),
        # The same question asked of the wheel-legged machine: reward = v_x
        # only, on v5 ground. On a body whose feet are driven hub wheels the
        # reward cannot distinguish rolling from galloping, and
        # `hound_forward_v5_env_cfg.py` argues that the ambiguity IS the
        # experiment. First Hound task with an `experiment_name` of its own —
        # arms 1 and 2 filed themselves under `anymal_c_rough`.
        (
            "Bestiary-ForwardV5-Hound-v0",
            f"{_HOUND_FWD_V5_CFG_MODULE}:HoundForwardV5EnvCfg",
            _HOUND_FWD_V5_RSL_RL_CFG,
        ),
        (
            "Bestiary-ForwardV5-Hound-Play-v0",
            f"{_HOUND_FWD_V5_CFG_MODULE}:HoundForwardV5EnvCfg_PLAY",
            _HOUND_FWD_V5_RSL_RL_CFG,
        ),
        # The COMMANDED Hound: the Spyder's steering pipeline — dead-zoned rate
        # commands, heading mode off, the arc-corrected terrain curriculum — on
        # the wheel-legged body, on the same v5 ground the forward probe ran on,
        # with the reward cut to command-tracking income plus `action_rate_l2`
        # and `lin_vel_z_l2`. First Hound task that can be DRIVEN: the forward
        # probe's reward never read a command, and its final block shows what
        # that buys (error_vel_xy 11.39, 203.56 m per episode).
        # `feet_air_time` is deliberately absent — a cadence term is undefined on
        # a rolling wheel — and `hound_overnight_env_cfg.py` argues it, along with
        # the kernel widths that are NOT rescaled with the ±1.5 box.
        (
            "Bestiary-Overnight-Hound-v0",
            f"{_HOUND_OVERNIGHT_CFG_MODULE}:HoundOvernightEnvCfg",
            _HOUND_OVERNIGHT_RSL_RL_CFG,
        ),
        (
            "Bestiary-Overnight-Hound-Play-v0",
            f"{_HOUND_OVERNIGHT_CFG_MODULE}:HoundOvernightEnvCfg_PLAY",
            _HOUND_OVERNIGHT_RSL_RL_CFG,
        ),
        # The reward-ablation LADDER: three arms, each paying the gentle task's
        # full command-tracking income plus AT MOST ONE penalty, and each
        # commanding strafe (lin_vel_y ±0.4, which the gentle task pins to
        # zero). Unlike the forward diagnostic these are steerable by
        # construction — the point is to find which single penalty tames the
        # gait on a policy that can still be driven.
        # `spyder_ladder_env_cfg.py` carries the argument; `check_spyder.py`'s
        # ladder check pins each rung's reward table and its one command diff.
        (
            "Bestiary-Ladder-Bare-Spyder-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderBareEnvCfg",
            _SPYDER_LADDER_BARE_RSL_RL_CFG,
        ),
        (
            "Bestiary-Ladder-Bare-Spyder-Play-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderBareEnvCfg_PLAY",
            _SPYDER_LADDER_BARE_RSL_RL_CFG,
        ),
        (
            "Bestiary-Ladder-ActionRate-Spyder-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderActionRateEnvCfg",
            _SPYDER_LADDER_ACTIONRATE_RSL_RL_CFG,
        ),
        (
            "Bestiary-Ladder-ActionRate-Spyder-Play-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderActionRateEnvCfg_PLAY",
            _SPYDER_LADDER_ACTIONRATE_RSL_RL_CFG,
        ),
        (
            "Bestiary-Ladder-Tilt-Spyder-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderTiltEnvCfg",
            _SPYDER_LADDER_TILT_RSL_RL_CFG,
        ),
        (
            "Bestiary-Ladder-Tilt-Spyder-Play-v0",
            f"{_SPYDER_LADDER_CFG_MODULE}:SpyderLadderTiltEnvCfg_PLAY",
            _SPYDER_LADDER_TILT_RSL_RL_CFG,
        ),
        # The LONG RUN: the ladder's measured winner (`action_rate_l2`) plus the
        # two terms that price the shape of a step — `feet_air_time` and
        # `lin_vel_z_l2` — on the ladder's own command envelope, strafe
        # included. One arm, one seed, ten times the iteration count: it spends
        # the ladder's answer rather than asking a new question, and
        # `spyder_overnight_env_cfg.py` says so in as many words.
        (
            "Bestiary-Overnight-Spyder-v0",
            f"{_SPYDER_OVERNIGHT_CFG_MODULE}:SpyderOvernightEnvCfg",
            _SPYDER_OVERNIGHT_RSL_RL_CFG,
        ),
        (
            "Bestiary-Overnight-Spyder-Play-v0",
            f"{_SPYDER_OVERNIGHT_CFG_MODULE}:SpyderOvernightEnvCfg_PLAY",
            _SPYDER_OVERNIGHT_RSL_RL_CFG,
        ),
        # The FINE-TUNE: the overnight task's reward, robot, terrain and
        # observation, commanded over a box 2.5x wider forward (±1.5 m/s),
        # 1.5x wider laterally (±0.6 m/s) and 1.875x wider in yaw
        # (±1.5 rad/s). One variable against the overnight task, and the run
        # it is launched for LOADS that task's `model_14999.pt` rather than
        # starting from scratch — provenance records it as a fine-tune, never
        # as a fresh arm. `spyder_fast_env_cfg.py` carries the argument,
        # including why the tracking kernel widths deliberately do not move
        # with the ranges.
        (
            "Bestiary-Fast-Spyder-v0",
            f"{_SPYDER_FAST_CFG_MODULE}:SpyderFastEnvCfg",
            _SPYDER_FAST_RSL_RL_CFG,
        ),
        (
            "Bestiary-Fast-Spyder-Play-v0",
            f"{_SPYDER_FAST_CFG_MODULE}:SpyderFastEnvCfg_PLAY",
            _SPYDER_FAST_RSL_RL_CFG,
        ),
        # PLAY ONLY, and there is deliberately no training twin. The demo strip
        # is one continuous surface with difficulty ramped along +x — a camera
        # subject, not a curriculum. Registering a `-v0` beside it would invite
        # someone to train on ground no ledger row should ever cite.
        # `spyder_demo_env_cfg.py` carries the argument.
        (
            "Bestiary-Demo-Spyder-Play-v0",
            f"{_SPYDER_DEMO_CFG_MODULE}:SpyderDemoEnvCfg_PLAY",
            _SPYDER_RSL_RL_CFG,
        ),
        (
            "Bestiary-Demo-Hound-Play-v0",
            "bestiary.isaac.hound_demo_env_cfg:HoundDemoEnvCfg_PLAY",
            "bestiary.isaac.rl_cfg:HoundForwardV5PPORunnerCfg",
        ),
    )
    for task_id, cfg_entry_point, rl_cfg in specs:
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": cfg_entry_point,
                "rsl_rl_cfg_entry_point": rl_cfg,
            },
        )


register()
