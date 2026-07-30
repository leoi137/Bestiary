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

The agent configs are reused verbatim from Isaac Lab's anymal_c package. For the
ANYmal tasks that is exactly right: they differ from theirs in the terrain and
nothing else, so the PPO hyperparameters should be identical -- otherwise a
throughput or reward comparison against their rough task measures two changes at
once.

For the HOUND tasks the same reuse is a placeholder, not a control. A 16-DoF
wheel-legged machine with two action groups is not the robot those
hyperparameters were tuned for, and the reward they optimise is ANYmal's too.
The Hound ids exist so a viewer and an oracle can reach a config; see
`hound_desert_env_cfg.py`'s docstring for what is still missing before either
should be trained.
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


def register() -> None:
    """Register the Bestiary tasks. Idempotent -- safe to call more than once."""
    specs = (
        ("Bestiary-Desert-Anymal-C-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg"),
        ("Bestiary-Desert-Coarse-Anymal-C-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertCoarseEnvCfg"),
        ("Bestiary-Desert-Anymal-C-Play-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg_PLAY"),
        # Hound. NOT ready to train: the reward is ANYmal-C's, inherited whole,
        # and `feet_air_time` on a wheel pays the machine to hop. The env cfg's
        # module docstring says so at length. Registered because a task id is
        # how a viewer and an oracle reach a config, not because it is finished.
        ("Bestiary-Desert-Hound-v0", f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg"),
        ("Bestiary-Desert-Hound-Play-v0", f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg_PLAY"),
    )
    for task_id, cfg_entry_point in specs:
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": cfg_entry_point,
                "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
            },
        )


register()
