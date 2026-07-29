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

The agent configs are reused verbatim from Isaac Lab's anymal_c package. Our
tasks differ from theirs in the terrain and nothing else, so the PPO
hyperparameters should be identical -- otherwise a throughput or reward
comparison against their rough task measures two changes at once.
"""

from __future__ import annotations

import gymnasium as gym

#: Isaac Lab's own rsl_rl PPO config for ANYmal-C rough. Reused, not copied, so
#: our runs stay comparable to Isaac-Velocity-Rough-Anymal-C-v0.
_RSL_RL_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg"
    ":AnymalCRoughPPORunnerCfg"
)

_ENV_CFG_MODULE = "bestiary.isaac.anymal_desert_env_cfg"


def register() -> None:
    """Register the Bestiary tasks. Idempotent -- safe to call more than once."""
    specs = (
        ("Bestiary-Desert-Anymal-C-v0", "AnymalCDesertEnvCfg"),
        ("Bestiary-Desert-Coarse-Anymal-C-v0", "AnymalCDesertCoarseEnvCfg"),
        ("Bestiary-Desert-Anymal-C-Play-v0", "AnymalCDesertEnvCfg_PLAY"),
    )
    for task_id, cls_name in specs:
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{_ENV_CFG_MODULE}:{cls_name}",
                "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
            },
        )


register()
