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

#: Spyder gets its own runner cfg — ANYmal's agent with its own
#: `experiment_name`, so runs stop filing themselves under `anymal_c_rough`
#: (the Hound did exactly that on the rented box; STATE's handoff carries the
#: repair). Same lazy-string mechanism as everything else here.
_SPYDER_RSL_RL_CFG = "bestiary.isaac.rl_cfg:SpyderGentlePPORunnerCfg"


def register() -> None:
    """Register the Bestiary tasks. Idempotent -- safe to call more than once."""
    specs = (
        ("Bestiary-Desert-Anymal-C-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg"),
        ("Bestiary-Desert-Coarse-Anymal-C-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertCoarseEnvCfg"),
        ("Bestiary-Desert-Anymal-C-Play-v0", f"{_ANYMAL_CFG_MODULE}:AnymalCDesertEnvCfg_PLAY"),
        # Hound. The reward is 0004 Part B's re-scoping of ANYmal-C's table
        # (contact-timing terms deleted, joint penalties split by group);
        # trained as arms 1 and 2 on the rented box, 2026-07-30. What the
        # reward is still KNOWN to get wrong is in STATE and the env cfg's
        # module docstring — point-and-park is measured, not hypothetical.
        ("Bestiary-Desert-Hound-v0", f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg"),
        ("Bestiary-Desert-Hound-Play-v0", f"{_HOUND_CFG_MODULE}:HoundDesertEnvCfg_PLAY"),
        # Spyder on the gentle terrain: the first Bestiary robot registered
        # here as READY to train — commands dead-zoned, heading mode off,
        # reward retargeted term by term. `spyder_gentle_env_cfg.py`'s module
        # docstring carries the design; `check_spyder.py` is the oracle.
        ("Bestiary-Gentle-Spyder-v0", f"{_SPYDER_CFG_MODULE}:SpyderGentleEnvCfg"),
        ("Bestiary-Gentle-Spyder-Play-v0", f"{_SPYDER_CFG_MODULE}:SpyderGentleEnvCfg_PLAY"),
    )
    for task_id, cfg_entry_point in specs:
        if task_id in gym.registry:
            continue
        rl_cfg = _SPYDER_RSL_RL_CFG if "Spyder" in task_id else _RSL_RL_CFG
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
