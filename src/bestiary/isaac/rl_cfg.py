"""rsl_rl runner configs for the Bestiary tasks.

One subclass per robot-and-terrain-and-reward pairing, each changing exactly
one thing against the config it descends from: the `experiment_name`.

WHY THE EXPERIMENT NAME IS THE WHOLE FILE
-----------------------------------------
The Hound trained on the rented Blackwell box with the ANYmal agent config
verbatim, so its runs filed themselves under `logs/rsl_rl/anymal_c_rough/` —
three Hound seeds indistinguishable from ANYmal runs by path alone, flagged in
STATE's handoff ("Fix `experiment_name` so Hound stops logging under
`anymal_c_rough`"). A run's directory is the first fact anyone learns about
it; it should not be a lie about which robot ran.

Everything else is INHERITED DELIBERATELY, per the same logic `tasks.py`
records for the ANYmal tasks: these PPO hyperparameters are the five-year
recipe (Rudin's operating point at 4096 x 24 = 98,304 samples per update at
the env counts we run), and a first Spyder run that changed them alongside the
robot, the terrain, the commands and the kernel widths would measure five
things at once. They are a control, not an endorsement — the network sizes
and the clip/entropy settings were shaped on 235-ish obs quadrupeds, which
Spyder now is (235 = 3+3+3+3+12+12+12+187).
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.agents.rsl_rl_ppo_cfg import (
    AnymalCRoughPPORunnerCfg,
)


@configclass
class SpyderGentlePPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """Spyder-12 on the gentle mix. ANYmal's agent, Spyder's name."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_gentle"


@configclass
class SpyderForwardPPORunnerCfg(SpyderGentlePPORunnerCfg):
    """The forward-velocity-only diagnostic. Same agent, different run dir.

    Subclasses the gentle runner rather than `AnymalCRoughPPORunnerCfg`
    directly, and that is the whole safety argument: the diagnostic's claim is
    that it changes EXACTLY ONE variable against the gentle task (the reward),
    so its PPO hyperparameters must be identical by construction, not by two
    lists of numbers that happen to match today. Inheriting from the sibling
    makes a future edit to the gentle agent land on both arms automatically —
    which is what keeps them comparable — while re-deriving from ANYmal would
    let the two drift apart silently.

    Only `experiment_name` moves. It MUST: rsl_rl files runs under
    `logs/rsl_rl/<experiment_name>/`, and seed 1 of the gentle arm already
    lives in `spyder_gentle/` (`runs/spyder_gentle_s1/`). A diagnostic writing
    into that directory would put two different rewards' checkpoints in one
    folder, which is the `anymal_c_rough` mis-filing this file exists to stop.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_forward"
