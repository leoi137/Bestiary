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


# ---------------------------------------------------------------------------
# The reward-ablation ladder: full command-tracking income + at most ONE
# penalty, three ways. `spyder_ladder_env_cfg.py` carries the question.
#
# All three subclass `SpyderGentlePPORunnerCfg` for the reason spelled out
# above: the ladder's whole claim is that its arms differ from the gentle task
# — and from each other — in the reward table and the lateral command range and
# NOTHING else, so the PPO hyperparameters have to be identical by
# construction. Three re-derivations from `AnymalCRoughPPORunnerCfg` would be
# three lists of numbers that happen to match today; one shared parent cannot
# drift. Only `experiment_name` moves in each, and it MUST: rsl_rl files runs
# under `logs/rsl_rl/<experiment_name>/`, and three rungs sharing a directory
# is three rewards' checkpoints in one folder — the `anymal_c_rough`
# mis-filing this file exists to stop, times three.
# ---------------------------------------------------------------------------
@configclass
class SpyderLadderBarePPORunnerCfg(SpyderGentlePPORunnerCfg):
    """Ladder rung 1: command-tracking income only, no penalty."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_ladder_bare"


@configclass
class SpyderLadderActionRatePPORunnerCfg(SpyderGentlePPORunnerCfg):
    """Ladder rung 2: income + `action_rate_l2` at the gentle task's -0.01."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_ladder_actionrate"


@configclass
class SpyderLadderTiltPPORunnerCfg(SpyderGentlePPORunnerCfg):
    """Ladder rung 3: income + `ang_vel_xy_l2` at the gentle task's -0.05."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_ladder_tilt"


@configclass
class SpyderOvernightPPORunnerCfg(SpyderGentlePPORunnerCfg):
    """The long run: the ladder's winner plus the two gait-shaping terms.

    `spyder_overnight_env_cfg.py` carries the reward argument. Same subclassing
    logic as everything above — the PPO hyperparameters are the gentle task's
    by construction, so the only thing this run changes against the ladder is
    the reward table, the run length and the seed.

    `experiment_name` moves and it must: rsl_rl files runs under
    `logs/rsl_rl/<experiment_name>/`, and three ladder arms plus the gentle
    seed already occupy four directories. A long run sharing one of them would
    put two rewards' checkpoints in one folder, which is the `anymal_c_rough`
    mis-filing this file exists to stop.

    MAX_ITERATIONS IS NOT PINNED HERE, DELIBERATELY
    -----------------------------------------------
    It stays at the inherited 1500 and is overridden per launch:
    `bestiary.isaac.train_desert` forwards its argv verbatim to Isaac Lab's
    `train_rsl_rl.run()`, whose parser carries `--max_iterations` (verified by
    `train_desert --help`, 2026-08-07) and whose `main()` writes it onto the
    agent config before the runner is built. Pinning 15000 in the class would
    make every future short smoke-test of this task a 15000-iteration run
    unless someone remembered the flag; leaving it on the CLI makes the run
    length a property of the launch, which is where the wall-clock ceiling that
    authorises it also lives.

    SAVE_INTERVAL: INHERITED AT 50, AND HERE IS WHY THAT IS AFFORDABLE
    ------------------------------------------------------------------
    rsl_rl saves at every iteration index divisible by `save_interval` and once
    more at the end, so N iterations produce `floor((N-1)/50) + 2` checkpoints.
    Measured on the ladder arms, which ran the identical network at 1500
    iterations: 31 files of 6,882,293 B each
    (`runs/spyder_ladder_s1/spyder_ladder_*/*/model_*.pt`), and
    `floor(1499/50) + 2 = 31` reproduces the count exactly.

    At 15000 iterations that formula gives **301 checkpoints x 6,882,293 B =
    2,071,570,193 B = 2.07 GB (1.93 GiB)**, plus a tensorboard event file that
    was 1.96 MB over 1500 iterations and should land near 20 MB over 15000.
    Call it 2.1 GB for the run.

    That is under the ~5 GB bar this cfg was asked to respect, so the interval
    is INHERITED rather than set — one fewer number that differs from the
    ladder's arms, and the checkpoint grid stays at the 50-iteration spacing
    every earlier Spyder run used, which is what makes a checkpoint from this
    run comparable to a checkpoint from those. Raising it to 100 would halve
    the cost to 1.04 GB and halve the resolution of the training curve's
    playable history; at 2.07 GB there is no reason to buy that.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_overnight"
