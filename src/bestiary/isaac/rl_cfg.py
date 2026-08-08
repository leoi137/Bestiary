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


@configclass
class SpyderForwardV5PPORunnerCfg(SpyderForwardPPORunnerCfg):
    """The forward diagnostic on v5 ground. Same agent, different run dir.

    Subclasses the v4 forward runner for the reason every class in this file
    subclasses its sibling: this task's claim is that it changes EXACTLY ONE
    variable against `Bestiary-Forward-Spyder-v0` — the terrain — so its PPO
    hyperparameters must be identical by construction rather than by two lists
    of numbers that agree today.

    `experiment_name` moves and it MUST. rsl_rl files runs under
    `logs/rsl_rl/<experiment_name>/`, and `spyder_forward/` already holds the
    seed-1 run of `research/episodes/014` — trained on v4, whose crests reach
    47 degrees. Two DIFFERENT GROUNDS in one log directory is the quietest
    version of the `anymal_c_rough` mis-filing this file exists to stop: the
    checkpoints load, the reward is the same, the returns are in the same units,
    and nothing in the directory records that they were earned on different
    worlds (`research/decisions/0007`, and the terrain invariant in CLAUDE.md).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_forward_v5"


@configclass
class HoundForwardV5PPORunnerCfg(AnymalCRoughPPORunnerCfg):
    """The Hound's v_x-only diagnostic on v5 ground. ANYmal's agent, Hound's name.

    Descends from `AnymalCRoughPPORunnerCfg` directly rather than from a Spyder
    class, and that is not laziness: the Spyder runners are the same ANYmal
    hyperparameters, so inheriting one of them would make this task's agent
    config depend on a Spyder task's `experiment_name` chain for no shared
    claim. The hyperparameters are the five-year recipe kept as a CONTROL, per
    this file's module docstring; a first arm on a new body that also changed
    them would measure two things at once.

    `experiment_name` is the only field this class sets, and it is the whole
    point. Hound arms 1 and 2 trained under ANYmal's agent config verbatim, so
    their runs filed themselves under `logs/rsl_rl/anymal_c_rough/` — three
    Hound seeds indistinguishable from ANYmal runs by path alone. This task does
    not repeat that: it files under `hound_forward_v5/`, which names the robot,
    the reward and the ground.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "hound_forward_v5"


@configclass
class HoundOvernightPPORunnerCfg(HoundForwardV5PPORunnerCfg):
    """The commanded Hound long run. Same agent as the forward probe, new run dir.

    Subclasses the FORWARD-V5 Hound runner rather than re-deriving from
    `AnymalCRoughPPORunnerCfg`, and the shared claim is specific: these two tasks
    run the SAME BODY on the SAME GROUND, so the only thing that makes their
    wall-clock numbers comparable is that the network, the rollout length
    (`num_steps_per_env` 24) and the update size are identical by construction.
    The launch ceiling for this run is sized off the forward probe's measured
    iteration time (`runs/hound_forward_v5_s1/box_console.log`, 4096 envs,
    ~2.5 s/iter in steady state), and that arithmetic is only valid while the
    agent config is inherited rather than restated.

    `experiment_name` moves and it MUST. rsl_rl files runs under
    `logs/rsl_rl/<experiment_name>/`, and `hound_forward_v5/` already holds the
    2026-08-08 probe — a run whose reward is `v_x` alone and whose policy cannot
    read a command. Two different rewards' checkpoints in one directory is the
    `anymal_c_rough` mis-filing this file exists to stop, and here it would be
    worse than indistinguishable: both runs are Hound-on-v5 at 4096 envs, so the
    directory would read as one lineage and is two.

    MAX_ITERATIONS IS NOT PINNED HERE, for `SpyderOvernightPPORunnerCfg`'s
    reason: it stays at the inherited 1500 and is overridden per launch, so the
    run length is a property of the launch line — which is where the declared
    wall-clock ceiling that authorises it also lives.

    SAVE_INTERVAL: INHERITED AT 50, and the disk arithmetic is this body's, not
    the Spyder's. rsl_rl saves at every iteration index divisible by
    `save_interval` and once more at the end, so N iterations produce
    `floor((N-1)/50) + 2` checkpoints. Measured on the forward probe, the same
    network on the same observation: 31 files of 6,987,125 B each
    (`runs/hound_forward_v5_s1/box_logs/2026-08-08_05-13-54/model_*.pt`), and
    `floor(1499/50) + 2 = 31` reproduces the count exactly. At 12,000 iterations
    that gives **241 checkpoints x 6,987,125 B = 1,683,897,125 B = 1.68 GB
    (1.57 GiB)**, plus a tensorboard event file that should land near 15 MB.
    Comfortably inside the run budget, so the interval is inherited rather than
    set — one fewer number that differs from the probe beside it.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "hound_overnight"


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


@configclass
class SpyderFastPPORunnerCfg(SpyderOvernightPPORunnerCfg):
    """The fine-tune: the overnight agent, a wider command box, a new log tree.

    Subclasses the OVERNIGHT runner rather than the gentle one, and that is the
    same safety argument every class above makes, one rung further along: this
    task's claim is that it changes exactly one thing against the overnight run
    (the command ranges), so its PPO hyperparameters must be identical to that
    run's by construction. Inheriting from the config the checkpoint was trained
    under also matters mechanically here in a way it does not for a fresh arm —
    the fine-tune restores an optimiser state that was built against these
    hyperparameters, and a network-size or clip change would either fail the
    strict load or silently resume under a different algorithm.

    `experiment_name` moves and it MUST. rsl_rl files runs under
    `logs/rsl_rl/<experiment_name>/`, so a fine-tune left at `spyder_overnight`
    would write its checkpoints into the 15,000-iteration run's own log tree —
    beside, and eventually intermixed with, the numbered checkpoints it is
    resuming FROM. That is worse than the `anymal_c_rough` mis-filing this file
    exists to stop: there the two runs were merely indistinguishable, here the
    later run's `model_15000.pt` onwards would extend a sequence that reads as
    one continuous training curve and is not one.

    MAX_ITERATIONS STAYS OFF THE CLASS, for `SpyderOvernightPPORunnerCfg`'s
    reason, with one addition specific to resuming. `OnPolicyRunner.load`
    restores `current_learning_iteration` from the checkpoint, and
    `OnPolicyRunner.learn` runs `range(start_it, start_it + num_learning_iterations)`
    — so `--max_iterations N` on a resumed run means N ADDITIONAL iterations,
    not "train until iteration N". Pinning a number in the class would make
    that arithmetic invisible at the launch line, which is the one place the
    wall-clock ceiling is declared.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "spyder_fast"
