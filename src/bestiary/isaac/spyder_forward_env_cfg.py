"""Spyder-12, gentle terrain, ONE reward: forward speed. The stack diagnostic.

    Bestiary-Forward-Spyder-v0        the diagnostic config
    Bestiary-Forward-Spyder-Play-v0   few robots, no noise, no shoving

WHAT THIS VARIANT IS FOR
------------------------
`spyder_gentle_env_cfg.py` trains under eleven inherited reward terms. That
table is a control, not a measurement (`research/decisions/0004` Part A), and
it is entangled with every other unproven piece of the Isaac port: the
MJCF→USD transform, the PD drive constants, the height-scan footprint, the
terrain curriculum, ANYmal's PPO hyperparameters. If the gentle task fails to
walk, the failure has eleven plausible authors and no way to tell them apart.

So this config removes the reward from the suspect list by removing the
reward. The whole table is replaced by base-frame `v_x` at weight 1.0 —
the unshaped objective the 2016-era MuJoCo benchmarks solved with SAC, and the
objective this repository's own Spyder-12 walked the desert under
(`research/learnings/001`, 0.37 m/s). `isaac/rewards.py` carries the full
rationale, including the one way this term is *stricter* than 001's baseline
(no alive bonus, no control cost, so neither of 001's confounders exists).

    If this stack cannot reproduce forward walking from reward = v_x,
    the fault is in the stack, not in the reward design.

**The operator ordered this variant explicitly (2026-08-06).**

EXACTLY ONE VARIABLE MOVES, AND THE FILE IS SHAPED TO PROVE IT
--------------------------------------------------------------
Everything else is inherited from `SpyderGentleEnvCfg` untouched: the robot,
the terrain mix and its 0.1 m sampling, the dead-zoned commands, the arc-
corrected terrain curriculum, the observations (235 wide — a one-way door;
it does not move for a diagnostic), the actions, the events and reset scatter,
and the terminations. `check_spyder.py`'s `forward-variant-changes-only-reward`
asserts that literally: it dumps both configs with `to_dict()` and requires the
difference to be the `rewards` key and nothing else.

Terminations in particular STAY. A fall reset is episode-reset machinery, not
a reward — nothing pays or charges for it, and deleting it would let a
collapsed machine slide downhill accumulating v_x forever, which is a
different experiment.

THE COMMANDS ARE NOW UNREWARDED, AND THAT IS DELIBERATE — WITH ONE CAVEAT
-------------------------------------------------------------------------
`v_x` does not read the command, so the command channel in the observation is
paid for nothing and the policy is free to ignore it. That is the point: this
run measures whether the machine can walk at all, not whether it obeys. The
commands stay in the config anyway because removing them would move the
observation width, and the observation width is a one-way door that orphans
every checkpoint (`CLAUDE.md`, Invariants).

The caveat, stated so it is not discovered: `curriculums.terrain_levels_vel_arc`
still reads the command to compute its demote bar. A forward-only policy
commanded to turn drives straight instead, so its displacement generally
EXCEEDS half the arc bar and it promotes; a standing command has a reachable
distance of 0, which no displacement can fall below, so it never demotes
either. The interaction therefore biases the curriculum toward promotion, not
toward the yaw-blindness trap `curriculums.py` was written to close. Left as
is, per the order to change one variable.

WHAT THE TENSORBOARD NUMBER MEANS HERE, WHICH IS UNUSUALLY DIRECT
------------------------------------------------------------------
`RewardManager` scales every term by `weight * step_dt`, so at weight 1.0 the
per-step reward is `v_x * 0.02 s` = metres travelled that step, and the
EPISODE RETURN IS METRES OF FORWARD TRAVEL. The 0.37 m/s of `learnings/001`
over a 20 s episode is a return of 7.4; the gentle task's seed 1 reached a mean
return of 14.17 under a completely different (kernel-based, bounded) reward, so
the two numbers are not comparable and must never be put in the same column.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from bestiary.isaac import rewards
from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg, SpyderGentleEnvCfg_PLAY

#: The one term's name, weight, and the function it must resolve to. Named as
#: constants because three places assert against them: this module's own
#: post-init guard, the oracle's check, and anything reading the run later.
REWARD_TERM_NAME = "forward_velocity"

#: Weight 1.0. Not tuned and not tunable — there is no second term to trade
#: against, so the scale is a pure relabelling of the return. 1.0 is chosen
#: because `RewardManager` multiplies by `step_dt`, which makes the episode
#: return read as metres of forward travel (see the module docstring).
REWARD_WEIGHT = 1.0


@configclass
class ForwardOnlyRewardsCfg:
    """The entire reward: one term.

    Written as a fresh config class rather than as `RewardsCfg` with ten terms
    set to `None`, and the difference is the point. Upstream's `RewardsCfg` is
    a moving target — Isaac Lab has added terms to it across releases — and a
    deletion list silently stops being complete the day a twelfth term ships,
    re-shaping this diagnostic without anyone editing this file. A class with
    one field cannot acquire a term it does not declare.

    `RewardManager` reads `cfg.__dict__`, not a declared type, so nothing
    downstream requires this to descend from upstream's class.
    """

    forward_velocity: RewTerm = RewTerm(func=rewards.forward_velocity, weight=REWARD_WEIGHT)


def single_reward_term(rewards_cfg) -> tuple[str, RewTerm]:
    """The one live (non-`None`) reward term, or a loud failure naming them all.

    Shared by this module's post-init guard and `check_spyder.py`, so the
    oracle and the config cannot disagree about what "exactly one term" means.
    """
    live = {
        name: term
        for name, term in vars(rewards_cfg).items()
        if not name.startswith("_") and term is not None
    }
    if len(live) != 1:
        raise AssertionError(
            f"the forward diagnostic must carry exactly ONE reward term; this "
            f"config has {len(live)}: {sorted(live)}. Any second term makes the "
            "run a two-variable experiment and forfeits the whole diagnostic."
        )
    return next(iter(live.items()))


def use_forward_velocity_only(cfg) -> None:
    """Replace `cfg.rewards` wholesale with the single forward-velocity term.

    Called after `super().__post_init__()`, so it discards the gentle config's
    kernel-width and body-name retargets along with the terms they targeted —
    which is correct: those retargets exist to keep `feet_air_time` and
    `undesired_contacts` resolvable, and neither term survives here.
    """
    cfg.rewards = ForwardOnlyRewardsCfg()
    name, term = single_reward_term(cfg.rewards)
    if name != REWARD_TERM_NAME:
        raise AssertionError(f"the surviving reward term is {name!r}, not {REWARD_TERM_NAME!r}")
    if term.func is not rewards.forward_velocity:
        raise AssertionError(
            f"{name}.func is {getattr(term.func, '__name__', term.func)!r}, not "
            "bestiary.isaac.rewards.forward_velocity"
        )
    if term.weight != REWARD_WEIGHT:
        raise AssertionError(f"{name}.weight is {term.weight}, not {REWARD_WEIGHT}")


@configclass
class SpyderForwardEnvCfg(SpyderGentleEnvCfg):
    """The gentle config with its entire reward table replaced by `v_x`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_forward_velocity_only(self)


@configclass
class SpyderForwardEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """Viewer config for the diagnostic: few robots, nothing random.

    Descends from `SpyderGentleEnvCfg_PLAY`, NOT from `SpyderForwardEnvCfg`,
    so the play overrides (16 envs, native terrain sampling, corruption and
    pushes off) are inherited rather than copied — a copy is what drifts. The
    cost is that a later change to `SpyderForwardEnvCfg` does not reach here;
    both classes call `use_forward_velocity_only`, which is the only thing the
    variant actually changes, so keep it that way. If this variant ever grows a
    second override, put it in that function too.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        use_forward_velocity_only(self)
