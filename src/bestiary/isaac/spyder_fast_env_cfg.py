"""Spyder-12, gentle terrain: the overnight task with a WIDER command box.

    Bestiary-Fast-Spyder-v0        the fine-tune config
    Bestiary-Fast-Spyder-Play-v0   few robots, no noise, no shoving

WHAT THIS IS: A FINE-TUNE, AND THE RECORD MUST SAY SO
------------------------------------------------------
This task is not a fresh arm. It exists to continue training the policy the
overnight run produced — `Bestiary-Overnight-Spyder-v0`, 15,000 iterations,
4096 envs, seed 1, finishing at mean reward 18.39, mean episode length 934.48
of 1000, `Metrics/base_velocity/error_vel_xy` 0.2245
(`runs/spyder_overnight_s1/box_console.log`, final iteration block) — under
command ranges the overnight policy never saw. Its checkpoint is loaded and
its optimiser state is restored; nothing about the network, the observation or
the reward moves. Any ledger row, learning or episode written from a run of
this task states that it INHERITED weights, and from which checkpoint. A
fine-tune reported as an arm would make its numbers look like evidence about
the reward table when they are evidence about one policy's continued training.

ONE VARIABLE: THE COMMAND RANGES
--------------------------------
Everything except three numbers is `SpyderOvernightEnvCfg`, by inheritance
rather than by restatement — the reward table (income + `action_rate_l2` +
`feet_air_time` + `lin_vel_z_l2`), the robot, the terrain mix, the arc-corrected
terrain curriculum, the observations (235 wide, a one-way door), the actions,
the events, the reset scatter, the terminations, and both dead zones.

    channel        overnight        fast            factor
    lin_vel_x      ±0.6 m/s         ±1.5 m/s        2.5x
    lin_vel_y      ±0.4 m/s         ±0.6 m/s        1.5x
    ang_vel_z      ±0.8 rad/s       ±1.5 rad/s      1.875x

The dead zones are NOT touched: `min_lin_vel_x` stays 0.25 m/s and
`min_ang_vel_z` stays 0.2 rad/s. That is deliberate and it changes what they
mean. The v_x resample maps |u| ~ U(0, hi) onto U(0.25, hi), so widening `hi`
moves the driving distribution up rather than merely stretching it: the median
commanded speed goes from 0.425 to 0.875 m/s. The yaw snap keeps its 0.2 rad/s
threshold, so the fraction of driving envs snapped to exactly straight falls
from 0.2/0.8 = 25% to 0.2/1.5 = 13.3% — see the curriculum section below,
because those are the envs that carry promotion.

THE KERNEL WIDTHS ARE DELIBERATELY NOT MOVED, AND THIS IS THE COST
-------------------------------------------------------------------
`spyder_gentle_env_cfg.py` derives its kernel widths by preserving upstream's
discrimination ratio std/v_max = 0.5: std_lin = 0.3 against ±0.6 m/s,
std_ang = 0.4 against ±0.8 rad/s. This task widens the ranges and leaves both
stds exactly where they are, so that ratio moves:

    linear    0.3 / 0.6 = 0.500   ->   0.3 / 1.5 = 0.200
    angular   0.4 / 0.8 = 0.500   ->   0.4 / 1.5 = 0.267
    2-D corner of the commanded box:
              sqrt(0.6² + 0.4²) = 0.721 m/s   ->   sqrt(1.5² + 0.6²) = 1.616 m/s
              0.3 / 0.721 = 0.416             ->   0.3 / 1.616 = 0.186

Read that as TIGHTER RELATIVE TRACKING AT SPEED. The kernel `exp(-(e/std)²)`
is a function of the ABSOLUTE error, so nothing about what a given error costs
has changed — a 0.25 m/s error is worth exp(-(0.25/0.3)²) = 0.50 of the linear
term today and was worth the same yesterday. What changes is that the errors
themselves get larger when the command does: a machine that tops out at 0.6 m/s
and is commanded 1.5 m/s eats a 0.9 m/s error and collects exp(-9) = 1.2e-4 —
effectively nothing. The tracking income at the top of the new box is therefore
close to all-or-nothing, which is the pressure this task exists to apply.

Moving std with the range would be a SECOND VARIABLE, and it would point the
opposite way: std_lin = 0.75 would restore the 0.5 ratio and make that same
0.9 m/s shortfall worth exp(-1.44) = 0.24 instead of 1.2e-4, i.e. it would pay
a machine for continuing to walk at its old speed. One of those two changes is
the experiment and the other is its confound, so only the ranges move here. If
the fine-tune stalls — reward collapses and stays collapsed because the top of
the box is unreachable and pays nothing — the kernel width is the first thing
to reconsider, and it is reconsidered as its own one-variable change with its
own launch, not folded into this one.

One consequence worth stating rather than discovering: standing gets CHEAPER
to punish. `check_spyder.check_the_money`'s expected drive-cell share for a
motionless machine falls from 27.2% to 11.6% under the wider box, because a
parked machine now eats a larger commanded error on both channels. The
parked-seed door this stack was built to close (`commands.py`) closes further,
not less.

WHAT THE WIDER BOX DOES TO THE TERRAIN CURRICULUM
--------------------------------------------------
`curriculums.terrain_levels_vel_arc` promotes an env whose robot ends the
episode more than `tile/2` = 4 m from spawn, and demotes one that displaced
less than half of `arc_displacement_m(|v|, w, T)` = 2(|v|/w)|sin(wT/2)|, which
reduces to |v|T for w = 0. **That formula is correct for any commanded speed —
it is kinematics, not a tuned constant — so nothing in `curriculums.py` needs
to change.** What changes is where the two bars now sit:

  * The PROMOTE bar gets EASIER to clear. It is a fixed 4 m of displacement, so
    at the top straight command it takes 4/1.5 = 2.67 s of driving instead of
    4/0.6 = 6.67 s of it, against a 20 s episode. And a turning env promotes
    only if its arc DIAMETER 2v/w exceeds 4 m, i.e. only if w < v/2: at the top
    forward command that threshold moves from 0.3 to 0.75 rad/s, so the share
    of the yaw range above the 0.2 rad/s snap that can promote at all goes from
    12.5% to 36.7%. Pulling the other way, the snap now makes only 13.3% of
    driving envs straight rather than 25%. The net effect on promotion rate is
    a training outcome to read off `Curriculum/terrain_levels`, not a claim
    this docstring is entitled to make.

  * The DEMOTE bar gets HARDER, and it is the one to watch. For a straight
    command it is half the commanded distance: 0.5 x 1.5 x 20 = 15 m, against
    0.5 x 0.6 x 20 = 6 m before. A policy that inherits the overnight run's
    ~0.6 m/s legs and is commanded 1.5 m/s covers 12 m against a 15 m bar and
    is demoted, so terrain levels should be expected to FALL early in the
    fine-tune and recover only as speed arrives. The overnight run ended at
    `Curriculum/terrain_levels` 2.9177; a dip below that in the first hundreds
    of iterations is the mechanism working, not a fault. If it never recovers,
    the machine did not find the speed and the run says so.

TERRAIN AND SENSING, THE PART THAT IS NOT FREE
-----------------------------------------------
The height scanner is inherited untouched — it must be, the observation is 235
wide and a one-way door, and the whole point of a fine-tune is that the
checkpoint loads. `spyder_gentle_env_cfg.py` sized its 2.56 x 1.6 m footprint
so the 1.28 m of forward reach is "1.28 m of terrain ahead at commanded speeds
up to 0.6 m/s". At 1.5 m/s that same 1.28 m is 0.85 s of lookahead instead of
2.13 s. Nothing here can widen it without orphaning the checkpoint, so the
honest statement is: this task asks the machine to move 2.5x faster over the
same gentle mix while seeing 2.5x less far ahead in time. That is a real
limitation of the fine-tune and a reason a failure at speed on rough tiles
should not be read as a reward-table failure.

The lateral channel keeps the property `spyder_ladder_env_cfg.py` documents:
`DeadZoneVelocityCommand` remaps only v_x and w_z, so v_y ~ U(-0.6, 0.6)
including the near-zero band the other two channels exclude. Widening y makes
strafe LESS optional, not more — a perfect v_x tracker that never sidesteps now
collects 44.1% of the linear kernel in expectation, down from 62.5% at ±0.4
(`check_spyder`'s fast check prints both). Giving v_y its own dead zone is
still the obvious repair and is still deliberately not made here, for the
reason it was not made on the ladder: it would move a second variable.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.spyder_gentle_env_cfg import (
    STD_ANG,
    STD_LIN,
    VX_MAX_MS as OVERNIGHT_VX_MAX_MS,
    WZ_MAX_RADS as OVERNIGHT_WZ_MAX_RADS,
)
from bestiary.isaac.spyder_ladder_env_cfg import (
    VY_MAX_MS as OVERNIGHT_VY_MAX_MS,
    live_reward_names,
)
from bestiary.isaac.spyder_overnight_env_cfg import (
    OVERNIGHT_TERMS,
    SpyderOvernightEnvCfg,
    SpyderOvernightEnvCfg_PLAY,
)

#: The widened envelope. Three numbers, and they are the entire task.
#: 2.5x forward, 1.5x lateral, 1.875x yaw against the overnight run's box.
#: The three are named separately rather than as one scale factor because they
#: are not one decision. v_x is the axis the run exists to move. w_z is widened
#: because the tightest constant-twist radius a command can ask for is v/w, and
#: leaving yaw at ±0.8 while v_x went to 1.5 would loosen it from 0.6/0.8 =
#: 0.75 m to 1.88 m — the machine would only be able to drive fast in near
#: straight lines; ±1.5 rad/s puts it back at 1.0 m. v_y trails both because the
#: lateral channel has no dead zone and its range is the one the record has
#: least evidence for.
VX_MAX_MS = 1.5
VY_MAX_MS = 0.6
WZ_MAX_RADS = 1.5

#: field name on `commands.base_velocity.ranges` -> the widened range.
#: A dict rather than three assignments, so the surgery, its post-condition and
#: `check_spyder.py`'s fast check all iterate ONE declaration of "which ranges
#: this task moves". A fourth entry added here is a fourth range checked, with
#: no second edit needed anywhere.
FAST_RANGES: dict[str, tuple[float, float]] = {
    "lin_vel_x": (-VX_MAX_MS, VX_MAX_MS),
    "lin_vel_y": (-VY_MAX_MS, VY_MAX_MS),
    "ang_vel_z": (-WZ_MAX_RADS, WZ_MAX_RADS),
}

#: The same three fields with the ranges the OVERNIGHT task commands, so
#: "widened, not narrowed" is checkable at import time and the widening factors
#: this task is named for have a source. Built from the constants the two
#: upstream modules own — `spyder_gentle_env_cfg` for x and yaw,
#: `spyder_ladder_env_cfg` for y — never retyped, so a change to either one
#: turns this module's import-time assertions red instead of leaving a stale
#: number behind.
OVERNIGHT_RANGES: dict[str, tuple[float, float]] = {
    "lin_vel_x": (-OVERNIGHT_VX_MAX_MS, OVERNIGHT_VX_MAX_MS),
    "lin_vel_y": (-OVERNIGHT_VY_MAX_MS, OVERNIGHT_VY_MAX_MS),
    "ang_vel_z": (-OVERNIGHT_WZ_MAX_RADS, OVERNIGHT_WZ_MAX_RADS),
}

#: The kernel widths this task must NOT move, imported rather than typed. The
#: docstring's whole argument for leaving them alone is that moving them would
#: be a second variable; this is the constant that makes "left alone" an
#: assertion rather than an intention.
UNMOVED_KERNEL_STDS: dict[str, float] = {
    "track_lin_vel_xy_exp": STD_LIN,
    "track_ang_vel_z_exp": STD_ANG,
}


def apply_fast(cfg) -> None:
    """Widen the command box on a constructed overnight config, in place.

    Called after `super().__post_init__()` by BOTH the training class and the
    Play twin, for the reason `spyder_ladder_env_cfg.apply_keep_list_and_strafe`
    gives: a Play config descends from the overnight Play class, so it cannot
    also inherit this variant's `__post_init__`, and a surgery that reached one
    class and not the other would let the viewer drive a policy under a
    different command distribution than the one it trained under.

    Three edits and four post-conditions. The post-conditions exist because
    every one of these failures is SILENT — a range that did not take still
    trains, as the overnight task under a different name; a dead zone that
    ended up outside its range still trains, with `DeadZoneVelocityCommand`
    either collapsing v_x onto the range edge or zeroing every yaw draw; a
    kernel std that drifted still trains, as a two-variable experiment; and a
    reward term that vanished still trains, as a different task entirely.
    """
    ranges = cfg.commands.base_velocity.ranges
    for name, value in FAST_RANGES.items():
        if not hasattr(ranges, name):
            raise AssertionError(
                f"the fast task wants to widen {name!r}, which "
                f"{type(ranges).__name__} does not have. It has "
                f"{sorted(n for n in vars(ranges) if not n.startswith('_'))}. "
                "Upstream renamed a command range, so this task would train on "
                "the overnight envelope under the fast task's name."
            )
        setattr(ranges, name, value)

    # (1) Every declared range took.
    got = {name: tuple(getattr(ranges, name)) for name in FAST_RANGES}
    if got != FAST_RANGES:
        raise AssertionError(
            f"the fast task ended up commanding {got}, not {FAST_RANGES} — the "
            "surgery in apply_fast did not take, and this is the overnight task "
            "relabelled."
        )

    # (2) Both dead zones still sit strictly inside their widened range. This
    # is `DeadZoneVelocityCommand.__init__`'s own validation, hoisted to config
    # time: there it raises when the command TERM is constructed, which is
    # after the app is up and minutes into a launch.
    cmd = cfg.commands.base_velocity
    for name, dz in (("lin_vel_x", cmd.min_lin_vel_x), ("ang_vel_z", cmd.min_ang_vel_z)):
        lo, hi = FAST_RANGES[name]
        if lo != -hi:
            raise AssertionError(
                f"the fast task's {name} range {(lo, hi)} is not symmetric; "
                "DeadZoneVelocityCommand treats both channels as a magnitude "
                "and a sign, so an asymmetric range biases the sign silently."
            )
        if not 0.0 <= dz < hi:
            raise AssertionError(
                f"the fast task's min_{name} = {dz} is outside [0, {hi}) — the "
                "v_x resample would collapse onto the range edge and the w_z "
                "snap would zero every draw."
            )

    # (3) The kernel widths did NOT move. The docstring argues at length that
    # leaving them alone is the experiment; this is the line that makes a
    # well-meaning "restore the 0.5 ratio" edit fail loudly instead of turning
    # the run into a two-variable one.
    for term_name, want_std in UNMOVED_KERNEL_STDS.items():
        term = getattr(cfg.rewards, term_name, None)
        if term is None:
            raise AssertionError(
                f"the fast task has no live {term_name!r} to check the kernel "
                "width of — the overnight reward table lost a tracking kernel, "
                "and a policy that cannot be driven cannot be asked to drive faster."
            )
        got_std = float(term.params["std"])
        if got_std != want_std:
            raise AssertionError(
                f"the fast task's {term_name} std is {got_std}, not the gentle "
                f"task's {want_std}. Widening the ranges is this task's ONE "
                "variable; moving the kernel width with them would pay the "
                "machine for keeping its old speed and make the run "
                "uninterpretable either way."
            )

    # (4) The reward table is still the overnight table, exactly. Implied by
    # (3) only for the two kernels; this covers the other three terms and, more
    # importantly, covers a term ARRIVING.
    live = live_reward_names(cfg.rewards)
    if live != set(OVERNIGHT_TERMS):
        raise AssertionError(
            f"the fast task pays {sorted(live)}, but the overnight table is "
            f"{sorted(OVERNIGHT_TERMS)}. This task inherits its reward and "
            "changes only the command ranges; a table that differs makes the "
            "fine-tune a two-variable change against the checkpoint it loads."
        )


def _assert_declaration_is_a_widening() -> None:
    """The two range tables agree on their channels, and every fast range is wider.

    Called at import — which is every hydra launch and every oracle run — so a
    declaration that violates the module docstring fails before the app boots
    rather than after hours of fine-tuning under an envelope nobody declared. A
    function rather than bare module-level statements so the loop variables do
    not survive as module attributes.
    """
    if set(FAST_RANGES) != set(OVERNIGHT_RANGES):
        raise AssertionError(
            f"the fast task widens {sorted(FAST_RANGES)} but the overnight "
            f"baseline is recorded for {sorted(OVERNIGHT_RANGES)}. Every widened "
            "range needs a baseline to be compared against, or the oracle checks "
            "a set of channels that is not the set that moved."
        )
    for name, (lo, hi) in FAST_RANGES.items():
        base = OVERNIGHT_RANGES[name]
        if lo != -hi:
            raise AssertionError(
                f"the fast {name} range {(lo, hi)} is not symmetric; the sampler "
                "reflects its uniform draw through zero and would bias the sign."
            )
        if hi <= base[1]:
            raise AssertionError(
                f"the fast {name} range {(lo, hi)} does not WIDEN the overnight "
                f"range {base}. This task is defined as a widening; an equal "
                "range makes it the overnight task relabelled, and a narrower "
                "one makes its name a lie."
            )


_assert_declaration_is_a_widening()


@configclass
class SpyderFastEnvCfg(SpyderOvernightEnvCfg):
    """The fine-tune: the overnight reward, commanded ±1.5 / ±0.6 / ±1.5."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_fast(self)


@configclass
class SpyderFastEnvCfg_PLAY(SpyderOvernightEnvCfg_PLAY):
    """The fine-tune for the viewer: few robots, nothing random."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_fast(self)
