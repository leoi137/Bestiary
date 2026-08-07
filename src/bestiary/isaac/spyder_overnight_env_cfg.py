"""Spyder-12, gentle terrain: the long run, five reward terms, strafe on.

    Bestiary-Overnight-Spyder-v0        the long-run training config
    Bestiary-Overnight-Spyder-Play-v0   few robots, no noise, no shoving

ONE TASK, AND WHERE ITS REWARD TABLE COMES FROM
-----------------------------------------------
`research/episodes/014-one-term-buys-speed.md` reduced the gentle task's eleven
reward terms to one — base-frame `v_x` — and the machine learned to run 44.7 m
per 20 s episode, played back at 4.2-5.4 m/s, with a bounding, airborne,
command-deaf gait. `spyder_ladder_env_cfg.py` then asked 014's open question —
which single penalty buys the most survivable gait *on top of a task the
machine can actually be driven with* — and three arms answered it.

The ladder's numbers, read from the arms' own event files
(`runs/spyder_ladder_s1/spyder_ladder_{bare,actionrate,tilt}/*/events.out.tfevents.*`),
1500 iterations, 4096 envs, ONE seed (seed 1) per arm:

    rung         Train/mean_reward   Train/mean_episode_length
                 final   last-10     final   last-10
    bare         17.45   16.77       866.4   818.8
    actionrate   18.01   17.33       869.7   847.9      <- the winner
    tilt         17.13   16.65       847.8   832.0

Two readings and one warning, in that order.

The reading that matters: `actionrate` scored the highest total reward *while
paying a term the control does not pay*. Reward is income minus penalty, so an
arm that carries an extra tax and still outscores the untaxed control must have
strictly more tracking income than it — the +0.56 gap understates the income
gap by exactly the action-rate tax. The ordering is the same on the final
iteration and on the mean of the last ten, so it is not a peak read against a
mean.

The warning: **this is one seed per arm, so it is a probe, not a finding**
(`CLAUDE.md`, the seed rule: no effect is claimed from fewer than three seeds
per arm). The winner's margin over the control is 3.2% of reward and 0.4% of
episode length; between-seed spread on this stack has never been measured, and
both gaps are small enough that a second seed could reorder them. What follows
therefore inherits `action_rate_l2` as *the measured winner of a single-seed
probe*, which is the best evidence in hand and is not the same thing as a
demonstrated effect. If a later multi-seed ladder reverses the ordering, this
task's term set is what has to change.

THE FIVE TERMS
--------------
    track_lin_vel_xy_exp   +1.0     income, std 0.3   (the ladder's income)
    track_ang_vel_z_exp    +0.5     income, std 0.4   (the ladder's income)
    action_rate_l2         -0.01    the ladder's measured winner
    feet_air_time          +0.125   threshold 0.5 s, on the tibias
    lin_vel_z_l2           -2.0     vertical torso speed, squared

The first three are the winning rung, imported from the ladder module rather
than retyped — `rung_reward_terms(WINNING_RUNG)` — so "the winner plus two" is
true by construction and cannot drift into a hand-copied list that is no longer
the winner's.

The two added terms are the two in the gentle table that price *the shape of a
step*, and neither was on the ladder:

  * `feet_air_time` is the only term in the entire table that reads foot
    CONTACT TIMING. It pays `sum(air_time - 0.5 s)` over the feet that touched
    down this step, while a linear command is active, so it prices how long
    each foot spends off the ground. Its arithmetic is worth stating rather
    than assuming: at a 50%-duty gait of f Hz per foot the term is worth
    `w·4·f·(0.5/f - 0.5)` per second — zero at 1 Hz per foot, NEGATIVE above
    it (`check_spyder.check_the_money` prints both: +0.000000 at 1 Hz and
    -0.005000 per policy step at 2 Hz, against +0.030000 of income).
    So at upstream's 0.5 s threshold it does not simply reward stepping;
    it rewards LONG swings and taxes a fast shuffle. Against 014's gait — a
    body thrown forward with no identifiable swing phase — that is the term
    that asks for a step to be a step.

  * `lin_vel_z_l2` is the largest-magnitude penalty in the whole gentle table
    (-2.0, 200x `action_rate_l2`) and it prices squared vertical velocity of
    the torso. 014's failure was described as bounding and airborne; a body
    that leaves the ground has vertical speed on the way up and again on the
    way down, and this is the term that charges for both. `ang_vel_xy_l2` — the
    ladder's losing rung — prices the torso TUMBLING; this one prices it
    LEAVING, which is a different failure and an untested one.

WHAT IS DELETED, AND WHY EACH
-----------------------------
The gentle task pays eleven terms. Five are kept; these six are deleted:

    ang_vel_xy_l2       -0.05      the ladder's losing rung: it scored below
                                   the untaxed control on both reward and
                                   episode length, so it is the one term the
                                   ladder gives a reason NOT to carry.
    undesired_contacts  -1.0       femur contact. Worth 66.7% of the tracking
                                   terms' ceiling if sustained (`check_the_money`
                                   prints it) with the frequency never measured,
                                   and the Hound deleted its analogue after
                                   measuring 106%.
                                   A 15,000-iteration run is the wrong place to
                                   find out.
    dof_torques_l2      -1e-05     joint-effort taxes, both silent in the mean:
    dof_acc_l2          -2.5e-07   `check_the_money` prices them at -0.000002
                                   and -0.000006 per policy step against
                                   +0.030000 of income (0.008% and 0.02%) at its
                                   assumed rms operating points — awake only at
                                   the extremes. Neither was on the ladder, and
                                   they are deleted for the ladder's reason: an
                                   unmeasured term is not carried into a long
                                   run just because upstream shipped it.
    dof_pos_limits       0.0       weight EXACTLY zero in the gentle table.
    flat_orientation_l2  0.0       weight EXACTLY zero in the gentle table.

The last two deserve their own sentence: at weight 0.0 they contribute exactly
nothing to the reward, so deleting them cannot change what the machine is paid.
They are removed so that the live table and the intended table are the same
list — a zero-weight term in a config is a term someone will one day switch on
by accident. This also means the run's reward differs from the winning ladder
rung's by exactly two terms with nonzero weight (`feet_air_time`,
`lin_vel_z_l2`) and from the gentle task's by exactly four (the four above).

Every weight and every param is the gentle task's, unchanged, and this file
never types one: the surgery keeps the term OBJECTS the gentle config built,
with its body-name retargets (`feet_air_time` onto the tibias) already applied.
`research/decisions/0004` Part A is the standing reason — keep upstream's
recipe knowingly, stop calling it tuned.

The penalty budget needs no new arithmetic. `check_spyder.check_the_money`
computes the gentle table's recurring penalties at labelled operating points
and holds them under `research/decisions/0005`'s 30%-of-income flag; this table
is a strict SUBSET of that one at identical weights, so its recurring penalties
are strictly smaller and the same flag is satisfied a fortiori.

THIS IS A PRODUCTION RUN, NOT AN EXPERIMENT — SAID PLAINLY
-----------------------------------------------------------
Two terms are added at once. If the resulting gait is worse than the winning
rung's, **this run cannot say which of the two did it**, and nothing written
from it may claim otherwise. That is a deliberate trade: the ladder was the
one-variable-at-a-time instrument and it has been run, this is the run that
spends its answer at 10x the iteration count. The single-arm, single-seed
design means the only claims it can support are about ITS OWN policy — what
that policy does, measured — never about which term caused what.

STRAFE, IDENTICAL TO THE LADDER
-------------------------------
`lin_vel_y = ±VY_MAX_MS`, imported from `spyder_ladder_env_cfg` rather than
restated, so there is one copy of the number and its derivation. The gentle
task pins the lateral range to (0, 0); the ladder opened it to ±0.4 m/s and
carries the whole argument in its module docstring — the commanded box's corner
at sqrt(0.6² + 0.4²) = 0.721 m/s, the missing y dead zone, the tightened
discrimination ratio, and the fact that `track_lin_vel_xy_exp` already reads
and prices the y channel so no new term is needed. Every one of those sentences
applies here unchanged, and this task's command config is byte-identical to a
ladder rung's.

WHAT IS INHERITED UNTOUCHED
---------------------------
Everything else, from `SpyderGentleEnvCfg`: the robot, the terrain mix and its
0.1 m sampling, the v_x and w_z ranges with their dead zones, heading mode off,
the arc-corrected terrain curriculum, the observations (235 wide — a one-way
door, and the reason a 15,000-iteration run can be started at all: nothing here
moves it), the actions, the events and reset scatter, and the terminations. A
fall reset is episode-reset machinery, not a reward, and nothing pays or
charges for it.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg, SpyderGentleEnvCfg_PLAY
from bestiary.isaac.spyder_ladder_env_cfg import (
    INCOME_TERMS,
    apply_keep_list_and_strafe,
    rung_reward_terms,
)

# `VY_MAX_MS` is deliberately NOT imported here. The strafe range is applied by
# `apply_keep_list_and_strafe`, which reads the ladder's constant directly, so
# this module never names the number — there is exactly one copy of ±0.4 m/s and
# one copy of its derivation, both in `spyder_ladder_env_cfg.py`.

#: The ladder rung whose reward table this task starts from: the arm that
#: scored highest on both reward and episode length at 1500 iterations
#: (docstring's table). Named, not inlined, so the provenance is a lookup into
#: the ladder rather than a claim in prose.
WINNING_RUNG = "actionrate"

#: The two terms added on top of the winner, in the order the docstring argues
#: them: foot contact timing, then vertical torso speed. Both are the gentle
#: task's own terms at the gentle task's own weights — this module adds no term
#: that the gentle table does not already carry, which is what makes "at the
#: gentle weights" a checkable statement rather than an intention.
GAIT_TERMS: tuple[str, str] = ("feet_air_time", "lin_vel_z_l2")

#: The complete reward table, DERIVED: the winning rung's terms (both income
#: kernels plus `action_rate_l2`) followed by the two gait-shaping terms. If a
#: later multi-seed ladder changes which rung wins, `WINNING_RUNG` is the one
#: line that moves and this table follows it.
OVERNIGHT_TERMS: tuple[str, ...] = rung_reward_terms(WINNING_RUNG) + GAIT_TERMS

#: What the keep list removes from the gentle table, with the gentle weight of
#: each. NOT used by the surgery — the surgery deletes whatever is live and not
#: kept, so a term Isaac Lab adds tomorrow is deleted tomorrow without a code
#: change. This is the DOCUMENTED EXPECTATION, and `check_spyder.py` asserts the
#: live gentle table minus `OVERNIGHT_TERMS` equals exactly these six names. The
#: split is deliberate: the config is safe by construction, and the oracle is
#: what goes red so a human finds out an upstream release moved the table.
EXPECTED_DELETED_TERMS: tuple[str, ...] = (
    "ang_vel_xy_l2",
    "dof_acc_l2",
    "dof_pos_limits",
    "dof_torques_l2",
    "flat_orientation_l2",
    "undesired_contacts",
)


def apply_overnight(cfg) -> None:
    """Turn a constructed gentle config into the long-run task, in place.

    The surgery is the ladder's `apply_keep_list_and_strafe` over
    `OVERNIGHT_TERMS`: keep exactly the declared names with the gentle config's
    own term objects, None out everything else whatever it is called, and open
    the lateral command range to the ladder's ±VY_MAX_MS. Called by both the
    training class and the Play twin, for the reason that function's docstring
    gives — a Play config inherits the gentle PLAY overrides, so it cannot also
    inherit this variant's, and a surgery that reached one class and not the
    other would let the viewer play a different reward than the one that
    trained.
    """
    apply_keep_list_and_strafe(cfg, OVERNIGHT_TERMS, "the overnight task")


# Import-time, not test-time: these three facts are what the module docstring
# claims, and a launch that violates one should fail at the first import —
# which is every hydra launch and every oracle run — rather than train for
# hours under a reward table nobody declared.
if set(INCOME_TERMS) - set(OVERNIGHT_TERMS):
    raise AssertionError(
        f"the overnight table {sorted(OVERNIGHT_TERMS)} drops "
        f"{sorted(set(INCOME_TERMS) - set(OVERNIGHT_TERMS))} — without both "
        "command-tracking kernels the policy cannot be driven, which is the one "
        "property every task descended from the ladder exists to keep."
    )
if not set(rung_reward_terms(WINNING_RUNG)) <= set(OVERNIGHT_TERMS):
    raise AssertionError(
        f"the overnight table {sorted(OVERNIGHT_TERMS)} does not contain the "
        f"winning rung {WINNING_RUNG!r} = {sorted(rung_reward_terms(WINNING_RUNG))}. "
        "This task is defined as that rung plus two terms; if it no longer is, "
        "the ladder's measurement is not its provenance."
    )
if len(set(OVERNIGHT_TERMS)) != len(OVERNIGHT_TERMS):
    raise AssertionError(
        f"the overnight table repeats a term: {OVERNIGHT_TERMS}. A duplicate "
        "here is silent — the keep list is a set — and would mean the winning "
        "rung and GAIT_TERMS overlap, so the table is one term shorter than it "
        "reads."
    )


@configclass
class SpyderOvernightEnvCfg(SpyderGentleEnvCfg):
    """The long run: income + `action_rate_l2` + `feet_air_time` + `lin_vel_z_l2`."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_overnight(self)


@configclass
class SpyderOvernightEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """The long run for the viewer: few robots, nothing random."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_overnight(self)
