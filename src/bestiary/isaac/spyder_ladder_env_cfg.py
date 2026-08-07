"""Spyder-12, gentle terrain: a three-rung reward ladder between 1 term and 11.

    Bestiary-Ladder-Bare-Spyder-v0         income only, no penalty at all
    Bestiary-Ladder-ActionRate-Spyder-v0   income + action_rate_l2
    Bestiary-Ladder-Tilt-Spyder-v0         income + ang_vel_xy_l2
    ...-Play-v0                            few robots, no noise, no shoving

THE QUESTION, AND WHERE IT COMES FROM
-------------------------------------
`research/episodes/014-one-term-buys-speed.md` deleted ten of the gentle task's
eleven reward terms, kept base-frame `v_x`, and the machine learned to run:
44.7 m per 20 s episode by iteration 1499, played back at 4.2-5.4 m/s. It also
learned to run *badly* — a bounding, airborne, visibly violent gait that is
deaf to commands by construction, because `v_x` never reads one. 014's third
reading names the gap it left open:

    "What this does not show: that eleven terms are *needed* for grace. The
     gap between 1 and 11 is unexplored; the cheapest next experiment is a
     ladder — add one term at a time and watch which single term buys the
     most survivable gait."

This module is that ladder, with one correction to 014's sentence. 014 added
terms to `r = v_x`, and `v_x` is not a task: a policy trained on it cannot be
driven, so "the most survivable gait" would be measured on a machine nobody
can steer, and the winner would be un-deployable whatever it scored. So every
rung here pays the **full COMMAND-TRACKING income** — both of the gentle task's
kernels, at the gentle task's weights and kernel widths — and then adds **at
most one penalty**. The rungs are therefore steerable by construction, and the
thing being measured is what one penalty buys *on top of a task*, which is the
question that has an answer worth acting on.

THE THREE RUNGS, AND WHY THESE TWO PENALTIES
--------------------------------------------
    1. bare        income only.                      The control.
    2. actionrate  income + action_rate_l2  (-0.01)
    3. tilt        income + ang_vel_xy_l2   (-0.05)

Rung 1 is not filler. Reading rungs 2 and 3 requires knowing what the tracking
kernels alone already do to the gait: the forward diagnostic's violence was
measured under a reward with NO velocity target, and a kernel that peaks at
0.6 m/s (2.2 km/h) may by itself remove most of a 5 m/s bound. Without rung 1
any tameness seen in rungs 2 and 3 is unattributable.

Rungs 2 and 3 are the two terms 014's reading points at. Its diagnosis of the
violence was "nothing priced flailing, slamming, or airtime": `action_rate_l2`
prices flailing directly — it is the L2 norm of the change in the action vector
between consecutive policy steps, so it is a tax on the joint targets moving
fast, which is what a thrashing limb is. `ang_vel_xy_l2` prices the torso
tumbling — roll and pitch rate — which is what a bounding, airborne gait does
to a body. Nine terms were deleted; these are the two whose stated job most
directly matches the observed failure, so they are the two the ladder spends
its rungs on. The other seven remain unexplored, and this module claims
nothing about them.

Weights and params are the gentle task's, unchanged, in every case — the
ladder is a question about which term, not about what it should cost. Both are
upstream's legged_gym constants inherited through `LocomotionVelocityRoughEnvCfg`
(`research/decisions/0004` Part A: keep the recipe knowingly, stop calling it
tuned).

STRAFE IS ON, AND THE COMMAND SET GROWS BY EXACTLY ONE CHANNEL
---------------------------------------------------------------
The gentle task pins `lin_vel_y = (0.0, 0.0)` — the command SLOT exists in the
observation but is never asked for. `spyder_gentle_env_cfg.py` says why the
slot was reserved: "a spider can sidestep (legs, not wheels — the Hound's
geometric argument does not apply), and the slot in the observation is what
makes adding lateral commands later a config change instead of a one-way
obs-width door." This is later. The range opens to ±0.4 m/s and the
observation width does not move, exactly as that comment promised.

**No new reward term is needed and none is added.** `track_lin_vel_xy_exp` is
a 2-D kernel: it already reads the y channel of the command and already prices
the y channel of the error. Under `lin_vel_y = (0, 0)` that y term was pricing
uncontrolled sideways drift; under ±0.4 it prices tracking. Same term, same
weight, same std — a wider command distribution is the whole change.

±0.4 rather than ±0.6, and the number is a claim exactly the way the forward
ceiling is a claim. What the machine has to reach is the CORNER of the
commanded box, not either edge: with v_x ±0.6 and v_y ±0.4 the largest
commanded speed is |v| = sqrt(0.6^2 + 0.4^2) = 0.721 m/s, already 20% past the
0.6 m/s that `spyder_gentle_env_cfg.py` itself calls "the weakest number in
this file" — a stride-geometry estimate against a single measured 0.37 m/s
(`research/learnings/001`). Opening y to ±0.6 would put the corner at
0.849 m/s, 2.3x that measurement, and every rung would then be partly a test
of an unreachable command rather than of a penalty. ±0.4 is a compromise, not
a derivation: large enough that strafe is a mode rather than a rounding error
(the top y command is 67% of the top x command), small enough that the corner
grows by a fifth instead of a half. It is on the list of things the first
measured v_x/v_y span should replace.

TWO CONSEQUENCES OF STRAFE, BOTH STATED SO THEY ARE NOT DISCOVERED
-------------------------------------------------------------------
1. **The v_y channel has NO dead zone**, and this is a property of the
   inherited sampler, not a choice made here.
   `commands_impl.DeadZoneVelocityCommand._resample_command` calls the parent
   and then remaps index 0 (v_x magnitude resample) and index 2 (w_z snap);
   index 1 is left exactly as `UniformVelocityCommand` sampled it,
   `r.uniform_(*ranges.lin_vel_y)`. Standing envs are still zeroed on all
   three channels afterwards by the parent's `_update_command`, and
   `DeadZoneVelocityCommand.__init__` never validates `lin_vel_y` (it checks
   symmetry and dead-zone bounds on the x and z channels only; ±0.4 is
   symmetric regardless). So v_y ~ U(-0.4, 0.4) including the ambiguous
   near-zero band that both other channels exclude.

   That does NOT re-open the parked-seed door the dead zones were built to
   close. Standing is caught by the x channel: every driving env still carries
   |v_x| >= 0.25, so a motionless machine still eats at least a 0.25 m/s
   linear error no matter what v_y is. What the missing y dead zone does open
   is one rung down in severity — a machine that tracks v_x perfectly and
   simply never strafes collects E[exp(-(v_y/0.3)^2)] = 62.5% of the linear
   kernel across the y distribution (computed in `check_spyder.py`'s ladder
   check, which prints it). Strafe is therefore worth learning but not
   mandatory, and a rung that comes back walking straight has not failed the
   run — it has answered a different question. Giving v_y its own dead zone is
   the obvious repair and it is deliberately NOT made here: it would move a
   second variable across the ladder's three arms.

2. **The linear kernel's discrimination ratio tightens, and it is left
   tightened.** `spyder_gentle_env_cfg.py` derives std_lin = 0.3 by preserving
   upstream's std/range = 0.5 against a ±0.6 x range. Against the 2-D box's
   corner of 0.721 m/s that ratio is now 0.3/0.721 = 0.416 — a sharper kernel
   than upstream's operating point, not a slacker one, so it errs toward
   discriminating more. The std is NOT retuned: the ladder's rungs must differ
   from the gentle task in the reward table and the y range and nothing else,
   and `check_spyder.py`'s ladder check asserts that as a whole-config diff.
   `check_kernel_widths_preserve_the_ratio` still reads the x channel and
   still passes, which is correct — it is asserting the derivation that is
   actually in force on the axis it was derived for.

WHAT IS INHERITED UNTOUCHED
---------------------------
Everything else, from `SpyderGentleEnvCfg`: the robot, the terrain mix and its
0.1 m sampling, the v_x and w_z ranges with their dead zones, heading mode off,
the arc-corrected terrain curriculum, the observations (235 wide — a one-way
door), the actions, the events and reset scatter, and the terminations.
Terminations in particular stay: a fall reset is episode-reset machinery, not a
reward, and nothing pays or charges for it.

The terrain curriculum needs no change for strafe, which is worth one sentence
because it is not obvious. `curriculums.terrain_levels_vel_arc` computes its
demote bar from `norm(command[:, :2])` and the yaw rate, and a constant twist
with a lateral component traces the same circle of radius |v|/w as one without
— the arc formula 2(|v|/w)|sin(wT/2)| is already correct for a strafing
command, and reduces to |v|T for w = 0 exactly as before.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg, SpyderGentleEnvCfg_PLAY

#: The command-tracking income every rung keeps: both of the gentle task's
#: kernels, at the gentle task's weights (1.0 and 0.5, upstream's) and the
#: gentle task's rescaled stds (0.3 and 0.4). Nothing in this module touches
#: either term — they arrive fully configured from
#: `SpyderGentleEnvCfg.__post_init__` and are simply not deleted.
INCOME_TERMS: tuple[str, str] = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")

#: rung id -> the ONE penalty term added back, or None for the control rung.
#: The term is kept AT THE GENTLE TASK'S WEIGHT AND PARAMS, untouched: -0.01
#: for `action_rate_l2` and -0.05 for `ang_vel_xy_l2`, both upstream's
#: legged_gym constants inherited through `LocomotionVelocityRoughEnvCfg`.
RUNGS: dict[str, str | None] = {
    "bare": None,
    "actionrate": "action_rate_l2",
    "tilt": "ang_vel_xy_l2",
}

#: Lateral command envelope, m/s. The docstring's arithmetic: ±0.4 puts the
#: corner of the commanded box at |v| = 0.721 m/s, 20% past the gentle task's
#: already-unproven 0.6 m/s forward ceiling. Only this one number is new:
#: `VX_MAX_MS`, `VX_MIN_MS`, `WZ_MAX_RADS`, `WZ_MIN_RADS` and `REL_STANDING`
#: stay in force unchanged, inherited from the gentle config rather than
#: restated here, so there is no second copy of them to drift.
VY_MAX_MS = 0.4


def rung_reward_terms(rung: str) -> tuple[str, ...]:
    """The exact set of live reward terms a rung must end up with.

    One function, three readers — the config surgery, its own post-condition,
    and `check_spyder.py`'s ladder check — so the oracle and the config cannot
    disagree about what a rung IS. Raises on an unknown rung rather than
    returning the income terms, because a typo that silently produced the bare
    rung would make two arms of a three-arm comparison identical.
    """
    if rung not in RUNGS:
        raise AssertionError(f"unknown ladder rung {rung!r}; the ladder is {sorted(RUNGS)}")
    penalty = RUNGS[rung]
    return INCOME_TERMS if penalty is None else INCOME_TERMS + (penalty,)


def live_reward_names(rewards) -> set[str]:
    """The names of the reward terms a constructed config is actually paying.

    One definition of "live", used by the surgery, by its own post-condition and
    by `check_spyder.py`. A term is live iff it is a public attribute that is not
    `None`, which is exactly the test `RewardManager._prepare_terms` applies when
    it decides what to register.
    """
    return {
        name
        for name, term in vars(rewards).items()
        if not name.startswith("_") and term is not None
    }


def apply_keep_list_and_strafe(cfg, keep, what: str) -> None:
    """Reduce a constructed gentle config to a declared reward table, plus strafe.

    Two edits and two post-conditions, in place. Called after
    `super().__post_init__()` by both the training and the Play class of every
    variant — the same shared-mutator pattern
    `spyder_forward_env_cfg.use_forward_velocity_only` uses, and for the same
    reason: a Play config that inherits the gentle Play overrides cannot also
    inherit the variant's, so the variant has to live in a function both call.

    `what` names the caller in every failure message ("ladder rung 'tilt'", "the
    overnight task"), because the same surgery now serves two modules and a bare
    "the config" would not say which one is wrong.

    The reward surgery is a KEEP LIST, not a delete list, and that is the whole
    safety argument. `spyder_forward_env_cfg` met the same problem — upstream's
    `RewardsCfg` is a moving target, Isaac Lab has added terms to it across
    releases, and a list of nine deletions silently stops being complete the
    day a twelfth term ships, re-shaping every variant without anyone editing
    this file. It solved it with a fresh one-field config class; that is not
    available here, because these tasks must carry the gentle task's terms with
    the gentle task's exact weights, params and body-name retargets already
    applied, which means keeping the objects the gentle config built. So
    instead: enumerate what is live, keep exactly the declared names, None out
    everything else whatever it is called. A term upstream adds tomorrow is
    deleted tomorrow, without a code change.

    `RewardManager._prepare_terms` skips `None` terms (`continue` on
    `term_cfg is None`, reward_manager.py:226 in the pinned release), which is
    the same mechanism the gentle Play config already uses to disable events.
    """
    keep = set(keep)
    live = live_reward_names(cfg.rewards)
    missing = sorted(keep - live)
    if missing:
        raise AssertionError(
            f"{what} wants to keep {missing}, but the gentle config does not "
            f"have those reward terms live. It has {sorted(live)}. Upstream "
            "renamed a term, or the gentle config deleted one — either way "
            "this task would train on a reward table nobody declared."
        )
    for name in sorted(live - keep):
        setattr(cfg.rewards, name, None)

    # Strafe. Only the y range moves; x, yaw, both dead zones, the standing
    # fraction and heading-off are the gentle task's, untouched.
    cfg.commands.base_velocity.ranges.lin_vel_y = (-VY_MAX_MS, VY_MAX_MS)

    # Post-conditions. Both failures are silent otherwise: a reward table with
    # a term that should have died still trains, and a strafe range that did
    # not take still trains — as the gentle task, mis-labelled.
    got = sorted(live_reward_names(cfg.rewards))
    if got != sorted(keep):
        raise AssertionError(
            f"{what} ended up with reward terms {got}, not {sorted(keep)} — "
            "the surgery in apply_keep_list_and_strafe did not take."
        )
    if tuple(cfg.commands.base_velocity.ranges.lin_vel_y) != (-VY_MAX_MS, VY_MAX_MS):
        raise AssertionError(
            f"{what} has lin_vel_y = {cfg.commands.base_velocity.ranges.lin_vel_y}, "
            f"not {(-VY_MAX_MS, VY_MAX_MS)} — strafe is off and this is a "
            "relabelled gentle task."
        )


def apply_rung(cfg, rung: str) -> None:
    """Turn a constructed gentle config into one rung of the ladder, in place.

    The surgery itself is `apply_keep_list_and_strafe`; this function is the
    ladder's name for it. The split exists because
    `spyder_overnight_env_cfg.py` — the run that spends the ladder's answer —
    needs the identical surgery over a different keep list, and a second copy
    of it would be a second thing to keep correct.
    """
    apply_keep_list_and_strafe(cfg, rung_reward_terms(rung), f"ladder rung {rung!r}")


@configclass
class SpyderLadderBareEnvCfg(SpyderGentleEnvCfg):
    """Rung 1 — command-tracking income, no penalty. The ladder's control."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "bare")


@configclass
class SpyderLadderBareEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """Rung 1 for the viewer: few robots, nothing random."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "bare")


@configclass
class SpyderLadderActionRateEnvCfg(SpyderGentleEnvCfg):
    """Rung 2 — income + `action_rate_l2` at the gentle task's -0.01.

    The term prices the L2 norm of the per-step change in the action vector, so
    it taxes joint targets that move fast between policy steps. Episode 014's
    diagnosis of the forward policy's violence names flailing first; this is
    the term that prices it.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "actionrate")


@configclass
class SpyderLadderActionRateEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """Rung 2 for the viewer: few robots, nothing random."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "actionrate")


@configclass
class SpyderLadderTiltEnvCfg(SpyderGentleEnvCfg):
    """Rung 3 — income + `ang_vel_xy_l2` at the gentle task's -0.05.

    The term prices squared roll and pitch RATE of the torso, so it taxes a
    body that is tumbling rather than one that is merely tilted (that is
    `flat_orientation_l2`, which the gentle task carries at weight 0.0 and this
    ladder therefore never sees). Episode 014's forward policy bounded through
    the air; this is the term that prices the bounding.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "tilt")


@configclass
class SpyderLadderTiltEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """Rung 3 for the viewer: few robots, nothing random."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_rung(self, "tilt")


#: rung id -> (training cfg class, Play cfg class). Built here rather than
#: written down by each reader so `check_spyder.py` iterates THE LADDER, not a
#: hand-copied list of three names that a fourth rung would silently escape.
LADDER_CFGS: dict[str, tuple[type, type]] = {
    "bare": (SpyderLadderBareEnvCfg, SpyderLadderBareEnvCfg_PLAY),
    "actionrate": (SpyderLadderActionRateEnvCfg, SpyderLadderActionRateEnvCfg_PLAY),
    "tilt": (SpyderLadderTiltEnvCfg, SpyderLadderTiltEnvCfg_PLAY),
}

# Import-time, not test-time: adding a rung to RUNGS without a config class (or
# a class without a rung) fails the first import, which is every hydra launch
# and every oracle run, instead of producing an unchecked task id.
if set(LADDER_CFGS) != set(RUNGS):
    raise AssertionError(
        f"LADDER_CFGS covers {sorted(LADDER_CFGS)} but the ladder is {sorted(RUNGS)}. "
        "Every rung needs a config pair, and every config pair needs a rung — "
        "otherwise the oracle checks a set of rungs that is not the set that trains."
    )
