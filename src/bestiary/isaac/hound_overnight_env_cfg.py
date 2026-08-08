"""HOUND-16 on v5 ground: the COMMANDED long run. Four reward terms, wide box.

    Bestiary-Overnight-Hound-v0        the long-run training config
    Bestiary-Overnight-Hound-Play-v0   few robots, no noise, no shoving

WHAT THIS TASK IS, IN ONE SENTENCE
----------------------------------
The Spyder's proven steering pipeline — dead-zoned rate commands, heading mode
off, the arc-corrected terrain curriculum, a reward reduced to income plus the
two terms that price the shape of a step — applied unchanged to the body whose
feet are driven hub wheels, on the v5 terrain `research/decisions/0007` makes
mandatory for every new arm.

Nothing in it is a new mechanism. Every piece is machinery that already exists
in this repository and already trained something: `commands.py`'s dead-zone
sampler, `curriculums.py`'s arc bar, `spyder_ladder_env_cfg.apply_keep_list`'s
reward surgery, `spyder_forward_v5_env_cfg.use_gentle_v5_mix`'s terrain swap.
What is new is the pairing: this is the first Hound task that can be DRIVEN.

WHY A COMMANDED HOUND, AND WHY NOW
-----------------------------------
`Bestiary-ForwardV5-Hound-v0` — the same body, the same ground, reward = `v_x`
alone — finished 1500 iterations at 4096 envs on 2026-08-08. Read from its own
final iteration block (`runs/hound_forward_v5_s1/box_console.log`):

    Mean reward                        203.56      (metres of forward travel)
    Mean episode length                606.28      of 1000 steps (12.1 s)
    Metrics/base_velocity/error_vel_xy  11.3859
    Curriculum/terrain_levels            5.7881
    Episode_Termination/base_contact     0.4097

Two facts in that block set up this task.

**The Hound's legs outrun its drive by an order of magnitude.** The wheel drive
saturates at `wheel_action_scale()` = 10.665 rad/s, which on an 85 mm (3.35 in)
wheel is 0.9066 m/s (2.03 mph) of rim speed — 18.13 m (59.5 ft) in a 20 s
episode. The forward probe returned 203.56 m per episode. An unshaped speed
objective on this body does not select rolling; it selects the gallop, and it
selects it hard. Nothing in that reward priced the mode, which was the probe's
whole point (`hound_forward_v5_env_cfg.py`).

**And it cannot be driven.** `error_vel_xy` of 11.39 is not a tracking failure,
it is the absence of tracking: `forward_velocity` never reads a command, so the
command channel in the observation was paid for and ignored. The operator's
goal is a steerable machine. A policy that runs at 16 m/s in whatever direction
it likes is not a step toward one.

So this task pays the command-tracking kernels, prices the vertical motion a
gallop is made of, and asks whether the same recipe that produced a driveable
Spyder produces a driveable Hound.

THE FOUR TERMS, AND WHERE EACH COMES FROM
------------------------------------------
    track_lin_vel_xy_exp   +1.0     income, std 0.5   (the Hound's own table)
    track_ang_vel_z_exp    +0.5     income, std 0.5   (the Hound's own table)
    action_rate_l2         -0.01    the ladder's measured winner
    lin_vel_z_l2           -2.0     vertical trunk speed, squared

Every weight and every param is `HoundDesertEnvCfg`'s, unchanged, and this file
never types one: the surgery keeps the term OBJECTS the desert config built
(`spyder_ladder_env_cfg.apply_keep_list`), so "at the Hound table's weights" is
true by construction rather than by two lists that agree today.

The first three are `spyder_overnight_env_cfg.py`'s recipe: the ladder's income
terms plus `action_rate_l2`, the rung that scored highest on both reward and
episode length at 1500 iterations. That measurement is a SINGLE-SEED PROBE on a
DIFFERENT ROBOT, and both qualifiers travel with it — it is the best evidence in
hand for which single penalty buys a survivable gait, and it is not a
demonstrated effect, still less a demonstrated effect on this body.

`lin_vel_z_l2` at -2.0 is the largest-magnitude penalty in the inherited table
and it is the one term that prices the failure the forward probe measured. A
body that leaves the ground has vertical speed on the way up and again on the
way down; a rolling machine has neither. On the Spyder this term was one of two
"shape of a step" additions. On the Hound it is doing a second job: it is the
only surviving line that expresses a preference between rolling and galloping,
now that `dof_acc_wheel_l2`'s 100x thumb on the scale is gone (below).

THE ONE DELIBERATE DIFFERENCE FROM THE SPYDER OVERNIGHT: NO `feet_air_time`
---------------------------------------------------------------------------
The Spyder's overnight table is these four terms plus `feet_air_time` at
+0.125. This task has four, and the missing one is not an oversight.

`feet_air_time` pays `sum(air_time - threshold) * first_contact` while a command
is active. It reads contact TIMING, and a driven hub wheel is supposed to stay
in rolling contact — so the only way to earn it is to break contact. Re-scoping
its body regex from `.*FOOT` onto `.*_wheel` fixes the `ValueError` and keeps a
bounty on hopping, which is worse than the crash. `HoundRewardsCfg` deleted it
for that reason (`research/decisions/0004` Part B, and `fan-ziqi/robot_lab`'s
Go2-W config, our exact topology), `check_hound.py` has a structural assertion
against any contact-timing term reaching a wheel body, and this task inherits
both.

There is a second reason, specific to this run: the term's own arithmetic is
undefined here. On the Spyder, `check_spyder.check_the_money` prices it at a
50%-duty gait of f Hz per foot as `w*4*f*(0.5/f - 0.5)` per second — zero at
1 Hz, negative above it. A rolling wheel has no duty cycle and no cadence, so
there is no f to evaluate it at. A term whose value cannot be computed for the
intended behaviour is not a term this run can carry.

**What would justify revisiting it.** Not a re-scoping — the incentive is wrong
under every scope on this body. It would take a NEW term with the opposite
sign: something that pays for KEEPING contact (a rolling-duty bonus) rather
than for breaking it. That is new reward mathematics, which is reserved work
with its own gate (`research/decisions/0005` records why the last attempt was
voided), and it becomes worth opening only if this run's policy converges on a
bounding gait — high `Curriculum/terrain_levels` and low `error_vel_xy` reached
while airborne. If instead the machine rolls, the question never arises.

WHAT ELSE IS DELETED, AND THE ONE DELETION THAT NEEDS AN ARGUMENT
------------------------------------------------------------------
The Hound desert table pays nine terms. Four are kept; these five are deleted:

    ang_vel_xy_l2       -0.05      the ladder's LOSING rung: it scored below
                                   the untaxed control on both reward and
                                   episode length, so it is the one term the
                                   ladder gives a reason not to carry.
    dof_torques_l2      -1e-05     leg-scoped joint-effort taxes. Neither was
    dof_acc_l2          -2.5e-07   on the ladder, and they go for the ladder's
                                   reason: an unmeasured term is not carried
                                   into a long run because upstream shipped it.
    dof_pos_limits       0.0       weight EXACTLY zero in the Hound table — it
                                   contributes nothing, and a zero-weight term
                                   in a config is a term someone switches on by
                                   accident one day.
    dof_acc_wheel_l2    -2.5e-09   the argued one; see below.

`dof_acc_wheel_l2` is the wheel half of `research/decisions/0004` Part B's
leg/wheel split, and deleting it deserves more than a line.

The split exists because the wheel drive is DERIVED to reach a commanded speed
in one control period (`hound_cfg.wheel_velocity_gain`), so it accelerates the
wheel hard by design — a step to full command is 533 rad/s^2 with nothing
wrong. Charging that at the leg weight would bill the machine for driving,
which is `research/learnings/011`'s failure written into the reward. 0004's
repair was to charge the wheels 100x less than the legs.

This table deletes BOTH sides. That is the same direction the split points, one
step further: with no joint-acceleration term at all, wheel acceleration is
free, which is what the 100x was reaching for. The magnitude is settled by
`check_hound.check_reward_budget_against_011_and_015`, which prices the wheel
term at the drive's own design acceleration and asserts it stays under 1% of
achievable income — so removing it moves the economics by well under a percent.

What is given up, stated plainly: **nothing prices leg-joint acceleration or
torque any more.** A violent leg motion is now taxed only indirectly, through
`action_rate_l2` (the joint TARGETS moving fast between policy steps) and
`lin_vel_z_l2` (the trunk leaving the ground). That is precisely the bet the
ladder measured on the Spyder — the untaxed control lost to the action-rate arm
— and it is a bet, not a result, on this body.

THE COMMAND ENVELOPE: THE FAST BOX FROM THE START
--------------------------------------------------
    lin_vel_x   ±1.5 m/s    (±3.4 mph)
    lin_vel_y   ±0.6 m/s    (±1.3 mph)
    ang_vel_z   ±1.5 rad/s
    min_lin_vel_x 0.25 m/s, min_ang_vel_z 0.2 rad/s, rel_standing_envs 0.1,
    heading mode OFF

These are `spyder_fast_env_cfg.py`'s three ranges and `spyder_gentle_env_cfg.py`'s
three dead-zone/standing parameters, and they are adopted at the START here
rather than reached by a fine-tune, because there is no Hound checkpoint to
fine-tune. The Spyder's ±0.6 forward box was sized against a machine measured at
0.37 m/s (`research/learnings/001`); the Hound's drive alone commands 0.9066 m/s
and its legs demonstrably produce an order of magnitude more, so the narrow box
would be the wrong instrument on this body.

Every one of the six numbers is a claim, and three of them carry arithmetic that
belongs here rather than in a later post-mortem.

**1. The yaw ceiling and the tightest commandable turn.** `spyder_fast_env_cfg`
widens yaw with v_x because the tightest constant-twist radius a command can ask
for is v/w: at v_x = 1.5 m/s, ±1.5 rad/s puts that radius at 1.0 m (3.3 ft).
Leaving yaw at the Hound's inherited ±1.0 would loosen it to 1.5 m and make fast
driving a near-straight-line-only mode.

**2. The lateral channel has no dead zone, and this task widens it anyway.**
`DeadZoneVelocityCommand._resample_command` remaps index 0 (v_x magnitude) and
index 2 (w_z snap) and leaves index 1 exactly as `UniformVelocityCommand`
sampled it, so v_y ~ U(-0.6, 0.6) including the ambiguous near-zero band the
other two channels exclude. That does not re-open the parked-seed door — the
x channel still guarantees |v_x| >= 0.25 m/s on every driving env, so a
motionless machine still eats at least that error — it makes STRAFE OPTIONAL,
and the price of ignoring it is computable: a machine that tracks v_x perfectly
and never sidesteps still collects

    E[exp(-(v_y/std)^2)],  |v_y| ~ U(0, 0.6),  std = 0.5   =   67.23%

of the linear kernel (`check_hound.py`'s overnight check prints it). For scale,
the same number for the already-launched `Bestiary-Fast-Spyder-v0` — ±0.6 at
std 0.3 — is 44.10%. **This envelope charges the Hound less for not strafing
than the Spyder fine-tune charges the Spyder**, which is the honest way to say
that ±0.6 is not a harsher ask here than one this repository has already made.

The desert task's own lateral note applies and is not repeated: the Hound CAN
make body-frame lateral velocity (a common abduct roll gives all four wheels the
same axle, and the no-slip constraint then has a nullspace containing near-pure
lateral translation; measured |v_y|/|v| = 0.332 at phi = 0.8 rad on a 20-degree
slope). ±0.3 was that task's compromise under an unmeasured ceiling. The
worst-case reading — v_y wholly unachievable, c_y ~ U(-0.6, 0.6), std 0.5 —
puts the linear kernel's ceiling at 0.6723 against 0.8919 at ±0.3, so at most
22.0% more of that term's income becomes unearnable. It is a LOWER bound on the
ceiling, because the machine demonstrably makes some v_y.

    Revisit trigger: measure this policy's achieved |v_y| span. If no cell of a
    per-cell grid eval shows |v_y| above ~0.2 m/s, the channel is charging the
    unremovable (`research/learnings/011`) and it narrows — by measurement,
    which is what the desert task's ±0.3 was waiting for.

**3. The kernel widths are NOT rescaled with the range, and the reason is the
opposite of `spyder_fast_env_cfg`'s.** Both tracking terms keep std = 0.5, the
Hound table's inherited value (upstream's `sqrt(0.25)`), so

    linear    std/v_max  = 0.5 / 1.5   = 0.3333
    angular   std/w_max  = 0.5 / 1.5   = 0.3333
    2-D box corner  sqrt(1.5^2 + 0.6^2) = 1.6155 m/s,  std/corner = 0.3095

against upstream's discrimination ratio of 0.5. `spyder_gentle_env_cfg.py`
derives its widths by PRESERVING that ratio, which here would give
std = 0.75 on both channels. That is the change this file does not make, and
the arithmetic says why:

    standing's expected share of drive-cell tracking income
        std = 0.50   ->   21.40%
        std = 0.75   ->   37.23%

`research/decisions/0005`'s flag is 30%, `check_spyder`'s `STANDING_SHARE_FLAG`
is 30%, and 62.7% is where the Hound's arm-1 seed 2 sat when it parked and still
beat the do-nothing control in 13 of 13 eval cells
(`research/measurements/isaac_hound_arm1_s2.json`). Restoring the 0.5 ratio on a
±1.5 box would put this run over that flag before it started. So the ratio moves
in the direction `spyder_fast_env_cfg.py` documents — tighter, discriminating
MORE, not less — and the cost is stated there and holds here: the kernel is a
function of the ABSOLUTE error, so nothing a given error costs has changed
(a 0.6 m/s shortfall is still worth exp(-(0.6/0.5)^2) = 0.2369 of the linear
term), but the errors get larger when the command does, and a parked machine at
the top command collects exp(-9) = 1.2e-4. Tracking income at the top of this
box is close to all-or-nothing. **If the run stalls — reward collapses and stays
collapsed because the top of the box pays nothing — the kernel width is the
first thing to reconsider, and it is reconsidered as its own one-variable change
with its own launch, never folded into a rerun of this one.**

THE TERRAIN CURRICULUM: THE ARC BAR, AND WHERE ITS TWO BARS NOW SIT
--------------------------------------------------------------------
`curriculums.terrain_levels_vel_arc` replaces upstream's `terrain_levels_vel`.
The Hound tasks never adopted it — `hound_forward_v5_env_cfg.py` says so
explicitly and leaves it, correctly, because a command-deaf reward cannot be
demoted for failing to track. A COMMANDED machine can: upstream's demote bar is
`||cmd_xy|| * T / 2`, straight-line kinematics, so a perfect tracker of a
turning command drives a circle of radius v/w and is demoted every episode while
a yaw-blind straight driver is promoted. That is `research/learnings/015`'s
failure taught on purpose. The function is robot-agnostic — it reads the command
and the displacement and nothing about the machine — so it ports as an
assignment.

Where the bars sit on this envelope, at the desert task's 8 m tiles and 20 s
episodes:

  * PROMOTE: a fixed `tile/2` = 4 m of displacement from spawn. At the top
    straight command that is 4/1.5 = 2.67 s of driving out of 20.
  * DEMOTE: half the command's own reachable displacement. At the top straight
    command, 0.5 * 1.5 * 20 = 15 m (49.2 ft).

And the sanity check that matters on THIS body: a machine that merely rolls at
the drive's saturation speed covers 0.9066 * 20 = **18.13 m**, which clears the
15 m bar. So the terrain curriculum does not require the gallop. A rolling Hound
that tracks the top command as well as its drive allows is promoted, not demoted
— which is the property that makes it legitimate to ask this run for rolling.

A turning env promotes only if its arc DIAMETER 2v/w exceeds 4 m, i.e. only if
w < v/2 = 0.75 rad/s at the top forward command; the yaw snap makes
0.2/1.5 = 13.3% of driving envs straight drivers, and they carry promotion. The
net promotion rate is a training outcome to read off `Curriculum/terrain_levels`,
not a claim this docstring is entitled to make.

WHAT IS INHERITED UNTOUCHED, AND WHY EACH
------------------------------------------
Everything else, from `HoundDesertEnvCfg`:

  * THE OBSERVATION, 243 wide (3 + 3 + 3 + 3 + 12 + 16 + 16 + 187), with the
    four unbounded wheel ANGLES already dropped from `joint_pos` and all sixteen
    joint VELOCITIES kept. It is a ONE-WAY DOOR (`envs/obs_spec.py`,
    `research/learnings/003`) — the actor's first layer is sized to it — and
    nothing here moves it. That is the property that makes a multi-hour run
    startable at all.
  * THE ACTIONS: twelve leg position targets then four wheel speeds, in that
    order, with the wheels on a velocity drive at stiffness 0. A position drive
    on an unlimited joint dies past ±2*pi.
  * THE TERMINATIONS: time-out plus trunk contact. Episode-reset machinery, not
    a reward — nothing pays or charges for it.
  * The robot, the height scanner (upstream's 1.6 x 1.0 m at 0.1 m = 187 rays),
    the contact sensor, the events and the reset scatter, the env count and
    spacing, and the physics step.

THE GROUND, AND WHY IT IS THE FORWARD PROBE'S GROUND EXACTLY
-------------------------------------------------------------
`spyder_forward_v5_env_cfg.use_gentle_v5_mix`, the same call
`hound_forward_v5_env_cfg.py` makes: the desert tile leaves the mix and the v5
gentle tile takes its place, at whatever sampling and grid shape the config
being replaced declared, so the Play twin keeps its 3x3 native-sampled grid with
the curriculum off. `research/decisions/0007` makes v5 mandatory for every new
arm and leaves v4 committed and untouched.

`check_hound.py`'s overnight check asserts the ground is **byte-identical to
`Bestiary-ForwardV5-Hound-v0`'s** — an empty `scene` diff against that config —
which is the strongest available statement that this run and the forward probe
stood on the same world. Two Hound runs on the same body and the same ground,
differing in the reward and the command distribution, is a comparison worth
being able to make later.

THIS IS A PRODUCTION RUN, NOT AN EXPERIMENT — SAID PLAINLY
-----------------------------------------------------------
Four things move at once against the forward probe: the reward table, the
command sampler, the command envelope and the terrain curriculum. If the
resulting policy is worse than the probe's on some axis, **this run cannot say
which of the four did it**, and nothing written from it may claim otherwise.
That is a deliberate trade — the Spyder ran the one-variable ladder and this
spends its answer on a second body — and it means the only claims this run can
support are about ITS OWN policy: what it does, measured, one seed, one arm.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.commands import DeadZoneVelocityCommandCfg
from bestiary.isaac.curriculums import terrain_levels_vel_arc
from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg, HoundDesertEnvCfg_PLAY
from bestiary.isaac.spyder_forward_v5_env_cfg import use_gentle_v5_mix
from bestiary.isaac.spyder_ladder_env_cfg import INCOME_TERMS, apply_keep_list, live_reward_names

#: The commanded envelope. Three numbers, and they are `spyder_fast_env_cfg`'s
#: three exactly — restated rather than imported, deliberately, and this is the
#: one place in this file where a copy is the right answer. Importing them would
#: tie the Hound's command box to a SPYDER fine-tune's, so a later edit made for
#: that machine's reasons would silently re-scope a Hound run. The two tasks
#: agree on these numbers today for the reasons the docstring gives; they are not
#: the same decision.
VX_MAX_MS = 1.5
VY_MAX_MS = 0.6
WZ_MAX_RADS = 1.5

#: The dead zones and the standing fraction, `spyder_gentle_env_cfg`'s, restated
#: for the same reason. `min_lin_vel_x` is a magnitude RESAMPLE floor — every
#: driving env is commanded |v_x| in [0.25, 1.5]. `min_ang_vel_z` is a SNAP
#: threshold — |w_z| below it becomes exactly 0, which is what makes straight
#: drivers exist for the terrain curriculum to promote on, and what makes
#: "A/D released" a commandable state. `REL_STANDING` zeroes the full command
#: for that fraction of resamples, so standing is an explicit mode rather than
#: the small tail of a uniform.
VX_MIN_MS = 0.25
WZ_MIN_RADS = 0.2
REL_STANDING = 0.1

#: The tracking kernel width both terms must still carry, m/s and rad/s.
#:
#: 0.5 is upstream's `sqrt(0.25)`, inherited by `HoundDesertEnvCfg` and NOT
#: rescaled here. Pinned as a constant so the post-condition below can assert it:
#: the docstring's whole argument is that preserving upstream's std/range ratio
#: of 0.5 (which would give 0.75 against this ±1.5 box) takes standing's expected
#: share of drive-cell income from 21.40% to 37.23%, past decision 0005's 30%
#: flag and toward the 62.7% at which a Hound seed parked. A well-meaning
#: "restore the ratio" edit must fail loudly rather than train.
KERNEL_STD = 0.5

#: The complete reward table. The ladder's two income kernels plus the ladder's
#: measured winner plus the one inherited term that prices a body leaving the
#: ground. FOUR terms, one fewer than the Spyder overnight's five: `feet_air_time`
#: is absent by decision, and the docstring's own section argues it.
OVERNIGHT_TERMS: tuple[str, ...] = INCOME_TERMS + ("action_rate_l2", "lin_vel_z_l2")

#: The term whose ABSENCE is this task's one deliberate departure from
#: `spyder_overnight_env_cfg.OVERNIGHT_TERMS`. Named rather than left implicit so
#: the import-time assertion below can state it as a fact about this table, and
#: so `check_hound.py` can assert it without re-deriving which term is meant.
FORBIDDEN_CADENCE_TERM = "feet_air_time"

#: What the keep list removes from the Hound desert table, with the desert weight
#: of each. NOT used by the surgery — the surgery deletes whatever is live and
#: not kept, so a term Isaac Lab adds tomorrow is deleted tomorrow without a code
#: change. This is the DOCUMENTED EXPECTATION, and `check_hound.py` asserts the
#: live desert table minus `OVERNIGHT_TERMS` equals exactly these five names. The
#: split is deliberate: the config is safe by construction, and the oracle is what
#: goes red so a human finds out an upstream release moved the table.
EXPECTED_DELETED_TERMS: tuple[str, ...] = (
    "ang_vel_xy_l2",
    "dof_acc_l2",
    "dof_acc_wheel_l2",
    "dof_pos_limits",
    "dof_torques_l2",
)

#: field name on `commands.base_velocity.ranges` -> the commanded range.
#: A dict rather than three assignments, so the surgery, its post-condition and
#: `check_hound.py`'s overnight check all iterate ONE declaration of "which
#: ranges this task commands".
OVERNIGHT_RANGES: dict[str, tuple[float, float]] = {
    "lin_vel_x": (-VX_MAX_MS, VX_MAX_MS),
    "lin_vel_y": (-VY_MAX_MS, VY_MAX_MS),
    "ang_vel_z": (-WZ_MAX_RADS, WZ_MAX_RADS),
}


def _dead_zone_velocity_command(upstream) -> DeadZoneVelocityCommandCfg:
    """The commanded task's sampler, built from the term it replaces.

    `resampling_time_range` and `debug_vis` are read off the config being
    replaced rather than typed, so this task inherits upstream's resample cadence
    (10 s against a 20 s episode) instead of quietly declaring its own. Everything
    else is this module's declaration.

    `heading_command=False` is not a preference. Upstream defaults it True, which
    recomputes w_z from heading error every step — so a machine that turns to the
    target and stops has zeroed its own yaw command and collects the yaw income
    for standing, the point-and-park loop `research/decisions/0006` prices on
    this exact robot. `DeadZoneVelocityCommand.__init__` refuses heading mode
    outright; this is the config that never asks for it.
    """
    return DeadZoneVelocityCommandCfg(
        asset_name=upstream.asset_name,
        resampling_time_range=upstream.resampling_time_range,
        rel_standing_envs=REL_STANDING,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=upstream.debug_vis,
        ranges=DeadZoneVelocityCommandCfg.Ranges(
            lin_vel_x=OVERNIGHT_RANGES["lin_vel_x"],
            lin_vel_y=OVERNIGHT_RANGES["lin_vel_y"],
            ang_vel_z=OVERNIGHT_RANGES["ang_vel_z"],
            heading=None,
        ),
        min_lin_vel_x=VX_MIN_MS,
        min_ang_vel_z=WZ_MIN_RADS,
    )


def apply_overnight(cfg) -> None:
    """Turn a constructed Hound desert config into the commanded long run, in place.

    Four edits and four post-conditions, in the order the docstring argues them:
    the reward keep list, the command sampler, the terrain curriculum, the
    ground. Called after `super().__post_init__()` by BOTH the training class and
    the Play twin, for the reason `spyder_ladder_env_cfg.apply_keep_list_and_strafe`
    gives — a Play config descends from the desert Play class, so it cannot also
    inherit this variant's `__post_init__`, and a surgery that reached one class
    and not the other would let the viewer drive a policy under a different
    reward, a different command distribution or a different world than the one
    that trained.

    Every post-condition here guards a SILENT failure. A reward term that
    survived the keep list still trains, as a task nobody declared. A command
    term that did not take still trains, as the desert task's heading-driven
    uniform sampler under this task's name — the parked-seed distribution the
    whole dead-zone stack exists to close. A kernel width that drifted still
    trains, over the standing-share flag. A curriculum left upstream still
    trains, demoting every perfect turner.
    """
    apply_keep_list(cfg, OVERNIGHT_TERMS, "the overnight Hound task")

    cfg.commands.base_velocity = _dead_zone_velocity_command(cfg.commands.base_velocity)

    # Upstream's `terrain_levels_vel` compares displacement against
    # `||cmd_xy|| * T / 2` — straight-line kinematics applied to an arc — so a
    # perfect tracker of a turning command is demoted every episode while a
    # yaw-blind straight driver is promoted. `curriculums.py` carries the
    # derivation; for w = 0 the arc bar reduces exactly to upstream's.
    cfg.curriculum.terrain_levels.func = terrain_levels_vel_arc

    # The ground: the desert tile leaves the mix, the v5 gentle tile takes its
    # place, at this config's own sampling and grid shape.
    use_gentle_v5_mix(cfg)

    # -- Post-conditions ------------------------------------------------------
    got_terms = sorted(live_reward_names(cfg.rewards))
    if got_terms != sorted(OVERNIGHT_TERMS):
        raise AssertionError(
            f"the overnight Hound task ended up paying {got_terms}, not "
            f"{sorted(OVERNIGHT_TERMS)} — the keep-list surgery did not take."
        )

    cmd = cfg.commands.base_velocity
    if not isinstance(cmd, DeadZoneVelocityCommandCfg):
        raise AssertionError(
            f"the overnight Hound task's command term is {type(cmd).__name__}, not "
            "DeadZoneVelocityCommandCfg. The plain sampler draws v_x uniformly "
            "over a symmetric range, which puts a large share of commands where "
            "standing is nearly the right answer — the distribution that produced "
            "the parked arm-1 seed."
        )
    got_ranges = {name: tuple(getattr(cmd.ranges, name)) for name in OVERNIGHT_RANGES}
    if got_ranges != OVERNIGHT_RANGES:
        raise AssertionError(
            f"the overnight Hound task commands {got_ranges}, not "
            f"{OVERNIGHT_RANGES} — the envelope did not take."
        )

    for term_name in INCOME_TERMS:
        term = getattr(cfg.rewards, term_name, None)
        if term is None:
            raise AssertionError(
                f"the overnight Hound task has no live {term_name!r} to check the "
                "kernel width of — without both tracking kernels the policy cannot "
                "be driven, which is the one property this task exists to add."
            )
        got_std = float(term.params["std"])
        if got_std != KERNEL_STD:
            raise AssertionError(
                f"the overnight Hound task's {term_name} std is {got_std}, not the "
                f"inherited {KERNEL_STD}. Rescaling the kernel with the widened "
                "range (std 0.75 preserves upstream's 0.5 ratio) takes standing's "
                "expected share of drive-cell income from 21.40% to 37.23%, past "
                "decision 0005's 30% flag — see this module's docstring before "
                "changing it."
            )

    if cfg.curriculum.terrain_levels.func is not terrain_levels_vel_arc:
        raise AssertionError(
            "the overnight Hound task's terrain curriculum is "
            f"{getattr(cfg.curriculum.terrain_levels.func, '__name__', '?')!r}, not "
            "terrain_levels_vel_arc. Upstream's bar demotes a perfect turner every "
            "episode and promotes a yaw-blind straight driver, which on a task "
            "whose whole point is obedience is learnings/015 taught on purpose."
        )


def _assert_the_declaration_is_coherent() -> None:
    """The module's own constants say what its docstring claims.

    Called at import — which is every hydra launch and every oracle run — so a
    declaration that violates the docstring fails before the app boots rather
    than after hours of training under a table nobody wrote down. A function
    rather than bare module-level statements so the loop variables do not survive
    as module attributes.
    """
    if set(INCOME_TERMS) - set(OVERNIGHT_TERMS):
        raise AssertionError(
            f"the overnight Hound table {sorted(OVERNIGHT_TERMS)} drops "
            f"{sorted(set(INCOME_TERMS) - set(OVERNIGHT_TERMS))} — without both "
            "command-tracking kernels the policy cannot be driven, which is the "
            "one property this task adds to the Hound."
        )
    if len(set(OVERNIGHT_TERMS)) != len(OVERNIGHT_TERMS):
        raise AssertionError(
            f"the overnight Hound table repeats a term: {OVERNIGHT_TERMS}. A "
            "duplicate here is silent — the keep list is a set — so the table is "
            "one term shorter than it reads."
        )
    if FORBIDDEN_CADENCE_TERM in OVERNIGHT_TERMS:
        raise AssertionError(
            f"{FORBIDDEN_CADENCE_TERM!r} is in the overnight Hound table. It pays "
            "for the DURATION of flight, and a driven hub wheel earns that only by "
            "breaking the rolling contact it exists to keep: the term is a bounty "
            "on hopping on this body (research/decisions/0004 Part B). Its absence "
            "is this task's one deliberate difference from the Spyder overnight; "
            "adding it back needs a new term with the opposite sign, not this one."
        )
    if set(OVERNIGHT_TERMS) & set(EXPECTED_DELETED_TERMS):
        raise AssertionError(
            "the overnight Hound task both keeps and expects to delete "
            f"{sorted(set(OVERNIGHT_TERMS) & set(EXPECTED_DELETED_TERMS))}. One of "
            "the two declarations is wrong, and the keep list is the one that runs."
        )

    # `DeadZoneVelocityCommand.__init__`'s own validation, hoisted to import
    # time: there it raises when the command TERM is constructed, which is after
    # the app is up and minutes into a launch.
    for name, dz in (("lin_vel_x", VX_MIN_MS), ("ang_vel_z", WZ_MIN_RADS)):
        lo, hi = OVERNIGHT_RANGES[name]
        if lo != -hi:
            raise AssertionError(
                f"the overnight Hound {name} range {(lo, hi)} is not symmetric; "
                "DeadZoneVelocityCommand treats both channels as a magnitude and a "
                "sign, so an asymmetric range biases the sign silently."
            )
        if not 0.0 <= dz < hi:
            raise AssertionError(
                f"min_{name} = {dz} is outside [0, {hi}) — the v_x resample would "
                "collapse onto the range edge and the w_z snap would zero every draw."
            )
    if not 0.0 <= REL_STANDING < 1.0:
        raise AssertionError(
            f"rel_standing_envs = {REL_STANDING} is not a fraction; at 1.0 every "
            "env is commanded to stand and the task has no driving in it at all."
        )


_assert_the_declaration_is_coherent()


@configclass
class HoundOvernightEnvCfg(HoundDesertEnvCfg):
    """The commanded long run: income + `action_rate_l2` + `lin_vel_z_l2`, on v5."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_overnight(self)


@configclass
class HoundOvernightEnvCfg_PLAY(HoundDesertEnvCfg_PLAY):
    """The commanded long run for the viewer: few robots, nothing random.

    Descends from `HoundDesertEnvCfg_PLAY`, NOT from `HoundOvernightEnvCfg`, so
    the Play overrides (16 envs, native terrain sampling, a 3x3 grid with the
    curriculum off, corruption and pushes off) are inherited rather than copied.
    The cost is that a change to the training class does not reach here, which is
    why the variant lives in `apply_overnight` and both classes call it — and why
    `check_hound.py` checks BOTH configs.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_overnight(self)
