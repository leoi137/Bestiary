"""HOUND-16 on the Bestiary desert: the env config, and what it is and is not.

    Bestiary-Desert-Hound-v0        the training-shaped config
    Bestiary-Desert-Hound-Play-v0   few robots, no noise, no shoving

THE REWARD TABLE, AND WHAT AUTHORISES EVERY LINE OF IT
------------------------------------------------------
This began as ANYmal-C's table inherited whole, with terms in it that are wrong
for a machine whose feet are driven hub wheels. `HoundRewardsCfg` below is that
table re-scoped for this body, and **every change applies a decision already in
the record** -- `research/decisions/0004` Part B, whose starting point is
`fan-ziqi/robot_lab`'s Go2-W config: 12 leg joints plus 4 hub wheels whose joints
are literally named `*_foot_joint`, i.e. our exact topology.

**Nothing here is a new reward formulation.** Deriving one is reserved work with
its own gate; `research/decisions/0005` records why the last attempt was voided.

DELETED, not zero-weighted, because the term is meaningless on this body:

    feet_air_time         paid +0.125 per second of AIR TIME per foot. On a
                          legged robot that buys a gait. A driven hub wheel is in
                          continuous rolling contact, so a term that pays for
                          breaking contact PAYS THE MACHINE TO HOP -- the
                          opposite of what wheels are for. robot_lab disables it
                          and so do the other three wheeled-legged code bases
                          0004 surveyed. Deleted rather than left at weight 0
                          because a zero-weight term still reads as a term
                          somebody chose to keep.
    flat_orientation_l2   already zero upstream, and disabled by robot_lab too.
                          On a desert with 5.05 m of relief an L2 penalty on the
                          gravity projection charges terrain-following. The
                          wheeled sources that do keep an attitude term use a
                          multiplicative upright GATE instead, which is a
                          formulation choice rather than a re-scoping, so it is
                          NOT adopted here.

SPLIT, so the wheel joints are not charged for spinning:

    dof_torques_l2        legs only. Torque on a rolling wheel is what makes it
                          roll; charging it is charging the machine for driving,
                          which is learnings/011's failure written into the
                          reward. robot_lab scopes its torque and power penalties
                          to the legs for the same reason.
    dof_acc_l2            legs only, weight unchanged at -2.5e-7.
    dof_acc_wheel_l2      NEW term, wheels only, at robot_lab's -2.5e-9 --
                          exactly 100x weaker than the leg weight. The wheel
                          drive is derived to reach a commanded speed in one
                          control period (`hound_cfg.wheel_velocity_gain`), so it
                          accelerates the wheel hard BY DESIGN; charging that at
                          the leg rate charges the drive for working.
    dof_pos_limits        legs only, still weight 0. A joint with no limit has no
                          limit to be pushed past.

KEPT exactly as inherited, per 0004's instruction to keep the velocity-tracking
terms and per the brief that the rest is not this file's decision to make:
`track_lin_vel_xy_exp` +1.0, `track_ang_vel_z_exp` +0.5, `lin_vel_z_l2` -2.0,
`ang_vel_xy_l2` -0.05, `action_rate_l2` -0.01, and `undesired_contacts` -1.0
re-scoped from `.*THIGH` onto `.*_thigh`.

WHAT IS STILL OPEN, AND WILL NOT BE FIXED BY A LONGER RUN
---------------------------------------------------------
    * THE FOUR WHEEL ANGLES ARE STILL IN THE OBSERVATION. The inherited
      `joint_pos_rel` term covers all sixteen joints, and a hub wheel's angle
      integrates without bound -- at the drive's 10.665 rad/s it passes 200 rad
      inside a 20 s episode, handed to a network whose other inputs are order 1.
      0004 records that all four wheeled-legged code bases it surveyed delete or
      zero those entries. Fixing it moves the observation, which is a ONE-WAY
      DOOR in this repository (`envs/obs_spec.py`, learnings/003), so it is not
      done as a side effect of a reward re-scope.
    * `lin_vel_y` IS COMMANDED OVER +/-1 m/s (+/-3.3 ft/s) AND IS UNACHIEVABLE.
      Four wheels with fixed spin axes cannot hold sustained body-frame lateral
      velocity, and `track_lin_vel_xy_exp` is the largest term in the table.
      0005 B2 puts the unremovable ceiling at 0.4409 of that term's income.
    * THE PENALTY BUDGET IS LARGE AGAINST THE INHERITED +1.0/+0.5 TRACKING
      WEIGHTS. `check_hound.py`'s reward-budget check prints the table term by
      term; learnings/011 (control cost was 105.5% of the gap) and learnings/015
      (one fixed trot was income-optimal) are the two outcomes it exists to keep
      from repeating.

WHAT HAD TO BE RE-SCOPED, AND WHY EACH ONE WOULD HAVE FAILED
------------------------------------------------------------
Isaac Lab's shipped locomotion config names ANYmal's links, so most of it does
not merely underperform on Hound -- it raises at construction:

  `.*FOOT`, `.*THIGH`   ValueError("Not all regular expressions are matched!").
                        Hound's links are trunk, {FL,FR,RL,RR}_{abduct, thigh,
                        calf, wheel}. The wheel IS the foot here.
  `base`                same failure; the trunk is called `trunk`.
  `Robot/base`          a PRIM PATH, not a body name, and it fails differently:
                        the MJCF importer preserves the kinematic tree instead
                        of flattening it, so the trunk lives at
                        `Robot/Geometry/trunk`. See `hound_cfg.TRUNK_PRIM_SUBPATH`.
  `joint_names=[".*"]`  the quiet one. It would put all sixteen joints on the
                        position action term, wheels included, and a position
                        drive on an unlimited joint dies past +/-2*pi. Split into
                        twelve position and four velocity.

The contact sensor's `Robot/.*` needed no change HERE and still nearly sank the
whole config, which is worth recording because the error message points at this
file:

    ValueError: Not all regular expressions are matched!
        .*_wheel: []
    Available strings: ('trunk',)

The regex was right. The sensor had one body. `activate_contact_sensors` -- the
thing `ArticulationCfg.spawn.activate_contact_sensors = True` runs -- walks the
spawned prim for rigid bodies and stops descending at the first one it finds,
because in a URDF import every link is a sibling and a nested rigid body cannot
happen. In an MJCF import they are all nested under the trunk, so exactly one
body got the contact-report API. The fix is in the asset, in
`isaac/hound_usd.activate_contact_reporting`, and `check_hound.py` asserts all
seventeen carry it.
"""

from __future__ import annotations

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from bestiary.isaac.anymal_desert_env_cfg import desert_terrain_cfg
from bestiary.isaac.hound_cfg import (
    CONTROL_DT_S,
    HOUND16_CFG,
    LEG_JOINT_EXPR,
    TRUNK_BODY,
    TRUNK_PRIM_SUBPATH,
    WHEEL_JOINT_EXPR,
    wheel_action_scale,
)
from bestiary.robots.hound.build import SPEC
from bestiary.terrain.isaac_hf import DESERT_NATIVE_CELL_M

#: Terrain sampling for the Hound tasks, in metres per cell.
#:
#: 0.1 m, NOT the desert's native 0.078125 m. Native is what you want for
#: inspecting the asset, and it is what `check_desert_terrain.py` asserts
#: against; it is also what segfaults PhysX above ~64 envs at 200 tiles
#: (~4.2M faces), measured. Training uses the coarser grid, so the training
#: config declares it rather than inheriting a resolution nobody chose.
TRAIN_CELL_M = 0.1

#: Terrain sampling for the Play/viewer config. Native, because a viewer shows
#: few robots on few tiles and the point of looking is to see the real samples.
VIEW_CELL_M = DESERT_NATIVE_CELL_M


def retarget(field, key: str, entity: str, body_names: str | list[str]) -> None:
    """Point a manager term's `SceneEntityCfg` at Hound's body names.

    `field` is one entry of an events / rewards / terminations config. It is NOT
    always a term: Isaac Lab 3.0 wraps backend-dependent terms in a `PresetCfg`
    holding one alternative per physics backend, and `events.base_com` is one --
    an `EventTerm` under `default`/`physx` and `None` under `newton_mjwarp`.
    Reaching straight for `.params` on it fails with
    `AttributeError: _Preset object has no attribute 'params'`, which is a
    confusing thing to debug from a config file, so the unwrapping lives here and
    every call site looks the same.

    Raises if the field holds nothing to retarget, because a silent no-op here is
    a body-name regex left pointing at a robot that does not exist -- which is
    the whole failure this function is for.
    """
    if isinstance(field, PresetCfg):
        terms = [
            getattr(field, name)
            for name in field.__dataclass_fields__
            if getattr(field, name) is not None
        ]
    else:
        terms = [] if field is None else [field]
    if not terms:
        raise ValueError(
            f"nothing to retarget onto body_names={body_names!r}: the config field "
            f"is {field!r}. Upstream removed or disabled the term this expects, so "
            "the re-scoping is now silently absent."
        )
    for term in terms:
        term.params[key] = SceneEntityCfg(entity, body_names=body_names)


@configclass
class HoundActionsCfg:
    """Twelve position targets, then four wheel speeds. Sixteen actions.

    Declaration order IS the action vector's order -- `ActionManager` iterates
    `cfg.__dict__` -- so the twelve leg targets precede the four wheel speeds.

    **This is not the MuJoCo action vector.** `actuator_xml` emits per-leg blocks
    of four, (abduct, hip, knee, wheel) x FL, FR, RL, RR. Isaac Lab orders the
    articulation's joints by tree depth instead, measured on the loaded asset:

        FL_abduct FR_abduct RL_abduct RR_abduct
        FL_hip    FR_hip    RL_hip    RR_hip
        FL_knee   FR_knee   RL_knee   RR_knee
        FL_wheel  FR_wheel  RL_wheel  RR_wheel

    A checkpoint cannot cross between the two simulators anyway -- different
    solver, different observation -- but do not read an action index from one
    and use it in the other, and do not compare a per-index action log across
    them.
    """

    #: `use_default_offset=True` makes the target `default_joint_pos + scale *
    #: action`, and the default pose is the solved stance, so a ZERO ACTION IS
    #: THE STANDING STANCE. That is the same prior `envs/hound.py` gives the
    #: MuJoCo policy, and `SPEC.action_scale` is the same 0.5 rad half-width.
    #:
    #: BOTH fields carry an explicit annotation, and that is load-bearing rather
    #: than style. `ActionManager` builds the action vector by iterating
    #: `cfg.__dict__`, and `configclass` orders that by `__annotations__` --
    #: annotate one term and not the other and the unannotated one is appended
    #: last regardless of where it is written. Measured: with `joint_pos`
    #: unannotated the manager reported `{'joint_vel': 4, 'joint_pos': 12}`, i.e.
    #: the four wheel speeds came FIRST in a 16-vector that reads like legs-first.
    joint_pos: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_EXPR,
        scale=SPEC.action_scale,
        use_default_offset=True,
    )

    #: The wheels. `use_default_offset=False` because the offset would be the
    #: default joint VELOCITY, which is zero, and writing True would be a claim
    #: about a resting wheel speed that nothing here has.
    joint_vel: ActionTermCfg = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_EXPR,
        scale=wheel_action_scale(),
        use_default_offset=False,
    )


#: Weight of the wheel-scoped joint-acceleration penalty, and the factor by which
#: it is weaker than the leg one.
#:
#: -2.5e-9 and 100x are read from `research/decisions/0004` Part B, which read
#: them from `fan-ziqi/robot_lab`'s `unitree_go2w/rough_env_cfg.py`
#: (`joint_acc_wheel_l2`, wheels only) against its own `joint_acc_l2` at -2.5e-7
#: on the legs. 0004 also records that this is the one wheeled-legged coefficient
#: the sources DISAGREE on: `DreamWaQ_Go2W` charges the wheels the same -1e-7 it
#: charges the legs, i.e. 100x heavier than this. That disagreement is recorded,
#: not resolved, and 0004's "how we would know this was wrong" names a two-arm run
#: on it as the experiment that would settle it.
WHEEL_ACC_PENALTY_WEIGHT = -2.5e-9
WHEEL_ACC_PENALTY_RATIO = 100.0


@configclass
class HoundRewardsCfg(RewardsCfg):
    """ANYmal-C's inherited table, re-scoped for a body with driven hub wheels.

    Read the module docstring for what authorises each line. The short version:
    deletions and joint-scope splits taken from `research/decisions/0004` Part B,
    velocity tracking untouched, and no new mathematics.

    The class-level `= None` is Isaac Lab's own way of removing an inherited term:
    every manager skips a term whose cfg is None
    (`isaaclab/managers/reward_manager.py:224-227`), and `configclass` rebuilds the
    full annotation set on each subclass so a bare override in the body really does
    replace the parent's default (`isaaclab/utils/configclass.py:258-317`).
    """

    # -- Deleted: contact timing has no meaning on a rolling wheel -------------
    #: `feet_air_time` pays `sum(air_time - threshold) * first_contact` while a
    #: command is active. A driven hub wheel is supposed to STAY in contact, so
    #: the only way to earn this is to leave the ground: it is a bounty on
    #: hopping. Re-scoping its body regex from `.*FOOT` to `.*_wheel` fixes the
    #: ValueError and keeps the wrong incentive, which is worse than the crash.
    feet_air_time = None

    #: `flat_orientation_l2` was already weight 0 upstream and robot_lab disables
    #: it as well. Deleted rather than carried at 0 so the table says what it
    #: means. An attitude term for this machine, if one is ever wanted, is the
    #: multiplicative upright gate 0004 describes -- a formulation, not a scope.
    flat_orientation_l2 = None

    #: `undesired_contacts` counts BODIES in contact above 1 N at weight -1.0.
    #: `check_hound.py`'s budget check priced ONE thigh touching sand at 106.27%
    #: of all achievable tracking income -- the arithmetic of
    #: `research/learnings/011` (control cost 105.5% of the gap, driving stopped
    #: paying) reproduced on a different term. Deleting it takes the penalty
    #: budget from 134.42% to 28.15% of income, inside `decisions/0005`'s <=30%
    #: rule.
    #:
    #: Deleted rather than re-scoped onto the trunk, for two reasons: trunk
    #: contact ALREADY ends the episode (`terminations.base_contact`, retargeted
    #: below), so a trunk penalty is redundant with a termination; and on ground
    #: whose roughness reaches 1.38 m of elevation std, a thigh or calf brushing
    #: sand is normal traversal rather than damage. 0005 anticipated exactly this
    #: -- "calf contact is legitimate in sand -- start permissive, tighten by
    #: measurement".
    #:
    #: What this gives up: nothing protects the LEGS from grinding. That is a
    #: real gap and it is a reward-design question, not a scope one -- a
    #: force-shaped term rather than a body count. Being designed separately.
    undesired_contacts = None

    # -- Split: the wheel joints are not charged for spinning ------------------
    #: Torque on a rolling wheel is the thing that makes it roll. Charging it is
    #: charging the machine for driving, which is exactly the arithmetic
    #: `research/learnings/011` measured (control cost 105.5% of the gap). Weight
    #: unchanged from the inherited -1.0e-5; only the joint scope moves.
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)},
    )

    #: Leg accelerations, at the inherited weight, on the twelve leg joints only.
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)},
    )

    #: Wheel accelerations, 100x weaker. `hound_cfg.wheel_velocity_gain()` is
    #: derived so a commanded speed is reached in ONE control period, which means
    #: the drive accelerates the wheel by design: a step to the full 10.665 rad/s
    #: command is about 533 rad/s^2 with nothing wrong. At the leg weight four
    #: wheels doing that would cost more per step than the entire achievable
    #: tracking income, so the split is not a nicety.
    dof_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=WHEEL_ACC_PENALTY_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_EXPR)},
    )

    #: Still weight 0, as inherited -- turning it on is a reward decision this
    #: file does not make (0005 B6 prices one joint 0.1 rad past a soft limit at
    #: 30% of income). Scoped to the legs regardless, because the wheel's soft
    #: limit is 0.9 x FLT_MAX and "distance past a limit" is meaningless there.
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)},
    )


@configclass
class HoundDesertEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Hound on the desert-and-builtin mix, at the 0.1 m grid that trains."""

    actions: HoundActionsCfg = HoundActionsCfg()
    rewards: HoundRewardsCfg = HoundRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = HOUND16_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = desert_terrain_cfg(TRAIN_CELL_M)

        # The gain in hound_cfg is derived for one control period. If the env
        # runs at a different rate the gain is for a machine this is not, so the
        # two are tied together here rather than left to agree by memory.
        env_dt = self.decimation * self.sim.dt
        if abs(env_dt - CONTROL_DT_S) > 1e-12:
            raise ValueError(
                f"this env steps the policy every decimation*dt = {self.decimation}"
                f"*{self.sim.dt} = {env_dt} s, but hound_cfg.CONTROL_DT_S is "
                f"{CONTROL_DT_S} s and the wheel velocity gain "
                "(wheel_velocity_gain()) is derived from it. Change both or "
                "neither."
            )

        # A prim path, not a body name: the MJCF importer keeps the kinematic
        # tree, so the trunk is not a direct child of the robot prim.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{TRUNK_PRIM_SUBPATH}"

        # -- The observation. `joint_pos_rel` covers all sixteen joints upstream,
        # and a hub wheel's angle is an UNBOUNDED INTEGRATOR: at this drive's
        # 10.665 rad/s it passes 200 rad inside a 20 s episode, handed to a
        # network whose every other input is order 1. `envs/hound.py` drops the
        # same four entries from the MuJoCo observation for the same reason, and
        # `decisions/0004` records that all four wheeled-legged code bases it
        # surveyed delete or zero them.
        #
        # Joint VELOCITY stays on all sixteen: a wheel's speed is bounded, is the
        # drive's own state, and is what a policy needs to know it is rolling.
        #
        # This moves the observation width, which is a ONE-WAY DOOR here
        # (`envs/obs_spec.py`, `learnings/003`) -- the actor's first layer is
        # sized to it, so every checkpoint trained at the old width fails to
        # load rather than degrading. It is done NOW, before the first run,
        # precisely because no checkpoint exists yet to orphan. After a single
        # long run this same edit costs a full retrain.
        # The TERM is `joint_pos`; the FUNCTION it calls is `mdp.joint_pos_rel`
        # (`velocity_env_cfg.py:177`). Reaching for the function's name here
        # raises `AttributeError: 'PolicyCfg' object has no attribute
        # 'joint_pos_rel'`, which is how this was first written.
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_EXPR)
        }

        # -- The command. `lin_vel_y` is inherited at +/-1.0 m/s (+/-3.3 ft/s) and
        # this machine CANNOT PRODUCE IT. Verified from the generated MJCF: abduct
        # turns about (1,0,0) and hip, knee and wheel all about (0,1,0), so the
        # wheel axle is +Y in the leg frame and the only rotation that can tilt it
        # is the abduct roll about X, which maps it to (0, cos phi, sin phi). The
        # axle never acquires an x-component for ANY joint configuration, so the
        # rolling direction is always sagittal and no joint steers. Hound is a
        # skid-steer unicycle: its controllable command set is (v_x, omega_z).
        #
        # Left at +/-1.0 it charges the policy for failing at something
        # geometrically impossible, and `check_hound.py` measured the cost: 46.87%
        # of the largest reward term's income is unearnable at ANY competence,
        # which is `research/learnings/011`'s "charging the unremovable" one level
        # up -- on the term the whole table is built around.
        #
        # Set to zero rather than deleted, deliberately: the command slot stays in
        # the observation (width, spec hash and every checkpoint untouched -- NOT a
        # one-way door), and `track_lin_vel_xy_exp` keeps reading a 2-D error, so
        # with c_y == 0 the y channel becomes a calibrated price on uncontrolled
        # sideways SKID on a dune face. Reversible in config if a lateral stepping
        # gait is ever wanted.
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

        # -- Body names. Every one of these is `base` or an ANYmal link upstream.
        retarget(self.events.add_base_mass, "asset_cfg", "robot", TRUNK_BODY)
        retarget(self.events.base_external_force_torque, "asset_cfg", "robot", TRUNK_BODY)
        retarget(self.events.base_com, "asset_cfg", "robot", TRUNK_BODY)
        retarget(self.terminations.base_contact, "sensor_cfg", "contact_forces", TRUNK_BODY)

        # No contact-driven REWARD survives on this body. `feet_air_time` was
        # deleted for rewarding a wheel for leaving the ground; `undesired_contacts`
        # for costing 106% of income when one thigh touches sand. Both are
        # `= None` in `HoundRewardsCfg` with the reasoning. The contact SENSOR is
        # still live and still feeds `terminations.base_contact`, which is what
        # protects the trunk.


@configclass
class HoundDesertEnvCfg_PLAY(HoundDesertEnvCfg):
    """What the viewer and a human should look at: few robots, nothing random.

    Mirrors `AnymalCRoughEnvCfg_PLAY`'s overrides rather than inheriting them,
    because the Hound config descends from the generic locomotion cfg and not
    from the ANYmal one.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # Spawn across the grid instead of on the easiest terrain level.
        self.scene.terrain.max_init_terrain_level = None

        cfg = desert_terrain_cfg(VIEW_CELL_M)
        cfg.num_rows = 3
        cfg.num_cols = 3
        cfg.curriculum = False
        self.scene.terrain.terrain_generator = cfg

        # No observation noise and no random shoving: a machine being pushed
        # over by an invisible force is not a useful thing to watch.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
