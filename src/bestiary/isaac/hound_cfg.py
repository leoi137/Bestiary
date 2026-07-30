"""HOUND-16 as an Isaac Lab articulation: the PD variant, two actuator groups.

Every number here is imported from `robots/hound/build.py`'s `Spec` or derived
from it in code below. Nothing is retyped -- `Spec` is the single source of
truth for this machine, and a second copy of `kp_knee` is a second machine
waiting to disagree with the first.

Load the asset with `bestiary.isaac.hound_usd` before using this; the USD it
points at is generated, not hand-written.


THE ONE THING THAT IS NOT OPTIONAL: TWO ACTUATOR GROUPS
------------------------------------------------------
Twelve leg joints on POSITION control, four wheels on VELOCITY control with
`stiffness = 0`.

The wheel is a MuJoCo hinge with `limited="false"`: its angle integrates without
bound. Isaac Lab's importer brings it in as a revolute joint with no limit
attributes, which is correct -- but a *position* drive on such a joint is not.
PhysX refuses a drive target outside +/-2*pi, so a position-driven wheel works
for about 3.2 revolutions and then stops working, silently, mid-episode, after
the policy has already learned to rely on it. `research/decisions/0001` predicted
this before the port existed and the sibling robot's card records it as
measured; `isaac/check_hound.py` asserts `stiffness == 0` on all four wheels so
it cannot come back.

A velocity drive is the right shape for the hardware too. The MuJoCo model
drives the wheels with `<motor gear="3.0">`, i.e. torque, and the traction
budget in `robots/hound/CARD.md` sizes that 3.0 N*m against what the ground will
accept rather than what a motor could deliver. A velocity drive with a finite
gain and that same effort ceiling is a torque source at every velocity error
past saturation, which is where a driving wheel lives.


WHAT IS DELIBERATELY NOT PORTED
-------------------------------
THE STANCE SPRINGS. `Spec.stiffness()` returns 12.12 / 18.58 / 32.77 N*m/rad
with `Spec.springref()` preloads, and the CARD labels them a crutch in their own
section: real Go2 joints have no springs, and these exist so the MuJoCo machine
is self-supporting at reset. Isaac Lab replaces the crutch with two things it
does better -- an implicit PD drive that holds the stance from the first step,
and reset ranges that scatter the joints around it -- so porting the springs
would double up the restoring torque and make the leg stiffer than either model.

They also do not arrive by accident: the importer files MuJoCo joint attributes
into a separate `mujoco` USD variant and the asset selects `physx`, so the
twelve drive stiffnesses below are the `<position>` actuators' kp and the
springs are in no layer the solver reads. `check_hound.py` asserts that.

THE WHEEL'S JOINT DAMPING AS A DRIVE GAIN. `Spec.wheel_damping = 0.05` is
passive coast damping -- "a wheel is supposed to coast" -- and it is not a
velocity-tracking gain. Using it as one would need a 60 rad/s error to reach
peak torque. The gain is derived instead, below.


VELOCITY LIMITS: SOURCED ON THE LEGS, STILL OPEN ON THE WHEELS
--------------------------------------------------------------
`Spec` has no joint speed limit of its own, because Go2's rated joint velocities
were never read into it. The twelve leg joints therefore arrived with whatever
the converted USD happened to carry, measured at **1.7e4 rad/s** -- no limit at
all, and a policy free to learn leg rates the real actuator does not have.

That number is now sourced rather than guessed. See
`LEG_JOINT_VELOCITY_LIMIT_RAD_S` below for the file and line it is read from, and
for why the field must be `velocity_limit_sim` and not `velocity_limit`.

THE WHEELS ARE STILL UNLIMITED, DELIBERATELY, and this is the gap that remains.
The hub drive is not a Unitree part -- `robots/hound/CARD.md` says so under
Provenance -- so no vendor sheet covers it and there is no wheel top speed
anywhere in this repository to read. An invented one would be worse than none:
`velocity_limit_sim` makes the solver *brake* a joint that exceeds it
(`isaaclab/actuators/actuator_base_cfg.py:96-99`), so a guessed number on a wheel
meant to coast is an invented brake nobody derived. `check_hound.py` asserts the
wheels stay unlimited, so a later guess cannot arrive quietly.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from bestiary import paths
from bestiary.robots.hound.build import SPEC

#: Body prim name of the trunk inside the converted asset, and the path from the
#: spawned robot prim down to it.
#:
#: Isaac Lab's own quadrupeds come from the URDF importer, which flattens every
#: link to a direct child of the asset root -- which is why the shipped configs
#: can say `{ENV_REGEX_NS}/Robot/base`. The MJCF importer does not flatten: it
#: preserves the kinematic tree under a `Geometry` scope, so the trunk is at
#: `Robot/Geometry/trunk` and a wheel is four levels below that. Anything that
#: takes a prim path rather than a body name -- the height scanner, for one --
#: has to be told.
TRUNK_BODY = "trunk"
TRUNK_PRIM_SUBPATH = f"Geometry/{TRUNK_BODY}"

#: Joint-name regexes, in the order the MuJoCo action vector uses per leg.
LEG_JOINT_EXPR = [".*_abduct", ".*_hip", ".*_knee"]
WHEEL_JOINT_EXPR = [".*_wheel"]

#: The solved standing stance, as joint-name regexes. `Spec.stance_knee` is
#: DERIVED (it puts each wheel axle under its own hip pivot) and is read here,
#: never typed -- change `Spec.stance_hip` and this follows.
STANCE = {
    ".*_abduct": SPEC.stance_abduct,
    ".*_hip": SPEC.stance_hip,
    ".*_knee": SPEC.stance_knee,
    # A wheel has no rest pose. Zero is the only defensible default and it means
    # "as authored", not "a pose to hold".
    ".*_wheel": 0.0,
}

#: Trunk-origin height with the wheels resting on z = 0, from `Spec.stand_z`
#: (axle drop 0.2784 m + wheel radius 0.085 m = 0.3634 m).
STAND_HEIGHT_M = SPEC.stand_z

#: Extra clearance added to the spawn height, in metres. 5 mm, the same value
#: `envs/hound.py` already uses on the MuJoCo desert, and for the same reason: a
#: spawn that intersects the ground is resolved by the solver as an explosion
#: rather than as a placement.
SPAWN_CLEARANCE_M = 0.005

#: Control period the wheel velocity gain is derived for, in seconds. This is
#: the inherited Isaac Lab locomotion env's `decimation * sim.dt` = 4 * 0.005 s,
#: i.e. 50 Hz. `hound_desert_env_cfg.py` asserts its own env matches, so the
#: gain cannot silently be derived for a rate the simulation does not run at.
CONTROL_DT_S = 0.02

#: Rated joint speed of a Go2 leg actuator, rad/s. Applies to the twelve leg
#: joints only; the wheels have no source and are left unlimited.
#:
#: READ, NOT GUESSED. `Spec` does not carry it, so it comes from the config Isaac
#: Lab ships for the robot whose kinematics, masses and rated torques Hound is
#: built on:
#:
#:     UNITREE_GO2_CFG.actuators["base_legs"] = DCMotorCfg(
#:         ..., effort_limit=23.5, velocity_limit=30.0, ...)
#:     ~/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:176
#:     (read 2026-07-30, Isaac Lab VERSION 3.0.0)
#:
#: The same file's Go1 actuator carries the identical 30.0 at line 43 with the
#: comment `# taken from spec sheet`, and its `effort_limit` there is 23.7 N*m --
#: the number `Spec.gear_abduct` and `Spec.gear_hip` already use. So this is the
#: same vendor sheet the torques came from, not a second opinion about the robot.
#:
#: The literal is pinned HERE rather than imported from `isaaclab_assets`, on
#: purpose: importing it would let an upstream edit change this machine's
#: dynamics silently between two runs. `check_hound.py` imports the shipped
#: config and compares, so a divergence is reported loudly instead of applied.
#:
#: 30 rad/s is reachable, not decorative: the knee's drive can put ~40 N*m into
#: roughly 0.035 kg*m^2 of joint-space inertia, which is over 1000 rad/s^2, so a
#: full-authority command crosses this limit inside two control periods.
LEG_JOINT_VELOCITY_LIMIT_RAD_S = 30.0


def wheel_spin_inertia() -> float:
    """Rotational inertia the wheel drive has to accelerate, kg*m^2.

    The wheel body is one massive geom -- a solid cylinder of `Spec.wheel_mass`
    at `Spec.wheel_r`, everything else on that body being `class="hub"` with
    `density="0"` -- so its inertia about the spin axis is (1/2) m r^2. The
    drive also feels the reflected rotor inertia, which `Spec.wheel_armature`
    adds directly to the joint-space inertia in both simulators.

    Verified against the compiled MuJoCo model rather than trusted: MuJoCo's
    `inertiafromgeom` gives FL_wheel a spin-axis inertia of 1.62563e-3 kg*m^2,
    which is (1/2)(0.45)(0.085^2) to every digit it prints.
    """
    return 0.5 * SPEC.wheel_mass * SPEC.wheel_r**2 + SPEC.wheel_armature


def wheel_velocity_gain() -> float:
    """Velocity-drive gain for one wheel, N*m/(rad/s).

    `Spec` does not contain this number, because MuJoCo drives these wheels with
    torque and never needed one. It is derived rather than chosen, so that it
    moves when the machine does.

    A velocity drive on a free wheel is a first-order system: with the wheel
    off the ground and the drive the only torque on it,

        I dw/dt = -k (w - w_ref)      =>      time constant T = I / k

    Pick T = one control period and the wheel reaches a commanded speed inside
    the interval the policy commanded it for -- faster is a gain the policy
    cannot observe the effect of, slower is a drive the policy has to plan
    around. So k = I / dt, with I from `wheel_spin_inertia()`:

        k = (1.62563e-3 + 4.0e-3) / 0.02 = 0.28128 N*m/(rad/s)

    The consequence to be aware of, and the reason this is honest rather than
    arbitrary: at 3.0 N*m of effort ceiling the drive saturates at a velocity
    error of 3.0 / 0.28128 = 10.665 rad/s, which is 0.906 m/s of rim speed. Past
    that error the wheel is a pure torque source at the traction limit, which is
    the regime the CARD's traction budget describes.
    """
    return wheel_spin_inertia() / CONTROL_DT_S


def wheel_action_scale() -> float:
    """Wheel speed one unit of action commands, rad/s.

    The largest velocity error the drive can still act on: past
    `gear_wheel / k` the effort ceiling clips and a bigger command buys nothing.
    Scaling the action to exactly that keeps the whole action range useful and
    keeps the saturation point at |action| = 1 rather than somewhere the policy
    has to discover.
    """
    return SPEC.gear_wheel / wheel_velocity_gain()


HOUND16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(paths.HOUND_ISAAC_USD),
        activate_contact_sensors=True,
        # -- STATIC FRICTION, WHICH THE CONVERTED ASSET DOES NOT HAVE ---------
        # This is the single most expensive thing found while porting this
        # robot, and it is invisible in every file you would think to check.
        #
        # MuJoCo has ONE sliding-friction coefficient per geom. USD Physics has
        # two, `physics:staticFriction` and `physics:dynamicFriction`, and the
        # MJCF importer writes only the dynamic one. `staticFriction` is then
        # whatever the USD schema defaults it to, which is ZERO -- so every
        # contact on the machine has full sliding friction and no grip at rest.
        #
        # Symptom: the robot is placed in a correct stance on flat ground, and
        # inside half a second it does the splits and lands on its side. It looks
        # like an unstable stance, or a bad wheel collider, or a missing spring;
        # `robots/hound/CARD.md` even predicts a splits failure from a different
        # cause, which makes the wrong explanation feel confirmed.
        #
        # `Spec.wheel_friction[0]` is 0.9 and `Spec.body_friction[0]` is 0.9 too,
        # so one material for the whole robot loses nothing: the MJCF's tyre and
        # body agree on the sliding coefficient, and the torsional and rolling
        # terms are inert at condim=3 anyway. Restitution is 0 because the MJCF
        # models none.
        #
        # `RigidBodyMaterialBaseCfg` is imported from `isaaclab.sim.spawners.
        # materials` rather than off `isaaclab.sim`, which lazy-loads a name list
        # that still only exports the deprecated `RigidBodyMaterialCfg`
        # (`AttributeError: No isaaclab.sim attribute RigidBodyMaterialBaseCfg`).
        # The base class is the solver-common one -- friction and restitution
        # only, no PhysX-specific combine modes -- which is what this needs.
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=SPEC.wheel_friction[0],
            dynamic_friction=SPEC.wheel_friction[0],
            restitution=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            # DEGREES per second, sitting next to a m/s field. Anything wrong by
            # a factor of 57.3 downstream is this.
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # MuJoCo already excludes parent-child contacts and this machine's
            # links are not authored to survive colliding with each other. The
            # converted asset says the same thing (`newton:selfCollisionEnabled
            # = 0`); saying it here keeps the two from drifting.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STAND_HEIGHT_M + SPAWN_CLEARANCE_M),
        joint_pos=STANCE,
        joint_vel={".*": 0.0},
    ),
    # Rotation is left at the class default (0, 0, 0, 1). Note that Isaac Lab
    # 3.0 changed quaternion order from WXYZ to XYZW, so (1, 0, 0, 0) -- which
    # was identity in 2.x and reads like identity everywhere in this repo's
    # MuJoCo models -- is now a 180-degree turn about +X. Not writing it is
    # safer than writing it from memory.
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # -- Twelve leg joints: POSITION control ------------------------------
        # stiffness/damping are `Spec`'s PD gains, the ones robots/hound/play.py
        # already drives this machine with; effort_limit_sim is the same Go2
        # torque ceiling the MuJoCo model has, so the machine is no easier to
        # drive here, only easier to command.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINT_EXPR,
            effort_limit_sim={
                ".*_abduct": SPEC.gear_abduct,
                ".*_hip": SPEC.gear_hip,
                ".*_knee": SPEC.gear_knee,
            },
            # `velocity_limit_sim`, NOT `velocity_limit`. On an implicit actuator
            # the second one is accepted, warned about, and then set to None --
            # "Since this parameter affects the simulation behavior, we continue
            # to not use it" (isaaclab/actuators/actuator_pd.py:81-91). So the
            # field that reads correctly is the field that does nothing, and only
            # the `_sim` suffix reaches the solver. Same trap as
            # `effort_limit_sim` above.
            velocity_limit_sim=LEG_JOINT_VELOCITY_LIMIT_RAD_S,
            stiffness={
                ".*_abduct": SPEC.kp_abduct,
                ".*_hip": SPEC.kp_hip,
                ".*_knee": SPEC.kp_knee,
            },
            damping={
                ".*_abduct": SPEC.kv_abduct,
                ".*_hip": SPEC.kv_hip,
                ".*_knee": SPEC.kv_knee,
            },
            armature={".*": SPEC.armature},
            friction={".*": SPEC.frictionloss},
        ),
        # -- Four wheels: VELOCITY control -----------------------------------
        # stiffness MUST be zero; see the module docstring. `friction` is the
        # hub drive's breakaway torque, which at condim=3 is the only thing that
        # stops a coasting wheel -- Spec derives it as 13.3% of rated torque and
        # explains why that is partly hardware and partly the heightfield.
        #
        # NO `velocity_limit_sim` HERE, and that is a decision rather than an
        # omission. The wheel is the one part of this machine with no vendor
        # sheet behind it, so there is no rated speed to read; and because
        # `velocity_limit_sim` makes the solver brake a joint past the limit, a
        # guessed number would be an undeclared brake on a wheel whose whole
        # point is that it coasts. What actually bounds the wheel is the drive:
        # `wheel_action_scale()` commands at most 10.665 rad/s (0.906 m/s of rim
        # speed, 2.97 ft/s) and `effort_limit_sim` caps the torque at 3.0 N*m,
        # which the CARD's traction budget already sizes against the ground.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_EXPR,
            effort_limit_sim=SPEC.gear_wheel,
            stiffness={".*": 0.0},
            damping={".*": wheel_velocity_gain()},
            armature={".*": SPEC.wheel_armature},
            friction={".*": SPEC.wheel_brake_torque()},
        ),
    },
)
