"""Spyder-12 as an Isaac Lab articulation: twelve position-driven leg joints.

The numbers here trace to exactly two places: `assets/spyder12.xml` (the
authored machine — masses, ranges, armature, the 40 N·m effort ceiling) and
the derivations below (the PD gains, each with its formula and its inputs
named). There is no `Spec` to import, because Spyder has no generator; the
authored XML is the source of truth and `check_spyder.py` re-reads it to hold
this file honest.

ONE ACTUATOR GROUP, AND WHY THAT IS THE WHOLE STORY
---------------------------------------------------
Every Hound lesson about actuator groups was a wheel lesson: the unbounded
joint, the velocity drive, the split acceleration budget. Spyder has no
wheels. Twelve hinge joints, all limited (±40/±60/±50 degrees), all position
controlled, one group. The port is boring by construction, which is the reason
this robot was chosen for the first Isaac training rung.

THE PD GAINS ARE DERIVED, NOT COPIED, AND NOT TUNED
---------------------------------------------------
The MuJoCo machine holds its arch with a joint spring: `stiffness="30"` on
every joint, a labelled reset crutch (spyder12.xml's default block). The
conversion input zeroes that spring precisely so the implicit PD drive can
take its job — see `spyder_usd.py` delta 3 — which makes the drive stiffness

    KP = 30 N·m/rad,  the authored spring constant.

Same restoring torque toward the same stance the MuJoCo machine has at zero
torque; a zero ACTION here is the standing arch, exactly the prior
`envs/spyder.py` gives the SAC policy. Choosing any other number would make
"the machine that trained in MuJoCo" and "the machine training here" two
different machines for no stated reason.

Damping closes the loop the spring never had to. MuJoCo's `damping="1"` gave
the free oscillation a damping ratio of 1/(2*sqrt(30*1.04)) ≈ 0.09 — fine for
a torque-controlled machine whose policy IS the damper, wrong for a position
drive that must settle where it is sent. So KD comes from the standard
second-order form, at half-critical:

    KD = 2 * zeta * sqrt(KP * I_joint),   zeta = 0.5
       = 2 * 0.5 * sqrt(30 * 1.04) ≈ 5.6 N·m·s/rad

with I_joint = 1.04 kg·m², measured from the compiled model (mj_fullM at
qpos0: M_ii = 1.010..1.037 — the `armature="1"` rotor term dominates; the limb
itself contributes at most 0.037). Half-critical rather than critical is a
saturation argument, not a taste: at critical (KD ≈ 11.2) a routine 3.6 rad/s
swing puts the damping torque alone at the 40 N·m effort ceiling, and a drive
that saturates on damping cannot track anything. zeta = 0.5 halves that and
accepts ~16% overshoot on a step. The natural frequency sqrt(30/1.04) ≈ 5.4
rad/s is the same one the MuJoCo spring gives this machine — the drive does
not make the joint faster, only settled.

WHAT IS DELIBERATELY ABSENT
---------------------------
NO `velocity_limit_sim`. The authored MJCF has no joint speed limit and no
vendor sheet exists for a machine that was never hardware — Hound's 30 rad/s
came from Go2's datasheet, and Spyder has no datasheet to read. An invented
number would be an undeclared brake (`velocity_limit_sim` makes the solver
brake a joint past it), so none is set and `check_spyder.py` asserts none
arrives by accident. What actually bounds leg speed is the effort ceiling
against the armature: 40 N·m into 1.04 kg·m² is 38 rad/s² of authority.

NO joint `friction`. The MJCF has no frictionloss; inventing one would slow a
machine nobody measured.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

from bestiary import paths

#: Body prim name of the torso inside the converted asset, and the path from
#: the spawned robot prim down to it. The MJCF importer preserves the kinematic
#: tree under a `Geometry` scope (measured on the Hound port), so anything that
#: takes a prim path — the height scanner — needs the subpath, not the name.
TORSO_BODY = "torso"
TORSO_PRIM_SUBPATH = f"Geometry/{TORSO_BODY}"

#: Joint-name regexes. One group; the order here is documentation, not a
#: contract — Isaac Lab orders the articulation's joints by tree depth
#: (hip_1..4, lift_1..4, knee_1..4), NOT by the MuJoCo actuator block's
#: per-leg (hip, lift, knee) x 4. Do not read an action index from one
#: simulator's logs and use it in the other.
JOINT_EXPR = ["hip_.*", "lift_.*", "knee_.*"]

#: The standing stance IS qpos 0: spyder12.xml authors the arch at zero joint
#: angle (the spring's return target, the shape the shell was modelled in).
STANCE = {".*": 0.0}

#: Torso height with all twelve joints at stance and the tibia tips touching
#: z = 0. Read from the authored model: torso sits at 0.35 with foot-capsule
#: centres 0.27 below it and radius 0.08 — 0.35 = 0.27 + 0.08 exactly, so the
#: authored spawn is the standing contact height. `spyder_usd.py` asserts the
#: authored value still is 0.35 before zeroing it out of the asset.
STAND_HEIGHT_M = 0.35

#: Same 5 mm the Hound and the MuJoCo desert envs use: a spawn that intersects
#: the ground is resolved by the solver as an explosion, not a placement.
SPAWN_CLEARANCE_M = 0.005

#: Effort ceiling, N·m: `<motor ctrlrange="-1 1" gear="40"/>` in the authored
#: default block — gear x ctrlrange = 40. The conversion input drops the motor
#: elements (spyder_usd delta 3), so this constant is where the ceiling lives
#: now; check_spyder re-reads the MJCF and asserts the two agree.
EFFORT_LIMIT_NM = 40.0

#: Rotor inertia, kg·m², from the authored `armature="1"` — passed through the
#: conversion (it is dynamics, not a crutch) and repeated here for the solver
#: config and the KD derivation. check_spyder asserts XML and cfg agree.
ARMATURE = 1.0

#: Joint-space inertia the drive acts against, kg·m². Measured: mj_fullM diag
#: at qpos0 spans 1.010 (knee) .. 1.037 (hip); one number for one gain, taken
#: at the heavy end so zeta is a floor, not an average.
JOINT_INERTIA = 1.04

#: Drive stiffness: the authored spring constant. See the module docstring.
KP = 30.0

#: Damping ratio. Half-critical — the saturation arithmetic is in the module
#: docstring; the oracle prints the damping torque at a gait-speed swing so
#: the number stays confronted with what it costs.
ZETA = 0.5

#: Drive damping, derived. 2 * 0.5 * sqrt(30 * 1.04) = 5.586.
KD = 2.0 * ZETA * math.sqrt(KP * JOINT_INERTIA)


SPYDER12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(paths.SPYDER_ISAAC_USD),
        activate_contact_sensors=True,
        # The importer writes only dynamicFriction; staticFriction defaults to
        # ZERO and the robot does the splits at rest. Measured on the Hound
        # port (hound_cfg.py tells it at length). The authored sliding
        # coefficient is 1.0 on every load-bearing geom (`friction="1 0.5
        # 0.5"`), torsional/rolling inert at condim=3; restitution unmodelled.
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            # Degrees per second, next to a m/s field. Same trap as the Hound.
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # False matches the MuJoCo model's effective behaviour. Computed
            # from the MJCF, not eyeballed: neighbouring coxa bases sit 0.40 m
            # apart (authored at (±0.2, ±0.2)), a 0.24 m surface gap at 0.08 m
            # capsule radii, and over the full joint envelope neighbouring
            # legs never close within ~0.13 m — there is nothing between legs
            # for self-collision to catch. What WOULD fire is each coxa
            # against its own torso stub (they share a mount point and
            # interpenetrate by construction) — exactly the parent-child pair
            # MuJoCo excludes by default, so True here would add a permanent
            # phantom contact force the MuJoCo machine never felt.
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STAND_HEIGHT_M + SPAWN_CLEARANCE_M),
        joint_pos=STANCE,
        joint_vel={".*": 0.0},
        # Rotation stays at the class default. Isaac Lab 3.0 is XYZW; writing
        # (1, 0, 0, 0) from 2.x memory would be a 180-degree flip about +X.
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=JOINT_EXPR,
            # `_sim` suffixes, not the bare names: the bare `velocity_limit` is
            # accepted, warned about, and ignored on an implicit actuator, and
            # `effort_limit` mirrors the same trap (hound_cfg.py, measured).
            effort_limit_sim=EFFORT_LIMIT_NM,
            stiffness={".*": KP},
            damping={".*": KD},
            armature={".*": ARMATURE},
        ),
    },
)
