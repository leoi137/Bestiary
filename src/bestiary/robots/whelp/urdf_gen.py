"""Emit the WHELP-16 URDF for Isaac Lab, from robot.json.

    python -m bestiary.robots.whelp.urdf_gen           # write the URDF + the actuator cfg
    python -m bestiary.robots.whelp.urdf_gen --print   # to stdout

READS robot.json, NOT spec.py. That is deliberate: the URDF is downstream of the
export, so if robot.json is stale this file emits a stale URDF and check.py
catches the mismatch, rather than the two silently agreeing because they both
recomputed from source and happened to agree. One direction of data flow.

WHAT THIS FILE KNOWS THAT A GENERIC URDF WRITER DOES NOT
--------------------------------------------------------
Six things, each of which is a documented way an Isaac Lab import goes wrong,
and each of which is silent. None of them raises. They present as "the robot
explodes on step 1" or, worse, as a policy that trains fine and does not
transfer.

1. WHEELS MUST NOT BE POSITION-DRIVEN. A URDF `continuous` joint imports as a
   PhysicsRevoluteJoint with sentinel limits of 3.4028235e+38, and PhysX then
   refuses a drive target outside [-2pi, 2pi]. A position-controlled wheel
   therefore breaks after about 3.2 revolutions of accumulated command. The
   wheels get `type="continuous"` here AND the generated actuator config puts
   them in a separate VELOCITY-driven group. Twelve leg joints, four wheels; two
   groups, not one.

2. WHEEL COLLISION MUST BE A CYLINDER PRIMITIVE, NOT A MESH. A meshed wheel
   under the importer's default Convex Hull is cooked down to a vertex limit --
   it becomes an N-gon, and it rolls with N contact impulses per revolution.
   That is the actual mechanism of "wheel chatter". Omniverse states plainly
   that cylinder prims get custom-geometry collision and are what wheels should
   use. The pretty tire mesh stays as VISUAL only.

3. EVERY LINK GETS AN EXPLICIT, VALID INERTIA. PhysX reads a zero inertia as
   INFINITE inertia about that axis, and Isaac Lab's link_density defaults to
   0.0, so a link with no <inertial> gets density-from-volume instead. Both give
   a robot whose dynamics nobody designed. Worse, PhysX does not appear to check
   the triangle inequality, so a tensor describing an impossible object
   simulates happily and a policy will find the exploit. `_assert_inertia`
   refuses to emit a link that fails it.

4. MESH SCALE. URDF is metres by definition; OpenSCAD authors in millimetres.
   UrdfConverterCfg exposes no distance_scale, so this must be fixed here or not
   at all -- hence `scale="0.001 0.001 0.001"` on every mesh, and a check that
   the assembled robot is under a metre.

5. HIGH-ASPECT-RATIO CONVEX HULLS FALL BACK TO CPU. A 110 mm thigh is exactly
   the long thin shape that produces a convex hull incompatible with GPU
   simulation; the symptom is a warning in the log and an order of magnitude of
   lost throughput. Collision geometry here is therefore PRIMITIVES for every
   link, not hulls of the printed shape.

6. EFFORT AND VELOCITY LIMITS ARE NOT DECORATIVE. They are the only thing
   stopping a policy from learning a gait out of joint rates the servo cannot
   produce -- and the STS3215 is about six times slower than the 30 rad/s that
   published legged-RL configs assume. Note the Isaac Lab trap: ImplicitActuator
   IGNORES velocity_limit; the one the solver enforces is velocity_limit_sim.
   The generated actuator config sets both.
"""
from __future__ import annotations

import json
import sys

from bestiary.robots.whelp.export import OUT_DIR, ROBOT_JSON, robot_dict
from bestiary.robots.whelp.spec import SPEC, Spec

URDF_PATH = OUT_DIR / "whelp16.urdf"
ACTUATOR_CFG_PATH = OUT_DIR / "whelp16_actuators.py"

#: Meshes are authored in millimetres by OpenSCAD; URDF is metres by definition.
MESH_SCALE = "0.001 0.001 0.001"


def _f(x: float) -> str:
    return f"{x:.9g}"


def _xyz(v) -> str:
    return " ".join(_f(c) for c in v)


def _assert_inertia(link: dict) -> None:
    """Refuse to emit a link whose inertia is not physically possible.

    The single highest-value assertion between CAD and simulation. See point 3
    of the module docstring: nothing downstream will catch this, because nothing
    downstream checks.
    """
    if not link["inertia_valid"]:
        raise ValueError(
            f"link {link['name']!r} has an invalid inertia and will not be emitted: "
            f"{link['inertia_problem']}. Fix massmodel.py rather than relaxing this check -- "
            f"PhysX will simulate an impossible tensor without complaint and a policy will "
            f"learn to exploit it."
        )
    if link["mass_kg"] <= 0:
        raise ValueError(f"link {link['name']!r} has mass {link['mass_kg']}")


def _inertial(link: dict, indent: str) -> str:
    _assert_inertia(link)
    i = link["inertia"]
    return (
        f'{indent}<inertial>\n'
        f'{indent}  <origin xyz="{_xyz(link["com_m"])}" rpy="0 0 0"/>\n'
        f'{indent}  <mass value="{_f(link["mass_kg"])}"/>\n'
        f'{indent}  <inertia ixx="{_f(i["ixx"])}" ixy="{_f(i["ixy"])}" ixz="{_f(i["ixz"])}"'
        f' iyy="{_f(i["iyy"])}" iyz="{_f(i["iyz"])}" izz="{_f(i["izz"])}"/>\n'
        f'{indent}</inertial>\n'
    )


def _collision_geometry(link_name: str, spec: Spec) -> str:
    """PRIMITIVES only. See points 2 and 5 of the module docstring.

    Every shape here is a deliberate simplification of the printed part, chosen
    to be cheap and stable in PhysX rather than accurate. The visual meshes carry
    the real shape; contact does not need it and is much better without it.
    """
    m = 0.001
    if link_name == "trunk":
        return (f'<box size="{_f(spec.trunk_len_mm * m)} {_f(spec.trunk_width_mm * m)} '
                f'{_f(spec.trunk_height_mm * m)}"/>')
    if link_name.endswith("_hip"):
        return f'<box size="{_f(0.050)} {_f(0.048)} {_f(0.045)}"/>'
    if link_name.endswith("_thigh"):
        # A capsule would be better but URDF has no capsule; a slightly-inset box
        # is stable and cheap, and the thigh is not the contact surface anyway.
        return (f'<box size="{_f(spec.thigh_section_fore_aft_mm * m)} '
                f'{_f(spec.thigh_section_lateral_mm * m)} {_f(spec.thigh_len_mm * m)}"/>')
    if link_name.endswith("_calf"):
        return (f'<box size="{_f(spec.calf_section_fore_aft_mm * m)} '
                f'{_f(spec.calf_section_lateral_mm * m)} {_f(spec.calf_len_mm * m)}"/>')
    if link_name.endswith("_wheel"):
        # THE one that matters. A cylinder prim gets custom-geometry collision in
        # PhysX and rolls smoothly; a convex hull of the tire mesh is an N-gon and
        # is where wheel chatter comes from.
        return (f'<cylinder radius="{_f(spec.wheel_radius_mm * m)}" '
                f'length="{_f(spec.wheel_width_mm * m)}"/>')
    raise ValueError(f"no collision primitive defined for link {link_name!r}")


def _collision_origin(link_name: str, spec: Spec) -> str:
    """Where the primitive sits in the link frame.

    Link frames are at the driving JOINT, so a leg segment's box is centred half
    its length down the link, not at the origin. Getting this wrong puts the
    collider somewhere the visual is not, and the symptom -- a robot that snags
    on nothing -- looks like a physics bug rather than an authoring one.
    """
    m = 0.001
    if link_name.endswith("_thigh"):
        return f'xyz="0 0 {_f(-spec.thigh_len_mm * m / 2)}" rpy="0 0 0"'
    if link_name.endswith("_calf"):
        return f'xyz="0 0 {_f(-spec.calf_len_mm * m / 2)}" rpy="0 0 0"'
    if link_name.endswith("_wheel"):
        # A URDF cylinder's axis is +Z; the wheel spins about +Y, so roll it 90
        # degrees about X. A wheel collider left on the default axis is a disc
        # lying flat on the ground and the robot simply sinks.
        return 'xyz="0 0 0" rpy="1.5707963 0 0"'
    return 'xyz="0 0 0" rpy="0 0 0"'


def _mesh_name(link_name: str) -> str | None:
    if link_name == "trunk":
        return "trunk_front"
    for suffix, mesh in (("_hip", "abduct_bracket"), ("_thigh", "thigh"),
                         ("_calf", "calf"), ("_wheel", "wheel_hub")):
        if link_name.endswith(suffix):
            return mesh
    return None


def build_urdf(robot: dict, spec: Spec = SPEC, with_meshes: bool = True) -> str:
    out: list[str] = []
    w = out.append

    w('<?xml version="1.0"?>')
    w("<!--")
    w("  WHELP-16 — GENERATED by `python -m bestiary.robots.whelp.urdf_gen`. DO NOT EDIT.")
    w("")
    w(f"  {len(robot['joints'])} actuated joints: 4 legs x (abduct + hip + knee + wheel).")
    w(f"  {robot['total_mass_kg']:.3f} kg, wheelbase {robot['wheelbase_m'] * 1000:.0f} mm, "
      f"track {robot['track_m'] * 1000:.0f} mm.")
    w(f"  Servo: {robot['servo_variant']}. Structure: {robot['material']}.")
    w("")
    w("  READ THIS BEFORE IMPORTING:")
    w("    * The four wheel joints are `continuous` and MUST be velocity-driven. A position")
    w("      target on a continuous joint fails in PhysX past +/-2*pi of accumulated")
    w("      command. Use the two actuator groups in whelp16_actuators.py.")
    w("    * Wheel collision is a <cylinder> primitive on purpose. Do not let the importer")
    w("      replace it with a convex hull of the tire mesh; that is what wheel chatter is.")
    w("    * All meshes are millimetre-authored and carry scale=\"0.001 0.001 0.001\".")
    w("    * Effort limits are the servo's STALL torque. Its CONTINUOUS rating is a third of")
    w(f"      that ({spec.leg_servo_rated_nm:.2f} N.m). A policy that lives near the effort")
    w("      limit will cook twelve servos in an afternoon; penalise torque and action rate.")
    w("-->")
    w(f'<robot name="{robot["name"]}">')
    w("")

    # ── Links ────────────────────────────────────────────────────────────────
    for link in robot["links"]:
        name = link["name"]
        w(f'  <link name="{name}">')
        w(_inertial(link, "    ").rstrip("\n"))
        mesh = _mesh_name(name) if with_meshes else None
        if mesh:
            w('    <visual>')
            w(f'      <origin {_collision_origin(name, spec)}/>'
              if name.endswith("_wheel") else '      <origin xyz="0 0 0" rpy="0 0 0"/>')
            w('      <geometry>')
            w(f'        <mesh filename="stl/{mesh}.stl" scale="{MESH_SCALE}"/>')
            w('      </geometry>')
            w('    </visual>')
        w('    <collision>')
        w(f'      <origin {_collision_origin(name, spec)}/>')
        w('      <geometry>')
        w(f'        {_collision_geometry(name, spec)}')
        w('      </geometry>')
        w('    </collision>')
        w('  </link>')
        w("")

    # ── Joints ───────────────────────────────────────────────────────────────
    for j in robot["joints"]:
        w(f'  <joint name="{j["name"]}" type="{j["type"]}">')
        w(f'    <parent link="{j["parent"]}"/>')
        w(f'    <child link="{j["child"]}"/>')
        w(f'    <origin xyz="{_xyz(j["origin_xyz"])}" rpy="{_xyz(j["origin_rpy"])}"/>')
        w(f'    <axis xyz="{_xyz(j["axis"])}"/>')
        if j["type"] == "continuous":
            # A continuous joint has no position limits by definition, but it MUST
            # still carry effort and velocity or the wheel spins unbounded in sim
            # and the policy learns a speed the motor does not have.
            w(f'    <limit effort="{_f(j["effort_nm"])}" '
              f'velocity="{_f(j["velocity_rad_s"])}"/>')
        else:
            w(f'    <limit lower="{_f(j["limit_lower"])}" upper="{_f(j["limit_upper"])}" '
              f'effort="{_f(j["effort_nm"])}" velocity="{_f(j["velocity_rad_s"])}"/>')
        w(f'    <dynamics damping="{_f(j["damping"])}" friction="{_f(j["friction"])}"/>')
        w('  </joint>')
        w("")

    w('</robot>')
    return "\n".join(out) + "\n"


# ── Isaac Lab actuator configuration ─────────────────────────────────────────
def build_actuator_cfg(robot: dict, spec: Spec = SPEC) -> str:
    """The companion Isaac Lab config, because the URDF alone cannot express this.

    A URDF has no way to say "these twelve joints are position-controlled and
    those four are velocity-controlled", and no way to carry reflected rotor
    inertia -- which at 1:345 is the single most consequential dynamic parameter
    on this robot. Both live here.
    """
    knee = robot["stance_solved_knee_rad"]
    return f'''"""Isaac Lab actuator and articulation config for WHELP-16.

GENERATED by `python -m bestiary.robots.whelp.urdf_gen`. DO NOT EDIT.

WHY THIS FILE EXISTS AT ALL
---------------------------
A URDF cannot express three things this robot depends on:

  1. That the twelve leg joints are POSITION-controlled and the four wheels are
     VELOCITY-controlled. There is no URDF tag for it, and getting it wrong is
     not a subtle degradation: a position target on a `continuous` joint fails
     in PhysX once the accumulated command passes +/-2*pi, i.e. after about
     three revolutions of driving.

  2. Reflected rotor inertia. At 1:345 the STS3215 presents
     {spec.servo_armature_kgm2:.3f} kg.m^2 at the joint, roughly a hundred times
     the thigh's own inertia. Leave it out and the simulated leg is a nearly
     massless whip that a policy will flick at rates the real leg cannot reach.
     It is the most-omitted parameter in hobby-robot URDFs and on a machine this
     light it dominates.

  3. That effort_limit and velocity_limit are IGNORED by ImplicitActuatorCfg.
     The ones the solver enforces are effort_limit_sim and velocity_limit_sim.
     Setting only the first pair is a config that looks correct, changes
     nothing, and lets the policy learn joint speeds the servo does not have.

THE ACTUATOR IS NOT AN EFFORT SOURCE
------------------------------------
The STS3215 has four operating modes -- position, constant speed, open-loop PWM,
and step -- and no torque or current mode at all. The policy emits a POSITION
TARGET and the servo closes its own loop internally, at its own rate, with
firmware integer gains (P=32, D=32, I=0 by default), its own deadband and its own
saturation. An ideal-effort actuator model in simulation is therefore a model of
a robot nobody owns, and that mismatch is the classic cause of "it worked in
simulation".

stiffness and damping below are a STARTING POINT, not an identification. There is
no published mapping from the firmware's integer P/D to N.m/rad. Fit them to a
measured step response on a real servo at your real bus voltage before a long run,
and expect two units from one batch to differ.
"""
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Solved standing stance. The knee is NOT typed: geometry.solve_stance_knee()
# places the wheel axle directly under the hip pivot, which makes the hip's
# static holding torque exactly zero.
STANCE = {{
    ".*_abduct": {robot["stance"]["FL_abduct"]:.6f},
    ".*_hip": {robot["stance"]["FL_hip"]:.6f},
    ".*_knee": {knee:.6f},
    ".*_wheel": 0.0,
}}

#: Trunk-origin height with the wheels resting on the ground, at the LOADED wheel
#: radius. Spawning at the free radius puts the robot into the floor, and PhysX
#: resolves that by launching it.
STAND_HEIGHT = {robot["stand_height_m"]:.6f}

WHELP16_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="{{USD_PATH}}",   # produced by scripts/tools/convert_urdf.py
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            # NOTE: documented in DEGREES per second while max_linear_velocity
            # next to it is m/s. Any number that looks wrong by 57.3x is this.
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STAND_HEIGHT),
        joint_pos=STANCE,
        joint_vel={{".*": 0.0}},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={{
        # ── The twelve leg joints: POSITION control ──────────────────────────
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_abduct", ".*_hip", ".*_knee"],
            effort_limit_sim={spec.leg_servo_stall_nm:.4f},
            velocity_limit_sim={spec.leg_servo_no_load_rad_s:.4f},
            stiffness={{".*": 8.0}},
            damping={{".*": 0.4}},
            armature={{".*": {spec.servo_armature_kgm2:.6f}}},
            friction={{".*": {spec.joint_friction_nm:.6f}}},
        ),
        # ── The four wheels: VELOCITY control ────────────────────────────────
        # stiffness MUST be zero. A non-zero position gain on a continuous joint
        # is the failure described at the top of this file.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel"],
            effort_limit_sim={spec.wheel_drive_stall_nm:.4f},
            velocity_limit_sim={spec.wheel_drive_no_load_rad_s:.4f},
            stiffness={{".*": 0.0}},
            damping={{".*": 0.6}},
            armature={{".*": {spec.servo_armature_kgm2:.6f}}},
            friction={{".*": {spec.wheel_friction_nm:.6f}}},
        ),
    }},
)

# ── The envelope this hardware actually has ─────────────────────────────────
# These are not advisory. They are what the machine can do, and a policy trained
# outside them cannot be run on it.
#
#   top speed              {robot["envelope"]["top_speed_m_s"]:.2f} m/s   (wheel drive, no load)
#   leg joint rate         {robot["envelope"]["leg_joint_rad_s"]:.2f} rad/s   about a sixth of what
#                                    published legged-RL configs assume
#   design drop height     {robot["envelope"]["design_drop_m"] * 1000:.0f} mm     above this the knee is over
#                                    its peak rating; the gearbox cannot yield
#                                    on impact timescales, so it goes into plastic
#
# The control loop is bounded by FEEDBACK, not commands: a broadcast sync-write
# to sixteen servos returns nothing and is fast, while a sync-read serialises
# sixteen replies on one half-duplex wire. Train at 50 Hz decimation and treat
# anything faster as headroom you have not measured.
MAX_SPEED_M_S = {robot["envelope"]["top_speed_m_s"]:.4f}
MAX_JOINT_RATE_RAD_S = {robot["envelope"]["leg_joint_rad_s"]:.4f}
DESIGN_DROP_M = {robot["envelope"]["design_drop_m"]:.4f}
CONTROL_HZ = 50.0
'''


def main(argv: list[str]) -> int:
    if not ROBOT_JSON.exists():
        print(f"{ROBOT_JSON} does not exist; run `python -m bestiary.robots.whelp.export` first")
        return 2
    robot = json.loads(ROBOT_JSON.read_text())

    # Compare the WHOLE document, not one summary number. Total mass is invariant
    # under most edits that matter -- a changed joint limit, a moved axis, a new
    # stance angle -- so a mass-only comparison would let a stale robot.json
    # through for exactly the changes worth catching.
    fresh = robot_dict(SPEC)
    if json.dumps(fresh, sort_keys=True) != json.dumps(robot, sort_keys=True):
        print("robot.json is STALE: spec.py has changed since it was written. "
              "Re-run `python -m bestiary.robots.whelp.export`.")
        return 2

    try:
        urdf = build_urdf(robot, SPEC)
    except ValueError as exc:
        print(f"refusing to emit URDF: {exc}")
        return 1

    if "--print" in argv:
        print(urdf)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    URDF_PATH.write_text(urdf, encoding="utf-8")
    ACTUATOR_CFG_PATH.write_text(build_actuator_cfg(robot, SPEC), encoding="utf-8")
    print(f"wrote {URDF_PATH}")
    print(f"wrote {ACTUATOR_CFG_PATH}")
    n_cont = sum(1 for j in robot["joints"] if j["type"] == "continuous")
    print(f"  {len(robot['links'])} links, {len(robot['joints'])} joints "
          f"({n_cont} continuous), {robot['total_mass_kg']:.3f} kg")
    print()
    print("  Import with:")
    print("    ./isaaclab.sh -p scripts/tools/convert_urdf.py \\")
    print(f"      {URDF_PATH} <out.usd> --joint-stiffness 0 --joint-damping 0 "
          "--joint-target-type none")
    print("  then set the drives from whelp16_actuators.py rather than at import time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
