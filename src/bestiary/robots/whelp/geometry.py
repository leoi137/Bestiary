"""The kinematic chain of WHELP-16, derived once and consumed by everything.

    python -m bestiary.robots.whelp.geometry        # print the chain and the stance

ONE DERIVATION, FOUR CONSUMERS
------------------------------
The joint origins, axes and the standing stance are computed here and nowhere
else. They are then read by:

    urdf_gen.py   -> the URDF Isaac Lab imports
    export.py     -> derived_gen.scad, so the CAD assembly preview is posed at
                     the same angles the URDF is, from the same numbers
    torque.py     -> the free-body diagram that sizes the servos
    check.py      -> the assertions

That list is the reason this file exists. A robot whose CAD preview and whose
URDF are posed by two independently-typed sets of angles will disagree, and it
will disagree *quietly*: the render looks fine, the sim runs, and the physical
machine is the one that discovers the leg is 4 degrees off. Deriving once and
generating both removes the possibility rather than the mistake.

UNITS: MILLIMETRES IN, METRES OUT
---------------------------------
Spec is in millimetres because it is a CAD spec -- walls, fastener clearances
and insert bores are natural in mm and unnatural in m, and writing `0.0025` for
a 2.5 mm wall is how a factor of a thousand gets in. Every Spec attribute that
is a length is therefore named `*_mm`.

The conversion to metres happens in exactly one function, `_m()`, at the URDF
boundary. check.py asserts the assembled robot's bounding box is under one
metre, which is the cheap catch for the conversion having been skipped: a
1000x-too-large robot is not subtle, and a 1000x-too-small one is not either.

AXIS CONVENTION
---------------
    +X forward   +Y left   +Z up   metres   radians   right-handed

Rotation of theta about +Y maps the straight-down link vector (0, 0, -L) to
(-L sin(theta), 0, -L cos(theta)). So POSITIVE PITCH SWINGS A LINK BACKWARD
and up. This is the same convention robots/hound/build.py uses, deliberately:
the two machines are meant to be readable side by side.

Abduction uses a single GLOBAL axis (1, 0, 0) on all four legs rather than
mirroring the axis per side. That is Unitree's convention and hound's, so joint
signs here match anything written for either. The consequence is worth stating
plainly because it surprises people and check.py tests it:

    a symmetric ACTION vector is not a symmetric POSE.

Positive abduction swings the LEFT legs out and the RIGHT legs in. Mirror
symmetry of the machine still holds exactly -- reflecting the robot in the XZ
plane maps left legs onto right legs under

    (abduct, hip, knee, wheel)  ->  (-abduct, hip, knee, wheel)

which is what `mirror_pose()` implements and what check.py asserts against the
generated URDF. The hip/knee signs are unchanged under that reflection because
a reflection in the plane normal to Y leaves a rotation about Y alone
(M R_y M = R_y for M = diag(1, -1, 1)), and flips one about X.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from bestiary.robots.whelp.spec import SPEC, Spec

#: Front-left, front-right, rear-left, rear-right. Same order as hound, which is
#: also the order the action vector is laid out in.
LEGS: tuple[str, ...] = ("FL", "FR", "RL", "RR")

#: Per leg: (+1 for a front leg, +1 for a left leg).
SIGNS: dict[str, tuple[int, int]] = {
    "FL": (+1, +1),
    "FR": (+1, -1),
    "RL": (-1, +1),
    "RR": (-1, -1),
}

#: The four joints of a leg, outboard order. This tuple IS the action layout.
JOINTS: tuple[str, ...] = ("abduct", "hip", "knee", "wheel")

X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)


def _m(mm: float) -> float:
    """Millimetres to metres. The ONLY place this conversion happens."""
    return mm / 1000.0


# ── Chain description ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Link:
    """One rigid body. Masses and inertias are filled in by massmodel.py."""

    name: str
    #: Which printed/bought parts make this body up. A link is not a part: the
    #: thigh link is the printed thigh PLUS the knee servo bolted to it, and
    #: forgetting the servo is how a leg link ends up at a third of its mass.
    parts: tuple[str, ...]
    #: True for the body the floating base attaches to.
    is_root: bool = False


@dataclass(frozen=True)
class Joint:
    """One actuated degree of freedom, in URDF terms. Metres and radians."""

    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    #: (lower, upper) in radians, or None for a continuous joint. None is not a
    #: default: it is the wheel, and it is the reason half of check.py exists.
    limit: tuple[float, float] | None
    effort_nm: float
    velocity_rad_s: float
    damping: float
    friction: float
    #: The stance value of this joint, radians. Continuous joints have 0.
    stance: float = 0.0
    leg: str = ""
    kind: str = ""

    @property
    def continuous(self) -> bool:
        return self.limit is None


@dataclass
class Chain:
    links: list[Link] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)

    def joint(self, name: str) -> Joint:
        for j in self.joints:
            if j.name == name:
                return j
        raise KeyError(f"no joint {name!r}; have {[j.name for j in self.joints]}")

    def leg_joints(self, leg: str) -> list[Joint]:
        return [self.joint(f"{leg}_{k}") for k in JOINTS]

    @property
    def dof(self) -> int:
        return len(self.joints)


# ── The stance solve ─────────────────────────────────────────────────────────
def solve_stance_knee(spec: Spec = SPEC) -> float:
    """Knee angle putting the wheel axle directly under the hip pivot.

    Forward kinematics in the leg's sagittal plane, both pitch joints about +Y.
    With a rotation of theta about +Y mapping (0, 0, -L) to
    (-L sin(theta), 0, -L cos(theta)), the axle's fore-aft offset from the hip
    pivot is

        x_axle = -L_thigh sin(hip) - L_calf sin(hip + knee)

    and requiring it to vanish gives

        L_thigh sin(hip) = -L_calf sin(hip + knee)
        knee = -asin(L_thigh sin(hip) / L_calf) - hip.

    WHY IT MUST BE ZERO, not merely small. Three separate reasons, and the
    third is the one that decides the servo choice:

      1. A contact patch offset from the hip pivot pushes the machine along the
         ground the moment it takes weight. That reads as a robot that drifts
         while standing still, and a policy will spend capacity cancelling it.

      2. On a wheeled machine the offset also steers: the vertical load at an
         offset contact is a constant fore-aft force on a free-spinning wheel,
         so the robot rolls away rather than merely leaning.

      3. It makes the HIP servo's static holding torque zero. The hip's moment
         arm about its own axis is exactly x_axle, so solving x_axle = 0 does
         not reduce the hip's stance torque, it removes it. On a 2.5 kg machine
         held up by hobby servos, an actuator that has to hold nothing all day
         is an actuator that does not cook, does not creep, and does not eat its
         torque budget before the policy asks for anything.

    The assert is the geometric statement "you cannot fold a long link back
    under a short one": if the thigh throws the knee further out than the calf
    can reach back, no knee angle brings the axle home.
    """
    reach = spec.thigh_len_mm * math.sin(spec.stance_hip_rad)
    if abs(reach) > spec.calf_len_mm:
        raise ValueError(
            f"stance_hip_rad={spec.stance_hip_rad:.4f} throws the knee {reach:.1f} mm out, "
            f"which the {spec.calf_len_mm:.1f} mm calf cannot pull back under the hip. "
            f"Reduce stance_hip_rad below {math.asin(spec.calf_len_mm / spec.thigh_len_mm):.4f} rad "
            f"or lengthen the calf."
        )
    knee = -math.asin(reach / spec.calf_len_mm) - spec.stance_hip_rad
    lo, hi = spec.knee_range_rad
    if not lo <= knee <= hi:
        raise ValueError(
            f"solved stance knee {knee:.4f} rad ({math.degrees(knee):.1f} deg) is outside the "
            f"knee range {spec.knee_range_rad}. The stance is geometrically fine but the joint "
            f"cannot reach it -- widen knee_range_rad or change stance_hip_rad."
        )
    return knee


def axle_drop_mm(spec: Spec = SPEC) -> float:
    """Hip pivot to wheel axle, vertically, in the standing stance."""
    knee = solve_stance_knee(spec)
    return (spec.thigh_len_mm * math.cos(spec.stance_hip_rad)
            + spec.calf_len_mm * math.cos(spec.stance_hip_rad + knee))


def knee_lever_mm(spec: Spec = SPEC) -> float:
    """Horizontal distance from the knee axis to the wheel axle, at stance.

    This IS the knee servo's moment arm against the ground reaction force, and
    because the stance solve puts the axle under the hip, it is also just the
    fore-aft displacement of the knee itself:

        |x_knee - x_axle| = |x_knee - 0| = L_thigh sin(hip).

    Every newton the leg carries is multiplied by this number before the knee
    servo sees it, so it is the single geometric term the whole actuator budget
    turns on. Standing taller shrinks it; crouching grows it.
    """
    return abs(spec.thigh_len_mm * math.sin(spec.stance_hip_rad))


def stand_height_mm(spec: Spec = SPEC) -> float:
    """Trunk-origin height with the wheels resting on z = 0.

    The abduction axes sit at the trunk origin's z, so this is the axle drop
    plus the wheel's LOADED radius -- not its free radius. A tire that carries
    load is a tire that is squashed, and building the URDF at the free radius
    spawns the robot a few millimetres into the floor, which PhysX resolves by
    launching it. The sag comes from the tire stiffness in spec.py.
    """
    return axle_drop_mm(spec) + spec.wheel_radius_mm - spec.tire_static_sag_mm


def mirror_pose(q: dict[str, float]) -> dict[str, float]:
    """Reflect a per-joint pose through the robot's XZ plane.

    Left and right legs swap, abduction negates, hip/knee/wheel are unchanged.
    See the module docstring for why the pitch joints do not flip sign.
    """
    swap = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
    out: dict[str, float] = {}
    for name, value in q.items():
        leg, kind = name.split("_", 1)
        out[f"{swap[leg]}_{kind}"] = -value if kind == "abduct" else value
    return out


# ── Frames ───────────────────────────────────────────────────────────────────
def abduct_axis_y_mm(spec: Spec = SPEC) -> float:
    """Lateral position of the abduction axis, from the trunk centreline.

    Two different quantities are easy to confuse here, and confusing them was a
    real bug in an earlier draft:

        abduct_y_mm                    WHERE the pivot is, relative to the trunk.
                                       Sets the TRACK. Constrained by the trunk's
                                       side wall and by the servo fitting inside.

        abduct_axis_to_wheel_plane_mm  HOW FAR the leg steps outboard of its own
                                       pivot. Sets the abduction servo's MOMENT
                                       ARM, and is completely independent of the
                                       above -- moving the whole leg outboard
                                       widens the track and changes no torque,
                                       because the wheel moves with it.

    The bug was defining the second as the first, which put all four abduction
    axes on the trunk centreline. Every torque in the report was still right,
    which is why it survived: the arm was unchanged. What was wrong was the
    track (102 mm under a 104 mm trunk) and the fact that the mass model, which
    places the abduction servos against the trunk's side walls, no longer agreed
    with the kinematics about where the legs were. check.py section 3 now asserts
    those two agree, which is the assertion that would have caught it.
    """
    return spec.abduct_y_mm


def build_chain(spec: Spec = SPEC) -> Chain:
    """The whole robot, as links and joints, in metres and radians."""
    knee = solve_stance_knee(spec)
    chain = Chain()
    chain.links.append(Link("trunk", spec.trunk_parts, is_root=True))

    y_ab = abduct_axis_y_mm(spec)

    for leg in LEGS:
        fx, fy = SIGNS[leg]

        # ── abduct: trunk -> hip carrier ─────────────────────────────────────
        # Origin is the abduction pivot in the trunk frame. The axis is the
        # GLOBAL +X on all four legs (see the module docstring).
        chain.links.append(Link(f"{leg}_hip", (f"{leg}_abduct_bracket", "sts3215")))
        chain.joints.append(Joint(
            name=f"{leg}_abduct",
            parent="trunk", child=f"{leg}_hip",
            origin_xyz=(_m(fx * spec.abduct_x_mm), _m(fy * y_ab), 0.0),
            origin_rpy=(0.0, 0.0, 0.0),
            axis=X_AXIS,
            limit=spec.abduct_range_rad,
            effort_nm=spec.leg_servo_stall_nm,
            velocity_rad_s=spec.leg_servo_no_load_rad_s,
            damping=spec.joint_damping,
            friction=spec.joint_friction_nm,
            stance=spec.stance_abduct_rad,
            leg=leg, kind="abduct",
        ))

        # ── hip: hip carrier -> thigh ────────────────────────────────────────
        # Steps outboard to the leg's own sagittal plane. The sign follows the
        # side, so the two sides are mirror images even though the abduction
        # AXIS is shared.
        chain.links.append(Link(f"{leg}_thigh", (f"{leg}_thigh", "sts3215")))
        chain.joints.append(Joint(
            name=f"{leg}_hip",
            parent=f"{leg}_hip", child=f"{leg}_thigh",
            origin_xyz=(0.0, _m(fy * spec.abduct_to_hip_mm), _m(-spec.abduct_to_hip_drop_mm)),
            origin_rpy=(0.0, 0.0, 0.0),
            axis=Y_AXIS,
            limit=spec.hip_range_rad,
            effort_nm=spec.leg_servo_stall_nm,
            velocity_rad_s=spec.leg_servo_no_load_rad_s,
            damping=spec.joint_damping,
            friction=spec.joint_friction_nm,
            stance=spec.stance_hip_rad,
            leg=leg, kind="hip",
        ))

        # ── knee: thigh -> calf ──────────────────────────────────────────────
        # Straight down the thigh. Zero lateral step: the leg is planar from
        # here out, which is what makes the sagittal IK above exact rather than
        # approximate.
        chain.links.append(Link(f"{leg}_calf", (f"{leg}_calf", spec.wheel_drive_part)))
        chain.joints.append(Joint(
            name=f"{leg}_knee",
            parent=f"{leg}_thigh", child=f"{leg}_calf",
            origin_xyz=(0.0, 0.0, _m(-spec.thigh_len_mm)),
            origin_rpy=(0.0, 0.0, 0.0),
            axis=Y_AXIS,
            limit=spec.knee_range_rad,
            effort_nm=spec.leg_servo_stall_nm,
            velocity_rad_s=spec.leg_servo_no_load_rad_s,
            damping=spec.joint_damping,
            friction=spec.joint_friction_nm,
            stance=knee,
            leg=leg, kind="knee",
        ))

        # ── wheel: calf -> wheel ─────────────────────────────────────────────
        # The one joint with no limit, no stance and no spring. Its angle
        # integrates without bound, so it is not a state a policy can be given:
        # only its velocity is. Everything downstream that treats a joint as
        # "an angle inside a range" has to special-case it, which is why
        # check.py tests the wheel separately at every level.
        #
        # The lateral step puts the wheel's centreplane where the abduction
        # moment arm was budgeted for it.
        chain.links.append(Link(f"{leg}_wheel", (f"{leg}_wheel_hub", f"{leg}_tire")))
        chain.joints.append(Joint(
            name=f"{leg}_wheel",
            parent=f"{leg}_calf", child=f"{leg}_wheel",
            origin_xyz=(0.0, _m(fy * spec.calf_to_wheel_plane_mm), _m(-spec.calf_len_mm)),
            origin_rpy=(0.0, 0.0, 0.0),
            axis=Y_AXIS,
            limit=None,
            effort_nm=spec.wheel_drive_stall_nm,
            velocity_rad_s=spec.wheel_drive_no_load_rad_s,
            damping=spec.wheel_damping,
            friction=spec.wheel_friction_nm,
            stance=0.0,
            leg=leg, kind="wheel",
        ))

    return chain


# ── Forward kinematics, for the checks and the report ────────────────────────
def _rot_x(t: float) -> list[list[float]]:
    c, s = math.cos(t), math.sin(t)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(t: float) -> list[list[float]]:
    c, s = math.cos(t), math.sin(t)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _apply(r, v):
    return tuple(sum(r[i][k] * v[k] for k in range(3)) for i in range(3))


Frame = tuple[list[list[float]], tuple[float, float, float]]


def link_frames(chain: Chain, q: dict[str, float] | None = None,
                ) -> tuple[dict[str, tuple[float, float, float]], dict[str, Frame]]:
    """Walk the chain once. Returns (joint origins, link frames), trunk frame, metres.

    q defaults to the standing stance. Written out longhand rather than pulled
    from a robotics library because it is thirty lines, adds no dependency, and
    because check.py must be able to test the generated URDF against something
    that did not come out of the same code path that produced it.

    A link's frame is (rotation, origin) with its origin AT ITS PARENT JOINT --
    the URDF convention, where a link's geometry and inertia are expressed in
    the frame the joint that drives it establishes.
    """
    if q is None:
        q = {j.name: j.stance for j in chain.joints}

    ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    joint_pos: dict[str, tuple[float, float, float]] = {}
    frames: dict[str, Frame] = {"trunk": (ident, (0.0, 0.0, 0.0))}

    for j in chain.joints:
        rp, pp = frames[j.parent]
        off = _apply(rp, j.origin_xyz)
        origin = (pp[0] + off[0], pp[1] + off[1], pp[2] + off[2])
        joint_pos[j.name] = origin
        theta = q.get(j.name, 0.0)
        local = _rot_x(theta) if j.axis == X_AXIS else _rot_y(theta)
        frames[j.child] = (_mul(rp, local), origin)
    return joint_pos, frames


def forward_kinematics(chain: Chain, q: dict[str, float] | None = None,
                       ) -> dict[str, tuple[float, float, float]]:
    """Origin of every joint AND every link in the trunk frame, metres, for pose q."""
    joint_pos, frames = link_frames(chain, q)
    out = dict(joint_pos)
    for name, (_r, p) in frames.items():
        out.setdefault(name, p)
    return out


def joint_axis_world(chain: Chain, name: str,
                     q: dict[str, float] | None = None) -> tuple[float, float, float]:
    """A joint's rotation axis in the trunk frame, for pose q.

    Needed by torque.py: a joint's holding torque is the component of the
    applied moment along its OWN axis, and after abduction that axis is no
    longer the one written in the URDF.
    """
    _joints, frames = link_frames(chain, q)
    j = chain.joint(name)
    r, _p = frames[j.parent]
    return _apply(r, j.axis)


def distal_links(chain: Chain, joint_name: str) -> list[str]:
    """Every link outboard of a joint, including the joint's own child.

    These are exactly the bodies whose weight the joint has to hold, which is
    the whole content of a static free-body diagram. Computed by walking the
    parent pointers rather than assuming the leg is a straight chain, so adding
    a branch later does not silently produce a torque budget that ignores it.
    """
    child_of = {j.parent: [] for j in chain.joints}
    for j in chain.joints:
        child_of.setdefault(j.parent, []).append(j.child)
    root = chain.joint(joint_name).child
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(child_of.get(n, []))
    return out


def contact_points(chain: Chain, spec: Spec = SPEC,
                   q: dict[str, float] | None = None) -> dict[str, tuple[float, float, float]]:
    """Where each wheel touches the ground, trunk frame, metres."""
    pos = forward_kinematics(chain, q)
    r = _m(spec.wheel_radius_mm - spec.tire_static_sag_mm)
    return {leg: (pos[f"{leg}_wheel"][0], pos[f"{leg}_wheel"][1], pos[f"{leg}_wheel"][2] - r)
            for leg in LEGS}


def main() -> int:
    spec = SPEC
    chain = build_chain(spec)
    knee = solve_stance_knee(spec)
    deg = math.degrees

    print("WHELP-16 kinematics")
    print(f"  {chain.dof} joints, {len(chain.links)} links")
    print()
    print("  STANCE (abduct and hip are inputs; knee is SOLVED)")
    print(f"    abduct  {spec.stance_abduct_rad:+.4f} rad ({deg(spec.stance_abduct_rad):+6.1f} deg)")
    print(f"    hip     {spec.stance_hip_rad:+.4f} rad ({deg(spec.stance_hip_rad):+6.1f} deg)")
    print(f"    knee    {knee:+.4f} rad ({deg(knee):+6.1f} deg)   SOLVED: axle under hip")
    print("    wheel   continuous, no stance")
    print()
    print(f"    axle drop below hip pivot   {axle_drop_mm(spec):7.2f} mm")
    print(f"    knee lever arm at stance    {knee_lever_mm(spec):7.2f} mm  <- sizes the knee servo")
    print(f"    trunk origin above ground   {stand_height_mm(spec):7.2f} mm")
    print(f"    abduction axis, from centre {abduct_axis_y_mm(spec):7.2f} mm")
    print(f"    abduction moment arm        {spec.abduct_axis_to_wheel_plane_mm:7.2f} mm"
          "  <- sizes the abduct servo")
    print()

    pos = forward_kinematics(chain)
    contacts = contact_points(chain, spec)
    print("  STANDING POSE, trunk frame, metres")
    print(f"    {'joint':<12} {'x':>9} {'y':>9} {'z':>9}")
    for leg in LEGS:
        for kind in JOINTS:
            p = pos[f"{leg}_{kind}"]
            print(f"    {leg + '_' + kind:<12} {p[0]:>9.4f} {p[1]:>9.4f} {p[2]:>9.4f}")
        c = contacts[leg]
        print(f"    {leg + ' contact':<12} {c[0]:>9.4f} {c[1]:>9.4f} {c[2]:>9.4f}")
        print()

    xs = [c[0] for c in contacts.values()]
    ys = [c[1] for c in contacts.values()]
    print(f"    wheelbase {max(xs) - min(xs):.4f} m,  track {max(ys) - min(ys):.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
