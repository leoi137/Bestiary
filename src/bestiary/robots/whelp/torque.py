"""What every joint of WHELP-16 has to hold, and what breaks when it cannot.

    python -m bestiary.robots.whelp.torque          # the report
    python -m bestiary.robots.whelp.torque --json   # machine-readable, for check.py

THE QUESTION THIS FILE ANSWERS
------------------------------
Not "is the servo big enough to stand up" -- it obviously is, a 2.5 kg machine
on four legs puts about 6 N through each one. The question is the other one:

    when this robot lands, or gets pushed, or falls over,
    WHAT GIVES FIRST -- and is that thing cheap and reversible?

A design where the servo backs off is a design that gets picked up and carried
on. A design where the calf snaps, or the output gear strips, or the horn shears,
is a design that ends the afternoon. Both look identical in simulation. The
difference is entirely in the ordering of failure thresholds, and that ordering
is something you choose at design time or discover at 3 pm on a Saturday.

So the report has two halves. The first is the ordinary static budget the brief
asked for: holding torque per joint at stance against the servo's rating, with a
2x margin required. It passes comfortably, and it is the least interesting result
here. The second is THE YIELD CHAINS -- every failure threshold in the leg,
sorted, in newtons at the contact patch.

WHY STATIC MARGIN IS NOT THE BINDING CONSTRAINT
-----------------------------------------------
Static stance is a fraction of the servo's rating. Then the machine lands from a
small drop and the same joint sees several times that, because impact force is
not set by weight -- it is set by how much STROKE the landing is absorbed over:

    linear absorber, drop height h, stroke s:   F_peak = 2 m g (h + s) / s

That ratio has no mass in it, which is the unintuitive part: a LIGHTER ROBOT
LANDS JUST AS HARD. Only stroke helps.

THE FINDING THAT CHANGED THE DESIGN
-----------------------------------
The obvious way to buy stroke is to let the joint fold: the servo has a Torque
Limit register, so cap it and the knee gives way instead of the leg breaking. An
earlier version of this file was built on exactly that, and it is WRONG for the
case that matters.

A 1:345 reduction reflects rotor inertia to the joint by the SQUARE of the ratio
-- about 0.027 kg.m^2, roughly a hundred times the thigh's own. Backdriving that
through a useful rotation takes on the order of a hundred milliseconds
(Spec.impact_rigid_below_ms). A landing's contact lasts tens. On that timescale
the gear train is a solid block no matter what any register says.

So the torque limit protects against LEANING and PUSHING and nothing else, the
tire's squash is the only mechanical stroke the leg has, and the drop envelope is
a hard limit rather than advice. Three separate yield chains follow from that,
because a leg loaded slowly, struck from below, and struck from the side fail in
three different orders -- and only two of the three have a reversible first item.
See `yield_chain()` and `lateral_chain()`.

WHAT THIS FILE IS NOT
---------------------
It is not FEA and it does not pretend to be. Structural thresholds here are
first-order beam and bearing calculations on idealised sections, and several of
their inputs are `Kind.ASSUMED` in spec.py. Their job is to establish the
ORDERING of failures with a wide enough gap that being 40% wrong about any one of
them does not reorder the chain. Where the gap is not wide, the report says so,
and spec.py names the test that would replace the estimate.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np

from bestiary.robots.whelp import geometry as geo
from bestiary.robots.whelp.massmodel import Body, link_bodies
from bestiary.robots.whelp.spec import SPEC, Spec

G = 9.80665  # m/s^2, standard gravity (BIPM SI brochure). Not 9.81: the third
             # digit costs nothing and the habit is worth more than the digit.


# ── The general free-body solver ─────────────────────────────────────────────
def holding_torques(
    chain: geo.Chain,
    bodies: dict[str, Body],
    contact_forces: dict[str, tuple[float, float, float]],
    q: dict[str, float] | None = None,
) -> dict[str, float]:
    """Actuator torque required to hold each joint, N.m, for one loading case.

    One routine for all sixteen joints rather than a hand-derived formula per
    joint. Hand-derived formulas are where sign errors live, and they have to be
    re-derived every time the geometry changes -- which, in a parametric design,
    is every time anyone edits Spec.

    Static equilibrium of everything OUTBOARD of joint j, taking moments about
    j's axis:

        tau_actuator = - sum_over_distal( (p - o_j) x F ) . a_j

    where F ranges over each distal link's weight at its own centre of mass and
    over the ground reaction at any contact distal to j. `a_j` is the joint axis
    IN THE WORLD, which after abduction is not the axis written in the URDF --
    getting that wrong is the classic error and it hides at zero abduction,
    which is exactly the pose everyone tests.

    Returned values are magnitudes-with-sign: positive means the actuator must
    push in its own positive direction. Sizing uses the absolute value; the sign
    is kept because it tells you which way a joint sags when the servo gives up.
    """
    joint_pos, frames = geo.link_frames(chain, q)
    out: dict[str, float] = {}

    for j in chain.joints:
        o = joint_pos[j.name]
        r_parent, _ = frames[j.parent]
        axis = geo._apply(r_parent, j.axis)

        mx = my = mz = 0.0
        for link in geo.distal_links(chain, j.name):
            body = bodies.get(link)
            if body is not None and body.mass_kg > 0:
                r_link, p_link = frames[link]
                c = geo._apply(r_link, body.com_m)
                p = (p_link[0] + c[0], p_link[1] + c[1], p_link[2] + c[2])
                f = (0.0, 0.0, -body.mass_kg * G)
                d = (p[0] - o[0], p[1] - o[1], p[2] - o[2])
                mx += d[1] * f[2] - d[2] * f[1]
                my += d[2] * f[0] - d[0] * f[2]
                mz += d[0] * f[1] - d[1] * f[0]

            if link in contact_forces:                     # link is "<LEG>_wheel"
                f = contact_forces[link]
                _r_link, p_link = frames[link]
                # Contact is directly below the axle by the loaded radius, in the
                # WORLD frame rather than the link's -- the patch is where the
                # ground is, and the ground does not rotate with the leg. The
                # loaded radius, not the free one: a squashed tire has a shorter
                # moment arm, and using the free radius overstates every torque.
                p = (p_link[0], p_link[1], p_link[2] - _loaded_radius_m(SPEC))
                d = (p[0] - o[0], p[1] - o[1], p[2] - o[2])
                mx += d[1] * f[2] - d[2] * f[1]
                my += d[2] * f[0] - d[0] * f[2]
                mz += d[0] * f[1] - d[1] * f[0]

        out[j.name] = -(mx * axis[0] + my * axis[1] + mz * axis[2])
    return out


def _loaded_radius_m(spec: Spec) -> float:
    return (spec.wheel_radius_mm - spec.tire_static_sag_mm) / 1000.0


def _total_mass(bodies: dict[str, Body]) -> float:
    return sum(b.mass_kg for b in bodies.values())


# ── Loading cases ────────────────────────────────────────────────────────────
@dataclass
class Case:
    """One loading case, and the answer for it."""

    name: str
    why: str
    #: Multiple of the machine's own weight that reaches the ground.
    load_factor: float
    #: Legs sharing that load.
    contacts: tuple[str, ...]
    #: WHICH RATING THIS CASE IS JUDGED AGAINST. Three, not two, because three
    #: physically different things are being asked, and using one number for all
    #: of them is a 5x error at the extremes:
    #:
    #:   "rated"  0.98 N.m   the CONTINUOUS rating. For poses the robot HOLDS,
    #:                       seconds to minutes. It is a thermal limit: exceed it
    #:                       and the servo cooks, slowly, while looking fine.
    #:   "stall"  2.94 N.m   the momentary peak. For dynamic gait loads lasting a
    #:                       fraction of a second -- the motor can produce this,
    #:                       it just cannot keep producing it.
    #:   "gear"   5.30 N.m   the gear train's estimated failure torque. For
    #:                       IMPACTS. Exceeding stall during a landing is not a
    #:                       failure: the servo simply cannot hold, and since the
    #:                       reflected rotor inertia stops it yielding anyway, the
    #:                       load goes into the gear train regardless. What
    #:                       matters is whether the TEETH survive, so that is what
    #:                       an impact is measured against.
    #:
    #: Judging a landing against stall (an earlier version of this file) demands
    #: the motor hold a load it is not being asked to hold, and shrinks the drop
    #: envelope by roughly 3x for no physical reason.
    rating: str
    torques: dict[str, float] = field(default_factory=dict)

    def worst(self, kind: str) -> tuple[str, float]:
        """(joint name, |torque|) for the worst joint of a given kind."""
        cands = {k: abs(v) for k, v in self.torques.items() if k.endswith("_" + kind)}
        name = max(cands, key=cands.get)
        return name, cands[name]

    @property
    def sustained(self) -> bool:
        return self.rating == "rated"

    def rating_nm(self, spec: Spec) -> float:
        return {
            "rated": spec.leg_servo_rated_nm,
            "stall": spec.leg_servo_stall_nm,
            "gear": spec.leg_servo_stall_nm * spec.servo_gear_break_multiple,
        }[self.rating]


def ground_reactions(chain, bodies, spec, contacts, total_load_n: float) -> dict[str, float]:
    """Vertical reaction at each contact, N. Solved, not divided by the count.

    THE ERROR THIS REPLACES, because it was wrong by 50% on the case that binds:
    an earlier version set every contact to W/n. For four contacts on a symmetric
    rectangle that is right. For THREE it is not right and it is not even
    admissible -- a rigid body on three point contacts is statically determinate,
    so there is no freedom to share the load, and W/3 each leaves a net roll
    moment of about 0.66 N.m on a trunk that is supposed to be in equilibrium.

    The true answer for a rectangular footprint with one corner lifted is the
    classic one, and it is worse than the average: the diagonal opposite the
    lifted leg takes W/2 EACH and the remaining leg takes nothing. That is 1.5x
    what equal-thirds assumed, and it is independent of where the centre of mass
    sits -- moving the battery does not help.

    Solved as the minimum-norm solution of the three equilibrium equations

        sum(f_i) = W,   sum(x_i f_i) = W x_com,   sum(y_i f_i) = W y_com

    which is exact for three contacts, reduces to equal shares for a symmetric
    four-contact stance, and stays well defined for two. Reactions are asserted
    non-negative: a negative one means the pose needs a leg to PULL on the
    ground, which is a real result about an unstable stance rather than a number
    to pass on quietly.
    """
    joint_pos, frames = geo.link_frames(chain)
    pts = geo.contact_points(chain, spec)

    m_tot = 0.0
    cx = cy = 0.0
    for name, body in bodies.items():
        r, p = frames[name]
        c = geo._apply(r, body.com_m)
        m_tot += body.mass_kg
        cx += body.mass_kg * (p[0] + c[0])
        cy += body.mass_kg * (p[1] + c[1])
    cx /= m_tot
    cy /= m_tot

    legs = list(contacts)

    if len(legs) == 2:
        # TWO CONTACTS ARE NOT A STATIC STANCE, and pretending otherwise is how a
        # trot case gets silently mis-loaded. A rigid body on two point contacts
        # can only balance the moment ALONG the line joining them; the moment
        # across that line is what tips it over, and no distribution of vertical
        # reactions can resist it.
        #
        # The physical statement is that to trot, the machine must put its centre
        # of mass over the diagonal -- which real trotting robots do, by leaning.
        # So the CoM is projected onto the support line and the reactions solved
        # exactly along it. That is the correct idealisation of a balanced trot,
        # and it is stated rather than hidden because the alternative (a
        # least-squares fudge) would quietly return a load set that is not in
        # equilibrium at all.
        p1, p2 = pts[legs[0]], pts[legs[1]]
        u = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        span = float(np.linalg.norm(u))
        u = u / span
        s_com = float(np.dot(np.array([cx - p1[0], cy - p1[1]]), u))
        f2 = total_load_n * s_com / span
        f1 = total_load_n - f2
        f = np.array([f1, f2])
    else:
        a = np.array([[1.0] * len(legs),
                      [pts[leg][0] for leg in legs],
                      [pts[leg][1] for leg in legs]])
        b = np.array([total_load_n, total_load_n * cx, total_load_n * cy])
        f, *_ = np.linalg.lstsq(a, b, rcond=None)

        resid = float(np.abs(a @ f - b).max())
        if resid > 1e-6 * max(1.0, total_load_n):
            raise ValueError(
                f"no equilibrium exists for contacts {legs}: residual {resid:.4g} N. "
                f"The centre of mass at ({cx * 1000:.1f}, {cy * 1000:.1f}) mm is outside "
                f"the support polygon, so this pose falls over rather than standing."
            )

    if float(f.min()) < -1e-9:
        raise ValueError(
            f"contact {legs[int(np.argmin(f))]} needs a NEGATIVE reaction "
            f"({f.min():.3f} N) -- the ground would have to pull. This stance is not "
            f"statically supportable."
        )
    return {leg: float(v) for leg, v in zip(legs, f)}


def _case(chain, bodies, spec, name, why, load_factor, contacts,
          rating: str, fx_per_leg: float = 0.0) -> Case:
    total = _total_mass(bodies) * G * load_factor
    fz = ground_reactions(chain, bodies, spec, contacts, total)
    forces = {f"{leg}_wheel": (fx_per_leg, 0.0, fz[leg]) for leg in contacts}
    c = Case(name, why, load_factor, contacts, rating)
    c.torques = holding_torques(chain, bodies, forces)
    # Joints on a leg that is not in contact still hold their own leg's weight,
    # which the solver already handles: they simply get no contact force.
    return c


def landing_load_factor(spec: Spec, stroke_m: float, drop_m: float | None = None) -> float:
    """Peak vertical load, as a multiple of static weight, for a drop landing.

    Energy balance for a linear absorber. The machine falls `h`, then the
    absorber compresses `s` while the machine keeps descending, so the work done
    on the absorber is m g (h + s) and, for a spring, equals half the peak force
    times the stroke:

        (1/2) F_peak s = m g (h + s)   ->   F_peak / (m g) = 2 (h + s) / s

    The factor of two is the linear-spring assumption and it is deliberately the
    pessimistic end. A real TPU tire stiffens as it compresses, which stores more
    energy for a given peak force and lands between this and the constant-force
    bound of (h + s) / s. Using the linear bound means the report understates the
    margin rather than overstating it, which is the right way round for a number
    that decides whether a leg survives.

    NOTE the shape of this: the load factor depends on the RATIO of drop to
    stroke, not on mass at all. A heavier robot lands harder in newtons but at
    the same load factor, so making the machine lighter does NOT protect it --
    only more stroke does. That is unintuitive and it is why the fix here is a
    yielding joint rather than a diet.
    """
    h = spec.design_drop_height_mm / 1000.0 if drop_m is None else drop_m
    if stroke_m <= 0:
        raise ValueError("stroke must be positive; a rigid landing has unbounded force")
    return 2.0 * (h + stroke_m) / stroke_m


def yield_stroke_m(chain: geo.Chain, spec: Spec) -> float:
    """Vertical travel available if the knee is allowed to fold, metres.

    The knee's stance lever arm is the horizontal distance from the knee axis to
    the contact patch, so rotating the knee by `dtheta` lowers the hip by
    approximately `lever * dtheta` for small rotations. `yield_knee_rad` is how
    much rotation the design is willing to spend before something else has to
    stop it -- limited by the knee's own range, and by the leg not folding so far
    that the trunk lands on the ground.

    Added to the tire's own stroke, this is the total absorber the landing has.
    """
    lever = geo.knee_lever_mm(spec) / 1000.0
    return spec.tire_stroke_mm / 1000.0 + lever * spec.yield_knee_rad


# ── The yield chain ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Threshold:
    """One way the leg can stop resisting, and the wheel-contact force that does it."""

    name: str
    #: Force at the WHEEL CONTACT PATCH that reaches this threshold, newtons.
    #:
    #: Everything is referred to the contact patch rather than to a joint torque,
    #: because the thresholds are not all torques: the sacrificial fuse fails in
    #: DIRECT SHEAR and has no moment arm at all. Expressing a shear web as an
    #: equivalent knee torque -- which an earlier draft of this file did -- makes
    #: it look 30x stronger than it is, and the fuse silently stops being a fuse.
    #: One common unit at one common place is what makes the chain comparable.
    wheel_force_n: float
    reversible: bool
    basis: str
    #: How confident the number is. Mirrors provenance.Kind for the inputs that
    #: dominate it, so the report can say which thresholds are estimates.
    confidence: str
    #: Whether this threshold is available on IMPACT timescales. The servo's own
    #: yielding is not: see `impact_rigid_below_ms` and the note in yield_chain.
    on_impact: bool = True


def yield_chain(spec: Spec, impact: bool = False) -> list[Threshold]:
    """Every failure threshold in a leg, expressed as an equivalent knee torque.

    Sorted lowest-first by the caller. The design requirement is that the lowest
    entry is `reversible=True`. That one ordering separates a robot that survives
    its own mistakes from one that does not, and it is worth more than making any
    individual part stronger: strengthening a part that was never first to fail
    changes nothing at all.

    Everything is referred to the KNEE, because the knee has the longest lever
    arm against a vertical load and therefore reaches any given threshold first.
    A force that would break the calf is converted to the knee torque that
    produces it, so the comparison is like for like.

    THE TWO CHAINS ARE DIFFERENT, AND THAT IS THE POINT
    ---------------------------------------------------
    `impact=False` is the SLOW chain: leaning on the robot, a bad pose, a push,
    a foot caught on something. Here the servo's Torque Limit register is real
    protection -- exceed it and the joint simply stops resisting.

    `impact=True` is the LANDING chain, and the servo drops out of it entirely.
    A 1:345 reduction reflects rotor inertia to the joint by the square of the
    ratio: ~0.027 kg.m^2, about a hundred times the thigh's own. Backdriving
    that through the yield rotation takes on the order of a hundred milliseconds
    (Spec.impact_rigid_below_ms), while a landing's contact lasts tens. On that
    timescale the gear train is a solid block no matter what the register says,
    and every newton goes into plastic.

    This is why the tire's stroke and the sacrificial fuse are structural parts
    of this design rather than accessories, and it is the single correction that
    most changes what the robot should be built like.
    """
    lever = geo.knee_lever_mm(spec) / 1000.0        # knee axis -> contact, horizontally
    calf_m = spec.calf_len_mm / 1000.0
    out: list[Threshold] = []

    def from_knee_torque(nm: float) -> float:
        """A knee-torque threshold, as the vertical wheel force that reaches it."""
        return nm / lever

    if not impact:
        # A register write. Free, and reversible by definition: the joint stops
        # resisting, the leg folds, the servo is unharmed.
        out.append(Threshold(
            "servo torque-limit register",
            from_knee_torque(spec.leg_servo_stall_nm * spec.servo_torque_limit_frac),
            reversible=True,
            basis=f"{spec.servo_torque_limit_frac:.0%} of {spec.leg_servo_stall_nm:.2f} N.m "
                  f"stall = {spec.leg_servo_stall_nm * spec.servo_torque_limit_frac:.2f} N.m "
                  f"at the {lever * 1000:.0f} mm knee lever",
            confidence="derived from the servo rating",
            on_impact=False,
        ))
        # Stall. The servo cannot produce more; it heats, and its overload
        # protection eventually cuts torque. Reversible, but living here cooks
        # the motor, so it is a worse place to be than the register limit.
        out.append(Threshold(
            "servo stalls", from_knee_torque(spec.leg_servo_stall_nm), reversible=True,
            basis=f"{spec.leg_servo_stall_nm:.2f} N.m datasheet peak stall at 12 V",
            confidence="datasheet", on_impact=False,
        ))

    # The sacrificial fuse: a printed shear web between the calf and the wheel
    # mount. It breaks -- but it is a two-gram part and a two-minute swap, and it
    # is available on EVERY timescale, which after the note above is the property
    # that matters most.
    #
    # DIRECT SHEAR, no moment arm: the wheel force passes straight through the
    # web. This is why the whole chain is in newtons at the contact patch.
    if spec.fuse_enable:
        out.append(Threshold(
            "sacrificial fuse link shears",
            spec.fuse_shear_area_mm2 * spec.print_shear_strength_mpa,
            reversible=True,
            basis=f"{spec.fuse_shear_area_mm2:.0f} mm^2 in direct shear at "
                  f"{spec.print_shear_strength_mpa:.0f} MPa",
            confidence="ESTIMATE -- shear strength assumed; one vice-and-scale test replaces it",
        ))

    # Servo output gear train. The number nobody publishes, and the one that
    # decides whether an overload is reversible.
    out.append(Threshold(
        "servo output gear strips",
        from_knee_torque(spec.leg_servo_stall_nm * spec.servo_gear_break_multiple),
        reversible=False,
        basis=f"{spec.servo_gear_break_multiple:.1f}x stall = "
              f"{spec.leg_servo_stall_nm * spec.servo_gear_break_multiple:.2f} N.m",
        confidence="ASSUMED -- no manufacturer figure and no citable teardown found",
    ))

    # The horn interface. Torque leaves a small metal disc into printed plastic
    # through a ring of small screws, and the plastic bears on the screw shanks.
    # Bearing stress, not screw shear: the screws are steel and the plastic is
    # not, so the plastic yields first every time.
    bolt_force = (spec.horn_bolt_count * spec.horn_bolt_dia_mm * spec.horn_boss_thick_mm
                  * spec.print_bearing_strength_mpa)
    out.append(Threshold(
        "horn bolts crush their bosses",
        from_knee_torque(bolt_force * (spec.horn_bolt_circle_mm / 2.0) / 1000.0),
        reversible=False,
        basis=f"{spec.horn_bolt_count} bolts bearing on {spec.horn_boss_thick_mm:.1f} mm of "
              f"plastic at {spec.print_bearing_strength_mpa:.0f} MPa, on a "
              f"{spec.horn_bolt_circle_mm:.1f} mm bolt circle",
        confidence="ESTIMATE -- bearing strength and bolt circle both assumed",
    ))

    # M3 heat-set insert stripping out of its boss.
    out.append(Threshold(
        "M3 insert strips from its boss",
        from_knee_torque(spec.insert_torque_fail_nm * 2.0),
        reversible=False,
        basis=f"{spec.insert_torque_fail_nm:.1f} N.m per insert in PETG, two per joint face",
        confidence="measured (CNC Kitchen instrumented series, PETG)",
    ))

    # The calf in sagittal bending: a cantilever with the ground reaction at its
    # tip. sigma = M c / I on the idealised box section. The allowable differs
    # between the chains -- one bad landing is an ULTIMATE question, while slow
    # loading is about a part that has to survive 1e6 cycles at 42% of UTS.
    allow = spec.print_ultimate_stress_mpa if impact else spec.print_design_stress_mpa
    z_sag = _calf_section_modulus_m3(spec, "sagittal")
    out.append(Threshold(
        "calf breaks in sagittal bending", allow * 1e6 * z_sag / calf_m, reversible=False,
        basis=f"box section Z = {z_sag * 1e9:.0f} mm^3 over the {spec.calf_len_mm:.0f} mm "
              f"calf, at {allow:.1f} MPa "
              f"({'ultimate, one event' if impact else 'fatigue allowable at 1e6 cycles'})",
        confidence="first-order beam theory on an idealised section",
    ))

    return out


def lateral_chain(spec: Spec) -> list[Threshold]:
    """What breaks when the robot is hit SIDEWAYS, in newtons at the wheel.

    The case that actually destroys printed legs, and the one a vertical torque
    budget says nothing about. A robot tipping over from standing drops its
    centre of mass roughly 100 mm, which is a couple of joules arriving at a
    wheel over a few millimetres of local deformation -- hundreds of newtons,
    sideways, at the end of a 100 mm cantilever.

    It is also the direction in which NOTHING can yield. The servos rotate about
    Y and the load is about X and Z; the gearbox is simply not in the load path,
    so there is no register to set and no compliance to spend. Only the fuse and
    the tire's own squash stand between a side impact and a broken calf.

    The numbers here are estimates -- the stopping distance dominates and is not
    something first-order analysis pins down -- so what they establish is the
    ORDERING, not the values. `sensitivity()` shows how hard the ordering is.
    """
    calf_m = spec.calf_len_mm / 1000.0
    out: list[Threshold] = []

    if spec.fuse_enable:
        out.append(Threshold(
            "sacrificial fuse link shears",
            spec.fuse_shear_area_mm2 * spec.print_shear_strength_mpa,
            reversible=True,
            basis=f"{spec.fuse_shear_area_mm2:.0f} mm^2 in direct shear at "
                  f"{spec.print_shear_strength_mpa:.0f} MPa",
            confidence="ESTIMATE -- shear strength assumed",
        ))

    z_lat = _calf_section_modulus_m3(spec, "lateral")
    out.append(Threshold(
        "calf breaks in LATERAL bending",
        spec.print_ultimate_stress_mpa * 1e6 * z_lat / calf_m,
        reversible=False,
        basis=f"box section Z = {z_lat * 1e9:.0f} mm^3 about the lateral axis, over the "
              f"{spec.calf_len_mm:.0f} mm calf, at {spec.print_ultimate_stress_mpa:.1f} MPa",
        confidence="first-order beam theory; also the WEAK print direction",
    ))

    # The abduction servo, resisting a sideways push at the contact patch. Its
    # moment arm is the whole leg's height, so it is the most heavily levered
    # joint on the machine for this load case even though it is the least loaded
    # one standing still.
    height_m = (geo.axle_drop_mm(spec) + spec.wheel_radius_mm
                - spec.tire_static_sag_mm) / 1000.0
    out.append(Threshold(
        "abduction servo is back-driven",
        spec.leg_servo_stall_nm / height_m,
        reversible=True,
        basis=f"{spec.leg_servo_stall_nm:.2f} N.m stall at the {height_m * 1000:.0f} mm "
              f"lever from the abduction axis to the contact patch",
        confidence="datasheet stall; the servo turns rather than breaking",
    ))

    return sorted(out, key=lambda t: t.wheel_force_n)


def _calf_section_modulus_m3(spec: Spec, axis: str) -> float:
    """Section modulus of the calf's hollow-box cross-section, m^3.

    `axis` is "sagittal" (the fore-aft depth resists) or "lateral" (the width
    does). Both matter and they differ: the calf is deliberately deeper than it
    is wide, because the load it carries constantly is sagittal -- which means it
    is correspondingly WEAKER in the direction a side impact loads it. That
    asymmetry is a design choice and lateral_chain() is where its cost is paid.
    """
    d, b, t = (spec.calf_section_fore_aft_mm, spec.calf_section_lateral_mm, spec.calf_wall_mm)
    if axis == "lateral":
        d, b = b, d
    di, bi = max(d - 2 * t, 0.0), max(b - 2 * t, 0.0)
    i_mm4 = (b * d ** 3 - bi * di ** 3) / 12.0
    return (i_mm4 / (d / 2.0)) * 1e-9


# ── Wheel drive ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WheelBudget:
    traction_limit_nm: float
    slope_nm: float
    accel_nm: float
    wheelie_limit_nm: float
    rolling_nm: float
    accel_ground_nm: float
    #: Highest forward acceleration the machine can actually reach, m/s^2, and
    #: which of the three ceilings sets it.
    accel_max_m_s2: float
    binding: str
    top_speed_m_s: float
    wheel_rpm: float
    #: Apparent mass for longitudinal acceleration, including the reflected
    #: rotor inertia of all four drives. On this machine it is ~27x the real mass.
    effective_mass_kg: float


def wheel_budget(chain: geo.Chain, bodies: dict[str, Body], spec: Spec) -> WheelBudget:
    """What the wheel drive must produce, and what actually limits it.

    Four separate ceilings, and which one binds is the interesting part. hound's
    equivalent analysis found the answer was not the one it expected -- the
    friction cone was barely used and a WHEELIE was the real bound -- so the same
    four are computed here rather than assumed to rank the same way at a fifth of
    the mass and half the wheel radius.
    """
    m = _total_mass(bodies)
    r = _loaded_radius_m(spec)
    fz = m * G / 4.0

    traction = spec.ground_friction * fz * r
    slope = m * G * math.sin(math.radians(spec.design_slope_deg)) * r / 4.0
    rolling = spec.rolling_resistance * fz * r

    # ACCELERATION IS NOT m*a*r/4, AND THE DIFFERENCE IS A FACTOR OF 25.
    #
    # That expression is the torque needed to push the CHASSIS. The motor must
    # also spin up its own rotor, and at 1:345 the reflected rotor inertia is
    # 0.027 kg.m^2 -- so accelerating the machine at 1 m/s^2 costs 0.025 N.m of
    # thrust and 0.62 N.m of winding up the gearbox.
    #
    # The consequence is worth stating in its own right: for LONGITUDINAL
    # acceleration this robot behaves as though it weighed
    #
    #     m_eff = m + 4 I_reflected / r^2  =  2.2 + 57  =  59 kg
    #
    # It is a 2.2 kg machine with the acceleration of a 59 kg one. Nothing about
    # the chassis causes that; it is entirely the gearbox. A policy trained
    # against a simulator that models the wheels as ideal torque sources with no
    # armature will learn accelerations the hardware cannot produce, and this is
    # the single clearest reason the armature term belongs in the actuator config.
    # TWO DIFFERENT DEMANDS, and conflating them made the report wrong in both
    # directions at once. The rotor spin-up torque is produced INSIDE the motor
    # and never passes through the contact patch, so it must not be compared
    # against a friction or wheelie ceiling; and the chassis thrust alone must
    # not be compared against the motor's rating.
    i_wheel = float(bodies["FL_wheel"].inertia_kgm2[1][1])
    i_eff = spec.servo_armature_kgm2 + i_wheel
    accel_ground = m * spec.design_accel_m_s2 * r / 4.0        # through the tyre
    accel = accel_ground + i_eff * spec.design_accel_m_s2 / r  # out of the motor
    m_eff = m + 4.0 * spec.servo_armature_kgm2 / (r * r)

    # What the machine can actually do, from each side. Computed after the
    # wheelie ceiling exists, below.

    # Wheelie: all thrust acts at ground level while the mass sits h_cg up, so
    # driving pitches the machine back about the rear contacts. The rear pair
    # takes everything when the front pair unloads, which happens at
    #     F_total * h_cg = m g * (wheelbase / 2)
    contacts = geo.contact_points(chain, spec)
    xs = [c[0] for c in contacts.values()]
    wheelbase = max(xs) - min(xs)
    # CoM height is DERIVED from the mass model and the stance, not assumed. It
    # came out at 0.745 of trunk-origin height against an initial guess of 0.85,
    # and since the wheelie ceiling is inversely proportional to it, that 14%
    # would have gone straight into the acceleration limit.
    from bestiary.robots.whelp.massmodel import cg_height_measured
    h_cg = geo.stand_height_mm(spec) / 1000.0 * cg_height_measured(spec)
    wheelie_total = m * G * (wheelbase / 2.0) / max(h_cg, 1e-6)
    wheelie = wheelie_total * r / 4.0

    # The ground/geometry ceiling caps the force the TYRE may transmit; the motor
    # caps what the drive can produce, and on this machine the reflected rotor
    # inertia makes the motor bind far sooner. hound's equivalent analysis found
    # the opposite -- a wheelie was its real bound -- which is the whole point of
    # recomputing rather than inheriting the conclusion at a fifth of the mass.
    a_motor = spec.wheel_drive_rated_nm / (m * r / 4.0 + i_eff / r)
    a_ground = 4.0 * min(traction, wheelie) / r / m
    ceilings = {"traction (friction cone)": a_ground if traction <= wheelie else 1e9,
                "wheelie (geometry)": a_ground if wheelie < traction else 1e9,
                "motor + reflected rotor inertia": a_motor}
    binding = min(ceilings, key=ceilings.get)
    a_max = min(a_motor, a_ground)

    rpm = spec.wheel_drive_no_load_rad_s * 60.0 / (2.0 * math.pi)
    return WheelBudget(
        traction_limit_nm=traction, slope_nm=slope, accel_nm=accel,
        accel_ground_nm=accel_ground, accel_max_m_s2=a_max,
        wheelie_limit_nm=wheelie, rolling_nm=rolling, binding=binding,
        top_speed_m_s=spec.wheel_drive_no_load_rad_s * r, wheel_rpm=rpm,
        effective_mass_kg=m_eff,
    )


# ── The whole report ─────────────────────────────────────────────────────────
def analyse(spec: Spec = SPEC) -> dict:
    chain = geo.build_chain(spec)
    bodies = link_bodies(spec)
    m = _total_mass(bodies)

    stroke = yield_stroke_m(chain, spec)
    tire_only = spec.tire_stroke_mm / 1000.0

    cases = [
        _case(chain, bodies, spec, "stance_4",
              "standing, all four wheels down -- the case the brief asks about",
              1.0, geo.LEGS, rating="rated"),
        _case(chain, bodies, spec, "stance_3",
              "one leg lifted; the remaining three carry everything",
              1.0, ("FL", "RL", "RR"), rating="rated"),
        _case(chain, bodies, spec, "stance_2_diag",
              "trotting, one diagonal pair in contact -- transient, so judged against stall",
              1.0, ("FL", "RR"), rating="stall"),
        _case(chain, bodies, spec, "landing_design",
              f"{spec.design_drop_height_mm:.0f} mm drop into {tire_only * 1000:.0f} mm of "
              "tire stroke. The gearbox is RIGID at this timescale, so this is the real case",
              landing_load_factor(spec, tire_only), geo.LEGS, rating="gear"),
        _case(chain, bodies, spec, "landing_crooked",
              "the same drop, but landing on three wheels because it came down crooked",
              landing_load_factor(spec, tire_only), ("FL", "RL", "RR"), rating="gear"),
        _case(chain, bodies, spec, "landing_if_knee_could_yield",
              f"the same drop if the knee COULD fold {spec.yield_knee_rad:.2f} rad "
              f"({stroke * 1000:.0f} mm total). It cannot at 1:345 -- shown to size the gain "
              "a backdrivable actuator would buy",
              landing_load_factor(spec, stroke), geo.LEGS, rating="gear"),
    ]

    slow_chain = sorted(yield_chain(spec, impact=False), key=lambda t: t.wheel_force_n)
    impact_chain = sorted(yield_chain(spec, impact=True), key=lambda t: t.wheel_force_n)
    side_chain = lateral_chain(spec)
    wb = wheel_budget(chain, bodies, spec)
    static_leg_n = m * G / 4.0
    design_landing_n = static_leg_n * landing_load_factor(spec, tire_only)

    return {
        "mass_kg": m,
        "stance": {
            "abduct_rad": spec.stance_abduct_rad,
            "hip_rad": spec.stance_hip_rad,
            "knee_rad": geo.solve_stance_knee(spec),
            "knee_lever_mm": geo.knee_lever_mm(spec),
            "abduct_lever_mm": spec.abduct_axis_to_wheel_plane_mm,
            "stand_height_mm": geo.stand_height_mm(spec),
        },
        "servo": {
            "leg_stall_nm": spec.leg_servo_stall_nm,
            "leg_rated_nm": spec.leg_servo_rated_nm,
            "required_margin": spec.required_torque_margin,
            "usable_sustained_nm": spec.leg_servo_rated_nm / spec.required_torque_margin,
            "usable_transient_nm": spec.leg_servo_stall_nm / spec.required_torque_margin,
            "rigid_below_ms": spec.impact_rigid_below_ms,
        },
        "cases": [
            {
                "name": c.name, "why": c.why, "load_factor": c.load_factor,
                "contacts": list(c.contacts), "sustained": c.sustained, "rating": c.rating,
                "rating_nm": c.rating_nm(spec),
                "worst": {k: {"joint": c.worst(k)[0], "nm": c.worst(k)[1],
                              "margin": (c.rating_nm(spec) / c.worst(k)[1]
                                         if c.worst(k)[1] > 1e-9 else math.inf)}
                          for k in ("abduct", "hip", "knee")},
            }
            for c in cases
        ],
        "static_leg_n": static_leg_n,
        "design_landing_n": design_landing_n,
        "yield_chain_slow": [
            {"name": t.name, "n": t.wheel_force_n, "reversible": t.reversible,
             "basis": t.basis, "confidence": t.confidence}
            for t in slow_chain
        ],
        "yield_chain_impact": [
            {"name": t.name, "n": t.wheel_force_n, "reversible": t.reversible,
             "basis": t.basis, "confidence": t.confidence}
            for t in impact_chain
        ],
        "yield_chain_lateral": [
            {"name": t.name, "n": t.wheel_force_n, "reversible": t.reversible,
             "basis": t.basis, "confidence": t.confidence}
            for t in side_chain
        ],
        "yield_chain_slow_ok": slow_chain[0].reversible,
        "yield_chain_lateral_ok": side_chain[0].reversible,
        # THE VERTICAL IMPACT CHAIN HAS NO REVERSIBLE FIRST ITEM, AND CANNOT.
        #
        # This is a real conclusion, not a gap in the design. The servo is out of
        # the chain (rigid on impact timescales), so the only candidate is the
        # fuse -- and the fuse sits in series with the wheel, so to trip before
        # the gearbox at ~92 N it would have to be about a 2 mm^2 web. That is
        # both unprintable and only ~2x above a three-legged design landing,
        # which makes it a part that nuisance-trips. A fuse that nuisance-trips
        # is worse than no fuse, because the person building this will leave it
        # out and then have neither.
        #
        # So vertical impact is bounded by the ENVELOPE rather than by a part,
        # and the thing to assert is headroom: the lowest vertical-impact
        # threshold must sit well above the worst landing the design permits.
        # check.py tests that ratio instead of testing reversibility.
        "impact_headroom": (impact_chain[0].wheel_force_n / design_landing_n
                            if design_landing_n > 0 else math.inf),
        "impact_first_failure": impact_chain[0].name,
        # The fuse must be strong enough never to break in normal use and weak
        # enough to break before the calf. Both halves are checked, because a
        # fuse that nuisance-trips is worse than no fuse: it teaches whoever is
        # building this to leave it out.
        "fuse_headroom_over_landing": (
            (spec.fuse_shear_area_mm2 * spec.print_shear_strength_mpa) / design_landing_n
            if design_landing_n > 0 else math.inf),
        "landing": {
            "drop_mm": spec.design_drop_height_mm,
            "tire_stroke_mm": tire_only * 1000,
            "total_stroke_mm": stroke * 1000,
            "load_factor_tire_only": landing_load_factor(spec, tire_only),
            "load_factor_yielding": landing_load_factor(spec, stroke),
            "stroke_needed_for_margin_mm": _stroke_for_margin(chain, bodies, spec) * 1000,
            "max_drop_4leg_mm": _max_drop_mm(chain, bodies, spec, geo.LEGS),
            "max_drop_3leg_mm": _max_drop_mm(chain, bodies, spec, ("FL", "RL", "RR")),
            "max_drop_3leg_with_margin_mm": _max_drop_mm(
                chain, bodies, spec, ("FL", "RL", "RR"), spec.required_torque_margin),
            "contact_time_ms": 2000.0 * tire_only / max(
                math.sqrt(2 * G * spec.design_drop_height_mm / 1000.0), 1e-9),
        },
        "wheel": {
            "traction_nm": wb.traction_limit_nm, "slope_nm": wb.slope_nm,
            "accel_nm": wb.accel_nm, "wheelie_nm": wb.wheelie_limit_nm,
            "rolling_nm": wb.rolling_nm, "binding": wb.binding,
            "effective_mass_kg": wb.effective_mass_kg,
            "accel_ground_nm": wb.accel_ground_nm, "accel_max_m_s2": wb.accel_max_m_s2,
            "rated_nm": spec.wheel_drive_rated_nm,
            "drive_stall_nm": spec.wheel_drive_stall_nm,
            "top_speed_m_s": wb.top_speed_m_s, "wheel_rpm": wb.wheel_rpm,
        },
        "speed": {
            "top_speed_m_s": wb.top_speed_m_s,
            "body_lengths_per_s": wb.top_speed_m_s / (spec.trunk_len_mm / 1000.0),
            "leg_joint_rad_s": spec.leg_servo_no_load_rad_s,
            "swing_time_s": math.pi / 2 / max(spec.leg_servo_no_load_rad_s, 1e-6),
        },
    }


def _stroke_for_margin(chain, bodies, spec) -> float:
    """Absorber stroke needed for the landing to stay inside the torque margin.

    Inverts the landing case: find the load factor whose worst knee torque hits
    the usable limit, then invert the energy balance for the stroke that produces
    it. Reported rather than asserted, because the honest answer at this scale is
    "more stroke than a tire can give you", and that conclusion is the reason the
    yielding-knee design exists at all.
    """
    unit = _case(chain, bodies, spec, "unit", "", 1.0, geo.LEGS, rating="gear")
    per_g = unit.worst("knee")[1]
    if per_g <= 1e-9:
        return 0.0
    allowed = (spec.leg_servo_stall_nm * spec.servo_gear_break_multiple
               / spec.required_torque_margin)
    n = allowed / per_g
    if n <= 2.0:
        return math.inf     # even an infinitely soft landing does not help
    h = spec.design_drop_height_mm / 1000.0
    return 2.0 * h / (n - 2.0)


def _max_drop_mm(chain, bodies, spec, contacts=geo.LEGS, margin: float = 1.0) -> float:
    """Tallest drop the leg survives, on the tire's stroke alone.

    The envelope number. Not "what margin does the design drop have" but "how far
    can this machine actually fall before a joint is over its rating", which is
    the question anyone deploying it asks on the first day.

    Computed over the WORST joint of any kind, not just the knee. At this
    geometry the abduction servo is nearly as loaded as the knee -- its lever is
    51 mm against the knee's 57 mm -- and quoting a knee-only envelope would be
    optimistic by the difference.
    """
    unit = _case(chain, bodies, spec, "unit", "", 1.0, contacts, rating="gear")
    per_g = max(unit.worst(k)[1] for k in ("abduct", "hip", "knee"))
    if per_g <= 1e-9:
        return math.inf
    n = spec.leg_servo_stall_nm * spec.servo_gear_break_multiple / margin / per_g
    if n <= 2.0:
        return 0.0
    return spec.tire_stroke_mm * (n / 2.0 - 1.0)


def _fmt_report(a: dict, spec: Spec) -> str:
    L: list[str] = []
    w = L.append
    sv = a["servo"]
    need = sv["required_margin"]

    w("=" * 78)
    w("WHELP-16 TORQUE REPORT")
    w("=" * 78)
    w(f"  total mass            {a['mass_kg']:.3f} kg")
    w(f"  leg servo             {spec.servo_variant}")
    w(f"    rated (continuous)  {sv['leg_rated_nm']:.2f} N.m  -> "
      f"{sv['usable_sustained_nm']:.2f} N.m usable at {need:.0f}x   [sustained poses]")
    w(f"    stall (momentary)   {sv['leg_stall_nm']:.2f} N.m  -> "
      f"{sv['usable_transient_nm']:.2f} N.m usable at {need:.0f}x   [transients]")
    w("")
    w(f"  knee lever at stance  {a['stance']['knee_lever_mm']:.1f} mm"
      "   <- multiplies every newton the leg carries")
    w(f"  abduct lever          {a['stance']['abduct_lever_mm']:.1f} mm")
    w("  hip lever              0.0 mm   <- ZERO BY CONSTRUCTION, from the stance solve")
    w("")

    w("1  LOADING CASES              worst joint of each kind, |N.m| and margin vs its rating")
    w("")
    w(f"   {'case':<28} {'load':>6} {'vs':>6}  {'abduct':>14} {'hip':>14} {'knee':>14}")
    for c in a["cases"]:
        row = (f"   {c['name']:<28} {c['load_factor']:>5.2f}g "
               f"{c['rating']:>6} ")
        for k in ("abduct", "hip", "knee"):
            x = c["worst"][k]
            mark = " " if x["margin"] >= need else ("!" if x["margin"] >= 1.0 else "X")
            row += f" {x['nm']:>6.3f} {x['margin']:>5.1f}x{mark}"
        w(row)
    w("")
    w("   ' ' inside the 2x margin   '!' holds but under margin   'X' EXCEEDS THE RATING")
    w("")
    for c in a["cases"]:
        w(f"     {c['name']:<28} {c['why']}")
    w("")

    lg = a["landing"]
    w("2  THE LANDING, AND WHY THE SERVO CANNOT HELP")
    w("")
    w("   Load factor for a drop absorbed over a stroke is 2(h+s)/s. It depends on the")
    w("   RATIO of drop to stroke and not on mass at all -- so making the robot lighter")
    w("   does not soften a landing. Only stroke does.")
    w("")
    w(f"     design drop                       {lg['drop_mm']:.0f} mm")
    w(f"     tire stroke                       {lg['tire_stroke_mm']:.0f} mm"
      f"   -> {lg['load_factor_tire_only']:.1f}g")
    w(f"     contact duration                  {lg['contact_time_ms']:.0f} ms")
    w(f"     servo is RIGID below              {sv['rigid_below_ms']:.0f} ms of contact")
    w("")
    if lg["contact_time_ms"] < sv["rigid_below_ms"]:
        w(f"   {lg['contact_time_ms']:.0f} ms < {sv['rigid_below_ms']:.0f} ms, so THE GEARBOX "
          "DOES NOT YIELD DURING A LANDING.")
        w("   A 1:345 reduction reflects rotor inertia to the joint by the square of the")
        w(f"   ratio ({spec.servo_armature_kgm2:.3f} kg.m^2, ~100x the thigh's own). Backdriving")
        w("   that takes ~100 ms; a landing lasts tens. The Torque Limit register protects")
        w("   against LEANING and PUSHING, and against landing it does nothing at all.")
    else:
        w("   Contact lasts long enough for the joint to yield; the register limit applies.")
    w("")
    need_mm = lg["stroke_needed_for_margin_mm"]
    if math.isinf(need_mm):
        w("     stroke needed for a 2x margin:    UNREACHABLE at this drop height")
    else:
        w(f"     stroke needed for a 2x margin:    {need_mm:.0f} mm"
          f"   (tire gives {lg['tire_stroke_mm']:.0f} mm)")
    w("")
    w("     THE DROP ENVELOPE, on tire stroke alone:")
    w(f"       flat, four wheels                {lg['max_drop_4leg_mm']:>4.0f} mm")
    w(f"       crooked, three wheels            {lg['max_drop_3leg_mm']:>4.0f} mm"
      "   <- the binding case")
    w(f"       crooked, with the {need:.0f}x margin     "
      f"{lg['max_drop_3leg_with_margin_mm']:>4.0f} mm"
      f"   <- design_drop_height_mm is {lg['drop_mm']:.0f}")
    w("")
    w("   Three wheels is the binding case because a robot that comes down crooked lands")
    w("   on three, and that is the normal way to land rather than an unlucky one. Above")
    w("   the envelope a joint is over its peak rating and the load goes into plastic and")
    w("   gear teeth. Enforce it in the reward and the terrain curriculum -- not by hoping")
    w("   the policy is gentle. It will not be: a policy optimises, and free-falling is")
    w("   fast.")
    w("")
    w(f"   For scale: if the knee COULD fold {spec.yield_knee_rad:.2f} rad the stroke would be")
    w(f"   {lg['total_stroke_mm']:.0f} mm and the same drop would be "
      f"{lg['load_factor_yielding']:.1f}g instead of "
      f"{lg['load_factor_tire_only']:.1f}g. That is what a backdrivable")
    w("   actuator buys, and it is why every reference machine at this scale uses one.")
    w("")

    w("3  YIELD CHAINS       what gives first, in NEWTONS AT THE CONTACT PATCH.")
    w("")
    w("   One unit at one place, because the thresholds are not all torques -- the")
    w("   sacrificial fuse fails in direct shear and has no moment arm at all. Three")
    w("   chains, because a leg loaded slowly, struck from below, and struck from the")
    w("   side fail in three different orders.")
    w("")
    w(f"   For scale: each wheel carries {a['static_leg_n']:.1f} N standing and "
      f"{a['design_landing_n']:.0f} N in a design landing.")
    w("")
    for title, key, ok_key, note in (
        ("SLOW OVERLOAD -- leaning, pushing, a bad pose, a foot caught",
         "yield_chain_slow", "yield_chain_slow_ok",
         "the joint backs off, nothing breaks, pick it up and carry on"),
        ("VERTICAL IMPACT -- a landing. The servo is ABSENT from this chain.",
         "yield_chain_impact", None, ""),
        ("LATERAL IMPACT -- a fall, or a wheel into a table leg. Nothing can yield here.",
         "yield_chain_lateral", "yield_chain_lateral_ok",
         "the fuse breaks before the calf does, which is a part swap not a reprint"),
    ):
        w(f"   {title}")
        w("")
        w(f"     {'threshold':<34} {'N':>7}  {'kind':<11} basis")
        for i, t in enumerate(a[key]):
            kind = "REVERSIBLE" if t["reversible"] else "BREAKS"
            lead = "  -> " if i == 0 else "     "
            w(f"{lead}{t['name']:<34} {t['n']:>7.0f}  {kind:<11} {t['basis']}")
        if ok_key is None:
            hd = a["impact_headroom"]
            w("     NOTHING here is reversible, and nothing can be. The servo is rigid on")
            w(f"     impact timescales, and a fuse that tripped below "
              f"{a[key][0]['n']:.0f} N would have to")
            w("     be a ~2 mm^2 web: unprintable, and close enough to a real landing that it")
            w("     would nuisance-trip. A fuse that nuisance-trips gets left out, and then you")
            w("     have neither. So this case is bounded by the ENVELOPE, not by a part:")
            w("")
            w(f"       first failure   {a['impact_first_failure']} at {a[key][0]['n']:.0f} N")
            w(f"       design landing  {a['design_landing_n']:.0f} N")
            w(f"       headroom        {hd:.1f}x"
              + ("" if hd >= 3.0 else "   TOO LOW -- reduce design_drop_height_mm"))
            w("")
            w("     Which is why the drop envelope in section 2 is a hard limit and not")
            w("     advice, and why it belongs in the reward function.")
        elif a[ok_key]:
            w(f"     PASS: the lowest threshold is reversible -- {note}.")
        else:
            w("     FAIL: the first thing to give BREAKS. Reduce fuse_shear_area_mm2 until")
            w("     the fuse is first, or this overload costs a servo, not a two-gram part.")
        w("")
    hr = a["fuse_headroom_over_landing"]
    w(f"   Fuse headroom over a design landing: {hr:.1f}x")
    if hr < 3.0:
        w("   TOO LOW -- a fuse that nuisance-trips is worse than no fuse, because whoever")
        w("   is building this will simply leave it out. Increase fuse_shear_area_mm2.")
    w("")
    seen = set()
    for t in a["yield_chain_slow"] + a["yield_chain_impact"] + a["yield_chain_lateral"]:
        if t["confidence"].startswith(("ESTIMATE", "ASSUMED")) and t["name"] not in seen:
            seen.add(t["name"])
            w(f"     {t['name']:<34} {t['confidence']}")
    w("")

    wh = a["wheel"]
    w("4  WHEEL DRIVE      two different budgets, and which ceiling binds is the point")
    w("")
    w("   THROUGH THE TYRE (what the ground has to accept):")
    w(f"     rolling resistance, flat      {wh['rolling_nm']:>7.3f} N.m")
    w(f"     {spec.design_slope_deg:.0f} deg slope                 {wh['slope_nm']:>7.3f} N.m")
    w(f"     {spec.design_accel_m_s2:.1f} m/s^2 acceleration        "
      f"{wh['accel_ground_nm']:>7.3f} N.m")
    w(f"     friction-cone ceiling         {wh['traction_nm']:>7.3f} N.m"
      f"   mu = {spec.ground_friction}")
    w(f"     wheelie ceiling (geometry)    {wh['wheelie_nm']:>7.3f} N.m")
    w("")
    w("   OUT OF THE MOTOR (what the drive has to produce):")
    w(f"     {spec.design_accel_m_s2:.1f} m/s^2 acceleration        {wh['accel_nm']:>7.3f} N.m"
      f"   <- {1 - wh['accel_ground_nm'] / wh['accel_nm']:.0%} of it is")
    w("                                                     spinning up the ROTOR, not")
    w("                                                     pushing the robot")
    w(f"     drive rated (continuous)      {wh['rated_nm']:>7.3f} N.m")
    w(f"     drive stall (momentary)       {wh['drive_stall_nm']:>7.3f} N.m")
    w("")
    w("   Keeping these apart matters: rotor spin-up torque is produced inside the motor")
    w("   and never crosses the contact patch, so comparing it against a friction cone is")
    w("   a category error. An earlier version of this report did exactly that.")
    w("")
    w(f"   Effective mass for acceleration: {wh['effective_mass_kg']:.0f} kg, "
      f"against a real {a['mass_kg']:.2f} kg.")
    w(f"   Highest reachable acceleration:  {wh['accel_max_m_s2']:.2f} m/s^2, "
      f"limited by {wh['binding']}.")
    w("")
    w("   That is a 2.2 kg machine with the acceleration of a 59 kg one, and the cause is")
    w("   entirely the 1:345 gearbox. Note which ceiling binds: hound, on big direct-drive")
    w("   hub motors, was bound by a WHEELIE. This machine is bound by its own rotors long")
    w("   before the ground or the geometry has an opinion. Same topology, opposite answer,")
    w("   which is why it is recomputed rather than inherited.")
    w("")

    sp = a["speed"]
    w("5  SPEED BUDGET                   the constraint that actually binds this design")
    w("")
    w(f"   top speed              {sp['top_speed_m_s']:.2f} m/s   "
      f"({sp['body_lengths_per_s']:.1f} body lengths/s, wheel at {wh['wheel_rpm']:.0f} rpm)")
    w(f"   leg joint no-load rate {sp['leg_joint_rad_s']:.2f} rad/s")
    w(f"   90 deg leg swing takes {sp['swing_time_s']:.2f} s at no load, longer under load")
    w("")
    w("   This is the number to design the policy around, and the most likely single")
    w("   cause of a failed transfer. A policy trained without these limits will learn")
    w("   a gait built out of joint rates the hardware cannot produce, and it will not")
    w("   fail gracefully on the robot -- it will fail on the first step. Set the URDF's")
    w("   velocity limits to these values and keep the action rate below them.")
    w("")
    w("=" * 78)
    return "\n".join(L)


def main(argv: list[str]) -> int:
    a = analyse(SPEC)
    if "--json" in argv:
        print(json.dumps(a, indent=2))
        return 0
    print(_fmt_report(a, SPEC))

    need = SPEC.required_torque_margin
    bad = [f"{c['name']}/{k}" for c in a["cases"] for k in ("abduct", "hip", "knee")
           if c["worst"][k]["margin"] < need]
    if bad:
        print(f"UNDER THE {need:.1f}x MARGIN: {', '.join(bad)}")
    if not a["yield_chain_slow_ok"]:
        print("SLOW YIELD CHAIN FAILS: the first thing to give is not reversible.")
    if not a["yield_chain_lateral_ok"]:
        print("LATERAL YIELD CHAIN FAILS: a side impact breaks a leg before the fuse.")
    if a["impact_headroom"] < 3.0:
        print(f"VERTICAL IMPACT HEADROOM IS ONLY {a['impact_headroom']:.1f}x: "
              "lower design_drop_height_mm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
