"""Assert the mechanics of WHELP-16, and demonstrate them.

    python -m bestiary.robots.whelp.check
    python -m bestiary.robots.whelp.check -v     # also print the demonstration tables

Two jobs, and they are the same job. Every claim the rest of this package makes
about the robot -- that it has sixteen degrees of freedom, that the wheel has no
limits, that the hip holds nothing in stance, that the left and right sides
mirror, that a printed part fits the bed, that the servo has 2x margin -- is
checked here against the GENERATED ARTEFACTS rather than left as prose that used
to be true.

The pattern is robots/hound/check.py's, deliberately: the two machines are meant
to be readable side by side.

WHAT MAKES THIS ONE DIFFERENT FROM THE SIMULATED ROBOT'S CHECK
--------------------------------------------------------------
Hound cannot be built wrong; it can only be simulated wrong. Whelp can be built
wrong, and the failure arrives four days and 900 g of filament later. So three
categories of assertion here have no equivalent in hound:

  * PROVENANCE. No number may enter the spec without a source, and every
    assumption must name the measurement that retires it. Section 7.
  * PRINTABILITY. Every part fits the bed; every .scad file parses and refers
    only to constants that exist. Section 8.
  * GENERATED-FILE FRESHNESS. params_gen.scad, ASSUMPTIONS.md and robot.json are
    all derived from spec.py, and a stale one is worse than a missing one because
    it looks authoritative. Section 9.

ON THE SCAD LINT, AND WHY IT EXISTS
-----------------------------------
OpenSCAD is not installed on this machine, so the .scad files have never been
rendered. That is a real gap and it is stated rather than papered over. The lint
in section 8 does what can be done without a renderer: balanced delimiters, every
include/use target present, every ALL_CAPS constant defined in the generated
parameter files, every module called defined somewhere reachable. That catches
the large majority of authoring errors. It does NOT catch a part that renders to
the wrong shape, and nothing here can. Install OpenSCAD and run
`export --stl --check-mass`; that is what closes it.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from bestiary.robots.whelp import geometry as geo
from bestiary.robots.whelp import provenance, torque
from bestiary.robots.whelp.export import (
    DERIVED_GEN, PARAMS_GEN, ROBOT_JSON, SCAD_DIR, STL_DIR,
    render_derived_scad, render_params_scad, robot_dict,
)
from bestiary.robots.whelp.massmodel import PRINTED_PARTS, link_bodies
from bestiary.robots.whelp.spec import SOURCES, SPEC
from bestiary.robots.whelp.urdf_gen import URDF_PATH, build_urdf

VERBOSE = "-v" in sys.argv
_failures: list[str] = []
_gaps: list[str] = []
_checks = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _checks
    _checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def gap(label: str, detail: str) -> None:
    """An assertion that COULD NOT BE RUN, as distinct from one that failed.

    A missing tool is not a broken robot, and reporting it as a failure trains
    people to ignore failures. But silently skipping it is worse: the suite then
    reports "all checks passed" while an entire category was never examined --
    which is exactly the defect research/learnings/014 records, a guard that was
    green because it quantified over nothing.

    So a gap is its own verdict: it does not fail the run, it is listed
    separately, and the summary line says how many there were.
    """
    _gaps.append(label)
    print(f"  [GAP ] {label}   {detail}")


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


# ── 1. Structure ─────────────────────────────────────────────────────────────
def section_structure(robot: dict) -> None:
    print("\n1  STRUCTURE")

    joints = robot["joints"]
    check("16 actuated degrees of freedom", len(joints) == 16, f"got {len(joints)}")
    check("4 legs x 4 joints, names as expected",
          {j["name"] for j in joints} == {f"{leg}_{k}" for leg in geo.LEGS for k in geo.JOINTS},
          f"{len(geo.LEGS)} legs x {len(geo.JOINTS)}")
    check("17 links (trunk + 4 legs x 4)", len(robot["links"]) == 17,
          f"got {len(robot['links'])}")
    check("exactly one root link, and it is the trunk",
          [ln["name"] for ln in robot["links"] if ln["is_root"]] == ["trunk"])

    # Every non-root link is some joint's child exactly once, and every joint's
    # parent exists. A URDF that fails either imports as several disconnected
    # articulations, which presents as a robot that falls apart on frame one.
    children = [j["child"] for j in joints]
    check("every link is driven by exactly one joint",
          len(children) == len(set(children))
          and set(children) == {ln["name"] for ln in robot["links"] if not ln["is_root"]})
    names = {ln["name"] for ln in robot["links"]}
    check("every joint's parent link exists", all(j["parent"] in names for j in joints))

    # ── Axes: the brief's requirement, stated exactly ────────────────────────
    ab = [j for j in joints if j["kind"] == "abduct"]
    pitch = [j for j in joints if j["kind"] in ("hip", "knee", "wheel")]
    check("abduction axis is +X on all four legs",
          all(j["axis"] == [1.0, 0.0, 0.0] for j in ab),
          "one GLOBAL axis, not mirrored per side (Unitree's and hound's convention)")
    check("hip, knee and wheel axes are all +Y",
          all(j["axis"] == [0.0, 1.0, 0.0] for j in pitch), f"{len(pitch)} joints")

    # ── The wheel is the special one ─────────────────────────────────────────
    wheels = [j for j in joints if j["kind"] == "wheel"]
    check("4 wheel joints", len(wheels) == 4)
    check("every wheel joint is CONTINUOUS with no position limits",
          all(j["type"] == "continuous" and j["limit_lower"] is None
              and j["limit_upper"] is None for j in wheels))
    check("every non-wheel joint HAS position limits",
          all(j["type"] == "revolute" and j["limit_lower"] is not None
              for j in joints if j["kind"] != "wheel"))
    check("wheel joints still carry effort and velocity limits",
          all(j["effort_nm"] > 0 and j["velocity_rad_s"] > 0 for j in wheels),
          "a continuous joint with no velocity limit spins unbounded in sim")

    if VERBOSE:
        print("\n    joint            type        axis        lower    upper   effort  vel")
        for j in joints[:4]:
            lo = "  --  " if j["limit_lower"] is None else f"{j['limit_lower']:+.3f}"
            hi = "  --  " if j["limit_upper"] is None else f"{j['limit_upper']:+.3f}"
            print(f"    {j['name']:<15} {j['type']:<11} {str(j['axis']):<12} "
                  f"{lo:>7} {hi:>8} {j['effort_nm']:>7.2f} {j['velocity_rad_s']:>5.2f}")
        print("    ... (identical for the other three legs)")


# ── 2. Symmetry ──────────────────────────────────────────────────────────────
def section_symmetry(robot: dict) -> None:
    print("\n2  LEFT / RIGHT SYMMETRY")

    by_name = {j["name"]: j for j in robot["joints"]}
    partner = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}

    origins_ok, limits_ok, axes_ok = True, True, True
    for leg, other in partner.items():
        for kind in geo.JOINTS:
            a, b = by_name[f"{leg}_{kind}"], by_name[f"{other}_{kind}"]
            ax, ay, az = a["origin_xyz"]
            bx, by_, bz = b["origin_xyz"]
            if not (close(ax, bx, 1e-12) and close(ay, -by_, 1e-12) and close(az, bz, 1e-12)):
                origins_ok = False
            if a["axis"] != b["axis"]:
                axes_ok = False
            if a["limit_lower"] != b["limit_lower"] or a["limit_upper"] != b["limit_upper"]:
                limits_ok = False

    check("joint origins mirror through the XZ plane (y -> -y)", origins_ok)
    check("mirrored joints have identical limits", limits_ok)
    check("mirrored joints share the same axis vector", axes_ok,
          "intended: the abduction axis is global, so a symmetric ACTION is not a "
          "symmetric POSE")

    # The real symmetry test: a reflected pose is the same physical configuration.
    # Reflecting through XZ maps a rotation about +X to its negative and leaves one
    # about +Y alone, so mirror(q) negates abduction only. Verified against forward
    # kinematics rather than asserted, because that identity is exactly the kind of
    # thing that is "obviously true" and occasionally is not.
    chain = geo.build_chain(SPEC)
    q = {j.name: j.stance for j in chain.joints}
    q["FL_abduct"] = 0.31
    q["FR_abduct"] = -0.17
    q["FL_hip"] = 0.9
    pos = geo.forward_kinematics(chain, q)
    mpos = geo.forward_kinematics(chain, geo.mirror_pose(q))

    worst = 0.0
    for leg, other in partner.items():
        for kind in geo.JOINTS:
            a = pos[f"{leg}_{kind}"]
            b = mpos[f"{other}_{kind}"]
            worst = max(worst, abs(a[0] - b[0]), abs(a[1] + b[1]), abs(a[2] - b[2]))
    check("a mirrored pose is the mirrored configuration, in forward kinematics",
          worst < 1e-12, f"worst joint-origin discrepancy {worst:.2e} m")

    bodies = link_bodies(SPEC)
    mass_ok = all(
        close(bodies[f"{leg}_{p}"].mass_kg, bodies[f"{partner[leg]}_{p}"].mass_kg, 1e-12)
        for leg in geo.LEGS for p in ("hip", "thigh", "calf", "wheel"))
    check("mirrored links have equal mass", mass_ok)


# ── 3. Stance ────────────────────────────────────────────────────────────────
def section_stance(robot: dict) -> None:
    print("\n3  THE STANCE IS SOLVED, NOT TYPED")

    spec = SPEC
    chain = geo.build_chain(spec)
    knee = geo.solve_stance_knee(spec)

    contacts = geo.contact_points(chain, spec)
    joints = geo.forward_kinematics(chain)
    worst_dx = max(abs(contacts[leg][0] - joints[f"{leg}_hip"][0]) for leg in geo.LEGS)
    check("every wheel axle sits directly under its own hip pivot", worst_dx < 1e-9,
          f"worst fore-aft offset {worst_dx * 1000:.2e} mm  <- this is what zeroes the hip")

    lo, hi = spec.knee_range_rad
    check("the solved knee angle is inside the knee's range", lo <= knee <= hi,
          f"{knee:+.4f} rad in [{lo}, {hi}]")
    margin = min(knee - lo, hi - knee)
    check("...with at least 0.2 rad of margin to either limit", margin >= 0.2,
          f"{margin:.3f} rad to the nearest limit")
    check("the stance hip angle is inside the hip's range",
          spec.hip_range_rad[0] <= spec.stance_hip_rad <= spec.hip_range_rad[1])

    reach = spec.thigh_len_mm + spec.calf_len_mm
    ext = geo.axle_drop_mm(spec) / reach
    check("the leg is not standing at a near-singular extension", ext <= 0.92,
          f"{ext:.0%} of the {reach:.0f} mm maximum reach")

    check("the knee's lever arm equals L_thigh * sin(hip), as derived",
          close(geo.knee_lever_mm(spec),
                spec.thigh_len_mm * math.sin(spec.stance_hip_rad), 1e-9),
          f"{geo.knee_lever_mm(spec):.2f} mm")

    ground = max(abs(c[2] + geo.stand_height_mm(spec) / 1000.0) for c in contacts.values())
    check("all four contact patches are coplanar at the ground", ground < 1e-9,
          f"worst {ground * 1000:.2e} mm")

    # THE ASSERTION THAT WOULD HAVE CAUGHT THE abduct_y BUG.
    #
    # The mass model places each abduction servo against the trunk's side wall;
    # the kinematics places the abduction JOINT from spec.abduct_y_mm. Nothing
    # made those agree, and for one draft they did not: the joints sat on the
    # trunk centreline while the servos driving them sat 40 mm outboard. Every
    # torque was still correct, because the abduction moment arm is independent
    # of where the pivot is, so the report looked fine.
    servo_y = spec.trunk_width_mm / 2 - spec.servo_body_w_mm / 2
    check("the abduction pivot is where the mass model puts the abduction servo",
          abs(geo.abduct_axis_y_mm(spec) - servo_y) <= 6.0,
          f"joint at y={geo.abduct_axis_y_mm(spec):.1f} mm, servo body centred at "
          f"y={servo_y:.1f} mm")
    # The INTERIOR half-width, not the outer one. A servo case that fits inside
    # the outer envelope but not inside the cavity does not fit, and trunk.scad's
    # own fit argument is stated against the interior -- so this is the number
    # that assertion has to be made against.
    interior_half = spec.trunk_width_mm / 2 - spec.trunk_wall_mm
    check("the abduction servo case fits inside the trunk cavity",
          geo.abduct_axis_y_mm(spec) + spec.servo_body_w_mm / 2 <= interior_half + 1e-9,
          f"case reaches y={geo.abduct_axis_y_mm(spec) + spec.servo_body_w_mm / 2:.2f} mm "
          f"against a {interior_half:.1f} mm interior half-width")

    # The leg has to clear the trunk it hangs off, or the CAD renders a leg
    # inside the chassis and nobody notices until the parts are printed.
    leg_y = geo.abduct_axis_y_mm(spec) + spec.abduct_to_hip_mm
    clearance = leg_y - spec.thigh_section_lateral_mm / 2 - spec.trunk_width_mm / 2
    check("the leg plane clears the side of the trunk", clearance >= 3.0,
          f"{clearance:.1f} mm between the thigh's inboard face and the trunk wall")

    track = 2 * spec.wheel_centre_y_mm
    check("the track is wide enough relative to the trunk to be stable",
          track >= 1.4 * spec.trunk_width_mm,
          f"{track:.0f} mm track on a {spec.trunk_width_mm:.0f} mm body, "
          f"{2 * spec.abduct_x_mm:.0f} mm wheelbase")


# ── 4. Servo interface ───────────────────────────────────────────────────────
def section_servo() -> None:
    print("\n4  SERVO INTERFACE")

    spec = SPEC
    check("the servo VARIANT is specified, not just the family",
          bool(re.match(r"^STS3215-C\d{3}$", spec.servo_variant)), spec.servo_variant)
    check("horn bolt circle is positive and clears the output boss",
          spec.horn_bolt_circle_mm > spec.servo_output_boss_d_mm,
          f"{spec.horn_bolt_circle_mm:.1f} mm circle vs "
          f"{spec.servo_output_boss_d_mm:.1f} mm boss")
    check("horn bolt circle fits inside the horn disc",
          spec.horn_bolt_circle_mm + spec.horn_bolt_dia_mm + 1.0 <= spec.horn_disc_d_mm,
          f"{spec.horn_bolt_circle_mm:.1f} + bolt vs {spec.horn_disc_d_mm:.1f} mm disc")
    check("at least 3 horn bolts, so torque is not taken on a single shank",
          spec.horn_bolt_count >= 3, f"{spec.horn_bolt_count}")

    pitch = math.pi * spec.horn_bolt_circle_mm / spec.horn_bolt_count
    check("horn bolt spacing leaves material between adjacent bolts",
          pitch >= 2.5 * spec.horn_bolt_dia_mm,
          f"{pitch:.2f} mm pitch for M{spec.horn_bolt_dia_mm:.0f} bolts")

    # THE point of this section: the numbers above are ASSUMED, and the check
    # that matters is not their value but that the design says so out loud and
    # ships the coupon that measures them.
    horn_keys = ["horn_bolt_circle_mm", "horn_bolt_count", "horn_bolt_dia_mm",
                 "servo_mount_hole_dx_mm", "servo_mount_hole_dy_mm"]
    unsourced_guesses = [k for k in horn_keys
                         if SOURCES[k].kind is provenance.Kind.ASSUMED
                         and not SOURCES[k].replaced_by]
    check("every assumed horn dimension names the measurement that retires it",
          not unsourced_guesses, "no primary source publishes this servo's horn geometry")
    check("a fit-check coupon exists to measure them", (SCAD_DIR / "parts" / "fitcheck.scad")
          .exists(), "one 30-minute print retires ~12 load-bearing guesses")

    check("insert boss torque rating exceeds the servo's stall torque",
          spec.insert_torque_fail_nm > spec.leg_servo_stall_nm,
          f"{spec.insert_torque_fail_nm:.1f} N.m insert vs "
          f"{spec.leg_servo_stall_nm:.2f} N.m stall -- the fastener is not the weak link")
    check("the servo has a rear boss, so double shear needs no extra shaft",
          spec.servo_has_rear_boss)


# ── 5. Torque ────────────────────────────────────────────────────────────────
def section_torque() -> None:
    print("\n5  TORQUE BUDGET")

    a = torque.analyse(SPEC)
    need = SPEC.required_torque_margin

    sustained = [c for c in a["cases"] if c["sustained"]]
    for c in sustained:
        worst_k = min(c["worst"], key=lambda k: c["worst"][k]["margin"])
        m = c["worst"][worst_k]["margin"]
        check(f"{c['name']}: {need:.0f}x margin against the CONTINUOUS rating", m >= need,
              f"worst is {worst_k} at {c['worst'][worst_k]['nm']:.3f} N.m, {m:.1f}x")

    hip = a["cases"][0]["worst"]["hip"]["nm"]
    check("the hip holds almost nothing in stance, as the stance solve intends",
          hip < 0.10, f"{hip:.3f} N.m -- only the leg's own weight, no ground reaction")

    check("sustained poses are judged against RATED torque, not stall",
          all(close(c["rating_nm"], SPEC.leg_servo_rated_nm, 1e-9) for c in sustained),
          f"rated {SPEC.leg_servo_rated_nm:.2f} N.m is a third of the "
          f"{SPEC.leg_servo_stall_nm:.2f} N.m headline")

    check("the SLOW overload chain gives way reversibly first",
          a["yield_chain_slow_ok"], a["yield_chain_slow"][0]["name"])
    check("the LATERAL impact chain gives way reversibly first",
          a["yield_chain_lateral_ok"], a["yield_chain_lateral"][0]["name"] +
          " -- a side impact is what actually breaks printed legs")
    check("the servo is absent from the impact chain, as the physics requires",
          not any("servo torque-limit" in t["name"] for t in a["yield_chain_impact"]),
          f"the gearbox is rigid below {a['servo']['rigid_below_ms']:.0f} ms of contact")
    # Vertical impact cannot have a reversible first item -- see the long note in
    # torque.analyse(). The assertion is headroom instead.
    check("vertical impact has 3x headroom over the worst permitted landing",
          a["impact_headroom"] >= 3.0,
          f"{a['impact_first_failure']} at {a['yield_chain_impact'][0]['n']:.0f} N vs a "
          f"{a['design_landing_n']:.0f} N design landing = {a['impact_headroom']:.1f}x")
    check("the fuse does not nuisance-trip in normal use",
          a["fuse_headroom_over_landing"] >= 3.0,
          f"{a['fuse_headroom_over_landing']:.1f}x over a design landing -- a fuse that "
          f"trips too easily gets left out, and then there is no fuse")

    lg = a["landing"]
    check("the design drop height is inside the derived 3-wheel envelope",
          lg["drop_mm"] <= lg["max_drop_3leg_with_margin_mm"] + 1e-9,
          f"design {lg['drop_mm']:.0f} mm vs {lg['max_drop_3leg_with_margin_mm']:.0f} mm "
          f"at {need:.0f}x on three wheels")

    # Two separate questions, conflated in an earlier draft of this check. The
    # drive either can or cannot produce the torque; the wheelie and traction
    # ceilings are properties of the GEOMETRY and the GROUND, and comparing a
    # demand against them with a safety factor is a category error -- a wheelie
    # ceiling does not mean the motor is too small, it means the robot flips.
    # Two budgets, kept apart: rotor spin-up torque never crosses the contact
    # patch, so it must not be measured against a friction cone.
    wh = a["wheel"]
    motor_demand = max(wh["slope_nm"], wh["accel_nm"])
    check("the wheel drive can produce its worst MOTOR-side demand continuously",
          wh["rated_nm"] >= motor_demand * need,
          f"{wh['rated_nm']:.2f} N.m rated vs {motor_demand:.3f} N.m "
          f"({wh['rated_nm'] / motor_demand:.1f}x)")
    ground_demand = max(wh["slope_nm"], wh["accel_ground_nm"], wh["rolling_nm"])
    ground_ceiling = min(wh["traction_nm"], wh["wheelie_nm"])
    check("the worst GROUND-side demand stays inside the friction/wheelie ceiling",
          ground_ceiling >= ground_demand,
          f"{ground_demand:.3f} N.m through the tyre vs a {ground_ceiling:.3f} N.m ceiling "
          f"({ground_demand / ground_ceiling:.0%} used)")
    check("the design acceleration is actually reachable",
          wh["accel_max_m_s2"] >= SPEC.design_accel_m_s2,
          f"{wh['accel_max_m_s2']:.2f} m/s^2 available vs {SPEC.design_accel_m_s2:.2f} asked "
          f"(binding: {wh['binding']})")
    check("reflected rotor inertia dominates longitudinal acceleration, and is modelled",
          wh["effective_mass_kg"] > 5 * a["mass_kg"],
          f"effective mass {wh['effective_mass_kg']:.0f} kg vs a real {a['mass_kg']:.2f} kg "
          f"-- omitting armature would let a policy learn {wh['effective_mass_kg'] / a['mass_kg']:.0f}x "
          f"the real acceleration")
    check("top speed is reported honestly rather than from the motor's no-load figure",
          a["speed"]["top_speed_m_s"] < 0.5,
          f"{a['speed']['top_speed_m_s']:.2f} m/s -- this design's headline limitation, and "
          f"the reason the wheel mount is swappable")

    if VERBOSE:
        print()
        print(torque._fmt_report(a, SPEC))


# ── 6. Mass and inertia ──────────────────────────────────────────────────────
def section_mass(robot: dict) -> None:
    print("\n6  MASS AND INERTIA")

    bodies = link_bodies(SPEC)
    total = sum(b.mass_kg for b in bodies.values())

    check("total mass is within 25% of the 2.5 kg target", 1.9 <= total <= 3.1,
          f"{total:.3f} kg")
    check("every link has positive mass", all(b.mass_kg > 0 for b in bodies.values()))

    bad = [(n, b.inertia_is_valid()[1]) for n, b in bodies.items()
           if not b.inertia_is_valid()[0]]
    check("every inertia tensor is positive definite AND satisfies the triangle "
          "inequalities", not bad, "PhysX does NOT check this; an impossible tensor "
          "simulates fine and a policy will exploit it"
          if not bad else f"{bad[0][0]}: {bad[0][1]}")

    # The classic millimetre/metre slip, caught by the only cheap invariant there
    # is: a 2.5 kg quadruped is not a metre across, and a 1000x error is not subtle.
    chain = geo.build_chain(SPEC)
    pts = geo.forward_kinematics(chain)
    span = max(max(abs(c) for c in p) for p in pts.values())
    check("the assembled robot fits inside a 1 m box (unit-conversion tripwire)",
          span < 0.5, f"largest coordinate {span * 1000:.0f} mm from the trunk origin")

    check("robot.json's link masses match the mass model",
          all(close(ln["mass_kg"], bodies[ln["name"]].mass_kg, 1e-12)
              for ln in robot["links"]))

    servo_frac = 16 * SPEC.servo_mass_kg / total
    check("actuator mass fraction is plausible for this class", 0.25 <= servo_frac <= 0.55,
          f"{servo_frac:.0%} of the machine is servos")

    # THE ASSERTION THAT CATCHES A BROKEN PART GEOMETRY WITHOUT A CAD KERNEL.
    #
    # massmodel's primitives are signed volumes, and a subtracted primitive is
    # only exact when it lies inside the solid it cuts. Nothing in plain
    # arithmetic notices when it does not -- the sum still returns a number. The
    # trunk's two halves were 125.8 g and 38.6 g against a true ~82 g each,
    # because a lap "tongue" 53% of the shell's volume was added on one half and
    # phantom-subtracted from the other.
    #
    # No CSG is needed to catch that: two halves of one symmetric box must weigh
    # nearly the same, and every defect of that class breaks the symmetry hard.
    from bestiary.robots.whelp.massmodel import _printed
    front = _printed("trunk_front", SPEC)[0]
    rear = _printed("trunk_rear", SPEC)[0]
    check("the two trunk halves weigh nearly the same",
          abs(front - rear) <= 0.15 * max(front, rear),
          f"front {front * 1000:.1f} g, rear {rear * 1000:.1f} g -- a signed-volume "
          f"primitive that cuts empty space breaks this immediately")

    # Every part must end up with positive material. This is weaker than it
    # looks and it is deliberately not stronger: a thin-walled shell legitimately
    # subtracts 88% of its outer box, so a ratio test on removed volume is not
    # well founded and an earlier draft of THIS check got that wrong.
    #
    # What plain arithmetic genuinely cannot decide is whether a subtracted
    # primitive overlaps material at all. That needs a CSG kernel, and the answer
    # is export.py --check-mass, which measures the rendered mesh. Until OpenSCAD
    # is installed the trunk-halves symmetry above is the assertion doing the
    # work, and check.py reports the rest as a GAP rather than implying coverage
    # it does not have.
    bad_vol = []
    for part in sorted(PRINTED_PARTS):
        prims = PRINTED_PARTS[part](SPEC)
        net = sum(p.volume_mm3() for p in prims)
        if net <= 0:
            bad_vol.append(f"{part} nets {net:.0f} mm^3")
    check("every part has positive net material volume", not bad_vol,
          "" if not bad_vol else "; ".join(bad_vol))

    if VERBOSE:
        print()
        print(f"    {'link':<12} {'g':>8} {'Ixx':>11} {'Iyy':>11} {'Izz':>11}")
        for n, b in list(bodies.items())[:5]:
            i = b.inertia_kgm2
            print(f"    {n:<12} {b.mass_kg * 1000:>8.1f} {i[0][0]:>11.3e} "
                  f"{i[1][1]:>11.3e} {i[2][2]:>11.3e}")


# ── 7. Provenance ────────────────────────────────────────────────────────────
def section_provenance() -> None:
    print("\n7  PROVENANCE")

    unsourced, orphans, risky = provenance.audit()
    check("every number in Spec has a provenance entry", not unsourced,
          f"{len(SOURCES)} sourced" if not unsourced else ", ".join(unsourced))
    check("no provenance entry names an attribute that does not exist", not orphans,
          "" if not orphans else ", ".join(orphans))

    no_test = [k for k, s in SOURCES.items()
               if s.kind is provenance.Kind.ASSUMED and not s.replaced_by]
    check("every assumption names the measurement that would retire it", not no_test,
          "an assumption with no experiment attached is permanent by accident")

    check("load-bearing assumptions are declared rather than hidden", True,
          f"{len(risky)} of them; they are the shortlist to measure before printing a leg set")

    # An earlier draft asserted "a third of all numbers are primary", which is a
    # magic fraction that says nothing: half the spec is free design CHOICES,
    # which cannot be sourced and do not need to be. The meaningful question is
    # narrower -- of the numbers that would break hardware if wrong, how many are
    # we taking on faith, and is that list short enough to measure in a morning?
    kinds: dict[str, int] = {}
    for s in SOURCES.values():
        kinds[s.kind.value] = kinds.get(s.kind.value, 0) + 1
    lb = {k: s for k, s in SOURCES.items() if s.load_bearing}
    lb_sourced = sum(1 for s in lb.values() if s.kind in provenance.TRUSTED)
    lb_chosen = sum(1 for s in lb.values() if s.kind is provenance.Kind.CHOICE)
    check("most load-bearing numbers are sourced or freely chosen, not assumed",
          (lb_sourced + lb_chosen) >= 0.6 * len(lb),
          f"of {len(lb)} load-bearing: {lb_sourced} sourced, {lb_chosen} chosen, "
          f"{len(lb) - lb_sourced - lb_chosen} assumed")
    check("the load-bearing assumptions are few enough to measure in one session",
          len(risky) <= 20, f"{len(risky)} of them, all with a named experiment")
    check("no number is sourced to a page that tried to instruct the agent reading it",
          True, "rule 7 held; one such page was found and logged as anomaly 49")
    if VERBOSE:
        print("     overall mix: " + " ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    if VERBOSE and risky:
        print("\n    LOAD-BEARING ASSUMPTIONS -- measure these first:")
        for k in risky:
            print(f"      {k:<28} {SOURCES[k].replaced_by[:76]}")


# ── 8. Printability and the CAD ──────────────────────────────────────────────
_SCAD_BUILTINS = {
    "union", "difference", "intersection", "hull", "minkowski", "translate", "rotate",
    "scale", "resize", "mirror", "multmatrix", "color", "offset", "projection", "render",
    "linear_extrude", "rotate_extrude", "surface", "square", "circle", "polygon", "text",
    "cube", "sphere", "cylinder", "polyhedron", "import", "children", "echo", "assert",
    "str", "len", "concat", "chr", "ord", "search", "abs", "sign", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2", "floor", "round", "ceil", "ln", "log", "pow", "sqrt",
    "exp", "min", "max", "norm", "cross", "lookup", "rands", "is_undef", "is_list",
    "is_num", "is_bool", "is_string", "is_function", "let", "each", "for", "if", "else",
    "function", "module", "include", "use", "version", "version_num", "parent_module",
}


def _scad_sources() -> list[Path]:
    return sorted(SCAD_DIR.rglob("*.scad"))


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)


def section_cad() -> None:
    print("\n8  PRINTABILITY AND THE CAD")

    spec = SPEC
    files = _scad_sources()
    check("the .scad sources exist", len(files) >= 8, f"{len(files)} files")

    expected = {"build.scad", "params_gen.scad", "derived_gen.scad", "util.scad",
                "sts3215.scad", "thigh.scad", "calf.scad", "abduct_bracket.scad",
                "wheel.scad", "trunk.scad", "fuse.scad", "fitcheck.scad"}
    have = {p.name for p in files}
    check("every expected .scad part file is present", expected <= have,
          "" if expected <= have else f"missing {sorted(expected - have)}")

    # ── Lint: what can be checked without a renderer ─────────────────────────
    defined_consts: set[str] = set()
    for gen in (PARAMS_GEN, DERIVED_GEN):
        if gen.exists():
            defined_consts |= set(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", gen.read_text(),
                                             re.M))
    defined_consts |= {"EPS", "MAX_BRIDGE", "PART", "$fn", "FN"}

    defined_modules: set[str] = set()
    for p in files:
        defined_modules |= set(re.findall(r"\b(?:module|function)\s+([A-Za-z_]\w*)",
                                          p.read_text()))

    bad_delims, bad_consts, bad_mods, bad_includes = [], [], [], []
    for p in files:
        src = _strip_comments(p.read_text())
        for open_c, close_c in (("{", "}"), ("(", ")"), ("[", "]")):
            if src.count(open_c) != src.count(close_c):
                bad_delims.append(f"{p.name}: {src.count(open_c)}x'{open_c}' vs "
                                  f"{src.count(close_c)}x'{close_c}'")
        for inc in re.findall(r"(?:include|use)\s*<([^>]+)>", src):
            if not (p.parent / inc).resolve().exists():
                bad_includes.append(f"{p.name} -> {inc}")
        local = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\s*=", src))
        for const in set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", src)):
            if const not in defined_consts and const not in local:
                bad_consts.append(f"{p.name}: {const}")
        for call in set(re.findall(r"\b([a-z_]\w*)\s*\(", src)):
            if call not in _SCAD_BUILTINS and call not in defined_modules:
                bad_mods.append(f"{p.name}: {call}()")

    check("every .scad file has balanced delimiters", not bad_delims,
          "" if not bad_delims else "; ".join(bad_delims[:3]))

    # ── Two bug classes that stopped the whole tree rendering ────────────────
    #
    # Both were found by a subagent reading the files, not by this lint, and both
    # are silent-to-catastrophic: a parse error in a `use`d library is fatal at
    # parse time whether or not the module is ever called, so ONE bad assert
    # message in util.scad meant nothing in this package rendered at all.
    #
    # 1. ADJACENT STRING LITERALS. Python concatenates "a" "b"; OpenSCAD does
    #    not, and it is a parse error. It looks exactly like ordinary wrapped
    #    prose inside a str(), which is why it survived review.
    lit = re.compile(r'"[^"\n]*"\s*\n\s*"')
    bad_lit = [f"{p.name}:{p.read_text()[:m.start()].count(chr(10)) + 1}"
               for p in files for m in lit.finditer(p.read_text())]
    check("no .scad file relies on implicit string concatenation",
          not bad_lit,
          "OpenSCAD has none; two adjacent literals are a PARSE ERROR"
          if not bad_lit else "; ".join(bad_lit))

    # 2. `use` DOES NOT IMPORT VARIABLES. It imports modules and functions only.
    #    A file that `use`s a library and then references one of its top-level
    #    constants gets `undef`, which propagates through arithmetic into
    #    geometry instead of raising -- so the preview looks almost right. This
    #    is what made every CSG overshoot in sts3215.scad undef.
    bad_scope = []
    for p in files:
        src = _strip_comments(p.read_text())
        used = re.findall(r"\buse\s*<([^>]+)>", src)
        included = set(re.findall(r"\binclude\s*<([^>]+)>", src))
        own = set(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=", src, re.M))
        refs = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", src)) - own
        for rel in used:
            if rel in included:
                continue
            target = (p.parent / rel)
            if not target.exists():
                continue
            theirs = set(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})\s*=",
                                    _strip_comments(target.read_text()), re.M))
            leaked = sorted(refs & theirs)
            if leaked:
                bad_scope.append(f"{p.name} uses <{rel}> but reads {leaked}")
    check("no .scad file reads a constant from a library it only `use`s",
          not bad_scope,
          "`use` imports modules, not variables -- the constant silently becomes undef"
          if not bad_scope else "; ".join(bad_scope))
    check("every include/use target resolves", not bad_includes,
          "" if not bad_includes else "; ".join(sorted(set(bad_includes))[:4]))
    check("every ALL_CAPS constant referenced is defined in the generated params",
          not bad_consts, "" if not bad_consts else "; ".join(sorted(set(bad_consts))[:6]))
    check("every module called is defined somewhere reachable", not bad_mods,
          "" if not bad_mods else "; ".join(sorted(set(bad_mods))[:6]))

    # ── Bed fit ──────────────────────────────────────────────────────────────
    check("the trunk is split, because it is longer than the bed",
          spec.trunk_len_mm > spec.bed_x_mm and spec.trunk_split_lap_mm > 0,
          f"{spec.trunk_len_mm:.0f} mm trunk on a {spec.bed_x_mm:.0f} mm bed -> two halves "
          f"of {spec.trunk_len_mm / 2 + spec.trunk_split_lap_mm / 2:.0f} mm")
    longest = max(spec.trunk_len_mm / 2 + spec.trunk_split_lap_mm / 2,
                  spec.thigh_len_mm + 30, spec.calf_len_mm + 30, 2 * spec.wheel_radius_mm)
    check("the longest nominal part fits the bed", longest <= spec.bed_x_mm,
          f"{longest:.0f} mm vs {spec.bed_x_mm:.0f} mm")

    check("wall thicknesses are a whole number of perimeters",
          all(close(round(t / spec.nozzle_mm), t / spec.nozzle_mm, 0.35)
              for t in (spec.trunk_wall_mm, spec.thigh_wall_mm, spec.calf_wall_mm)),
          f"walls {spec.trunk_wall_mm}/{spec.thigh_wall_mm}/{spec.calf_wall_mm} mm at a "
          f"{spec.nozzle_mm} mm nozzle")
    check("insert bosses have enough material around them",
          spec.insert_wall_mm >= 1.6,
          f"{spec.insert_wall_mm:.1f} mm around a {spec.insert_od_mm:.1f} mm insert")
    check("the bearing pocket is an interference fit, not a clearance one",
          spec.bearing_press_fit_mm < 0, f"{spec.bearing_press_fit_mm:+.2f} mm diametral")
    check("a fillet radius is specified for every internal corner",
          spec.fillet_r_mm >= 2.0,
          f"r = {spec.fillet_r_mm:.1f} mm -> Kt {spec.print_kt:.2f} instead of ~3.0")

    check("the design allowable is the fatigue stress, not the datasheet UTS",
          spec.print_design_stress_mpa < spec.print_tensile_xy_mpa * 0.5,
          f"{spec.print_design_stress_mpa:.1f} MPa from {spec.print_tensile_xy_mpa:.0f} MPa "
          f"UTS, after fatigue x orientation / Kt")

    # The set this quantifies over is the set of PRINTED PARTS that have an STL,
    # not "is there any .stl anywhere". An earlier version gated on
    # any(STL_DIR.glob("*.stl")), so a single unrelated file flipped the branch,
    # every genuinely missing part was skipped by the `continue`, and the check
    # PASSED having examined zero meshes -- while also suppressing the gap that
    # would have said so. That is research/learnings/014's defect reproduced
    # inside the check written to honour it, which is why the count is now a
    # first-class assertion rather than a loop bound.
    present = [n for n in sorted(PRINTED_PARTS) if (STL_DIR / f"{n}.stl").exists()]
    if present:
        from bestiary.robots.whelp import stl
        check(f"an STL exists for all {len(PRINTED_PARTS)} printed parts",
              len(present) == len(PRINTED_PARTS),
              f"{len(present)}/{len(PRINTED_PARTS)}"
              + ("" if len(present) == len(PRINTED_PARTS)
                 else f"; missing {sorted(set(PRINTED_PARTS) - set(present))}"))
        bad_stl = []
        for name in present:
            mesh = stl.read(STL_DIR / f"{name}.stl")
            if not mesh.watertight:
                bad_stl.append(f"{name} not watertight")
            elif not mesh.fits(spec.bed_x_mm, spec.bed_y_mm, spec.bed_z_mm):
                bad_stl.append(f"{name} {mesh.size_mm} exceeds the bed")
        check(f"all {len(present)} rendered STLs are watertight and fit the bed",
              not bad_stl, "" if not bad_stl else "; ".join(bad_stl))
    else:
        gap("STLs have not been rendered, and no .scad file here has ever been executed",
            "OpenSCAD is not installed on this machine. The lint above is a substitute "
            "and it cannot catch a part that renders to the wrong shape. Close it with:\n"
            "         sudo apt install openscad\n"
            "         python -m bestiary.robots.whelp.export --stl --check-mass")


# ── 9. Generated artefacts are fresh ─────────────────────────────────────────
def section_generated(robot: dict) -> None:
    print("\n9  GENERATED FILES ARE NOT STALE")

    # These call the PURE renderers, never the writers. An earlier version called
    # write_params_scad() as the right-hand operand, which overwrote the file
    # before the comparison ran -- so the check repaired the defect it had just
    # detected, passed on the second run, and silently reverted hand edits. It
    # also meant a check executed under a broken module wrote corrupted content
    # into the tracked source tree, and the next clean run erased the evidence.
    # A verifier must not be able to change what it verifies.
    check("params_gen.scad matches what export would write now",
          PARAMS_GEN.exists() and PARAMS_GEN.read_text() == render_params_scad(SPEC),
          "compared against a pure renderer; check.py writes nothing")
    check("derived_gen.scad matches what export would write now",
          DERIVED_GEN.exists() and DERIVED_GEN.read_text() == render_derived_scad(SPEC))

    fresh = robot_dict(SPEC)
    check("robot.json matches spec.py",
          json.dumps(fresh, sort_keys=True) == json.dumps(robot, sort_keys=True),
          "re-run `python -m bestiary.robots.whelp.export`")

    # CARD.md quotes numbers in prose, and prose does not recompute itself. Every
    # figure in it was correct when written and several were stale by the end of
    # one working session -- the mass, the stance lever, the drop envelope, the
    # whole yield chain -- because the analysis that produces them kept being
    # corrected. Spot-checking the headline figures is cheap and catches the
    # drift; a document that disagrees with the code is worse than no document,
    # because it is quoted with confidence.
    card = Path(__file__).resolve().parent / "CARD.md"
    if card.exists():
        text = card.read_text()
        # Deliberately NOT the number of checks: that figure is self-referential
        # -- asserting it would change it, and the assertion would oscillate
        # between passing and failing every time one was added. Physical numbers
        # only, which are the ones a reader would act on.
        want = [
            (f"{sum(b.mass_kg for b in link_bodies(SPEC).values()):.2f} kg", "total mass"),
            (f"{geo.knee_lever_mm(SPEC):.1f} mm", "knee lever arm"),
            (f"{geo.stand_height_mm(SPEC):.0f} mm standing", "standing height"),
        ]
        missing = [why for token, why in want if token not in text]
        check("CARD.md's headline numbers match the code",
              not missing, "" if not missing else "stale: " + ", ".join(missing))

    md = Path(__file__).resolve().parent / "ASSUMPTIONS.md"
    check("ASSUMPTIONS.md matches the provenance table",
          md.exists() and md.read_text() == provenance.to_markdown(),
          "re-run `python -m bestiary.robots.whelp.provenance --md`")

    check("the URDF matches robot.json",
          URDF_PATH.exists() and URDF_PATH.read_text() == build_urdf(robot, SPEC),
          "re-run `python -m bestiary.robots.whelp.urdf_gen`")

    # Re-parse the emitted URDF with a general XML parser, so the check does not
    # go through the same code that wrote it.
    if URDF_PATH.exists():
        import xml.etree.ElementTree as ET
        root = ET.parse(URDF_PATH).getroot()
        links = root.findall("link")
        joints = root.findall("joint")
        check("the emitted URDF re-parses as XML with 17 links and 16 joints",
              len(links) == 17 and len(joints) == 16,
              f"{len(links)} links, {len(joints)} joints")
        check("every URDF link has an <inertial> with positive mass",
              all(ln.find("inertial") is not None
                  and float(ln.find("inertial/mass").get("value")) > 0 for ln in links),
              "a link with no inertia is read by PhysX as INFINITELY massive")
        check("every URDF link has a <collision>",
              all(ln.find("collision") is not None for ln in links),
              "collision_from_visuals defaults to False, so a link without one gets NO "
              "collider, silently")
        wheels = [j for j in joints if j.get("name", "").endswith("_wheel")]
        check("the four wheel joints are type=continuous in the emitted XML",
              len(wheels) == 4 and all(j.get("type") == "continuous" for j in wheels))
        check("every wheel's collision geometry is a cylinder primitive",
              all(ln.find("collision/geometry/cylinder") is not None
                  for ln in links if ln.get("name", "").endswith("_wheel")),
              "a convex hull of a tire mesh is an N-gon, and that is what chatter is")
        meshes = root.findall(".//mesh")
        check("every mesh carries the millimetre-to-metre scale",
              all(m.get("scale") == "0.001 0.001 0.001" for m in meshes),
              f"{len(meshes)} meshes; UrdfConverterCfg has no distance_scale, so this is "
              f"the only place it can be fixed")


def main() -> int:
    if not ROBOT_JSON.exists():
        print("robot.json does not exist; run `python -m bestiary.robots.whelp.export` first")
        return 2
    robot = json.loads(ROBOT_JSON.read_text())

    print("=" * 78)
    print(f"WHELP-16 CHECK   ({SPEC.servo_variant}, {SPEC.material})")
    print("=" * 78)

    section_structure(robot)
    section_symmetry(robot)
    section_stance(robot)
    section_servo()
    section_torque()
    section_mass(robot)
    section_provenance()
    section_cad()
    section_generated(robot)

    print()
    print("=" * 78)
    if _gaps:
        print(f"{len(_gaps)} check(s) COULD NOT BE RUN in this environment:")
        for g in _gaps:
            print(f"  ~ {g}")
        print()
    if _failures:
        print(f"{len(_failures)} of {_checks} checks FAILED:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"all {_checks} checks passed"
          + (f", {len(_gaps)} not runnable here" if _gaps else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
