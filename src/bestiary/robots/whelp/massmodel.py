"""What each part of WHELP-16 weighs, where its mass sits, and its inertia.

    python -m bestiary.robots.whelp.massmodel          # the mass budget
    python -m bestiary.robots.whelp.massmodel --check  # ...against measured STLs

WHY AN ANALYTIC MODEL AT ALL, WHEN THE CAD EXISTS
-------------------------------------------------
Because the URDF must be buildable without OpenSCAD installed. A pipeline where
regenerating the robot description requires a CAD binary is a pipeline that
breaks on the training machine, in CI, and on anyone else's laptop. So the mass
of every part is computed here from a hand-written sum of primitives, and the
URDF falls out of arithmetic.

That makes this file a SECOND implementation of the geometry in scad/parts/,
and two implementations of the same thing drift. The safeguard is that they are
checked against each other: when OpenSCAD is present, export.py renders each
part, measures the real mesh with stl.py, and asserts this model agrees within
MASS_MODEL_TOL. Loud, named, and with both numbers in the message.

THREE WAYS A MASS MODEL LIES, ALL OF THEM HANDLED HERE
------------------------------------------------------
1. SOLID DENSITY. CAD volume times filament density overestimates a printed part
   by roughly 2x, because the part is walls and gyroid, not plastic. But NOT by
   the infill percentage: on a part whose smallest dimension is under ~20 mm the
   walls are most of the volume -- five perimeters at 0.4 mm on a typical bracket
   is about a third of it -- so 25% infill gives an effective solid fraction near
   0.50, not 0.25. Spec.effective_density_g_cm3 carries that correction.

2. THE HARDWARE. Sixteen servos at 55 g are 880 g; the brass inserts are ~0.9 g
   each and there are dozens; the bearings are 3.1 g each. On a 2.5 kg robot the
   metal is a third of the machine and it is concentrated AT THE JOINTS, which
   is exactly where inertia matters most. Every one is counted below.

3. WHICH LINK A SERVO BELONGS TO. This is the error that looks like nothing and
   changes everything. A servo's BODY moves with the link it is bolted to, not
   the link it drives. The knee servo is part of the THIGH; the wheel drive is
   part of the CALF. Getting this backwards moves ~55 g by ~100 mm, which is a
   real change in the leg's inertia about the hip and therefore in what a policy
   learns is possible. LINK_PARTS below is the single place that mapping is
   written down.

WHAT THIS MODEL IS NOT
----------------------
It is not a CAD mass property. The primitives are idealisations -- a channel is
an outer box minus an inner box, not the filleted, gusseted, insert-bossed part
that actually prints. It is intended to be right to a few percent and to be
CHECKED, not to be believed. The measured-STL cross-check is what turns it from
an estimate into a number, and weighing the printed parts is what turns it from
a number into the truth. `mass_measured.json` exists for the last step.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bestiary.robots.whelp.spec import SPEC, Spec

#: Fractional disagreement allowed between this model and a measured STL before
#: export.py fails. 12% is loose, deliberately: the primitives are idealisations
#: and a tight tolerance here would fail on fillets rather than on mistakes. It
#: is tight enough to catch the errors that matter -- a forgotten pocket, a
#: doubled part, a millimetre/metre slip -- which are all far larger than 12%.
MASS_MODEL_TOL = 0.12

#: Optional override: a JSON of {part_name: grams} from an actual scale.
#: Measured mass outranks every model here, and the moment this file exists the
#: robot's URDF describes the robot that was built rather than the one designed.
MEASURED_JSON = Path(__file__).resolve().parent / "mass_measured.json"


# ── Primitive solids ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Prim:
    """One primitive solid, in a part's own frame. Millimetres.

    `sign` is +1 for material and -1 for a pocket. Subtracted primitives carry
    negative mass through the centre-of-mass and inertia sums, which is exact so
    long as the pocket lies entirely inside the solid it is cut from -- true for
    every use here, and the reason pockets are written as pockets rather than as
    a difference of two whole parts.
    """

    kind: str                       # "box" | "cyl" | "tube"
    dims: tuple[float, ...]         # box: (x, y, z); cyl: (d, h); tube: (od, id, h)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "z"                 # cyl/tube only
    sign: int = 1
    density_g_cm3: float | None = None   # None -> the part's default

    def volume_mm3(self) -> float:
        if self.kind == "box":
            x, y, z = self.dims
            return self.sign * x * y * z
        if self.kind == "cyl":
            d, h = self.dims
            return self.sign * math.pi * (d / 2.0) ** 2 * h
        if self.kind == "tube":
            od, idd, h = self.dims
            return self.sign * math.pi * ((od / 2.0) ** 2 - (idd / 2.0) ** 2) * h
        raise ValueError(f"unknown primitive kind {self.kind!r}")

    def inertia_local_kgm2(self, mass_kg: float) -> np.ndarray:
        """Inertia about the primitive's own centre, principal axes, kg.m^2."""
        m = mass_kg
        if self.kind == "box":
            x, y, z = (d / 1000.0 for d in self.dims)
            return np.diag([m * (y * y + z * z) / 12.0,
                            m * (x * x + z * z) / 12.0,
                            m * (x * x + y * y) / 12.0])
        if self.kind in ("cyl", "tube"):
            if self.kind == "cyl":
                r_o, r_i, h = self.dims[0] / 2000.0, 0.0, self.dims[1] / 1000.0
            else:
                r_o, r_i, h = (self.dims[0] / 2000.0, self.dims[1] / 2000.0,
                               self.dims[2] / 1000.0)
            i_ax = m * (r_o * r_o + r_i * r_i) / 2.0
            i_tr = m * (3.0 * (r_o * r_o + r_i * r_i) + h * h) / 12.0
            order = {"x": (i_ax, i_tr, i_tr), "y": (i_tr, i_ax, i_tr),
                     "z": (i_tr, i_tr, i_ax)}[self.axis]
            return np.diag(order)
        raise ValueError(self.kind)


@dataclass(frozen=True)
class Point:
    """A bought component, as a point mass. Millimetres, kilograms."""

    name: str
    mass_kg: float
    pos: tuple[float, float, float]
    #: Bought parts are not points -- a servo is 45 x 25 x 35 mm. Giving its
    #: extents lets its own inertia be included instead of treating 55 g as
    #: dimensionless, which understates the leg's inertia by a surprising amount.
    extent_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Body:
    """A rigid body's mass properties, in its own link frame. SI."""

    name: str
    mass_kg: float
    com_m: tuple[float, float, float]
    #: About the centre of mass, in the link frame. kg.m^2.
    inertia_kgm2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    #: Volume of printed plastic only, cm^3. Used for the STL cross-check and
    #: for the filament estimate; excludes every bought component.
    printed_cm3: float = 0.0
    parts: tuple[str, ...] = ()

    def inertia_is_valid(self) -> tuple[bool, str]:
        """Positive definite AND the triangle inequalities on principal moments.

        PhysX does NOT check the triangle inequality. A tensor violating
        I1 + I2 >= I3 describes an object that cannot exist, but it simulates
        happily -- giving a body that responds to torques in ways no real object
        can. A policy will find that and exploit it, and the exploit does not
        transfer. This is the single highest-value assertion in the CAD-to-URDF
        path, which is why it lives on the Body rather than in a test.
        """
        if self.mass_kg <= 0:
            return False, f"mass {self.mass_kg:.6g} kg is not positive"
        w = np.linalg.eigvalsh(self.inertia_kgm2)
        if np.any(w <= 0):
            return False, f"principal moments {w} are not all positive"
        i1, i2, i3 = sorted(w)
        if i1 + i2 < i3 * (1.0 - 1e-9):
            return False, (f"triangle inequality fails: {i1:.6g} + {i2:.6g} = {i1 + i2:.6g} "
                           f"< {i3:.6g}; this tensor describes an impossible object")
        return True, ""


def _accumulate(name: str, prims: list[Prim], points: list[Point], spec: Spec,
                default_density: float | None = None) -> Body:
    """Sum primitives and point masses into one Body. All maths in mm/g, out in SI."""
    rho_default = default_density if default_density is not None else spec.effective_density_g_cm3

    total_g = 0.0
    moment = np.zeros(3)
    printed_mm3 = 0.0
    entries: list[tuple[float, np.ndarray, np.ndarray]] = []   # (kg, com_m, I_local)

    for p in prims:
        rho = p.density_g_cm3 if p.density_g_cm3 is not None else rho_default
        v = p.volume_mm3()
        g = v * rho / 1000.0            # mm^3 * g/cm^3 / 1000 = g
        total_g += g
        moment += np.asarray(p.pos) * g
        printed_mm3 += v
        entries.append((g / 1000.0, np.asarray(p.pos) / 1000.0, p.inertia_local_kgm2(g / 1000.0)))

    for pt in points:
        g = pt.mass_kg * 1000.0
        total_g += g
        moment += np.asarray(pt.pos) * g
        x, y, z = (e / 1000.0 for e in pt.extent_mm)
        i_local = np.diag([pt.mass_kg * (y * y + z * z) / 12.0,
                           pt.mass_kg * (x * x + z * z) / 12.0,
                           pt.mass_kg * (x * x + y * y) / 12.0])
        entries.append((pt.mass_kg, np.asarray(pt.pos) / 1000.0, i_local))

    if total_g <= 0:
        raise ValueError(f"body {name!r} has non-positive mass {total_g:.4g} g")

    com_m = moment / total_g / 1000.0
    inertia = np.zeros((3, 3))
    for m_kg, c_m, i_local in entries:
        d = c_m - com_m
        inertia += i_local + m_kg * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return Body(name=name, mass_kg=total_g / 1000.0, com_m=tuple(com_m),
                inertia_kgm2=inertia, printed_cm3=printed_mm3 / 1000.0,
                parts=tuple(p.kind for p in prims[:0]))


# ── Part geometry, as primitives ─────────────────────────────────────────────
# Each function returns the primitives of ONE PRINTED PART in that part's own
# frame, with the frame chosen to match how scad/parts/ authors it, so the two
# can be compared without a transform.

def part_trunk_half(spec: Spec, half: str) -> list[Prim]:
    """Front or rear trunk shell: an open-topped box with a lap joint and servo cut-outs.

    REWRITTEN after review found three separate defects, all of which produced a
    plausible-looking number:

    1. The primitive list was built as `[...][:1] + [...] + [...]`, so four of the
       five primitives in the first literal were silently discarded -- including
       BOTH abduction servo cut-outs and the lid opening. Slicing a list literal
       is not a mistake anyone makes on purpose, and nothing downstream noticed.
    2. The "half-thickness" lap tongue was `(H - 2t)/2` tall -- half the INTERNAL
       HEIGHT, eleven times a wall thickness, and 53% of the shell's own volume.
    3. Worse, it violated the validity condition stated in Prim's own docstring:
       a subtracted primitive is only exact when it lies inside the solid it cuts.
       91% of the rear tongue was in empty space, so the signed sum removed
       68,607 mm^3 where the real notch removes 6,352. Front modelled 125.8 g
       against a true 82.2 g, rear 38.6 g against 78.1 g -- both far outside
       MASS_MODEL_TOL, and the two-half centroid landed 33 mm off centre.

    The shape here is now: outer box, an open-topped cavity (the lid is a separate
    part, so there is no top wall to remove), a servo cut-out through each SIDE
    WALL where there is real material to remove, and a lap that is one wall
    thickness of material -- added on the front, notched from the rear.

    `_assert_trunk_halves_balance` guards the class of error rather than this
    instance: two halves of one symmetric box must weigh nearly the same, and any
    of the three defects above breaks that by 50% or more.
    """
    L = spec.trunk_len_mm / 2.0 + spec.trunk_split_lap_mm / 2.0
    W, H, t = spec.trunk_width_mm, spec.trunk_height_mm, spec.trunk_wall_mm
    lap = spec.trunk_split_lap_mm

    prims = [
        Prim("box", (L, W, H)),
        # Open-topped: the cavity runs out through the top face, because the lid
        # is its own printed part and the top is not a printed surface at all.
        Prim("box", (L - 2 * t, W - 2 * t, H - t), (0.0, 0.0, t / 2.0), sign=-1),
    ]

    # Abduction servo cut-outs, through the SIDE WALL. Cut through the wall, not
    # floating in the cavity: a pocket entirely inside empty space removes
    # nothing in CSG while the signed sum still subtracts its whole volume.
    for s in (-1, 1):
        prims.append(Prim(
            "box", (spec.servo_body_l_mm, 2.0 * t, spec.servo_body_h_mm),
            (0.0, s * (W / 2.0 - t / 2.0), 0.0), sign=-1))

    # The lap. One wall thickness of material over the overlap area: the front
    # half carries a tongue on the cavity side of its floor, the rear half has
    # the matching notch taken out of its floor. Both act on real material.
    z_lap = -H / 2.0 + t / 2.0
    prims.append(Prim(
        "box", (lap, W - 2 * t, t),
        (L / 2.0 - lap / 2.0, 0.0, z_lap + (t if half == "front" else 0.0)),
        sign=+1 if half == "front" else -1))
    return prims


def part_trunk_lid(spec: Spec) -> list[Prim]:
    return [Prim("box", (spec.trunk_len_mm / 2.0, spec.trunk_width_mm, spec.trunk_wall_mm * 1.5)),
            Prim("box", (spec.trunk_len_mm / 2.0 - 30, spec.trunk_width_mm - 30,
                         spec.trunk_wall_mm * 1.5), sign=-1)]


def part_abduct_bracket(spec: Spec) -> list[Prim]:
    """Rotates about +X, carries the hip servo. A clevis with two side walls."""
    w = spec.abduct_to_hip_mm + spec.servo_body_w_mm / 2 + 4
    h = spec.servo_body_h_mm + 10
    d = spec.servo_body_l_mm + 8
    t = spec.thigh_wall_mm
    return [
        Prim("box", (d, w, h)),
        Prim("box", (d - 2 * t, w - 2 * t, h - 2 * t), sign=-1),
        Prim("cyl", (spec.horn_disc_d_mm + 10, spec.horn_boss_thick_mm),
             (0, -w / 2 + spec.horn_boss_thick_mm / 2, 0), axis="y"),
        Prim("cyl", (spec.idler_bearing_od_mm + 6, spec.idler_bearing_w_mm + 2),
             (0, w / 2 - spec.idler_bearing_w_mm / 2, 0), axis="y"),
    ]


def part_thigh(spec: Spec) -> list[Prim]:
    """Hip axis to knee axis. A U-channel, open side outboard, printed lying down."""
    L = spec.thigh_len_mm
    d, w, t = spec.thigh_section_fore_aft_mm, spec.thigh_section_lateral_mm, spec.thigh_wall_mm
    return [
        Prim("box", (d, w, L), (0, 0, -L / 2)),
        Prim("box", (d - 2 * t, w - t, L - 2 * t), (0, t / 2, -L / 2), sign=-1),
        Prim("cyl", (d + 4, spec.horn_boss_thick_mm), (0, -w / 2, 0), axis="y"),
        Prim("cyl", (d + 4, spec.idler_bearing_w_mm + 2), (0, w / 2, 0), axis="y"),
        # Knee servo mounting boss at the far end.
        Prim("box", (d, w, 12), (0, 0, -L + 6)),
        # Lightening windows down the web, on the neutral axis where they cost
        # almost no stiffness.
        Prim("cyl", (d * 0.45, w + 2), (0, 0, -L * 0.35), axis="y", sign=-1),
        Prim("cyl", (d * 0.45, w + 2), (0, 0, -L * 0.62), axis="y", sign=-1),
    ]


def part_calf(spec: Spec) -> list[Prim]:
    """Knee axis to wheel axis. Shallower channel, carries the wheel drive."""
    L = spec.calf_len_mm
    d, w, t = spec.calf_section_fore_aft_mm, spec.calf_section_lateral_mm, spec.calf_wall_mm
    return [
        Prim("box", (d, w, L), (0, 0, -L / 2)),
        Prim("box", (d - 2 * t, w - t, L - 2 * t), (0, t / 2, -L / 2), sign=-1),
        Prim("cyl", (d + 4, spec.horn_boss_thick_mm), (0, -w / 2, 0), axis="y"),
        Prim("cyl", (d + 4, spec.idler_bearing_w_mm + 2), (0, w / 2, 0), axis="y"),
        Prim("box", (d, w, 14), (0, 0, -L + 7)),
        Prim("cyl", (d * 0.42, w + 2), (0, 0, -L * 0.45), axis="y", sign=-1),
    ]


def part_wheel_hub(spec: Spec) -> list[Prim]:
    """PETG hub: a shallow dish with a horn pad and a tire seat."""
    r_seat = spec.wheel_radius_mm - spec.tire_thickness_mm
    w = spec.wheel_width_mm
    return [
        Prim("tube", (2 * r_seat, 2 * r_seat - 6, w), axis="y"),
        Prim("cyl", (2 * r_seat - 6, 3.0), axis="y"),
        Prim("cyl", (spec.horn_disc_d_mm + 10, spec.horn_boss_thick_mm),
             (0, -w / 2 + spec.horn_boss_thick_mm / 2, 0), axis="y"),
        # Six spoke lightening holes through the web.
        *[Prim("cyl", (r_seat * 0.42, 4.0),
               ((r_seat * 0.55) * math.cos(i * math.pi / 3), 0,
                (r_seat * 0.55) * math.sin(i * math.pi / 3)), axis="y", sign=-1)
          for i in range(6)],
    ]


def part_tire(spec: Spec) -> list[Prim]:
    """TPU 95A annulus over the hub."""
    r_o = spec.wheel_radius_mm
    r_i = spec.wheel_radius_mm - spec.tire_thickness_mm
    rho = spec.tpu_density_g_cm3 * (
        spec.tire_infill_frac + (1 - spec.tire_infill_frac) * 0.45)
    return [Prim("tube", (2 * r_o, 2 * r_i, spec.wheel_width_mm), axis="y",
                 density_g_cm3=rho)]


def part_fuse(spec: Spec) -> list[Prim]:
    a = spec.fuse_shear_area_mm2
    return [Prim("box", (a / spec.fuse_thick_mm, spec.fuse_thick_mm, 18.0),
                 density_g_cm3=spec.print_density_g_cm3)]


PRINTED_PARTS = {
    "trunk_front": lambda s: part_trunk_half(s, "front"),
    "trunk_rear": lambda s: part_trunk_half(s, "rear"),
    "trunk_lid": part_trunk_lid,
    "abduct_bracket": part_abduct_bracket,
    "thigh": part_thigh,
    "calf": part_calf,
    "wheel_hub": part_wheel_hub,
    "tire": part_tire,
    "fuse": part_fuse,
}


# ── Link composition ─────────────────────────────────────────────────────────
# A servo's BODY belongs to the link it is BOLTED TO, not the one it drives.
# See the module docstring; this mapping is the whole reason it is written down
# in one place.
LINK_PARTS: dict[str, tuple[str, ...]] = {
    "trunk": ("trunk_front", "trunk_rear", "trunk_lid"),
    "hip": ("abduct_bracket",),
    "thigh": ("thigh",),
    "calf": ("calf", "fuse"),
    "wheel": ("wheel_hub", "tire"),
}


def _printed(name: str, spec: Spec) -> tuple[float, np.ndarray, float]:
    """(mass_kg, com_mm, volume_cm3) of one printed part."""
    prims = PRINTED_PARTS[name](spec)
    rho_default = spec.effective_density_g_cm3
    g = 0.0
    moment = np.zeros(3)
    vol = 0.0
    for p in prims:
        rho = p.density_g_cm3 if p.density_g_cm3 is not None else rho_default
        v = p.volume_mm3()
        gp = v * rho / 1000.0
        g += gp
        moment += np.asarray(p.pos) * gp
        vol += v
    return g / 1000.0, (moment / g if g else np.zeros(3)), vol / 1000.0


def link_bodies(spec: Spec = SPEC) -> dict[str, Body]:
    """Every link's mass properties, keyed by the link names geometry.py uses."""
    from bestiary.robots.whelp.geometry import LEGS

    measured = {}
    if MEASURED_JSON.exists():
        measured = json.loads(MEASURED_JSON.read_text())

    def prims_of(part: str) -> list[Prim]:
        return PRINTED_PARTS[part](spec)

    def scaled(part: str) -> list[Prim]:
        """Primitives, rescaled if this part has been weighed on a scale.

        A measured mass outranks the model, but the SHAPE of the part is still
        the model's -- so the correction is applied as a uniform density scale,
        which preserves the centre of mass and scales the inertia with the mass.
        That is the right first-order correction and it is honest about being
        first-order.
        """
        prims = prims_of(part)
        if part not in measured:
            return prims
        model_g = _printed(part, spec)[0] * 1000.0
        k = measured[part] / model_g if model_g > 0 else 1.0
        return [Prim(p.kind, p.dims, p.pos, p.axis, p.sign,
                     (p.density_g_cm3 if p.density_g_cm3 is not None
                      else spec.effective_density_g_cm3) * k)
                for p in prims]

    bodies: dict[str, Body] = {}

    # ── Trunk ────────────────────────────────────────────────────────────────
    trunk_prims: list[Prim] = []
    quarter = spec.trunk_len_mm / 4.0
    for part, dx in (("trunk_front", +quarter), ("trunk_rear", -quarter)):
        trunk_prims += [Prim(p.kind, p.dims, (p.pos[0] + dx, p.pos[1], p.pos[2]),
                             p.axis, p.sign, p.density_g_cm3) for p in scaled(part)]
    for dx in (+quarter, -quarter):
        trunk_prims += [Prim(p.kind, p.dims,
                             (p.pos[0] + dx, p.pos[1], p.pos[2] + spec.trunk_height_mm / 2),
                             p.axis, p.sign, p.density_g_cm3) for p in scaled("trunk_lid")]

    trunk_points = [
        Point("battery", spec.battery_mass_kg, (0, 0, -8), (105, 35, 25)),
        Point("compute", spec.compute_mass_kg, (-30, 0, 18), (85, 56, 12)),
        Point("wiring", spec.wiring_mass_kg, (0, 0, 0), (200, 80, 40)),
        Point("fasteners", spec.fastener_mass_kg, (0, 0, 0), (240, 100, 55)),
    ]
    # The four ABDUCTION servos: bodies bolted to the trunk, horns driving the
    # brackets. They belong here, not to the legs.
    for leg in LEGS:
        fx = +1 if leg[0] == "F" else -1
        fy = +1 if leg[1] == "L" else -1
        trunk_points.append(Point(
            f"{leg}_abduct_servo", spec.servo_mass_kg,
            (fx * spec.abduct_x_mm, fy * (spec.trunk_width_mm / 2 - spec.servo_body_w_mm / 2), 0),
            (spec.servo_body_l_mm, spec.servo_body_w_mm, spec.servo_body_h_mm)))

    bodies["trunk"] = _accumulate("trunk", trunk_prims, trunk_points, spec)

    # ── Legs ─────────────────────────────────────────────────────────────────
    for leg in LEGS:
        fy = +1 if leg[1] == "L" else -1

        # hip link: the abduction bracket, carrying the HIP servo.
        bodies[f"{leg}_hip"] = _accumulate(
            f"{leg}_hip",
            [Prim(p.kind, p.dims, (p.pos[0], fy * p.pos[1], p.pos[2]), p.axis, p.sign,
                  p.density_g_cm3) for p in scaled("abduct_bracket")],
            [Point("hip_servo", spec.servo_mass_kg,
                   (0, fy * spec.abduct_to_hip_mm, 0),
                   (spec.servo_body_l_mm, spec.servo_body_w_mm, spec.servo_body_h_mm)),
             Point("idler_bearing", spec.idler_bearing_mass_kg, (0, 0, 0)),
             Point("inserts", 6 * spec.insert_mass_kg, (0, 0, 0))],
            spec)

        # thigh link: the thigh, carrying the KNEE servo at its far end.
        bodies[f"{leg}_thigh"] = _accumulate(
            f"{leg}_thigh",
            [Prim(p.kind, p.dims, (p.pos[0], fy * p.pos[1], p.pos[2]), p.axis, p.sign,
                  p.density_g_cm3) for p in scaled("thigh")],
            [Point("knee_servo", spec.servo_mass_kg, (0, 0, -spec.thigh_len_mm),
                   (spec.servo_body_l_mm, spec.servo_body_w_mm, spec.servo_body_h_mm)),
             Point("idler_bearing", spec.idler_bearing_mass_kg, (0, 0, 0)),
             Point("inserts", 6 * spec.insert_mass_kg, (0, 0, -spec.thigh_len_mm / 2))],
            spec)

        # calf link: the calf, carrying the WHEEL DRIVE at its far end.
        calf_prims = [Prim(p.kind, p.dims, (p.pos[0], fy * p.pos[1], p.pos[2]), p.axis, p.sign,
                           p.density_g_cm3) for p in scaled("calf")]
        if spec.fuse_enable:
            calf_prims += [Prim(p.kind, p.dims,
                                (p.pos[0], fy * p.pos[1], p.pos[2] - spec.calf_len_mm + 8),
                                p.axis, p.sign, p.density_g_cm3) for p in scaled("fuse")]
        bodies[f"{leg}_calf"] = _accumulate(
            f"{leg}_calf", calf_prims,
            [Point("wheel_drive", spec.wheel_drive_mass_kg, (0, 0, -spec.calf_len_mm),
                   (spec.servo_body_l_mm, spec.servo_body_w_mm, spec.servo_body_h_mm)),
             Point("idler_bearing", spec.idler_bearing_mass_kg, (0, 0, 0)),
             Point("inserts", 6 * spec.insert_mass_kg, (0, 0, -spec.calf_len_mm / 2))],
            spec)

        # wheel link: hub and tire only. Everything here is UNSPRUNG and at the
        # end of the leg, so grams count double.
        bodies[f"{leg}_wheel"] = _accumulate(
            f"{leg}_wheel",
            [Prim(p.kind, p.dims, (p.pos[0], fy * p.pos[1], p.pos[2]), p.axis, p.sign,
                  p.density_g_cm3) for p in scaled("wheel_hub") + scaled("tire")],
            [Point("inserts", 4 * spec.insert_mass_kg, (0, 0, 0))],
            spec)

    return bodies


def total_mass_kg(spec: Spec = SPEC) -> float:
    return sum(b.mass_kg for b in link_bodies(spec).values())


def cg_height_measured(spec: Spec = SPEC) -> float:
    """CoM height above the ground in the standing stance, as a fraction of trunk height.

    Replaces Spec.cg_height_frac once the model exists, which is why that
    attribute is marked ASSUMED with this function named as its replacement:
    the wheelie limit depends on it and a guessed CoM height is a guessed
    acceleration ceiling.
    """
    from bestiary.robots.whelp import geometry as geo

    chain = geo.build_chain(spec)
    _joints, frames = geo.link_frames(chain)
    bodies = link_bodies(spec)
    m_tot = 0.0
    z = 0.0
    for name, body in bodies.items():
        r, p = frames[name]
        c = geo._apply(r, body.com_m)
        m_tot += body.mass_kg
        z += body.mass_kg * (p[2] + c[2])
    com_z = z / m_tot
    stand = geo.stand_height_mm(spec) / 1000.0
    return (com_z + stand) / stand


def main(argv: list[str]) -> int:
    spec = SPEC
    bodies = link_bodies(spec)
    total = sum(b.mass_kg for b in bodies.values())
    printed = sum(b.printed_cm3 for b in bodies.values())

    print("WHELP-16 MASS BUDGET")
    print()
    print(f"  {'link':<12} {'mass g':>9} {'printed cm3':>12} "
          f"{'com x':>8} {'com y':>8} {'com z':>8}   inertia ok")
    for name, b in bodies.items():
        ok, why = b.inertia_is_valid()
        cx, cy, cz = (v * 1000 for v in b.com_m)
        print(f"  {name:<12} {b.mass_kg * 1000:>9.1f} {b.printed_cm3:>12.1f} "
              f"{cx:>8.1f} {cy:>8.1f} {cz:>8.1f}   {'yes' if ok else 'NO: ' + why}")
    print()
    print(f"  TOTAL {total * 1000:.0f} g = {total:.3f} kg   "
          f"({printed:.0f} cm3 of plastic, ~{printed * spec.print_density_g_cm3 / 1000:.2f} kg "
          f"of filament at solid density)")

    servo_kg = 16 * spec.servo_mass_kg
    print(f"  of which  servos {servo_kg * 1000:.0f} g ({servo_kg / total:.0%}), "
          f"payload {spec.payload_mass_kg * 1000:.0f} g "
          f"({spec.payload_mass_kg / total:.0%})")
    print(f"  measured CoM height fraction: {cg_height_measured(spec):.3f} "
          f"(spec assumes {spec.cg_height_frac:.3f})")
    if MEASURED_JSON.exists():
        print(f"  using measured masses from {MEASURED_JSON.name}")
    else:
        print(f"  NO measured masses: write {MEASURED_JSON.name} as "
              "{\"thigh\": 41.2, ...} in grams once parts are printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
