"""Assertions on the Hound MJCF-to-USD port. The oracle for `hound_usd`/`hound_cfg`.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.check_hound

Exit status is 0 on all-pass and 1 on any failure, so it can gate a launch.
Takes about a minute, almost all of it Kit starting up.


WHY THIS EXISTS
---------------
`robots/hound/check.py` is the MuJoCo robot's oracle -- 38 assertions that the
machine has not moved. It cannot see this port at all: a different solver reads a
different file, produced by a converter nobody in this repository wrote, and
every one of the failure modes below leaves that file loading cleanly.

  * A stale USD. The `.usda` is generated and committed, so it can silently
    describe last week's robot while `build.py` describes this week's.
  * The wheel drive. A position gain on the wheel joint works for about 3.2
    revolutions and then stops, because PhysX refuses a drive target outside
    +/-2*pi. It is not an error, it is a joint that quietly no longer tracks --
    and the joint arrives with +/-FLT_MAX limits, which is exactly the condition
    that makes it look fine.
  * The stance springs. They are a labelled MuJoCo crutch and must not become
    drive gains. They travel INTO the asset (in a `mujoco` USD variant) and the
    thing that keeps them out of the physics is a variant selection, which is a
    one-token change away from being wrong.
  * Degrees. The importer writes angular limits in degrees. Every joint limit in
    this machine would be wrong by 57.2958 and the robot would still stand.
  * A ground plane. The conversion input has its world stripped for a reason; if
    that ever stops happening, every environment gets an 80 m collision plane
    cloned under it and the terrain is no longer what the robot stands on.
  * Mass. Nothing downstream notices a mass that arrived scaled.
  * STATIC FRICTION, which the importer does not write at all. MuJoCo has one
    sliding coefficient; USD has two, and the missing one defaults to zero. A
    robot with no grip at rest looks exactly like a robot with an unstable
    stance, and this machine's card documents a real splits failure from another
    cause -- so the wrong explanation arrives pre-confirmed.

So this suite reads three things and compares them: `Spec` (the source of
truth), `assets/hound16pd.xml` (the committed MuJoCo model), and the loaded
articulation (what the solver actually got). Anything only one of the three
believes is a failure.

AND THE REWARD, WHICH IS A FOURTH KIND OF SILENT FAILURE
--------------------------------------------------------
Section 4 checks `Bestiary-Desert-Hound-v0`'s reward table, because the ways it
goes wrong on this body all construct, step and train:

  * A CONTACT-TIMING TERM ON A WHEEL. Re-scoping ANYmal's `feet_air_time` from
    `.*FOOT` onto `.*_wheel` fixes the ValueError and keeps the incentive, and the
    incentive on a driven hub wheel is to leave the ground. A reward that pays for
    hopping trains perfectly well.
  * A JOINT PENALTY THAT REACHES THE WHEELS. The wheel drive is derived to
    accelerate the wheel inside one control period, so an acceleration penalty at
    the leg weight bills the drive for working. `research/learnings/011` measured
    that shape: control cost was 105.5% of the gap and driving did not pay.
  * A VELOCITY LIMIT ON THE FIELD THAT IS IGNORED. `ImplicitActuatorCfg` accepts
    `velocity_limit`, warns, and discards it; only `velocity_limit_sim` reaches the
    solver. The wrong field is a config that reads correctly and changes nothing.

These are structural checks, not name checks: they resolve each term's regex
against the loaded articulation's real body and joint names and then read the
reward function's own source to decide what it charges for. A future term under a
new name is caught for the same reason the current one is.

AND THE TASK VARIANTS, WHICH NEED NO SIMULATOR AT ALL
------------------------------------------------------
Section 5 pins the two tasks built on top of the desert config. Both groups of
assertions read whole-config `to_dict()` dumps rather than the lines somebody
remembered to write, because "one variable moved" is a claim about a config and
not about a diff a human eyeballed.

  * `Bestiary-ForwardV5-Hound-v0` — exactly one reward term (`v_x`, weight 1.0),
    the v5 ground of `research/decisions/0007` measured from the committed bytes,
    and a whole-config diff limited to `rewards` and the two sub-terrain keys
    inside `scene`.
  * `Bestiary-Overnight-Hound-v0` — the COMMANDED run: four reward terms at the
    desert table's own weights, the dead-zone sampler on the ±1.5 / ±0.6 / ±1.5
    envelope with heading mode off, the arc-corrected terrain curriculum, and a
    diff that is four sections against the desert task and THREE against the
    forward probe, whose ground it must match byte for byte. It also prices the
    thing that ended arm 1: standing's expected share of drive-cell income,
    flagged past 30%.

Those configs construct before the simulation app exists, so that group runs at a
desk with no GPU — which is the only part of this oracle a session without the
card can still hold.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import sys
import traceback
from typing import Callable

import numpy as np
from isaaclab.app import AppLauncher

#: RELATIVE tolerance for a mass. The USD stores masses as float32, so the
#: trunk's 6.921 kg comes back as 6.921000481 -- a 7e-8 relative error, which is
#: the float32 round trip and nothing else. Four times float32's epsilon
#: (4 x 1.1920929e-7 = 4.77e-7) leaves room for that without admitting any real
#: scaling bug: those are factors, not parts per million. Chosen after 1e-9
#: absolute failed on exactly this, which is worth leaving on the record.
MASS_RTOL = 4.0 * float(np.finfo(np.float32).eps)

#: Absolute tolerance for a joint angle, in radians. 1e-6 is well inside the
#: float32 round trip through degrees and back that the importer performs on
#: every limit; the smallest error that would matter physically is ~1e-3.
ANGLE_TOL_RAD = 1e-6

#: Absolute tolerance for a gain or a torque, in the joint's own units.
GAIN_TOL = 1e-6

#: Below this, a joint limit is "not a limit", in radians.
#:
#: An unlimited MuJoCo hinge does NOT arrive with the limit attribute absent; it
#: arrives with a huge finite one, and how huge depends on the solver: PhysX
#: reports +/-3.4028235e38 (FLT_MAX) and Newton/MJWarp reports +/-1e10 for the
#: same asset, both measured. So "unlimited" is a magnitude test, and the
#: threshold has to sit below the smaller of the two. 1e6 rad is 159,155
#: revolutions -- no limit anyone authored, and comfortably past the +/-2*pi that
#: a position drive would actually break at.
UNLIMITED_RAD = 1e6

#: Above this, a wheel's joint VELOCITY limit is "not a limit", in rad/s.
#:
#: The wheel is the one part of this machine with no vendor sheet behind it, so
#: `hound_cfg` deliberately sets no `velocity_limit_sim` on it and the converted
#: USD's own default (measured: 1.7e4 rad/s) stands. 1e3 rad/s is 94x the largest
#: speed the drive can command (`wheel_action_scale()` = 10.665 rad/s), so nothing
#: below it can be an artifact of the drive -- anything below it is a number
#: somebody chose, which is what this threshold exists to catch.
WHEEL_VEL_UNLIMITED_RAD_S = 1e3

#: Contact-sensor fields that carry contact TIMING rather than contact force.
#:
#: A reward whose source reads one of these pays for the DURATION of contact or of
#: flight. On a driven hub wheel -- which is supposed to stay in rolling contact --
#: that is a bounty on leaving the ground, i.e. on hopping. These are the names
#: `mdp.feet_air_time` and its relatives index on `ContactSensorData`.
CONTACT_TIMING_FIELDS = (
    "current_air_time",
    "last_air_time",
    "current_contact_time",
    "last_contact_time",
)

#: Articulation fields that make a reward term a per-joint effort penalty.
#:
#: A term reading any of these charges a joint for moving or for pushing. On a
#: driven wheel that is charging the machine for rolling, which is the arithmetic
#: `research/learnings/011` measured: control cost was 105.5% of the entire
#: policy-versus-control gap, so driving did not pay.
JOINT_EFFORT_FIELDS = ("applied_torque", "joint_vel", "joint_acc", "joint_pos")

#: The one reward term allowed to include a wheel joint in a per-joint effort
#: penalty: the 100x-weaker acceleration term `research/decisions/0004` Part B
#: takes from `fan-ziqi/robot_lab`'s Go2-W config.
WHEEL_EFFORT_TERM = "dof_acc_wheel_l2"

#: Wheel angular acceleration used to price the wheel-acceleration penalty, rad/s^2.
#:
#: DERIVED, not assumed: `hound_cfg.wheel_velocity_gain()` is chosen so the drive's
#: time constant is exactly one control period, so a step to the largest commandable
#: speed is `wheel_action_scale() / CONTROL_DT_S` of acceleration. That is the drive
#: working as designed, which is the whole reason the wheel term must be cheap.
#: Computed in `check_reward_budget_against_011_and_015`, not typed.
#:
#: Fraction of achievable tracking income the four wheels may cost at that
#: acceleration. 1% is generous: at robot_lab's -2.5e-9 the true figure is ~0.3%,
#: and at the LEG weight of -2.5e-7 it is ~27%, so this bound separates the two
#: without pretending to know where between them the right answer is.
WHEEL_ACC_BUDGET_FRACTION = 0.01

#: Fraction of achievable tracking income above which the penalty budget is
#: reported as a flag rather than a number.
#:
#: 30% is the threshold this cycle was asked to flag at, and it sits under both
#: measured failures it is meant to anticipate: `learnings/011` at 105.5% of the
#: gap (driving did not pay) and `learnings/015` at 62.3% of income (driving paid,
#: but one fixed trot was the income-optimal way to do it).
PENALTY_BUDGET_FLAG_FRACTION = 0.30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    AppLauncher.add_app_launcher_args(parser)
    # An oracle has nothing to look at. `--viz none` is 3.0's headless.
    parser.set_defaults(visualizer=["none"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers. Everything below runs with the app up.
# ---------------------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    """Read an Isaac Lab 3.0 `.data.*` field as numpy, whatever it happens to be.

    3.0 returns `warp.array` from some of these fields and a torch tensor from
    others, and the torch ones live on the GPU. `.numpy()` exists on a CUDA
    tensor and RAISES, so the torch branch has to be tested first -- probing for
    a `numpy` attribute finds the wrong one.
    """
    if hasattr(x, "detach"):  # torch.Tensor, possibly on cuda
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):  # warp.array
        return np.asarray(x.numpy())
    return np.asarray(x)


def mjcf_geom_masses(xml: str) -> dict[str, float]:
    """Body name -> mass, read out of a Bestiary MJCF's `<geom ... mass=...>`.

    Reads the committed model rather than `Spec` on purpose: `Spec` is already
    the input to the conversion, so comparing the USD against it would only
    prove the converter is self-consistent. The committed XML is the artifact a
    MuJoCo run actually loaded.

    Every massive geom in `hound16pd.xml` is named `<body>_geom`, which is what
    maps a mass onto a body without parsing the tree.
    """
    masses: dict[str, float] = {}
    for tag in re.finditer(r"<geom\b[^>]*?/>", xml, re.S):
        text = tag.group(0)
        name = re.search(r'\bname="([^"]+)"', text)
        mass = re.search(r'\bmass="([^"]+)"', text)
        if name is None or mass is None:
            continue  # decorative geom: no mass attribute, by design
        if not name.group(1).endswith("_geom"):
            raise AssertionError(
                f"geom {name.group(1)!r} carries a mass but is not named "
                "'<body>_geom', so this parser cannot attribute it to a body"
            )
        masses[name.group(1)[: -len("_geom")]] = float(mass.group(1))
    return masses


def _usd_stage():
    """The converted asset, composed exactly as Isaac Lab composes it."""
    from pxr import Usd

    from bestiary import paths

    if not paths.HOUND_ISAAC_USD.is_file():
        raise AssertionError(
            f"{paths.HOUND_ISAAC_USD} does not exist. Generate it with\n"
            "    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.hound_usd"
        )
    stage = Usd.Stage.Open(str(paths.HOUND_ISAAC_USD))
    if stage is None:
        raise AssertionError(f"USD failed to open {paths.HOUND_ISAAC_USD}")
    return stage


# ---------------------------------------------------------------------------
# 1. The generated files still describe the machine `build.py` describes.
# ---------------------------------------------------------------------------
def check_conversion_input_is_current() -> None:
    """The committed conversion MJCF is byte-for-byte what `build.py` emits now.

    This is the load-bearing staleness check. A change to `Spec` -- a length, a
    mass, a gain -- moves the MuJoCo models the moment anyone re-runs
    `robots.hound.build`, and leaves this file and the USD behind it describing
    the old machine with nothing raising.
    """
    from bestiary import paths
    from bestiary.isaac.hound_usd import conversion_mjcf
    from bestiary.robots.hound.build import SPEC

    if not paths.HOUND_ISAAC_MJCF.is_file():
        raise AssertionError(f"{paths.HOUND_ISAAC_MJCF} does not exist; run hound_usd")
    on_disk = paths.HOUND_ISAAC_MJCF.read_text()
    fresh = conversion_mjcf(SPEC)
    if on_disk != fresh:
        raise AssertionError(
            f"{paths.HOUND_ISAAC_MJCF} is {len(on_disk)} chars and build.py now "
            f"emits {len(fresh)}; they differ, so the committed USD was converted "
            "from a different robot. Re-run:\n"
            "    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.hound_usd"
        )


def check_conversion_input_is_the_pd_robot() -> None:
    """The conversion input's robot is `assets/hound16pd.xml`'s robot.

    Exactly two deltas are permitted and both are asserted rather than trusted:
    the trunk is authored at z = 0 instead of `Spec.stand_z`, and there is no
    world. Anything else -- a different stance, a different limit, an extra geom
    -- means the Isaac Lab port and the MuJoCo runs are two machines.
    """
    from bestiary import paths
    from bestiary.isaac.hound_usd import CONVERSION_TRUNK_Z_M
    from bestiary.robots.hound.build import SPEC, robot_xml

    mujoco_subtree = robot_xml(SPEC)
    isaac_subtree = robot_xml(SPEC, trunk_z=CONVERSION_TRUNK_Z_M)

    pd_xml = paths.HOUND_PD_XML.read_text()
    if mujoco_subtree not in pd_xml:
        raise AssertionError(
            f"the robot subtree build.py emits is not present verbatim in "
            f"{paths.HOUND_PD_XML}. That file is stale -- re-run "
            "`venv/bin/python -m bestiary.robots.hound.build`."
        )
    isaac_mjcf = paths.HOUND_ISAAC_MJCF.read_text()
    if isaac_subtree not in isaac_mjcf:
        raise AssertionError(
            f"the conversion input {paths.HOUND_ISAAC_MJCF} does not contain the "
            "robot subtree with the trunk authored at "
            f"z = {CONVERSION_TRUNK_Z_M}"
        )
    # ...and the two subtrees differ ONLY in that one number.
    if mujoco_subtree.replace(
        f'pos="0 0 {SPEC.stand_z:.4f}"', f'pos="0 0 {CONVERSION_TRUNK_Z_M:.4f}"'
    ) != isaac_subtree:
        raise AssertionError(
            "the Isaac Lab conversion input's robot differs from "
            f"{paths.HOUND_PD_XML}'s by more than the trunk height. The port and "
            "the MuJoCo runs are no longer the same machine."
        )


# ---------------------------------------------------------------------------
# 2. What the converter produced, read off the USD.
# ---------------------------------------------------------------------------
def check_usd_has_no_ground() -> None:
    """No plane, no ground, no floor anywhere in the asset.

    `assets/hound16pd.xml` has an 80 x 80 m `<geom type="plane">` and the
    importer converts it to a `Plane` prim with a collider. Isaac Lab clones a
    robot asset once per environment, so one surviving plane is N planes lying on
    top of the terrain -- and the robot would stand on the wrong one without
    anything raising.
    """
    stage = _usd_stage()
    offenders = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetTypeName() == "Plane" or prim.GetName().lower() in {"floor", "ground"}
    ]
    if offenders:
        raise AssertionError(
            f"the converted asset contains ground geometry: {offenders}. The "
            "conversion input is supposed to have no world at all -- see "
            "hound_usd.py's docstring, delta 1."
        )


def check_usd_selects_the_physx_variant() -> None:
    """The asset's `Physics` variant selection is `physx`, not `mujoco`.

    This one line is what keeps the stance springs out of the solver. The
    importer files MuJoCo joint attributes (`mjc:stiffness`, `mjc:springref`)
    into a `mujoco` variant and the PhysX drives into a `physx` variant; select
    the wrong one and the twelve leg joints gain a spring the MuJoCo model calls
    a crutch, on top of the PD drive that replaced it.
    """
    stage = _usd_stage()
    prim = stage.GetDefaultPrim()
    if not prim:
        raise AssertionError("the converted asset has no default prim")
    variant_sets = prim.GetVariantSets()
    if not variant_sets.HasVariantSet("Physics"):
        raise AssertionError(
            f"{prim.GetPath()} has no 'Physics' variant set; the importer's "
            "layout changed and this check no longer describes the asset"
        )
    selection = variant_sets.GetVariantSet("Physics").GetVariantSelection()
    if selection != "physx":
        raise AssertionError(
            f"the asset selects the {selection!r} Physics variant, not 'physx'. "
            "The 'mujoco' variant carries the stance springs."
        )


def check_no_springs_reach_the_solver() -> None:
    """No `mjc:springref` or `mjc:stiffness` in the composed asset...

    ...and, more to the point, the twelve leg drive stiffnesses are the
    `<position>` actuators' kp (60 / 80 / 90 N*m/rad) and not the spring rates
    `Spec.stiffness()` derives (12.12 / 18.58 / 32.77 N*m/rad). The second half
    is the one that matters: a spring rate arriving as a drive gain is a number
    in the right field with the wrong meaning.
    """
    from pxr import Usd

    from bestiary.robots.hound.build import SPEC

    stage = _usd_stage()
    leaked = [
        f"{prim.GetPath()}.{attr.GetName()}"
        for prim in stage.Traverse()
        for attr in prim.GetAttributes()
        if attr.GetName().startswith("mjc:spring")
    ]
    if leaked:
        raise AssertionError(
            f"MuJoCo spring attributes are visible in the composed asset: "
            f"{leaked[:4]} ({len(leaked)} total). The physx variant is supposed "
            "to exclude them."
        )

    spring_k = SPEC.stiffness()
    expected_kp = {"abduct": SPEC.kp_abduct, "hip": SPEC.kp_hip, "knee": SPEC.kp_knee}
    seen: dict[str, list[float]] = {"abduct": [], "hip": [], "knee": []}
    for prim in stage.Traverse():
        attr = prim.GetAttribute("drive:angular:physics:stiffness")
        if not attr or not attr.HasAuthoredValue():
            continue
        for joint in seen:
            if prim.GetName().endswith(f"_{joint}"):
                seen[joint].append(float(attr.Get()))
    for joint, values in seen.items():
        if len(values) != 4:
            raise AssertionError(
                f"expected 4 authored drive stiffnesses on the {joint} joints, "
                f"found {len(values)}: {values}"
            )
        for value in values:
            if abs(value - expected_kp[joint]) > GAIN_TOL:
                raise AssertionError(
                    f"a {joint} drive stiffness is {value} N*m/rad. Spec's PD gain "
                    f"is {expected_kp[joint]} and its STANCE SPRING rate is "
                    f"{spring_k[joint]:.2f} -- if it is the latter, the crutch "
                    "became a drive gain."
                )
    assert isinstance(stage, Usd.Stage)  # keeps the import honest


def check_wheel_collider_is_a_cylinder_primitive() -> None:
    """Each wheel's collider is a `UsdGeomCylinder`, at `Spec`'s radius and width.

    Not a cooked convex hull. A meshed wheel under a convex-hull approximation
    becomes an N-gon and rolls with N contact impulses per revolution -- that is
    what wheel chatter is, and it is invisible in a still frame.
    """
    from pxr import UsdGeom, UsdPhysics

    from bestiary.robots.hound.build import SPEC

    stage = _usd_stage()
    found = 0
    for prim in stage.Traverse():
        if not prim.GetName().endswith("_wheel_geom"):
            continue
        found += 1
        if prim.GetTypeName() != "Cylinder":
            raise AssertionError(
                f"{prim.GetPath()} is a {prim.GetTypeName()!r}, not a Cylinder. "
                "A meshed wheel rolls on its facets."
            )
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise AssertionError(f"{prim.GetPath()} is a cylinder with no collider")
        cyl = UsdGeom.Cylinder(prim)
        radius = float(cyl.GetRadiusAttr().Get())
        height = float(cyl.GetHeightAttr().Get())
        if abs(radius - SPEC.wheel_r) > 1e-9:
            raise AssertionError(
                f"{prim.GetPath()} radius is {radius} m, Spec.wheel_r is {SPEC.wheel_r} m"
            )
        if abs(height - SPEC.wheel_w) > 1e-9:
            raise AssertionError(
                f"{prim.GetPath()} height is {height} m, Spec.wheel_w is {SPEC.wheel_w} m"
            )
    if found != 4:
        raise AssertionError(f"expected 4 wheel colliders, found {found}")


def check_all_bodies_report_contacts() -> None:
    """All 17 rigid bodies carry `PhysxContactReportAPI`.

    Not something the spawn flag can be trusted for. `activate_contact_sensors`
    walks for rigid bodies and STOPS descending at the first one, on the
    documented assumption that nested rigid bodies cannot happen -- true of a
    URDF import, false of an MJCF one, where all sixteen links are nested under
    the trunk. So one body reports contacts, and the whole inherited reward set
    fails at env construction with a message that blames a body-name regex.

    `hound_usd.activate_contact_reporting` puts the API in the asset instead.
    This asserts it is still there, because a re-conversion that skipped that
    step would leave a USD that loads, spawns, and then breaks the env.
    """
    from pxr import UsdPhysics

    stage = _usd_stage()
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 17:
        raise AssertionError(
            f"{len(bodies)} rigid bodies in the asset, expected 17: "
            f"{[p.GetName() for p in bodies]}"
        )
    missing = [
        prim.GetName()
        for prim in bodies
        if "PhysxContactReportAPI" not in prim.GetAppliedSchemas()
    ]
    if missing:
        raise AssertionError(
            f"these bodies do not report contacts: {missing}. Re-run "
            "`-m bestiary.isaac.hound_usd`, which applies the API; without it a "
            "contact sensor resolves only the trunk and every contact-based "
            "reward and termination raises."
        )


def check_static_friction_is_supplied_by_the_cfg() -> None:
    """The asset has no static friction, and `hound_cfg` puts one back.

    THE MOST EXPENSIVE FINDING OF THIS PORT, so it gets an assertion on both
    halves rather than a comment.

    MuJoCo has one sliding-friction coefficient per geom. USD Physics has two,
    and the importer writes only `physics:dynamicFriction`. The USD schema
    defaults `physics:staticFriction` to 0, so every contact on the converted
    machine slides freely from rest: placed in a correct stance on flat ground it
    did the splits and landed on its side inside half a second, four times out of
    four.

    Half one asserts the asset is still missing it, because that is the fact the
    override exists for -- if a future importer starts writing it, the override
    becomes a silent second opinion and someone should look. Half two asserts the
    override is present and equals `Spec`'s sliding coefficient.
    """
    from pxr import UsdPhysics

    from bestiary.isaac.hound_cfg import HOUND16_CFG
    from bestiary.robots.hound.build import SPEC

    stage = _usd_stage()
    authored_static = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.MaterialAPI)
        and prim.GetAttribute("physics:staticFriction")
        and prim.GetAttribute("physics:staticFriction").HasAuthoredValue()
    ]
    if authored_static:
        raise AssertionError(
            "the converted asset now authors physics:staticFriction on "
            f"{authored_static} -- the MJCF importer's behaviour changed. "
            "hound_cfg's physics_material override was written because it did "
            "NOT, and two opinions about friction is worse than one."
        )

    material = HOUND16_CFG.spawn.physics_material
    if material is None:
        raise AssertionError(
            "hound_cfg.HOUND16_CFG.spawn.physics_material is None. The converted "
            "asset has staticFriction 0 by schema default, so without an override "
            "the machine has no grip at rest and cannot stand."
        )
    want = SPEC.wheel_friction[0]
    for field in ("static_friction", "dynamic_friction"):
        got = float(getattr(material, field))
        if abs(got - want) > GAIN_TOL:
            raise AssertionError(
                f"hound_cfg's physics_material.{field} is {got}, "
                f"Spec.wheel_friction[0] is {want}"
            )


def check_only_the_load_bearing_geoms_collide() -> None:
    """Seventeen colliders: one per link, and none on the decoration.

    The MJCF's `class="deco"` and `class="hub"` geoms are `contype=0
    conaffinity=0` so that the physics is exactly what it would be with them
    deleted -- the same invariant `robots/spyder/check.py` holds for the
    spider's shell. `collision_from_visuals=True` on the converter would give
    all twenty-nine of them colliders and change the machine.
    """
    from pxr import UsdPhysics

    stage = _usd_stage()
    colliders = sorted(
        prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if len(colliders) != 17:
        raise AssertionError(
            f"{len(colliders)} colliders in the asset, expected 17 (trunk + 4 x "
            f"(abduct, thigh, calf, wheel)): {colliders}"
        )
    decorative = [n for n in colliders if not n.endswith("_geom")]
    if decorative:
        raise AssertionError(
            f"these colliders are not link geoms, so decoration is colliding: {decorative}"
        )


# ---------------------------------------------------------------------------
# 3. What the solver got, read off the loaded articulation.
# ---------------------------------------------------------------------------
def check_body_count(robot) -> None:
    """Seventeen bodies.

    Named in the mandate because `merge_fixed_joints` may not merge since Isaac
    Sim 5.1, and a merge that half-happened changes the body list without
    changing anything that raises. This model has no fixed joints to merge, so 17
    is also the assertion that none appeared.
    """
    if len(robot.body_names) != 17:
        raise AssertionError(
            f"len(robot.body_names) == {len(robot.body_names)}, expected 17. "
            f"Got: {robot.body_names}"
        )
    expected = {"trunk"} | {
        f"{leg}_{part}"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("abduct", "thigh", "calf", "wheel")
    }
    if set(robot.body_names) != expected:
        raise AssertionError(
            f"body names are not the MJCF's link names.\n  missing: "
            f"{sorted(expected - set(robot.body_names))}\n  extra:   "
            f"{sorted(set(robot.body_names) - expected)}"
        )


def check_sixteen_actuated_dof(robot) -> None:
    """Sixteen joints, and they are the MJCF's sixteen actuated joints.

    The trunk's six free-joint DoF are the articulation ROOT here, not joints, so
    16 and not 22 is the correct number -- the same split MuJoCo reports as
    nu = 16 with nv = 22.
    """
    if len(robot.joint_names) != 16:
        raise AssertionError(
            f"len(robot.joint_names) == {len(robot.joint_names)}, expected 16. "
            f"Got: {robot.joint_names}"
        )
    expected = {
        f"{leg}_{joint}"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("abduct", "hip", "knee", "wheel")
    }
    if set(robot.joint_names) != expected:
        raise AssertionError(
            f"joint names are not the MJCF's.\n  missing: "
            f"{sorted(expected - set(robot.joint_names))}\n  extra:   "
            f"{sorted(set(robot.joint_names) - expected)}"
        )


def check_actuator_split(robot) -> None:
    """Twelve position-driven leg joints, four velocity-driven wheels.

    And the wheels at `stiffness == 0` exactly. This is the assertion that stops
    the port's known failure from coming back: a position drive on a joint with
    +/-FLT_MAX limits refuses targets past +/-2*pi, so it works for ~3.2
    revolutions and then silently stops tracking.
    """
    groups = {name: list(act.joint_names) for name, act in robot.actuators.items()}
    if set(groups) != {"legs", "wheels"}:
        raise AssertionError(f"actuator groups are {sorted(groups)}, expected legs + wheels")
    if len(groups["legs"]) != 12:
        raise AssertionError(f"the 'legs' group has {len(groups['legs'])} joints: {groups['legs']}")
    if len(groups["wheels"]) != 4:
        raise AssertionError(
            f"the 'wheels' group has {len(groups['wheels'])} joints: {groups['wheels']}"
        )
    if any(not name.endswith("_wheel") for name in groups["wheels"]):
        raise AssertionError(f"non-wheel joints in the 'wheels' group: {groups['wheels']}")
    if any(name.endswith("_wheel") for name in groups["legs"]):
        raise AssertionError(f"a wheel is in the 'legs' group: {groups['legs']}")

    stiffness = _to_numpy(robot.data.joint_stiffness)[0]
    for name in groups["wheels"]:
        k = float(stiffness[robot.joint_names.index(name)])
        if k != 0.0:
            raise AssertionError(
                f"{name} has drive stiffness {k}, must be exactly 0. A position "
                "gain on an unlimited joint breaks past +/-2*pi -- about 3.2 "
                "wheel revolutions of driving."
            )


def check_joint_limits_match_spec(robot) -> None:
    """Every leg limit is `Spec`'s, in radians, and every wheel is unlimited.

    The importer writes limits in DEGREES: the knee's [-2.60, -0.60] rad is
    [-148.969, -34.377] in the USD. This check is what catches a 57.2958x error,
    which is a robot that stands up and cannot fold a leg.
    """
    from bestiary.robots.hound.build import SPEC

    limits = _to_numpy(robot.data.joint_pos_limits)[0]
    expected = {
        "abduct": SPEC.abduct_range,
        "hip": SPEC.hip_range,
        "knee": SPEC.knee_range,
    }
    for index, name in enumerate(robot.joint_names):
        lo, hi = (float(v) for v in limits[index])
        if name.endswith("_wheel"):
            if min(abs(lo), abs(hi)) < UNLIMITED_RAD:
                raise AssertionError(
                    f"{name} has limits [{lo}, {hi}] rad, inside the "
                    f"+/-{UNLIMITED_RAD:.0e} that counts as unlimited. A MuJoCo "
                    'hinge with limited="false" must arrive free to spin; a real '
                    "range here means the wheel is clamped."
                )
            continue
        joint = name.rsplit("_", 1)[1]
        want_lo, want_hi = expected[joint]
        if abs(lo - want_lo) > ANGLE_TOL_RAD or abs(hi - want_hi) > ANGLE_TOL_RAD:
            deg_ratio = (lo / want_lo) if want_lo else float("nan")
            raise AssertionError(
                f"{name} limits are [{lo:.6f}, {hi:.6f}] rad, Spec says "
                f"[{want_lo}, {want_hi}]. Ratio {deg_ratio:.4f}; a factor of "
                f"{180 / math.pi:.4f} means degrees were read as radians."
            )


def check_default_pose_is_the_solved_stance(robot) -> None:
    """The default joint positions are the solved standing stance.

    Load-bearing beyond "the robot looks right": the position action term uses
    `use_default_offset=True`, so the default pose IS the zero action. Get it
    wrong and a policy's zero action is some other pose, and the whole prior --
    "the policy starts at standing instead of having to discover it" -- is gone.

    `Spec.stance_knee` is derived, not typed: it is the knee angle that puts each
    wheel axle directly under its own hip pivot, which is what makes the hip's
    static holding torque nearly zero.
    """
    from bestiary.robots.hound.build import SPEC

    want = {
        "abduct": SPEC.stance_abduct,
        "hip": SPEC.stance_hip,
        "knee": SPEC.stance_knee,
        "wheel": 0.0,
    }
    pose = _to_numpy(robot.data.default_joint_pos)[0]
    for index, name in enumerate(robot.joint_names):
        joint = name.rsplit("_", 1)[1]
        got = float(pose[index])
        if abs(got - want[joint]) > ANGLE_TOL_RAD:
            raise AssertionError(
                f"{name} default position is {got:.6f} rad, the solved stance is "
                f"{want[joint]:.6f} rad ({math.degrees(want[joint]):.2f} deg)"
            )


def check_masses_match_the_mjcf(robot) -> None:
    """Every body's mass is the mass the committed MuJoCo model gives it.

    Read from `assets/hound16pd.xml` rather than from `Spec`, so this compares
    the port against the artifact a MuJoCo run loaded and not merely against the
    converter's own input.
    """
    from bestiary import paths

    want = mjcf_geom_masses(paths.HOUND_PD_XML.read_text())
    if len(want) != 17:
        raise AssertionError(
            f"parsed {len(want)} masses out of {paths.HOUND_PD_XML}, expected 17: "
            f"{sorted(want)}"
        )
    got = _to_numpy(robot.data.default_mass)[0]
    for index, name in enumerate(robot.body_names):
        if name not in want:
            raise AssertionError(f"body {name!r} has no mass in {paths.HOUND_PD_XML}")
        if not math.isclose(float(got[index]), want[name], rel_tol=MASS_RTOL):
            raise AssertionError(
                f"{name} is {float(got[index]):.9f} kg in the USD and "
                f"{want[name]:.9f} kg in {paths.HOUND_PD_XML.name} -- a relative "
                f"error of {abs(float(got[index]) / want[name] - 1):.3e}, past the "
                f"{MASS_RTOL:.3e} a float32 round trip explains"
            )
    total_usd = float(got.sum())
    total_mjcf = sum(want.values())
    if not math.isclose(total_usd, total_mjcf, rel_tol=MASS_RTOL):
        raise AssertionError(
            f"total mass {total_usd:.9f} kg in the USD vs {total_mjcf:.9f} kg in the MJCF"
        )


def check_leg_gains_and_torque_ceilings(robot) -> None:
    """PD gains and effort ceilings are `Spec`'s, per joint.

    kp 60 / 80 / 90 N*m/rad and kv 3 / 4 / 4.5 N*m/(rad/s) are the gains
    `robots/hound/play.py` already drives this machine with -- the only ones
    shown to hold the stance and re-pose it without fighting the physics into
    instability. The effort limits are Go2's rated torques, so the machine is no
    easier to drive here than in MuJoCo, only easier to command.
    """
    from bestiary.robots.hound.build import SPEC

    want_kp = {"abduct": SPEC.kp_abduct, "hip": SPEC.kp_hip, "knee": SPEC.kp_knee}
    want_kv = {"abduct": SPEC.kv_abduct, "hip": SPEC.kv_hip, "knee": SPEC.kv_knee}
    want_tau = {
        "abduct": SPEC.gear_abduct,
        "hip": SPEC.gear_hip,
        "knee": SPEC.gear_knee,
        "wheel": SPEC.gear_wheel,
    }
    stiffness = _to_numpy(robot.data.joint_stiffness)[0]
    damping = _to_numpy(robot.data.joint_damping)[0]
    effort = _to_numpy(robot.data.joint_effort_limits)[0]
    for index, name in enumerate(robot.joint_names):
        joint = name.rsplit("_", 1)[1]
        if abs(float(effort[index]) - want_tau[joint]) > GAIN_TOL:
            raise AssertionError(
                f"{name} effort limit is {float(effort[index])} N*m, Spec says "
                f"{want_tau[joint]}"
            )
        if joint == "wheel":
            continue
        if abs(float(stiffness[index]) - want_kp[joint]) > GAIN_TOL:
            raise AssertionError(
                f"{name} kp is {float(stiffness[index])}, Spec says {want_kp[joint]}"
            )
        if abs(float(damping[index]) - want_kv[joint]) > GAIN_TOL:
            raise AssertionError(
                f"{name} kv is {float(damping[index])}, Spec says {want_kv[joint]}"
            )


def check_armature_and_joint_friction(robot) -> None:
    """Reflected rotor inertia and dry friction survived the conversion.

    Armature is the most-omitted parameter in a ported robot and it is not
    cosmetic: leave it out and the leg is a nearly massless whip a policy will
    flick at rates the real actuator cannot reach. The wheel's 0.399 N*m
    breakaway is load-bearing for a different reason -- at `condim=3` it is the
    only thing that stops a coasting wheel, and `Spec` derives it as 13.3% of
    rated torque because 5% let 3 of 15 noisy resets end on the floor.
    """
    from bestiary.robots.hound.build import SPEC

    want_armature = {"wheel": SPEC.wheel_armature}
    want_friction = {"wheel": SPEC.wheel_brake_torque()}
    armature = _to_numpy(robot.data.joint_armature)[0]
    friction = _to_numpy(robot.data.joint_friction_coeff)[0]
    for index, name in enumerate(robot.joint_names):
        joint = name.rsplit("_", 1)[1]
        want_a = want_armature.get(joint, SPEC.armature)
        want_f = want_friction.get(joint, SPEC.frictionloss)
        if abs(float(armature[index]) - want_a) > GAIN_TOL:
            raise AssertionError(
                f"{name} armature is {float(armature[index])} kg*m^2, Spec says {want_a}"
            )
        if abs(float(friction[index]) - want_f) > GAIN_TOL:
            raise AssertionError(
                f"{name} joint friction is {float(friction[index])} N*m, Spec says {want_f:.4f}"
            )


def check_wheel_velocity_gain_is_derived(robot) -> None:
    """The wheel drive gain on the joints is the one `hound_cfg` derives.

    Recomputed here from `Spec` and the control period rather than compared to a
    literal, because the point of deriving it is that it moves when the wheel
    does. The wheel's spin inertia is (1/2) m r^2 + armature and the gain is that
    over one control period.
    """
    from bestiary.isaac.hound_cfg import (
        CONTROL_DT_S,
        wheel_action_scale,
        wheel_velocity_gain,
    )
    from bestiary.robots.hound.build import SPEC

    inertia = 0.5 * SPEC.wheel_mass * SPEC.wheel_r**2 + SPEC.wheel_armature
    expected = inertia / CONTROL_DT_S
    if abs(wheel_velocity_gain() - expected) > 1e-12:
        raise AssertionError(
            f"hound_cfg.wheel_velocity_gain() is {wheel_velocity_gain()} but "
            f"({inertia:.9f} kg*m^2) / ({CONTROL_DT_S} s) = {expected}"
        )
    expected_scale = SPEC.gear_wheel / expected
    if abs(wheel_action_scale() - expected_scale) > 1e-12:
        raise AssertionError(
            f"hound_cfg.wheel_action_scale() is {wheel_action_scale()} but the "
            f"drive saturates at {SPEC.gear_wheel} / {expected} = {expected_scale} rad/s"
        )
    damping = _to_numpy(robot.data.joint_damping)[0]
    for index, name in enumerate(robot.joint_names):
        if not name.endswith("_wheel"):
            continue
        if abs(float(damping[index]) - expected) > GAIN_TOL:
            raise AssertionError(
                f"{name} drive damping in the sim is {float(damping[index])}, "
                f"hound_cfg derives {expected}"
            )


# ---------------------------------------------------------------------------
# 4. The reward table. Not "does it construct" -- does it pay for the right
#    things on a body whose feet are driven wheels.
# ---------------------------------------------------------------------------
def _hound_env_cfg():
    """The training config, constructed. Cheap: no sim, no terrain generated.

    Read off a CONSTRUCTED cfg rather than off `HoundRewardsCfg()`, because
    `HoundDesertEnvCfg.__post_init__` is where `undesired_contacts` gets Hound's
    link names. Reading the class would check the declared half of the table and
    miss the patched half.
    """
    from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg

    return HoundDesertEnvCfg()


def _live_reward_terms(cfg) -> dict[str, object]:
    """The reward terms the manager will actually run, in the manager's own order.

    Deleting an inherited term means setting it to None, and every Isaac Lab
    manager skips a None term (`managers/reward_manager.py:224-227`). This reads
    `__dict__`, which is what the manager reads, so a term that is present-but-None
    is absent here for exactly the reason it is absent there.
    """
    return {name: term for name, term in vars(cfg.rewards).items() if term is not None}


def _resolve(names: list[str], patterns) -> list[str]:
    """The names a manager term's regex list resolves to.

    `utils/string.py::resolve_matching_names` uses `re.fullmatch`, so a pattern
    that reads like a prefix is not one. `None` there means "every joint/body",
    which has to be reproduced or an unscoped term looks like an empty one -- and
    an unscoped joint penalty reaching the wheels is the whole failure here.
    """
    if patterns is None:
        return list(names)
    if isinstance(patterns, str):
        patterns = [patterns]
    return [n for n in names if any(re.fullmatch(p, n) for p in patterns)]


def _reads(func, fields: tuple[str, ...]) -> list[str]:
    """Which of `fields` a reward function's own source reads off `.data`.

    Structural rather than name-based on purpose. The failure to prevent is not
    "a term called feet_air_time comes back", it is "some term that reads contact
    timing ends up scoped to a wheel", and only the source says which terms those
    are.
    """
    source = inspect.getsource(func)
    return [f for f in fields if f"data.{f}" in source]


def check_no_contact_timing_reward_on_a_wheel(robot) -> None:
    """No reward pays for the duration of contact, or of flight, on a wheel body.

    `feet_air_time` at +0.125 per second of air time was re-scoped from ANYmal's
    `.*FOOT` onto `.*_wheel`, which fixed the ValueError and kept the wrong
    incentive: a driven hub wheel is in continuous rolling contact, so the only way
    to earn air time is to leave the ground. That term pays the machine to HOP.
    `research/decisions/0004` Part B and `fan-ziqi/robot_lab`'s Go2-W config -- our
    exact topology -- both delete it.
    """
    cfg = _hound_env_cfg()
    terms = _live_reward_terms(cfg)

    if "feet_air_time" in terms:
        term = terms["feet_air_time"]
        raise AssertionError(
            f"feet_air_time is live at weight {term.weight} on "
            f"{term.params.get('sensor_cfg')}. On a driven hub wheel it pays the "
            "machine to HOP: air time can only be earned by breaking the rolling "
            "contact the wheel exists to keep. DELETE it (set it to None in "
            "HoundRewardsCfg), do not re-weight it -- research/decisions/0004 "
            "Part B."
        )

    offenders = []
    for name, term in terms.items():
        sensor_cfg = term.params.get("sensor_cfg")
        if sensor_cfg is None:
            continue
        bodies = _resolve(list(robot.body_names), sensor_cfg.body_names)
        if not bodies:
            raise AssertionError(
                f"reward term {name!r} scopes its contact sensor to "
                f"{sensor_cfg.body_names!r}, which matches NONE of Hound's bodies "
                f"{sorted(robot.body_names)}. Isaac Lab raises "
                "'Not all regular expressions are matched!' for this at env "
                "construction; this check says so before the app starts."
            )
        wheels = [b for b in bodies if b.endswith("_wheel")]
        if not wheels:
            continue
        timing = _reads(term.func, CONTACT_TIMING_FIELDS)
        if timing:
            offenders.append(
                f"{name} (weight {term.weight}) reads {timing} on {wheels}"
            )
    if offenders:
        raise AssertionError(
            "these reward terms pay for contact TIMING on a wheel body, i.e. for "
            f"breaking rolling contact: {offenders}. A wheel is not a foot; see "
            "research/decisions/0004 Part B."
        )


def check_wheel_joints_are_not_charged_for_spinning(robot) -> None:
    """No per-joint effort penalty reaches a wheel, except the 100x weaker one.

    The inherited table charges `dof_torques_l2` and `dof_acc_l2` across all
    sixteen joints. Torque on a rolling wheel is the thing that makes it roll, and
    `hound_cfg.wheel_velocity_gain()` is derived so the drive reaches a commanded
    speed in one control period -- so it accelerates the wheel hard by design.
    Both terms therefore bill the machine for driving, which is
    `research/learnings/011`'s failure written straight into the reward.

    `research/decisions/0004` Part B scopes them to the legs and adds robot_lab's
    wheel-only acceleration term at 100x less weight. Both halves are asserted:
    the structural one (nothing reaches a wheel under any name) and the concrete
    one (the split is the one 0004 specifies, not merely "some split").
    """
    from bestiary.isaac.hound_desert_env_cfg import (
        WHEEL_ACC_PENALTY_RATIO,
        WHEEL_ACC_PENALTY_WEIGHT,
    )

    cfg = _hound_env_cfg()
    terms = _live_reward_terms(cfg)
    wheels = sorted(n for n in robot.joint_names if n.endswith("_wheel"))
    legs = sorted(n for n in robot.joint_names if not n.endswith("_wheel"))
    if len(wheels) != 4 or len(legs) != 12:
        raise AssertionError(
            f"expected 12 leg joints and 4 wheel joints, got {legs} and {wheels}"
        )

    def scope(term) -> list[str]:
        asset_cfg = term.params.get("asset_cfg")
        names = None if asset_cfg is None else asset_cfg.joint_names
        return sorted(_resolve(list(robot.joint_names), names))

    offenders = []
    for name, term in terms.items():
        fields = _reads(term.func, JOINT_EFFORT_FIELDS)
        if not fields:
            continue
        billed = sorted(set(scope(term)) & set(wheels))
        if billed and name != WHEEL_EFFORT_TERM:
            offenders.append(
                f"{name} (weight {term.weight}) reads {fields} on {billed}"
            )
    if offenders:
        raise AssertionError(
            "these reward terms charge a WHEEL joint for moving or pushing: "
            f"{offenders}. Only {WHEEL_EFFORT_TERM!r} may, and only at "
            f"{WHEEL_ACC_PENALTY_WEIGHT}. Scope the rest to the legs with "
            "SceneEntityCfg('robot', joint_names=LEG_JOINT_EXPR)."
        )

    for name in ("dof_torques_l2", "dof_acc_l2", "dof_pos_limits"):
        if name not in terms:
            raise AssertionError(
                f"reward term {name!r} is not in the live table {sorted(terms)}. "
                "It was scoped to the legs, not deleted -- if upstream renamed it, "
                "the leg-only scoping is now silently absent."
            )
        got = scope(terms[name])
        if got != legs:
            raise AssertionError(
                f"{name} is scoped to {got}, expected exactly the 12 leg joints "
                f"{legs}. A wheel in this term is a charge for spinning."
            )

    if WHEEL_EFFORT_TERM not in terms:
        raise AssertionError(
            f"{WHEEL_EFFORT_TERM!r} is missing from the live table {sorted(terms)}. "
            "With no wheel-scoped acceleration penalty at all the wheel drive is "
            "unpriced, which is not what 0004 Part B specifies."
        )
    wheel_term = terms[WHEEL_EFFORT_TERM]
    got = scope(wheel_term)
    if got != wheels:
        raise AssertionError(
            f"{WHEEL_EFFORT_TERM} is scoped to {got}, expected exactly the 4 wheel "
            f"joints {wheels}"
        )
    if abs(wheel_term.weight - WHEEL_ACC_PENALTY_WEIGHT) > 1e-15:
        raise AssertionError(
            f"{WHEEL_EFFORT_TERM} weight is {wheel_term.weight}, "
            f"research/decisions/0004 Part B records robot_lab's "
            f"{WHEEL_ACC_PENALTY_WEIGHT}"
        )
    leg_weight = terms["dof_acc_l2"].weight
    ratio = abs(leg_weight / wheel_term.weight)
    if abs(ratio - WHEEL_ACC_PENALTY_RATIO) > 1e-6 * WHEEL_ACC_PENALTY_RATIO:
        raise AssertionError(
            f"dof_acc_l2 is {leg_weight} and {WHEEL_EFFORT_TERM} is "
            f"{wheel_term.weight}, a ratio of {ratio:.4f}. 0004 Part B records "
            f"{WHEEL_ACC_PENALTY_RATIO:.0f}x."
        )


def check_leg_velocity_limit_is_the_go2_rated_speed(robot) -> None:
    """The legs carry Go2's rated joint speed; the wheels deliberately carry none.

    `Spec` has no joint speed limit, so the twelve leg joints arrived with the
    converted USD's own default -- measured at 1.7e4 rad/s, i.e. none, and a policy
    free to learn leg rates the real actuator does not have.

    This asserts three separate things, because there are three ways to get it
    wrong: the number still matches the shipped Go2 config it was read from (a
    silent upstream change is reported, not adopted); it is set on
    `velocity_limit_sim` and not on `velocity_limit`, which `ImplicitActuator`
    accepts, warns about and then discards (`actuators/actuator_pd.py:81-91`); and
    the solver actually got it.
    """
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    from bestiary.isaac.hound_cfg import HOUND16_CFG, LEG_JOINT_VELOCITY_LIMIT_RAD_S

    shipped = UNITREE_GO2_CFG.actuators["base_legs"].velocity_limit
    if shipped is None:
        raise AssertionError(
            "UNITREE_GO2_CFG.actuators['base_legs'].velocity_limit is None. That "
            "is the source hound_cfg.LEG_JOINT_VELOCITY_LIMIT_RAD_S cites "
            "(isaaclab_assets/robots/unitree.py:176); it is gone, so the citation "
            "no longer points at anything."
        )
    if abs(float(shipped) - LEG_JOINT_VELOCITY_LIMIT_RAD_S) > GAIN_TOL:
        raise AssertionError(
            f"hound_cfg.LEG_JOINT_VELOCITY_LIMIT_RAD_S is "
            f"{LEG_JOINT_VELOCITY_LIMIT_RAD_S} rad/s, but the shipped "
            f"UNITREE_GO2_CFG leg actuator now says {float(shipped)} rad/s. Ours "
            "is pinned on purpose so a run's dynamics cannot move under it -- so "
            "this is a decision to make, not a number to sync blindly."
        )

    legs_cfg = HOUND16_CFG.actuators["legs"]
    if legs_cfg.velocity_limit_sim is None:
        raise AssertionError(
            "hound_cfg's 'legs' actuator has no velocity_limit_sim, so the leg "
            "joints are unlimited in the solver. Note that setting "
            f"'velocity_limit' instead does nothing: it is currently "
            f"{legs_cfg.velocity_limit!r}, and ImplicitActuator logs a warning and "
            "discards it."
        )
    if abs(float(legs_cfg.velocity_limit_sim) - LEG_JOINT_VELOCITY_LIMIT_RAD_S) > GAIN_TOL:
        raise AssertionError(
            f"the 'legs' actuator's velocity_limit_sim is "
            f"{float(legs_cfg.velocity_limit_sim)} rad/s, the constant says "
            f"{LEG_JOINT_VELOCITY_LIMIT_RAD_S}"
        )

    wheels_cfg = HOUND16_CFG.actuators["wheels"]
    if wheels_cfg.velocity_limit_sim is not None:
        raise AssertionError(
            f"the 'wheels' actuator sets velocity_limit_sim="
            f"{wheels_cfg.velocity_limit_sim}. There is no vendor source for this "
            "hub drive's top speed -- CARD.md's Provenance says the wheel is not a "
            "Unitree part -- and velocity_limit_sim makes the solver BRAKE a joint "
            "past the limit, so a guessed number here is an undeclared brake on a "
            "wheel whose whole point is that it coasts."
        )

    limits = _to_numpy(robot.data.joint_vel_limits)[0]
    for index, name in enumerate(robot.joint_names):
        got = float(limits[index])
        if name.endswith("_wheel"):
            if got < WHEEL_VEL_UNLIMITED_RAD_S:
                raise AssertionError(
                    f"{name} has a solver velocity limit of {got} rad/s, below the "
                    f"{WHEEL_VEL_UNLIMITED_RAD_S:.0e} that counts as unlimited. "
                    "Somebody chose a hub-drive top speed; there is no source for "
                    "one."
                )
            continue
        if abs(got - LEG_JOINT_VELOCITY_LIMIT_RAD_S) > GAIN_TOL:
            raise AssertionError(
                f"{name} has a solver velocity limit of {got} rad/s, Go2's rated "
                f"joint speed is {LEG_JOINT_VELOCITY_LIMIT_RAD_S}. 1.7e4 means the "
                "USD default is still in place and velocity_limit_sim never "
                "reached the solver."
            )


def check_reward_budget_against_011_and_015(robot) -> None:
    """Price the penalty table against achievable tracking income, and print it.

    `research/learnings/011` and `015` are the two ways this reward has already
    gone wrong on this robot, and both were arithmetic that could have been done
    before the run. In 011 the control cost was 105.5% of the whole
    policy-versus-control gap, so driving did not pay. In 015 a halved control cost
    made driving pay, and one fixed trot then collected 76% of all the speed income
    there was, so tracking was never required.

    So this prints a per-term budget and flags anything worth more than 30% of
    achievable income. It ASSERTS only the one line in it that needs no assumed
    operating point: that the wheel-acceleration penalty is negligible at the
    acceleration the wheel drive is DERIVED to produce. That is the claim the
    leg/wheel split makes, and it is what breaks if the wheel weight is ever raised
    to the leg weight.

    Every line marked [ASSUMED] rests on an operating point nobody has measured on
    this machine, because no Hound policy exists yet. **None of those numbers may
    be cited in the record**; a committed script under `research/scripts/` is what
    earns a number that right.
    """
    from bestiary.isaac.hound_cfg import CONTROL_DT_S, wheel_action_scale
    from bestiary.robots.hound.build import SPEC

    cfg = _hound_env_cfg()
    terms = _live_reward_terms(cfg)
    dt = cfg.decimation * cfg.sim.dt
    n_wheels = sum(1 for n in robot.joint_names if n.endswith("_wheel"))
    n_legs = len(robot.joint_names) - n_wheels

    # Achievable value of track_lin_vel_xy_exp for a machine that cannot make
    # lateral velocity. Four wheels with fixed spin axes cannot hold body-frame
    # v_y, so the best any competence reaches is e_y = c_y; with c_y ~ U(-1, 1)
    # and the kernel exp(-(e_x^2 + e_y^2)/std^2) at e_x = 0,
    #
    #     E[exp(-(c_y/std)^2)] = (1/2) * int_{-1}^{1} exp(-(x/std)^2) dx
    #                          = (std * sqrt(pi) / 2) * erf(1/std)
    #
    # Computed here rather than copied: decision 0005 B2 states 0.4409 but its
    # Part B has no committed script, so under the number rule it is re-derived.
    lin = terms["track_lin_vel_xy_exp"]
    ang = terms["track_ang_vel_z_exp"]
    std = float(lin.params["std"])
    lo, hi = cfg.commands.base_velocity.ranges.lin_vel_y
    half = 0.5 * (hi - lo)
    # Mean of exp(-(e/std)^2) over a command uniform on [-half, +half], where the
    # policy holds the achievable axis perfectly and eats the unachievable one.
    #
    # half == 0 is not a degenerate case, it is the FIXED case: with the command
    # pinned to a single value the machine can actually hold, the whole channel is
    # earnable and the ceiling is 1. Taking the limit,
    #     lim_{h->0} (std*sqrt(pi)/(2h)) * erf(h/std) = 1
    # because erf(x) -> 2x/sqrt(pi) as x -> 0. Written out because the closed form
    # divides by zero there, and a ZeroDivisionError inside the budget check is a
    # confusing way to learn that a command range was correctly set to zero.
    if half == 0.0:
        lin_ceiling = 1.0
    else:
        lin_ceiling = (std * math.sqrt(math.pi) / (2.0 * half)) * math.erf(half / std)
    income = (lin.weight * lin_ceiling + ang.weight * 1.0) * dt

    # The wheel drive's own design acceleration: its time constant is one control
    # period, so a step to the largest commandable speed is that speed over that
    # period. Derived, not assumed -- which is why this line is asserted.
    wheel_acc = wheel_action_scale() / CONTROL_DT_S
    wheel_cost = abs(terms[WHEEL_EFFORT_TERM].weight) * n_wheels * wheel_acc**2 * dt

    # Assumed operating points, all labelled. Chosen to bracket rather than to
    # flatter: the two joint-acceleration figures are decision 0005 B6's bracket.
    vz_rms, wxy_rms, tau_rms, acc_rms, dact_rms = 0.15, 0.5, 5.0, 250.0, 0.1
    rows = [
        ("track_lin_vel_xy_exp  (income, ceiling)", lin.weight * lin_ceiling * dt, "v_y unachievable"),
        ("track_ang_vel_z_exp   (income, perfect)", ang.weight * 1.0 * dt, "yaw tracked exactly"),
        (
            f"{WHEEL_EFFORT_TERM} @ {wheel_acc:.0f} rad/s^2",
            -wheel_cost,
            "DERIVED from the drive",
        ),
        (
            "dof_acc_l2            (12 legs)",
            -abs(terms["dof_acc_l2"].weight) * n_legs * acc_rms**2 * dt,
            f"[ASSUMED] {acc_rms:.0f} rad/s^2 rms",
        ),
        (
            "dof_torques_l2        (12 legs)",
            -abs(terms["dof_torques_l2"].weight) * n_legs * tau_rms**2 * dt,
            f"[ASSUMED] {tau_rms:.1f} N*m rms",
        ),
        (
            "lin_vel_z_l2",
            -abs(terms["lin_vel_z_l2"].weight) * vz_rms**2 * dt,
            f"[ASSUMED] {vz_rms:.2f} m/s rms ({vz_rms * 3.2808:.2f} ft/s)",
        ),
        (
            "ang_vel_xy_l2",
            -abs(terms["ang_vel_xy_l2"].weight) * 2 * wxy_rms**2 * dt,
            f"[ASSUMED] {wxy_rms:.2f} rad/s rms per axis",
        ),
        (
            "action_rate_l2        (16 actions)",
            -abs(terms["action_rate_l2"].weight)
            * len(robot.joint_names)
            * dact_rms**2
            * dt,
            f"[ASSUMED] {dact_rms:.2f} rms step change",
        ),
    ]

    # `undesired_contacts` is priced ONLY IF THE TABLE STILL HAS IT. It was
    # deleted (`HoundRewardsCfg.undesired_contacts = None`) after this very check
    # priced one thigh in contact at 106.27% of achievable income -- the
    # arithmetic of `research/learnings/011` on a different term.
    #
    # Discovered rather than hardcoded, and this is the point: the first version
    # of this list indexed `terms["undesired_contacts"]` unconditionally, so
    # deleting the term the check had just condemned made the CHECK raise
    # KeyError. That is exactly the defect `research/anomalies.jsonl` 43 records
    # against `track_eval.TERMS` -- an instrument carrying a hardcoded term list
    # cannot survive the reward changing, which is the one thing rewards do.
    # Guard `decomposition-completeness` exists for the same reason.
    if "undesired_contacts" in terms:
        rows.append(
            (
                "undesired_contacts    (ONE body)",
                -abs(terms["undesired_contacts"].weight) * 1.0 * dt,
                "exact: the term is an integer COUNT",
            )
        )

    print(f"      operating point: dt = {dt} s, achievable income = {income:.6f}/step", flush=True)
    print(
        f"      (income ceiling {lin_ceiling:.4f} on the lin term because "
        f"lin_vel_y is commanded over {lo:+.1f}..{hi:+.1f} m/s and cannot be made)",
        flush=True,
    )
    penalties = 0.0
    for label, per_step, note in rows:
        share = abs(per_step) / income
        flag = " <<< FLAG" if share > PENALTY_BUDGET_FLAG_FRACTION and per_step < 0 else ""
        print(
            f"      {label:<40s} {per_step:+.6f}/step  {share * 100:6.2f}% of income"
            f"  {note}{flag}",
            flush=True,
        )
        if per_step < 0:
            penalties += -per_step
    print(
        f"      {'PENALTY SUM':<40s} {-penalties:+.6f}/step  "
        f"{penalties / income * 100:6.2f}% of income"
        f"{'  <<< FLAG' if penalties / income > PENALTY_BUDGET_FLAG_FRACTION else ''}",
        flush=True,
    )
    print(
        f"      for scale: SPEC.gear_wheel = {SPEC.gear_wheel} N*m, so a wheel at "
        f"the traction limit is not what costs here -- accelerations are.",
        flush=True,
    )

    share = wheel_cost / income
    if share > WHEEL_ACC_BUDGET_FRACTION:
        raise AssertionError(
            f"{WHEEL_EFFORT_TERM} costs {wheel_cost:.3e}/step at the drive's own "
            f"design acceleration of {wheel_acc:.1f} rad/s^2, which is "
            f"{share * 100:.2f}% of the {income:.6f}/step of achievable tracking "
            f"income -- over the {WHEEL_ACC_BUDGET_FRACTION * 100:.0f}% this split "
            "exists to hold. At the LEG weight the same acceleration costs "
            f"{wheel_cost * abs(terms['dof_acc_l2'].weight / terms[WHEEL_EFFORT_TERM].weight):.3e}"
            "/step, which is what research/decisions/0004 Part B's 100x is for."
        )


# ---------------------------------------------------------------------------
# 5. The task VARIANTS. Nothing here needs the articulation or the app: a config
#    is a config, and every one of these configs constructs pre-app.
# ---------------------------------------------------------------------------
#: Basename of the committed v5 heightfield, asserted literally.
#:
#: `paths.GENTLE_V5_HFIELD` is asserted too, but a check that ONLY compares
#: against the constant still passes when the constant itself is repointed at
#: other bytes. The literal is the independent half of that pair.
V5_HFIELD_BASENAME = "gentle_v5_hfield.bin"

#: Metres of elevation the v5 asset spans, from `research/decisions/0007`.
#: Typed here rather than imported for the same reason: the env cfg's own
#: constant is one of the things being checked.
V5_Z_SPAN_M = 2.25

#: The dotted paths inside `scene` at which the forward-v5 task may differ from
#: `Bestiary-Desert-Hound-v0`. Two, and they are one change: the desert tile
#: leaves the mix and the gentle-v5 tile takes its place, so the sub-terrain
#: DICT gains one key and loses another. `scene` is an unavoidable section —
#: `terrain_generator` lives under it — and this list is what stops "unavoidable
#: section" from becoming a licence to move the robot, the sensors, the env
#: count or the spacing along with the ground.
V5_SCENE_DIFF_PATHS = [
    "terrain.terrain_generator.sub_terrains.bestiary_desert",
    "terrain.terrain_generator.sub_terrains.bestiary_gentle",
]


def _diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths at which two `to_dict()` trees disagree, deepest-wins.

    A key present in one tree and absent from the other reports as that key's
    own path rather than recursing into it — which is what makes the desert tile
    leaving and the gentle tile arriving read as two paths instead of a dozen
    leaves.

    Deliberately a duplicate of `check_spyder.py`'s `_dict_diff_paths` rather
    than an import. The two oracles already duplicate `_to_numpy`, `_run` and
    `_exit` for the same reason: they gate launches independently, and a syntax
    error or a bad edit in one must not be able to take the other down with it.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}{key}"
            if key not in a or key not in b:
                out.append(path)
            else:
                out.extend(_diff_paths(a[key], b[key], f"{path}."))
        return out
    return [] if a == b else [prefix.rstrip(".")]


def check_forward_v5_is_vx_on_v5_ground() -> None:
    """`Bestiary-ForwardV5-Hound-v0` pays v_x alone, on v5 ground, and moves nothing else.

    The task (`hound_forward_v5_env_cfg.py`) asks what an UNSHAPED speed reward
    selects on a body that can both roll and gallop. That question only has an
    answer if the reward really is unshaped: a single surviving penalty from the
    desert table would be a thumb on the scale between the two modes, and
    `HoundRewardsCfg` is full of terms that are exactly that (`lin_vel_z_l2` at
    -2.0 prices the vertical bouncing a gallop is made of; the 100x split
    between `dof_acc_l2` and `dof_acc_wheel_l2` prices stepping over rolling).
    So "exactly one term" is asserted structurally, not by reading the file:

      (a) ONE live reward term — `vars(cfg.rewards)` minus the Nones, which is
          what `RewardManager` iterates — named `forward_velocity`, at weight
          1.0, resolving to the `bestiary.isaac.rewards.forward_velocity`
          function object itself rather than to something that shares its name.
      (b) THE GROUND IS v5. The bestiary tile reads a path ending in
          `gentle_v5_hfield.bin` that equals `paths.GENTLE_V5_HFIELD`, declares
          decision 0007's 2.25 m span, and those bytes MEASURE that span through
          the same bridge Isaac Lab reads them through. The desert tile is gone
          from the mix: this task does not train on the 5.05 m desert.
      (c) THE DIFF against `Bestiary-Desert-Hound-v0` is `rewards` and `scene`,
          and inside `scene` it is exactly the two sub-terrain keys. Everything
          else the desert task carries is inherited on purpose — the wheel-aware
          action split, the observation with the four unbounded wheel angles
          already dropped, the trunk-contact termination, the command ranges and
          the retargeted body names.

    Both the training config and the Play twin, because the Play class descends
    from the desert Play class and gets its surgery from a SECOND call to
    `apply_forward_v5`. An edit reaching one call and not the other would put
    the viewer on different ground, or under a different reward, than the run.
    """
    import numpy as _np

    from bestiary import paths
    from bestiary.isaac import rewards as bestiary_rewards
    from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg, HoundDesertEnvCfg_PLAY
    from bestiary.isaac.hound_forward_v5_env_cfg import (
        HoundForwardV5EnvCfg,
        HoundForwardV5EnvCfg_PLAY,
    )
    from bestiary.isaac.spyder_forward_env_cfg import REWARD_TERM_NAME, REWARD_WEIGHT
    from bestiary.isaac.spyder_forward_v5_env_cfg import GENTLE_SUBTERRAIN_KEY
    from bestiary.terrain.isaac_hf import load_desert_m

    arms = (
        ("train", HoundForwardV5EnvCfg(), HoundDesertEnvCfg()),
        ("play", HoundForwardV5EnvCfg_PLAY(), HoundDesertEnvCfg_PLAY()),
    )

    for arm, v5, base in arms:
        # (a) Exactly one reward term, and it is the one the task is named for.
        terms = _live_reward_terms(v5)
        if sorted(terms) != [REWARD_TERM_NAME]:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) pays {sorted(terms)}, but the "
                f"whole task is [{REWARD_TERM_NAME!r}]. A second term prices "
                "rolling against galloping, which is the one thing this run must "
                "not do: the question is which mode an UNSHAPED speed objective "
                "selects on a body that has both."
            )
        term = terms[REWARD_TERM_NAME]
        if term.func is not bestiary_rewards.forward_velocity:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) pays {REWARD_TERM_NAME!r} "
                f"through {getattr(term.func, '__name__', term.func)!r}, not "
                "bestiary.isaac.rewards.forward_velocity — it is measuring "
                "something other than base-frame forward speed."
            )
        if term.weight != REWARD_WEIGHT:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) pays {REWARD_TERM_NAME!r} at "
                f"weight {term.weight}, not {REWARD_WEIGHT}. At 1.0 the episode "
                "return reads as metres of forward travel; any other weight is a "
                "relabelling that makes the run's own number mean nothing."
            )

        # (b) The ground is the committed v5 asset, and the desert is gone.
        gen = v5.scene.terrain.terrain_generator
        sub = gen.sub_terrains.get(GENTLE_SUBTERRAIN_KEY)
        if sub is None:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) has no "
                f"{GENTLE_SUBTERRAIN_KEY!r} sub-terrain; it carries "
                f"{sorted(gen.sub_terrains)}. The terrain swap did not run."
            )
        if "bestiary_desert" in gen.sub_terrains:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) still carries a "
                "'bestiary_desert' tile alongside the gentle one. Half its tiles "
                "would be the 5.05 m desert, which is not the ground "
                "research/decisions/0007 puts new arms on."
            )
        if not sub.hfield_path.endswith(V5_HFIELD_BASENAME):
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) reads {sub.hfield_path}, whose "
                f"name is not {V5_HFIELD_BASENAME!r}. v4's crests reach 47 degrees "
                "— past any angle of repose — which is why 0007 made v5 mandatory "
                "for every new arm."
            )
        if sub.hfield_path != str(paths.GENTLE_V5_HFIELD):
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) reads {sub.hfield_path}, but "
                f"paths.GENTLE_V5_HFIELD is {paths.GENTLE_V5_HFIELD}"
            )
        if abs(sub.z_span_m - V5_Z_SPAN_M) > 1e-12:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) declares z_span_m = "
                f"{sub.z_span_m}, decision 0007 says {V5_Z_SPAN_M}. Every slope on "
                "this terrain scales with that number, so the wrong span is a "
                "different world at the same task id."
            )
        measured = float(_np.ptp(load_desert_m(paths.GENTLE_V5_HFIELD, V5_Z_SPAN_M)))
        if abs(measured - V5_Z_SPAN_M) > 1e-6:
            raise AssertionError(
                f"the committed v5 bytes span {measured} m, not decision 0007's "
                f"{V5_Z_SPAN_M}. `terrain/gentle.py` rescales the field until "
                "these are equal by construction, so a disagreement means the "
                "asset was not written by that generator."
            )

        # (c) Two sections, and inside `scene` exactly the two sub-terrain keys.
        base_d, v5_d = base.to_dict(), v5.to_dict()
        moved = sorted(k for k in set(base_d) | set(v5_d) if base_d.get(k) != v5_d.get(k))
        if moved != ["rewards", "scene"]:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) differs from "
                f"Bestiary-Desert-Hound-v0 in {moved}, but it may differ ONLY in "
                "['rewards', 'scene']. 'scene' is unavoidable — the terrain "
                "generator lives there — and everything else is inherited on "
                "purpose: the wheel-aware action split, the observation (a "
                "one-way door), the trunk-contact termination, the commands, the "
                "events and the retargeted body names. Two variables and the run "
                "cannot say which one moved the result."
            )
        scene_paths = _diff_paths(base_d["scene"], v5_d["scene"])
        if scene_paths != V5_SCENE_DIFF_PATHS:
            raise AssertionError(
                f"the forward-v5 Hound task ({arm}) moves these scene fields: "
                f"{scene_paths}. The only legal diff is {V5_SCENE_DIFF_PATHS} — "
                "the desert tile leaving the mix and the v5 gentle tile taking "
                "its place. The robot, the height scanner, the contact sensor, "
                "the env count and the spacing are the desert task's; moving one "
                "of them here hides a second change inside a terrain swap, which "
                "is the quietest failure this repository has "
                "(CLAUDE.md, the terrain invariant)."
            )


#: The commanded long run's envelope, dead zones, standing fraction and kernel
#: width — PINNED HERE, independent of the module that declares them.
#:
#: Pinned rather than imported, for the reason `check_spyder.py` pins the fast
#: task's ranges: an oracle whose expectation is read out of the module it checks
#: cannot catch that module changing. `hound_overnight_env_cfg` derives every one
#: of its ranges from three constants in one file, so editing `VX_MAX_MS` there
#: would move the config AND the expectation together and every assertion would
#: stay green — which is the one failure this check most needs to catch, because
#: the envelope is not a free parameter. Widening the box changes what fraction
#: of the tracking kernel is earnable and where the terrain curriculum's demote
#: bar sits; narrowing it changes what the run is for. Either is a new
#: experiment, and it should go red here until the record says why.
OVERNIGHT_COMMAND_RANGES: dict[str, tuple[float, float]] = {
    "lin_vel_x": (-1.5, 1.5),
    "lin_vel_y": (-0.6, 0.6),
    "ang_vel_z": (-1.5, 1.5),
}
OVERNIGHT_DEAD_ZONES: dict[str, float] = {"min_lin_vel_x": 0.25, "min_ang_vel_z": 0.2}
OVERNIGHT_REL_STANDING = 0.1
OVERNIGHT_KERNEL_STD = 0.5

#: The four reward terms the commanded run pays, sorted, and the five it deletes
#: from the Hound desert table. Both pinned; the deletions are also RECOMPUTED
#: from the live desert config inside the check, so an upstream release that adds
#: or renames a term goes red here instead of being silently swallowed by a keep
#: list that is safe by construction.
OVERNIGHT_REWARD_TERMS: tuple[str, ...] = (
    "action_rate_l2",
    "lin_vel_z_l2",
    "track_ang_vel_z_exp",
    "track_lin_vel_xy_exp",
)
OVERNIGHT_DELETED_TERMS: tuple[str, ...] = (
    "ang_vel_xy_l2",
    "dof_acc_l2",
    "dof_acc_wheel_l2",
    "dof_pos_limits",
    "dof_torques_l2",
)

#: Drive-cell tracking income a MOTIONLESS machine may collect in expectation
#: before this oracle goes red. `check_spyder.py` carries the same constant for
#: the same reason and `research/decisions/0005` is the source: 30%. The number
#: it exists to keep out is 62.7% — where the Hound's arm-1 seed 2 sat when it
#: parked and still beat the do-nothing control in 13 of 13 eval cells
#: (`research/measurements/isaac_hound_arm1_s2.json`).
STANDING_SHARE_FLAG = 0.30

#: Top-level config sections the commanded run may differ in, against each of the
#: two baselines it is checked against. Four against the desert task it descends
#: from (`scene` is unavoidable — the terrain generator lives there) and THREE
#: against the forward-v5 probe, whose ground it must match exactly.
OVERNIGHT_SECTIONS_VS_DESERT = ["commands", "curriculum", "rewards", "scene"]
OVERNIGHT_SECTIONS_VS_FORWARD_V5 = ["commands", "curriculum", "rewards"]


def _dead_zone_mean_kernel(dz: float, hi: float, std: float) -> float:
    """E[exp(-(c/std)^2)] for |c| ~ U(dz, hi) — what a machine holding zero on
    that channel still earns. Closed form via erf; symmetric in the sign.

    Deliberately a duplicate of `check_spyder.py`'s function of the same name,
    for the reason `_diff_paths` above is a duplicate: the two oracles gate
    launches independently, and a bad edit in one must not be able to take the
    other down with it.
    """
    if hi <= dz:
        raise ValueError(f"empty dead-zone range [{dz}, {hi}]")
    return (std * math.sqrt(math.pi) / (2.0 * (hi - dz))) * (
        math.erf(hi / std) - math.erf(dz / std)
    )


def _standing_share(cmd, lin, ang, std: float) -> float:
    """Expected share of DRIVE-CELL tracking income a motionless machine earns.

    The two channels are shaped differently and that is the whole arithmetic: the
    linear dead zone is a magnitude RESAMPLE, so a parked machine faces
    |v_x| ~ U(dz, hi); the yaw dead zone is a SNAP, so a fraction dz/hi of driving
    envs carry w_z == 0 exactly — straight drivers, where a motionless machine
    scores the full yaw kernel — and the survivors are uniform on ±[dz, hi]. The
    policy-step dt cancels, so it is not a parameter.

    `std` is passed rather than read off the terms so the check can price the
    SAME configuration at a counterfactual kernel width — the 0.75 that would
    preserve upstream's std/range ratio against this ±1.5 box, and that
    `hound_overnight_env_cfg.py` argues against.

    Standing ENVS are excluded throughout: a machine commanded to stand and
    standing is earning honestly.
    """
    vx_dz, vx_hi = cmd.min_lin_vel_x, float(cmd.ranges.lin_vel_x[1])
    wz_dz, wz_hi = cmd.min_ang_vel_z, float(cmd.ranges.ang_vel_z[1])
    stand_lin = _dead_zone_mean_kernel(vx_dz, vx_hi, std)
    p_straight = wz_dz / wz_hi
    stand_ang = p_straight * 1.0 + (1.0 - p_straight) * _dead_zone_mean_kernel(
        wz_dz, wz_hi, std
    )
    return (lin.weight * stand_lin + ang.weight * stand_ang) / (lin.weight + ang.weight)


def measured_span(sub) -> float:
    """Metres of relief a sub-terrain's committed bytes actually span.

    A one-line helper so the dump below can print the measurement beside the
    declaration; the ASSERTION on the same quantity lives in
    `_assert_overnight_variant`, where a mismatch is a failure rather than a
    printed curiosity.
    """
    from bestiary.terrain.isaac_hf import load_desert_m

    return float(np.ptp(load_desert_m(sub.hfield_path, sub.z_span_m)))


def _assert_overnight_variant(cfg, desert_base, v5_base, arm: str) -> None:
    """One commanded-Hound config against its two baselines. Eight assertions.

    `arm` says which of the pair is being read ("train" or "play"). Both are
    checked, and that is not decoration: `HoundOvernightEnvCfg_PLAY` descends
    from the DESERT Play class, so it gets its whole surgery from a second call
    to `apply_overnight`, and an edit that reached one call and not the other
    would give the viewer a policy driven under a different reward, a different
    command distribution or a different world than the one that trained.

    Every assertion catches something the others cannot, and each one is a
    failure that otherwise TRAINS:

      (a) The reward table is exactly the four declared names, each term
          byte-identical to the desert task's own — so a weight typed by hand, a
          param dropped with a retarget, or an upstream term surviving the keep
          list all fail here.
      (b) No live term reads contact TIMING, under any name. Structural, and it
          needs no articulation: this body has no feet, so a contact-timing term
          is wrong at every scope, not merely wrong on the wheels.
      (c) The command term is the dead-zone sampler with the lazy `class_type`,
          heading off, and exactly the pinned ranges, dead zones and standing
          fraction — and the DESERT baseline is not already dead-zoned, so the
          comparison is not a config against itself.
      (d) Both kernel widths are the pinned 0.5 AND equal the desert task's own,
          so the deliberate non-rescaling is asserted against its source.
      (e) The terrain curriculum is the arc bar, and the desert baseline's is not.
      (f) The ground is the committed v5 asset at decision 0007's span, measured
          from the bytes, with the desert tile gone from the mix.
      (g) Against the DESERT task: four sections move, and inside `scene` exactly
          the two sub-terrain keys.
      (h) Against the FORWARD-V5 probe: three sections move and `scene` does not
          move at all — the strongest available statement that this run stands on
          byte-identical ground to the arm that already ran here.
    """
    from bestiary import paths
    from bestiary.isaac.commands import DeadZoneVelocityCommandCfg
    from bestiary.isaac.curriculums import terrain_levels_vel_arc
    from bestiary.isaac.hound_overnight_env_cfg import FORBIDDEN_CADENCE_TERM
    from bestiary.isaac.spyder_forward_v5_env_cfg import GENTLE_SUBTERRAIN_KEY
    from bestiary.terrain.isaac_hf import load_desert_m

    base_d, var_d = desert_base.to_dict(), cfg.to_dict()

    # (a) The reward table: the declared names, at the desert table's own values.
    live = {
        name: term
        for name, term in var_d["rewards"].items()
        if not name.startswith("_") and term is not None
    }
    if sorted(live) != sorted(OVERNIGHT_REWARD_TERMS):
        raise AssertionError(
            f"the overnight Hound task ({arm}) pays {sorted(live)}, but its "
            f"declared table is {sorted(OVERNIGHT_REWARD_TERMS)}. An extra term is "
            "a reward nobody wrote down; a missing one makes this a different task "
            "than the record says trained."
        )
    for name, term in sorted(live.items()):
        base_term = base_d["rewards"].get(name)
        if base_term is None:
            raise AssertionError(
                f"the overnight Hound task ({arm}) pays {name!r}, which "
                "Bestiary-Desert-Hound-v0 does not have live — it invented a reward "
                "term, so 'at the Hound table's weights' is meaningless for it."
            )
        if term != base_term:
            raise AssertionError(
                f"the overnight Hound task ({arm}) term {name!r} is {term}, the "
                f"desert task's is {base_term}. A keep-list variant chooses WHICH "
                "terms are paid, never what a term should cost."
            )

    # (b) No contact-timing reward survives, under any name. `feet_air_time` is
    # named separately because its absence is this task's one deliberate
    # departure from the Spyder overnight's recipe, and a check that only tested
    # the structural property would let it back in under a rename.
    terms = _live_reward_terms(cfg)
    if FORBIDDEN_CADENCE_TERM in terms:
        raise AssertionError(
            f"the overnight Hound task ({arm}) has {FORBIDDEN_CADENCE_TERM!r} live "
            f"at weight {terms[FORBIDDEN_CADENCE_TERM].weight}. On a driven hub "
            "wheel it pays the machine to HOP — air time can only be earned by "
            "breaking the rolling contact the wheel exists to keep. Its absence is "
            "the one deliberate difference from Bestiary-Overnight-Spyder-v0, and "
            "restoring a cadence incentive here needs a NEW term with the opposite "
            "sign, not this one (research/decisions/0004 Part B, 0005's gate)."
        )
    timing_offenders = sorted(
        f"{name} reads {_reads(term.func, CONTACT_TIMING_FIELDS)}"
        for name, term in terms.items()
        if _reads(term.func, CONTACT_TIMING_FIELDS)
    )
    if timing_offenders:
        raise AssertionError(
            f"the overnight Hound task ({arm}) pays for contact TIMING: "
            f"{timing_offenders}. This body's feet are wheels, so a timing term is "
            "wrong at every scope — re-scoping it onto other bodies does not repair "
            "the incentive."
        )

    # (c) The command sampler, and that it really replaced something else.
    cmd = cfg.commands.base_velocity
    if not isinstance(cmd, DeadZoneVelocityCommandCfg):
        raise AssertionError(
            f"the overnight Hound task ({arm}) commands through "
            f"{type(cmd).__name__}, not DeadZoneVelocityCommandCfg. The plain "
            "sampler draws v_x uniformly over a symmetric range, so a large share "
            "of commands sit where standing is nearly the right answer — the "
            "distribution that produced the parked arm-1 seed."
        )
    if isinstance(desert_base.commands.base_velocity, DeadZoneVelocityCommandCfg):
        raise AssertionError(
            "Bestiary-Desert-Hound-v0 is ALREADY using the dead-zone sampler, so "
            "this check is comparing a config against itself. If the desert task "
            "adopted it deliberately, this check needs a different baseline; if it "
            "did not, something rewrote a shared config object."
        )
    if not (
        isinstance(cmd.class_type, str)
        and str(cmd.class_type).endswith("commands_impl:DeadZoneVelocityCommand")
    ):
        raise AssertionError(
            f"the overnight Hound task ({arm}) has class_type {cmd.class_type!r} — "
            "it must be the lazy string "
            "'bestiary.isaac.commands_impl:DeadZoneVelocityCommand'. An eager class "
            "object imports the runtime chain (and the pip pxr) before Kit boots, "
            "which heap-corrupts the app 1.5 s into every launch."
        )
    facts = {
        "heading_command": (cmd.heading_command, False),
        "rel_standing_envs": (cmd.rel_standing_envs, OVERNIGHT_REL_STANDING),
        "min_lin_vel_x": (cmd.min_lin_vel_x, OVERNIGHT_DEAD_ZONES["min_lin_vel_x"]),
        "min_ang_vel_z": (cmd.min_ang_vel_z, OVERNIGHT_DEAD_ZONES["min_ang_vel_z"]),
    }
    for name, want in OVERNIGHT_COMMAND_RANGES.items():
        facts[name] = (tuple(getattr(cmd.ranges, name)), want)
    wrong = {k: f"{got!r} (want {want!r})" for k, (got, want) in facts.items() if got != want}
    if wrong:
        raise AssertionError(
            f"the overnight Hound task ({arm}) command config drifted from the "
            f"pinned design: {wrong}. The envelope and both dead zones are what "
            "make every driving command distinguishable from standing, and where "
            "the terrain curriculum's demote bar sits."
        )

    # (d) The kernel widths, against the pin AND against the desert task's own.
    for term_name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
        got = float(getattr(cfg.rewards, term_name).params["std"])
        inherited = float(getattr(desert_base.rewards, term_name).params["std"])
        if got != OVERNIGHT_KERNEL_STD or inherited != OVERNIGHT_KERNEL_STD:
            raise AssertionError(
                f"the overnight Hound task ({arm}) has {term_name} std {got} and "
                f"the desert task has {inherited}; this oracle pins "
                f"{OVERNIGHT_KERNEL_STD}. Rescaling the kernel with the widened "
                "range (0.75 preserves upstream's 0.5 ratio) takes standing's "
                "expected share of drive-cell income from 21.4% to 37.2%, past "
                f"the {STANDING_SHARE_FLAG:.0%} flag — which is the door the whole "
                "dead-zone stack exists to keep shut."
            )

    # (e) The terrain curriculum, and that it really replaced something else.
    if cfg.curriculum.terrain_levels.func is not terrain_levels_vel_arc:
        raise AssertionError(
            f"the overnight Hound task ({arm}) curriculum is "
            f"{getattr(cfg.curriculum.terrain_levels.func, '__name__', '?')!r}, not "
            "terrain_levels_vel_arc. Upstream's demote bar is straight-line "
            "kinematics applied to an arc, so a perfect tracker of a turning "
            "command is demoted every episode while a yaw-blind straight driver is "
            "promoted — learnings/015's failure taught on purpose, on a task whose "
            "whole point is obedience."
        )
    if desert_base.curriculum.terrain_levels.func is terrain_levels_vel_arc:
        raise AssertionError(
            "Bestiary-Desert-Hound-v0 ALREADY uses the arc bar, so the curriculum "
            "half of this check compares a config against itself. If the desert "
            "task adopted it, this check needs a different baseline."
        )

    # (f) The ground is the committed v5 asset, and the desert is gone.
    gen = cfg.scene.terrain.terrain_generator
    sub = gen.sub_terrains.get(GENTLE_SUBTERRAIN_KEY)
    if sub is None:
        raise AssertionError(
            f"the overnight Hound task ({arm}) has no {GENTLE_SUBTERRAIN_KEY!r} "
            f"sub-terrain; it carries {sorted(gen.sub_terrains)}. The terrain swap "
            "did not run."
        )
    if "bestiary_desert" in gen.sub_terrains:
        raise AssertionError(
            f"the overnight Hound task ({arm}) still carries a 'bestiary_desert' "
            "tile. Half its tiles would be the 5.05 m desert, which is not the "
            "ground research/decisions/0007 puts new arms on."
        )
    if not sub.hfield_path.endswith(V5_HFIELD_BASENAME):
        raise AssertionError(
            f"the overnight Hound task ({arm}) reads {sub.hfield_path}, whose name "
            f"is not {V5_HFIELD_BASENAME!r}."
        )
    if sub.hfield_path != str(paths.GENTLE_V5_HFIELD):
        raise AssertionError(
            f"the overnight Hound task ({arm}) reads {sub.hfield_path}, but "
            f"paths.GENTLE_V5_HFIELD is {paths.GENTLE_V5_HFIELD}"
        )
    if abs(sub.z_span_m - V5_Z_SPAN_M) > 1e-12:
        raise AssertionError(
            f"the overnight Hound task ({arm}) declares z_span_m = {sub.z_span_m}, "
            f"decision 0007 says {V5_Z_SPAN_M}. Every slope on this terrain scales "
            "with that number."
        )
    measured = float(np.ptp(load_desert_m(paths.GENTLE_V5_HFIELD, V5_Z_SPAN_M)))
    if abs(measured - V5_Z_SPAN_M) > 1e-6:
        raise AssertionError(
            f"the committed v5 bytes span {measured} m, not decision 0007's "
            f"{V5_Z_SPAN_M}."
        )

    # (g) Against the desert task: four sections, and inside `scene` two keys.
    moved = sorted(k for k in set(base_d) | set(var_d) if base_d.get(k) != var_d.get(k))
    if moved != OVERNIGHT_SECTIONS_VS_DESERT:
        raise AssertionError(
            f"the overnight Hound task ({arm}) differs from "
            f"Bestiary-Desert-Hound-v0 in {moved}, but it may differ ONLY in "
            f"{OVERNIGHT_SECTIONS_VS_DESERT}. The observation (243 wide — a ONE-WAY "
            "door, and the reason a multi-hour run can be started at all), the "
            "wheel-aware action split, the trunk-contact termination, the events "
            "and the reset scatter are all inherited on purpose."
        )
    scene_paths = _diff_paths(base_d["scene"], var_d["scene"])
    if scene_paths != V5_SCENE_DIFF_PATHS:
        raise AssertionError(
            f"the overnight Hound task ({arm}) moves these scene fields: "
            f"{scene_paths}. The only legal diff is {V5_SCENE_DIFF_PATHS} — the "
            "desert tile leaving the mix and the v5 gentle tile taking its place. "
            "Moving the robot, the height scanner, the contact sensor, the env "
            "count or the spacing here hides a second change inside a terrain "
            "swap, which is the quietest failure this repository has "
            "(CLAUDE.md, the terrain invariant)."
        )

    # (h) Against the forward-v5 probe: the ground does not move AT ALL.
    fwd_d = v5_base.to_dict()
    moved_v5 = sorted(k for k in set(fwd_d) | set(var_d) if fwd_d.get(k) != var_d.get(k))
    if moved_v5 != OVERNIGHT_SECTIONS_VS_FORWARD_V5:
        raise AssertionError(
            f"the overnight Hound task ({arm}) differs from "
            f"Bestiary-ForwardV5-Hound-v0 in {moved_v5}, but it may differ ONLY in "
            f"{OVERNIGHT_SECTIONS_VS_FORWARD_V5}. `scene` in particular must be "
            "byte-identical: these two runs are the same body at the same env "
            "count, and 'same ground' is the fact that makes anything measured "
            "across them comparable at all."
        )


def check_overnight_task_is_the_commanded_hound() -> None:
    """`Bestiary-Overnight-Hound-v0` is the declared table, envelope and ground.

    The commanded Hound (`hound_overnight_env_cfg.py`) applies the Spyder's
    steering pipeline to the wheel-legged body: a reward cut to command-tracking
    income plus `action_rate_l2` and `lin_vel_z_l2`, the dead-zone sampler on the
    ±1.5 / ±0.6 / ±1.5 envelope with heading mode off, the arc-corrected terrain
    curriculum, and the v5 ground the forward probe already ran on.

    It is a PRODUCTION run with no arm beside it, which raises the stakes on this
    check rather than lowering them: a ladder arm under the wrong reward costs 46
    minutes and is caught by the arm next to it; this one runs for hours and is
    the only thing that will be measured.

    Eight assertions per config in `_assert_overnight_variant`, on both the
    training config and the Play twin, plus four this function does itself:

      * THE PINS AND THE DECLARATION AGREE. This file's six numbers and
        `hound_overnight_env_cfg`'s are two independent statements of the same
        envelope, and this is what makes editing `VX_MAX_MS` a RED check rather
        than a silently self-consistent one.
      * THE DELETIONS ARE THE DECLARED ONES, recomputed from the LIVE desert
        table rather than trusted from a tuple. The keep-list surgery deletes
        whatever is live and not kept — safe by construction, and silent if
        Isaac Lab ships a tenth reward term. This is the loud half.
      * THE MONEY. Standing's expected share of drive-cell income is computed and
        flagged past 30%, and the counterfactual at the ratio-preserving std of
        0.75 is printed beside it so the docstring's central argument is a
        number this oracle produces rather than a claim it repeats.
      * NON-VACUOUSNESS. Every assertion above is a comparison; a comparison over
        an empty table or against an identical baseline passes silently, which is
        `research/learnings/014`'s shape exactly.

    No simulator is needed: every config here constructs pre-app, the same
    property `commands_impl.py` depends on.
    """
    from bestiary.isaac.curriculums import arc_displacement_m
    from bestiary.isaac.hound_cfg import wheel_action_scale
    from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg, HoundDesertEnvCfg_PLAY
    from bestiary.isaac.hound_forward_v5_env_cfg import (
        HoundForwardV5EnvCfg,
        HoundForwardV5EnvCfg_PLAY,
    )
    from bestiary.isaac.hound_overnight_env_cfg import (
        EXPECTED_DELETED_TERMS,
        KERNEL_STD,
        OVERNIGHT_RANGES,
        OVERNIGHT_TERMS,
        REL_STANDING,
        VX_MIN_MS,
        WZ_MIN_RADS,
        HoundOvernightEnvCfg,
        HoundOvernightEnvCfg_PLAY,
    )
    from bestiary.isaac.spyder_ladder_env_cfg import live_reward_names
    from bestiary.robots.hound.build import SPEC

    # The pins and the module's own declaration: two statements of one design.
    declared = {
        "ranges": {name: tuple(value) for name, value in OVERNIGHT_RANGES.items()},
        "dead zones": {"min_lin_vel_x": VX_MIN_MS, "min_ang_vel_z": WZ_MIN_RADS},
        "rel_standing_envs": REL_STANDING,
        "kernel std": KERNEL_STD,
        "reward terms": tuple(sorted(OVERNIGHT_TERMS)),
        "deleted terms": tuple(sorted(EXPECTED_DELETED_TERMS)),
    }
    pinned = {
        "ranges": OVERNIGHT_COMMAND_RANGES,
        "dead zones": OVERNIGHT_DEAD_ZONES,
        "rel_standing_envs": OVERNIGHT_REL_STANDING,
        "kernel std": OVERNIGHT_KERNEL_STD,
        "reward terms": tuple(sorted(OVERNIGHT_REWARD_TERMS)),
        "deleted terms": tuple(sorted(OVERNIGHT_DELETED_TERMS)),
    }
    drifted = {k: (declared[k], pinned[k]) for k in pinned if declared[k] != pinned[k]}
    if drifted:
        raise AssertionError(
            f"`hound_overnight_env_cfg` declares {[(k, v[0]) for k, v in drifted.items()]}; "
            f"this oracle pins {[(k, v[1]) for k, v in drifted.items()]}. The "
            "envelope, the dead zones and the reward table are the whole content "
            "of this run — changing one is a new experiment, so update the pin, "
            "the module docstring's arithmetic and the record together."
        )

    desert, desert_play = HoundDesertEnvCfg(), HoundDesertEnvCfg_PLAY()
    train_cfg = HoundOvernightEnvCfg()
    _assert_overnight_variant(train_cfg, desert, HoundForwardV5EnvCfg(), "train")
    _assert_overnight_variant(
        HoundOvernightEnvCfg_PLAY(), desert_play, HoundForwardV5EnvCfg_PLAY(), "play"
    )

    # The deletions, recomputed from the live desert table.
    desert_live = live_reward_names(desert.rewards)
    deleted = sorted(desert_live - set(OVERNIGHT_TERMS))
    if not deleted:
        raise AssertionError(
            f"the overnight table {sorted(OVERNIGHT_TERMS)} deletes NOTHING from "
            f"the desert table {sorted(desert_live)} — it is the desert task "
            "relabelled, and every assertion above compares a config against "
            "itself."
        )
    if deleted != sorted(OVERNIGHT_DELETED_TERMS):
        raise AssertionError(
            f"the overnight Hound task deletes {deleted}, but this oracle pins "
            f"{sorted(OVERNIGHT_DELETED_TERMS)}. The keep-list surgery has already "
            "deleted the difference silently and correctly — this is the "
            "notification, not the failure. Either Isaac Lab's RewardsCfg "
            "gained/renamed a term or HoundRewardsCfg did: write down which, then "
            "update the pin, the module's EXPECTED_DELETED_TERMS and its docstring "
            "enumeration together."
        )

    # -- The dump. Everything a reader needs to know what will train. ----------
    cmd = train_cfg.commands.base_velocity
    lin = train_cfg.rewards.track_lin_vel_xy_exp
    ang = train_cfg.rewards.track_ang_vel_z_exp
    dt = train_cfg.decimation * train_cfg.sim.dt
    income = (lin.weight * 1.0 + ang.weight * 1.0) * dt
    sub = train_cfg.scene.terrain.terrain_generator.sub_terrains["bestiary_gentle"]

    print(
        f"      the commanded Hound, {len(OVERNIGHT_TERMS)} terms "
        f"(income + action_rate_l2 + lin_vel_z_l2; NO cadence term):",
        flush=True,
    )
    for name in sorted(OVERNIGHT_TERMS):
        term = getattr(train_cfg.rewards, name)
        std = term.params.get("std")
        extra = f"   std {std}" if std is not None else ""
        print(f"        {name:<24} {term.weight:+g}{extra}", flush=True)
    print(
        f"        {'deleted (' + str(len(deleted)) + ')':<24} "
        + "  ".join(f"{n} {getattr(desert.rewards, n).weight:+g}" for n in deleted),
        flush=True,
    )
    print(
        f"        {'commands':<24} v_x {tuple(cmd.ranges.lin_vel_x)}  "
        f"v_y {tuple(cmd.ranges.lin_vel_y)}  w_z {tuple(cmd.ranges.ang_vel_z)}",
        flush=True,
    )
    print(
        f"        {'dead zones':<24} |v_x| >= {cmd.min_lin_vel_x} m/s (resample), "
        f"|w_z| < {cmd.min_ang_vel_z} rad/s -> 0 (snap), stand "
        f"{cmd.rel_standing_envs:.0%}, heading {cmd.heading_command}",
        flush=True,
    )
    func = train_cfg.curriculum.terrain_levels.func
    print(
        f"        {'curriculum':<24} {func.__module__}:{func.__name__}",
        flush=True,
    )
    print(
        f"        {'terrain':<24} {sub.hfield_path}  span {sub.z_span_m} m  "
        f"(measured {measured_span(sub):.6f} m)",
        flush=True,
    )

    # Kernel geometry: the ratio this task deliberately does NOT restore.
    corner = math.hypot(
        OVERNIGHT_COMMAND_RANGES["lin_vel_x"][1], OVERNIGHT_COMMAND_RANGES["lin_vel_y"][1]
    )
    ratio_std = 0.5 * OVERNIGHT_COMMAND_RANGES["lin_vel_x"][1]
    print(
        f"      kernel std {KERNEL_STD} unchanged: std/v_max "
        f"{KERNEL_STD / OVERNIGHT_COMMAND_RANGES['lin_vel_x'][1]:.4f}, std/w_max "
        f"{KERNEL_STD / OVERNIGHT_COMMAND_RANGES['ang_vel_z'][1]:.4f}, "
        f"2-D corner {corner:.4f} m/s (std/corner {KERNEL_STD / corner:.4f})",
        flush=True,
    )
    share = _standing_share(cmd, lin, ang, KERNEL_STD)
    share_ratio = _standing_share(cmd, lin, ang, ratio_std)
    print(
        f"      standing share, drive cells {share:.2%} at std {KERNEL_STD} "
        f"vs {share_ratio:.2%} at the ratio-preserving std {ratio_std} "
        f"(flag {STANDING_SHARE_FLAG:.0%}; Hound's parked seed: 63%)",
        flush=True,
    )
    never_strafes = _dead_zone_mean_kernel(
        0.0, OVERNIGHT_COMMAND_RANGES["lin_vel_y"][1], KERNEL_STD
    )
    print(
        f"      strafe is optional (v_y has no dead zone): a perfect v_x tracker "
        f"that never sidesteps still earns {never_strafes:.2%} of the linear kernel"
        f" (Bestiary-Fast-Spyder-v0, ±0.6 at std 0.3: "
        f"{_dead_zone_mean_kernel(0.0, 0.6, 0.3):.2%})",
        flush=True,
    )
    # The worst-case reading of the lateral channel, printed at BOTH lateral
    # ranges the Hound has now commanded. It is a LOWER bound on the linear
    # kernel's ceiling — the machine demonstrably makes some v_y (the desert
    # task's ARM 2 note, measured |v_y|/|v| = 0.332) — and it is the number that
    # `research/learnings/011`'s "charging the unremovable" would be read from.
    ceilings = {
        half: (KERNEL_STD * math.sqrt(math.pi) / (2.0 * half)) * math.erf(half / KERNEL_STD)
        for half in (0.3, OVERNIGHT_COMMAND_RANGES["lin_vel_y"][1])
    }
    print(
        "      worst case, v_y wholly unachievable: linear-kernel ceiling "
        + ", ".join(f"{c:.4f} at ±{h}" for h, c in sorted(ceilings.items())),
        flush=True,
    )

    # The two penalties, at labelled operating points. [ASSUMED] means no Hound
    # policy has been measured at them, so none of these may enter the record.
    dact_rms, vz_rms = 0.1, 0.15
    for label, value, note in (
        (
            f"action_rate_l2 (16 actions, {dact_rms} rms)",
            -abs(train_cfg.rewards.action_rate_l2.weight) * 16 * dact_rms**2 * dt,
            "[ASSUMED]",
        ),
        (
            f"lin_vel_z_l2 ({vz_rms} m/s rms)",
            -abs(train_cfg.rewards.lin_vel_z_l2.weight) * vz_rms**2 * dt,
            "[ASSUMED]",
        ),
    ):
        print(
            f"      {label:<40} {value:+.6f}/step  {abs(value) / income:6.2%} of "
            f"income  {note}",
            flush=True,
        )

    # The curriculum's two bars against what this machine's DRIVE alone can do.
    # This is the number that makes it legitimate to ask this run for rolling:
    # if the demote bar sat above the drive's own reach, the curriculum would
    # require the gallop the forward probe found.
    rim_speed = wheel_action_scale() * SPEC.wheel_r
    episode_s = float(train_cfg.episode_length_s)
    vx_hi = OVERNIGHT_COMMAND_RANGES["lin_vel_x"][1]
    demote_m = 0.5 * float(arc_displacement_m(vx_hi, 0.0, episode_s))
    promote_m = float(train_cfg.scene.terrain.terrain_generator.size[0]) / 2.0
    rolled_m = rim_speed * episode_s
    print(
        f"      curriculum @ top straight cmd: demote below {demote_m:.1f} m, "
        f"promote above {promote_m:.1f} m; the wheel drive alone reaches "
        f"{rim_speed:.4f} m/s = {rolled_m:.2f} m per {episode_s:.0f} s episode",
        flush=True,
    )
    if rolled_m <= demote_m:
        raise AssertionError(
            f"a machine rolling at the drive's saturation speed covers "
            f"{rolled_m:.2f} m per episode, which does NOT clear the {demote_m:.1f} m "
            "demote bar at the top straight command. The terrain curriculum would "
            "then require the machine to gallop in order to be promoted — the mode "
            "the forward-v5 probe already found at 203.56 m/episode — and this task "
            "would be asking for rolling while paying for the opposite. Narrow "
            "lin_vel_x or re-derive the bar."
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
#: Checks that need nothing but the files.
FILE_CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("conversion input is current with build.py", check_conversion_input_is_current),
    ("conversion input is hound16pd's robot, trunk at 0", check_conversion_input_is_the_pd_robot),
    ("converted asset carries no ground plane", check_usd_has_no_ground),
    ("asset selects the physx Physics variant", check_usd_selects_the_physx_variant),
    ("no stance spring reaches the solver", check_no_springs_reach_the_solver),
    ("wheel collider is a cylinder primitive", check_wheel_collider_is_a_cylinder_primitive),
    ("17 colliders, none decorative", check_only_the_load_bearing_geoms_collide),
    ("all 17 bodies report contacts", check_all_bodies_report_contacts),
    ("static friction is supplied by the cfg", check_static_friction_is_supplied_by_the_cfg),
)

#: Checks that need a loaded articulation.
SIM_CHECKS: tuple[tuple[str, Callable[[object], None]], ...] = (
    ("17 bodies with the MJCF's link names", check_body_count),
    ("16 actuated DoF with the MJCF's joint names", check_sixteen_actuated_dof),
    ("12 position + 4 velocity, wheels at stiffness 0", check_actuator_split),
    ("joint limits match Spec (radians, not degrees)", check_joint_limits_match_spec),
    ("default pose is the solved stance", check_default_pose_is_the_solved_stance),
    ("masses match assets/hound16pd.xml", check_masses_match_the_mjcf),
    ("PD gains and torque ceilings match Spec", check_leg_gains_and_torque_ceilings),
    ("armature and joint friction match Spec", check_armature_and_joint_friction),
    ("wheel velocity gain is the derived one", check_wheel_velocity_gain_is_derived),
    # -- The reward table. These need the articulation because every one of them
    #    resolves a body-name or joint-name regex against the REAL names, not
    #    against a list retyped here.
    ("no contact-timing reward on a wheel body", check_no_contact_timing_reward_on_a_wheel),
    ("wheels are not charged for spinning", check_wheel_joints_are_not_charged_for_spinning),
    ("leg velocity limit is Go2's rated speed", check_leg_velocity_limit_is_the_go2_rated_speed),
    ("reward budget priced against 011 and 015", check_reward_budget_against_011_and_015),
)

#: Checks that need only a constructed config — no articulation, no app.
#:
#: A third group rather than a fourth entry in SIM_CHECKS, because these are the
#: only checks in this file that would run on a laptop with no GPU: every config
#: here constructs before the simulation app exists (the same property
#: `commands_impl.py` depends on), so a task-variant claim can be checked at the
#: desk instead of on a rented box. `check_spyder.py` has carried this group
#: since the forward diagnostic; this is its Hound twin.
CFG_CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("forward-v5 task is v_x on v5 ground, nothing else", check_forward_v5_is_vx_on_v5_ground),
    ("overnight task is the commanded Hound", check_overnight_task_is_the_commanded_hound),
)


def _run(name: str, fn: Callable[[], None]) -> int:
    """Run one check. A check reports; it never crashes the suite."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  {name}", flush=True)
        print(f"      {type(exc).__name__}: {exc}", flush=True)
        if not isinstance(exc, AssertionError):
            traceback.print_exc()
        return 1
    print(f"ok    {name}", flush=True)
    return 0


def main() -> int:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from bestiary import paths
    from bestiary.isaac.hound_cfg import HOUND16_CFG

    print(f"[bestiary] MJCF : {paths.HOUND_PD_XML}", flush=True)
    print(f"[bestiary] input: {paths.HOUND_ISAAC_MJCF}", flush=True)
    print(f"[bestiary] USD  : {paths.HOUND_ISAAC_USD}", flush=True)
    print(flush=True)

    failures = sum(_run(name, fn) for name, fn in FILE_CHECKS)

    # One robot, no ground, no terrain: these checks read the articulation's
    # parsed properties, and none of them needs the machine to stand on anything.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    robot = Articulation(HOUND16_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    failures += sum(_run(name, lambda fn=fn: fn(robot)) for name, fn in SIM_CHECKS)
    failures += sum(_run(name, fn) for name, fn in CFG_CHECKS)

    total = len(FILE_CHECKS) + len(SIM_CHECKS) + len(CFG_CHECKS)
    print(f"\n{total - failures}/{total} checks pass", flush=True)
    return 1 if failures else 0


def _exit(status: int) -> None:
    """Leave the process with `status`, and actually mean it.

    `SimulationApp.close()` ENDS THE PROCESS ITSELF, with status 0. Measured:
    a script that prints, calls `close()`, and then calls `os._exit(7)` never
    reaches the second line and exits 0. So `sys.exit(1)` placed after `close()`
    -- the obvious shape, and the one this file had first -- makes a failing
    oracle report success, which is worse than having no oracle.

    Therefore: flush, then `os._exit`, and no `close()` at all. Skipping Kit's
    teardown costs nothing in a headless process that is about to die anyway, and
    it is the only path Kit's shutdown cannot overwrite. The same trap explains
    why every print in this file passes `flush=True`.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    args = _parse_args()
    AppLauncher(args)
    try:
        _exit(main())
    except Exception:
        traceback.print_exc()
        _exit(1)
