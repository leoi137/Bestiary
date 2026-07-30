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

    total = len(FILE_CHECKS) + len(SIM_CHECKS)
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
