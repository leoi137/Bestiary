"""The Spyder-12 Isaac port oracle: every way this port could be quietly wrong.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.check_spyder

Run after ANY change to `assets/spyder12.xml`, `spyder_usd.py`, `spyder_cfg.py`,
`commands.py`, `spyder_gentle_env_cfg.py`, `rewards.py` or
`spyder_forward_env_cfg.py`, and before every training launch.

Three groups, mirroring `check_hound.py`'s structure because the failure modes
are the Hound port's failure modes minus the wheels:

  1. FILE checks — the committed conversion input is still what the transform
     produces, the USD has no world, the physx variant is selected, all 13
     bodies report contacts, exactly the 17 load-bearing geoms collide, the
     gentle terrain asset parses.
  2. SIM checks — the loaded articulation has Spyder's bodies, joints, limits,
     masses, gains, effort ceilings and armature, and nothing extra arrived.
  3. CFG checks — the training config's commands are dead-zoned with heading
     off, the kernel widths preserve upstream's discrimination ratio, every
     retargeted regex resolves on this robot, and the money adds up: standing
     earns a bounded fraction of income and the penalty budget stays under the
     30% flag (`research/decisions/0005`'s rule, `learnings/011` and `015` the
     failures it exists to keep from repeating). Last in this group, the
     forward-velocity DIAGNOSTIC variant is asserted to carry exactly one
     reward term and to differ from the training config in nothing else — a
     one-variable experiment is only one variable if something checks.

The one check this file CANNOT make: whether the policy actually drives. That
is `vx_span_ratio` and the per-cell grid, measured after training — an oracle
asserts the machine and its incentives, never the outcome.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from typing import Callable

import numpy as np

from isaaclab.app import AppLauncher

#: Tolerances, same meanings as check_hound's.
GAIN_TOL = 1e-6
ANGLE_TOL_RAD = 1e-5
#: A joint limit past this magnitude counts as "unlimited" (PhysX reports
#: FLT_MAX, Newton 1e10 — both far past any real range).
UNLIMITED_RAD = 1e6

#: Penalty budget flag, as a fraction of achievable tracking income.
#: `research/decisions/0005`'s 30% rule, same constant as check_hound.
PENALTY_BUDGET_FLAG_FRACTION = 0.30

#: Standing-share flag: the drive-cell income a MOTIONLESS machine may collect
#: in expectation before this oracle goes red. The Hound's inherited settings
#: put it at 63% and one seed in three parked
#: (`research/measurements/isaac_hound_arm1_s2.json`); the dead zones plus
#: rescaled kernels are DESIGNED to put it at 27.2% (the yaw snap's straight
#: drivers hand a parked machine the full yaw kernel on ~25% of drive envs,
#: which is why the v_x floor moved 0.2 -> 0.25 to buy the share back down),
#: and this assertion is what keeps a later "harmless" command-range edit from
#: quietly re-opening the door. Standing ENVS are excluded: a machine
#: commanded to stand and standing is earning honestly.
STANDING_SHARE_FLAG = 0.30

#: Body masses of the compiled MuJoCo model, kg. PINNED, WITH PROVENANCE:
#: computed 2026-08-05 from the committed asset with the local venv
#: (mujoco 3.8.1) —
#:
#:     m = mujoco.MjModel.from_xml_path(str(paths.SPYDER_XML))
#:     m.body("torso").mass   -> 0.4838802
#:     m.body("coxa_1").mass  -> 0.0277840
#:     m.body("femur_1").mass -> 0.0508476
#:     m.body("tibia_1").mass -> 0.0716421
#:
#: Seven significant figures and a 1e-5 tolerance, deliberately: the first
#: draft pinned 5-figure roundings with a 5e-4 tolerance, and an adversarial
#: review computed that three of the four roundings were off in their last
#: digit while the tolerance was 23x the largest error — a check that could
#: never catch the drift it existed for. The total is DERIVED from the pins
#: (torso + 4 per-leg chains), never typed, so the five numbers cannot
#: disagree with each other.
#:
#: Pinned rather than recomputed because THIS interpreter deliberately has no
#: mujoco (`terrain/isaac_hf.py` explains the independence); asserting the USD
#: against numbers a different toolchain computed from the same source file is
#: the point, not a compromise. Re-run the command above if spyder12.xml's
#: geometry or density ever changes.
MJCF_BODY_MASS_KG = {
    "torso": 0.4838802,
    "coxa": 0.0277840,
    "femur": 0.0508476,
    "tibia": 0.0716421,
}
MJCF_TOTAL_MASS_KG = MJCF_BODY_MASS_KG["torso"] + 4 * (
    MJCF_BODY_MASS_KG["coxa"] + MJCF_BODY_MASS_KG["femur"] + MJCF_BODY_MASS_KG["tibia"]
)
MASS_TOL_KG = 1e-5

#: Authored joint ranges, degrees, from assets/spyder12.xml. The importer
#: writes limits in degrees; the articulation reports radians. A factor of
#: 57.2958 anywhere downstream is this table read wrong.
JOINT_RANGE_DEG = {"hip": 40.0, "lift": 60.0, "knee": 50.0}

LEGS = ("1", "2", "3", "4")
PARTS = ("hip", "lift", "knee")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    return parser.parse_args()


def _to_numpy(x) -> np.ndarray:
    """Isaac Lab 3.0 `.data.*` as numpy: torch-on-cuda first, then warp."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _usd_stage():
    from pxr import Usd

    from bestiary import paths

    if not paths.SPYDER_ISAAC_USD.is_file():
        raise AssertionError(
            f"{paths.SPYDER_ISAAC_USD} does not exist. Generate it with\n"
            "    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.spyder_usd"
        )
    stage = Usd.Stage.Open(str(paths.SPYDER_ISAAC_USD))
    if stage is None:
        raise AssertionError(f"USD failed to open {paths.SPYDER_ISAAC_USD}")
    return stage


# ---------------------------------------------------------------------------
# 1. Files: the generated artifacts still describe the authored machine.
# ---------------------------------------------------------------------------
def check_conversion_input_is_current() -> None:
    """The committed conversion input is what the transform produces TODAY.

    `spyder12.xml` is hand-authored, so nothing regenerates the conversion
    input when it changes — this string comparison is the only thing standing
    between an edited robot and a stale USD that still loads cleanly.
    """
    from bestiary import paths
    from bestiary.isaac.spyder_usd import conversion_mjcf

    if not paths.SPYDER_ISAAC_MJCF.is_file():
        raise AssertionError(f"{paths.SPYDER_ISAAC_MJCF} does not exist; run spyder_usd")
    committed = paths.SPYDER_ISAAC_MJCF.read_text()
    current = conversion_mjcf()
    if committed != current:
        raise AssertionError(
            "the committed conversion input differs from what spyder_usd."
            "conversion_mjcf() produces from the live spyder12.xml. Either the "
            "authored robot changed (re-run spyder_usd, re-convert, re-commit) "
            "or the transform changed without regenerating. Either way the USD "
            "may describe a machine that no longer exists."
        )


def check_usd_has_no_ground() -> None:
    """No plane, no floor: one surviving plane is N planes, cloned per env."""
    stage = _usd_stage()
    offenders = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetTypeName() == "Plane" or prim.GetName().lower() in {"floor", "ground"}
    ]
    if offenders:
        raise AssertionError(
            f"the converted asset contains ground geometry: {offenders}. "
            "spyder_usd delta 1 strips the world; it did not run, or it missed."
        )


def check_usd_selects_the_physx_variant() -> None:
    """The Physics variant selection is `physx`, not `mujoco`."""
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
        raise AssertionError(f"the asset selects {selection!r}, not 'physx'")


def check_no_spring_and_no_stray_drive_reaches_the_solver() -> None:
    """The zeroed springs actually took, read from the layer that holds them.

    Spyder's trap is nastier than the Hound's because of a coincidence: the
    authored MuJoCo spring is 30 N·m/rad and `spyder_cfg.KP` is ALSO 30 (the
    cfg adopts the spring's constant deliberately). A value assertion on the
    loaded gain cannot tell "drive only" from "drive plus leaked spring", so
    the assertion has to happen at the asset level — and it has to happen in
    the RIGHT LAYER. The first draft of this check traversed the composed
    stage for `mjc:spring*` attributes, and an adversarial review proved that
    scan dead twice over: the importer writes the spring CONSTANT as
    `mjc:stiffness` (`mjc:springref` is the rest angle), and every `mjc:*`
    attribute lives only in the `mujoco` variant's payload layer, which the
    composed stage — selected to `physx` — never composes. A green result
    from that scan was `research/learnings/014`'s green guard that checked
    nothing.

    So: open `payloads/Physics/mujoco.usda` DIRECTLY. The importer authors
    `mjc:*` attributes ONLY FOR NONZERO VALUES (measured 2026-08-05: this
    asset's payload has zero `mjc:stiffness` entries but the Hound's — whose
    conversion input keeps nonzero springs — carries `mjc:stiffness = 32.77`),
    so "springs zeroed" manifests as ABSENCE, and a bare absence check is
    exactly 014's vacuous green. The non-vacuousness anchor is `mjc:armature`:
    authored on all twelve joints in the same layer, same namespace, same
    importer, because armature is nonzero by design (it is dynamics, not a
    crutch — spyder_usd delta 3). Twelve visible armatures prove the scan
    works; zero visible spring attributes then prove delta 3 took. Then assert
    no authored `physics:drive` stiffness exists in the composed asset at all:
    the input has no `<position>` actuators, so an authored drive would be the
    importer inventing one to stack with the cfg's.
    """
    from pxr import Usd

    from bestiary import paths

    mj_layer = paths.SPYDER_ISAAC_USD.parent / "payloads" / "Physics" / "mujoco.usda"
    if not mj_layer.is_file():
        raise AssertionError(
            f"{mj_layer} does not exist — the importer's layer layout changed, "
            "and this check no longer knows where the MuJoCo joint attributes "
            "live. Find them before trusting any spring assertion."
        )
    mj = Usd.Stage.Open(str(mj_layer))
    if mj is None:
        raise AssertionError(f"USD failed to open {mj_layer}")
    armatures: list[str] = []
    springs: list[str] = []
    # TraverseAll, not Traverse: the payload holds `over` specs for prims
    # defined in a sibling layer, and a plain Traverse of the layer alone
    # visits zero of them (measured — the canary caught exactly this).
    for prim in mj.TraverseAll():
        for attr in prim.GetAttributes():
            if not attr.HasAuthoredValue():
                continue
            if attr.GetName() == "mjc:armature":
                armatures.append(f"{prim.GetPath()}")
            elif attr.GetName() in ("mjc:stiffness", "mjc:springref"):
                springs.append(f"{prim.GetPath()}.{attr.GetName()} = {float(attr.Get())}")
    if len(armatures) != 12:
        raise AssertionError(
            f"the canary failed: {len(armatures)} authored mjc:armature "
            "attributes in the mujoco payload, expected 12. Either the importer "
            "stopped writing mjc:* attributes there — in which case this check "
            "is no longer looking at the springs (research/learnings/014) — or "
            "the armature was zeroed out of the conversion input, which delta 3 "
            "explicitly must not do."
        )
    if springs:
        raise AssertionError(
            f"authored MuJoCo spring attributes in {mj_layer.name}: "
            f"{springs[:4]} ({len(springs)} total). The importer only authors "
            "nonzero values, so delta 3 did not take; with KP == the authored "
            "spring constant, no gain assertion downstream can see this."
        )

    stage = _usd_stage()
    authored_drives = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if (attr := prim.GetAttribute("drive:angular:physics:stiffness"))
        and attr.HasAuthoredValue()
        and float(attr.Get()) != 0.0
    ]
    if authored_drives:
        raise AssertionError(
            f"the asset authors nonzero drive stiffness on {authored_drives[:4]} "
            f"({len(authored_drives)} total), but the conversion input has no "
            "<position> actuators — the importer invented a drive, and it will "
            "stack with spyder_cfg's."
        )


def check_all_bodies_report_contacts() -> None:
    """All 13 rigid bodies carry PhysxContactReportAPI.

    The spawn flag stops descending at the first rigid body of an MJCF import
    (nested tree), so only the asset-level patch makes the contact sensor see
    the tibias — without it, `feet_air_time`'s regex resolves nothing and env
    construction dies blaming the regex.
    """
    from pxr import UsdPhysics

    stage = _usd_stage()
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 13:
        raise AssertionError(
            f"{len(bodies)} rigid bodies in the asset, expected 13: "
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
            "`-m bestiary.isaac.spyder_usd`."
        )


def check_only_the_load_bearing_geoms_collide() -> None:
    """Seventeen colliders — sphere + 16 capsules — and none on the shell.

    The visual shell is 25 mesh geoms at contype=0 conaffinity=0 density=0;
    `collision_from_visuals=False` keeps them decorative. 17 = torso_geom +
    4 stubs + 4 coxa + 4 femur + 4 tibia, the same physics
    `robots/spyder/check.py` holds for the MuJoCo model.
    """
    from pxr import UsdPhysics

    stage = _usd_stage()
    colliders = sorted(
        prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    expected = sorted(
        ["torso_geom"]
        + [f"stub_{leg}" for leg in LEGS]
        + [f"{part}_{leg}_geom" for leg in LEGS for part in ("coxa", "femur", "tibia")]
    )
    if colliders != expected:
        raise AssertionError(
            f"colliders are not the 17 load-bearing geoms.\n  got:      "
            f"{colliders}\n  expected: {expected}"
        )


def check_static_friction_is_supplied_by_the_cfg() -> None:
    """The asset authors no static friction; spyder_cfg puts back 1.0.

    The importer writes only dynamicFriction; the schema defaults static to 0
    and the machine does the splits at rest. Both halves asserted, same as the
    Hound — half one so a future importer that starts writing it is noticed,
    half two so the override stays the MJCF's 1.0.
    """
    from pxr import UsdPhysics

    from bestiary.isaac.spyder_cfg import SPYDER12_CFG

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
            f"the asset now authors physics:staticFriction on {authored_static} "
            "— the importer changed, and the cfg override is a second opinion."
        )
    material = SPYDER12_CFG.spawn.physics_material
    if material is None:
        raise AssertionError("SPYDER12_CFG.spawn.physics_material is None: no grip at rest")
    for field in ("static_friction", "dynamic_friction"):
        got = float(getattr(material, field))
        if abs(got - 1.0) > GAIN_TOL:
            raise AssertionError(
                f"physics_material.{field} is {got}; spyder12.xml's sliding "
                "coefficient is 1.0"
            )


def check_effort_ceiling_and_armature_agree_with_the_mjcf() -> None:
    """spyder_cfg's pinned constants still say what the authored XML says.

    The conversion input DROPS the `<motor>` block (delta 3), so gear x
    ctrlrange lives on only as `spyder_cfg.EFFORT_LIMIT_NM` — this is the
    assertion that keeps that constant from silently diverging from the
    machine it summarises. Same for the armature.
    """
    import xml.etree.ElementTree as ET

    from bestiary import paths
    from bestiary.isaac import spyder_cfg

    root = ET.parse(paths.SPYDER_XML).getroot()
    motor = root.find("default/motor")
    joint = root.find("default/joint")
    if motor is None or joint is None:
        raise AssertionError("spyder12.xml's default block lost its motor or joint element")
    gear = float(motor.get("gear"))
    lo, hi = (float(v) for v in motor.get("ctrlrange").split())
    if hi != -lo:
        raise AssertionError(f"authored ctrlrange {lo}..{hi} is not symmetric")
    if abs(gear * hi - spyder_cfg.EFFORT_LIMIT_NM) > GAIN_TOL:
        raise AssertionError(
            f"authored gear x ctrlrange = {gear * hi}, spyder_cfg.EFFORT_LIMIT_NM "
            f"= {spyder_cfg.EFFORT_LIMIT_NM}"
        )
    armature = float(joint.get("armature"))
    if abs(armature - spyder_cfg.ARMATURE) > GAIN_TOL:
        raise AssertionError(
            f"authored armature = {armature}, spyder_cfg.ARMATURE = {spyder_cfg.ARMATURE}"
        )


def check_gentle_asset_parses() -> None:
    """The gentle heightfield exists, parses, and spans exactly its constant.

    `save_hfield_bin` normalises to an exact [0, 1] span, so the metres the
    bridge returns must span exactly `gentle.Z_SPAN_M`. A truncated file, a
    wrong grid or a drifted constant all raise here, on the desk, instead of
    on a rented box mid-preflight.
    """
    from bestiary import paths
    from bestiary.terrain.gentle import Z_SPAN_M
    from bestiary.terrain.isaac_hf import load_desert_m

    z = load_desert_m(paths.GENTLE_HFIELD, Z_SPAN_M)
    span = float(np.ptp(z))
    if abs(span - Z_SPAN_M) > 1e-6:
        raise AssertionError(
            f"gentle terrain spans {span} m, Z_SPAN_M says {Z_SPAN_M}. The "
            "generator rescales to make these equal by construction."
        )


# ---------------------------------------------------------------------------
# 2. The loaded articulation.
# ---------------------------------------------------------------------------
def check_body_count(robot) -> None:
    """Thirteen bodies with the MJCF's names."""
    expected = {"torso"} | {f"{part}_{leg}" for leg in LEGS for part in ("coxa", "femur", "tibia")}
    if set(robot.body_names) != expected or len(robot.body_names) != 13:
        raise AssertionError(
            f"body names are not the MJCF's 13.\n  missing: "
            f"{sorted(expected - set(robot.body_names))}\n  extra:   "
            f"{sorted(set(robot.body_names) - expected)}"
        )


def check_twelve_actuated_dof(robot) -> None:
    """Twelve joints, the MJCF's names, one actuator group covering all."""
    expected = {f"{part}_{leg}" for leg in LEGS for part in PARTS}
    if set(robot.joint_names) != expected or len(robot.joint_names) != 12:
        raise AssertionError(
            f"joint names are not the MJCF's 12.\n  missing: "
            f"{sorted(expected - set(robot.joint_names))}\n  extra:   "
            f"{sorted(set(robot.joint_names) - expected)}"
        )
    groups = {name: list(act.joint_names) for name, act in robot.actuators.items()}
    if set(groups) != {"legs"} or len(groups["legs"]) != 12:
        raise AssertionError(f"expected one 'legs' group of 12, got {groups}")


def check_joint_limits_match_the_mjcf(robot) -> None:
    """±40/±60/±50 degrees, in radians, on the right joints."""
    limits = _to_numpy(robot.data.joint_pos_limits)[0]
    for index, name in enumerate(robot.joint_names):
        part = name.split("_")[0]
        want = math.radians(JOINT_RANGE_DEG[part])
        lo, hi = (float(v) for v in limits[index])
        if abs(lo + want) > ANGLE_TOL_RAD or abs(hi - want) > ANGLE_TOL_RAD:
            raise AssertionError(
                f"{name} limits are [{lo:.6f}, {hi:.6f}] rad, the MJCF says "
                f"±{want:.6f}. A ratio near {180 / math.pi:.4f} means degrees "
                "were read as radians."
            )


def check_default_pose_is_the_stance(robot) -> None:
    """Default joints all zero, root at stand height: zero action = the arch.

    Load-bearing: `use_default_offset=True` makes the default pose the zero
    action, the same prior `envs/spyder.py` gives the SAC policy.
    """
    from bestiary.isaac.spyder_cfg import SPAWN_CLEARANCE_M, STAND_HEIGHT_M

    default = _to_numpy(robot.data.default_joint_pos)[0]
    off = {robot.joint_names[i]: float(v) for i, v in enumerate(default) if abs(v) > ANGLE_TOL_RAD}
    if off:
        raise AssertionError(f"default joint positions are not the zero stance: {off}")
    root_z = float(_to_numpy(robot.data.default_root_state)[0][2])
    want = STAND_HEIGHT_M + SPAWN_CLEARANCE_M
    if abs(root_z - want) > 1e-6:
        raise AssertionError(f"default root z is {root_z}, cfg says {want}")


def check_masses_match_the_mjcf(robot) -> None:
    """Per-body and total masses are the compiled MuJoCo model's.

    The pinned numbers' provenance is on `MJCF_BODY_MASS_KG`. Density-derived
    masses crossing an importer are exactly the kind of number that drifts.
    """
    # `body_mass`, not `default_mass`: the latter is deprecation-warned in this
    # install ("will be deprecated in IsaacLab 4.0. Please use body_mass").
    masses = _to_numpy(robot.data.body_mass)[0]
    total = float(masses.sum())
    if abs(total - MJCF_TOTAL_MASS_KG) > MASS_TOL_KG * 13:
        raise AssertionError(
            f"total mass {total:.7f} kg, MuJoCo compiles {MJCF_TOTAL_MASS_KG:.7f}"
        )
    for index, name in enumerate(robot.body_names):
        part = name.split("_")[0]
        want = MJCF_BODY_MASS_KG[part]
        got = float(masses[index])
        if abs(got - want) > MASS_TOL_KG:
            raise AssertionError(f"{name} mass is {got:.7f} kg, MuJoCo compiles {want:.7f}")


def check_gains_ceilings_and_armature(robot) -> None:
    """kp 30, kd 5.586, effort 40 N·m, armature 1.0 — on all twelve.

    kp is the authored spring constant doing its old job through the drive;
    kd is the half-critical derivation in spyder_cfg (its docstring carries
    the saturation arithmetic that ruled out critical damping).
    """
    from bestiary.isaac import spyder_cfg

    stiffness = _to_numpy(robot.data.joint_stiffness)[0]
    damping = _to_numpy(robot.data.joint_damping)[0]
    effort = _to_numpy(robot.data.joint_effort_limits)[0]
    armature = _to_numpy(robot.data.joint_armature)[0]
    for index, name in enumerate(robot.joint_names):
        for label, arr, want in (
            ("kp", stiffness, spyder_cfg.KP),
            ("kd", damping, spyder_cfg.KD),
            ("effort limit", effort, spyder_cfg.EFFORT_LIMIT_NM),
            ("armature", armature, spyder_cfg.ARMATURE),
        ):
            got = float(arr[index])
            if abs(got - want) > GAIN_TOL:
                raise AssertionError(f"{name} {label} is {got}, spyder_cfg says {want}")


def check_no_velocity_limit_arrived(robot) -> None:
    """No joint velocity limit — absent by decision, so its arrival is a bug.

    There is no vendor sheet for a machine that was never hardware, and
    `velocity_limit_sim` BRAKES a joint past the limit — an invented number
    would be an undeclared brake. What bounds leg speed is 40 N·m against
    1.04 kg·m². If a sourced number ever exists, set it in spyder_cfg and
    update this check to assert it instead.
    """
    limits = _to_numpy(robot.data.joint_vel_limits)[0]
    clamped = {
        robot.joint_names[i]: float(v) for i, v in enumerate(limits) if float(v) < 1e3
    }
    if clamped:
        raise AssertionError(
            f"velocity limits arrived from nowhere: {clamped}. Nothing in this "
            "repository sources one for Spyder."
        )


# ---------------------------------------------------------------------------
# 3. The training config's incentives.
# ---------------------------------------------------------------------------
def _spyder_env_cfg():
    from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg

    return SpyderGentleEnvCfg()


def _dead_zone_mean_kernel(dz: float, hi: float, std: float) -> float:
    """E[exp(-(c/std)^2)] for |c| ~ U(dz, hi) — what standing earns on a
    dead-zoned channel. Closed form via erf; symmetric in the sign."""
    if hi <= dz:
        raise ValueError(f"empty dead-zone range [{dz}, {hi}]")
    return (std * math.sqrt(math.pi) / (2.0 * (hi - dz))) * (
        math.erf(hi / std) - math.erf(dz / std)
    )


def check_commands_are_dead_zoned_with_heading_off(cfg) -> None:
    """The command term is the dead-zone sampler, configured as designed."""
    from bestiary.isaac.commands import DeadZoneVelocityCommandCfg
    from bestiary.isaac.spyder_gentle_env_cfg import (
        REL_STANDING,
        VX_MAX_MS,
        VX_MIN_MS,
        WZ_MAX_RADS,
        WZ_MIN_RADS,
    )

    cmd = cfg.commands.base_velocity
    if not isinstance(cmd, DeadZoneVelocityCommandCfg):
        raise AssertionError(f"command cfg is {type(cmd).__name__}, not DeadZoneVelocityCommandCfg")
    # The class_type must be the LAZY STRING, never the class object: hydra
    # imports env-cfg modules before the app exists, and an eager class_type
    # drags VisualizationMarkers -> pip pxr into the process pre-Kit — a
    # measured free(): invalid pointer at boot, on two machines (2026-08-06).
    if not (isinstance(cmd.class_type, str)
            and str(cmd.class_type).endswith("commands_impl:DeadZoneVelocityCommand")):
        raise AssertionError(
            f"class_type is {cmd.class_type!r} — it must be the lazy string "
            "'bestiary.isaac.commands_impl:DeadZoneVelocityCommand'. An eager "
            "class object here imports the runtime chain (and pip pxr) before "
            "Kit boots, which heap-corrupts the app 1.5 s into every launch."
        )
    facts = {
        "heading_command": (cmd.heading_command, False),
        "rel_standing_envs": (cmd.rel_standing_envs, REL_STANDING),
        "lin_vel_x": (cmd.ranges.lin_vel_x, (-VX_MAX_MS, VX_MAX_MS)),
        "lin_vel_y": (cmd.ranges.lin_vel_y, (0.0, 0.0)),
        "ang_vel_z": (cmd.ranges.ang_vel_z, (-WZ_MAX_RADS, WZ_MAX_RADS)),
        "min_lin_vel_x": (cmd.min_lin_vel_x, VX_MIN_MS),
        "min_ang_vel_z": (cmd.min_ang_vel_z, WZ_MIN_RADS),
    }
    wrong = {k: got for k, (got, want) in facts.items() if got != want}
    if wrong:
        raise AssertionError(f"command config drifted from the design: {wrong}")


def check_kernel_widths_preserve_the_ratio(cfg) -> None:
    """std/range = 0.5 on both channels — upstream's own discrimination ratio.

    Inheriting std = 0.5 against a ±0.6 range would score standing at the top
    command at exp(-(0.6/0.5)^2) = 0.24 of perfect; upstream's operating point
    is 0.018. The ratio, not the constant, is the recipe.
    """
    lin_std = float(cfg.rewards.track_lin_vel_xy_exp.params["std"])
    ang_std = float(cfg.rewards.track_ang_vel_z_exp.params["std"])
    vx_hi = float(cfg.commands.base_velocity.ranges.lin_vel_x[1])
    wz_hi = float(cfg.commands.base_velocity.ranges.ang_vel_z[1])
    for label, std, hi in (("lin", lin_std, vx_hi), ("ang", ang_std, wz_hi)):
        ratio = std / hi
        if abs(ratio - 0.5) > 1e-9:
            raise AssertionError(
                f"{label} kernel std/range = {std}/{hi} = {ratio:.4f}, not the "
                "0.5 upstream discriminates at. If the range moved, move the std "
                "with it."
            )


def check_retargets_resolve_on_this_robot(cfg, robot) -> None:
    """Every body-name regex the config carries resolves on Spyder's links.

    The loud failures raise at env construction anyway; this is for the QUIET
    one — a regex that matches nothing in a term whose manager tolerates an
    empty list, which is a reward silently not being paid.
    """
    import re as _re

    body_names = list(robot.body_names)
    expectations = {
        "feet_air_time sensor": (cfg.rewards.feet_air_time.params["sensor_cfg"].body_names, 4),
        "undesired_contacts sensor": (
            cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names, 4),
        "base_contact termination": (
            cfg.terminations.base_contact.params["sensor_cfg"].body_names, 1),
    }
    for label, (patterns, want) in expectations.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        matched = {n for p in patterns for n in body_names if _re.fullmatch(p, n)}
        if len(matched) != want:
            raise AssertionError(
                f"{label}: {patterns!r} resolves {sorted(matched)} "
                f"({len(matched)} bodies), expected {want} on {body_names}"
            )


def check_height_scan_covers_the_feet(cfg) -> None:
    """Same 187 rays as upstream, on a footprint that reaches Spyder's feet.

    Two half-checks, both load-bearing: the ray COUNT is part of the
    observation width (a one-way door — change it and every checkpoint
    orphans), and the FOOTPRINT must contain the foot centres at (±0.76,
    ±0.76) in the torso frame — upstream's ANYmal-sized 1.6 x 1.0 m grid
    leaves all four Spyder feet on unseen ground, found in adversarial
    review before it could cost a run.
    """
    pat = cfg.scene.height_scanner.pattern_cfg
    nx = int(round(pat.size[0] / pat.resolution)) + 1
    ny = int(round(pat.size[1] / pat.resolution)) + 1
    if nx * ny != 187:
        raise AssertionError(
            f"height_scan pattern is {nx}x{ny} = {nx * ny} rays, not upstream's "
            "187 — the observation width moved, which orphans every checkpoint."
        )
    foot_xy = 0.76  # torso-frame |x| and |y| of each foot centre, from the MJCF
    half_x, half_y = pat.size[0] / 2.0, pat.size[1] / 2.0
    if half_x < foot_xy or half_y < foot_xy:
        raise AssertionError(
            f"height_scan footprint half-spans are ({half_x:.2f}, {half_y:.2f}) m "
            f"but the foot centres sit at ±{foot_xy} m — the policy would place "
            "feet on ground it cannot see."
        )


def check_curriculum_uses_the_arc_bar(cfg) -> None:
    """The terrain curriculum is the arc-corrected term, and its kinematics hold.

    Upstream's demote bar applies straight-line kinematics to arc commands, so
    a PERFECT tracker of (0.6 m/s, 0.8 rad/s) — 1.48 m reachable against a 6 m
    bar — is demoted every episode while a yaw-blind straight driver promotes:
    a curriculum that teaches learning 015's failure on purpose. The wiring and
    three spot values of the pure kinematics are asserted; the derivation is
    `curriculums.py`'s docstring.
    """
    import math as _math

    from bestiary.isaac.curriculums import arc_displacement_m, terrain_levels_vel_arc

    if cfg.curriculum.terrain_levels.func is not terrain_levels_vel_arc:
        raise AssertionError(
            f"curriculum.terrain_levels.func is "
            f"{getattr(cfg.curriculum.terrain_levels.func, '__name__', '?')!r}, "
            "not terrain_levels_vel_arc — upstream's bar demotes perfect turners."
        )
    # w = 0 reduces exactly to upstream's straight bar; the two arc points are
    # 2(v/w)|sin(wT/2)| by hand.
    cases = [
        ((0.6, 0.0), 12.0),
        ((0.6, 0.8), 2.0 * (0.6 / 0.8) * abs(_math.sin(0.8 * 10.0))),
        ((0.25, 0.2), 2.0 * (0.25 / 0.2) * abs(_math.sin(0.2 * 10.0))),
    ]
    for (v, w), want in cases:
        got = float(arc_displacement_m(v, w, 20.0))
        if abs(got - want) > 1e-9:
            raise AssertionError(
                f"arc_displacement_m({v}, {w}, 20) = {got}, expected {want} — "
                "the kinematics moved."
            )


def check_reset_scatter_is_not_degenerate(cfg) -> None:
    """Joint resets use the OFFSET form, because the stance is all zeros.

    Upstream's `reset_joints_by_scale` multiplies the default pose — on a
    zero-stance machine that is 0 x U(0.5, 1.5) = 0, i.e. no reset diversity
    at all, silently. This is the assertion that keeps a well-meaning
    "restore upstream's event" edit from re-introducing the degeneracy.
    """
    import isaaclab.envs.mdp as mdp

    ev = cfg.events.reset_robot_joints
    if ev.func is not mdp.reset_joints_by_offset:
        raise AssertionError(
            f"reset_robot_joints.func is {getattr(ev.func, '__name__', ev.func)!r}; "
            "on a zero-default-pose robot the scale form scatters nothing."
        )
    lo, hi = ev.params["position_range"]
    if not (lo < 0.0 < hi):
        raise AssertionError(
            f"reset position_range {ev.params['position_range']} does not "
            "straddle zero — offsets from a zero stance must go both ways."
        )


def check_terrain_is_the_gentle_mix(cfg) -> None:
    """The generator carries the gentle asset at its declared span, 0.1 m grid."""
    from bestiary import paths
    from bestiary.terrain.gentle import Z_SPAN_M

    gen = cfg.scene.terrain.terrain_generator
    if abs(gen.horizontal_scale - 0.1) > 1e-12:
        raise AssertionError(f"training horizontal_scale is {gen.horizontal_scale}, not 0.1")
    sub = gen.sub_terrains.get("bestiary_gentle")
    if sub is None:
        raise AssertionError(f"no bestiary_gentle sub-terrain; got {sorted(gen.sub_terrains)}")
    if sub.hfield_path != str(paths.GENTLE_HFIELD):
        raise AssertionError(f"gentle tile reads {sub.hfield_path}, not {paths.GENTLE_HFIELD}")
    if abs(sub.z_span_m - Z_SPAN_M) > 1e-12:
        raise AssertionError(f"gentle tile z_span_m is {sub.z_span_m}, gentle.py says {Z_SPAN_M}")
    if not gen.curriculum:
        raise AssertionError("curriculum is off; terrain_levels_vel would be inert")


def check_forward_variant_is_reward_only(cfg) -> None:
    """The v_x diagnostic carries ONE reward term and moves nothing else.

    `Bestiary-Forward-Spyder-v0` exists to tell "the reward table is wrong"
    apart from "the port is wrong" (`spyder_forward_env_cfg.py` carries the
    argument). It can only do that if it is a ONE-VARIABLE experiment, and
    "one variable" is a claim about a whole config, not about the lines
    someone remembered to write — so it is asserted the strongest available
    way: dump both configs and require the difference to be the `rewards` key
    and nothing else.

    That form catches what a hand-written list of assertions cannot: a
    termination quietly dropped, a command range nudged, an observation term
    added, an event disabled. Two constructions of the SAME config dict-compare
    equal (measured 2026-08-06), so the comparison has no false positives to
    tolerate — any key that moves is a real second variable.

    No simulator is needed: both configs construct pre-app, which is the same
    property `commands.py` depends on.
    """
    from bestiary.isaac import rewards as bestiary_rewards
    from bestiary.isaac.spyder_forward_env_cfg import (
        REWARD_TERM_NAME,
        REWARD_WEIGHT,
        SpyderForwardEnvCfg,
        single_reward_term,
    )

    fwd = SpyderForwardEnvCfg()
    # Raises unless exactly one non-None term survives, naming what it found.
    name, term = single_reward_term(fwd.rewards)
    if name != REWARD_TERM_NAME:
        raise AssertionError(f"the one reward term is {name!r}, not {REWARD_TERM_NAME!r}")
    if term.func is not bestiary_rewards.forward_velocity:
        raise AssertionError(
            f"{name}.func is {getattr(term.func, '__name__', term.func)!r}, not "
            "bestiary.isaac.rewards.forward_velocity — the diagnostic is measuring "
            "something other than base-frame forward speed."
        )
    if term.weight != REWARD_WEIGHT:
        raise AssertionError(f"{name}.weight is {term.weight}, not {REWARD_WEIGHT}")

    gentle_d, fwd_d = cfg.to_dict(), fwd.to_dict()
    moved = sorted(k for k in set(gentle_d) | set(fwd_d) if gentle_d.get(k) != fwd_d.get(k))
    if moved != ["rewards"]:
        raise AssertionError(
            f"the forward diagnostic differs from the gentle config in {moved}, "
            "but it may differ ONLY in 'rewards'. Anything else here makes the "
            "run a multi-variable experiment: whatever it shows about the stack "
            "would be uninterpretable. (Terminations, commands, observations, "
            "actions, terrain and curriculum are all inherited on purpose — the "
            "episode-reset machinery is not a reward.)"
        )


def check_the_money(cfg) -> None:
    """Standing's share and the penalty budget, computed from the live config.

    Two assertions and a printed table. The assertions:

      * A motionless machine's expected share of drive-cell tracking income
        stays under STANDING_SHARE_FLAG. This is the number that was 63% when
        the Hound's seed parked, and it moves whenever someone edits a range,
        a dead zone or a kernel width — which is exactly when it should bite.
      * Recurring penalties at labelled operating points stay under
        PENALTY_BUDGET_FLAG_FRACTION of income (decisions/0005's 30% rule).

    Contact prices are PRINTED, not asserted: their cost is price x frequency
    and frequency is unknown before training. The femur line exists because
    the Hound's thigh line was 106% and nobody had printed it.
    """
    cmd = cfg.commands.base_velocity
    dt = cfg.decimation * cfg.sim.dt
    lin = cfg.rewards.track_lin_vel_xy_exp
    ang = cfg.rewards.track_ang_vel_z_exp
    lin_std = float(lin.params["std"])
    ang_std = float(ang.params["std"])
    vx_dz, vx_hi = cmd.min_lin_vel_x, float(cmd.ranges.lin_vel_x[1])
    wz_dz, wz_hi = cmd.min_ang_vel_z, float(cmd.ranges.ang_vel_z[1])

    # Income: every commanded velocity is achievable in principle (v_y is
    # commanded 0), so the perfect-tracking ceiling is 1.0 on both kernels.
    income = (lin.weight * 1.0 + ang.weight * 1.0) * dt

    # Standing's expected drive-cell share. The yaw channel is SNAP-shaped:
    # a fraction dz/hi of drive envs carry w_z == 0 exactly (straight
    # drivers), where a motionless machine scores the FULL yaw kernel; the
    # survivors are uniform on ±[dz, hi]. The linear channel is resampled,
    # so its dead zone is a pure magnitude window.
    stand_lin = _dead_zone_mean_kernel(vx_dz, vx_hi, lin_std)
    p_straight = wz_dz / wz_hi
    stand_ang = p_straight * 1.0 + (1.0 - p_straight) * _dead_zone_mean_kernel(
        wz_dz, wz_hi, ang_std
    )
    stand_share = (lin.weight * stand_lin + ang.weight * stand_ang) * dt / income

    # feet_air_time is CADENCE-SHAPED, not income: the formula pays
    # (air_time - threshold) per touchdown while a command is active, so at a
    # 50% duty gait of f Hz per foot it is 4·f·(0.5/f - threshold) per second
    # — zero at 1 Hz, NEGATIVE above it. The first draft modelled it as
    # "+weight·0.5·dt income" and an adversarial review computed that as
    # 12-25x wrong with the wrong sign for any brisk gait. Printed at two
    # cadences; the 2 Hz value is counted WITH the penalties, conservatively.
    fat_w = cfg.rewards.feet_air_time.weight
    fat_at = lambda f: fat_w * 4.0 * f * (0.5 / f - 0.5) * dt  # noqa: E731

    # Recurring penalties at labelled operating points. The two joint figures
    # are ASSUMED and generous: 40 N·m into 1.04 kg·m² allows 38 rad/s², and a
    # sedate gait holds a fraction of that.
    tau_rms, acc_rms, dact_rms, vz_rms, wxy_rms = 1.0, 10.0, 0.3, 0.1, 0.5
    n = 12
    rows = [
        ("track_lin_vel_xy_exp (income)", lin.weight * 1.0 * dt, "ceiling: all commands achievable"),
        ("track_ang_vel_z_exp  (income)", ang.weight * 1.0 * dt, "ceiling"),
        ("feet_air_time @ 1 Hz/foot", fat_at(1.0), "cadence-shaped: zero at 1 Hz, 50% duty"),
        ("feet_air_time @ 2 Hz/foot", fat_at(2.0),
         "[ASSUMED] 2 Hz, 50% duty — a tax on brisk stepping, counted below"),
        ("dof_torques_l2", -abs(cfg.rewards.dof_torques_l2.weight) * n * tau_rms**2 * dt,
         f"[ASSUMED] {tau_rms} N·m rms"),
        ("dof_acc_l2", -abs(cfg.rewards.dof_acc_l2.weight) * n * acc_rms**2 * dt,
         f"[ASSUMED] {acc_rms} rad/s² rms"),
        ("action_rate_l2", -abs(cfg.rewards.action_rate_l2.weight) * n * dact_rms**2 * dt,
         f"[ASSUMED] {dact_rms} action-units/step rms"),
        ("lin_vel_z_l2", -abs(cfg.rewards.lin_vel_z_l2.weight) * vz_rms**2 * dt,
         f"[ASSUMED] {vz_rms} m/s rms"),
        ("ang_vel_xy_l2", -abs(cfg.rewards.ang_vel_xy_l2.weight) * 2 * wxy_rms**2 * dt,
         f"[ASSUMED] {wxy_rms} rad/s rms per axis"),
    ]
    print("      money, per policy step (dt = {:.3f} s):".format(dt), flush=True)
    for label, value, note in rows:
        print(f"        {label:<32} {value:+.6f}   {note}", flush=True)
    penalties = -sum(v for _, v, _ in rows if v < 0)
    print(f"        {'recurring penalties':<32} {penalties:.6f}   "
          f"{penalties / income:.1%} of income "
          f"(flag {PENALTY_BUDGET_FLAG_FRACTION:.0%})", flush=True)
    print(f"        {'standing share, drive cells':<32} {stand_share:.1%}   "
          f"(Hound at the parked seed: 63%)", flush=True)
    print(f"        {'one femur in contact':<32} {-1.0 * abs(cfg.rewards.undesired_contacts.weight) * dt:+.6f}   "
          f"{abs(cfg.rewards.undesired_contacts.weight) * dt / income:.1%} of income "
          "IF sustained — price x frequency, frequency unmeasured", flush=True)

    if stand_share > STANDING_SHARE_FLAG:
        raise AssertionError(
            f"a motionless machine collects {stand_share:.1%} of drive-cell "
            f"tracking income in expectation (flag: {STANDING_SHARE_FLAG:.0%}). "
            "This is the parked-seed door re-opening — check the dead zones "
            "and the kernel stds together."
        )
    if penalties > PENALTY_BUDGET_FLAG_FRACTION * income:
        raise AssertionError(
            f"recurring penalties at the stated operating points cost "
            f"{penalties / income:.1%} of income (flag: "
            f"{PENALTY_BUDGET_FLAG_FRACTION:.0%}) — learnings/011's failure "
            "shape. Re-derive before training."
        )


# ---------------------------------------------------------------------------
# Harness.
# ---------------------------------------------------------------------------
FILE_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("conversion-input-current", check_conversion_input_is_current),
    ("usd-has-no-ground", check_usd_has_no_ground),
    ("usd-selects-physx-variant", check_usd_selects_the_physx_variant),
    ("no-spring-no-stray-drive", check_no_spring_and_no_stray_drive_reaches_the_solver),
    ("all-bodies-report-contacts", check_all_bodies_report_contacts),
    ("only-load-bearing-geoms-collide", check_only_the_load_bearing_geoms_collide),
    ("static-friction-from-cfg", check_static_friction_is_supplied_by_the_cfg),
    ("effort-and-armature-vs-mjcf", check_effort_ceiling_and_armature_agree_with_the_mjcf),
    ("gentle-asset-parses", check_gentle_asset_parses),
]

SIM_CHECKS: list[tuple[str, Callable]] = [
    ("body-count", check_body_count),
    ("twelve-actuated-dof", check_twelve_actuated_dof),
    ("joint-limits-vs-mjcf", check_joint_limits_match_the_mjcf),
    ("default-pose-is-stance", check_default_pose_is_the_stance),
    ("masses-vs-mjcf", check_masses_match_the_mjcf),
    ("gains-ceilings-armature", check_gains_ceilings_and_armature),
    ("no-velocity-limit-arrived", check_no_velocity_limit_arrived),
]

CFG_CHECKS: list[tuple[str, Callable]] = [
    ("commands-dead-zoned-heading-off", check_commands_are_dead_zoned_with_heading_off),
    ("kernel-widths-preserve-ratio", check_kernel_widths_preserve_the_ratio),
    ("height-scan-covers-feet", check_height_scan_covers_the_feet),
    ("curriculum-arc-bar", check_curriculum_uses_the_arc_bar),
    ("reset-scatter-not-degenerate", check_reset_scatter_is_not_degenerate),
    ("terrain-is-gentle-mix", check_terrain_is_the_gentle_mix),
    ("the-money", check_the_money),
    ("forward-variant-is-reward-only", check_forward_variant_is_reward_only),
]


def _run(name: str, fn: Callable[[], None]) -> int:
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
    from bestiary.isaac.spyder_cfg import SPYDER12_CFG

    print(f"[bestiary] MJCF : {paths.SPYDER_XML}", flush=True)
    print(f"[bestiary] input: {paths.SPYDER_ISAAC_MJCF}", flush=True)
    print(f"[bestiary] USD  : {paths.SPYDER_ISAAC_USD}", flush=True)
    print(flush=True)

    failures = sum(_run(name, fn) for name, fn in FILE_CHECKS)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    robot = Articulation(SPYDER12_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    failures += sum(_run(name, lambda fn=fn: fn(robot)) for name, fn in SIM_CHECKS)

    cfg = _spyder_env_cfg()
    failures += sum(
        _run(
            name,
            lambda fn=fn: fn(cfg, robot) if fn is check_retargets_resolve_on_this_robot else fn(cfg),
        )
        for name, fn in CFG_CHECKS + [("retargets-resolve", check_retargets_resolve_on_this_robot)]
    )

    total = len(FILE_CHECKS) + len(SIM_CHECKS) + len(CFG_CHECKS) + 1
    print(f"\n{total - failures}/{total} checks pass", flush=True)
    return 1 if failures else 0


def _exit(status: int) -> None:
    """Flush, then `os._exit`. `SimulationApp.close()` exits 0 no matter what
    (measured on the Hound oracle), so Kit teardown is skipped entirely."""
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
