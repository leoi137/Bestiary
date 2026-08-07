"""The Spyder-12 Isaac port oracle: every way this port could be quietly wrong.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.check_spyder

Run after ANY change to `assets/spyder12.xml`, `spyder_usd.py`, `spyder_cfg.py`,
`commands.py`, `spyder_gentle_env_cfg.py`, `rewards.py`,
`spyder_forward_env_cfg.py`, `spyder_ladder_env_cfg.py` or
`spyder_overnight_env_cfg.py`, and before every training launch.

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
     failures it exists to keep from repeating). Last in this group, the three
     REWARD-TABLE VARIANTS are pinned the same way: the forward-velocity
     diagnostic carries exactly one reward term and differs from the training
     config in nothing else; each rung of the reward LADDER carries the income
     terms plus at most one declared penalty and differs in nothing but
     `rewards` and one command range; and the OVERNIGHT task carries the
     ladder's winning rung plus its two declared gait-shaping terms, deleting
     exactly the six the record names. A one-variable experiment is only one
     variable if something checks, and a long run's reward table is only the
     declared one if something checks that too.

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


def _dict_diff_paths(a, b, prefix: str = "") -> list[str]:
    """Dotted paths at which two `to_dict()` trees disagree, deepest-wins.

    `check_forward_variant_is_reward_only` compares top-level keys, which is
    the right granularity when a variant may move exactly one section. The
    ladder may move two — `rewards` and `commands` — and the second one is only
    allowed to move in a single leaf, so the diff has to descend. A key present
    in one tree and absent from the other reports as that key's own path
    rather than recursing into it; both are real differences, and naming the
    shallower one is what makes the failure message readable.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        paths: list[str] = []
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}{key}"
            if key not in a or key not in b:
                paths.append(path)
            else:
                paths.extend(_dict_diff_paths(a[key], b[key], f"{path}."))
        return paths
    return [] if a == b else [prefix.rstrip(".")]


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


#: The ONE command range a keep-list variant is allowed to move against the
#: gentle task. A dotted path into `to_dict()`, so the assertion below reads as
#: the sentence the ladder's docstring makes: "strafe, and nothing else". The
#: overnight task shares it, because it commands the ladder's envelope exactly.
LADDER_COMMAND_DIFF_PATH = "base_velocity.ranges.lin_vel_y"


def _assert_keep_list_variant(what: str, want_terms: list[str], cfg, base_cfg, arm: str) -> None:
    """One keep-list variant against one baseline: two sections, one leaf, one table.

    Shared by the ladder's three rungs and by the overnight task, because they
    are the same shape of claim: `spyder_ladder_env_cfg.apply_keep_list_and_strafe`
    reduced the gentle config to a declared list of reward terms and opened the
    lateral command range, and nothing else moved. `what` names the variant in
    every message ("ladder rung 'tilt'", "the overnight task") and `arm` says
    which of its two configs is being read ("train" or "play").

    Every variant is checked against BOTH the gentle training config and the
    gentle Play config. The Play twins are not decoration:
    `spyder_forward_env_cfg.py`'s Play class descends from the gentle Play class
    rather than from its own training class, and both the ladder and the
    overnight task copy that pattern, so a Play config gets its reward surgery
    from a SECOND call to the mutator — an edit that reaches one call and not
    the other produces a Play config that silently plays a different reward than
    the one that trained. Checking both is what makes that impossible rather
    than merely unlikely.
    """
    from bestiary.isaac.spyder_ladder_env_cfg import VY_MAX_MS

    base_d, var_d = base_cfg.to_dict(), cfg.to_dict()

    # (a) Which SECTIONS moved. Two, named, and no others.
    moved = sorted(k for k in set(base_d) | set(var_d) if base_d.get(k) != var_d.get(k))
    if moved != ["commands", "rewards"]:
        raise AssertionError(
            f"{what} ({arm}) differs from the gentle config in "
            f"{moved}, but it may differ ONLY in ['commands', 'rewards']. "
            "Anything else makes it a multi-variable change against the gentle "
            "task and against every sibling that shares this baseline: whatever "
            "the run then shows would be unattributable. "
            "(Terminations, observations, actions, events, terrain and "
            "curriculum are all inherited on purpose.)"
        )

    # (b) WHERE inside `commands` it moved. Exactly one leaf: the strafe range.
    cmd_paths = _dict_diff_paths(base_d["commands"], var_d["commands"])
    if cmd_paths != [LADDER_COMMAND_DIFF_PATH]:
        raise AssertionError(
            f"{what} ({arm}) moves these command fields: "
            f"{cmd_paths}. Opening the LATERAL range is the only command change "
            f"allowed — the only legal diff is [{LADDER_COMMAND_DIFF_PATH!r}]. "
            "The v_x and w_z ranges, both dead zones, the standing fraction and "
            "heading-off are the gentle task's, and moving one of them here "
            "would confound the reward change with a command-distribution one."
        )
    got_y = tuple(var_d["commands"]["base_velocity"]["ranges"]["lin_vel_y"])
    base_y = tuple(base_d["commands"]["base_velocity"]["ranges"]["lin_vel_y"])
    if base_y != (0.0, 0.0) or got_y != (-VY_MAX_MS, VY_MAX_MS):
        raise AssertionError(
            f"{what} ({arm}) has lin_vel_y {base_y} -> {got_y}; "
            f"expected (0.0, 0.0) -> {(-VY_MAX_MS, VY_MAX_MS)}. Either the "
            "gentle task stopped pinning strafe to zero (in which case this "
            "task no longer adds a channel) or its range is not the declared "
            "one."
        )

    # (c) The reward table: exactly the declared terms, at the gentle task's
    # weights and params. Comparing against the LIVE gentle config, never
    # against numbers typed here — a weight that drifts in one place should
    # fail, a weight that is deliberately changed in the gentle task should
    # follow through to every variant.
    live = {
        name: term
        for name, term in var_d["rewards"].items()
        if not name.startswith("_") and term is not None
    }
    if sorted(live) != want_terms:
        raise AssertionError(
            f"{what} ({arm}) pays {sorted(live)}, but its declared table is "
            f"{want_terms}. A keep-list variant is exactly the terms it "
            "declares: an extra one is a reward nobody wrote down, and a "
            "missing one silently makes it a different task than the record "
            "says trained."
        )
    for name, term in sorted(live.items()):
        base_term = base_d["rewards"].get(name)
        if base_term is None:
            raise AssertionError(
                f"{what} ({arm}) pays {name!r}, which the gentle "
                "config does not have live — it invented a reward term, so "
                "'at the gentle task's weight' is meaningless for it."
            )
        if term != base_term:
            raise AssertionError(
                f"{what} ({arm}) term {name!r} is {term}, the "
                f"gentle task's is {base_term}. Keep-list variants inherit "
                "weights, params and functions untouched: they choose WHICH "
                "terms are paid, never what a term should cost."
            )


def _assert_one_rung(rung: str, rung_cfg, base_cfg, label: str) -> None:
    """One ladder rung, named as a rung. The assertions are the shared ones."""
    from bestiary.isaac.spyder_ladder_env_cfg import rung_reward_terms

    _assert_keep_list_variant(
        f"ladder rung {rung!r}", sorted(rung_reward_terms(rung)), rung_cfg, base_cfg, label
    )


def check_ladder_rungs_are_income_plus_one(cfg) -> None:
    """Each ladder rung: full income + at most one penalty, plus strafe. Only.

    The ladder (`spyder_ladder_env_cfg.py`) asks which single penalty tames the
    gait that `research/episodes/014` measured — 4.2-5.4 m/s, bounding,
    airborne, command-deaf — while keeping a policy that can still be driven.
    Three rungs, one penalty apart, are only comparable if the rest of the
    config is identical, so "identical" is asserted the strongest available
    way: dump every config and require the difference to be two named sections,
    one named command leaf, and a declared reward table.

    Three assertions per rung per baseline, each catching what the others
    cannot:

      (a) SECTIONS: only `rewards` and `commands` move. Catches a termination
          quietly dropped, an observation term added, an event disabled, a
          terrain parameter nudged.
      (b) COMMAND LEAF: inside `commands`, only `lin_vel_y` moves, from
          (0, 0) to the declared ±0.4. Catches the failure (a) is blind to —
          a rung that also widens v_x, softens a dead zone, or re-enables
          heading mode would still show `commands` as one moved section.
      (c) REWARD TABLE: exactly the declared term names, each byte-identical
          to the gentle task's own term. Catches a weight typed by hand, a
          param dropped with a retarget, and an upstream term that survived
          the keep-list.

    Both the training config AND the Play twin of every rung are checked, for
    the reason `_assert_one_rung` gives.

    No simulator is needed: every config here constructs pre-app, the same
    property `commands.py` depends on.
    """
    from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg_PLAY
    from bestiary.isaac.spyder_ladder_env_cfg import (
        LADDER_CFGS,
        RUNGS,
        VY_MAX_MS,
        rung_reward_terms,
    )

    play_base = SpyderGentleEnvCfg_PLAY()
    print(f"      the ladder, {len(RUNGS)} rungs (income + at most one penalty):", flush=True)
    # Declaration order, not sorted: `RUNGS` is written bare-first because that
    # is the order the rungs are meant to be READ in — the control, then the
    # two terms measured against it.
    for rung in RUNGS:
        train_cls, play_cls = LADDER_CFGS[rung]
        train_cfg = train_cls()
        _assert_one_rung(rung, train_cfg, cfg, "train")
        _assert_one_rung(rung, play_cls(), play_base, "play")

        terms = {
            name: term.weight
            for name, term in vars(train_cfg.rewards).items()
            if not name.startswith("_") and term is not None
        }
        ranges = train_cfg.commands.base_velocity.ranges
        table = "  ".join(f"{n} {w:+g}" for n, w in sorted(terms.items()))
        print(f"        {rung:<11} {table}", flush=True)
        print(
            f"        {'':<11} v_x {tuple(ranges.lin_vel_x)}  v_y "
            f"{tuple(ranges.lin_vel_y)}  w_z {tuple(ranges.ang_vel_z)}",
            flush=True,
        )

    # Printed, not asserted, and it is the honest cost of adding strafe without
    # adding a strafe dead zone: `DeadZoneVelocityCommand` remaps only v_x and
    # w_z, so v_y ~ U(-0.4, 0.4) including the near-zero band the other two
    # channels exclude. A machine that tracks v_x perfectly and never sidesteps
    # therefore still collects this share of the LINEAR kernel in expectation.
    # Standing is unaffected — the v_x dead zone still guarantees a >= 0.25 m/s
    # error for a motionless machine — so the parked-seed door stays shut; what
    # this number prices is how optional strafe is, which is a training
    # outcome to read, not a config error to fail on.
    lin_std = float(cfg.rewards.track_lin_vel_xy_exp.params["std"])
    never_strafes = _dead_zone_mean_kernel(0.0, VY_MAX_MS, lin_std)
    print(
        f"        strafe is optional: a perfect v_x tracker that never sidesteps "
        f"still earns {never_strafes:.1%} of the linear kernel "
        f"(std {lin_std}, |v_y| ~ U(0, {VY_MAX_MS}))",
        flush=True,
    )

    # Non-vacuousness. Every assertion above is a comparison, and a comparison
    # over an empty ladder passes silently — learnings/014's shape exactly.
    if sorted(LADDER_CFGS) != sorted(RUNGS) or len(RUNGS) < 2:
        raise AssertionError(
            f"the ladder is {sorted(RUNGS)} with configs {sorted(LADDER_CFGS)} — "
            "a ladder needs at least two rungs and a config per rung, or this "
            "check compares nothing and reports green."
        )
    if len({tuple(rung_reward_terms(r)) for r in RUNGS}) != len(RUNGS):
        raise AssertionError(
            f"two ladder rungs declare the same reward table: "
            f"{ {r: rung_reward_terms(r) for r in sorted(RUNGS)} }. Two identical "
            "arms measure a seed, not a term."
        )


def check_overnight_task_is_the_declared_table(cfg) -> None:
    """The long run pays exactly five declared terms, and deletes exactly six.

    `Bestiary-Overnight-Spyder-v0` (`spyder_overnight_env_cfg.py`) is the run
    that spends the ladder's answer: the winning rung's table plus the two
    terms that price the shape of a step, at 10x the ladder's iteration count.
    It is a production run, not an experiment, which RAISES the stakes on this
    check rather than lowering them — a ladder arm that trains under the wrong
    reward costs 46 minutes and is caught by the arm beside it, and this one has
    no arm beside it and runs ten times as long.

    Four assertions, each catching what the others cannot:

      (a-c) The three shared keep-list assertions (`_assert_keep_list_variant`,
            which the ladder rungs use too): only `rewards` and `commands`
            move; inside `commands` only `lin_vel_y` moves, from (0, 0) to the
            ladder's declared range; and the reward table is exactly the
            declared names, each term byte-identical to the gentle task's own.
            Both the training config and the Play twin.
      (d)   PROVENANCE: the table contains every term of the winning ladder
            rung. This is what makes the docstring's "the measured winner plus
            two" a checked statement — a table that quietly stopped containing
            `action_rate_l2` would still satisfy (a-c) while the module's whole
            argument for its existence had evaporated.
      (e)   THE DELETIONS ARE THE DECLARED ONES. The surgery is a keep list, so
            it deletes whatever is live and not kept — safe by construction, and
            SILENT if Isaac Lab ships a twelfth reward term. This assertion is
            the loud half: the live gentle table minus the declared table must
            be exactly `EXPECTED_DELETED_TERMS`, so an upstream release that
            adds or renames a term turns this check red and someone writes down
            what changed instead of discovering it in a run.

    No simulator is needed: every config here constructs pre-app, the same
    property `commands.py` depends on.
    """
    from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg_PLAY
    from bestiary.isaac.spyder_ladder_env_cfg import live_reward_names, rung_reward_terms
    from bestiary.isaac.spyder_overnight_env_cfg import (
        EXPECTED_DELETED_TERMS,
        OVERNIGHT_TERMS,
        WINNING_RUNG,
        SpyderOvernightEnvCfg,
        SpyderOvernightEnvCfg_PLAY,
    )

    want_terms = sorted(OVERNIGHT_TERMS)
    train_cfg = SpyderOvernightEnvCfg()
    _assert_keep_list_variant("the overnight task", want_terms, train_cfg, cfg, "train")
    _assert_keep_list_variant(
        "the overnight task", want_terms, SpyderOvernightEnvCfg_PLAY(), SpyderGentleEnvCfg_PLAY(),
        "play",
    )

    # (d) Provenance: the ladder's winner is still inside the table.
    winner = sorted(rung_reward_terms(WINNING_RUNG))
    if not set(winner) <= set(OVERNIGHT_TERMS):
        raise AssertionError(
            f"the overnight table {want_terms} does not contain the winning "
            f"ladder rung {WINNING_RUNG!r} = {winner}. The task is declared as "
            "that rung plus the two gait-shaping terms; if it is not, the "
            "ladder's measurement is not its provenance and the module "
            "docstring is a story about a different config."
        )

    # (e) The deletions are the declared six, computed from the LIVE gentle
    # table rather than trusted from the tuple.
    gentle_live = live_reward_names(cfg.rewards)
    deleted = sorted(gentle_live - set(OVERNIGHT_TERMS))
    if not deleted:
        raise AssertionError(
            f"the overnight table {want_terms} deletes NOTHING from the gentle "
            f"table {sorted(gentle_live)} — it is the gentle task relabelled, "
            "and every assertion above compares a config against itself."
        )
    if deleted != sorted(EXPECTED_DELETED_TERMS):
        raise AssertionError(
            f"the overnight task deletes {deleted}, but "
            f"`EXPECTED_DELETED_TERMS` says {sorted(EXPECTED_DELETED_TERMS)}. "
            "The keep-list surgery has already deleted the difference silently "
            "and correctly — this is the notification, not the failure. Either "
            "Isaac Lab's RewardsCfg gained/renamed a term, or the gentle task "
            "did: write down which, then update the tuple and the module "
            "docstring's enumeration together."
        )

    missing_from_gentle = sorted(set(OVERNIGHT_TERMS) - gentle_live)
    if missing_from_gentle:
        raise AssertionError(
            f"the overnight table declares {missing_from_gentle}, which the "
            "gentle config does not pay — 'at the gentle task's weights' cannot "
            "be true of a term the gentle task does not have."
        )

    terms = {
        name: term.weight
        for name, term in vars(train_cfg.rewards).items()
        if not name.startswith("_") and term is not None
    }
    ranges = train_cfg.commands.base_velocity.ranges
    print(
        f"      the overnight task, {len(terms)} terms "
        f"(ladder rung {WINNING_RUNG!r} + {len(OVERNIGHT_TERMS) - len(winner)} "
        "gait-shaping terms):",
        flush=True,
    )
    for name, weight in sorted(terms.items()):
        origin = "winner" if name in winner else "added"
        print(f"        {name:<24} {weight:+g}   [{origin}]", flush=True)
    print(
        f"        {'commands':<24} v_x {tuple(ranges.lin_vel_x)}  "
        f"v_y {tuple(ranges.lin_vel_y)}  w_z {tuple(ranges.ang_vel_z)}",
        flush=True,
    )
    dropped = "  ".join(f"{n} {cfg.rewards.__dict__[n].weight:+g}" for n in deleted)
    print(f"        {'deleted (' + str(len(deleted)) + ')':<24} {dropped}", flush=True)


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
    ("ladder-rungs-are-income-plus-one", check_ladder_rungs_are_income_plus_one),
    ("overnight-task-is-declared-table", check_overnight_task_is_the_declared_table),
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
