"""Convert Spyder-12 from MJCF to USD for Isaac Lab.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.spyder_usd

Writes two files, both generated and both committed:

    assets/spyder/isaac/spyder12_isaac.xml                  the conversion input
    assets/spyder/isaac/spyder12_isaac/spyder12_isaac.usda  what Isaac Lab loads

Re-run after any change to `assets/spyder12.xml`. `isaac/check_spyder.py`
recomputes the transform below from the committed spyder12.xml and fails if the
committed conversion input no longer matches, so a stale USD is not a silent
condition.

WHY THE INPUT IS NOT `assets/spyder12.xml` ITSELF
-------------------------------------------------
Unlike Hound, Spyder has no `build.py` — `assets/spyder12.xml` is the authored
source of truth. So this module cannot re-emit the robot from a generator's
fragments; instead it PARSES the authored XML and applies a fixed, enumerated
set of deltas. The deltas are the same two the Hound port needed, plus one this
robot's authoring style adds:

1. NO WORLD. The floor plane, the directional light, the skybox and floor
   textures and the floor material all go. Isaac Lab clones the asset once per
   environment; an 80 x 80 m collision plane inside the robot prim would be
   cloned under every env origin, on top of the terrain the whole exercise
   exists to stand on. The `track` camera goes for the same reason — N robots
   would carry N cameras.

2. THE TORSO IS AUTHORED AT THE ORIGIN, not at z = 0.35. Isaac Lab ADDS an
   asset's authored transform to the spawn position it is given, so a stance
   height baked into the asset is an unrequested 0.35 m of drop.
   `isaac/spyder_cfg.py` supplies the stand height as `init_state.pos` instead.

3. THE PASSIVE JOINT DYNAMICS MOVE TO THE CFG. The authored default class puts
   `stiffness="30" damping="1"` on every joint — a labelled MuJoCo reset
   crutch: the spring returns each joint to the authored stance so the machine
   is self-supporting at reset (spyder12.xml says so at the default block).
   Isaac Lab replaces that crutch with an implicit PD drive
   (`spyder_cfg.KP` / `spyder_cfg.KD`), so the conversion input zeroes both —
   leaving them would risk spring and drive stacking into a stiffer leg than
   either model has, and unlike the Hound case the coincidence KP == authored
   stiffness == 30 would make the doubling invisible to a gain assertion.
   The `<motor>` actuators are dropped for the same reason: the MJCF importer
   cannot convert a `<motor>` to a PhysX drive anyway (it warns and emits
   nothing — measured on the Hound's wheels), and the effort ceiling they
   carried (gear 40 x ctrlrange ±1 = 40 N·m) moves to
   `spyder_cfg.EFFORT_LIMIT_NM`, where `check_spyder.py` re-reads the MJCF and
   asserts the two still agree.

   `armature="1"` SURVIVES, deliberately: rotor inertia is real dynamics, not
   a crutch, and it dominates this machine's joint-space inertia (measured
   M_ii = 1.010..1.037 kg·m², link contribution 0.010..0.037). It arrives as
   `physxJoint:armature`, the same path the Hound's did.

4. MESH PATHS GAIN A `meshdir`. The visual shell's OBJs live in
   `assets/meshes/`, referenced as `meshes/...` relative to the authored XML in
   `assets/`. The conversion input lives two levels deeper, so the compiler
   block gets `meshdir="../.."` — which COMPOSES with the authored
   `file="meshes/..."` refs to land on `assets/meshes/`. The shell is KEPT — `class="visual"`
   geoms are contype=0 conaffinity=0 density=0, so they carry no physics, and
   the operator watches these robots; a recognisable machine in the viewer is
   worth one attribute. If the importer ever chokes on the OBJs, `--no-shell`
   strips them and the physics is bit-identical.

Everything else — geometry, masses, joint ranges, friction — passes through
untouched, and `check_spyder.py` holds the transform to exactly these deltas by
recomputing it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET

from bestiary import paths

# `isaaclab` imports live inside the functions that need Kit, NOT at module
# scope — a deliberate difference from hound_usd.py. `conversion_mjcf` below is
# pure ElementTree, and keeping the module importable by the plain venv is what
# lets `check.py`-style tooling and a human validate the transform against
# MuJoCo locally, without a GPU session. (`check_spyder.py` relies on this.)

#: Rigid bodies the machine has: torso + 4 legs x (coxa, femur, tibia).
#: Asserted after conversion, because a partial `merge_fixed_joints` changes
#: the body list without raising.
EXPECTED_BODY_COUNT = 13

#: Where the torso is authored in the conversion input (delta 2).
CONVERSION_TORSO_Z_M = 0.0

#: What the authored asset says, re-stated here only to be ASSERTED against —
#: the transform reads the live XML; these catch the authored file drifting
#: under this module's assumptions.
AUTHORED_TORSO_Z_M = 0.35
AUTHORED_JOINT_STIFFNESS = 30.0
AUTHORED_JOINT_DAMPING = 1.0


def conversion_mjcf(no_shell: bool = False) -> str:
    """The robot-only MJCF that gets converted, as a string.

    Parses the committed `assets/spyder12.xml` and applies the module
    docstring's four deltas. Raises if the authored file has drifted from the
    shape this transform expects — a silent partial transform is exactly the
    "same regex, different machine" failure the Hound port documented.
    """
    tree = ET.parse(paths.SPYDER_XML)
    root = tree.getroot()

    # -- Delta 4a: mesh resolution. Relative to the conversion input's dir,
    # and COMPOSED with the authored `file="meshes/..."` refs — so this is
    # "../..", which resolves to assets/, and the refs complete the path.
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"{paths.SPYDER_XML} has no <compiler> element")
    compiler.set("meshdir", "../..")

    # -- Delta 3: passive dynamics and actuation move to spyder_cfg.py -------
    default_joint = root.find("default/joint")
    if default_joint is None:
        raise ValueError(f"{paths.SPYDER_XML} has no <default><joint> element")
    for attr, expect in (("stiffness", AUTHORED_JOINT_STIFFNESS),
                         ("damping", AUTHORED_JOINT_DAMPING)):
        got = float(default_joint.get(attr, "nan"))
        if got != expect:
            raise ValueError(
                f"authored default joint {attr} is {got}, this transform was "
                f"written against {expect}. Re-read spyder_usd.py's delta 3 "
                "before updating the constant — if the authored spring changed, "
                "the PD gains in spyder_cfg.py were derived against the old one."
            )
        default_joint.set(attr, "0")
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"{paths.SPYDER_XML} has no <actuator> block")
    if len(actuator) != 12:
        raise ValueError(f"expected 12 motors in {paths.SPYDER_XML}, found {len(actuator)}")
    root.remove(actuator)

    # -- Delta 1: no world ---------------------------------------------------
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"{paths.SPYDER_XML} lacks <asset> or <worldbody>")
    removed = []
    for el in list(asset):
        name = el.get("name", el.get("type", "?"))
        if el.tag == "texture" or name == "MatPlane":
            asset.remove(el)
            removed.append(f"{el.tag}:{name}")
    for el in list(worldbody):
        if el.tag in ("light", "geom"):  # the only world-level geom is the floor
            worldbody.remove(el)
            removed.append(f"{el.tag}:{el.get('name', '?')}")
    if sorted(removed) != sorted(
        ["texture:skybox", "texture:texplane", "material:MatPlane",
         "light:?", "geom:floor"]
    ):
        raise ValueError(
            f"world-stripping removed {removed!r}, not the five elements this "
            "transform enumerates. The authored world changed; re-derive the "
            "deltas rather than trusting a partial strip."
        )

    # -- Delta 2: torso at the origin; the per-env camera goes ---------------
    torso = worldbody.find("body[@name='torso']")
    if torso is None:
        raise ValueError("no <body name='torso'> under <worldbody>")
    if torso.get("pos") != f"0 0 {AUTHORED_TORSO_Z_M}":
        raise ValueError(
            f"authored torso pos is {torso.get('pos')!r}; this transform was "
            f"written against '0 0 {AUTHORED_TORSO_Z_M}'. If the stance height "
            "moved, update spyder_cfg.STAND_HEIGHT_M with it."
        )
    torso.set("pos", f"0 0 {CONVERSION_TORSO_Z_M:g}")
    cam = torso.find("camera[@name='track']")
    if cam is None:
        raise ValueError("no <camera name='track'> on the torso; delta 2 expects it")
    torso.remove(cam)

    # -- Optional: strip the visual shell (fallback, see module docstring) ---
    if no_shell:
        for parent in root.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("class") == "visual":
                    parent.remove(geom)
        for el in list(asset):
            if el.tag in ("mesh", "material"):
                asset.remove(el)

    root.set("model", "spyder12_isaac")
    body = ET.tostring(root, encoding="unicode")
    header = (
        "<!-- GENERATED BY bestiary/isaac/spyder_usd.py FROM assets/spyder12.xml.\n"
        "     DO NOT EDIT BY HAND. Re-generate with\n"
        "\n"
        "         PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.spyder_usd\n"
        "\n"
        "     This is spyder12.xml's robot with its world removed, its torso\n"
        "     authored at the origin, and its passive joint dynamics zeroed in\n"
        "     favour of spyder_cfg.py's PD drive. It is not a model to simulate\n"
        "     in MuJoCo: it has no ground, so a machine released in it falls\n"
        "     forever. The deltas and their reasons are spyder_usd.py's module\n"
        "     docstring; isaac/check_spyder.py asserts they are the ONLY deltas.\n"
        "-->\n"
    )
    # ET.tostring emits everything after the root tag opening; put the header
    # inside the root element so the result stays one well-formed document.
    first_break = body.index(">") + 1
    return body[:first_break] + "\n" + header + body[first_break:] + "\n"


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="re-run the conversion even if the output exists")
    parser.add_argument("--no-shell", action="store_true",
                        help="strip the visual shell meshes (physics unchanged)")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    return parser.parse_args()


def main(args_cli: argparse.Namespace) -> None:
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    # Reused, not copied: the truncated-tree-walk bug this patches is described
    # at its definition, and one definition is how it stays fixed. hound_usd
    # imports isaaclab.app at module scope, so the import lives here.
    from bestiary.isaac.hound_usd import activate_contact_reporting

    paths.SPYDER_ISAAC_DIR.mkdir(parents=True, exist_ok=True)

    # --force must REMOVE the old output: the importer writes a numeric-suffix
    # sibling rather than overwriting, measured on the Hound port.
    if args_cli.force and paths.SPYDER_ISAAC_USD.parent.is_dir():
        shutil.rmtree(paths.SPYDER_ISAAC_USD.parent)
        print(f"[bestiary] removed {paths.SPYDER_ISAAC_USD.parent} for --force", flush=True)

    paths.SPYDER_ISAAC_MJCF.write_text(conversion_mjcf(no_shell=args_cli.no_shell))
    print(f"[bestiary] wrote conversion input : {paths.SPYDER_ISAAC_MJCF}", flush=True)

    cfg = MjcfConverterCfg(
        asset_path=str(paths.SPYDER_ISAAC_MJCF),
        usd_dir=str(paths.SPYDER_ISAAC_DIR),
        force_usd_conversion=args_cli.force,
        # The load-bearing geoms are MuJoCo primitives (sphere + capsules) and
        # carry across as USD primitives; the shell is OBJ meshes that stay
        # visual-only because...
        merge_mesh=False,
        # ...this is False: the shell is contype=0 conaffinity=0 in the MJCF,
        # and True here would give every mesh a collider and change the physics.
        collision_from_visuals=False,
        # MuJoCo already excludes parent-child contacts, and that exclusion is
        # the load-bearing one here: each coxa interpenetrates its own torso
        # stub at the shared mount point, so self-collision would be a
        # permanent phantom contact force at rest. (Between LEGS there is
        # nothing to catch — neighbouring coxa bases are 0.40 m apart, a
        # 0.24 m surface gap, and the joint envelope never closes it under
        # ~0.13 m. Numbers from the MJCF; spyder_cfg.py repeats them.)
        self_collision=False,
        # `<option integrator="RK4" timestep="0.01"/>` belongs to the MuJoCo
        # runs. The env cfg owns dt here.
        import_physics_scene=False,
        robot_type="Quadruped",
    )
    converter = MjcfConverter(cfg)
    print(f"[bestiary] wrote USD             : {converter.usd_path}", flush=True)

    if converter.usd_path != str(paths.SPYDER_ISAAC_USD):
        raise AssertionError(
            f"MjcfConverter wrote {converter.usd_path!r} but paths.SPYDER_ISAAC_USD "
            f"is {str(paths.SPYDER_ISAAC_USD)!r}. Every downstream reader uses the "
            "constant, so they now disagree."
        )
    print("[bestiary] path constant matches the importer's layout", flush=True)

    bodies = activate_contact_reporting(converter.usd_path)
    print(f"[bestiary] contact reporting on {bodies} rigid bodies", flush=True)
    if bodies != EXPECTED_BODY_COUNT:
        raise AssertionError(
            f"patched contact reporting onto {bodies} rigid bodies, expected "
            f"{EXPECTED_BODY_COUNT} (torso + 4 x (coxa, femur, tibia)). Either "
            "the conversion lost links or it gained some."
        )


if __name__ == "__main__":
    # Same two traps as hound_usd: Kit teardown can drop stdout, and
    # SimulationApp.close() exits 0 no matter what. Flush and os._exit.
    import os
    import traceback

    from isaaclab.app import AppLauncher

    _args = _parse_args()
    AppLauncher(_args)
    try:
        main(_args)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
