"""Convert HOUND-16 (the PD variant) from MJCF to USD for Isaac Lab.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.hound_usd

Writes two files, both generated and both committed:

    assets/hound/isaac/hound16pd_isaac.xml                 the conversion input
    assets/hound/isaac/hound16pd_isaac/hound16pd_isaac.usda  what Isaac Lab loads

Re-run after any change to `robots/hound/build.py`. `isaac/check_hound.py` fails
if the two stop describing the same machine, so a stale USD is not a silent
condition.


WHY THE INPUT IS NOT `assets/hound16pd.xml` ITSELF
--------------------------------------------------
Two deltas, both required, both narrow. Every dimension, mass, limit, gain and
the solved stance still come from `build.py`'s `Spec` — this module imports the
generator's own fragments rather than re-emitting geometry, so there is exactly
one description of the robot and this is a re-wrapping of it.

1. NO WORLD. `hound16pd.xml` carries a `<geom name="floor" type="plane"
   size="40 40 0.1"/>`, and the MJCF importer converts it: the first attempt at
   this port produced a USD whose default prim contained an 80x80 m collision
   plane next to the robot. Isaac Lab spawns a robot asset once per environment,
   so that plane would be cloned under every env origin, on top of the terrain
   the whole exercise exists to stand on. The conversion input therefore has a
   `<worldbody>` containing the robot and nothing else.

   The light goes for the same reason: a directional light inside the robot
   prim is N lights in an N-robot scene, and the viewer supplies its own.

2. THE TRUNK IS AUTHORED AT THE ORIGIN, not at `spec.stand_z`. See
   `build.robot_xml`'s docstring — Isaac Lab ADDS an asset's authored transform
   to the spawn position it was given, so a stance height baked into the asset
   is an unrequested 0.3634 m of drop. `isaac/hound_cfg.py` supplies the stand
   height as `init_state.pos` instead, which is where it belongs.

Not a delta: the stance springs. `Spec.stiffness()` and `Spec.springref()` are a
labelled MuJoCo reset crutch, and they must not become PhysX drive gains — but
they do not, and the mechanism is worth knowing rather than trusting. The
importer writes MuJoCo joint attributes into a separate `mujoco` USD variant
(`mjc:stiffness`, `mjc:springref`) and the PhysX/Newton path into a `physx`
variant, and the interface layer selects `physx`. So the twelve `physics:drive`
stiffnesses in the loaded asset are the `<position>` actuators' kp (60 / 80 /
90) and the springs (12.12 / 18.58 / 32.77) appear in no layer PhysX reads.
`check_hound.py` asserts both halves of that.


WHAT THE IMPORTER GETS RIGHT, MEASURED RATHER THAN ASSUMED
----------------------------------------------------------
Checked by reading the emitted USD (and re-checked on every run by
`check_hound.py`), because each of these was a plausible way for the port to be
quietly wrong:

  * The wheel's collider is a `UsdGeomCylinder` primitive with radius 0.085 m
    and height 0.05 m — not a cooked convex mesh. A meshed wheel becomes an
    N-gon and rolls with N contact impulses per revolution, which is what wheel
    chatter is.
  * The wheel joint arrives as a revolute joint with NO authored limit
    attributes. What the SOLVER reports is not "unlimited" but a huge finite
    range, and which one depends on the backend: PhysX says ±3.4028235e38
    (FLT_MAX), Newton/MJWarp says ±1e10, both measured on this asset. Either way
    it is far past the ±2π where a position drive would break, which is why the
    wheels are velocity-driven in `hound_cfg.py`. The joint also arrives with no
    drive at all, because a MuJoCo `<motor>` has no PhysX position-drive
    equivalent — the importer says so out loud ("Gain type or bias type not
    available or supported for actuator ... FL_wheel").
  * Colliders exist on exactly the seventeen load-bearing geoms. The
    `class="deco"` and `class="hub"` geoms — `contype=0 conaffinity=0` in the
    MJCF — arrive with no collision API, so the physics is the same as if they
    were deleted, which is the invariant the MuJoCo model already holds.
  * Masses arrive exactly: 6.921 / 0.678 / 1.152 / 0.241 / 0.45 kg.
  * `armature` and `frictionloss` survive as `physxJoint:armature` and
    `physxJoint:jointFriction` (0.01 / 0.2 for the legs, 0.004 / 0.399 for the
    wheels).
  * Angular limits arrive in DEGREES. The knee's [-2.60, -0.60] rad is written
    as [-148.969, -34.377]. Anything wrong by a factor of 57.3 downstream is
    this, and `check_hound.py` converts back before comparing to `Spec`.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from isaaclab.app import AppLauncher

from bestiary import paths
from bestiary.robots.hound.build import (
    SPEC,
    Spec,
    actuator_xml,
    common_head,
    keyframe_xml,
    robot_xml,
)

# IMPORTING THIS MODULE MUST NOT LAUNCH KIT. `check_hound.py` imports
# `conversion_mjcf` to prove the committed input is still what `build.py`
# produces, and it already has an app of its own -- a second AppLauncher at
# import time would be a second Isaac Sim in one process. So the launch and the
# heavy `isaaclab.sim.converters` import both live inside `main()`.
# `isaaclab.app` itself is safe at module scope; it is what launches Kit, not
# something that needs Kit.

#: Where the trunk body is authored in the conversion input. Zero, not
#: `SPEC.stand_z` -- see the module docstring, delta 2.
CONVERSION_TRUNK_Z_M = 0.0

#: Rigid bodies the machine has: trunk + 4 legs x (abduct, thigh, calf, wheel).
#: Asserted after conversion rather than assumed, because the sibling robot's
#: card warns that `merge_fixed_joints` may not merge since Isaac Sim 5.1 and a
#: partial merge changes the body list without raising.
EXPECTED_BODY_COUNT = 17


def conversion_mjcf(spec: Spec) -> str:
    """The robot-only MJCF that gets converted.

    Every fragment is `build.py`'s own, so this function contains no geometry,
    no masses and no gains. What it chooses is the *world*, and the world is
    empty.
    """
    return f"""<mujoco model="hound16pd_isaac">
  <!-- GENERATED BY bestiary/isaac/hound_usd.py FROM robots/hound/build.py.
       DO NOT EDIT BY HAND. Re-generate with

           PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.hound_usd

       This is assets/hound16pd.xml's robot, with its world removed and its
       trunk authored at the origin, as the input to Isaac Lab's MJCF importer.
       It is not a model to simulate in MuJoCo: it has no ground, so a machine
       released in it falls forever. Use assets/hound16pd.xml for that.

       The two deltas from assets/hound16pd.xml, and why they exist, are in
       bestiary/isaac/hound_usd.py's module docstring. isaac/check_hound.py
       asserts that they are the ONLY two deltas.
  -->
{common_head(spec, "hound16pd_isaac").format(extra_assets="")}
  <worldbody>{robot_xml(spec, trunk_z=CONVERSION_TRUNK_Z_M)}
  </worldbody>

  <actuator>
{actuator_xml(spec, "pd")}
  </actuator>

{keyframe_xml(spec)}
</mujoco>
"""


def activate_contact_reporting(usd_path: str) -> int:
    """Give every rigid body in the converted asset `PhysxContactReportAPI`.

    Returns the number of bodies touched. Edits the asset's interface layer, so
    the additions appear as `over` statements next to the importer's own output
    and are rewritten on every conversion.

    WHY THIS IS NEEDED, AND WHY NOTHING ELSE DOES IT
    ------------------------------------------------
    `ArticulationCfg.spawn.activate_contact_sensors = True` is supposed to be
    exactly this. It calls `isaaclab.sim.schemas.activate_contact_sensors`, which
    walks the spawned prim and applies the API to rigid bodies -- and stops
    descending the moment it finds one:

        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            ...add PhysxContactReportAPI...
        else:
            all_prims += child_prim.GetChildren()

    with the comment "nested rigid bodies are not allowed by SDK so we can safely
    assume that if a prim has a rigid body API, it is a rigid body and we don't
    need to check its children". That assumption holds for a URDF import, which
    flattens every link to a sibling. It does not hold for an MJCF import, which
    keeps the kinematic tree: Hound's `trunk` is the first rigid body found AND
    the ancestor of the other sixteen, so the walk applies the API to the trunk
    and returns.

    The symptom is a contact sensor that resolves ONE body. Isaac Lab's shipped
    locomotion rewards then die at env construction with

        ValueError: Not all regular expressions are matched!
            .*_wheel: []
        Available strings: ('trunk',)

    which reads as a bad regex and is a truncated tree walk. Applying the API in
    the asset makes the sensor's own predicate -- which does recurse fully --
    find all seventeen.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"USD failed to open {usd_path} for contact-report patching")
    stage.SetEditTarget(stage.GetRootLayer())
    touched = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        over = stage.OverridePrim(prim.GetPath())
        if "PhysxContactReportAPI" not in over.GetAppliedSchemas():
            over.AddAppliedSchema("PhysxContactReportAPI")
        # Threshold 0 N: report every contact. The same value Isaac Lab's own
        # helper writes, and the sensors downstream apply their own thresholds.
        over.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)
        touched += 1
    stage.GetRootLayer().Save()
    return touched


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run the USD conversion even if the output already exists. The "
            "converter is otherwise lazy and keyed on a hash of the INPUT MJCF, "
            "so a change to build.py forces a rebuild on its own -- this is for "
            "when the importer itself changed."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    # A converter has nothing to look at. `--viz none` is the 3.0 spelling of
    # headless; `--headless` is deprecated.
    parser.set_defaults(visualizer=["none"])
    return parser.parse_args()


def main(args_cli: argparse.Namespace) -> None:
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    paths.HOUND_ISAAC_DIR.mkdir(parents=True, exist_ok=True)

    # A forced re-conversion has to REMOVE the old output, not just ask for a
    # rebuild. The importer does not overwrite an existing output directory; it
    # writes a sibling with a numeric suffix, so `--force` alone produced
    # `hound16pd_isaac_1/hound16pd_isaac.usda` and left the committed asset
    # untouched while reporting success. The path assertion at the end of this
    # function is what caught that, and this is the fix.
    if args_cli.force and paths.HOUND_ISAAC_USD.parent.is_dir():
        shutil.rmtree(paths.HOUND_ISAAC_USD.parent)
        print(f"[bestiary] removed {paths.HOUND_ISAAC_USD.parent} for --force", flush=True)

    paths.HOUND_ISAAC_MJCF.write_text(conversion_mjcf(SPEC))
    print(f"[bestiary] wrote conversion input : {paths.HOUND_ISAAC_MJCF}", flush=True)

    cfg = MjcfConverterCfg(
        asset_path=str(paths.HOUND_ISAAC_MJCF),
        usd_dir=str(paths.HOUND_ISAAC_DIR),
        force_usd_conversion=args_cli.force,
        # Every collider in this model is already a MuJoCo primitive -- box,
        # capsule or cylinder -- and the importer carries primitives across as
        # USD primitives. Nothing here is a mesh, so `collision_type` never
        # gets a chance to cook one, and in particular the wheel stays a
        # cylinder rather than becoming a convex hull of one.
        merge_mesh=False,
        # The MJCF's decorative geoms are contype=0 conaffinity=0 on purpose.
        # True here would give all of them colliders and change the physics.
        collision_from_visuals=False,
        # MuJoCo already excludes parent-child contacts, and this machine's
        # links are not authored to survive being collided with each other.
        self_collision=False,
        # `<option integrator="implicitfast" timestep="0.005"/>` and the visual
        # block belong to the MuJoCo runs. Isaac Lab's env cfg owns dt and
        # gravity here, and importing a second physics scene would fight it.
        import_physics_scene=False,
        robot_type="Quadruped",
    )
    converter = MjcfConverter(cfg)
    print(f"[bestiary] wrote USD             : {converter.usd_path}", flush=True)

    # The path constant and the importer's own output layout must agree, or
    # `hound_cfg.py` loads a file that is not the one just written. The
    # importer derives that layout from the input stem, so this is exactly the
    # assertion that catches a rename.
    if converter.usd_path != str(paths.HOUND_ISAAC_USD):
        raise AssertionError(
            f"MjcfConverter wrote {converter.usd_path!r} but "
            f"paths.HOUND_ISAAC_USD is {str(paths.HOUND_ISAAC_USD)!r}. Every "
            "downstream reader uses the constant, so they now disagree."
        )
    print("[bestiary] path constant matches the importer's layout", flush=True)

    bodies = activate_contact_reporting(converter.usd_path)
    print(f"[bestiary] contact reporting on {bodies} rigid bodies", flush=True)
    if bodies != EXPECTED_BODY_COUNT:
        raise AssertionError(
            f"patched contact reporting onto {bodies} rigid bodies, expected "
            f"{EXPECTED_BODY_COUNT} (trunk + 4 x (abduct, thigh, calf, wheel)). "
            "Either the conversion lost links or it gained some."
        )


if __name__ == "__main__":
    # The traceback is printed and flushed HERE, and the process is ended with
    # `os._exit` rather than by closing Kit. Two separate traps, both measured:
    # Kit's teardown can end the process without draining Python's stdout, and
    # `SimulationApp.close()` ends it with status 0 no matter what -- so a
    # `SystemExit(1)` placed after `close()` never happens and a failed
    # conversion reports success. See `check_hound._exit`.
    import os
    import traceback

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
