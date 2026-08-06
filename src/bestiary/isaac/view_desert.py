"""Look at the Bestiary desert -- and at a robot standing on it -- inside Isaac Lab.

This is the "is it actually our ground?" check, and it is deliberately a viewer
rather than a training script. A heightfield that loads without raising can
still be wrong in ways only a picture shows: transposed, mirrored, flattened by
a bad vertical scale, or tiled from the same patch 200 times. Numbers cannot
catch those; eyes can, in about three seconds.

The same argument is why `--robot hound` exists. `check_hound.py` can prove that
sixteen joints arrived with the right limits, the right masses and the right
gains, and still not notice a leg mirrored, a wheel inside a calf, or a machine
standing 30 cm into the sand. To see HOUND-16 on the desert:

    cd Bestiary && export PYTHONPATH=$PWD/src
    VIRTUAL_ENV=$HOME/isaaclab-env ~/IsaacLab/isaaclab.sh -p \\
        -m bestiary.isaac.view_desert \\
        --robot hound --robots 4 --rows 3 --cols 3 \\
        --spawn flat --frame robot --hold-pose \\
        --terrain-color 0.85 0.82 0.72

`VIRTUAL_ENV` is not optional: `isaaclab.sh` falls back to system Python without
it and then cannot import `isaaclab` at all. Check its first `[INFO]` line.

TWO THINGS THAT LOOK LIKE PHYSICS BUGS AND ARE NOT, BOTH MEASURED HERE

1. A ROBOT SPAWNED AT AN ENV ORIGIN CAN FALL METRES. Isaac Lab computes a
   sub-terrain's origin z as the MAXIMUM height in a 2 x 2 m window at the tile
   centre (`height_field_to_mesh` in isaaclab/terrains/height_field/utils.py).
   On their own terrains that maximum is the surface, because `random_rough` has
   two to ten centimetres of noise. On a real dune face it is not: measured tile
   origins here reached 2.31 m while the surface directly under the origin was
   0.4 m, so a robot placed at "origin + standing height" free-fell 1.9 m and
   landed on its back. `--spawn flat` asks Isaac Lab's own flat-patch sampler
   for real ground instead, and after that the drop is ~2 cm.

2. WITHOUT A CONTROLLER, HOUND ROLLS AWAY AND TIPS OVER ON A SLOPE, and that is
   the machine being correct rather than the port being wrong.
   `robots/hound/CARD.md` says it outright: a wheel is free along its rolling
   direction, so the stance is an inverted pendulum over a contact patch that
   translates, and "a real wheel-legged robot holds still by BRAKING its wheels,
   and a policy here has to learn the same thing." A velocity drive commanded to
   zero cannot hold a static load on a slope either -- it only makes torque from
   a velocity error -- so the machine creeps downhill and eventually trips.
   Measured over 8 s on four sampled patches: two stood in the stance
   (trunk 0.36 m above the patch, trunk-to-axle 0.267 and 0.249 m against a
   nominal 0.278 m), one settled leaning ~49 degrees, one fell. `--hold-pose`
   pins it for inspection; it does not fix this, because there is nothing here to
   fix.

   (What DID have to be fixed to get that far was a missing static-friction
   coefficient in the converted asset -- see `hound_cfg.py`. Before that, all
   four went onto their backs inside half a second, every time.)

Four mixes are offered so the comparisons are direct:

    --mix desert        every tile is a patch of assets/terrain/desert_hfield.bin
    --mix blend         the desert alongside Isaac Lab's own slopes and rock fields
    --mix gentle        every tile is a patch of assets/terrain/gentle_hfield.bin
    --mix gentle-blend  the gentle terrain alongside the same Isaac Lab tiles —
                        EXACTLY what Bestiary-Gentle-Spyder-v0 trains on

`blend` is the one that matters for Hound training; `gentle-blend` for Spyder.
Isaac Lab's shipped rough config uses `noise_range=(0.02, 0.10)` — two to ten
centimetres — while the desert at difficulty 1.0 has metres of relief and the
gentle asset tops out at 1.0 m, so each mix gives its curriculum a span instead
of a single texture.

DEFAULTS TO THE NEWTON VIEWER, ON PURPOSE

Isaac Sim's Kit viewport loads ~110 extensions and ~1800 threads and is not
usable on a 16-thread desktop that is also running an editor. The Newton
visualizer is a pyglet/OpenGL window with none of that. Pass `--viz kit` if you
want the heavy one anyway.

Example:

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p \\
        -m bestiary.isaac.view_desert --mix blend
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mix",
    choices=["desert", "blend", "gentle", "gentle-blend"],
    default="desert",
    help=(
        "desert/gentle: every tile from that heightfield. blend/gentle-blend: "
        "the heightfield plus Isaac Lab's built-ins — gentle-blend is the "
        "Spyder training mix."
    ),
)
parser.add_argument(
    "--tile-m", type=float, default=8.0, help="Square sub-terrain size in metres (default 8.0)."
)
parser.add_argument(
    "--horizontal-scale",
    type=float,
    default=None,
    help=(
        "Terrain sampling in metres. Defaults to the desert's native 0.078125 m "
        "(hardcoded here rather than imported: isaac_hf pulls in isaaclab, which must "
        "not be imported before AppLauncher runs). Native is what you want for "
        "INSPECTING the asset; pass 0.1 to see what actually TRAINS -- native "
        "segfaults PhysX above ~64 envs at 200 tiles, so training uses 0.1 m. Viewing "
        "both is the only way to judge what the coarser grid costs visually."
    ),
)
parser.add_argument(
    "--rows", type=int, default=5, help="Sub-terrain grid rows (default 5)."
)
parser.add_argument(
    "--cols", type=int, default=5, help="Sub-terrain grid columns (default 5)."
)
parser.add_argument(
    "--curriculum",
    action="store_true",
    help="Ramp difficulty across rows: flattest real ground at row 0, roughest at the last.",
)
parser.add_argument(
    "--color-scheme",
    choices=["height", "random", "none"],
    default="none",
    help=(
        "Tile colouring. Defaults to 'none' because 'height' is BROKEN in this "
        "install: Isaac Lab's color_meshes_by_height passes a colormap name that "
        "the installed trimesh rejects ('Included color maps are: magma, inferno, "
        "plasma, viridis'), and TerrainImporter raises before a window opens. "
        "Not our bug and not worth patching theirs -- the robots give scale."
    ),
)
parser.add_argument(
    "--terrain-color",
    type=float,
    nargs=3,
    metavar=("R", "G", "B"),
    default=(1.0, 1.0, 1.0),
    help=(
        "Terrain diffuse colour, 0-1 per channel. Defaults to white because "
        "TerrainImporterCfg.visual_material defaults to "
        "PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)) -- literally black, which "
        "makes real relief nearly invisible under a dome light. Try 0.85 0.82 0.72 "
        "for something desert-coloured, or lower values if pure white washes out "
        "the shading that shows slope."
    ),
)
parser.add_argument(
    "--light-intensity",
    type=float,
    default=3000.0,
    help="Dome light intensity (default 3000). Raise if the terrain reads flat.",
)
parser.add_argument(
    "--robots",
    type=int,
    default=4,
    help=(
        "Quadrupeds to stand on the terrain, for scale (default 4). "
        "NOT cosmetic: the Newton visualizer builds its render model by walking the "
        "stage for articulated bodies, and refuses to initialize on a scene that has "
        "none -- 'Model must be set before calling set_visible_worlds()'. So a "
        "terrain-only scene is only viewable under --viz kit. 0 is allowed for "
        "headless geometry checks."
    ),
)
parser.add_argument(
    "--robot",
    choices=["anymal-c", "hound", "spyder"],
    default="anymal-c",
    help=(
        "Which machine to stand on the tiles. 'anymal-c' is Isaac Lab's own and "
        "the default, because for a TERRAIN check a known-good robot is the one "
        "that isolates the terrain. 'hound' and 'spyder' are ours, from "
        "assets/hound/isaac/ and assets/spyder/isaac/ -- generate them first "
        "with `-m bestiary.isaac.hound_usd` / `-m bestiary.isaac.spyder_usd`."
    ),
)
parser.add_argument(
    "--frame",
    choices=["grid", "robot"],
    default="grid",
    help=(
        "Where to point the camera. 'grid' frames the whole tiled area, which is "
        "what a terrain check wants. 'robot' sits under two metres from robot 0, "
        "which is what LOOKING AT A ROBOT wants: Hound stands 0.3634 m tall, and "
        "from across a 24 m grid that is about two pixels. Either way the "
        "viewer's own WASD / mouse controls take over from there."
    ),
)
parser.add_argument(
    "--spawn",
    choices=["origin", "flat"],
    default="origin",
    help=(
        "Where to stand the robots. 'origin' uses each env's terrain origin, "
        "which is the DEFAULT ONLY FOR BACKWARD COMPATIBILITY -- Isaac Lab takes "
        "that z as the maximum height in a 2 m window at the tile centre, and on "
        "a dune face the surface below it can be nearly two metres lower "
        "(measured). 'flat' asks Isaac Lab's own flat-patch sampler for ground "
        "that is actually flat and actually there, and is what you want with a "
        "robot. See --max-height-diff if 'flat' cannot find any."
    ),
)
parser.add_argument(
    "--max-height-diff",
    type=float,
    default=0.06,
    help=(
        "With --spawn flat: how much relief a spawn patch may have, in metres, "
        "across --patch-radius (default 0.06). This is a real constraint on real "
        "terrain, not a formality: at 0.05 m over a 0.35 m radius the sampler "
        "exhausted 10,000 rejection-sampling iterations and raised 'Failed to "
        "find valid patches', because the desert has no ground that flat at that "
        "radius. Raise it if it raises. It is also added to the spawn height, "
        "because the sampler's z is a ray-cast on the patch RING and the ground "
        "inside the ring can be this much higher -- at 0.15 m of allowed relief "
        "a wheel started 11 cm inside the mesh and PhysX depenetration threw the "
        "machine onto its back in a tenth of a second."
    ),
)
parser.add_argument(
    "--patch-radius",
    type=float,
    default=None,
    help=(
        "With --spawn flat: radius of the patch that must be flat, in metres. "
        "DEFAULTS PER ROBOT (see ROBOT_CHOICES): 0.25 for ANYmal-C and Hound "
        "(Hound's contact patches sit at sqrt(0.1934^2 + 0.142^2) = 0.240 m "
        "from its centre), 1.2 for Spyder, whose X-stance puts each foot at "
        "sqrt(0.76^2 + 0.76^2) = 1.075 m from the torso axis — a 0.25 m patch "
        "would certify flat ground under the body and let all four feet hang "
        "over edges. Larger is stricter and finds fewer patches."
    ),
)
parser.add_argument(
    "--physics",
    choices=["newton", "physx"],
    default="newton",
    help=(
        "Which solver to run. Defaults to 'newton' BECAUSE THE NEWTON VISUALIZER "
        "REQUIRES IT: the visualizer draws `NewtonManager.get_model()`, and under "
        "PhysX there is no Newton model, so it initializes with None, fails "
        "`set_visible_worlds()` and is removed before the first frame -- a run "
        "that steps forever and never opens a window. Isaac Lab's own message "
        "for that is 'Model must be set before calling set_visible_worlds()', "
        "which reads like a scene-content problem and is a backend problem. Use "
        "'physx' for a headless geometry check, or with --viz kit."
    ),
)
parser.add_argument(
    "--hold-pose",
    action="store_true",
    help=(
        "Pin each robot: write its root pose, zero root velocity and the stance "
        "joint state every step, so it stands exactly where it was placed. THIS "
        "IS A DISPLAY MODE AND IT PROVES NOTHING ABOUT THE PHYSICS -- it is for "
        "looking at the machine's geometry, its wheels and its stance on real "
        "ground from any angle. Leave it off to watch what actually happens, "
        "which for Hound is a slow roll downhill and a tumble (see the module "
        "docstring, point 2)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
# The whole point of this script is to look at something, so a visualizer is the
# default -- but the light one. See the module docstring.
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Everything below needs the app running before it can be imported. -------

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.terrains import (  # noqa: E402
    FlatPatchSamplingCfg,
    TerrainImporter,
    TerrainImporterCfg,
)
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg  # noqa: E402
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa: E402

from bestiary import paths  # noqa: E402
from bestiary.isaac.hound_cfg import HOUND16_CFG  # noqa: E402
from bestiary.isaac.spyder_cfg import SPYDER12_CFG  # noqa: E402
from bestiary.terrain.gentle import Z_SPAN_M as GENTLE_Z_SPAN_M  # noqa: E402
from bestiary.terrain.isaac_hf import (  # noqa: E402
    DESERT_NATIVE_CELL_M,
    DESERT_Z_SPAN_M,
    HfBestiaryDesertTerrainCfg,
    load_desert_m,
)

#: Key the flat-patch sampler files its results under, in
#: `TerrainImporter.flat_patches`. One name, used by the config and the reader.
FLAT_PATCH_NAME = "spawn"

#: Sliding-friction coefficient of the desert ground, read off the committed
#: model: `assets/hound16pd_desert.xml`'s hfield floor carries
#: `friction="0.8 0.5 0.5"`, and MuJoCo's first friction term is the sliding one.
DESERT_GROUND_FRICTION = 0.8

#: The machines this viewer can stand on the terrain, and the label to print.
#:
#: Each config's OWN `init_state.pos[2]` is used as the spawn height, rather than
#: a number written here. That is not tidiness: a spawn that intersects the mesh
#: is resolved by the solver as an explosion rather than as a placement, and the
#: right clearance is a property of the robot (0.6 m for ANYmal-C, 0.3684 m for
#: Hound -- its 0.3634 m stance plus 5 mm) which the config already states.
#: (label, articulation cfg, flat-patch radius in metres). The radius is the
#: smallest circle the STANDING machine's contacts fit inside, computed from
#: each model's geometry — see --patch-radius's help for both derivations.
ROBOT_CHOICES: dict[str, tuple[str, ArticulationCfg, float]] = {
    "anymal-c": ("ANYmal-C", ANYMAL_C_CFG, 0.25),
    "hound": ("HOUND-16", HOUND16_CFG, 0.25),
    "spyder": ("Spyder-12", SPYDER12_CFG, 1.2),
}

#: Which committed heightfield each --mix draws from, and the span its
#: normalised samples cover. One place, because main()'s printout and
#: build_terrain_cfg() must agree on which world is on screen.
MIX_ASSET: dict[str, tuple[str, float]] = {
    "desert": (str(paths.DESERT_HFIELD), DESERT_Z_SPAN_M),
    "blend": (str(paths.DESERT_HFIELD), DESERT_Z_SPAN_M),
    "gentle": (str(paths.GENTLE_HFIELD), GENTLE_Z_SPAN_M),
    "gentle-blend": (str(paths.GENTLE_HFIELD), GENTLE_Z_SPAN_M),
}


def build_terrain_cfg() -> TerrainGeneratorCfg:
    """The sub-terrain mix named by ``--mix``.

    `horizontal_scale` is set to the desert's own cell size rather than Isaac
    Lab's default 0.1 m. At 0.078125 m the resample in `isaac_hf` is very nearly
    the identity, so what appears on screen is the committed samples and not a
    smoothed interpretation of them — which is the only way this script can
    answer the question it exists to answer.
    """
    hfield_path, z_span = MIX_ASSET[args_cli.mix]
    blended = args_cli.mix in ("blend", "gentle-blend")
    desert = HfBestiaryDesertTerrainCfg(
        proportion=0.5 if blended else 1.0,
        hfield_path=hfield_path,
        z_span_m=z_span,
        border_width=0.25,
    )

    sub_terrains: dict = {"bestiary_desert": desert}
    if blended:
        # Isaac Lab's own, kept at their shipped parameters so the contrast with
        # the desert is visible rather than tuned away.
        sub_terrains["isaac_slope"] = terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        )
        sub_terrains["isaac_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        )

    if args_cli.spawn == "flat":
        # EVERY sub-terrain gets the sampler, not just the desert. The generator
        # allocates one flat-patch tensor per NAME and fills it only for the
        # (row, col) whose sub-terrain asked for it; a tile that did not ask
        # keeps zeros, and a zero is a perfectly valid-looking position at the
        # world origin. Robots placed there would appear buried at (0, 0, 0),
        # which reads as a spawn bug rather than as an unfilled tensor.
        # Per-robot default: the radius is a property of the machine standing
        # on the patch, so an unset flag reads it from ROBOT_CHOICES.
        patch_radius = (
            ROBOT_CHOICES[args_cli.robot][2]
            if args_cli.patch_radius is None
            else args_cli.patch_radius
        )
        for cfg in sub_terrains.values():
            cfg.flat_patch_sampling = {
                FLAT_PATCH_NAME: FlatPatchSamplingCfg(
                    # One per tile is all this viewer uses; asking for more is
                    # rejection sampling nobody reads.
                    num_patches=1,
                    patch_radius=patch_radius,
                    max_height_diff=args_cli.max_height_diff,
                )
            }

    h_scale = (
        DESERT_NATIVE_CELL_M if args_cli.horizontal_scale is None else args_cli.horizontal_scale
    )
    return TerrainGeneratorCfg(
        size=(args_cli.tile_m, args_cli.tile_m),
        border_width=20.0,
        num_rows=args_cli.rows,
        num_cols=args_cli.cols,
        horizontal_scale=h_scale,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=args_cli.curriculum,
        color_scheme=args_cli.color_scheme,
        sub_terrains=sub_terrains,
    )


def build_visualizer_cfgs() -> list:
    """Pre-create the Newton visualizer's config, so its camera can be aimed.

    `sim.set_camera_view()` DOES NOT MOVE THE NEWTON CAMERA. `BaseVisualizer.
    set_camera_view` is a bare `pass` and `NewtonVisualizer` does not override
    it; the Newton camera is read once, from `cfg.eye` / `cfg.lookat`, inside
    `initialize()`. Measured: the window opened at the cfg default (4, -4, 3)
    while this script had asked for a pose next to robot 0, both before and after
    `sim.reset()`.

    So the config has to exist before `SimulationContext` is built and be edited
    before `reset()` creates the visualizer from it. Returning it here rather than
    letting Isaac Lab create a default is the whole mechanism.

    Empty list for anything but Newton: `--viz kit` and `--viz none` are resolved
    by Isaac Lab, and Kit's camera does respond to `set_camera_view`.
    """
    if "newton" not in (args_cli.visualizer or []):
        return []
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    return [NewtonVisualizerCfg()]


def build_physics_cfg():
    """The solver named by ``--physics``.

    The Newton settings are Isaac Lab's own `RoughPhysicsCfg.newton_mjwarp`,
    copied rather than imported so this viewer does not depend on a task package
    -- including the 1 cm shape margin, which their changelog calls "the single
    most important Newton setting for rough terrain" because without it
    non-ANYmal-D robots fail to make stable contact on triangle-mesh ground.

    Returns None for PhysX, which is `SimulationCfg.physics`'s own default.
    """
    if args_cli.physics == "physx":
        return None
    from isaaclab_newton.physics import (
        MJWarpSolverCfg,
        NewtonCfg,
        NewtonCollisionPipelineCfg,
        NewtonShapeCfg,
    )

    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=200,
            nconmax=100,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
    )


def resolve_spawn_points(terrain: TerrainImporter, count: int) -> list[tuple[float, float, float]]:
    """Ground positions to stand `count` robots on, one per tile, world frame.

    `--spawn origin` returns the env origins, which is what this script always
    did. `--spawn flat` returns one sampled flat patch per tile instead, taken in
    row-major order so the robots spread across the grid rather than piling into
    whichever tile happened to sample first.

    The difference is not cosmetic. An env origin's z is
    `max(heights[centre 2x2 m]) * vertical_scale`, so on ground with metres of
    relief it can sit far above the surface beneath it -- measured 2.31 m against
    a surface at ~0.4 m, a 1.9 m free fall that lands a wheeled machine on its
    back. A flat patch's z is a ray-cast hit on the mesh, so the drop is the
    clearance you asked for and nothing else.
    """
    if args_cli.spawn == "origin":
        origins = terrain.env_origins
        return [tuple(float(v) for v in origins[i]) for i in range(count)]

    patches = terrain.flat_patches.get(FLAT_PATCH_NAME)
    if patches is None:
        raise RuntimeError(
            f"--spawn flat but the terrain reports no {FLAT_PATCH_NAME!r} flat "
            f"patches (it has {sorted(terrain.flat_patches)}). Flat-patch "
            "sampling is only available for terrain_type='generator'."
        )
    # (num_rows, num_cols, num_patches, 3) -> one patch per tile, row-major.
    per_tile = patches[:, :, 0, :].reshape(-1, 3)
    if per_tile.shape[0] < count:
        raise RuntimeError(
            f"--spawn flat can place at most one robot per tile: "
            f"{args_cli.rows}x{args_cli.cols} = {per_tile.shape[0]} tiles but "
            f"--robots {count}. Raise --rows/--cols or lower --robots."
        )
    return [tuple(float(v) for v in per_tile[i]) for i in range(count)]


def main() -> None:
    # Report what is about to be drawn, so a wrong asset is caught before the
    # window opens rather than squinted at afterwards.
    # flush=True throughout: Kit's teardown can end the process without draining
    # Python's stdout buffer, which silently swallowed these lines the first time.
    hfield_path, z_span = MIX_ASSET[args_cli.mix]
    desert = load_desert_m(hfield_path, z_span)
    cells = int(round(args_cli.tile_m / DESERT_NATIVE_CELL_M))
    print(f"[bestiary] heightfield : {hfield_path}", flush=True)
    print(f"[bestiary] terrain     : {desert.shape[0]}x{desert.shape[1]} cells, "
          f"{DESERT_NATIVE_CELL_M * 100:.4f} cm/cell, relief {desert.max():.3f} m", flush=True)
    print(f"[bestiary] tile        : {args_cli.tile_m:.2f} m = {cells}x{cells} native cells",
          flush=True)
    h_scale = (
        DESERT_NATIVE_CELL_M if args_cli.horizontal_scale is None else args_cli.horizontal_scale
    )
    print(f"[bestiary] sampling    : {h_scale:.6f} m/cell "
          f"({'NATIVE' if args_cli.horizontal_scale is None else 'resampled'}) -> "
          f"{int(args_cli.tile_m / h_scale)}x{int(args_cli.tile_m / h_scale)} px per tile",
          flush=True)
    print(f"[bestiary] grid        : {args_cli.rows}x{args_cli.cols} tiles, mix={args_cli.mix}, "
          f"curriculum={args_cli.curriculum}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, physics=build_physics_cfg(), visualizer_cfgs=build_visualizer_cfgs()
    )
    print(f"[bestiary] physics     : {args_cli.physics}", flush=True)
    sim = sim_utils.SimulationContext(sim_cfg)

    light = sim_utils.DomeLightCfg(intensity=args_cli.light_intensity, color=(1.0, 1.0, 1.0))
    light.func("/World/Light", light)

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=build_terrain_cfg(),
        num_envs=max(args_cli.robots, 1),
        env_spacing=args_cli.tile_m,
        max_init_terrain_level=None,
        debug_vis=False,
    )
    # Ground friction. TerrainImporterCfg defaults to
    # RigidBodyMaterialCfg(static=0.5, dynamic=0.5), which is nobody's terrain;
    # the desert MJCF gives its hfield floor friction="0.8 0.5 0.5", so the
    # sliding coefficient of this ground is 0.8. Set from the asset rather than
    # left at a library default, because whether a wheeled machine holds station
    # is decided here.
    terrain_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=DESERT_GROUND_FRICTION,
        dynamic_friction=DESERT_GROUND_FRICTION,
        restitution=0.0,
    )
    if args_cli.color_scheme in ("height", "random"):
        # A visual material would paint over the colour scheme we asked for.
        terrain_cfg.visual_material = None
    else:
        terrain_cfg.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(args_cli.terrain_color)
        )
    terrain = TerrainImporter(terrain_cfg)

    # Stand quadrupeds on the tiles. See --robots: without at least one
    # articulation the Newton visualizer has no model and raises rather than
    # opening a window. They also give the relief a human scale, which is the
    # difference between "that looks bumpy" and "that dune is taller than the dog".
    label, robot_cfg, _ = ROBOT_CHOICES[args_cli.robot]
    # With --spawn flat, the sampled z is a ray-cast on the patch's RING, so the
    # ground inside the ring may be up to --max-height-diff higher. Spawning at
    # the ring height therefore risks starting a wheel INSIDE the mesh, and PhysX
    # resolves that at max_depenetration_velocity, which is a kick, not a
    # placement. Adding the allowed relief turns the worst case from "buried" into
    # "a short drop", which is what --max-height-diff's help says.
    spawn_z = float(robot_cfg.init_state.pos[2])
    if args_cli.spawn == "flat":
        spawn_z += args_cli.max_height_diff
    spots = resolve_spawn_points(terrain, args_cli.robots)
    robots: list[Articulation] = []
    for i in range(args_cli.robots):
        cfg = robot_cfg.replace(prim_path=f"/World/envs/env_{i}/Robot")
        x, y, z = spots[i]
        cfg.init_state.pos = (x, y, z + spawn_z)
        robots.append(Articulation(cfg))
    print(f"[bestiary] robots      : {len(robots)} {label}, spawn={args_cli.spawn}, "
          f"{spawn_z:.4f} m above the resolved ground", flush=True)

    # Guarded because camera control is a visualizer-dependent capability, and a
    # viewer that cannot aim is not a reason to abandon a scene that built
    # correctly.
    span = max(args_cli.rows, args_cli.cols) * args_cli.tile_m
    if args_cli.frame == "robot" and robots:
        # Stand just under 2 m off robot 0 and a little above it. At the Newton
        # viewer's 65 degree field of view, 1.9 m of range covers about 2.4 m
        # vertically, so a 0.3634 m machine is roughly 15% of the frame height --
        # visible as a machine rather than as a speck, with enough ground around
        # it to see what it is standing on. At 3 m it was a speck.
        x, y, z = spots[0]
        eye = (x - 1.3, y - 1.3, z + spawn_z + 0.7)
        target = (x, y, z + spawn_z)
    else:
        eye = (span * 0.7, span * 0.7, span * 0.5)
        target = (0.0, 0.0, 0.0)
    # Two camera routes, because the two visualizers take it differently. Newton
    # reads its cfg at initialize() and ignores set_camera_view entirely (see
    # build_visualizer_cfgs); Kit responds to set_camera_view and has no cfg here.
    # Writing both costs nothing and neither one alone covers both viewers.
    for viz_cfg in sim_cfg.visualizer_cfgs:
        viz_cfg.eye = eye
        viz_cfg.lookat = target
    try:
        sim.set_camera_view(eye=eye, target=target)
    except Exception as exc:  # noqa: BLE001
        print(f"[bestiary] camera not settable ({type(exc).__name__}: {exc}); "
              "use the viewer's own controls", flush=True)
    print(f"[bestiary] camera      : eye {tuple(round(v, 3) for v in eye)} "
          f"-> target {tuple(round(v, 3) for v in target)}", flush=True)

    sim.reset()

    # PUT THE JOINTS IN THE STANCE. Nothing else does, and the failure is subtle
    # enough to be worth the paragraph.
    #
    # `ArticulationCfg.init_state.joint_pos` becomes `data.default_joint_pos`,
    # which is what an ACTION TERM offsets from and what a reset EVENT writes.
    # A standalone script has neither, so the joint state in the simulation is
    # whatever the USD authored -- and for a converted MJCF that is the model's
    # rest pose. Hound's rest pose is all-zeros, legs straight down: measured
    # trunk-to-axle 0.385 m instead of the stance's 0.278 m. The PD drive then
    # folds twelve joints by up to 1.6 rad while all four wheels are already on
    # the ground and free to roll, the contact patches squirt outward, and the
    # machine does the splits and lands on its back. It looks exactly like a
    # broken port. It is a missing write.
    for robot in robots:
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel)
    print("[bestiary] joints written to the default stance", flush=True)

    if args_cli.hold_pose:
        # Snapshot the placed pose ONCE, before physics has touched it, so the
        # pin below holds where the robot was put rather than where it drifted.
        held = [
            (
                robot.data.root_pos_w.clone(),
                robot.data.root_quat_w.clone(),
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            for robot in robots
        ]
        print("[bestiary] --hold-pose: robots pinned; this is a DISPLAY mode and "
              "says nothing about whether the machine can stand", flush=True)

    print("[bestiary] terrain built. Ctrl+C in this terminal to quit.", flush=True)

    dt = sim.get_physics_dt()
    zero_vel = torch.zeros((1, 6), device=sim.device)
    while simulation_app.is_running():
        # Hold the default stance. This is a viewer, not a controller -- the
        # robots are here to be looked at and to give the renderer a model, so
        # they are commanded to their nominal joint state and left to settle.
        #
        # Both targets are written, not just the position one. Hound's four
        # wheels are VELOCITY-driven at stiffness 0, so a position target on them
        # produces exactly nothing; what holds them still is a zero velocity
        # target against the drive's damping. Writing only positions would leave
        # the wheels entirely free.
        for robot in robots:
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.set_joint_velocity_target(robot.data.default_joint_vel)
            robot.write_data_to_sim()
        if args_cli.hold_pose:
            for robot, (pos, quat, jpos, jvel) in zip(robots, held):
                robot.write_root_pose_to_sim_index(
                    root_pose=torch.cat([pos, quat], dim=-1)
                )
                robot.write_root_velocity_to_sim_index(root_velocity=zero_vel)
                robot.write_joint_position_to_sim_index(position=jpos)
                robot.write_joint_velocity_to_sim_index(velocity=jvel)
        sim.step()
        for robot in robots:
            robot.update(dt)


if __name__ == "__main__":
    # The traceback is printed and flushed HERE rather than left to propagate.
    # `simulation_app.close()` can end the process without draining Python's
    # buffers, so a bare `finally: close()` turns any failure into a silent
    # exit 0 with no window and no message -- which is exactly how the missing
    # Newton render model presented the first time.
    import traceback

    try:
        main()
    except KeyboardInterrupt:
        print("[bestiary] interrupted", flush=True)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
        raise SystemExit(1)
    simulation_app.close()
