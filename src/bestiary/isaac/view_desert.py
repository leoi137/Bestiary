"""Look at the Bestiary desert inside Isaac Lab.

This is the "is it actually our ground?" check, and it is deliberately a viewer
rather than a training script. A heightfield that loads without raising can
still be wrong in ways only a picture shows: transposed, mirrored, flattened by
a bad vertical scale, or tiled from the same patch 200 times. Numbers cannot
catch those; eyes can, in about three seconds.

Two terrains are offered so the comparison is direct:

    --mix desert   every tile is a patch of assets/terrain/desert_hfield.bin
    --mix blend    the desert alongside Isaac Lab's own slopes and rock fields

`blend` is the one that matters for training. Isaac Lab's shipped rough config
uses `noise_range=(0.02, 0.10)` — two to ten centimetres — while the desert at
difficulty 1.0 has metres of relief, so mixing them gives a curriculum that
spans "gentle bumps" to "real dune" instead of only one of the two.

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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mix",
    choices=["desert", "blend"],
    default="desert",
    help="desert: every tile from our heightfield. blend: ours plus Isaac Lab's built-ins.",
)
parser.add_argument(
    "--tile-m", type=float, default=8.0, help="Square sub-terrain size in metres (default 8.0)."
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
    default="height",
    help="Tile colouring. 'height' makes relief obvious and is the default here.",
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
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: E402
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg  # noqa: E402

from bestiary import paths  # noqa: E402
from bestiary.terrain.isaac_hf import (  # noqa: E402
    DESERT_NATIVE_CELL_M,
    HfBestiaryDesertTerrainCfg,
    load_desert_m,
)


def build_terrain_cfg() -> TerrainGeneratorCfg:
    """The sub-terrain mix named by ``--mix``.

    `horizontal_scale` is set to the desert's own cell size rather than Isaac
    Lab's default 0.1 m. At 0.078125 m the resample in `isaac_hf` is very nearly
    the identity, so what appears on screen is the committed samples and not a
    smoothed interpretation of them — which is the only way this script can
    answer the question it exists to answer.
    """
    desert = HfBestiaryDesertTerrainCfg(
        proportion=1.0 if args_cli.mix == "desert" else 0.5,
        hfield_path=str(paths.DESERT_HFIELD),
        border_width=0.25,
    )

    sub_terrains: dict = {"bestiary_desert": desert}
    if args_cli.mix == "blend":
        # Isaac Lab's own, kept at their shipped parameters so the contrast with
        # the desert is visible rather than tuned away.
        sub_terrains["isaac_slope"] = terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        )
        sub_terrains["isaac_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        )

    return TerrainGeneratorCfg(
        size=(args_cli.tile_m, args_cli.tile_m),
        border_width=20.0,
        num_rows=args_cli.rows,
        num_cols=args_cli.cols,
        horizontal_scale=DESERT_NATIVE_CELL_M,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=args_cli.curriculum,
        color_scheme=args_cli.color_scheme,
        sub_terrains=sub_terrains,
    )


def main() -> None:
    # Report what is about to be drawn, so a wrong asset is caught before the
    # window opens rather than squinted at afterwards.
    # flush=True throughout: Kit's teardown can end the process without draining
    # Python's stdout buffer, which silently swallowed these lines the first time.
    desert = load_desert_m(paths.DESERT_HFIELD)
    cells = int(round(args_cli.tile_m / DESERT_NATIVE_CELL_M))
    print(f"[bestiary] heightfield : {paths.DESERT_HFIELD}", flush=True)
    print(f"[bestiary] desert      : {desert.shape[0]}x{desert.shape[1]} cells, "
          f"{DESERT_NATIVE_CELL_M * 100:.4f} cm/cell, relief {desert.max():.3f} m", flush=True)
    print(f"[bestiary] tile        : {args_cli.tile_m:.2f} m = {cells}x{cells} native cells",
          flush=True)
    print(f"[bestiary] grid        : {args_cli.rows}x{args_cli.cols} tiles, mix={args_cli.mix}, "
          f"curriculum={args_cli.curriculum}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)

    light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)

    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=build_terrain_cfg(),
        num_envs=1,
        env_spacing=args_cli.tile_m,
        max_init_terrain_level=None,
        debug_vis=False,
    )
    if args_cli.color_scheme in ("height", "random"):
        # A visual material would paint over the colour scheme we asked for.
        terrain_cfg.visual_material = None
    TerrainImporter(terrain_cfg)

    # Park the camera above the middle of the grid, looking down the diagonal,
    # so the whole tiled area is in frame at whatever --rows/--cols was asked.
    span = max(args_cli.rows, args_cli.cols) * args_cli.tile_m
    sim.set_camera_view(eye=(span * 0.7, span * 0.7, span * 0.5), target=(0.0, 0.0, 0.0))

    sim.reset()
    print("[bestiary] terrain built. Ctrl+C in this terminal to quit.", flush=True)
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
