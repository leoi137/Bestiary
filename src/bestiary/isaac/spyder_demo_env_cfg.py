"""Spyder-12 on the DEMO strip: one continuous surface, flat at -x, hard at +x.

WHAT THIS IS, AND WHAT IT IS NOT

Not a training task. `Bestiary-Demo-Spyder-Play-v0` exists so a policy that was
already trained on `spyder_gentle_env_cfg`'s tiled curriculum can be WATCHED
crossing a single unbroken surface whose difficulty is a smooth function of
position. Nothing here should ever appear in the ledger as the ground a run
stood on, and no reward, observation or command is changed from the task the
policy actually trained under — the whole point is that the machine cannot tell
this env from its own, apart from where its feet land.

The observation is therefore INHERITED, not redeclared. It is 235 wide and it
is a one-way door (CLAUDE.md); a demo config that quietly changed it would
produce a policy that loads, runs, and is being fed a permuted world.

WHY THE TRAINING TERRAIN COULD NOT JUST BE FILMED

`spyder_gentle_env_cfg.gentle_terrain_cfg` is a 10x20 grid of 8 m tiles, each
cut from a different patch of the asset and ranked by roughness, because the
grid IS the curriculum. Three consequences, all fine for training and all fatal
for a demo: neighbouring tiles are unrelated crops so every boundary is a
vertical step; `border_width = 20.0` wraps the grid in a flat plane, which is
what the machine ends up filmed on once it leaves the grid; and
`slope_threshold = 0.75` asks the mesh converter to INSERT vertical faces at
steep cells. Measured 2026-08-07: a 5 m/s policy cleared the 3x3 play grid in
about 7 s and spent the rest of the take on the flat border.

So the demo replaces the generator with a single tile the size of the whole
strip, zero border, and no slope correction. `terrain/demo_hf.py` derives that
tile from the committed asset's own bytes under a smooth envelope, so it is one
continuous surface by construction rather than by inspection.

WHERE THE MACHINE STARTS, AND WHY IT IS PINNED

Upstream's `reset_base` scatters yaw over the full circle, which is correct for
training and useless for a shot that is supposed to travel up a ramp. Here the
spawn is pinned: mid-pad at the flat end, facing +x, at rest. Everything the
policy does after the first step is its own.

A caveat to state rather than discover: the forward-only diagnostic
(`spyder_forward_env_cfg`) does not read commands at all, so it holds no
heading — pinning its spawn aims it up the ramp but nothing keeps it there. A
command-following checkpoint driven at (v_x, 0, 0) is the one that will
actually run the strip end to end.
"""

from __future__ import annotations

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from bestiary import paths
from bestiary.isaac.spyder_gentle_env_cfg import SpyderGentleEnvCfg_PLAY
from bestiary.terrain.demo_hf import HfDemoRampTerrainCfg
from bestiary.terrain.gentle import Z_SPAN_M as GENTLE_Z_SPAN_M

#: Strip length along +x, metres — the direction difficulty increases and the
#: direction the machine runs. 78 m fits inside the 80 m asset without tiling
#: it, and at the forward policy's measured 4-5.4 m/s it is 15-20 s of running.
DEMO_LENGTH_M = 78.0

#: Strip width along y, metres. As wide as the strip is long, because the
#: forward diagnostic does not hold a heading — measured 2026-08-07 on this
#: very strip, it was 10 m off-axis 4 s after a spawn pinned to +x — and
#: running off the side of a borderless strip is a fall, not a shot. 78 m is
#: also the widest square the 80 m asset supplies without tiling it.
DEMO_WIDTH_M = 78.0

#: Metres per cell of the demo mesh. 0.05 m — half the training config's 0.1,
#: because this surface is looked at rather than learned from and the asset's
#: 7.8 cm native sampling is worth resolving.
DEMO_CELL_M = 0.05

#: Spawn x, metres from the strip centre. The pad spans x in [-39.0, -29.6]
#: (12% of 78 m); -34.0 is its middle, so the machine gets ~4.4 m of flat
#: ground behind it and ~4.7 m ahead before the terrain begins.
SPAWN_X_M = -34.0


def demo_terrain_cfg(cell_m: float = DEMO_CELL_M) -> TerrainGeneratorCfg:
    """ONE tile, no border, no slope correction — see the module docstring.

    Every field that differs from `gentle_terrain_cfg` differs on purpose and
    is the difference between a curriculum and a camera subject.
    """
    return TerrainGeneratorCfg(
        # The tile IS the strip. num_rows/num_cols of 1 is what removes the
        # seams: there are no neighbours to step to.
        size=(DEMO_LENGTH_M, DEMO_WIDTH_M),
        num_rows=1,
        num_cols=1,
        border_width=0.0,
        horizontal_scale=cell_m,
        # 1 mm, not the training config's 5 mm. Heights are stored int16 in
        # units of this, so 5 mm quantises a shallow dune flank into visible
        # contour terraces — harmless to a foot, obvious on camera (measured
        # 2026-08-07: banding across the whole hard end of the first take).
        # int16 at 1 mm still spans +-32.7 m against this asset's 2.06 m.
        vertical_scale=0.001,
        # None, NOT the training config's 0.75. The threshold makes the mesh
        # converter insert vertical faces at steep cells — a cliff-maker, and
        # cliffs are the thing this terrain exists not to have.
        slope_threshold=None,
        use_cache=False,
        curriculum=False,
        # color_meshes_by_height crashes the installed trimesh; same as gentle.
        color_scheme="none",
        sub_terrains={
            "demo_ramp": HfDemoRampTerrainCfg(
                proportion=1.0,
                hfield_path=str(paths.GENTLE_HFIELD),
                z_span_m=GENTLE_Z_SPAN_M,
                border_width=0.0,
                # 1.0 = the committed asset unmodified at the far end. The
                # right end is therefore exactly the hardest ground the policy
                # trained on, not something invented to look dramatic.
                max_gain=1.0,
            ),
        },
    )


@configclass
class SpyderDemoEnvCfg_PLAY(SpyderGentleEnvCfg_PLAY):
    """The gentle PLAY task with its terrain replaced by the demo strip."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.terrain_generator = demo_terrain_cfg()
        # One tile means there is no level to initialise to, and the terrain
        # curriculum has nothing to promote between. Both would raise or
        # silently index a 1x1 grid.
        self.scene.terrain.max_init_terrain_level = None
        if getattr(self, "curriculum", None) is not None:
            self.curriculum.terrain_levels = None

        # Pinned spawn: mid-pad, facing +x, at rest. Offsets are relative to
        # the env origin, which for a single tile is the strip's centre.
        self.events.reset_base.params["pose_range"] = {
            "x": (SPAWN_X_M, SPAWN_X_M),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
