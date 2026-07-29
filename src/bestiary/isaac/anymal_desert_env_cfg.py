"""ANYmal-C on the Bestiary desert: env configs for training.

WHY ANYMAL AND NOT HOUND

Hound is the point, and Hound is not here yet -- its MJCF still has to become
USD and its 38-assertion check has to be re-established against a different
solver. Putting a *known* robot on our *new* terrain isolates one variable: if
ANYmal-C trains here the way it trains on Isaac Lab's own rough terrain, the
terrain bridge is sound and every later Hound problem is a Hound problem. Swap
the order and a failure is unattributable.

THE MEASUREMENT THIS FILE EXISTS TO ENABLE

`terrain/isaac_hf.py` serves tiles at the desert's native 0.078125 m sampling.
Isaac Lab's shipped rough config uses 0.1 m, so our tiles carry about 1.65x the
geometry -- roughly 10,600 vertices and 20,808 faces each, against their ~6,400
-- and the training grid is 200 tiles, not the 36 a viewer shows. Denser
collision meshes are not free, and nobody has measured what they cost here.

So two task ids are registered, differing in exactly one number:

    Bestiary-Desert-Anymal-C-v0         horizontal_scale = 0.078125  (native)
    Bestiary-Desert-Coarse-Anymal-C-v0  horizontal_scale = 0.1       (theirs)

Run both for the same iteration count and the difference is the price of
native resolution. The baseline to beat is 13,520 steps/s, measured on
Isaac-Velocity-Rough-Anymal-C-v0 at 1024 envs on this machine.

WHAT IS INHERITED AND WHAT IS REPLACED

Everything comes from `AnymalCRoughEnvCfg` -- rewards, observations, the height
scanner, terminations, the curriculum -- and only `scene.terrain.terrain_generator`
is replaced. That is deliberate: a reward tuned by NVIDIA for their terrain is a
control, not a liability. When Hound arrives the rewards become ours; until
then, changing terrain and rewards together would make the result unreadable.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.rough_env_cfg import (
    AnymalCRoughEnvCfg,
    AnymalCRoughEnvCfg_PLAY,
)

from bestiary import paths
from bestiary.terrain.isaac_hf import DESERT_NATIVE_CELL_M, HfBestiaryDesertTerrainCfg

#: Tiles per grid. Matches ROUGH_TERRAINS_CFG (10x20 = 200) so throughput here is
#: directly comparable to the 13,520 steps/s measured on their rough task.
NUM_ROWS = 10
NUM_COLS = 20

#: Sub-terrain footprint, in metres. Also matches theirs.
TILE_M = 8.0


def desert_terrain_cfg(horizontal_scale: float) -> TerrainGeneratorCfg:
    """A desert-and-builtin mix at the requested horizontal sampling.

    Proportions: half the tiles are real desert, half are Isaac Lab's own at
    their shipped parameters. The blend is the point -- their `random_rough`
    spans two to ten centimetres of noise while the desert reaches metres, so a
    policy that only ever saw one of the two has seen a narrow world. Keeping
    their parameters untouched also leaves them as a control.
    """
    return TerrainGeneratorCfg(
        size=(TILE_M, TILE_M),
        border_width=20.0,
        num_rows=NUM_ROWS,
        num_cols=NUM_COLS,
        horizontal_scale=horizontal_scale,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        # NOT "height": Isaac Lab's color_meshes_by_height passes a colormap name
        # the installed trimesh rejects, and TerrainImporter raises. See
        # bestiary.isaac.view_desert's --color-scheme help.
        color_scheme="none",
        sub_terrains={
            "bestiary_desert": HfBestiaryDesertTerrainCfg(
                proportion=0.5,
                hfield_path=str(paths.DESERT_HFIELD),
                border_width=0.25,
            ),
            "isaac_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.25, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            ),
            "isaac_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
        },
    )


@configclass
class AnymalCDesertEnvCfg(AnymalCRoughEnvCfg):
    """ANYmal-C on the desert at its native 7.8 cm sampling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = desert_terrain_cfg(DESERT_NATIVE_CELL_M)


@configclass
class AnymalCDesertCoarseEnvCfg(AnymalCRoughEnvCfg):
    """Same terrain, resampled to Isaac Lab's 0.1 m grid.

    The control arm for the density measurement. If this is materially faster
    than the native config, native resolution is what costs -- not the desert.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = desert_terrain_cfg(0.1)


@configclass
class AnymalCDesertEnvCfg_PLAY(AnymalCRoughEnvCfg_PLAY):
    """Watchable variant: few robots, no observation noise, no random shoving.

    Inherits the PLAY overrides (50 envs, corruption off, pushes removed) and
    then shrinks the tile grid, because 200 tiles of terrain to look at one
    robot is a waste of a minute of startup.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg = desert_terrain_cfg(DESERT_NATIVE_CELL_M)
        cfg.num_rows = 5
        cfg.num_cols = 5
        self.scene.terrain.terrain_generator = cfg
