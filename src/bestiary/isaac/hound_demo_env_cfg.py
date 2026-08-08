"""Hound on the DEMO strip — the camera surface, not a training ground.

Same contract as `spyder_demo_env_cfg` (which carries the full argument for
why the training grid cannot be filmed): the forward-v5 Hound Play task with
its terrain replaced by the one continuous flat-to-hard strip, spawn pinned
mid-pad facing +x so an unsteered sprinter runs up the difficulty gradient.
Nothing here ever enters the ledger as ground a run stood on.

The spawn y is SCATTERED ±9 m for the same reason the Spyder demo's is: the
player simulates nine physics twins on a one-tile world (one env renders no
robot on this install), and nine machines at one pinned point spawn inside
each other.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.hound_forward_v5_env_cfg import HoundForwardV5EnvCfg_PLAY
from bestiary.isaac.spyder_demo_env_cfg import SPAWN_X_M, demo_terrain_cfg


@configclass
class HoundDemoEnvCfg_PLAY(HoundForwardV5EnvCfg_PLAY):
    """The forward-v5 Hound Play task with its terrain swapped for the strip."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain.terrain_generator = demo_terrain_cfg()
        self.scene.terrain.max_init_terrain_level = None
        if getattr(self, "curriculum", None) is not None:
            self.curriculum.terrain_levels = None

        self.events.reset_base.params["pose_range"] = {
            "x": (SPAWN_X_M, SPAWN_X_M),
            "y": (-9.0, 9.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
