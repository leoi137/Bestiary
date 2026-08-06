"""Terrain curriculum for command sets that include turning in place… almost.

THE DEFECT THIS REPAIRS, WITH ITS NUMBERS
-----------------------------------------
Upstream `terrain_levels_vel` (isaaclab_tasks .../velocity/mdp/curriculums.py)
promotes an env whose robot ends the episode more than `tile/2` = 4 m from its
spawn, and demotes one that displaced less than HALF THE COMMANDED DISTANCE,
computed as `|cmd_xy| * T * 0.5` — i.e. as if the commanded path were a
straight line.

For a straight command that bar is right. For a command (v, w) with a yaw
rate, the machine that tracks PERFECTLY drives a circle of radius v/w: its
displacement from spawn is bounded by the diameter forever. At this task's
command envelope, v = 0.6 m/s with w = 0.8 rad/s gives a 1.48 m maximum
displacement against a 6.0 m demote bar — flawless tracking, demoted to
easier terrain every single episode, while a policy that IGNORED the yaw
command and drove straight would be promoted. The curriculum would actively
teach yaw-blindness, which is learning 015's failure taught on purpose.

THE BAR, DERIVED RATHER THAN INVENTED
-------------------------------------
Keep upstream's intent verbatim — "demote if it covered less than half of
what its command would have covered" — and compute what the command would
have covered with the constant-twist kinematics every unicycle textbook
carries. Integrating (x', y') = (v cos wt, v sin wt) from the spawn:

    displacement(t) = 2 (v/w) |sin(w t / 2)|        (w != 0)
                    = v t                            (w -> 0 limit)

The demote bar is half of that, evaluated at the episode length with the
env's own (v, w). For w = 0 this reduces EXACTLY to upstream's bar, so
straight-drive and standing envs behave identically to stock Isaac Lab; for
turners it is the same sentence with the right geometry. The promote bar is
untouched — a tight turner simply cannot promote (bounded displacement),
which is neutral, not punitive: straight-drive envs carry promotion, and
`commands.py`'s yaw snap guarantees ~25% of driving envs are straight.

CAVEAT, STATED: commands resample mid-episode (every 10 s against a 20 s
episode), and like upstream this reads only the CURRENT command. The bar is
exact for the last command segment and approximate across a resample. That
approximation is upstream's too; it is not made worse here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

# TYPE_CHECKING, not a runtime import, and it is load-bearing: this module is
# reached by hydra's PRE-APP env-cfg import, and resolving TerrainImporter
# through isaaclab.terrains' lazy loader drags terrain_importer -> pxr (the
# pip usd-core) into the process before Kit boots — the measured free():
# invalid pointer crash commands_impl.py documents. The annotation below is a
# string under `from __future__ import annotations`, so nothing needs the
# class at runtime.
if TYPE_CHECKING:
    from isaaclab.terrains import TerrainImporter

#: Below this |w| (rad/s) the arc formula is numerically the straight line;
#: 2(v/w)sin(wT/2) at w = 1e-3, T = 20 differs from vT by one part in 6e5.
_STRAIGHT_W_RADS = 1e-3


def arc_displacement_m(speed_ms, yaw_rate_rads, t_s: float):
    """|displacement| after t seconds of perfectly tracked (v, w). Tensors in,
    tensor out; also correct on floats wrapped in tensors. Pure, so the oracle
    can pin its values without a simulator."""
    v = torch.as_tensor(speed_ms, dtype=torch.float64)
    w = torch.as_tensor(yaw_rate_rads, dtype=torch.float64)
    straight = v * t_s
    # Guard the division on the straight branch too: torch.where evaluates
    # both arms, and v/0 is inf even where the result is discarded.
    w_safe = torch.where(w.abs() < _STRAIGHT_W_RADS, torch.ones_like(w), w)
    arc = 2.0 * (v / w_safe) * torch.sin(w_safe * t_s / 2.0).abs()
    return torch.where(w.abs() < _STRAIGHT_W_RADS, straight, arc)


def terrain_levels_vel_arc(
    env, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Upstream `terrain_levels_vel` with the demote bar's kinematics corrected.

    Line-for-line the upstream function (same promote bar, same
    `update_env_origins`, same return) except the one comparison the module
    docstring derives. Kept structurally identical on purpose, so a diff
    against upstream shows exactly one changed expression.
    """
    asset = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.linalg.norm(
        asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    reachable = arc_displacement_m(
        torch.linalg.norm(command[env_ids, :2], dim=1),
        command[env_ids, 2],
        env.max_episode_length_s,
    ).to(distance.dtype)
    move_down = distance < reachable * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
