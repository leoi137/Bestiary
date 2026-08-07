"""Diagnostic reward terms: forward speed, and deliberately nothing else.

WHY A REWARD THIS BARE EXISTS
-----------------------------
The classic MuJoCo locomotion benchmarks — Ant, HalfCheetah, Walker2d, the
2016-era Gym suite that SAC and every actor-critic since was validated on —
learn forward walking from a reward *dominated by forward velocity*, with no
gait shaping whatsoever: no tracking kernel, no contact-timing term, no
joint-acceleration penalty, no orientation term. This repository has its own
instance of that result: `research/learnings/001` reports a Spyder-12 SAC
policy walking the 5.05 m desert at 0.37 m/s. **Forward walking is a solved
problem under an unshaped reward, on this robot, in this repository.**

The Isaac Lab task trains under a different thing entirely: eleven inherited
reward terms (two tracking kernels, contact timing, four joint/base penalties,
two zero-weighted), five years of the legged_gym recipe kept as a control
(`spyder_gentle_env_cfg.py`). That inheritance is a *hypothesis*, and it is
entangled with every other hypothesis in the port — the MJCF→USD transform,
the PD drive constants, the terrain, the height scan, the PPO hyperparameters,
the observation layout.

So this term exists to split that entanglement in one run:

    if Isaac + PPO cannot reproduce forward walking under reward = v_x,
    the fault is in the STACK, not in the reward design.

A negative result here indicts the port. A positive result clears it and hands
the whole question back to the reward table, where it can be worked on with a
control that is known to walk. Neither conclusion is available from a run that
changes the reward *and* keeps eleven other terms alive.

**The operator ordered this variant explicitly (2026-08-06.)** It is a
diagnostic, not a proposal: nothing here is a claim that an unshaped reward is
the right reward for a command-following walker. It cannot be — it does not
read the command at all.

HONEST ABOUT THE BASELINE, BECAUSE THE RECORD IS CHECKABLE
-----------------------------------------------------------
`learnings/001`'s reward was NOT literally `v_x`. It was the Gym shape:
forward velocity **plus** a 1000-point alive bonus **minus** a control cost at
`ctrl_cost_weight = 0.1` — and 001 is precisely the learning that the control
cost inverted the incentive on rough ground (0.51 : 1 payoff-to-cost against
8.7 : 1 on the flat). This term is therefore **stricter than the baseline it
cites**: no alive bonus and no control cost, so neither of 001's two
confounders is present. If forward walking appears here, it appears with less
help than the 2016 recipe ever had; if it does not, the failure cannot be
blamed on a mis-weighted control cost, because there is none.

There is still an implicit liveness incentive and it is worth naming: v_x is
paid every step an episode survives, so falling early forfeits the remaining
income. Standing still scores exactly 0 — neither punished nor rewarded, which
is the sharpest possible version of the standing test `learnings/001` asks for.

IMPORT DISCIPLINE — THE SAME ONE `commands.py` CARRIES, FOR THE SAME REASON
---------------------------------------------------------------------------
This module is reached by hydra's env-cfg import, which happens **before the
simulation app exists**. Anything on that chain that drags `pxr` into the
process — directly, or transitively through a lazy loader — heap-corrupts Kit
about 1.5 s into every launch (`free(): invalid pointer`, exit 134; measured
locally and on the rented box, 2026-08-06, account in `commands_impl.py`).

Hence: `torch` and `isaaclab.managers.SceneEntityCfg` at runtime — both
already on the cfg chain, so neither adds a new edge — and everything heavier
behind `TYPE_CHECKING`, resolved as a string annotation under
`from __future__ import annotations`. A reward function is exactly the kind of
file where someone reaches for `isaaclab.assets` at runtime for a type hint
they never call; do not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

# TYPE_CHECKING, not a runtime import — see the module docstring's last
# section. Under `from __future__ import annotations` both names below are
# strings at runtime, so nothing needs the classes.
if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def forward_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base-frame forward speed, m/s, signed. One number per env, no kernel.

    Positive when the machine's own nose direction is where it is going, so
    faster forward is strictly more reward with no saturation point, and
    driving backwards is a genuine loss rather than a smaller gain. **Base
    frame, not world frame**, deliberately: a world-frame +x reward would pay a
    machine for facing the arena's x axis and would be un-learnable on a
    terrain curriculum that re-spawns envs at arbitrary origins.

    `.torch` is not optional and not decoration. In the pinned Isaac Lab
    (release/3.0.0-beta2, af1bab4), `data.root_lin_vel_b` is a **warp** array
    and `.torch` is the zero-copy torch view; upstream's own
    `isaaclab.envs.mdp.rewards.lin_vel_z_l2` reads
    `asset.data.root_lin_vel_b.torch[:, 2]` in exactly this install, which is
    the provenance for the line below. Indexing the warp array directly does
    not raise here — it returns something the reward manager will happily
    multiply — so this is a silent-wrongness edge, and the fix is to spell it
    the way upstream spells it. If a future Isaac Lab returns a plain tensor,
    upstream's rewards break in the same breath as this one, loudly.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b.torch[:, 0]
