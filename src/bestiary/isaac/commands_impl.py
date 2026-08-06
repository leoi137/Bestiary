"""Runtime half of the dead-zone velocity command. Import only under a running app.

WHY THIS FILE IS SEPARATE FROM `commands.py`, MEASURED
------------------------------------------------------
Hydra resolves a task's env-cfg entry point BEFORE the simulation app exists
(`isaaclab_tasks/utils/hydra.py:resolve_task_config` →
`load_cfg_from_registry` → `importlib.import_module(<env cfg module>)`), so
everything an env cfg module imports, transitively, is imported pre-app. A
runtime command TERM imports `isaaclab.markers.VisualizationMarkers`, which
imports `pxr` — and with no Kit running, that `pxr` comes from the pip
`usd-core` wheel. Booting Kit on top of a foreign USD already in the process
corrupts the heap: `free(): invalid pointer`, exit 134, reproduced identically
on the rented box AND locally (2026-08-06). It costs about 1.5 seconds to die
and nothing in the traceback names the cause.

This is exactly why upstream's `class_type` field is TYPED `type | str` and
DEFAULTED to a lazy string (`"{DIR}.velocity_command:UniformVelocityCommand"`)
— the cfg module never touches the runtime module; `configclass` wraps the
string in a `ResolvableString` resolved at term construction, which happens
after the app is up. The first draft of `commands.py` set `class_type` to the
class OBJECT, which quietly re-imported the runtime chain into every pre-app
cfg import. The oracle now asserts the cfg carries the string form, so the
eager import cannot come back.

The mechanics of the sampler itself are documented in `commands.py`'s module
docstring; this file is only the code that has to wait for the app.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from bestiary.isaac.commands import DeadZoneVelocityCommandCfg


class DeadZoneVelocityCommand(UniformVelocityCommand):
    """`UniformVelocityCommand` with a resampled v_x magnitude and snapped w_z.

    Only `_resample_command` changes, and only as a REMAP of what the parent
    already sampled: heading logic, standing-env bookkeeping, metrics and
    visualisers are inherited untouched, so this stays correct when upstream
    fixes theirs.
    """

    cfg: DeadZoneVelocityCommandCfg

    def __init__(self, cfg: DeadZoneVelocityCommandCfg, env) -> None:
        # Fail at construction, not mid-run: the remap below assumes symmetric
        # ranges (it reflects the parent's uniform sample through zero), and a
        # dead zone at or past the range edge would command nothing at all.
        for name, (lo, hi), dz in (
            ("lin_vel_x", cfg.ranges.lin_vel_x, cfg.min_lin_vel_x),
            ("ang_vel_z", cfg.ranges.ang_vel_z, cfg.min_ang_vel_z),
        ):
            if lo != -hi:
                raise ValueError(
                    f"DeadZoneVelocityCommand needs a symmetric {name} range, got "
                    f"({lo}, {hi}). Both remaps treat the range as a magnitude "
                    "and a sign, so an asymmetric range would silently bias the sign."
                )
            if not (0.0 <= dz < hi):
                raise ValueError(
                    f"min_{name} = {dz} must sit inside [0, {hi}) — past the range "
                    "edge the v_x resample collapses to the edge value and the w_z "
                    "snap zeroes every draw."
                )
        if cfg.heading_command:
            raise ValueError(
                "DeadZoneVelocityCommand with heading_command=True would let the "
                "yaw channel bypass the dead zone (heading mode recomputes w_z "
                "from heading error every step — the self-zeroing loop "
                "`decisions/0006` prices). Use rate commands, or the plain sampler."
            )
        super().__init__(cfg, env)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        r = self.vel_command_b

        # v_x: |u| ~ U(0, hi) -> dz + |u|(hi-dz)/hi ~ U(dz, hi); the sign is
        # torch.where, not torch.sign, so u == 0 cannot emit a zero command
        # outside a standing env.
        dz, hi = self.cfg.min_lin_vel_x, self.cfg.ranges.lin_vel_x[1]
        u = r[env_ids, 0]
        sign = torch.where(u >= 0.0, 1.0, -1.0)
        r[env_ids, 0] = sign * (dz + u.abs() * (hi - dz) / hi)

        # w_z: snap-to-zero (legged_gym's device). Small draws become exactly
        # zero — the straight drivers the terrain curriculum needs — and the
        # survivors keep their sampled value, so |w_z| is never in (0, dz).
        dz = self.cfg.min_ang_vel_z
        w = r[env_ids, 2]
        r[env_ids, 2] = torch.where(w.abs() < dz, torch.zeros_like(w), w)
