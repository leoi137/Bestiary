"""A velocity command sampler with a dead zone: drive means drive.

WHY THIS EXISTS — THE HOUND'S PARKED SEED, PRICED
-------------------------------------------------
`UniformVelocityCommand` samples v_x uniformly over a symmetric range, so a
large share of commands sit near zero, where standing still is nearly the
right answer. Under the tracking kernel exp(-(e/sigma)^2) a motionless machine
then collects a substantial fraction of the tracking income in expectation —
44.1% of the linear term at the Hound's inherited U(-1, 1) with sigma = 0.5,
computed in `check_hound.py`'s budget check and measured in the wild by
`isaac_hound_arm1_s2`: a seed that beat the do-nothing control in 13 of 13
eval cells while covering LESS ground than the control on fwd_fast
(`research/measurements/isaac_hound_arm1_s2.json`). The reward could not tell
parking from driving, so one seed in three parked.

The repair is a distribution, not a new reward term:

    v_x ~ U(-max, max)  ->  |v_x| ~ U(min_lin_vel_x, max), sign preserved
    w_z ~ U(-max, max)  ->  snapped to 0 where |w_z| < min_ang_vel_z

so every driving env is commanded something standing cannot fake, and STANDING
IS COMMANDED EXPLICITLY instead of arriving as the small tail of a uniform:
`rel_standing_envs` (upstream machinery, untouched) zeroes the full command
for that fraction of envs AFTER this sampler runs.

THE TWO CHANNELS GET DIFFERENT DEAD ZONES, and the difference is load-bearing:

    v_x  — magnitude RESAMPLED onto [min, max]. A driving env must drive;
           near-zero forward commands are what standing fakes.
    w_z  — sampled U(-max, max), then SNAPPED TO ZERO below min. Straight
           driving has to exist: `terrain_levels_vel`'s promote bar is
           displacement from spawn (> 4 m), and a machine whose every command
           carries |w_z| >= 0.2 rad/s drives arcs of radius v/w <= 3 m and
           can rarely displace 4 m — a curriculum in which perfect tracking
           never promotes. Snapping small yaw draws to exactly zero makes
           ~25% of driving envs straight drivers (they carry the curriculum)
           and the rest genuine turns, with the ambiguous 0 < |w_z| < min
           band never commanded. It is also the operator's interface: A/D
           released IS w_z = 0.

PRECEDENT, SO THIS IS A RE-SCOPING RATHER THAN AN INVENTION
-----------------------------------------------------------
legged_gym (Rudin et al. 2021) snaps any sampled command with norm under
0.2 m/s to exactly zero in `_resample_commands` — the lineage every config we
inherit descends from. The yaw channel here IS that mechanism. The v_x channel
re-bookkeeps it: snap-to-zero on the drive axis would couple the standing
fraction to the range geometry (U(-0.6, 0.6) snapped at 0.25 stands 42% of
the time, silently), while magnitude resampling keeps the standing fraction
where the config declares it, `rel_standing_envs`.

TERRAIN-CURRICULUM INTERACTION, STATED SO IT IS NOT DISCOVERED
--------------------------------------------------------------
Even with straight drivers restored, upstream `terrain_levels_vel` DEMOTES a
perfect turner: its bar is `|cmd_xy| * T / 2` — commanded distance as if the
path were straight — while a constant-twist tracker's displacement is bounded
by the arc diameter 2*v/w (1.48 m at v = 0.6, w = 0.8, against that 6 m bar).
`bestiary.isaac.curriculums.terrain_levels_vel_arc` replaces the bar with the
command's own reachable displacement; the kinematics live there.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils.configclass import configclass


class DeadZoneVelocityCommand(UniformVelocityCommand):
    """`UniformVelocityCommand` with a resampled v_x magnitude and snapped w_z.

    Only `_resample_command` changes, and only as a REMAP of what the parent
    already sampled: heading logic, standing-env bookkeeping, metrics and
    visualisers are inherited untouched, so this stays correct when upstream
    fixes theirs.
    """

    cfg: "DeadZoneVelocityCommandCfg"

    def __init__(self, cfg: "DeadZoneVelocityCommandCfg", env) -> None:
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
                "from heading error every step — the self-zeroing loop learning "
                "015 documents). Use rate commands, or the plain sampler."
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


@configclass
class DeadZoneVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for :class:`DeadZoneVelocityCommand`."""

    class_type: type = DeadZoneVelocityCommand

    min_lin_vel_x: float = 0.0
    """Smallest |v_x| a driving env may be commanded, m/s — the magnitude is
    RESAMPLED onto [min, max]. Zero disables the dead zone and reproduces the
    parent sampler exactly."""

    min_ang_vel_z: float = 0.0
    """Yaw-rate snap threshold, rad/s: sampled |w_z| below this becomes
    exactly 0 (a straight-drive command), at or above it survives unchanged.
    Zero disables the snap."""
