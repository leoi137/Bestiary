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

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

# NOTHING RUNTIME IS IMPORTED HERE, and that is the file's one invariant.
# Hydra imports env-cfg modules BEFORE the simulation app exists; a runtime
# command term imports VisualizationMarkers -> pxr from the pip usd-core, and
# booting Kit over a foreign USD is a measured heap corruption (free():
# invalid pointer, exit 134 — locally and on the rented box, 2026-08-06).
# The runtime class lives in `commands_impl.py`, reached only through the
# lazy class_type string below; its docstring carries the full account.


@configclass
class DeadZoneVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for `commands_impl.DeadZoneVelocityCommand`."""

    #: A LAZY STRING, deliberately — upstream's own pattern. `configclass`
    #: wraps it in a `ResolvableString` resolved at term construction, after
    #: the app is up. Assigning the class object here instead re-creates the
    #: pre-app pxr import and the 1.5-second heap-corruption crash; the
    #: oracle asserts the string form survives.
    class_type: type | str = "bestiary.isaac.commands_impl:DeadZoneVelocityCommand"

    min_lin_vel_x: float = 0.0
    """Smallest |v_x| a driving env may be commanded, m/s — the magnitude is
    RESAMPLED onto [min, max]. Zero disables the dead zone and reproduces the
    parent sampler exactly."""

    min_ang_vel_z: float = 0.0
    """Yaw-rate snap threshold, rad/s: sampled |w_z| below this becomes
    exactly 0 (a straight-drive command), at or above it survives unchanged.
    Zero disables the snap."""
