"""Spyder-12 on the gentle terrain: the env config, the commands, the reward.

    Bestiary-Gentle-Spyder-v0        the training-shaped config
    Bestiary-Gentle-Spyder-Play-v0   few robots, no noise, no shoving

THE GOAL THIS CONFIG ENCODES
----------------------------
A command-following walker: W/S/A/D later means (±v_x, ±w_z) now, and space —
stand still — is an explicit commanded mode, not the absence of one. The three
design choices below are all downstream of one measurement:
`isaac_hound_arm1_s2`, the seed that parked. It beat the do-nothing control in
13 of 13 eval cells while covering less ground than the control
(`research/measurements/isaac_hound_arm1_s2.json`), because the command
distribution and the kernel width together made standing worth 44% of tracking
income and heading mode made the yaw term free. Every choice here closes one
of those doors, and each one names the mechanism it closes.

1. DEAD-ZONED COMMANDS (`isaac/commands.py`). Driving envs are commanded
   |v_x| ∈ [0.25, 0.6] m/s, and w_z is either EXACTLY 0 (small draws snapped —
   the straight drivers the terrain curriculum promotes on) or a genuine turn,
   |w_z| ∈ [0.2, 0.8] rad/s. The ambiguous near-zero middle is never asked
   for on either channel. Standing is commanded separately and exactly:
   `rel_standing_envs = 0.1` zeroes the full command for 10% of resamples.
   The policy therefore sees unambiguous regimes, which is precisely the
   W/S/A/D-or-nothing interface the operator asked for (2026-08-05).
   Turn commands break upstream's terrain-curriculum demote bar (straight-line
   kinematics applied to an arc); `isaac/curriculums.py` carries the
   corrected bar and the derivation.

2. HEADING MODE OFF. Upstream defaults `heading_command=True`, which
   recomputes w_z from heading error every step — so a machine that turns to
   the target and stops has zeroed its own yaw command and collects the yaw
   income for standing: the point-and-park loop. `research/decisions/0006` §1
   prices it on the Hound's committed ±0.3 lateral range at 64.95%–80.19% of
   a competent driver's net (its 92.42% headline needs four extra assumptions
   0006 itself forbids quoting bare). With rate commands the yaw target is
   exogenous: parked and driving score what they earn. `commands.py` refuses
   heading mode at construction.

3. KERNEL WIDTHS RESCALED WITH THE COMMAND RANGE — same rule, not new tuning.
   Upstream discriminates with std/v_max = 0.5/1.0. Spyder's range is
   ±0.6 m/s, so std_lin = 0.3; yaw ±0.8 rad/s keeps upstream's own 0.8 there,
   so std_ang = 0.4 by the same ratio. Inheriting std = 0.5 against a ±0.6
   range would make standing at the TOP command worth exp(-(0.6/0.5)^2) = 0.24
   of perfect — at upstream's own operating point that number is 0.018, and
   preserving the ratio is what preserves the recipe. The oracle computes the
   standing share all three choices jointly buy — 27.2% of drive-cell income,
   against the Hound's 62.7% — and goes red past 30%.

THE SPEED CEILING IS A CLAIM, AND HERE IS ITS ARITHMETIC
--------------------------------------------------------
0.6 m/s commanded max against one measurement and one estimate: this machine
walked 0.37 m/s on the 5.05 m desert (SAC, torque control —
`research/learnings/001`), and a stride-geometry ceiling of roughly
stride 0.3 m x 2 Hz ≈ 0.6 m/s. So the top command is reachable-in-principle,
unproven-in-fact, on ground 5x gentler than the 0.37 was measured on — unlike
the Hound's inherited 1.0 m/s against a 0.907 m/s drive maximum, it is not
geometrically impossible, but it is the weakest number in this file. Re-derive
it from the seeds' measured v_x span after run 1; if no seed exceeds ~0.45,
the top cell was charging the unremovable and the range narrows.

WHAT IS INHERITED UNTOUCHED, AND WHY
------------------------------------
The reward table is `RewardsCfg` with only BODY-NAME retargets: air time onto
the tibias, undesired contacts onto the femurs, termination onto the torso.
No term is deleted and no weight moves — a spider has feet, so the
contact-timing terms the Hound had to delete (`feet_air_time` pays a wheel to
hop) mean on this body exactly what ANYmal's recipe intends: air time buys a
gait, femur contact is a stumble, torso contact ends the episode. Five years
of the recipe (legged_gym 2021 -> Isaac Lab 3.0, byte-identical scales — the
2026-07-29 reward research) is inherited as a control, per decision 0004's
Part A logic: keep it knowingly, stop calling it tuned.

`dof_torques_l2` at -1e-5 deserves its one sentence: all twelve joints at the
full 40 N·m simultaneously would price 12 x 40^2 x 1e-5 x 0.02 = 0.00384/step,
12.8% of income — but that is the ceiling-of-ceilings; at the ~1 N·m rms a
1.08 kg machine actually walks with (gravity torque at stance is 0.41 N·m on
the worst joint), it is 0.008% of income. Awake at the extremes, silent in
the mean. The oracle prints the whole table.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

from bestiary import paths
from bestiary.isaac.commands import DeadZoneVelocityCommandCfg
from bestiary.isaac.curriculums import terrain_levels_vel_arc
from bestiary.isaac.hound_desert_env_cfg import retarget
from bestiary.isaac.spyder_cfg import SPYDER12_CFG, TORSO_BODY, TORSO_PRIM_SUBPATH
from bestiary.terrain.gentle import Z_SPAN_M as GENTLE_Z_SPAN_M
from bestiary.terrain.isaac_hf import DESERT_NATIVE_CELL_M, HfBestiaryDesertTerrainCfg

#: Terrain sampling for training, metres per cell. 0.1 m — the same number the
#: Hound task declares and for the same reasons: it is Isaac Lab's own shipped
#: resolution (so throughput comparisons carry), and the native 0.078125 m
#: grid segfaults PhysX above ~64 envs at 200 tiles (measured, STATE 2026-07-29).
TRAIN_CELL_M = 0.1

#: Native sampling for the Play/viewer config — few tiles, real samples.
VIEW_CELL_M = DESERT_NATIVE_CELL_M

#: Command envelope. The v_x floor keeps every driving command distinguishable
#: from standing (0.25, not 0.2: with the yaw snap putting 25% of drive envs
#: at w_z = 0 — where a parked machine scores the full yaw kernel — 0.2 put
#: the joint standing share at 30.5%, over the oracle's 30% flag; 0.25 lands
#: it at 27.2%, computed in `check_the_money`). The w_z floor is a SNAP
#: threshold: below it the command becomes exactly 0. Ceilings are the
#: docstring's claim.
VX_MAX_MS = 0.6
VX_MIN_MS = 0.25
WZ_MAX_RADS = 0.8
WZ_MIN_RADS = 0.2

#: Kernel widths, derived by preserving upstream's std/range ratio of 0.5.
STD_LIN = 0.5 * VX_MAX_MS   # 0.3
STD_ANG = 0.5 * WZ_MAX_RADS  # 0.4

#: Fraction of resamples commanded to stand (full zero command). Upstream
#: default is 0.02 — enough to see standing, not enough to learn it. 0.1
#: makes stand a first-class mode: the operator's "space or nothing" key.
REL_STANDING = 0.1


def gentle_terrain_cfg(horizontal_scale: float) -> TerrainGeneratorCfg:
    """The gentle-and-builtin mix at the requested horizontal sampling.

    Same 50/25/25 shape as `anymal_desert_env_cfg.desert_terrain_cfg`, with the
    gentle asset in the desert's slot and the two Isaac Lab sub-terrains kept
    at their shipped parameters as controls. What "gentle" bounds here is
    FORMS, not cells: the pyramid tiles cap at 21.8 degrees of sustained
    slope (inside the gentle asset's P99 of 26.5), while `isaac_rough`'s
    2-10 cm noise steps can present cell-to-cell gradients well past both —
    single-cell ledges, the same texture legged robots train on upstream.
    Nothing in the mix re-introduces MOUNTAINS — metres of sustained climb —
    which is the thing the gentle asset exists to remove.

    Difficulty is honoured end to end: the bestiary tile ranks real patches by
    elevation std and indexes the ranking by difficulty (`terrain/isaac_hf.py`),
    so `terrain_levels_vel` walks from measured-flattest to measured-roughest
    ground — the check STATE's handoff asked for, done by reading the code.
    """
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=horizontal_scale,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        # Not "height": color_meshes_by_height crashes the installed trimesh.
        color_scheme="none",
        sub_terrains={
            "bestiary_gentle": HfBestiaryDesertTerrainCfg(
                proportion=0.5,
                hfield_path=str(paths.GENTLE_HFIELD),
                z_span_m=GENTLE_Z_SPAN_M,
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
class SpyderGentleEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Spyder-12 on the gentle mix, commands dead-zoned, heading off."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = SPYDER12_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_generator = gentle_terrain_cfg(TRAIN_CELL_M)

        # A prim path, not a body name: the MJCF importer keeps the kinematic
        # tree, so the torso is not a direct child of the robot prim.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/{TORSO_PRIM_SUBPATH}"
            # The FOOTPRINT must move with the robot, not just the mount.
            # Upstream's grid is 1.6 x 1.0 m at 0.1 m — sized for ANYmal,
            # whose feet stand inside ~0.6 x 0.4 m. Spyder's X-stance puts
            # its foot centres at (±0.76, ±0.76) in the torso frame, OUTSIDE
            # that grid entirely: the policy would place feet on ground it
            # cannot see. Same 17 x 11 = 187 rays — the obs width is a
            # one-way door and does not move — at 0.16 m spacing instead of
            # 0.1: a 2.56 x 1.6 m footprint whose y edge (±0.8) covers the
            # foot centres and whose x reach sees 1.28 m of terrain ahead at
            # commanded speeds up to 0.6 m/s. Coarser per-ray, wider per
            # machine; 0.16 m still samples the rubble band's 1.5 m floor.
            self.scene.height_scanner.pattern_cfg.resolution = 0.16
            self.scene.height_scanner.pattern_cfg.size = (2.56, 1.6)

        # -- Terrain curriculum: the arc-corrected demote bar. Upstream's
        # `terrain_levels_vel` compares displacement against |cmd| * T / 2 —
        # straight-line kinematics — so a PERFECT tracker of (0.6, 0.8) is
        # demoted every episode (1.48 m reachable vs a 6 m bar) while a
        # yaw-blind straight driver is promoted: a curriculum that teaches
        # learning 015's failure on purpose. `curriculums.py` derives the
        # constant-twist bar; for w = 0 it reduces exactly to upstream's.
        self.curriculum.terrain_levels.func = terrain_levels_vel_arc

        # -- Commands: the module docstring's items 1 and 2 -------------------
        up = self.commands.base_velocity
        self.commands.base_velocity = DeadZoneVelocityCommandCfg(
            asset_name=up.asset_name,
            resampling_time_range=up.resampling_time_range,
            rel_standing_envs=REL_STANDING,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=up.debug_vis,
            ranges=DeadZoneVelocityCommandCfg.Ranges(
                lin_vel_x=(-VX_MAX_MS, VX_MAX_MS),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-WZ_MAX_RADS, WZ_MAX_RADS),
                heading=None,
            ),
            min_lin_vel_x=VX_MIN_MS,
            min_ang_vel_z=WZ_MIN_RADS,
        )

        # lin_vel_y is (0, 0) but the command SLOT stays: a spider can sidestep
        # (legs, not wheels — the Hound's geometric argument does not apply),
        # and the slot in the observation is what makes adding lateral commands
        # later a config change instead of a one-way obs-width door. With
        # c_y == 0 the y channel of the 2-D tracking error prices uncontrolled
        # sideways drift, which on 26-degree slopes is the honest thing to do.

        # -- Kernel widths: the module docstring's item 3 ---------------------
        self.rewards.track_lin_vel_xy_exp.params["std"] = STD_LIN
        self.rewards.track_ang_vel_z_exp.params["std"] = STD_ANG

        # -- Body-name retargets. Upstream names ANYmal's links. --------------
        # Every term named here would otherwise raise at construction
        # ("Not all regular expressions are matched!") — except the quiet ones,
        # which is why retarget() raises on a term with nothing to retarget.
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="tibia_.*"
        )
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="femur_.*"
        )
        retarget(self.events.add_base_mass, "asset_cfg", "robot", TORSO_BODY)
        retarget(self.events.base_external_force_torque, "asset_cfg", "robot", TORSO_BODY)
        retarget(self.events.base_com, "asset_cfg", "robot", TORSO_BODY)
        retarget(self.terminations.base_contact, "sensor_cfg", "contact_forces", TORSO_BODY)

        # -- Reset scatter, restored. Upstream's `reset_joints_by_scale`
        # MULTIPLIES the default pose by U(0.5, 1.5) — on ANYmal (nonzero
        # defaults) that scatters the reset; on Spyder, whose stance is all
        # zeros, 0 x anything = 0 and every episode would start at the
        # identical joint state, silently. The offset form expresses the
        # recipe's intent — scattered starts — on a zero-stance machine.
        # ±0.1 rad is ~10-14% of each joint's range and stays well inside the
        # softest limit (knee ±0.87 x 0.9).
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        }

        # On femur contact at -1.0: the Hound deleted this term because ONE
        # thigh in sand scored at 106% of achievable income on terrain where
        # thigh contact is normal traversal. Spyder's femurs run UPWARD from
        # the coxa at stance (+0.25 m over 0.31 m of reach) — they touch
        # ground when the machine has collapsed, not when it is walking, so
        # here the term prices failure, which is its upstream meaning. The
        # oracle prints the arithmetic; if training shows healthy gaits paying
        # it (femur brushes on the steepest tiles), delete it THEN, citing the
        # measurement — not now, citing the Hound's.


@configclass
class SpyderGentleEnvCfg_PLAY(SpyderGentleEnvCfg):
    """What the viewer and a human should look at: few robots, nothing random.

    Mirrors the Hound Play config's overrides rather than inheriting ANYmal's,
    because this config descends from the generic locomotion cfg.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # Spawn across the grid instead of on the easiest terrain level.
        self.scene.terrain.max_init_terrain_level = None

        cfg = gentle_terrain_cfg(VIEW_CELL_M)
        cfg.num_rows = 3
        cfg.num_cols = 3
        cfg.curriculum = False
        self.scene.terrain.terrain_generator = cfg

        # No observation noise and no random shoving: a machine being pushed
        # over by an invisible force is not a useful thing to watch.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
