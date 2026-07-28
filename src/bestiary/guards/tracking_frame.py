"""The tracking reward measures velocity in the HEADING frame, not the world frame.

`docs/theory/command-tracking-reward.md` Section 1 calls this load-bearing and
Section 7 ranks it failure mode 6. The failure is quiet in the worst way: a
world-frame implementation trains, logs, checkpoints and plots normally, and
merely makes the objective unsatisfiable whenever the yaw command is nonzero.
Under a turn, a correctly driving body's world-frame velocity rotates
continuously, so tracking a fixed world-frame command caps Phi(u_v) near 0.5 on
every turning segment forever. The signature in the logs -- a gap between
turning and straight slices that no amount of training closes -- is only
legible to someone who already suspects the bug.

The note's suggested detection is to watch a conditional mean during training.
That costs a run to find out. This guard costs 0.3 s and runs before the run
starts, which is the difference between a lesson in prose and a lesson that
cannot ship.

WHAT IT ASSERTS

1. A body moving along its OWN +x axis, at any yaw, reads as purely forward.
   This is the assertion that fails loudly on a world-frame regression: at
   yaw = 40 degrees a world-frame reading returns (0.383, 0.321) where the
   heading frame returns (0.5, 0.0).
2. Lateral motion is not silently discarded -- the planar NORM is what enters
   u_v, so a sideways drift under a forward command must be penalized. A
   would-be "simplification" to `v_forward - vx_cmd` passes assertion 1 and
   fails this one.
3. yaw_rate reads the body-frame z component of the free joint's angular
   velocity, which is what Section 1 specifies.
4. The kernel is Cauchy, evaluated against values computed here rather than
   copied from the note -- the record's Gaussian/Cauchy inconsistency is
   exactly the kind of drift a guard should not inherit.
5. The freeride bounds the tolerances were DERIVED from still hold. sigma_v
   and sigma_w are not free parameters; they were sized by two inequalities
   (Section 2), and an innocent-looking retune that breaks one reopens the
   standing exploit the whole env exists to close. This is the assertion that
   makes the derivation permanent rather than a note somebody read once.
"""
from __future__ import annotations

import json

import numpy as np

from bestiary import paths
from bestiary.guards import Finding

# Section 2's derived bounds, restated as the inequalities they came from.
# These are the CONCLUSIONS of the derivation, so a guard may assert them; the
# derivation itself lives in the theory note and is not duplicated here.
#
# THE NOISE CONSTANTS ARE READ FROM THE MEASUREMENT FILE, BY ARM NAME, AND THAT
# IS THE WHOLE POINT. The first version of this guard hardcoded 0.127 rad/s as
# "the" yaw drift and used it in BOTH inequalities. 0.127 is the yaw of the
# `wheel_0.3` arm -- a machine that is DRIVING but not steering. The standing
# machine is the `wheel_0` arm and yaws at 0.01823, seven times less.
#
# The consequence was not a rounding error. Assertion 5 computed
# Phi(0.34/0.15) * Phi(0.127/0.10) = 0.0624 against a 0.16 cap and passed by
# 2.6x, while the quantity it claimed to be bounding -- a real standing machine
# under the easiest drive command -- is Phi(0.3354/0.15) * Phi(0.01855/0.10) =
# 0.1611, which is OVER that cap. A guard written to make a derivation
# permanent passed comfortably on a number the derivation never contained.
# Section 2 uses 0.968 for that factor, which is Phi(0.0182/0.10): the note had
# it right and the guard did not.
#
# Reading by arm name makes the substitution impossible rather than merely
# fixed. An arm that disappears from the file raises here instead of silently
# falling back to a plausible constant.
_NOISE = json.loads(
    (paths.RESEARCH / "measurements" / "tracking_noise.json").read_text()
)
_STANDING = _NOISE["arms"]["wheel_0"]        # zero wheel command: NOT driving
_DRIVING = _NOISE["arms"]["wheel_0.3"]       # driving, unsteered

STANDING_YAW_DRIFT = _STANDING["yaw"]["std"]        # 0.01823 rad/s
STANDING_DRIFT = abs(_STANDING["linear"]["mean_vx"])  # 0.03553 m/s, the creep
UNSTEERED_DRIVING_YAW = _DRIVING["yaw"]["std"]      # 0.12695 rad/s

MAX_UNSTEERED_YAW_FACTOR = 0.45          # Phi(0.127/sigma_w) <= 0.45, Section 2

# Section 2 aimed at 0.15 for this cap and reports its own chosen sigma_v
# binding it "at equality" at 0.158; with this repo's measured creep and
# standing yaw it lands at 0.1611. So sigma_v = 0.15 sits AT the cap by
# construction -- the note calls it "the rounded top" of the [0.12, 0.146]
# window -- and this bound exists to catch a WIDENING, not to re-litigate the
# choice. 0.17 leaves ~5% over the measured value: sigma_v = 0.16 already
# scores 0.179 and is caught, which is correct, because anything above 0.15 is
# outside the derived window.
MAX_STANDING_TAKE_EASIEST_DRIVE = 0.17
MIN_DRIVE_COMMAND = 0.3                  # = 2 * sigma_v, Section 3
# Backward drives need a LARGER minimum than forward ones, and the asymmetry is
# not a preference: the creep is backward, so it cancels part of the error under
# a backward command and the same |vx| buys standing a much larger take. Solving
# Phi((|vx| - 0.03553)/0.15) * 0.9678 <= 0.17 for |vx| gives 0.3756, and the
# nearest round value above it is 0.40 (take 0.1402). 0.35 does NOT clear it
# (0.1794). Derived from the cap, not chosen.
MIN_DRIVE_COMMAND_BACKWARD = 0.4


def run() -> list[Finding]:
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  -- registers the env ids
    from bestiary.envs.hound_track import (
        SIGMA_V,
        SIGMA_W,
        HoundTrackEnv,
        kernel,
    )

    out: list[Finding] = []
    env: HoundTrackEnv = gym.make("HoundPDTrackDesert-v0").unwrapped
    env.reset(seed=0)

    # --- 1. A body driving along its own +x reads as purely forward ----------
    speed, yaw = 0.5, np.deg2rad(40.0)
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]   # yaw-only quaternion
    qvel[:2] = [speed * np.cos(yaw), speed * np.sin(yaw)]      # world velocity along body +x
    env.set_state(qpos, qvel)

    v_b = env.heading_velocity
    world_would_read = np.array([speed * np.cos(yaw), speed * np.sin(yaw)])
    ok = np.allclose(v_b, [speed, 0.0], atol=1e-6)
    out.append(Finding(
        "velocity is measured in the heading frame, not the world frame",
        ok,
        f"at yaw={np.rad2deg(yaw):.0f}deg driving {speed} m/s along its own +x: "
        f"heading frame reads ({v_b[0]:.4f}, {v_b[1]:.4f}), "
        f"world frame would read ({world_would_read[0]:.4f}, {world_would_read[1]:.4f}); "
        f"expected ({speed:.4f}, 0.0000)",
        n=1,   # one constructed state: driving along its own +x at yaw=40deg
    ))

    # A forward command must be fully satisfied by that motion. This is the
    # assertion in the units the reward actually uses.
    phi_v = float(kernel(np.linalg.norm(v_b - np.array([speed, 0.0])) / SIGMA_V))
    phi_v_world = float(kernel(
        np.linalg.norm(world_would_read - np.array([speed, 0.0])) / SIGMA_V
    ))
    out.append(Finding(
        "a correctly-driving body under a turn command scores full tracking credit",
        phi_v > 0.999,
        f"Phi_v = {phi_v:.4f} in the heading frame; a world-frame reading would "
        f"cap it at {phi_v_world:.4f} on this segment and no training would close it",
        n=1,   # the same constructed state, read in reward units
    ))

    # --- 2. Lateral drift is penalized, not discarded ------------------------
    qvel[:2] = [speed * np.cos(yaw) - 0.20 * np.sin(yaw),
                speed * np.sin(yaw) + 0.20 * np.cos(yaw)]   # +0.2 m/s to its left
    env.set_state(qpos, qvel)
    v_b_side = env.heading_velocity
    err_side = float(np.linalg.norm(v_b_side - np.array([speed, 0.0])))
    out.append(Finding(
        "lateral velocity enters the error through the planar norm",
        abs(err_side - 0.20) < 1e-6 and abs(v_b_side[1] - 0.20) < 1e-6,
        f"0.20 m/s of sideways drift under a pure-forward command reads as "
        f"v_left={v_b_side[1]:.4f} and error {err_side:.4f} m/s "
        f"(Phi_v = {float(kernel(err_side / SIGMA_V)):.4f}); "
        f"a scalar v_forward - vx_cmd formulation would read error 0.0000",
        n=1,   # one constructed state: the same yaw, plus 0.2 m/s to its left
    ))

    # --- 3. yaw_rate is the body-frame z component ---------------------------
    qvel[:] = 0.0
    qvel[5] = 0.37
    env.set_state(qpos, qvel)
    out.append(Finding(
        "yaw_rate reads the trunk's body-frame z angular velocity",
        abs(env.yaw_rate - 0.37) < 1e-9,
        f"set qvel[5]=0.37, yaw_rate reads {env.yaw_rate:.6f}",
        n=1,   # one constructed state: pure body-frame yaw
    ))

    # --- 4. The kernel is Cauchy ---------------------------------------------
    # Phi(1) = 1/2 for Cauchy and exp(-1) = 0.3679 for the Gaussian the record
    # elsewhere claimed. One evaluation separates them unambiguously.
    out.append(Finding(
        "the tolerance kernel is Cauchy, 1/(1+u^2)",
        abs(float(kernel(1.0)) - 0.5) < 1e-12 and abs(float(kernel(1 / 3)) - 0.9) < 1e-9,
        f"Phi(1)={float(kernel(1.0)):.6f} (Gaussian would be 0.367879), "
        f"Phi(1/3)={float(kernel(1/3)):.6f} — the 3x rule's 10% cost",
        n=2,   # two evaluation points, u=1 and u=1/3
    ))

    # --- 5. The freeride inequalities sigma was derived from still hold ------
    # Standing (drifting backwards) under the EASIEST drive command. Both
    # factors use the STANDING arm, because the machine being bounded is
    # standing -- see the note on _STANDING above for what using the driving
    # arm's yaw here did.
    # The floors this assertion reasons about live in the ENV, and a guard that
    # bounds its own private copy of a constant bounds nothing. Two numbers that
    # must agree and are written in two files will eventually disagree, and this
    # guard's whole value is that it fails when they do.
    from bestiary.envs import hound_track as _ht

    out.append(Finding(
        "the guard's command floors are the env's command floors",
        _ht.VX_MIN == MIN_DRIVE_COMMAND
        and _ht.VX_MIN_BACKWARD == MIN_DRIVE_COMMAND_BACKWARD,
        f"env samples |vx| >= {_ht.VX_MIN} forward / {_ht.VX_MIN_BACKWARD} backward; "
        f"this guard asserts against {MIN_DRIVE_COMMAND} / "
        f"{MIN_DRIVE_COMMAND_BACKWARD}. If these drift apart the freeride bound "
        f"below is checking a distribution nothing samples from",
        n=2,   # two command floors compared against the env's own
    ))

    # BOTH SIGNS. The creep is BACKWARD (mean_vx = -0.03553), so it adds to the
    # error under a forward command and SUBTRACTS under a backward one. This
    # assertion used to compute the forward case only, and the sign it omitted
    # is the one that is not bounded:
    #
    #     forward  (+0.3):  err = 0.3 + 0.03553 = 0.33553 -> take 0.1112  OK
    #     backward (-0.3):  err = 0.3 - 0.03553 = 0.26447 -> take 0.2356  1.39x OVER
    #
    # Measured, not predicted: on the (-0.3, 0, 0) cell of
    # `hound_track_desert_s0`'s committed decomposition, zero action scores
    # 0.23567/step of tracking against the trained policy's 0.09737 and returns
    # 226.38 against 40.27. Backward commands are 16% of the command mass
    # (P_FORWARD = 0.8 within a drive mass of 0.8), so this is a standing
    # exploit sitting open on a sixth of training -- the third instance of the
    # failure this project has already paid for twice, shipped because the
    # guard against it only ever looked one way.
    #
    # The bound is the LARGEST take over the sign, so the smallest error.
    forward_err = MIN_DRIVE_COMMAND + STANDING_DRIFT
    backward_err = MIN_DRIVE_COMMAND_BACKWARD - STANDING_DRIFT
    worst_sign = "forward" if forward_err <= backward_err else "backward"
    standing_err = min(forward_err, backward_err)
    phi_v_standing = float(kernel(standing_err / SIGMA_V))
    phi_w_standing = float(kernel(STANDING_YAW_DRIFT / SIGMA_W))
    standing_take = phi_v_standing * phi_w_standing
    out.append(Finding(
        "standing cannot freeride the easiest drive command, either sign",
        standing_take <= MAX_STANDING_TAKE_EASIEST_DRIVE,
        f"worst sign is {worst_sign}: forward err {forward_err:.5f} "
        f"(cmd {MIN_DRIVE_COMMAND}), backward err {backward_err:.5f} "
        f"(cmd -{MIN_DRIVE_COMMAND_BACKWARD}); standing takes "
        f"{standing_take:.4f}/step = Phi_v {phi_v_standing:.4f} x Phi_w "
        f"{phi_w_standing:.4f}, at sigma_v={SIGMA_V} sigma_w={SIGMA_W}, "
        f"cap {MAX_STANDING_TAKE_EASIEST_DRIVE}. Both factors use the STANDING "
        f"arm (creep {STANDING_DRIFT:.5f} m/s BACKWARD, yaw "
        f"{STANDING_YAW_DRIFT:.5f} rad/s); the driving arm's yaw "
        f"{UNSTEERED_DRIVING_YAW:.5f} would give "
        f"{phi_v_standing * float(kernel(UNSTEERED_DRIVING_YAW / SIGMA_W)):.4f} "
        f"and bound nothing",
        n=2,   # both signs of the easiest drive command; the bound is the worse
    ))

    # This one genuinely IS about a driving machine, so it genuinely does use
    # the driving arm. The two assertions sitting next to each other with
    # different constants is the point.
    unsteered_yaw_factor = float(kernel(UNSTEERED_DRIVING_YAW / SIGMA_W))
    out.append(Finding(
        "at least half the reward rides on active yaw stabilization",
        unsteered_yaw_factor <= MAX_UNSTEERED_YAW_FACTOR,
        f"an unsteered but DRIVING machine yawing {UNSTEERED_DRIVING_YAW:.5f} "
        f"rad/s scores Phi_w={unsteered_yaw_factor:.4f} at sigma_w={SIGMA_W}, "
        f"cap {MAX_UNSTEERED_YAW_FACTOR}; a steered one scores "
        f"{float(kernel(0.03 / SIGMA_W)):.4f}",
        # A scalar threshold on one measured constant, not a quantification over
        # a set -- declaring a size here would be a lie.
        n=None,
    ))

    env.close()
    return out
