"""Arithmetic for docs/lessons/007-a-tolerance-that-cancels-the-command.md.

Every number in that lesson is printed here. Nothing that can be read from the
thing itself is hardcoded: the kernel and the tolerance rules are imported from
the env, the command mixture is drawn from the env's OWN sampler rather than a
re-implementation of it, and the drift a standing machine actually has is
recovered from the committed measurement
`research/measurements/track_rel_zero_action.json`.

The point of the lesson is that a tolerance proportional to the command cancels
the command out of a standing machine's score. A script that hardcoded the
score instead of evaluating the shipped kernel would be asserting exactly the
thing under examination.

    venv/bin/python docs/lessons/scripts/007_scale_free_tolerance.py
"""
from __future__ import annotations

import json

import gymnasium as gym
import numpy as np

import bestiary.envs  # noqa: F401  -- importing registers the env ids
from bestiary import paths
from bestiary.envs.hound_track_rel import (
    ALPHA_V_MIN,
    ALPHA_W_MIN,
    BETA_V,
    BETA_W,
    relative_kernel,
)

K_SHIPPED = BETA_W          # 0.5, what hound_track_rel.py ships
K_FIRST = 0.75              # the first value written, caught before it trained
ENV_ID = "HoundPDTrackRelDesert-v0"
N_DRAWS = 200_000
DRAW_SEED = 0

REL = json.loads((paths.RESEARCH / "measurements"
                  / "track_rel_zero_action.json").read_text())["zero_action"]
OLD = json.loads((paths.RESEARCH / "measurements"
                  / "hound_track_desert_s0_final_sac.json").read_text())

TURN_CELL = "(0.0, 0.0, 0.45)"
STOP_CELL = "(0.0, 0.0, 0.0)"


def invert_kernel(phi: float, alpha: float) -> float:
    """The error that scores phi under K(e;alpha) = exp(-(e/alpha)^2)."""
    return alpha * float(np.sqrt(-np.log(phi)))


# A standing machine's drift, recovered from the STOP cell, where the command
# is zero so the score IS the drift's score and the tolerance is the floor.
E_V_DRIFT = invert_kernel(REL["cells"][STOP_CELL]["mean_phi_v"], ALPHA_V_MIN)
E_W_DRIFT = invert_kernel(REL["cells"][STOP_CELL]["mean_phi_w"], ALPHA_W_MIN)
PHI_V_STANDING = REL["cells"][STOP_CELL]["mean_phi_v"]

print("=" * 72)
print("1. WHAT A STANDING MACHINE IS ACTUALLY DOING")
print("=" * 72)
print(f"  measured, stop cell {STOP_CELL}, n=20 seeds 1000-1019:")
print(f"    mean phi_v {PHI_V_STANDING:.7f}  ->  speed drift {E_V_DRIFT:.5f} m/s")
print(f"    mean phi_w {REL['cells'][STOP_CELL]['mean_phi_w']:.7f}"
      f"  ->  yaw drift   {E_W_DRIFT:.5f} rad/s")
print(f"  commanded turns run {0.3}-{0.6} rad/s, i.e. "
      f"{0.3 / E_W_DRIFT:.0f}-{0.6 / E_W_DRIFT:.0f}x that drift.")
print("  So on a turn command the standing machine's yaw error IS the command.")

print()
print("=" * 72)
print("2. THE COMMAND CANCELS:  exp(-(|w| / (k*|w|))^2) = exp(-1/k^2)")
print("=" * 72)
for w in (0.30, 0.40, 0.45, 0.60):
    for k in (K_FIRST, K_SHIPPED):
        print(f"  w_cmd = {w:.2f}  k = {k:.2f}   alpha_w = {k * w:.4f}"
              f"   K = {relative_kernel(w, k * w):.6f}")
print("  the score does not depend on w_cmd at all -- only on k:")
for k in (K_FIRST, K_SHIPPED):
    print(f"    k = {k:.2f}   exp(-1/k^2) = {np.exp(-1.0 / k ** 2):.6f}")
ratio_k = np.exp(-1.0 / K_FIRST ** 2) / np.exp(-1.0 / K_SHIPPED ** 2)
print(f"  going from k = {K_FIRST} to k = {K_SHIPPED} cuts it {ratio_k:.1f}x")

print()
print("=" * 72)
print("3. WHAT THAT PAYS PER STEP ON A TURN-IN-PLACE COMMAND")
print("=" * 72)
print(f"  a stander is correctly NOT moving on (0,0,w), so phi_v = "
      f"{PHI_V_STANDING:.6f}")
for k in (K_FIRST, K_SHIPPED):
    print(f"    k = {k:.2f}   income = {np.exp(-1.0 / k ** 2):.6f} x "
          f"{PHI_V_STANDING:.6f} = "
          f"{np.exp(-1.0 / k ** 2) * PHI_V_STANDING:.6f} / step")
best_drive = max(v["mean_track"] for c, v in OLD["trained"]["cells"].items()
                 if c != STOP_CELL and json.loads(c.replace("(", "[")
                                                  .replace(")", "]"))[2] == 0.0)
old_stander_turn = OLD["zero_action"]["cells"][TURN_CELL]["mean_track"]
print("  for scale, measured under the reward this design replaces:")
print(f"    trained policy, best straight-drive cell   {best_drive:.5f} / step")
print(f"    stander, turn cell {TURN_CELL}          {old_stander_turn:.5f} / step")

print()
print("  cross-check against the SHIPPED env, measured not predicted:")
for cell in ("(0.5, 0.0, 0.4)", "(0.5, 0.0, -0.4)", TURN_CELL):
    print(f"    {cell:<18} measured phi_w {REL['cells'][cell]['mean_phi_w']:.6f}"
          f"   predicted exp(-1/k^2) {np.exp(-1.0 / K_SHIPPED ** 2):.6f}")

# --- The training mixture ----------------------------------------------------
# Drawn from the env's own _resample_command, not a re-implementation, so a
# change to the command distribution moves this number automatically.
env = gym.make(ENV_ID).unwrapped
env.reset(seed=DRAW_SEED)
cmds = []
for _ in range(N_DRAWS):
    env._resample_command()
    cmds.append(env._cmd.copy())
cmds = np.asarray(cmds)
env.close()

# Every draw that asks for motion: DRIVE (vx != 0) plus TURN-IN-PLACE
# (vx = 0, w != 0). The STOP draws are excluded because a standing machine is
# genuinely doing the job there -- that income is earned, not freeride.
motion = cmds[np.any(cmds != 0.0, axis=1)]
drive = cmds[cmds[:, 0] != 0.0]
turn = cmds[(cmds[:, 0] == 0.0) & (cmds[:, 2] != 0.0)]


def stander_income(cmd: np.ndarray, k_w: float) -> float:
    """Track-term income per step for a machine that does nothing.

    Its planar velocity is the measured backward creep, so the speed error is
    |vx_cmd| + creep going forward and |vx_cmd| - creep going backward. Its yaw
    error is the yaw drift against the command. Shaping telescopes to ~0 over an
    episode and the control cost of doing nothing is exactly 0, so the track
    term is the whole income.
    """
    vx, _, w = cmd
    e_v = abs(vx - (-E_V_DRIFT))
    e_w = abs(w - E_W_DRIFT)
    alpha_v = max(ALPHA_V_MIN, BETA_V * abs(vx))
    alpha_w = max(ALPHA_W_MIN, k_w * abs(w))
    return relative_kernel(e_v, alpha_v) * relative_kernel(e_w, alpha_w)


print()
print("=" * 72)
print(f"4. A STANDER'S INCOME OVER EVERY DRAW THAT ASKS FOR MOTION "
      f"(n={len(motion)} of {N_DRAWS}: {len(drive)} drive, {len(turn)} turn)")
print("=" * 72)
inc = {k: float(np.mean([stander_income(c, k) for c in motion]))
       for k in (K_FIRST, K_SHIPPED)}
for k in (K_FIRST, K_SHIPPED):
    print(f"  k = {k:.2f}   {inc[k]:.5f} / step"
          f"   (drive draws {np.mean([stander_income(c, k) for c in drive]):.5f},"
          f" turn draws {np.mean([stander_income(c, k) for c in turn]):.5f})")
print(f"  k = {K_FIRST} pays a do-nothing machine "
      f"{inc[K_FIRST] / inc[K_SHIPPED]:.1f}x what k = {K_SHIPPED} pays it")
# The like-for-like comparison against the OLD reward is NOT this mixture rate.
# A mixture rate averages over the command DISTRIBUTION; drive_grid_track
# averages over six FIXED commands. Dividing one by the other was printed here
# as a "7.1x cut" and it compares two different denominators over two different
# command sets. The same-commands number is six-cell against six-cell:
old_drive = OLD["zero_action"]["drive_grid_track"]
new_drive = REL["drive_grid_track"]
print("  the SAME SIX COMMANDS under each reward (drive_grid_track):")
print(f"    old reward {old_drive:.5f} / step -> shipped k = {K_SHIPPED} "
      f"{new_drive:.5f} / step   = {old_drive / new_drive:.1f}x cut")
print(f"  do NOT divide {old_drive:.5f} (six fixed commands) by "
      f"{inc[K_SHIPPED]:.5f} (the mixture) -- different command sets.")

print()
print("=" * 72)
print("5. THE WHOLE-EPISODE NUMBERS THAT SHIPPED")
print("=" * 72)
old = json.loads((paths.RESEARCH / "measurements"
                  / "hound_track_desert_s0_final_sac.json").read_text())
print(f"  {'zero action, per episode':<30}{'old reward':>14}{'k=0.5':>12}")
print(f"  {'drive grid mean return':<30}"
      f"{old['zero_action']['drive_grid_mean']:14.2f}"
      f"{REL['drive_grid_mean']:12.2f}")
print(f"  {'drive grid track / step':<30}"
      f"{old['zero_action']['drive_grid_track']:14.5f}"
      f"{REL['drive_grid_track']:12.5f}")
print(f"  {'stop cell mean return':<30}"
      f"{old['zero_action']['stop_cell_mean']:14.2f}"
      f"{REL['stop_cell_mean']:12.2f}")
