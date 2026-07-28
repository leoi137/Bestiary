"""Arithmetic for docs/lessons/006-what-gamma-is-saying-about-the-future.md.

Every number in that lesson is printed here. Nothing is hardcoded that can be
read from the thing itself: gamma comes out of the SAC constructor's own
signature, the episode length out of the gym registry, the termination penalty
out of the env module, and the per-step rates out of the committed measurement
`research/measurements/hound_track_desert_s0_final_decomposition.json`.

That matters more than usual here. The lesson's whole point is that a constant
derived from an assumed per-step rate is only as good as the rate, so a script
that ASSERTS the rate instead of reading it would be making the same mistake it
describes.

    venv/bin/python docs/lessons/scripts/006_discounting_math.py
"""
from __future__ import annotations

import inspect
import json

import gymnasium as gym
from stable_baselines3 import SAC

import bestiary.envs  # noqa: F401  -- importing registers the env ids
from bestiary import paths
from bestiary.envs.hound_track import TERMINATION_PENALTY

# --- Where the two constants actually come from ------------------------------
# gamma is NEVER named in this repo. train.py builds SAC without passing it, so
# the discount is Stable-Baselines3's default, read here from the signature so
# a version bump that moved it would move this lesson too.
GAMMA = inspect.signature(SAC).parameters["gamma"].default
ENV_ID = "HoundPDTrackDesert-v0"
EPISODE_STEPS = gym.registry[ENV_ID].max_episode_steps
CONTROL_HZ = 20.0  # envs/__init__.py: "1000 steps at 20 Hz = 50 s"

# The rate the termination penalty was derived from, quoted from the derivation
# in docs/theory/command-tracking-reward.md Section 4 -- an ESTIMATE of an early
# flailing policy's net loss per step, ctrl ~0.053 + contact ~0.05, tracking ~0.
C_ASSUMED = 0.10
CONTACT_ASSUMED = 0.045  # envs/hound_track.py module docstring, per step

print("=" * 70)
print("1. WHAT gamma IS, AND WHERE IT COMES FROM")
print("=" * 70)
print(f"  gamma                        {GAMMA}   (SAC's default; train.py never sets it)")
print(f"  episode length               {EPISODE_STEPS} steps "
      f"= {EPISODE_STEPS / CONTROL_HZ:.0f} s at {CONTROL_HZ:.0f} Hz")

print()
print("=" * 70)
print("2. THE EFFECTIVE HORIZON, 1/(1-gamma)")
print("=" * 70)
for g in (0.95, GAMMA, 0.999):
    h = 1.0 / (1.0 - g)
    print(f"  gamma = {g:<6}  ->  horizon {h:8.1f} steps = {h / CONTROL_HZ:6.2f} s")

print()
print("=" * 70)
print(f"3. WHAT A REWARD n STEPS AWAY IS WORTH, AT gamma = {GAMMA}")
print("=" * 70)
for n in (10, 100, 500, EPISODE_STEPS):
    print(f"  {n:>5} steps ahead   gamma^n = {GAMMA ** n:.6g}"
          f"   ({GAMMA ** n * 100:.4f}% of the same reward now)")

print()
print("=" * 70)
print("4. THE GEOMETRIC SERIES, AND THE TERMINATION PENALTY IT PRODUCED")
print("=" * 70)
print("  sum_{t=0..inf} gamma^t * c  =  c / (1 - gamma)  =  c * horizon")
k_assumed = C_ASSUMED / (1.0 - GAMMA)
truncated = C_ASSUMED * (1.0 - GAMMA ** EPISODE_STEPS) / (1.0 - GAMMA)
print(f"  at c = {C_ASSUMED} per step   K = {C_ASSUMED} / {1 - GAMMA:.2f} = {k_assumed:.4f}")
print(f"  same sum cut off at the episode's {EPISODE_STEPS} steps: {truncated:.6f}"
      f"  (short by {k_assumed - truncated:.2e})")
print(f"  envs/hound_track.py TERMINATION_PENALTY = {TERMINATION_PENALTY}")
print(f"  match? {abs(k_assumed - TERMINATION_PENALTY) < 1e-9}")

# --- The measured rates ------------------------------------------------------
MEAS = paths.RESEARCH / "measurements" / \
    "hound_track_desert_s0_final_decomposition.json"
d = json.loads(MEAS.read_text())


def rates(arm: str) -> dict[str, float]:
    """Per-step rates over the six-cell drive grid, from episode totals.

    The denominator is that arm's own mean episode length, not 1000: the
    trained arm crashes on one cell, so its episodes are shorter and dividing
    by 1000 would understate every rate it earns.
    """
    a = d[arm]
    n = a["drive_grid_steps"]
    running = (a["drive_grid_reward_track"]
               + a["drive_grid_reward_ctrl"]
               + a["drive_grid_reward_contact"])
    return {
        "steps": n,
        "track": a["drive_grid_reward_track"] / n,
        "ctrl": a["drive_grid_reward_ctrl"] / n,
        "contact": a["drive_grid_reward_contact"] / n,
        "net": running / n,        # EXCLUDES the one-time termination penalty:
                                   # it is the thing being priced, not an input
    }


tr, zr = rates("trained"), rates("zero_action")

print()
print("=" * 70)
print("5. THE RATES THIS ROBOT ACTUALLY EARNS (six-cell drive grid, n=20/cell)")
print("=" * 70)
print(f"  {'':<28}{'trained':>12}{'zero action':>14}")
print(f"  {'mean episode length':<28}{tr['steps']:12.1f}{zr['steps']:14.1f}")
print(f"  {'tracking reward / step':<28}{tr['track']:12.5f}{zr['track']:14.5f}")
print(f"  {'control cost / step':<28}{tr['ctrl']:12.5f}{zr['ctrl']:14.5f}")
print(f"  {'contact cost / step':<28}{tr['contact']:12.5f}{zr['contact']:14.5f}")
print(f"  {'NET / step':<28}{tr['net']:12.5f}{zr['net']:14.5f}")

print()
print("  the derivation's inputs against the measurement:")
print(f"    contact cost   assumed {CONTACT_ASSUMED:.5f}/step,"
      f" measured {abs(zr['contact']):.5f}/step standing"
      f"  -> {CONTACT_ASSUMED / abs(zr['contact']):.2f}x too big")
print(f"    net rate       assumed {-C_ASSUMED:+.5f}/step,"
      f" measured {tr['net']:+.5f}/step trained"
      f"  -> {C_ASSUMED / abs(tr['net']):.1f}x too big")

# Cross-check the tracking rate against the OTHER published figure for the same
# quantity. `drive_grid_track` in the *_final_sac.json is a mean over cells of
# each cell's per-step mean, which weights a short crashed episode the same as
# a full one -- anomalies.jsonl row 20. The rates above divide summed reward by
# summed steps instead, so a crash's missing steps count. The two disagree only
# for the arm that crashes, and the size of that disagreement is printed rather
# than hidden.
sac = json.loads((paths.RESEARCH / "measurements"
                  / "hound_track_desert_s0_final_sac.json").read_text())
print()
print("  cross-check, the same tracking rate the other way "
      "(drive_grid_track, length-biased):")
for arm, r in (("trained", tr), ("zero_action", zr)):
    published = sac[arm]["drive_grid_track"]
    print(f"    {arm:<12} published {published:.5f}   here {r['track']:.5f}"
          f"   diff {published - r['track']:+.5f}")

print()
print("=" * 70)
print("6. THE SAME FORMULA, AT THE RATE THE MACHINE ACTUALLY ACHIEVES")
print("=" * 70)
k_measured = abs(tr["net"]) / (1.0 - GAMMA)
print(f"  K at the assumed -{C_ASSUMED}/step        {k_assumed:8.4f}   <- shipped")
print(f"  K at the measured {tr['net']:+.5f}/step   {k_measured:8.4f}")
print(f"  the shipped penalty is {k_assumed / k_measured:.1f}x the stream it prices")

print()
print("=" * 70)
print("7. THE OTHER READING OF THE SAME SUM: WHAT A LIFE IS WORTH")
print("=" * 70)
lo, hi = zr["track"], tr["track"]
print(f"  tracking income / step   trained {hi:.5f}   doing nothing {lo:.5f}")
print(f"  doing nothing collects {lo / hi * 100:.1f}% of what trying collects")
print(f"  capitalised, income * horizon   trained {hi / (1 - GAMMA):.4f}"
      f"   doing nothing {lo / (1 - GAMMA):.4f}")
print(f"  gap {(hi - lo) / (1 - GAMMA):.4f} -- the whole discounted worth of "
      f"tracking better than a corpse")
print()
print("  and once the control cost of earning it is subtracted (NET / step):")
print(f"    a discounted life of doing nothing   {zr['net'] / (1 - GAMMA):+8.4f}")
print(f"    a discounted life of trying          {tr['net'] / (1 - GAMMA):+8.4f}")
print(f"    trying is worth {(zr['net'] - tr['net']) / (1 - GAMMA):.4f} LESS, "
      f"which is {(zr['net'] - tr['net']) / (1 - GAMMA) / TERMINATION_PENALTY * 100:.0f}% "
      f"of the death penalty")
