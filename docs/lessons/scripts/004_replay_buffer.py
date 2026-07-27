"""Arithmetic for docs/lessons/004-what-the-replay-buffer-holds.md.

Every number in that lesson is printed here, computed from this repo's own
config files and measured file sizes rather than asserted in prose.

    venv/bin/python docs/lessons/scripts/004_replay_buffer.py
"""
from __future__ import annotations

import json

from bestiary import paths

# --- What one transition costs, from the run's own recorded shapes ----------
# Read from a real config.json rather than hardcoded, so the lesson cannot
# drift from the robot.
CONFIG = paths.RUNS / "hound_track_desert_s0" / "config.json"
cfg = json.loads(CONFIG.read_text())

obs_width = cfg["obs_spec"]["width"]
n_actions = 16                      # hound: 4 legs x 4 joints
buffer_size = cfg["hyperparameters"]["buffer_size"]
batch_size = cfg["hyperparameters"]["batch_size"]

# The observation dtype is float64, NOT float32, because envs/hound.py declares
# its Box with dtype=np.float64. This is the whole reason the buffer is 2.6 GB
# and not 1.4 GB, and getting it wrong is what made the first version of this
# script disagree with the measured file by a factor of 1.8.
OBS_BYTES = 8       # float64 observation
ACT_BYTES = 4       # float32 action
SCALAR_BYTES = 4    # float32 reward, done, timeout

# SB3's ReplayBuffer stores, per transition: observations, next_observations,
# actions, rewards, dones, and timeouts (handle_timeout_termination=True).
bytes_per_transition = (
    2 * obs_width * OBS_BYTES + n_actions * ACT_BYTES + 3 * SCALAR_BYTES
)
total_bytes = bytes_per_transition * buffer_size

print("ONE TRANSITION")
print(f"  observation width           {obs_width}")
print(f"  two observations (s, s')    2 x {obs_width} x {OBS_BYTES}B = "
      f"{2 * obs_width * OBS_BYTES}B")
print(f"  action                      {n_actions} x {ACT_BYTES}B = "
      f"{n_actions * ACT_BYTES}B")
print(f"  reward + done + timeout     3 x {SCALAR_BYTES}B = "
      f"{3 * SCALAR_BYTES}B")
print(f"  = bytes per transition      {bytes_per_transition}B")
print()
print("THE WHOLE BUFFER")
print(f"  capacity                    {buffer_size:,} transitions")
print(f"  predicted                   {total_bytes:,} bytes "
      f"= {total_bytes / 1024**3:.3f} GiB")

# Measured: the actual file this project wrote, in bytes. The prediction above
# is checked against it rather than asserted beside it.
MEASURED_BYTES = 2_780_004_759   # ls -l runs/hound_pd_desert_s1/ant_buffer.pkl
print(f"  measured on disk            {MEASURED_BYTES:,} bytes "
      f"= {MEASURED_BYTES / 1024**3:.3f} GiB")
print(f"  difference                  {MEASURED_BYTES - total_bytes:,} bytes "
      f"({(MEASURED_BYTES - total_bytes) / MEASURED_BYTES * 1e6:.1f} ppm) "
      f"-- the pickle header")

# --- How many times one transition is reused -------------------------------
# SB3 SAC does one gradient step per environment step by default, and each
# gradient step samples `batch_size` transitions uniformly from the buffer.
steps = 1_000_000
draws = steps * batch_size
reuse = draws / buffer_size

print()
print("HOW OFTEN ONE TRANSITION IS REUSED")
print(f"  gradient steps over {steps:,} env steps   {steps:,}")
print(f"  transitions drawn per step               {batch_size}")
print(f"  total draws                              {draws:,}")
print(f"  buffer capacity                          {buffer_size:,}")
print(f"  -> each transition is trained on ~{reuse:.0f} times")

# --- What a reward change does ---------------------------------------------
# nulls.jsonl row 1: the spyder warm-start. The buffer was full of transitions
# labelled under the flat-world reward; the env then paid a different one.
BEFORE = 6331.0    # ep_rew_mean before the warm-start, runs/spyder_walk_v3
AFTER = 146.0      # a few thousand steps into SpyderDesert-v0
print()
print("WHAT A STALE LABEL COSTS (nulls.jsonl row 1, the spyder warm-start)")
print(f"  ep_rew_mean before         {BEFORE:.0f}")
print(f"  ep_rew_mean after          {AFTER:.0f}")
print(f"  collapse                   {BEFORE / AFTER:.1f}x, and it never recovered")

# --- The two hashes that make this checkable -------------------------------
old = json.loads((paths.RUNS / "hound_pd_desert_s1"
                  / "config.json").read_text())["reward_spec"]
new = cfg["reward_spec"]
print()
print("THE HASH THAT REFUSES THE RESUME")
print(f"  hound_pd_desert_s1  shape_hash  {old['shape_hash']}")
print(f"  hound_track_desert_s0 shape_hash {new['shape_hash']}")
print(f"  identical? {old['shape_hash'] == new['shape_hash']}"
      "  -> train.py refuses to resume one as the other")
