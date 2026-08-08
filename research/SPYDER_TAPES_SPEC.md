# Spyder camera tapes — the format contract

*2026-08-08. This spec and `src/bestiary/isaac/record_spyder.py` change
together or not at all. The recorder asserts the load-bearing numbers below at
startup and refuses to tape if any have moved. Command schedules and captions
come from `src/bestiary/isaac/spyder_tape_commands.py`, the pure module the
closed-loop eval replays byte-identically.*

The stage-2 dataset for vision-language-action fine-tuning: the exact
(observation, action, command) tensors crossing a trained Spyder policy's
interface at the policy's own rate, plus first-person camera frames at 10 Hz
and a mechanical language caption per episode.

## The teacher, and why its scan matters

A self-trained checkpoint (default: `runs/spyder_gentle_s1`, seed 1 — PPO,
1500 iterations, ~90% episode survival on the gentle mix; the exact path is
saved into every episode's metadata). The teacher is **privileged**: its
observation includes a 187-ray height scan of the ground around and ahead of
it, and its foot placement responds to that scan. The camera films the same
ground, so a student trained frames→actions must recover the scan's
information from pixels — that correspondence is the entire point of these
tapes. A blind teacher's tapes would contain nothing for a camera to explain.

## Rates

| quantity | value |
|---|---|
| physics dt | 0.005 s (200 Hz), decimation 4 → policy at **50 Hz** |
| obs/act/cmd rows | one per policy step (50 Hz) |
| camera frames | **10 Hz** (`frame_stride = 5` policy steps) |
| alignment | frame *k* pairs with obs row `5k`; rendered ≤1 policy step earlier (`sim.render_interval = decimation × stride`) |

## Observation — 235 dims, float32, SI, body frame

Layout from `LocomotionVelocityRoughEnvCfg` with the gentle task's scan
footprint (`spyder_gentle_env_cfg`): asserted as a total width at startup;
joint order is the articulation's `joint_names`, saved into metadata.

| slice | content | units |
|---|---|---|
| `[0:3]` | base linear velocity, body frame | m/s |
| `[3:6]` | base angular velocity, body frame | rad/s |
| `[6:9]` | gravity direction, body frame | — |
| `[9:12]` | command (v_x, v_y, ω_z) | m/s, m/s, rad/s |
| `[12:24]` | joint positions − default | rad |
| `[24:36]` | joint velocities | rad/s |
| `[36:48]` | previous action (raw) | — |
| `[48:235]` | height scan, 17×11 rays, 0.16 m spacing, 2.56×1.6 m footprint | m |

The scan is recorded **deliberately** — it is the privileged signal. Distilling
against it (scan-supervision auxiliaries) is a training-side choice; a student
that consumes it at deployment is not vision-dependent and misses the point.
Observation noise is off (Play config): tapes are the teacher at its best.

## Action — 12 dims, float32, raw

The raw policy output as passed to the env (`clip_actions=None`). Position
targets are `default_pos + scale × action` downstream; scale and defaults are
the task's, not the tape's.

## Episodes — one command, one caption

Per episode (durations in `spyder_tape_commands.py`): 1 s settle (simulated,
**untaped**) → 1 s commanded stand (taped) → one driving command held 10 s to
the end. No stop tail: the caption must be true of every frame it covers.
10% of episodes are pure standing (`"stand still"`), matching the teacher's
trained `rel_standing_envs = 0.1`.

Commands are a pure function of the episode seed, drawn from the **live task
config's** ranges (read at runtime, never hardcoded) mirroring the trained
dead-zone regimes: |v_x| ∈ [0.25, 0.6] m/s signed; ω_z snapped to exactly 0
below 0.2 rad/s else |ω_z| ∈ [0.2, 0.8]; v_y exactly 0 for the gentle task.
Captions are deterministic per command class (`command_text`): "walk forward",
"walk backward, turning left", "turn right in place", "stand still", …
Paraphrase augmentation belongs to the training-side converter, seeded and
recorded there.

## Camera

| fact | value |
|---|---|
| resolution | 224×224 RGB (PaliGemma's input size — SigLIP-So400m/14, arXiv:2407.07726) |
| mount | torso frame, (0.18, 0, 0.22) m, pitched 25° down, ROS optical convention |
| lens | 12 mm focal, 20.955 mm aperture → ~82° horizontal FOV |
| geometry | view centres ~1 m ahead at stance; covers the scan's 1.28 m forward reach. Mount raised from (0.18, 0, 0.12)/20° after the first smoke filmed the torso shell over the bottom ~40% of frame (operator decision, 2026-08-08) |
| encoding | H.264 (libx264, CRF 18, yuv420p) at 10 fps, one MP4 per episode |

The recorder refuses to tape blank frames (per-env std guard at first grab —
the frozen-boot-camera / sky-only render anomaly class in STATE) and static
frames (staleness guard on the first driving survivor). Mount numbers are
geometry, not yet measurement — **verify on smoke frames** and update both
files together if the view is wrong.

## Files

`runs/spyder_tapes/{train,holdout,fallen}/ep_<seed>.{npz,mp4}` (gitignored):

| field | shape | dtype |
|---|---|---|
| `obs` | (T, 235) | float32 |
| `act` | (T, 12) | float32 |
| `cmd` | (T, 3) | float32 |
| `meta` | JSON string | seed, text, command, fell, steps, frames, rates, dof_names, checkpoint, task, terrain grid + hfield md5, camera config, spec |
| `ep_<seed>.mp4` | (F, 224, 224, 3) | uint8 via H.264 |

## Split and hygiene rules — inherited from the Spot spec verbatim

1. **Holdout at record time, by seed:** `seed % 10 == 0` → `holdout/`; never
   trained on, including normalization statistics.
2. **No normalization at record time.** Raw SI is the immutable ground truth.
3. **Falls are quarantined, not deleted** (`fallen/`): termination (torso
   contact) ends the tape at that step; the footage is debugging evidence.
4. **Terrain is provenance:** grid size and the gentle hfield's md5 are in
   every episode's metadata. Tapes from different ground are different data.

## Deliberately absent

Rewards (imitation has none), normalization statistics (training-side, from
`train/` only), LeRobot formatting (a separate converter's job — the tape is
the raw record it converts from), depth/segmentation (RGB is what the student
gets on real hardware), and stop transitions (see episode design; add a
"stop" phase type only with a caption scheme that stays true frame-by-frame).
