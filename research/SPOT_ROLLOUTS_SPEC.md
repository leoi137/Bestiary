# Spot rollout dataset — the format contract

*2026-08-06. This spec and `src/bestiary/isaac/record_spot.py` change
together or not at all. The recorder asserts the load-bearing numbers below
at startup and refuses to tape if any have moved.*

The stage-1 dataset for next-token imitation: the exact (observation, action)
tensors crossing the interface of a pretrained quadruped walking policy, at
the policy's own rate, with nothing normalized, reordered, or converted.

## The teacher

The Isaac Sim sample checkpoint, used as-is. The exact policy path is saved
into every episode's metadata, and `--policy-path`/`--env-config` can
substitute any compatible TorchScript checkpoint (e.g. a self-trained
`Isaac-Velocity-Flat-Spot-v0` run) without touching this format.

| fact | value | source |
|---|---|---|
| Policy | Spot flat-terrain locomotion, TorchScript | `/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt`, NVIDIA asset server |
| Trained by | The AI Institute, on Isaac Lab's `Isaac-Velocity-Flat-Spot-v0`, from Boston Dynamics specifications | isaac-sim/IsaacLab spot task README, retrieved 2026-08-06 |
| Recipe | PPO (rsl_rl), 4,096 envs, 15,000 iterations, ~4 h on one RTX 4090 | NVIDIA developer blog, 2024-06-17 |
| Loader | `SpotFlatTerrainPolicy` (Apache-2.0) | isaac-sim/IsaacSim GitHub, `isaacsim.robot.policy.examples`, retrieved 2026-08-06 |

## Rates — from the policy's own env config (`spot_env.yaml`, fetched 2026-08-06)

| quantity | value |
|---|---|
| sim dt | 0.002 s (500 Hz physics) |
| decimation | 10 → policy at **50 Hz** |
| action scale | 0.2 |
| joint targets applied | every physics step; recomputed every 10th |

One row per **policy step** (50 Hz), not per physics step. NVIDIA's own
`spot_standalone.py` example boots physics at 200 Hz against this 500
Hz-trained policy; the recorder does not inherit that — it sets the sim to the
yaml's dt and refuses to run otherwise.

## Observation — 48 dims, float32, SI, body frame

Layout verbatim from `SpotFlatTerrainPolicy._compute_observation` (GitHub,
retrieved 2026-08-06):

| slice | content | units |
|---|---|---|
| `[0:3]` | base linear velocity, body frame | m/s |
| `[3:6]` | base angular velocity, body frame | rad/s |
| `[6:9]` | gravity direction, body frame (unit vector) | — |
| `[9:12]` | command (v_x, v_y, ω_z) | m/s, m/s, rad/s |
| `[12:24]` | joint positions − default pose | rad |
| `[24:36]` | joint velocities − default | rad/s |
| `[36:48]` | previous action (raw net output) | — |

Joint order is the articulation's `dof_names`, saved verbatim into every
episode's metadata — never assumed, never hand-listed here.

## Action — 12 dims, float32, raw

The uncut TorchScript output (verified: no clip, no scale in
`PolicyController._compute_action`). The controller applies it as
`joint_target = default_pos + 0.2 × action` (rad). The dataset stores the
**raw** action; 0.2 and the default pose live in metadata and in the yaml —
a training pipeline that bakes them in is breaking the contract.

## Commands — sampled inside the trained distribution

Ranges are the trained distribution's edges, from `spot_env.yaml`:
v_x ∈ [−2.0, 3.0] m/s, v_y ∈ [−1.5, 1.5] m/s, ω_z ∈ [−2.0, 2.0] rad/s.
Each episode: 1 s stand → 2–4 driving phases of 2–4 s (one forced
pure-forward, v_x ∈ [0.5, 3.0]) → 1 s stop. Schedule is a pure function of
the episode seed.

## Files

`runs/spot_rollouts/{train,holdout,fallen}/ep_<seed>.npz` (gitignored), each:

| field | shape | dtype |
|---|---|---|
| `obs` | (T, 48) | float32 |
| `act` | (T, 12) | float32 |
| `cmd` | (T, 3) | float32 |
| `meta` | JSON string | seed, fell, dt_s, decimation, policy_rate_hz, action_scale, dof_names, engine, spec |

## Split and hygiene rules

1. **Holdout is decided at record time, by seed:** `seed % 10 == 0` →
   `holdout/`. Nothing in `holdout/` is ever trained on, including for
   normalization statistics.
2. **No normalization at record time.** Raw SI values are the immutable
   ground truth; whatever statistics training needs are computed from
   `train/` only, by the training code, and recorded with the model.
3. **Falls are quarantined, not deleted.** Torso below 0.3 m → `fallen/`.
   Imitating a fall teaches falling; the tape is kept as debugging evidence.
4. **The physics engine is recorded per episode** (`engine` in meta). PhysX
   and Newton episodes are different dynamics and are never mixed in one
   training set without saying so.

## Deliberately absent

Images (stage 1 is blind), terrain variation (flat grid only — the surface
this policy was trained for), rewards (imitation has none), our own robots
(this dataset proves the pipeline; the Spyder/Hound tapes get their own spec
when their turn comes).
