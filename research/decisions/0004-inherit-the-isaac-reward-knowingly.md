# 0004 — Inherit the Isaac reward knowingly, and re-scope it when Hound arrives

**Date:** 2026-07-29 · **Status:** accepted · **Robot:** Hound (16-DoF wheel-legged), on Isaac Lab

## The decision

Two parts, held at different confidences.

**Part A — keep the inherited reward for ANYmal-C, but stop calling it a tuned
control.** `AnymalCDesertEnvCfg` continues to inherit `AnymalCRoughEnvCfg`
unchanged. Nothing about the terrain-bridge measurement changes. What changes
is what we claim it is: **it is Isaac Lab's generic velocity reward, not a
reward NVIDIA tuned for ANYmal-C or for their terrain.** Confidence **high,
about 95%** — this was read directly from source.

**Part B — when Hound replaces ANYmal-C, the table is re-scoped, not
inherited.** Four of the eleven inherited terms are meaningless or wrong for a
robot with driven hub wheels instead of feet, and two of them will raise at env
construction rather than degrade quietly. The starting point is `robot_lab`'s
Go2-W table (below), cross-checked against Matsuzawa et al., not Isaac Lab's
base table. Confidence **moderate, about 65%** — the structure is corroborated
by four independent implementations, but **no one has published an ablation of
any wheeled-legged reward coefficient**, so the specific weights are convention
rather than evidence.

## Why we asked

`src/bestiary/isaac/anymal_desert_env_cfg.py` lines 33–34 state:

> Everything comes from `AnymalCRoughEnvCfg` — rewards, observations, the height
> scanner, terminations, the curriculum — and only `scene.terrain.terrain_generator`
> is replaced. That is deliberate: **a reward tuned by NVIDIA for their terrain
> is a control, not a liability.**

**That premise is false, and this decision exists to correct it in the record.**
`config/anymal_c/rough_env_cfg.py` overrides **zero** reward weights. It swaps
the robot USD and nothing else. What we inherited is the framework's generic
`RewardsCfg`, identical for every robot that does not override it — not an
ANYmal-tuned artifact, and not a terrain-tuned one.

The premise is wrong in a second way too. NVIDIA *does* re-tune per terrain, for
the same robot: `anymal_c/flat_env_cfg.py` changes three reward weights against
`rough_env_cfg.py` with the robot held fixed. So "tuned for their terrain" is
not merely unsupported — the vendor's own configs demonstrate the opposite
practice.

This matters now because Hound is next, and getting the reward wrong costs days
per iteration.

## What we actually verified

All source read **2026-07-29**. Isaac Lab paths are on branch `main` unless a
tag is named.

### The inherited table (`velocity_env_cfg.py`, `RewardsCfg`)

Control frequency **50 Hz** (decimation 4, sim dt 0.005 → 200 Hz physics),
`episode_length_s = 20.0`.

| Term | Weight | Params |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | `std = sqrt(0.25)` |
| `track_ang_vel_z_exp` | +0.5 | `std = sqrt(0.25)` |
| `lin_vel_z_l2` | −2.0 | — |
| `ang_vel_xy_l2` | −0.05 | — |
| `dof_torques_l2` | −1.0e−5 | — |
| `dof_acc_l2` | −2.5e−7 | — |
| `action_rate_l2` | −0.01 | — |
| `feet_air_time` | +0.125 | `threshold = 0.5`, `body_names = ".*FOOT"` |
| `undesired_contacts` | −1.0 | `body_names = ".*THIGH"`, `threshold = 1.0` |
| `flat_orientation_l2` | 0.0 (off) | — |
| `dof_pos_limits` | 0.0 (off) | — |

Source: `https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

**Weights are per-second rates.** `managers/reward_manager.py` computes
`value = term_cfg.func(...) * term_cfg.weight * dt`, and its class docstring
says so explicitly: *"The reward manager multiplies the reward term's `weight`
with the time-step interval `dt` of the environment. This is done to ensure
that the computed reward terms are balanced with respect to the chosen
time-step interval."* Per-step contribution is therefore `weight × 0.02`.
legged_gym uses the identical convention (`self.reward_scales[key] *= self.dt`,
with `self.dt = decimation * sim_dt`), so numbers from the two stacks are
directly comparable.

The tracking kernel is `exp(−error / std**2)` — note the square. With
`std = sqrt(0.25)` this is `exp(−error / 0.25)`, matching legged_gym's
`tracking_sigma = 0.25`.

### ANYmal-C overrides nothing; Go2 and Spot override plenty

`config/anymal_c/rough_env_cfg.py` sets `scene.robot = ANYMAL_C_CFG` and
nothing else. The `_PLAY` subclass changes `num_envs = 50`, `env_spacing = 2.5`,
`num_rows/num_cols = 5`, `curriculum = False`, `enable_corruption = False`, and
disables `base_external_force_torque` and `push_robot` — **no reward weights.**

Same robot, terrain change only (`anymal_c/flat_env_cfg.py`):

| Term | Rough | Flat |
|---|---|---|
| `flat_orientation_l2` | 0.0 | **−5.0** |
| `dof_torques_l2` | −1.0e−5 | **−2.5e−5** |
| `feet_air_time` | 0.125 | **0.5** |

plus `terrain_type = "plane"`, `height_scanner = None`,
`observations.policy.height_scan = None`, `curriculum.terrain_levels = None`,
`max_iterations` 1500 → 300, hidden dims [512,256,128] → [128,128,128].

Different robot (`config/go2/rough_env_cfg.py`) — Go2 is Hound's kinematic base,
so this is the most transferable vendor datapoint we have:

| Term | Base | Go2 |
|---|---|---|
| `track_lin_vel_xy_exp` | 1.0 | **1.5** |
| `track_ang_vel_z_exp` | 0.5 | **0.75** |
| `dof_torques_l2` | −1.0e−5 | **−2.0e−4** (20×) |
| `feet_air_time` | 0.125 | **0.01**, `body_names = ".*_foot"` |
| `undesired_contacts` | −1.0 | **None (disabled)** |
| action scale | 0.5 | **0.25** |

`config/spot/flat_env_cfg.py` is a complete rewrite — 14 terms from a private
`spot_mdp` module, none shared with the base task: `gait` +10.0 (`std = 0.1`,
`max_err = 0.2`), `air_time` +5.0 (`mode_time = 0.3`),
`base_linear_velocity` +5.0 (`std = 1.0`), `base_angular_velocity` +5.0
(`std = 2.0`), `foot_clearance` +0.5 (`target_height = 0.1`),
`base_orientation` −3.0, `base_motion` −2.0, `action_smoothness` −1.0,
`air_time_variance` −1.0, `joint_pos` −0.7, `foot_slip` −0.5, `joint_vel` −1e−2,
`joint_torques` −5e−4, `joint_acc` −1e−4. Action scale **0.2**, decimation
**10**, sim dt **0.002**.

**So NVIDIA re-tuned two of the three quadrupeds they ship. The one they did not
re-tune is the one the reward was originally derived on.**

### The table has not moved in five years

legged_gym's base `scales` (`envs/base/legged_robot_config.py`):
`termination −0.0`, `tracking_lin_vel 1.0`, `tracking_ang_vel 0.5`,
`lin_vel_z −2.0`, `ang_vel_xy −0.05`, `orientation −0.`, `torques −0.00001`,
`dof_vel −0.`, `dof_acc −2.5e−7`, `base_height −0.`, `feet_air_time 1.0`,
`collision −1.`, `feet_stumble −0.0`, `action_rate −0.01`, `stand_still −0.`;
`only_positive_rewards = True`, `tracking_sigma = 0.25`, `base_height_target 1.`,
`max_contact_force 100.`

Term-for-term identical to Isaac Lab's, **except `feet_air_time` 1.0 → 0.125**.
Both formulas were read and are the same —
`sum(air_time − 0.5) × first_contact × [norm(cmd_xy) > 0.1]` — so the 8× is a
real re-weighting, not a formulation artifact.

Isaac Lab's `RewardsCfg` is byte-identical between tags `v2.3.2` (2026-02-02)
and `v3.0.0-beta2.patch1` (2026-07-02). Five years, a full physics-backend
rewrite, no weight moved. Every 2026 commit touching `velocity_env_cfg.py` is
plumbing (`ea8ecd6e` core/contrib split, `530fa8b3` "Rename Newton preset to
MJWarp", `1a040c0f` "[Newton] Add Rough terrain locomotion Part 1").

`legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py` has
`class scales: pass` — all base defaults. It overrides only
`base_height_target = 0.5`, `max_contact_force = 500.`, stiffness 80
{HAA,HFE,KFE}, damping 2., `use_actuator_network = True`
(`anydrive_v3_lstm.pt`, 2 layers, hidden 8).

`unitreerobotics/unitree_rl_gym` `GO2RoughCfg` overrides exactly **two** scales
— `torques = −0.0002`, `dof_pos_limits = −10.0` — plus
`base_height_target = 0.25`, `soft_dof_pos_limit = 0.9`, stiffness 20,
damping 0.5, `action_scale = 0.25`.

**UNVERIFIED:** legged_gym's A1 config was not fetched. No A1 numbers are
recorded here.

### The wheeled-legged table — our exact topology

`fan-ziqi/robot_lab`, `.../locomotion/velocity/config/wheeled/unitree_go2w/rough_env_cfg.py`.
Go2-W is 12 leg joints plus 4 hub wheels, and its wheel joints are literally
named `*_foot_joint` — the wheel replaces the foot, exactly Hound's topology.
Read twice, independently, and the two readings agree.

| Term | Weight | Scope / params |
|---|---|---|
| `track_lin_vel_xy_exp` | **+3.0** | `std = sqrt(0.25)` |
| `track_ang_vel_z_exp` | **+1.5** | — |
| `upward` | +1.0 | `(1 − g_b,z)^2` |
| `feet_contact_without_cmd` | +0.1 | wheels |
| `lin_vel_z_l2` | −2.0 | — |
| `ang_vel_xy_l2` | −0.05 | — |
| `joint_torques_l2` | −2.5e−5 | legs only |
| `joint_acc_l2` | −2.5e−7 | legs only |
| **`joint_acc_wheel_l2`** | **−2.5e−9** | **wheels only — 100× weaker** |
| `joint_power` | −2e−5 | legs only, `sum(abs(qd * tau))` |
| `joint_pos_limits` | −5.0 | legs only |
| `joint_pos_penalty` | −1.0 | legs only, ×5 when standing |
| `stand_still` | −2.0 | legs only |
| `joint_mirror` | −0.05 | FR↔RL, FL↔RR |
| `action_rate_l2` | −0.01 | — |
| `undesired_contacts` | −1.0 | `"^(?!.*_foot).*"` |
| `contact_forces` | −1.5e−4 | feet, 100 N |
| `feet_air_time` | **0** | disabled |
| `feet_slide` | **0** | disabled |
| `feet_stumble` | **0** | disabled |
| `feet_gait` | **0** | disabled |
| `feet_height` | **0** | disabled |
| `base_height_l2` | **0** | disabled |
| `flat_orientation_l2` | **0** | disabled |

Actions are **two terms**:
`joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}` over the
12 leg joints, and `joint_vel.scale = 5.0` over the 4 wheels, whose actuator is
`stiffness = 0.0, damping = 0.5` — pure velocity target.
`base_lin_vel = None` and `height_scan = None` on the policy branch; the critic
keeps both.

**It also carries a global upright gate** that multiplies roughly 27 reward
functions, velocity tracking included:
`reward *= clamp(-projected_gravity_b[:, 2], 0, 0.7) / 0.7`.
Upright → 1.0; past horizontal → 0.0. Combined with
`terminations.illegal_contact = None` and ±3.14 rad reset randomisation, this
trains fall-recovery inside the locomotion policy. **Copying the repo adopts
this silently.** It is a design commitment, not a detail, and it is close in
spirit to the multiplicative composition our MuJoCo track already uses.

### Independent corroboration on real hardware

Matsuzawa, Irie, Yoshida, Suzuki, Hara, Tomono, *"Long-Distance Real-World
Navigation of the Legged-Wheeled Robot Go2-W Using Deep Reinforcement
Learning"*, arXiv **2606.21387v1**, submitted **2026-06-19**
(`https://arxiv.org/abs/2606.21387`, full text `https://arxiv.org/html/2606.21387v1`).
Same robot class, 16 DoF, **2.8 km autonomous traverse** including sidewalks,
park and stairs. Reward Table 3, rough / flat:

| Term | Rough | Flat |
|---|---|---|
| `tracking_lin_vel_var` | 1.0 | 2.0 |
| `tracking_ang_vel` | 0.5 | 1.0 |
| `action_rate` | −0.01 | −0.015 |
| `ang_vel_xy` | −0.05 | −0.05 |
| `collision` | −1.0 | −1.0 |
| `dof_acc` | −2.5e−7 | −5e−7 |
| `dof_pos_limits` | −10.0 | −10.0 |
| **`leg_effort_std`** | **−0.01** | **−0.03** |
| `lin_vel_z` | −2.0 | −2.0 |
| `orientation` | 0 | −1.0 |
| `slip` | −0.1 | 0 |
| `torque_limits` | −0.01 | 0 |
| **`action_curvature`** | **−0.1** | **−0.5** |

Their finding is the most mission-relevant thing in this sweep: *"wheeled
locomotion concentrates the load on the hip joints and causes heat concentration
that hinders sustained travel, and obtained a policy that suppresses it by
distributing the load."* `leg_effort_std` is the deviation of smoothed torque
**across legs**; `action_curvature` penalises jerk, promoting periodic
foot-lifting. Temperature is never an observation: *"Temperature information was
not directly used in policy training, owing to the difficulty of incorporating
it into the learning process."*

**A 2.8 km unsupervised traverse failed on thermals, not on locomotion.** No
other reward table in this literature contains a term for it.

Wheel action scale is the one number the sources disagree on: **5.0**
(robot_lab), **4.0 rough / 6.0 flat** (Matsuzawa), **10.0**
(`ShengqianChen/go2w_rl_gym`). Wheel acceleration penalty likewise:
**−2.5e−9** (robot_lab, 100× weaker than legs) against **−1e−7** equal-to-legs
(`DreamWaQ_Go2W`). Both disagreements are recorded rather than resolved.

### Wheel joints: how the field handles an unbounded angle

Isaac Lab has **no observation function that wraps or normalises a continuously
rotating joint** — `envs/mdp/observations.py` offers `joint_pos`,
`joint_pos_rel`, `joint_pos_limit_normalized`, `joint_vel`, `joint_vel_rel`, and
none wraps to [−pi, pi] or encodes sin/cos. `joint_pos_rel` on a hub wheel
returns a monotonically growing number, and the base config applies no `clip` to
`joint_pos`.

Convergent practice, four code sources, is **deletion, not encoding**:
robot_lab and `go2w_rl_gym` zero wheel positions in place
(`joint_pos_rel[:, wheel_ids] = 0`, dimension preserved); LimX Tron1 slices them
out. Wheel *velocity* is always kept.

### The swap fails loudly, not silently — verified

`utils/string.py::resolve_matching_names` raises
`ValueError("Not all regular expressions are matched! Please check that the
regular expressions are correct: ...")` when a pattern matches nothing, and
`Articulation.find_bodies`/`find_joints` both `return
string_utils.resolve_matching_names(...)`. So the inherited `.*FOOT` and
`.*THIGH` patterns will **raise at env construction** against Go2-derived link
names, not silently zero the reward. Each link of that chain was read; the
composition was not executed, so this is inference from source, not a run.

### Training configurations, and where our machine sits

| Stack | Envs | steps/env | Iters | Total samples | GPU | Wall-clock |
|---|---|---|---|---|---|---|
| Rudin et al. 2021 (paper) | 4096 | 24 | 1500 | 1.47e8 | RTX A6000, i9-11900k | under 20 min rough; under 4 min flat |
| IL ANYmal-C rough | 4096 | 24 | 1500 | 1.47e8 | not stated | not stated |
| IL ANYmal-C flat | 4096 | 24 | 300 | 2.95e7 | not stated | not stated |
| IL Go2 rough | 4096 | 24 | 1500 | 1.47e8 | not stated | not stated |
| IL Spot flat | 4096 | 24 | 20000 | 1.97e9 | not stated | not stated |
| Extreme Parkour | 6144 | 24 | 50000 | 7.4e9 | not stated | not stated |
| robot_lab Go2-W rough | 4096 | 24 | 20000 | 1.97e9 | **UNVERIFIED** | **UNVERIFIED** |
| Matsuzawa Go2-W rough | **UNVERIFIED** | — | 40000 | — | **UNVERIFIED** | — |

**UNVERIFIED across the board: no wheeled-legged wall-clock is published by
anyone.** Matsuzawa states neither `num_envs` nor GPU.

PPO hyperparameters are identical everywhere — `clip 0.2, gamma 0.99,
lam 0.95, desired_kl 0.01, adaptive lr 1e−3, 5 epochs, 4 minibatches,
max_grad_norm 1.0`, MLP [512,256,128] ELU. Only `entropy_coef` moves: 0.01
(legged_gym, Go2, robot_lab), 0.005 (ANYmal-C), 0.0025 (Spot).

Measured on this machine (decision 0003, 2026-07-29): **7,630 steps/s at 1024
envs, 4,649 MiB peak VRAM** on `Bestiary-Desert-Coarse-Anymal-C-v0`. The
module docstring of `anymal_desert_env_cfg.py` records **13,520 steps/s** at
1024 envs on Isaac Lab's own `Isaac-Velocity-Rough-Anymal-C-v0` — the desert
costs roughly 44% of throughput.

Two consequences, both arithmetic over the measured rate and marked as such:

- Reproducing the published 1.47e8-sample budget at 1024 envs needs **6000
  iterations**, not 1500. At 7,630 steps/s that is about **5.4 hours**.
- The wheeled-legged budget (1.97e9 samples) is about **72 hours** at that rate.
  That is the real cost of the published Go2-W recipe on this machine.
- **4096 envs is out of reach.** 4,649 MiB at 1024 envs puts 4096 envs over the
  8 GB card even allowing for fixed overhead.

**Our batch is 4× under the published operating point.** Rudin et al. Figure 4
concludes *"using 2048 to 4096 robots with a batch size of ~100k or ~200k
provides the best trade-off"* with *"nearly linear scaling up to 4000 robots"*.
At 1024 envs × 24 steps our batch is **24,576**. Raising `num_steps_per_env`
from 24 to 96 restores **98,304** at a rollout-storage cost of roughly 97 MB
(96 × 1024 × 247 floats × 4 bytes). This is the cheapest lever available and it
does not touch the reward.

### Terrain curriculum — the reusable thresholds

`velocity/mdp/curriculums.py::terrain_levels_vel`, with
`distance = norm(root_pos_w[:, :2] − env_origins[:, :2])`:

- **Promote** when `distance > terrain_generator.size[0] / 2`. With
  `size = (8.0, 8.0)` — which `desert_terrain_cfg` matches via `TILE_M = 8.0` —
  the threshold is **4.0 m**.
- **Demote** when `distance < norm(command[:, :2]) * max_episode_length_s * 0.5`
  and not promoting: less than **50%** of the commanded distance over the 20 s
  episode.
- legged_gym additionally resamples a robot that solves the top level to a
  random level.

`TerrainGenerator` computes `difficulty = (sub_row + U[0,1)) / num_rows`, mapped
into `difficulty_range`. **Rows are difficulty; columns are terrain type**,
assigned by cumulative `proportion`. This is load-bearing for us:
`HfBestiaryDesertTerrainCfg` receives a `difficulty` argument, and **if it
ignores that argument every row is the same desert, the curriculum promotes and
demotes through identical ground, and `terrain_levels_vel` is inert.** That has
not been checked and is listed as a trigger below.

`ROUGH_TERRAINS_CFG` for reference: size (8,8), border 20.0, 10 rows × 20 cols,
h-scale 0.1, v-scale 0.005, slope_threshold 0.75; `pyramid_stairs` 0.2 and
`pyramid_stairs_inv` 0.2 (step 0.05–0.23, width 0.3, platform 3.0), `boxes` 0.2
(grid 0.45, height 0.05–0.2), `random_rough` 0.2 (noise 0.02–0.10, step 0.02),
`hf_pyramid_slope` 0.1 and `_inv` 0.1 (slope 0.0–0.4).

### Domain randomization (inherited `EventCfg`)

| Event | Mode | Range |
|---|---|---|
| `physics_material` | startup | static 0.8, dynamic 0.6, restitution 0.0, 64 buckets |
| `add_base_mass` | startup | −5.0 to +5.0 kg, add |
| `base_com` | startup | x/y ±0.05, z ±0.01 |
| `base_external_force_torque` | reset | force 0.0, torque 0.0 (off) |
| `reset_base` | reset | pose x/y ±0.5, yaw ±3.14; vel x/y/z ±0.5, rpy ±0.5 |
| `reset_robot_joints` | reset | position scale 0.5–1.5, velocity 0.0 |
| `push_robot` | interval 10–15 s | vel x/y ±0.5 |

### Observation and action spec

Policy group, in order, `enable_corruption = True`, `concatenate_terms = True`:
`base_lin_vel` 3 (noise ±0.1) · `base_ang_vel` 3 (±0.2) · `projected_gravity` 3
(±0.05) · `velocity_commands` 3 · `joint_pos_rel` 12 (±0.01) · `joint_vel_rel`
12 (±1.5) · `last_action` 12 · `height_scan` 187 (±0.1, clip (−1, 1)).
**Total 235**, matching legged_gym's `num_observations = 235` exactly.

**Height scan is 187 points, a 17 × 11 grid.** From
`sensors/ray_caster/patterns/patterns.py::grid_pattern`:
`torch.arange(start=-size/2, end=size/2 + 1.0e-9, step=resolution)` with
`resolution = 0.1` and `size = (1.6, 1.0)` gives x in [−0.8, 0.8] (17 points)
and y in [−0.5, 0.5] (11 points). Scanner z-offset 20.0 m. legged_gym's
`measured_points_x/y` are the identical literal grid.

For Hound at 16 DoF the same layout gives **3+3+3+3+16+16+16+187 = 247** — but
robot_lab and Matsuzawa **both** make the deployed policy blind and give terrain
to the critic only. **That choice is a one-way door under this repo's own
invariant** (`envs/obs_spec.py`; the actor's first layer is `Linear(obs, 256)`),
so it must be locked before any multi-hour Hound run, not during one.

Action scales across stacks: Isaac Lab base 0.5 · Go2 0.25 · Spot 0.2 ·
unitree_rl_gym Go2 0.25 · robot_lab Go2-W legs 0.125 hip / 0.25 other,
wheels 5.0 (velocity).

PD gains: legged_gym ANYmal-C Kp 80 {HAA,HFE,KFE} / Kd 2.0 · unitree_rl_gym Go2
Kp 20 / Kd 0.5 · Matsuzawa Go2-W Kp 50 rough, 40 flat / Kd 1.0 · robot_lab
Go2-W wheels Kp 0.0 / Kd 0.5. **Isaac Lab's own ANYmal-C and Go2 PD gains are
UNVERIFIED** — they live in the actuator cfg (`ANYMAL_C_CFG`, `UNITREE_GO2_CFG`)
which was not read.

### Two mechanics worth knowing before porting

- Isaac Lab's `ActionManager` concatenates multiple action terms and slices the
  incoming tensor in config declaration order
  (`term_actions = action[:, idx : idx + term.action_dim]`,
  `total_action_dim = sum(action_term_dim)`), with no restriction on terms using
  different control modes over disjoint joint sets. Hound's 16-D action is
  therefore a 12-D `JointPositionActionCfg` plus a 4-D `JointVelocityActionCfg`,
  structurally supported.
- legged_gym's `only_positive_rewards = True` clips the per-step total at zero.
  **Isaac Lab has no equivalent.** Any weight ported from a legged_gym-lineage
  config arrives with its negatives unclipped.

## The trigger to revisit

Reopen when **any** of these becomes true:

1. **Hound's USD lands and the env is constructed.** The inherited `.*FOOT` and
   `.*THIGH` patterns raise a `ValueError` at construction. That exception is
   the trigger — it is scheduled, not hypothetical, and Part B is what answers
   it.
2. **`HfBestiaryDesertTerrainCfg` is checked against its `difficulty`
   argument.** If it ignores `difficulty`, the terrain curriculum is inert on
   the desert and `terrain_levels_vel` has been promoting through identical
   ground for every run so far. Cheap to check: read the sub-terrain function,
   no GPU.
3. **Anyone publishes a wheeled-legged reward-coefficient ablation.** None
   exists today. The specific weights in Part B are convention; one ablation
   would move them from convention to evidence.
4. **The wheel action scale is derived from geometry rather than copied.** The
   published range is 4.0–10.0 across three sources. The value that makes
   `a = 1` correspond to maximum commanded ground speed divided by Hound's wheel
   radius collapses that range analytically. When that number is computed, this
   decision's deference to robot_lab's 5.0 is superseded.
5. **`num_steps_per_env` is raised to 96 and the batch reaches ~98k.** If
   behaviour changes materially at the same iteration count, then batch size was
   confounding every Isaac-side comparison in the record so far, and the
   training-config half of this decision needs restating.

## What we gave up

**We gave up the claim that the ANYmal-C arm is a reward control.** It is still
a *terrain* control — same reward, same robot, different ground, which is the
comparison `anymal_desert_env_cfg.py` was actually built for and which survives
intact. What it is not is evidence that the reward suits ANYmal-C, because
nobody tuned it for ANYmal-C.

**We gave up starting Hound from our own multiplicative reward.** The MuJoCo
track has a hand-derived reward with real measurement behind it —
`research/measurements/tracking_noise.json`, the Cauchy kernel, sigma_v 0.15 and
sigma_w 0.1 — and Part B adopts a foreign additive table instead. That discards
accumulated tuning, exactly as decision 0003 discarded it for the algorithm.
The mitigation is that the two are not exclusive: a product of terms in [0,1] is
one `RewardTerm` whose function computes the product internally, since the
manager sums whatever terms it is given. Porting the multiplicative reward is
therefore a later, separable step, not a fork in the road.

**We gave up, for now, on the height scan for Hound.** Both wheeled sources go
blind-with-privileged-critic, and following them costs us the 187-point scan the
desert was built to exercise. That is a real loss on a 5.05 m relief terrain and
it is deferred rather than settled.

## How we would know this was wrong

- **The inherited table trains Hound fine once the body patterns are fixed.** If
  re-scoping `.*FOOT` to Hound's wheel links and leaving all eleven weights
  alone produces a policy that tracks commands on the desert, then Part B bought
  nothing and the convergent deletion of `feet_air_time` across four
  repositories was cargo cult after all. This is cheap to test and is the single
  experiment that would most improve the record, because **no published work
  runs the control arm.**
- **`joint_acc_wheel_l2` at −2.5e−9 turns out not to matter.** robot_lab and
  `DreamWaQ_Go2W` differ by 100× on this term. If a two-arm run at fixed seed
  shows no difference in tracking error or wheel-torque RMS, then the term is
  decorative and the disagreement recorded above was never worth recording.
- **The upright gate is what actually makes robot_lab's table work.** If Part B
  is adopted without the gate and the policy will not learn, then the
  transferable object was never the weights — it was the multiplicative posture
  factor, which is the thing our MuJoCo track already had and this decision
  proposed to set aside.
- **Thermals never bite in simulation.** `leg_effort_std` addresses a failure
  Matsuzawa found on hardware after 2.8 km. If Hound never leaves simulation,
  the term is unfalsifiable here and including it is faith, not evidence. It
  earns its place only when a run is long enough, or a motor model detailed
  enough, for load concentration to show up in a number we measure.
- **The whole comparison is confounded by batch size.** Every Isaac-side number
  in this record so far was taken at a batch of 24,576, four times under the
  published operating point. If raising `num_steps_per_env` to 96 changes
  outcomes materially, then reward differences and batch differences have been
  entangled in everything above, and the tables here describe two variables
  rather than one.
