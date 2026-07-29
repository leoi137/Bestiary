# 0005 — The Isaac Hound stack: what survives an adversarial pass

**Date:** 2026-07-29 · **Status:** accepted · **Robot:** hound

## The decision

A design pass over the Isaac Lab locomotion stack for Hound-16 was written and
then attacked by an independent refutation pass. **This decision records what
survived, not what was designed.** Two things gate the next run and both are
open, so nothing in the reward table may be launched yet:

1. **The reward-composition choice — additive vs. product — is VOID pending
   measurement.** Both published grounds for preferring the additive form are
   dead. Ground 1 (the product's dead-gradient pathology) does not occur under
   the kernel width the same design adopts. Ground 2 (command geometry replaces
   the freeride fence) rests on a comparison that differs on three axes at once,
   and the additive form gives back roughly the whole claimed gain.
2. **`lin_vel_y ∈ ±1 m/s` is physically unachievable for this morphology**, and
   it is the dominant term in the inherited reward. Four wheels with fixed +Y
   spin axes cannot produce sustained body-frame lateral velocity. Roughly 80%
   of the inherited 2-D command square asks for motion the machine cannot make,
   which puts an unremovable ceiling on `track_lin_vel_xy_exp` and inflates
   every freeride figure computed against it.

Everything else recorded below is either **verified against a named source**
(safe to build on) or **corrected** (the corrected form is what the record
carries).

Confidence, split by kind of claim, because they are not equally strong:

| claim class | confidence | why |
|---|---|---|
| source-read verifications (§What we verified, part A) | **high, ~95%** | read from named files at named lines; several reproduced from the module's own code |
| `lin_vel_y` is unachievable, qualitatively | **high, ~95%** | follows from fixed wheel axes, isotropic `condim=3` friction, and the CARD's own "a wheel rolls fore-aft but *grips* sideways" |
| the composition decision is VOID | **~85%** | the two grounds are refuted by arithmetic; the *conclusion* (ship additive) may still be right on convention grounds, which is why this says VOID rather than "ship the product" |
| the corrected maneuver laws | **~80% on form, 0% on any numeric margin** | the algebra is re-derivable; every number in it still hangs on an assumed knee Jacobian and an unfetched motor speed limit |
| the terrain-traversability finding | **~75%** | computed from the committed heightfield, but slope-vs-traction is a static criterion and a legged machine can step |

## Why we asked

`0004` inherited the Isaac reward table knowingly and left Part B — the
re-scoping for a wheel-legged body — as prose. The design pass was asked to turn
that prose into a stack: composition, weights, batch geometry, VRAM, and the
maneuver/payload/inverted capability envelope. Because it would set the shape of
the first multi-hour Isaac run and hand hardware constraints to a print effort,
it was attacked before any of it was allowed into the record.

The attack found enough that the honest output is a decision about *what may be
built on*, not an implementation plan.

## What we actually verified

Two parts. Part A is read from a named source and is safe. Part B is arithmetic
this pass performed; **it was computed in a scratch script that is not
committed, so under the number rule none of Part B's figures may be cited
downstream until a script lands under `research/scripts/`.** Each Part B block
names the script that would satisfy the rule.

### Part A — read from a named source

**The composition grounds, and the source lines that kill them.**

The three "stander fences" proposed to compensate for the additive form's
conceded pathology are all gated on a near-zero command, and the pathology is
*standing while commanded to move*. Read from the installed Isaac Lab tree:

- `stand_still_joint_deviation_l1` returns
  `joint_deviation_l1(...) * (torch.linalg.norm(command[:, :2], dim=1) < command_threshold)`
  with `command_threshold = 0.06`
  (`isaaclab_tasks/.../locomotion/velocity/mdp/rewards.py:117-123`).
- Spot's `joint_position_penalty` applies `stand_still_scale` only via
  `torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)`
  (`.../velocity/config/spot/mdp/rewards.py:266-274`).
- `feet_contact_without_cmd` pays only when there is no command
  (`0004:181`).

So all three are **exactly zero** in the regime they were shipped to fence.
This is a source fact, not an estimate.

**The tracking kernels.** `track_lin_vel_xy_exp` is
`exp(-sum(square(cmd[:, :2] - root_lin_vel_b[:, :2])) / std**2)` and
`track_ang_vel_z_exp` is `exp(-square(cmd[:, 2] - root_ang_vel_b[:, 2]) / std**2)`
(`isaaclab/envs/mdp/rewards.py:314-338`), with `std = math.sqrt(0.25)` so
`std**2 = 0.25`. Both are **body-frame**, not heading-frame. The lin term's
exponent sums `e_x² + e_y²`, which is what makes the 2-D freeride value the
square of the 1-D one.

`lin_vel_z_l2` is `square(root_lin_vel_b[:, 2])` — **body-frame** z
(`isaaclab/envs/mdp/rewards.py:77-81`). On a slope the body pitches with the
terrain, so the worry that this term charges unavoidable terrain-following is
materially smaller than the design pass claimed. Still measure it; it is not the
term to worry about most.

`undesired_contacts` returns an **integer count** of bodies whose max contact
force exceeds the threshold (`isaaclab/envs/mdp/rewards.py:272-282`). One
contacting body at weight −1.0 is a full −1.0 per step, not a small number.

**Commands.** `UniformVelocityCommandCfg` with `lin_vel_x=(-1,1)`,
`lin_vel_y=(-1,1)`, `ang_vel_z=(-1,1)`, `heading=(-pi,pi)`,
`heading_command=True`, `rel_heading_envs=1.0`, `heading_control_stiffness=0.5`,
`rel_standing_envs=0.02`, `resampling_time_range=(10.0, 10.0)`
(`velocity_env_cfg.py:140-151`). Because `rel_heading_envs=1.0`, the sampled
`ang_vel_z` is **always discarded** and replaced by
`clip(0.5 * wrap_to_pi(heading_target - heading_w), -1, +1)` — so
`ang_vel_z=(-1,1)` acts only as the clip bound on a heading-derived rate.

**Domain randomization — the design pass's sharpest correction, and it holds.**
`add_base_mass` is
`mass_distribution_params=(1/1.25, 1.25), operation="scale", distribution="log_uniform"`
on `body_names="base"`, `mode="startup"` (`velocity_env_cfg.py:212-224`). It is
multiplicative on the base body, **not** the additive −5..+5 kg that `0004:364`
records. Two things the design pass missed and this record should carry:
`randomize_rigid_body_mass` defaults `recompute_inertia=True`, so all nine
inertia components are rescaled by the same mass ratio
(`isaaclab/envs/mdp/events.py:546-559`); and `base_com` shifts the CoM
separately **without** re-deriving inertia about the shifted CoM.

`physics_material` is `static_friction_range=(0.8, 0.8)`,
`dynamic_friction_range=(0.6, 0.6)`, `restitution_range=(0.0, 0.0)`
(`velocity_env_cfg.py:200-210`) — **degenerate. Friction is not randomized at
all**, and `num_buckets=64` is inert. This strengthens the design pass's own
low-μ gap: the gap is not "the range is narrow", it is "there is no range".

`base_external_force_torque` has all-zero ranges — a no-op placeholder
(`velocity_env_cfg.py:239-247`). `reset_robot_joints` `position_range=(0.5, 1.5)`
is a **multiplicative** scale on the default joint position
(`events.py:1924-1962`), so any joint whose default is 0 is unrandomized by
construction.

**The curriculum, and why the demote criterion cannot fence a one-gait policy.**
`terrain_levels_vel` (`.../velocity/mdp/curriculums.py:42-54`):

```python
distance = torch.linalg.norm(root_pos_w[env_ids, :2] - env_origins[env_ids, :2], dim=1)
move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
move_down *= ~move_up
```

Four facts the design pass's "promotes at 0.20 m/s" claim needs and mostly had:

- The 4.0 m threshold is **terrain-derived**, `size[0]/2`, not a constant. It is
  4.0 m for us because `desert_terrain_cfg` sets `TILE_M = 8.0`
  (`src/bestiary/isaac/anymal_desert_env_cfg.py:58,71`).
- `distance` is a **displacement norm from the env origin**, not path length. A
  circle scores ~0, and `reset_base` spawn jitter of ±0.5 m in x and y
  (`velocity_env_cfg.py:253`) already gives up to 0.707 m for free.
- `episode_length_s = 20.0`, `decimation = 4`, `sim.dt = 0.005`
  (`velocity_env_cfg.py:360-366`) → 0.02 s steps, 1000 steps, so the promote
  threshold as a mean speed is 4.0/20 = **0.20 m/s**.
- **`move_down *= ~move_up` means promotion strictly wins.** The demote
  threshold for a mean command is larger than the distance a 0.27 m/s gait
  covers, so demote *would* fire — and is then zeroed. **The demote criterion
  cannot fence the one-gait exploit; it is dominated by the promote branch.**
  The real gate is survival, not steering: the same gait that terminates before
  ~14.8 s falls below 4 m and demotes.

Curriculum terms are computed **only on reset, for terminating envs**, as the
first statement in `_reset_idx`, before `scene.reset` and before
`command_manager.reset` — so `root_pos_w` is the terminal pose and `command` is
the finished episode's last command
(`isaaclab/envs/manager_based_rl_env.py:362-369`).

**`HfBestiaryDesertTerrainCfg` honours `difficulty`.** `isaac_hf.py:232-244`:
difficulty indexes an std-ranked patch list (flattest first,
`_rank_patches_by_roughness`, `isaac_hf.py:167-195`) and additionally
interpolates `z_gain_range`, whose shipped default `(1.0, 1.0)` makes that
second path a no-op. Patch size is derived from the tile size at call time
(`isaac_hf.py:217-218`), stride defaults to 32 native cells
(`isaac_hf.py:282-285`), and the ranking is cached per
`(path, patch_x, patch_y, stride)` (`isaac_hf.py:94,177`). Executing the
module's own code at its own `__main__` parameters (8.0 m → 102 cells, stride
32, 841 candidates) reproduces the ladder **0.1096 / 0.3580 / 0.6994 / 0.9325 /
1.3779 m** at difficulty 0 / .25 / .5 / .75 / 1. **The curriculum is live.**

One caveat that matters and was not stated: the shipped desert sub-terrain sets
`border_width=0.25` (`anymal_desert_env_cfg.py:88`), and `isaac_hf.py:214-216`
documents that `cfg.size` has already been shrunk by the decorator. So the
**training tiles are 7.5 m → 96 cells**, whose ladder is
0.0919 / 0.3457 / 0.6810 / 0.9311 / 1.4169 m with entirely different corners.
The published ladder is the self-check's, not the training tiles'.

**Actions are unclipped, and it does not matter.** `clip_actions: float | None = None`
by default, applied as `if self.clip_actions is not None: torch.clamp(...)`
(Isaac Lab's rsl_rl `vecenv_wrapper.py:37,182-183`; `rl_cfg.py:274`);
`ActionTermCfg.clip` is `None` and the velocity env never sets it; and
`JointAction.process_actions` would clip the **processed target** in radians,
not the raw action (`isaaclab/envs/mdp/actions/joint_actions.py:169-178`). So
nothing attenuates the target. The attenuation is **torque saturation** inside
the actuator model. The design pass's derivation already assumes that
saturation, so it is internally consistent — but the headline should read
*"the wheel action scale sets units and exploration noise; authority is capped
downstream by effort saturation, not by the scale."*

**The τ–ω model named as the cure is dead code on the robot that anchors every
measurement.** `DCMotor._clip_effort` implements the four-quadrant curve
(`isaaclab/actuators/actuator_pd.py:295-307`), but ANYmal-C uses
`ANYDRIVE_3_LSTM_ACTUATOR_CFG` (`isaaclab_assets/robots/anymal.py:122`), and
`ActuatorNetLSTM.compute` (`isaaclab/actuators/actuator_net.py:77-98`) **never
assigns `self._joint_vel`**, which `DCMotor.compute` does. With `_joint_vel`
left at its allocated zeros the envelope collapses to a flat
`±effort_limit = ±80 N·m`. Consequence for the plan: **the A0 arm cannot be a
one-variable control.** ANYmal-C's actuator cannot port to Hound, so A0 changes
the body **and** the actuator model **and** the reward body-name regexes.

**Hound's own numbers, from committed source.** `build.py`: `hip_range = (-1.20, 2.60)`,
`knee_range = (-2.60, -0.60)` (one-way fold), `abduct_range = (-0.80, 0.80)`,
`hip_x = 0.1934`, `thigh_len = 0.213`, `calf_len = 0.190`, `wheel_r = 0.085`,
`trunk_half = (0.1881, 0.04675, 0.057)`, link masses 0.678 / 1.152 / 0.241 /
0.45 and trunk 6.921 → **17.005 kg**, gears 23.7 / 23.7 / 40.0 / 3.0,
`stance_hip = 0.75` with the knee **solved** to put the axle under its hip
pivot. The trunk is a box geom with no `pos` and no `<inertial>`, so **its CoM
is exactly at the body origin.** The wheel joint carries `damping = 0.05`,
`armature = 0.004`, and `frictionloss = 0.133 × gear_wheel = 0.3990`, which
`build.py:602-610` states is *"partly hardware, partly numerics"* — chosen so
noisy resets stay upright.

**The Hound CARD contains no wheel top speed and no motor τ–ω model of any
kind.** The MuJoCo `<motor>` is an ideal torque source: 3.0 N·m at any ω. The
only τ–ω figures in the repository belong to *Whelp*, a different machine, and
must not be transferred.

**Storage layout (flagged provenance).** rsl_rl 5.0.1's `RolloutStorage`
allocates, per `(step, env)`: one float32 leaf per **obs group present** (all of
them, regardless of `obs_groups`), `actions` float32, `dones` **uint8**, and
`rewards`, `values`, `actions_log_prob`, `returns`, `advantages` as five
float32 scalars, plus a lazily allocated `distribution_params` tuple which for a
Gaussian is **two** full-shape tensors `(mean, std)` — not one `mu`/`sigma`
pair. `mini_batch_generator` flattens with views but the mini-batch index is
advanced indexing, which **copies**. This was read from the installed rsl_rl
source, which sits **outside the two-repo boundary**; it is recorded here as
read-once and **should be re-verified from an in-scope copy before anything is
built on it.**

### Part B — arithmetic performed by this pass (not yet committed)

Every figure in this section was computed. **None of it satisfies the number
rule yet.** What would satisfy it: one script,
`research/scripts/isaac_hound_stack_arithmetic.py`, containing §B1–§B6 below and
printing each figure it asserts.

**B1 — the composition arithmetic.** With `std = 0.5` and independent uniform
`c_x, c_y ∈ [-1,1]`, the best command-blind fixed velocity earns
`F = 0.4409` per axis, so the 2-D value is `0.4409² = 0.1944`. Under heading
mode a standing policy's yaw command is `clip(0.5·U(-π,π), ±1)`, and it earns
`F_w = 0.2876`. Then, on the *same* command geometry:

| composition | stander earns, fraction of nominal max |
|---|---|
| additive, weights 3.0 / 1.5 | **0.2255** |
| product `k_v · k_w` | **0.0559** |

**Additive is 4.03× more fakeable than the product it replaces**, on the exact
command set offered as the replacement fence. And the claimed
`0.764 → 0.194` fence improvement is not like-for-like: 0.764 is 1-D,
one-sided, command-scaled α, over the *trained* distribution. Like-for-like on
dimensionality alone gives 0.764 → 0.441 = 1.73×. Against the record's own
two-sided number — `learnings/015`'s six-cell grid at **F = 0.411**, which
`docs/theory/reward-composition.md:588-592` had already named as the right
comparison — the improvement is 2.11× before correcting for achievability, and
**1.15× after** (see B2). Not 4×.

`reward-composition.md:601-605` states that the F table has no committed script
and *"may not be cited downstream until one exists"*. That still holds; 0.4409,
0.1944, 0.2876, 0.2255 and 0.0559 are all inside that fence.

**B2 — the `lin_vel_y` ceiling.** If `v_y ≡ 0` is the only achievable lateral
behaviour, the best attainable value of `track_lin_vel_xy_exp` is
`E_c[exp(-(c_y/0.5)²)] = 0.4409` — **56% of the largest reward term's income is
unattainable at any competence.** That is `learnings/011`'s
punished-for-unremovable-error shape one level up, sitting on the term the whole
table is built around. Correcting both sides for achievability: the additive
stander gets `0.2255 / 0.6273 = 0.359` of what a perfect fixed-wheel-axis
tracker can get, against the record's like-for-like 0.411 — the 1.15× above.
Narrowing `lin_vel_y` to ±0.2 raises the 2-D freeride figure to 0.419, which
changes the composition arithmetic again. **This must be settled before the
table is fixed.**

**B3 — flooring, which survives and is what kills ground 1.** With α = 0.5 and
commands bounded by `|c_v| ≤ √2`, `|c_w| ≤ 1`, the worst standing case gives
`k_v = 3.355e-4`, `k_w = 0.0183`, so even the **product** is `6.14e-6` with
`∂/∂e_v = 6.95e-5` — seven orders above the `1e-12` threshold at which the
record measured 60.8% of early healthy steps. The measured underflow came from
`α_w` as small as 0.10 against **unbounded** rate error (`u_w` up to 40.23). The
design pass says exactly this in its own flooring paragraph. **So ground 1 for
abandoning the product does not survive the width the same design adopts.** No
floor is needed under either composition.

**B4 — the corrected maneuver laws.** The published form omits gravity's
contribution to the ground-reaction impulse. Gravity acts at the CoM and exerts
no moment there, so the angular impulse about the CoM is
`b · ∫F_GRF dt = b·M·(v_z + g·Δt)`, not `b·M·v_z`. Carrying that through the
same constant-force model with `Δt = v_z / (g(β−1))`, the `(β−1)` **cancels**:

```
theta_flight = 4·b·d·beta / k^2                (not 4·b·d·(beta-1)/k^2)
beta_req     = max(1,  pi·k^2 / (2·b·d))       (no  "+1")
v_z_req      = sqrt( pi·g·k^2·(beta-1) / (b·beta) )
omega_knee   >= v_z_req / rho(q),  rho(q) = |x_axle - x_knee| = |L2·sin(t1+t2)|
```

Three further corrections of the same law:

- **A backflip uses the front pair.** With x forward, y left, z up, a vertical
  force at the **rear** contact gives `M_y = +bF`, which is nose-down — a
  *front* flip. Nose-up needs the front contact and the arm
  `b_front = 0.2172 m`, 28% larger than `b_rear = 0.1696 m`.
- **`rho` is a Jacobian, not `L2/2`.** By virtual work the force arm and the
  velocity arm are the same number. Along the axle-under-hip stroke it runs
  0.1894 → 0.1452 (the CARD stance) → 0.0594 m at full extension. `L2/2 = 0.095`
  is not the stance value and not any mid-stroke value; it corresponds to a
  nearly-extended, nearly-singular leg with almost no travel left.
- **The assumed stroke is unreachable.** `d = 0.325 m` requires a 0.06 m
  axle-under-hip crouch, which needs `knee = -2.865 rad` against a **−2.6
  limit**. The true axle-under-hip stroke is **0.2725 m** (0.1125 → 0.3850).

**Because margin is linear in `rho`, ±30% on it spans pass and fail:** at
ω_max = 30 rad/s the backflip speed margin is 0.75× at ρ = 0.0665, 1.07× at
0.095, 1.39× at 0.1235, 1.63× at the stance Jacobian. ω_max = 24 / 30 / 36 at
ρ = 0.095 gives 0.85× / 1.07× / 1.28×. **The record carries no numeric margin
until `rho` is swept over the stroke and `omega_max` comes off a vendor sheet.**
The jump apex is affected the same way: the correct simultaneously-feasible
takeoff is `v_z = 4.11 m/s`, apex **0.86 m** — and the design pass's own
scratch script computed 4.14 m/s / 0.87 m while its report published 0.41 m.
Consequence: *"jumps are bought with knee speed, not torque; a lighter robot
gains almost nothing in apex"* does **not** survive — the torque and speed
envelopes cross mid-stroke, so mass matters at the margin.

**B5 — the payload law.** The static knee torque is **affine**, not
proportional: `tau = 0.3561·p + 5.2425` N·m, with a **−0.813 N·m offset** that
is the leg links' own weight and does not scale with payload. So the record
carries:

```
tau_knee(p) = (r_kc/4)·(M + p)·g  -  tau_0
r_kc = 0.1452 m   at stance_hip = 0.75    [POSTURE-DEPENDENT, not geometry]
tau_0 = 0.813 N m                          [leg links' own weight]
```

`c_k = tau(0)/(M·g) = 0.03143` is not a shape constant: it drifts to 0.02939 at
M = 12 kg (−6.5%) and 0.02594 at M = 8 kg (−17.4%), because it bundles the fixed
offset divided by `M·g`. Worse, `r_kc = L1·sin(theta_hip)` is a **control
choice** — and one the design pass frees further by recommending the stance
springs be stripped. Across `stance_hip` 0.40 → 0.95 the implied capacity swings
4.7×, and it trades directly against the wheelie margin
(`da/dh = -g·b/h² = -18.4 (m/s²)/m`): standing tall to carry more costs ~23% of
`a_max`. **No `p_max` number enters the record while the motor thermal curve is
open.** The assumed `tau_cont/tau_peak = 1/3` alone swings the answer from
+1.54 M (at 1/2, σ 1.5) to **−0.05 M** (at 1/4, σ 2.0) — the last meaning the
machine could not hold its own weight continuously.

**B6 — terrain traversability, and the reward budget.** Slope statistics of the
selected desert patches, measured at Isaac's 0.1 m grid, against the traction
cap `atan(mu = 0.9) = 42.0°`:

| difficulty | std (m) | median slope | p90 | fraction above 42° |
|---|---|---|---|---|
| 0.00 | 0.110 | 2.7° | 31° | 3.5% |
| 0.25 | 0.358 | 18.4° | 35° | 3.2% |
| 0.50 | 0.699 | **38.8°** | 57° | **42.6%** |
| 0.75 | 0.932 | **43.1°** | 69° | **51.3%** |
| 1.00 | 1.378 | **39.7°** | 64° | **46.6%** |

(the 7.8 cm native and 0.17 m wheel-diameter scales agree within 3°.) Above
difficulty 0.25 the median tile slope is at or past the traction limit and about
half the tile is unclimbable by a rolling wheel. **Expect the terrain level to
equilibrate near 0.25 regardless of policy quality** — which changes what the
curriculum measures and weakens "std up to 1.38 m" as a capability statement.

And the budget check offered as the reconciliation with `011` and `015` omits
the two terms most likely to reproduce them. At the design pass's own income
figure of 0.0331/step (e = 0.5 m/s, dt = 0.02):

| term omitted from the budget | per step | % of income |
|---|---|---|
| `undesired_contacts` −1.0, **one** body in contact | 0.0200 | **60.4%** |
| `dof_acc_l2` −2.5e−7, 16 joints at 500 rad/s² rms | 0.0200 | **60.4%** |
| `dof_acc_l2` at 250 rad/s² rms | 0.0050 | 15.1% |
| `joint_pos_limits` −5.0, one joint 0.1 rad past soft limit | 0.0100 | 30% |

`015` measured control cost at 62.3% of income. One contacting thigh reproduces
it. **The claim "all penalties ≤ 1.5% of income" is a three-of-eight
accounting.**

Two more Part B numbers the record should carry as corrections:

- Wheel reaction-wheel authority: the stator reaction on the body **is** the
  applied torque, so `dL = n·tau·dt = 4 × 3.0 × 0.3 = 3.60 kg·m²/s`, giving
  **5.38 rad/s** of body pitch — not the 1.0 kg·m²/s and 1.5 rad/s published,
  which scale `4·tau·dt` by `I_disc/(I_disc + armature) = 0.289`.
- The "low splayed" inverted pose is not a locomotion result. It puts the belly
  **2.8 cm** off the ground and consumes **15.58 N·m of the 23.7 N·m hip
  gear — 66%, statically, unloaded.**

### What the inverted-stance "reversal" actually was

There was no reversal. The earlier session required 17 cm of belly clearance and
found the inverted stance infeasible; the later one required 2 cm and found it
feasible. Re-running the later script confirms it prints `unstable` on **every**
row at `h ≥ 0.21` — the same sign and magnitude as the earlier verdict. **Both
computations agree that the inverted stance is infeasible at any clearance
useful for driving, and the record should not carry a correction notice.**

The design recommendation to the print effort — *"+0.55 rad of hip range or a
two-way knee"* — is unsupported by the computation cited for it: that script's
closing sweep prints an identical `max x_leg = +0.3754` for `hip_hi` of 2.6,
2.8, 3.0 and π, because it tests the low clearance. Computed at 17 cm clearance
the two options are **not** equivalent: extending the hip to π reaches a front
contact of +0.086 m (marginal), a two-way knee reaches +0.274 m (comfortable).
The direction is right; the equivalence is not.

## The trigger to revisit

Reopen when **any** of these becomes true:

1. **`lin_vel_y` is measured.** A zero-command lateral-push rollout, or a
   commanded pure-`v_y` rollout, on the Isaac desert — minutes, no training —
   reports the largest sustained body-frame `|v_y|` the machine can hold. That
   number sets the command range, and the command range sets every F figure in
   B1–B2. **This is the cheapest gate and it comes first.**
2. **The composition is measured rather than argued.** Two 1-seed probes on the
   same geometry, additive vs. product, 5.4 h each at the measured 7,630
   steps/s, judged on `vx_span_ratio` and a per-cell grid rather than on return.
   Whichever wins, this decision is superseded and the F arithmetic becomes a
   prediction that was scored.
3. **Hound's own VRAM footprint is measured at N=1024, T=24.** The `4,649 MiB`
   figure is a one-seed, five-iteration probe of a *different* robot — 12 DoF,
   13 bodies, 4 point feet, carrying an LSTM actuator net Hound cannot use — and
   no instrument for it is recorded anywhere in either tree. Until Hound's own
   number exists, no T=96 projection is a floor.
4. **A motor τ–ω and continuous-torque curve exists.** It reopens every
   maneuver number and the payload law simultaneously.
5. **`rho(q)` is swept.** One MuJoCo statics/Jacobian sweep over the
   axle-under-hip stroke replaces the assumed scalar and settles whether any
   flip closes.
6. **The terrain level is observed to climb past 0.25 and stay there.** That
   would falsify B6's traversability argument and mean a legged strategy is
   beating the static slope criterion — itself the most interesting result
   available from the first run.

## What we gave up

We gave up shipping a reward table this cycle. The design pass produced one, and
it is not launchable: its composition ground is void, its stander fences are
inert where they are needed, its dominant term has an unremovable 56% ceiling,
and its budget check omits the two terms that reproduce our two known failures.
Fixing it needs one measurement (`lin_vel_y`) and one comparison (composition),
not more argument.

We also gave up the maneuver envelope as a *quantified* thing. What survives is
the algebra: which quantities matter (`k²/b`, the stroke, the knee Jacobian,
the motor speed limit) and which pair does which flip. What does not survive is
every number, because they all hang on two assumed scalars and one of them spans
pass and fail under a 30% perturbation. Handing either to a print effort in
numeric form would be a decision that cannot be walked back.

And we gave up treating `0004`'s DR table as current. It records `add_base_mass`
as additive −5..+5 kg; the installed tree is multiplicative `(1/1.25, 1.25)` on
the base body. A later cycle should append that to `0004` rather than edit it.

## How we would know this was wrong

- **`v_y` turns out to be achievable to ±0.5 m/s or better** by abduction
  shuffling on the desert. Then the inherited command range is not absurd, the
  0.441 ceiling is wrong, and B1–B2's whole reframing goes with it.
- **The additive table trains a command-following policy anyway**, with
  `vx_span_ratio ≥ 0.5` across three seeds. Then the freeride arithmetic was
  predicting a pathology that the entropy bonus, the curriculum and the
  termination set already prevent, and F is a weaker predictor than this
  decision assumes.
- **The product form fails to train at all** under α = 0.5 with bounded
  commands, despite B3 showing its gradient is alive. That would mean the
  dead-gradient story was never the mechanism and something else — variance,
  reward scale, critic conditioning — is doing the work, which would make the
  additive choice right for a reason nobody has written down.
- **The terrain level climbs to 0.75+ and the policy keeps tracking.** Then the
  static slope-vs-traction criterion in B6 is the wrong criterion for a machine
  with legs, and the curriculum ladder is more useful than this says.
- **A Hound VRAM measurement lands close to 4,649 MiB.** Then inheriting the
  PhysX side of the ledger from ANYmal was fine and the caution in trigger 3
  cost a measurement for nothing — which is a cheap way to be wrong.
- **`rho` swept turns out to be near `L2/2` over most of the usable stroke.**
  Then the assumed scalar was defensible and the maneuver numbers were closer to
  right than this decision allows.

## Dangerous to act on as written

Recorded verbatim in substance, because it is the most operationally useful part:

- **Launching the reward table as designed.** The composition ground is void,
  the stander fences are inactive in the pathology's regime, and `lin_vel_y`
  puts an unremovable 56% floor on the largest term. The most likely outcome is
  `015`'s one-trot reproduced at Isaac scale — a wasted ~16 h across three
  seeds. Settle `lin_vel_y` and re-price the fence first; both are minutes.
- **Trusting ~850 MiB of VRAM headroom.** Every modelled delta is PyTorch-side
  on an unmeasured "peak" for a different robot with 33% more DoF and rolling
  cylinder-vs-mesh contacts instead of point feet. An OOM five hours in is the
  exact failure the projection was meant to prevent.
- **Handing the print effort "buy knee speed, hold `k²/b_rear` down; the
  backflip closes at 1.07×."** The margin is set by a guessed scalar that ±30%
  swings across pass and fail, the arm is the wrong one for the named maneuver,
  the `+1` in `beta_req` is a derivation error, and the same session's own
  script gives a 2.1× larger jump. A CAD decision taken on this is not
  recoverable later.
- **Handing the print effort "+0.55 rad of hip range or a two-way knee."**
  Direction right; the supporting printout shows no effect, and computed
  properly the two options differ by ~3× in the margin they buy.
- **Running A0 as a one-variable control arm.** It changes three things — body,
  actuator model, reward body-name regexes. Whatever it shows will not be
  attributable to the body.
- **Citing `0.764 → 0.194` anywhere.** The record already bars the F table
  pending a committed script, and the prior refutation already named that exact
  comparison as not like-for-like.

## Sources

- Installed Isaac Lab tree, `VERSION` 3.0.0 (`release/3.0.0-beta2`), read at the
  file:line references above. Read-only.
- Installed rsl_rl 5.0.1 — **outside the two-repo boundary**, read once, flagged
  above, to be re-verified from an in-scope copy.
- `src/bestiary/isaac/anymal_desert_env_cfg.py`,
  `src/bestiary/terrain/isaac_hf.py`, `src/bestiary/robots/hound/build.py`,
  `src/bestiary/robots/hound/CARD.md`, `src/bestiary/envs/hound.py`,
  `assets/terrain/desert_hfield.bin`.
- `research/decisions/0004-inherit-the-isaac-reward-knowingly.md`,
  `research/learnings/009`, `011`, `012`, `015`,
  `docs/theory/reward-composition.md` (including its appended refutation),
  `docs/theory/command-tracking-reward.md`.
- Part B arithmetic: **computed but not yet committed.** Lands as
  `research/scripts/isaac_hound_stack_arithmetic.py`.
