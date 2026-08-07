# 014 — The anatomy of the Spyder policy, input to output

**One sentence:** The trained Spyder brain is two ordinary four-layer
perceptrons — 235 floats in, 12 floats out for the actor, 235 floats in and 1
float out for the critic, 571,801 numbers in total — and everything that makes
it a *robot* rather than a curve fit happens on either side of it: what gets
packed into the 235, and what the simulator does with the 12.

Assumes [008 — what a policy is](008-what-a-policy-is.md),
[009 — actor and critic](009-actor-and-critic.md) and
[013 — what an observation is](013-what-an-observation-is.md).

Every number below is printed by
[`scripts/014_spyder_policy_anatomy.py`](scripts/014_spyder_policy_anatomy.py),
reading the trained checkpoint
`runs/spyder_gentle_s1/box_logs/2026-08-06_07-53-39/model_1499.pt` and that
run's own `params/env.yaml` and `params/agent.yaml`.

> A note on the one-page rule: this is a reference page, and it is longer than
> the standard allows. It buys that by replacing four short lessons that would
> each have to re-introduce the same checkpoint.

## 1 — The big picture, in five sentences

Fifty times a second the simulator builds one array of 235 floating-point
numbers describing the robot and the ground under it, and hands it to the
actor. The actor multiplies it through four weight matrices and returns 12
numbers, one per leg joint. Those 12 are not torques and not angles: each is
halved and added to that joint's rest angle to make a *target* angle, and a
spring-and-damper law in the simulator turns the gap between target and actual
into a torque, capped at 40 N·m (354 lbf·in). Three of the 235 inputs —
positions 9, 10 and 11 — are the command: forward speed, sideways speed, turn
rate, which is where W/A/S/D will enter (W/S sets slot 9 to ±0.6 m/s
(±1.34 mph), A/D sets slot 11 to ±0.8 rad/s, space sets all three to exactly
zero). The critic sees the same 235 numbers, returns one number — "how much
total reward do I expect from here?" — and is thrown away when training ends.

## 2 — The model

Verbatim `print(model)`, after loading `model_1499.pt` into a freshly built
rsl-rl model with `strict=True` (a listing the real weights refused to enter
would be a drawing, not a measurement):

    # actor — 286,616 parameters, loaded strict=True
    MLPModel(
      (obs_normalizer): Identity()
      (distribution): GaussianDistribution()
      (mlp): MLP(
        (0): Linear(in_features=235, out_features=512, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(in_features=128, out_features=12, bias=True)
      )
    )

    # critic — 285,185 parameters, loaded strict=True
    MLPModel(
      (obs_normalizer): Identity()
      (mlp): MLP(
        (0): Linear(in_features=235, out_features=512, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(in_features=128, out_features=1, bias=True)
      )
    )

Layer by layer, as measured in the checkpoint (`in` is a column of the weight
matrix, `out` a row):

| layer | tensor | shape | in → out | params |
|---|---|---|---|---|
| `mlp.0` | weight + bias | `(512, 235)` + `(512,)` | 235 → 512 | 120,832 |
| `mlp.1` | — | — | ELU, elementwise | 0 |
| `mlp.2` | weight + bias | `(256, 512)` + `(256,)` | 512 → 256 | 131,328 |
| `mlp.3` | — | — | ELU, elementwise | 0 |
| `mlp.4` | weight + bias | `(128, 256)` + `(128,)` | 256 → 128 | 32,896 |
| `mlp.5` | — | — | ELU, elementwise | 0 |
| `mlp.6` (actor) | weight + bias | `(12, 128)` + `(12,)` | 128 → **12** | 1,548 |
| `mlp.6` (critic) | weight + bias | `(1, 128)` + `(1,)` | 128 → **1** | 129 |
| `distribution.std_param` | parameter | `(12,)` | — | 12 |

One layer is one matrix multiply plus a bias, then ELU:

    h = ELU(W · x + b)

- `x` — the layer's input vector; for `mlp.0` it is the 235-float observation.
- `W` — the weight matrix, `out × in`. `mlp.0.weight` is `(512, 235)`: one
  **column per observation value**, one **row per hidden unit**.
- `b` — the bias, one number per output.
- `ELU(z)` — the nonlinearity: `z` when `z > 0`, `exp(z) − 1` when `z ≤ 0`.
  Without it, four stacked matrices collapse into one matrix and the whole
  network could only draw straight lines.

Two things the listing says that are easy to miss. `obs_normalizer` is
`Identity()` — `obs_normalization: false`, so the raw numbers go straight in,
joint velocities in rad/s sitting next to a gravity unit vector with no
rescaling. And `distribution` is a `GaussianDistribution` with
`std_type: scalar`: the network outputs only the **mean** action, and the
spread is 12 separate learned numbers that do not depend on the observation at
all. After 1500 iterations they are 0.357 to 0.614, mean 0.479.

## 3 — The observation: 235 numbers, eight blocks

Concatenated in declaration order. Noise is uniform, added at every step during
training and switched off in the Play config.

| indices | block | size | what it is, in plain words | units | noise |
|---|---|---|---|---|---|
| `[0:3]` | `base_lin_vel` | 3 | how fast the torso is moving, in the torso's own frame: forward, sideways, up | m/s | ±0.1 |
| `[3:6]` | `base_ang_vel` | 3 | how fast the torso is rotating: roll, pitch, yaw rate | rad/s | ±0.2 |
| `[6:9]` | `projected_gravity` | 3 | which way is down, as a unit vector in the torso frame — this is the robot's inner ear | none | ±0.05 |
| `[9:12]` | `velocity_commands` | 3 | **the joystick**: wanted forward speed, wanted sideways speed, wanted turn rate | m/s, m/s, rad/s | none |
| `[12:24]` | `joint_pos` | 12 | each joint's angle minus its rest angle (rest is 0 here, so: the angle) | rad | ±0.01 |
| `[24:36]` | `joint_vel` | 12 | how fast each joint is turning | rad/s | ±1.5 |
| `[36:48]` | `actions` | 12 | the 12 numbers the actor produced last step | none | none |
| `[48:235]` | `height_scan` | 187 | a 17 × 11 grid of downward rays: how far the ground is below the body at each grid point, clipped to [−1, 1] | m | ±0.1 |
| | **sum** | **235** | | | |

The script asserts this sum equals `mlp.0.weight.shape[1]`; if a block were
wrong the script fails rather than the table misleading you.

Where the height scan looks: a 2.56 × 1.6 m (8.4 × 5.2 ft) rectangle centred on
the torso, sampled every 0.16 m (6.3 in), which gives 17 × 11 = 187 rays. That
footprint is chosen so the ±0.76 m (±29.9 in) foot centres are inside it — the policy must
never place a foot on ground it cannot see — and so it reaches 1.28 m (4.2 ft)
ahead, about two seconds of travel at the top command.

Slots 9–11 are the whole human interface. `lin_vel_y` is pinned to `(0, 0)`
today, so slot 10 is always 0.0 — the slot exists anyway, because widening the
observation later would orphan every checkpoint ([013](013-what-an-observation-is.md)).
Commands are dead-zoned: a driving command has `|v_x| ∈ [0.25, 0.6]` m/s
(0.56–1.34 mph), a turn has `|w_z| ∈ [0.2, 0.8]` rad/s, and 10% of resamples
zero all three. There is no ambiguous middle, which is exactly the
key-or-nothing interface a keyboard gives you.

## 4 — The action: 12 numbers, and the two steps that make them torques

The actor's output is 12 raw floats, one per joint (`hip_1..4`, `lift_1..4`,
`knee_1..4`). They are **not clipped** — `clip: null` on both the action term
and the runner — so a Gaussian sample can be any size, and the effort ceiling
is the only thing that bounds what it can do.

**Step one, action → joint target.** `JointPositionAction` with `scale = 0.5`,
`use_default_offset = true`:

    q_cmd[j] = 0.5 · a[j] + q_default[j]

- `a[j]` — the actor's raw output for joint `j`, dimensionless.
- `q_default[j]` — that joint's rest angle. Spyder's standing arch **is** joint
  zero, so `q_default = 0` for all twelve and the law is just `q_cmd = 0.5·a`.
- `q_cmd[j]` — the angle the joint is told to be at, in radians.

**Step two, joint target → torque.** The implicit PD actuator, 200 times a
second (four physics steps per policy step):

    tau[j] = KP · (q_cmd[j] − q[j]) − KD · qd[j],   clipped to ±40 N·m

- `KP = 30.0` N·m/rad (266 lbf·in/rad) — stiffness. How hard the joint pulls
  per radian of error. It is the authored MuJoCo spring constant, kept so the
  Isaac machine and the MuJoCo machine hold the same arch.
- `q[j]` — the joint's actual angle, rad. `q_cmd − q` is the error.
- `KD = 5.585696` N·m·s/rad (49.4 lbf·in·s/rad) — damping, the brake on joint
  speed. Derived, not tuned: `KD = 2·ζ·√(KP·I)` with `ζ = 0.5` and
  `I = 1.04` kg·m² (3554 lb·in²), the measured joint-space inertia.
- `qd[j]` — the joint's actual angular velocity, rad/s.
- `±40` N·m (±354 lbf·in) — the effort ceiling, `gear × ctrlrange` from the
  authored MJCF.

### The arithmetic, on the trained weights

Compose the two steps at zero error and zero speed (`q = 0`, `qd = 0`):

    tau = KP · 0.5 · a = 15.0 · a

So **one unit of action is 15 N·m (133 lbf·in)**, and `|a| ≥ 40/15 = 2.667`
saturates the drive before physics is consulted. Now put the real learned noise
through it. The smallest of the twelve learned standard deviations is 0.3565:

    0.3565 × 0.5   = 0.1782 rad of jitter on the target angle
    0.3565 × 15.0  = 5.35 N·m (47.4 lbf·in) of jitter on the torque

**Physically: even after 1500 iterations, the policy is still shaking each leg
with about 5 N·m (44 lbf·in) of deliberate randomness — that is what exploration costs, in
torque, on this machine.** It is why a policy watched at training-time noise
looks drunk and the same policy played deterministically does not.

## 5 — The reward

**The gentle task** pays eleven terms every step, each multiplied by its weight
*and* by `step_dt = 0.02` s:

| term | weight | what it pays for |
|---|---|---|
| `track_lin_vel_xy_exp` | +1.0 | matching the commanded `(v_x, v_y)`, `std = 0.3` |
| `track_ang_vel_z_exp` | +0.5 | matching the commanded `w_z`, `std = 0.4` |
| `feet_air_time` | +0.125 | keeping a foot in the air past 0.5 s — buys a gait |
| `lin_vel_z_l2` | −2.0 | bouncing up and down |
| `ang_vel_xy_l2` | −0.05 | rolling and pitching |
| `action_rate_l2` | −0.01 | changing the action abruptly |
| `undesired_contacts` | −1.0 | a femur touching ground (a collapse, not a step) |
| `dof_torques_l2` | −1e−5 | raw effort |
| `dof_acc_l2` | −2.5e−7 | slamming the joints |
| `flat_orientation_l2` | 0.0 | off |
| `dof_pos_limits` | 0.0 | off |

Two tracking kernels, four penalties, one gait term, two off. Inherited whole
from the ANYmal recipe as a *control*, not as a tuned design — which is the
problem the next reward solves.

**The forward-only diagnostic** deletes all eleven and pays one thing:

    r = v_x

- `v_x` — the torso's forward speed **in its own frame**, m/s, signed. Body
  frame, not world frame, so the reward means "go where your nose points"
  rather than "go towards the arena's +x", which would be unlearnable on a
  terrain curriculum that respawns robots anywhere.

Weight 1.0, and the reward manager multiplies by `step_dt`:

    per-step reward = v_x · 0.02 s = metres travelled during that step

Summing metres-per-step over an episode gives metres. At the 0.37 m/s
(0.83 mph) this repository's SAC Spyder-12 actually walked
([`research/learnings/001`](../../research/learnings/001-flat-reward-breaks-on-terrain.md)):

    0.37 × 0.02          = 0.0074 per step
    0.0074 × 1000 steps  = 7.40

**Physically: the episode return is not a score, it is a distance — 7.40 means
7.40 m (24.3 ft) of ground covered in the 20-second episode.** That is the
whole point of the diagnostic: a number nobody has to interpret. Standing still
scores exactly 0, falling early forfeits the rest, and the gentle task's
kernel-based returns are a different currency that must never be put in the
same column.

## 6 — What it all weighs, and what survives

    actor    286,616 parameters   (286,604 in the MLP + 12 learned std)
    critic   285,185 parameters
    total    571,801 parameters   in a 6,882,293-byte checkpoint

The critic is 49.9% of the trained weights and **none of it ships**. It exists
to answer "how good is this state?", which PPO needs to tell a lucky step from
a good one; at deployment the robot only ever runs the actor. Measured, not
asserted: the exported `exported/policy.pt` beside the checkpoint holds
**286,604** parameters — the actor's four `Linear` layers, and not the critic,
not even the 12 learned standard deviations. Half the training cost buys a
teacher you then throw away — see [009](009-actor-and-critic.md) for why that
trade is not optional.

## Where it bites here

`src/bestiary/isaac/rl_cfg.py` declares **no network dimensions at all** — it
subclasses `AnymalCRoughPPORunnerCfg` and changes only `experiment_name`. So
`[512, 256, 128]` and `elu` are inherited from upstream and appear nowhere in
this repository's source; the only place they are written down for this run is
the run's own `params/agent.yaml`. If you go looking for the architecture in
the config files you will not find it, and that is worth knowing before you
conclude it is something else.

The second bite is the same one 013 names, now with the concrete number:
`mlp.0.weight` is `(512, 235)`, so one added observation value is 512 new
weights in the actor and 512 in the critic. Adding a sensor does not upgrade
this checkpoint; it orphans it.

## If you want to go deeper

- `src/bestiary/isaac/spyder_cfg.py` — the PD gain derivation, including why
  `ζ = 0.5` rather than 1.0 (at critical damping a routine 3.6 rad/s swing puts
  the damping term alone at the 40 N·m ceiling, and a drive that saturates on
  damping cannot track anything).
- `src/bestiary/isaac/rewards.py` — why the diagnostic reward is *stricter*
  than the 2016-era benchmark it cites: no alive bonus, no control cost.
- [011 — torque versus PD position targets](011-torque-versus-pd-position-targets.md)
  — the same 12 numbers meaning two different things in two envs.
