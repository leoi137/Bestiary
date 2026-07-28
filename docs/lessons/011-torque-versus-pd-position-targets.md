# 011 — Torque control versus PD position targets

**One sentence:** A torque-controlled policy has to output *how hard to push*
every joint, every step, and therefore has to learn to hold the joint still as
well as to move it; a PD position-target policy outputs *what angle to be at*
and a fixed controller underneath it works out the pushing — which in this
repo's one probe cut the samples needed to reach the same score by about 5x,
and did not raise the score.

Assumes [008 — what a policy is](008-what-a-policy-is.md).

## The idea

The hound has 16 actuated joints, and the policy emits 16 numbers each control
step. What those 16 numbers *mean* is a choice, and it is the only difference
between two of this project's environments.

Under **torque control** (`HoundDesert-v0`, `assets/hound16_desert.xml`) each
number is a fraction of the joint's peak torque, in [-1, 1]. A knee command of
0.5 means *apply half of 40 N·m to the knee and hold it there for the next
0.05 s*. Torque is a force, and a force applied to a mass produces
acceleration, not position. So a **constant** command does not hold a pose — it
accelerates the joint until something stops it. To stand still, the policy must
find, every step, the exact torque that cancels gravity at the current angle,
and it must find a different one the moment the angle changes. That is inverse
dynamics, and the policy has to learn it from scratch, from reward alone.

Under **PD position targets** (`HoundPDDesert-v0`,
`assets/hound16pd_desert.xml`) the twelve leg joints become MuJoCo `<position>`
servos. Each number is now a **target angle in radians**, and a small fixed
controller — *proportional–derivative*, hence PD — computes the torque itself:
push in proportion to how far the joint is from where it was told to be
(**P**), and brake in proportion to how fast it is moving (**D**). A constant
command now *holds a pose*, because once the joint arrives the error is zero
and so is the torque. The four wheels stay on torque; a wheel that spins
forever has no pose to hold.

Nothing about the machine got easier. The `forcerange` on each PD servo is set
to the same ceiling as the torque model's `gear` — 40 N·m at the knee in both.
Only the *command* got easier.

## The math

The controller MuJoCo runs internally is one line:

    tau = kp * (q_des - q) - kv * qdot

- `tau` — the torque actually applied at the joint, N·m
- `q` — the joint's current angle, rad; `qdot` — its speed, rad/s
- `q_des` — the target angle the policy asked for, rad (this *is* the action)
- `kp` — proportional gain, N·m/rad — how hard it pulls per radian of error
- `kv` — derivative gain, N·m/(rad/s) — the braking, per unit speed

Worked on the hound's front-left knee, from
[`scripts/pd_vs_torque_math.py`](scripts/pd_vs_torque_math.py), which reads
every constant out of the committed XML and the env:

    kp = 90.0 N*m/rad, kv = 4.5 N*m/(rad/s), forcerange +/- 40.0 N*m
    q     (standing stance)  -1.61973 rad
    q_des (policy commands)  -1.2 rad
    error                     0.41973 rad
    qdot                      0.0 rad/s

    tau = 90.0 * 0.41973 - 4.5 * 0.0 = 37.776 N*m      (ceiling 40.0, not saturated)

The identical torque under the other env would need the policy to emit
`37.776 / 40.0 = 0.9444`. Same physics, same instant — two completely different
numbers for the policy to learn to produce.

Now hold each command fixed for one step. The knee's inertia about its own axis
is 0.031047 kg·m² at this pose, so a constant 37.776 N·m gives

    alpha = tau / I = 37.776 / 0.031047 = 1216.7 rad/s^2
    swept in one 0.05 s step = 0.5 * alpha * dt^2 = 1.521 rad

The torque command sweeps the knee 1.52 rad — most of its 2.0 rad travel — in a
single control step, while the PD command settles at the requested angle and
holds there at zero torque. And the PD loop is not held constant across the
step: one control step is `frame_skip = 10` physics steps of 0.005 s, so the
inner controller re-reads `q` and `qdot` and recomputes `tau` **ten times** per
action, at 200 Hz, while the policy acts at 20 Hz.

**Physically: PD hands the policy an action space where doing nothing means
standing still, and where nearby actions mean nearby poses — the torque space
gives it one where doing nothing means falling over, and the mapping from
command to pose depends on the whole state.** That is why it is easier to
search, not why it would be better.

Two side facts the same script prints. The error at which the servo saturates
is `forcerange / kp = 40.0 / 90.0 = 0.4444 rad` (25.5°): past that the PD law
is clipped and the servo is just a torque source at full power, so the two
control modes are the *same thing* for large errors and differ only near the
target. And the inner loop's damping ratio is
`kv / (2*sqrt(kp*I)) = 1.346` — above 1, i.e. overdamped, so it approaches a
target without overshooting and cannot ring against the 20 Hz policy.

## What it bought, and what it did not

`research/ledger.jsonl` rows 1 and 2: same robot, same desert, same SAC, same
reward, one variable changed.

| | torque `hound_desert_v0` | PD `hound_pd_desert_v0` |
|---|---|---|
| steps to reach eval ≥ 1100 | 1,502,322 | **300,809** |
| peak eval return | **1218.3** | 1176.7 |
| mean eval after 400k steps | 887.5 | **1113.1** |

**4.99x fewer samples for the same band — and a peak 41.7 return points
LOWER.** Both readings are real and they are consistent: an easier action space
changes the *cost of the search*, not the *set of behaviours reachable*. Any
pose the PD servo can hold is a torque the torque model can also apply — the
ceiling is set by the robot, the terrain and the reward, none of which moved.
What moved is how much random flailing it takes to find that ceiling.

**One seed per arm.** Under this project's seed rule that is a **probe**, not a
finding: between-seed variance in SAC is comparable to most effects worth
claiming, and the two arms also ran to different step budgets. Treat "5x" as
the size of a thing worth measuring properly, not as a measured constant. See
[002 — why one training run is not a result](002-why-one-seed-is-not-a-result.md).

## Where it bites here

`src/bestiary/robots/hound/build.py` — `actuator_xml(spec, control=...)` is the
whole switch, and `kp_knee = 90.0` / `kv_knee = 4.5` are hand-chosen numbers,
not derived ones. Set `kp` too low and the leg cannot hold the robot up
whatever the policy commands; too high and the servo fights the 20 Hz command
stream and buzzes. Either failure looks like *the policy did not learn*, because
the reward curve is the only thing anyone reads — which is exactly the trap the
first PD run could have fallen into and did not.

## If you want to go deeper

Peng & van de Panne, *Learning Locomotion Skills Using DeepRL: Does the Choice
of Action Space Matter?* (2017), [arXiv:1611.01055](https://arxiv.org/abs/1611.01055)
— the paper that measured this comparison properly, with the seeds we did not
run.
