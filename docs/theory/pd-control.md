# PD position control

**Written:** 2026-07-25 · **Because:** the hound moved from torque control to
position targets (`research/episodes/002-pd-position-targets.md`), and the
gains had to be justified rather than guessed.

Every number below is this robot's, read out of `assets/hound16pd.xml`. None
of it is a textbook example.

---

## 1. The problem it solves

Under torque control the policy outputs a torque for each of the twelve leg
motors, twenty times a second. Gravity is pulling the machine down the whole
time, so *just standing still* requires the policy to output the exact torque
that cancels gravity at every joint, continuously, forever. Get it slightly
wrong and the leg sags; stop outputting and the robot collapses.

That is a control problem the policy has to solve **before** it can start
solving the problem we actually care about, which is where to put the feet.

PD position control moves the first problem into the simulator. The policy
says *where* it wants a joint to be, and a stiff spring-damper underneath
works out the torque. Standing still becomes "ask for the standing angles",
which under our action mapping is the **zero action**.

## 2. The control law

At every physics step, for each leg joint, MuJoCo computes:

```
τ = kp · (q* − q) − kv · q̇
```

| symbol | meaning | units |
|---|---|---|
| `τ` | torque the actuator applies | N·m |
| `q` | joint's current angle | rad |
| `q*` | target angle the policy asked for | rad |
| `q̇` | joint's current angular velocity | rad/s |
| `kp` | **proportional gain** — stiffness | N·m/rad |
| `kv` | **derivative gain** — damping | N·m/(rad·s⁻¹) |

Read it in plain English: **the first term pulls toward the target, harder the
further away you are. The second term resists motion, harder the faster you
are moving.** The first is a spring, the second is a shock absorber. That is
all a PD controller is.

There is no integral term (no `ki`), which is why this is PD and not PID. An
integral term would slowly wind up to cancel steady-state error, and every
production legged-RL stack leaves it out — winding up against a foot planted
on the ground is how you get a leg that suddenly kicks when it lifts.

## 3. Where the numbers came from

`kp` and `kv` are carried over from the hand controller in
`src/bestiary/robots/hound/play.py`. They are the only gains on this machine
already shown to hold a stance and re-pose it. Rather than trusting that,
here is the check.

### Effective inertia

A joint does not swing its own mass — it swings everything distal to it, plus
the motor's reflected rotor inertia (`armature = 0.01`). MuJoCo's mass matrix
`M` gives this directly; the diagonal entry for a joint's degree of freedom is
its effective inertia `I` in kg·m², evaluated at the standing stance.

### What else acts on the joint

Two things add to the PD gains and must not be forgotten:

- Each leg joint carries a **parallel spring** (`stiffness`, `springref`),
  sized in `build.py` to hold the stance passively. It adds to `kp`.
- Each joint has **damping** `b = 1.5` N·m/(rad·s⁻¹). It adds to `kv`.

So the honest totals are `kp_tot = kp + k_spring` and `kv_tot = kv + b`.

| joint | I (kg·m²) | kp | k_spring | **kp_tot** | kv | b | **kv_tot** |
|---|---|---|---|---|---|---|---|
| abduct | 0.08808 | 60 | 12.12 | **72.12** | 3.0 | 1.5 | **4.5** |
| hip | 0.07949 | 80 | 18.58 | **98.58** | 4.0 | 1.5 | **5.5** |
| knee | 0.03105 | 90 | 32.77 | **122.77** | 4.5 | 1.5 | **6.0** |

## 4. Natural frequency and damping ratio

A mass on a spring with a damper is the most-studied system in mechanics. Its
behaviour is fixed by two derived numbers.

**Natural frequency** — how fast it wants to oscillate:

```
ωₙ = √(kp_tot / I)            rad/s
fₙ = ωₙ / 2π                  Hz
```

**Damping ratio** — whether it oscillates at all:

```
ζ = kv_tot / (2·√(kp_tot · I))     dimensionless
```

`ζ < 1` overshoots and rings. `ζ = 1` is *critically damped* — fastest
approach with no overshoot. `ζ > 1` is overdamped: no overshoot, but sluggish.

Worked for the hip, with this robot's real numbers:

```
ωₙ = √(98.58 / 0.07949) = √1240.2 = 35.22 rad/s  →  5.60 Hz

ζ  = 5.5 / (2 · √(98.58 × 0.07949))
   = 5.5 / (2 · √7.836)
   = 5.5 / (2 × 2.799)
   = 5.5 / 5.598
   = 0.982
```

All three:

| joint | fₙ (Hz) | ζ | reads as |
|---|---|---|---|
| abduct | 4.55 | 0.893 | slightly underdamped — a touch of overshoot |
| hip | 5.60 | **0.982** | essentially critically damped |
| knee | 10.01 | 1.537 | overdamped, deliberately stiff |

**In plain English: the hip is almost exactly critically damped — it moves to
a commanded angle as fast as it possibly can without overshooting.** That was
not designed; it fell out of gains tuned by hand until the machine felt right.
Worth knowing, because it means there is little to gain from retuning the hip
and the abduction joint is the one with room to improve.

This also explains the measurement in episode 002: commanded to its own
stance, the machine settles with **0.4° of steady-state joint error** and no
visible ringing. With ζ near 1 there is nothing to ring.

## 5. Why the PD loop must run faster than the policy

This is the part that matters most, and the number is startling.

- The policy acts at **20 Hz** (`frame_skip = 10`).
- Physics runs at **200 Hz** (`timestep = 0.005`).
- MuJoCo evaluates `τ = kp(q* − q) − kv·q̇` **every physics step** — ten times
  per policy step.

Now apply the Nyquist–Shannon sampling theorem: to control a system
oscillating at `f`, you must sample faster than `2f`. Anything slower and the
controller cannot even *see* the oscillation it is supposed to damp.

The knee's natural frequency is **10.01 Hz**. Twice that is 20.02 Hz. The
policy rate is **20 Hz**.

**The knee sits exactly at the Nyquist limit of the policy's own control
rate.** A PD loop running at policy rate would be marginally stable at best —
it would be sampling the knee's dynamics almost exactly once per half-cycle.
At 200 Hz it samples about 20 times per cycle, which is comfortable.

This is the whole architectural argument in one number. The policy is far too
slow to stabilize this robot's joints. It does not have to be, because it is
not the thing stabilizing them.

## 6. From action to target

The policy's action stays in `[-1, 1]` (the repo convention, unchanged), and
`HoundEnv.action_to_ctrl` maps it:

```
q* = q_stance + a · action_scale          action_scale = 0.5 rad
```

Consequences worth stating plainly:

- **a = 0 commands the standing stance.** Standing is the origin of the
  action space, not a behaviour to be discovered. This is the prior that
  motivated the whole change.
- **a = ±1 commands ±0.5 rad (±28.6°)** from stance, then clipped to the
  joint's own travel, so an unreachable target cannot be requested.
- The wheels are **not** position-controlled. A wheel that turns forever has
  no pose to hold, so all four stay on direct torque. The action vector is
  still 16 slots in the same order; only the meaning of twelve of them
  changed.

## 7. Torque limits still bind

`forcerange` clamps `τ` to the same Go2 ceiling the torque model has: ±23.7
N·m at abduction and hip, ±40 N·m at the knee.

The angle at which the servo saturates is where `kp · error` hits that
ceiling:

```
error_sat = τ_max / kp
```

For the hip: `23.7 / 80 = 0.296 rad = 17.0°`. Ask for more than 17° of error
and the actuator is simply at full torque — the same full torque the torque
model had.

**PD does not make this machine stronger. It makes it easier to command.**
The 2.17 N·m/N torque-to-weight ratio that episode 001 flagged is untouched.

## 8. What this does not fix

Position targets remove pose-holding from the learning problem. They do not:

- add torque (§7),
- add samples — still ~120 steps/s in one environment, which decision 0001
  identified as the real bottleneck,
- decide *where* the feet should go, which is the actual locomotion problem.

If episode 002's run plateaus at roughly the torque run's number, that is
evidence the parameterization was never the binding constraint — and the
throughput argument carries the whole weight.

## See also

- `research/episodes/002-pd-position-targets.md` — the run this was written for
- `research/decisions/0001-defer-isaac-lab.md` — the throughput decision
- `src/bestiary/robots/hound/build.py` — `Spec.kp_*`, `Spec.kv_*`, `actuator_xml`
- `src/bestiary/envs/hound.py` — `HoundEnv.action_to_ctrl`
