# HOUND-16

A 16-DoF wheel-legged quadruped on Unitree Go2 kinematics and masses, with a
driven hub wheel where each point foot would be.

Generated, never hand-written: `build.py` is the single source of truth and
`check.py` is the 38-assertion oracle that says the machine has not moved.
Every number below is printed by

```bash
venv/bin/python -m bestiary.robots.hound.build --report
venv/bin/python -m bestiary.robots.hound.check
```

and nothing here is typed from memory. Where a value is *derived* rather than
chosen, this card says so and names the constraint it was derived from —
because the derived ones are the ones that silently go wrong when a length
changes.

---

## Provenance

Link lengths, hip offsets, per-link masses and torque limits are **Unitree
Go2's**, read from `unitree_go2/go2.xml` in MuJoCo Menagerie (Apache-2.0; the
underlying Unitree model is BSD-3). Using a real robot's inertial spec means
the mass distribution, the reach-to-mass ratio and the torque limits are
mutually consistent — a machine that could exist.

**The wheel is not from Unitree.** No vendor ships a wheel-legged MJCF, so the
fourth joint per leg is designed here against the published envelope of the
wheeled quadrupeds that do exist (Unitree B2-W / Go2-W, ANYbotics
ANYmal-on-Wheels, Swiss-Mile). The calf is shortened from Go2's 0.213 m to
0.190 m so calf + wheel radius lands near the original foot and the leg keeps
roughly its original reach.

---

## Morphology

```
                trunk  (free joint: 3 translations + 3 rotations)
                  |
  +---------------+---------------+
  |                               |
FL_abduct (hinge, axis +X)    ... x4
  |
FL_hip    (hinge, axis +Y)
  |
FL_knee   (hinge, axis +Y)
  |
FL_wheel  (hinge, axis +Y, UNLIMITED, no spring)
```

**4 legs x (abduct + hip + knee + wheel) = 16 actuated DoF**, plus 6
unactuated from the trunk's free joint.

Axis convention: **+X forward, +Y left, +Z up, metres, radians.** Abduction
uses a single global axis `(1 0 0)` on all four legs rather than mirroring per
side — Unitree's own convention, so joint signs match anything you read in Go2
code. It does mean **a symmetric action vector is not a symmetric pose**:
+abduction swings the left legs out and the right legs in.

### Spaces

```
nq 23   nv 22   nbody(non-world) 17   nu 16

obs = (nq-2-4) + nv + nbody*6 + reserved = 17 + 22 + 102 + 28 = 169
     ^ -2  drops world x,y (position-invariant locomotion)
     ^ -4  drops the WHEEL ANGLES: unbounded integrators. Velocities kept.
     ^ +28 RESERVED (3 command + 25 height scan), held at zero
```

The reserved slots exist so a later roadmap step cannot change the observation
width and orphan every checkpoint. **The observation width is a one-way door** —
the actor's first layer is `Linear(obs, 256)`, so a width change makes every
existing checkpoint fail to load, not degrade. See `research/learnings/003` and
`docs/lessons/013`.

---

## Geometry and mass

| | value | source |
|---|---|---|
| trunk box half-extents | 0.1881 x 0.04675 x 0.057 m | Go2 |
| thigh length | 0.213 m | Go2, unchanged |
| calf length | 0.190 m | Go2's 0.213, shortened for the wheel |
| wheel radius / width | 0.085 / 0.050 m | designed here |
| hip pivot (x, y) | 0.1934, 0.0465 m | Go2 |
| abduction offset | 0.0955 m | Go2 |
| wheelbase | 0.387 m | derived |
| **stand height** | **0.3634 m** | derived: axle drop 0.2784 + wheel r |

**Mass budget — 17.005 kg total**

| link | each | count | total |
|---|---|---|---|
| trunk | 6.921 kg | 1 | 6.921 |
| abduct link | 0.678 kg | 4 | 2.712 |
| thigh | 1.152 kg | 4 | 4.608 |
| calf | 0.241 kg | 4 | 0.964 |
| wheel | 0.450 kg | 4 | 1.800 |
| | | | **17.005 kg** |

The wheel mass is deliberately a little heavy for a 170 mm hub drive.
**Unsprung mass at the ankle is what makes wheeled legs hard to control**, and a
light wheel is the easy case.

---

## Joints

| | abduct | hip | knee | wheel |
|---|---|---|---|---|
| axis | +X | +Y | +Y | +Y |
| range (rad) | -0.80 … +0.80 | -1.20 … +2.60 | -2.60 … -0.60 | **none — unlimited** |
| range (deg) | ±46 | -69 … +149 | -149 … -34 | continuous |
| peak torque (N·m) | 23.7 | 23.7 | 40.0 | 3.0 |
| armature | 0.01 | 0.01 | 0.01 | 0.004 |
| damping | 1.5 | 1.5 | 1.5 | 0.05 |
| frictionloss (N·m) | 0.2 | 0.2 | 0.2 | **0.399** (derived) |

`ctrlrange` is `[-1, 1]` everywhere (repo convention), so **`gear` IS the peak
joint torque in N·m.** Leg values are Go2's rated torques; the knee is scaled
for the shorter calf (Go2 rates 45.43 N·m over 0.213 m). Go2 allows ±60° of
abduction; this machine is held to ±46°.

Wheel damping is low because **a wheel is supposed to coast.** Rolling
resistance is a derived joint brake torque (0.399 N·m) rather than joint
damping, which would wrongly resist the wheel while it is in the air.

### The wheel is a genuinely different kind of joint

Three consequences that propagate everywhere downstream:

1. **It never stops.** `limited="false"` — its angle integrates without bound,
   so it is not a state you can feed a policy. `envs/hound.py` drops the four
   wheel angles from the observation and keeps only their velocities.
2. **It has no rest pose.** The leg joints carry a spring toward the standing
   stance; a spring on a wheel would mean the machine rolls back to where it
   started.
3. **It is limited by the ground, not by the motor.** See the traction budget.

---

## Standing stance

| joint | value | how |
|---|---|---|
| abduct | +0.0000 rad (0.0°) | input |
| hip | +0.7500 rad (43.0°) | input |
| knee | **-1.6197 rad (-92.8°)** | **SOLVED** — puts the axle under its hip pivot |

The knee is solved, not typed. A stance with the contact patch offset from the
pivot pushes the machine sideways the moment it takes weight, which reads as a
drifting robot and is nobody's intent. `Spec.stance_knee` derives it; change
`stance_hip` and the knee follows.

**Static torque to hold the stance:** abduct +1.939, hip -0.992, knee -5.243 N·m
(signed gravity torque).

---

## The stance springs — a crutch, and labelled as one

Real Go2 joints have no springs. These exist so the machine is self-supporting
at reset and SAC spends its first 50k steps learning to move rather than
learning not to collapse. **Set the stiffnesses to 0 for the physically pure
machine** — it then has to learn to stand before it can learn to move.

The stiffnesses are **derived from a free-body diagram**, not guessed. The first
draft of `build.py` guessed 6 / 12 / 12 N·m/rad and the machine folded to a
0.18 m belly-flop the moment gravity was switched on.

| joint | holds (N·m) | spring k | stability k | springref | authority | reachable vs range | sized by |
|---|---|---|---|---|---|---|---|
| abduct | +1.94 | 12.116 | 0.0 | -0.16 | 1.96 rad | full range | **LOAD** |
| hip | -0.99 | 18.577 | 11.611 | +0.80 | 1.28 rad | **67% of range** | **STABILITY** |
| knee | -5.24 | 32.766 | 5.111 | -1.46 | 1.22 rad | full range | **LOAD** |

- **spring k** = `|holds| / sag_target`, with `sag_target = 0.16 rad (9.2°)`.
  Load only, so the motor keeps its travel.
- **springref** is the stance wound 0.16 rad *against* the load, so the spring
  already carries `holds` at the stance and t=0 is a true equilibrium. The
  machine stands exactly where it is drawn.
- **stability k** = 1.6 x the inverted-pendulum threshold `-N·d²h/dq²`.
  A wheeled leg is an inverted pendulum, and a spring merely strong enough to
  carry the static load lets the machine slide into the splits — which the
  second draft did (front wheels rolled 15 cm forward, trunk fell to 0.25 m).
  The 1.6 multiple is **measured**: sweeping the hip against a passive settle,
  a ±0.1 rad noisy reset and a 25 cm drop, it first stands cleanly in all three
  at ~18 N·m/rad. At exactly 1.0x it still folds.
- **Abduction is exempt from the stability term**: a wheel rolls fore-aft but
  *grips* sideways, so only the hip loses its stability.
- Coupled (hip, knee) Hessian minimum eigenvalue **+5.77 → STABLE**.

`sag_target = 0.16` is a **floor, not a preference.** The knee binds: its
reachable band is `springref ± gear/k`, and covering the full [-2.60, -0.60]
range needs `s ≥ 0.154`. Softer than that and the spring, not the mechanism,
decides how far the knee can straighten.

The cost is real: **the hip keeps ~2.5 rad of its 3.8 rad range** at full motor
torque.

---

## PD position control (`HoundPD-*` envs)

`build(control="pd")` turns the twelve leg joints into position servos. **The
four wheels stay on torque** — a wheel that spins forever has no pose to hold
and a position target on it is meaningless.

| | abduct | hip | knee |
|---|---|---|---|
| kp (N·m/rad) | 60.0 | 80.0 | 90.0 |
| kv (N·m/(rad/s)) | 3.0 | 4.0 | 4.5 |

`action_scale = 0.5` rad. The env maps `target = stance + action × action_scale`,
so **a zero action IS the standing stance.** That is the whole prior: the policy
starts at "standing" instead of having to discover it.

These are the gains `play.py`'s hand controller already uses — the only gains on
this machine shown to hold the stance and re-pose it without fighting the
physics into instability. Damping-ratio check in `docs/theory/pd-control.md`.

---

## Contact

MuJoCo friction is `(sliding, torsional, rolling)`. The wheel sets
`priority="1"` so **its** numbers win over the floor's instead of the usual
element-wise max — a wheel's contact is a property of the tyre, and we want the
same tyre on the plane and on sand.

| | value | note |
|---|---|---|
| sliding | 0.9 | traction; above this the wheel just spins |
| torsional | 0.03 | **inert at condim=3** |
| rolling | 0.002 | **inert at condim=3** |
| `condim` | **3** | sliding only |
| `margin` | **0.0** | not the 1 mm used elsewhere |

**`condim=3` cost a sweep to settle.** `condim=6` buys contact-level torsional
and rolling resistance, but on the desert it also settles 4 mm lower and drifts
twice as far: every extra constraint per contact is another chance for the
heightfield's imperfect contact normals to disagree. Rolling resistance moved to
the wheel joint instead. The values are left in place so switching back is a
one-character experiment.

**`margin=0` matters more than it looks.** Margin makes contacts appear *before*
touchdown, and on a heightfield those early contacts are found against the
vertical walls of the terrain-cell prisms — normals pointing sideways, into a
wheel that is supposed to be rolling. At `margin=0.001` the machine spawned with
11–18 horizontal-normal contacts and was kicked over. At `margin=0` it stands at
0.3630 with upright 1.000.

---

## Traction budget — why `gear_wheel` is small

```
static load per wheel        41.70 N   (4.25 kgf)
max friction, mu = 0.9       37.53 N
-> traction limit             3.19 N·m  at r = 0.085 m
gear_wheel                    3.00 N·m  (94% of the limit)

all four at full torque     141.18 N thrust -> 8.30 m/s^2  IDEAL
...but the wheelie limit     88.78 N        -> 5.22 m/s^2
measured saturation                         ~2.0 m/s^2
```

The leg motors are sized by what they must **hold**; the wheel motors by what
the ground will **accept**. A bigger hub drive would buy nothing.

**`check.py` finds this conclusion right for the wrong reason, which is worth
knowing before tuning anything.** Thrust does saturate — 5x the gear gives
*less* acceleration, not more — but it saturates at ~2 m/s², a quarter of the
`μg = 8.8` the friction cone allows, with the cone only 5% used. What actually
binds is a **wheelie**: all thrust acts at ground level while the mass sits
0.363 m up on a 0.387 m wheelbase, so hard driving pitches the machine onto its
rear pair, and past ~6 N·m lifts every wheel clear of the ground. **Friction is
the upper bound; geometry is the real one.**

---

## Known-open, carried forward

- **Backward creep, cause UNKNOWN.** With every motor at zero the machine
  drifts **-3.55 cm/s** (-1.7761 ± 0.0276 m per 1000-step episode) on the
  desert, and barely moves on a plane. The heightfield collider is implicated;
  the *cell-size* explanation for it was measured and **refuted** — halving the
  cell closes 0.7% of the gap. `research/learnings/009`.
- **The three command slots are used** by the tracking envs; Spyder's are not.
- **25 height-scan slots remain reserved and unused.**

---

## Variants

### As built — 16 DoF, driven wheels

`assets/hound16.xml` (flat) and `assets/hound16_desert.xml` (desert). The robot
subtree is **byte-identical** between the two, so nq/nv/nbody and therefore the
observation and action spaces are unchanged, and a flat-world checkpoint loads
on the desert directly.

Envs: `Hound-v0`, `HoundDesert-v0` (torque); `HoundPD-v0`, `HoundPDDesert-v0`
(PD); `HoundPDTrackDesert-v0`, `HoundPDTrackRelDesert-v0` (command tracking).

### Two variants that DO NOT EXIST YET

Recorded here because the difference between them is easy to get wrong, and
getting it wrong silently produces a different robot.

| variant | wheel joint | DoF | what the wheel does |
|---|---|---|---|
| **welded** | fixed / removed | **12** | rigid 8.5 cm cylinder that **slides** |
| **passive** | hinge, no actuator | 16 moving, 12 actuated | **rolls freely**, coasts |
| **driven** (built) | hinge + actuator | 16 | driven |

**A hinge with no actuator is free-rolling, not locked.** These are three
different machines and only the third is built.

The welded variant is the standard-quadruped control arm: 12 leg joints on Go2
kinematics is the morphology every published legged-locomotion reward is tuned
for. Note that **a welded 8.5 cm wheel is still not a foot** — a Go2 foot is a
~2–3 cm hemisphere, so contact-timing reward terms that assume a small point
foot making and breaking contact should be expected to misbehave, differently
rather than less.

Neither variant is built, neither has an oracle, and neither has been measured.
Do not read this section as describing something that exists.

---

## The oracle

```bash
venv/bin/python -m bestiary.robots.hound.check     # 38 assertions
```

**38/38 green means the machine is unchanged.** It says nothing about
`train.py`, `watch.py`, or anything else in the repository — nothing in the
suite imports them. This is a *robot* oracle, not a repository oracle, and
`research/learnings/006` is what happens when the two get confused.

Run it after **any** physics change.
