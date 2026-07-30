# WHELP-16 — a printable 16-DoF wheel-legged quadruped

The hardware sibling of [`hound`](../hound/CARD.md). Hound is the same topology
at Unitree Go2 scale and exists only in MuJoCo; Whelp is 2.21 kg of PETG and
brass that has to survive a floor.

```
4 legs x (abduct + hip + knee + wheel) = 16 actuated joints
2.21 kg · 250 mm trunk · 184 mm wheelbase · 178 mm track · 229 mm standing
12x Feetech STS3215-C018 leg servos, swappable wheel drive
```

| | |
|---|---|
| `spec.py` | every dimension, with provenance on each one |
| `geometry.py` | the stance solve and the kinematic chain |
| `massmodel.py` | mass, centre of mass and inertia per link |
| `torque.py` | the free-body budget and the failure ordering |
| `export.py` | CAD parameters, STLs, `robot.json` |
| `urdf_gen.py` | the URDF and the Isaac Lab actuator config |
| `check.py` | every claim above, asserted against the generated artefacts |
| `ASSUMPTIONS.md` | generated: every number that is still a guess |

```bash
python -m bestiary.robots.whelp.export      # params + robot.json  (no CAD needed)
python -m bestiary.robots.whelp.urdf_gen    # URDF + actuator cfg
python -m bestiary.robots.whelp.torque      # the report
python -m bestiary.robots.whelp.check -v    # assert all of it
```

---

## 1. Read this first: what this machine cannot do

Three limits, all derived rather than estimated, all consequences of one choice
— a 1:345 hobby serial-bus servo. None of them is a defect. All of them will
break a policy that was not trained knowing about them.

**Top speed 0.21 m/s.** The STS3215-C018 turns at 45 rpm no-load, and on a 45 mm
wheel that is 0.21 m/s. Comparable wheel-legged machines run 1.5–2.5 m/s, and
Froude scaling from Go2-W puts a realistic target for a 250 mm body at
1.2–1.5 m/s. Whelp is six times slower than its own envelope allows. The wheel
mount is parametric for exactly this reason; a drive that does ≥300 rpm at
≥0.20 N·m and ≤60 g closes the gap and is a bracket, not a redesign.

**Joint rate 4.7 rad/s.** Published legged-RL configs assume ~30 rad/s — Isaac
Lab's own Go2 config uses `velocity_limit = 30.0`. A policy trained at that rate
learns a gait built out of joint speeds this hardware does not have. It will not
degrade gracefully on the robot; it will fail on the first step.

**Free-fall drop envelope 8 mm at 2× margin, 38 mm before the gear train is at
its estimated failure torque.** This is the one that surprises people, and
section 3 derives it.

**Acceleration limited to 1.5 m/s², by the gearbox rather than by grip.** The
reflected rotor inertia makes the machine behave as though it weighed **59 kg**
for longitudinal acceleration — see section 3.

---

## 2. The stance is solved, and that is what makes the servos adequate

The knee angle is not typed. It is the solution of

```
L_thigh · sin(hip) + L_calf · sin(hip + knee) = 0
```

which puts each wheel axle directly under its own hip pivot. Three payoffs, and
the third is the one that decides whether hobby servos work at all:

1. A contact patch offset from the pivot pushes the machine along the ground the
   moment it takes weight — a robot that drifts while standing still.
2. On a *wheeled* machine it also steers: a constant fore-aft force on a
   free-spinning wheel makes it roll away rather than merely lean.
3. **The hip's static holding torque becomes exactly zero.** The hip's moment arm
   about its own axis *is* that offset, so solving it to zero does not reduce the
   hip's stance torque, it removes it. Measured in the report: 0.043 N·m, which
   is the leg's own weight and nothing else, against a 0.98 N·m continuous
   rating — a 23× margin on the joint that would otherwise hold load all day.

The knee's lever arm is then `L_thigh · sin(hip) = 48.8 mm`, and that single
number multiplies every newton the leg carries. Standing taller shrinks it and
costs travel; the trade is tabulated in `spec.py` next to `stance_hip_rad`.

The stance angle is set by the **three-wheel** case, not the four-wheel one.
Lifting one leg does not put a third of the weight on each remaining leg — a
rigid body on three point contacts is statically determinate, and the answer is
that the diagonal opposite the lifted leg takes **half the weight each** while
the third leg takes nothing. That 1.5× over the naive average is what makes the
stance angle a real constraint. It is also independent of where the centre of
mass sits, so moving the battery does not help.

---

## 3. Why the landing case is the whole design, and static torque is not

Standing is comfortable. Each wheel carries 5.4 N; the worst sustained joint
torque is 0.48 N·m (three wheels down) against a 0.98 N·m continuous rating.

Then it lands. Impact force is not set by weight — it is set by **stroke**:

```
linear absorber, drop h, stroke s:    F_peak / (m·g) = 2(h + s)/s
```

That ratio has no mass in it. **A lighter robot lands just as hard.** Only stroke
helps, and this machine has 8 mm of it, all in the tire.

### The servo cannot help, and this is the single most important finding

The obvious protection is the servo's Torque Limit register: cap it at 50% and
the joint backs off instead of breaking. That works — for *slow* overload.

It does nothing for a landing. A 1:345 reduction reflects rotor inertia to the
joint by the square of the ratio: **0.027 kg·m², about a hundred times the
thigh's own inertia**. Accelerating that through the yield rotation takes on the
order of 100 ms. A landing's contact lasts 40 ms. On that timescale the gear
train is a solid block, regardless of any register.

```
contact duration        40 ms
servo is rigid below   113 ms      →  the gearbox does not yield
```

So the energy goes into plastic and gear teeth, and the only protection is the
envelope:

```
THE DROP ENVELOPE, on tire stroke alone, against the gear-strip torque
  flat, four wheels                 87 mm
  crooked, three wheels             38 mm   ← binding: a crooked landing is normal
  crooked, with the 2x margin       15 mm   ← design_drop_height_mm is 8
```

An impact is measured against the **gear train's** failure torque, not against
stall. Exceeding stall during a landing is not a failure: the motor simply
cannot hold, and since the reflected inertia stops it yielding anyway, the load
reaches the teeth regardless. What matters is whether the teeth survive.

Enforce it in the reward and the terrain curriculum. A policy optimises, and
free-falling is fast.

For scale: if the knee *could* fold 0.35 rad the stroke would be 25 mm and the
same drop would be 2.6 g instead of 4.0 g. That is what a backdrivable actuator
buys, and it is why every reference machine at this scale — Pupper V3 (10:1),
Mini Cheetah (6:1), Upkie, DIABLO (direct drive) — uses one.

---

## 4. The yield chain: what breaks first, and is it cheap

Everything in newtons at the contact patch, because the thresholds are not all
torques — the fuse fails in direct shear and has no moment arm. One unit at one
place is what makes them comparable.

Each wheel carries **5.4 N** standing and **22 N** in a design landing.

| Slow overload | N | |
|---|---|---|
| servo torque-limit register | 30 | **reversible** ← first |
| servo stalls | 60 | reversible |
| output gear strips | 108 | breaks |
| M3 insert strips | 123 | breaks |
| calf, sagittal bending (fatigue) | 210 | breaks |
| sacrificial fuse shears | 280 | reversible |
| horn bolts crush their bosses | 313 | breaks |

| Lateral impact — a fall, or a wheel into a table leg | N | |
|---|---|---|
| abduction servo back-driven | 13 | **reversible** ← first |
| sacrificial fuse shears | 280 | **reversible** ← before the calf |
| calf, lateral bending | 473 | breaks |

The lateral case is what actually destroys printed legs, and it is the direction
in which nothing can yield — the servos rotate about Y and the load is about X
and Z, so the gearbox is not in the load path at all. The fuse is a 2 g printed
shear web that goes at 280 N, below the calf's 473 N. A fall costs a four-minute
reprint instead of a leg.

**Vertical impact has no reversible first item and cannot have one.** To trip
before the gear train's 108 N the fuse would have to be a ~2 mm² web:
unprintable, and close enough to a real landing that it would nuisance-trip. A
fuse that nuisance-trips gets left out, and then you have neither. Vertical
impact is bounded by the envelope, not by a part — 108 N first failure against a
22 N design landing is 5.0× headroom.

---

## 5. Material: why PETG, and when to switch

| | density g/cm³ | tensile XY MPa | HDT @0.45 MPa °C | note |
|---|---|---|---|---|
| PLA / Tough PLA | 1.24 | 51 / 45 | **55** | **disqualified** |
| **PETG** | 1.27 | 47 | 68–78 | marginal, chosen |
| PETG-CF | 1.27 | 47 | 96 | better, needs hardened nozzle |
| ASA | **1.07** | 42 | 93 | −190 g of frame, needs enclosure |
| PA6-CF | 1.17 | 105 dry / **82 wet** | Tg 56.6 | absorbs water from room air |

**PLA is out on heat, not strength.** Its HDT at 0.45 MPa is 55 °C and measured
compressive yield halves from ~80 MPa at 25 °C to ~40 MPa at 50 °C. A bench test
of an STS3215 measured +15 °C in ten minutes holding 1.47 N·m, and the servo's
own operating limit is 60 °C. A bracket bolted to a servo that has been holding
torque sits exactly in the band where PLA creeps. UltiMaker state in writing that
Tough PLA must not exceed 58 °C.

Heat-set inserts make it worse, not better: installing one deliberately melts the
polymer around it, so the material at every insert is the softest, most stressed,
most creep-prone volume in the part — and it is where the servo bolt preload
acts.

**Switching is three numbers.** `material`, `print_density_g_cm3`,
`print_tensile_xy_mpa` in `spec.py`, then re-run export. ASA saves ~190 g of
frame for the cost of an enclosure and the worst interlayer adhesion of the set
(11 MPa).

### The design allowable is a quarter of the datasheet

This is the correction that matters most for "it worked in simulation":

```
47 MPa  datasheet tensile yield
 × 0.42  fatigue at 1e6 cycles      (a leg joint reaches 1e6 in ~140 h of walking)
 × 0.85  orientation allowance
 ÷ 1.48  Kt at a 3 mm filleted corner
= 11.3 MPa  design allowable
```

Isaac Lab will hand you peak joint torques, and a printed part passes that static
check easily. What kills the robot is the fatigue allowable times a stress
concentration at the corner nobody filleted. A bracket "rated" 47 MPa has a real
allowable near 11.

Caveat, stated because it is load-bearing: **the 0.42 is PLA data.** A 2023
review of FFF fatigue found exactly one published study on PETG. It is
`Kind.ASSUMED` in `spec.py` with a named experiment attached.

---

## 6. Printing: the five things that change whether it survives

Everything below is measured, and cited in `spec.py`'s `SOURCES`.

**1. Walls, not infill.** 1→3 perimeters measured **+51%** tensile strength.
0→100% infill raised absolute strength 32% but dropped strength *per kilogram* by
38%. On a mass-budgeted robot that is the wrong direction. Whelp uses 5 walls and
25% gyroid; when you need to move mass, change wall count.

**2. Fillet every internal corner.** A 0.3 mm as-modelled corner has Kt = 2.99 in
bending; a 3 mm fillet, 1.48. Half the peak stress for zero mass and zero print
time. And the penalty is hidden by static testing: a 4 mm hole cost only 8% of
static strength but roughly **halved fatigue life**.

**3. Orientation follows the load, not the bed.** Print thighs and calves lying
down, long axis in the bed plane, channel opening up: sagittal bending then runs
along extrusion lines. The tire is the strict one — print it with the **wheel
axis vertical** so hoop stress is in-plane. TPU 95A is 23.7 MPa in XY and
6.4 MPa in Z, and elongation at break falls from >560% to 82%. Printed on its
side, every tread block hangs off an interlayer weld at a quarter strength.

**4. Interlayer adhesion is a thermal-history problem, not a temperature
setting.** With the nozzle *fixed* at 180 °C, raising bed/ambient from 30 to
120 °C raised joint strength 33% and dropped the fraction of specimens failing at
the weld from 92% to 8%. What you control is time-above-Tg at the interface: keep
the part warm, turn part-cooling **down** on thick load-bearing parts, run PETG
hot (245–255 °C), slow down.

**5. Do not anneal.** Measured gain is +6 to +8%. Measured cost on an ASTM bar is
±3.3 mm of length change, up to 22 mm worst case. Scaled to a 110 mm thigh that
is ~2 mm, which destroys the horn bore fit, the insert pitch and every bearing
seat.

**Fasteners.** M3 heat-set inserts, 4.2 mm bore, 1.8 mm of wall around them.
Measured in PETG: torque-to-failure 3 N·m for an insert against 1 N·m for a screw
straight into plastic — and 3 N·m is above every joint torque on this robot,
which is what keeps the fastener out of the failure chain. Orient bolt axes to
lie in the XY plane where you can; a boss pulled axially by a bolt loads pure
interlayer adhesion.

**Every joint is double shear.** The servo's output bushing is sized to transmit
torque, not to carry a leg's bending moment. Hang a 100 mm calf off one side and
every impact wears the bushing oval; the play becomes backlash, the backlash
becomes a policy that cannot repeat itself, and because it arrives slowly it gets
blamed on the controller. The STS3215 is a *dual-shaft* servo — there is an idler
boss opposite the horn — so the far-side 623ZZ rides on the servo's own casting
and is coaxial by construction rather than by assembly.

---

## 7. Weight

```
link            g     of which
trunk         837     4 abduct servos 220, battery 200, compute 75, wiring 120
4x hip        372     hip servo 55 each
4x thigh      373     knee servo 55 each
4x calf       352     wheel drive 55 each
4x wheel      276     hub + TPU tire (unsprung, at the end of the leg)
             ----
TOTAL        2210 g   servos 880 g (40%), payload 485 g (22%)
```

Two things this model gets right that a naive one does not:

**Effective density, not filament density.** CAD volume × 1.27 g/cm³
overestimates by ~2× — but *not* by the infill fraction. On a part whose smallest
dimension is under ~20 mm the walls are most of the volume: 5 perimeters at
0.4 mm is about a third of a typical bracket, so 25% gyroid gives an effective
solid fraction near **0.50**, not 0.25.

**Which link a servo belongs to.** A servo's body moves with the link it is
*bolted to*, not the one it drives. The knee servo is part of the thigh; the
wheel drive is part of the calf. Getting it backwards moves 55 g by 100 mm, which
is a real change in the leg's inertia about the hip and therefore in what a
policy learns is possible.

**Then weigh the parts.** Write `mass_measured.json` as `{"thigh": 41.2, ...}` in
grams and the model rescales density per part, preserving the centre of mass. Do
this before training anything long: an inertia tensor built on a 10%-wrong mass
is a sim-to-real gap you will spend a week blaming on the controller.

---

## 8. Sim-to-real

### The actuator is not an effort source

The STS3215 has four modes — position, constant speed, open-loop PWM, step — and
**no torque or current mode at all**. The policy emits a position target; the
servo closes its own loop internally with firmware integer gains (P=32, D=32,
I=0), its own deadband, and its own saturation. An ideal-effort actuator in
simulation is a model of a robot nobody owns.

It also has ~1.75° of total dead motion: 0.87° measured backlash plus the
deadband. With `action_scale = 0.25 rad`, a policy output change below ~0.035
commands *less than the deadband* and produces literally nothing on hardware.

Set these, in order of how often they are the actual cause of a failed transfer:

| | value | why |
|---|---|---|
| **latency** | randomise 0–3 control steps | ranked cause #1 by three independent papers |
| **armature** | 0.027 kg·m² | ~100× the link's own inertia; omitted → massless whip |
| **torque-speed clip** | `DCMotorCfg`-style saturation | else sim delivers 3 N·m at 15 rad/s |
| **velocity limit** | 4.7 rad/s, via `velocity_limit_sim` | `ImplicitActuatorCfg` **ignores** `velocity_limit` |
| **joint zero offset** | ±0.02 rad | horn splines quantise your assembly zero |
| **friction / damping** | 0.068 N·m / 0.56 | identified for this exact servo |

**Scale the randomisation to a 2.5 kg robot.** Isaac Lab's default
`add_base_mass` is (−5, +5) kg — twice the whole machine; copying it trains a
policy for a robot that sometimes weighs −2.5 kg. Use ±0.15 kg on the trunk,
U(0.9, 1.1) on link masses, ±10–30 mm on CoM. And note that Isaac Lab's default
friction range is `(0.8, 0.8)` and restitution `(0.0, 0.0)` — degenerate ranges
that randomise nothing, which people copy believing they do.

**Wheel friction is a trap that looks like success.** Ascento report it
verbatim: with too much grip "the robot learns to use its grip between the wheels
and the step to go up… it does not transfer due to unmodelled tire dynamics."
Train with a wide range (0.1–3.0) and specifically check whether the policy's
obstacle behaviour depends on grip. If it does, it will not transfer.

### Isaac Lab specifics that fail silently

- **Wheels must be velocity-driven.** A URDF `continuous` joint imports as a
  revolute with FLT_MAX limits, and PhysX then refuses a drive target outside
  ±2π — a position-driven wheel breaks after ~3.2 revolutions. Two actuator
  groups: 12 position, 4 velocity with `stiffness = 0`.
- **Wheel collision is a `<cylinder>` primitive**, deliberately. A meshed wheel
  under Convex Hull is cooked to a vertex limit, becomes an N-gon, and rolls with
  N contact impulses per revolution. That is what wheel chatter *is*.
- **A link with no `<collision>` gets no collider, silently.**
  `collision_from_visuals` defaults to False.
- **A link with zero inertia is read as infinitely massive.** And PhysX does not
  check the triangle inequality, so a physically impossible tensor simulates
  happily and a policy will find the exploit. `urdf_gen.py` refuses to emit one.
- **Degrees hide in three places**: the importer converts angular *limits* to
  degrees, `max_angular_velocity` is deg/s next to `max_linear_velocity` in m/s,
  and the converter multiplies revolute drive stiffness by π/180. Anything wrong
  by 57.3× is this.
- **`merge_fixed_joints` may not merge** since Isaac Sim 5.1. Print
  `len(robot.body_names)` after conversion and compare against 17.

### The control loop is bounded by feedback, not commands

Sixteen servos on one half-duplex 1 Mbps bus. A broadcast sync-write returns
nothing and is fast; a sync-read serialises sixteen replies. Measured on
comparable hardware: ~2300 Hz write against 35–57 Hz read.

- **Train at 50 Hz decimation.** The 284 Hz wire-time figure assumes zero return
  delay, zero host overhead and no Python.
- **Set Return Delay Time (register 7) to 0 and verify.** Sources disagree on
  its default; at 250 it adds 8 ms to every sync read and caps you at ~87 Hz on
  its own.
- **Set the USB-serial latency timer to 1 ms.** Linux defaults FTDI to 16 ms,
  which caps you at ~31–60 Hz regardless of baud, and presents as "the servos are
  slow" rather than "the driver is misconfigured".
- **Disable the servo's own motion profile** (Acceleration = 254). Otherwise a
  50 Hz policy is fighting a trapezoidal trajectory generator and the real joint
  follows a rate-limited version of the policy output that exists nowhere in your
  simulator.
- Registers are **P, D, I at 21, 22, 23** — not P, I, D. Goal/Present Position
  are **15-bit sign-magnitude**, not two's complement; packing a negative as
  int16 sends the leg 32768 steps away at full torque.

### Thermal death is real and unmodelled

The STS3215 has no thermal shutdown and a 60 °C limit. A bench test reached
71 °C on ±90° oscillation; a 1.5 kg static load added 15 °C in ten minutes. RL
policies produce high-frequency micro-corrections that a hand-tuned gait never
does. You need an action-rate penalty, a torque penalty, **and** a hardware
temperature watchdog on the bus — the servos report temperature, so read it.
Otherwise the first successful policy destroys twelve servos over an afternoon
while looking fine.

---

## 9. What to measure before printing a leg set

`ASSUMPTIONS.md` is generated and lists all of them. The short version — 15
load-bearing numbers are currently guesses, and most retire in under twenty
minutes:

| measure | how | retires |
|---|---|---|
| **the fit-check coupon** | one 30-min print | horn bolt circle, bolt count, insert bore, bearing fit, servo mount pattern, your own anisotropy ratio |
| fuse shear strength | one fuse in a vice + luggage scale | the entire lateral yield chain |
| tire stroke | press a tire against a scale, log force vs deflection | the whole impact budget |
| servo gear break torque | sacrifice one servo with a lever | whether an overload is reversible |
| a weighed 20 mm cube | one print | effective density → every URDF mass |
| part masses | a scale, `mass_measured.json` | every inertia tensor |

No primary source publishes the STS3215's horn bolt circle or its case
mounting-hole pattern. Feetech publish torque, speed and current and no
dimensioned drawing. That is why the coupon exists and why it is the first thing
to print.

---

## 10. Build order

1. `python -m bestiary.robots.whelp.check` — should be green.
2. `sudo apt install openscad`, then
   `python -m bestiary.robots.whelp.export --stl --check-mass`.
3. Print **`fitcheck`** first. Measure. Update `spec.py`. Re-run export.
4. Print one leg's worth. Assemble one leg. Check the double-shear pivots turn
   freely and that nothing fouls across the full joint ranges.
5. Weigh every part → `mass_measured.json` → re-run export → re-run `urdf_gen`.
6. Drop-test one leg from the envelope height onto foam. Then onto the floor.
7. Only then print the other three.

First deployment: gantry or tether, foam floor, a hard software clamp on
commanded joint velocity and on the delta between consecutive position targets,
and a bus-level current limit. Every published success at this scale had that.
The people who report "it worked in sim then it fell and broke" are the ones who
skipped latency modelling and went straight to the floor.

---

## 11. Known gaps

- **No `.scad` file in this package has ever been rendered.** OpenSCAD is not
  installed on the machine this was authored on. `check.py` lints the sources and
  reports this as a `GAP`, not a pass. The lint now also covers the two bug
  classes that a subagent found by reading and the lint had missed: OpenSCAD has
  no implicit string concatenation (two adjacent literals are a **parse error**,
  and one in a `use`d library is fatal whether or not the module is called — a
  single wrapped `assert` message in `util.scad` meant nothing in this tree
  rendered), and `use <>` imports modules but **not variables**, so `EPS` was
  silently `undef` throughout `sts3215.scad` and propagated into geometry rather
  than raising.

- **`abduct_bracket.scad` and `trunk.scad` do not agree on how the bracket leaves
  the trunk, and this is the top item to resolve.** The kinematics puts the hip
  pivot 36 mm outboard of the abduction pivot in **Y**, at |y| = 72 mm — 20 mm
  beyond the trunk's 52 mm half-width — so the bracket must cross the trunk's
  *side*. `trunk.scad` instead puts a plain journal through the *end* wall,
  coaxial with the abduction axis, which is a better bearing and a different
  topology. The bracket is authored to the kinematics, because the URDF is what a
  policy is trained against; the header of that file states the disagreement and
  lists the three ways out, cheapest first. Rendering both together is one
  `apt-get` away and is the only honest way to settle it.


- The fatigue knockdown is PLA data applied to PETG.
- The structural thresholds are first-order beam theory on idealised sections.
  They are intended to establish the *ordering* of failures with enough margin
  that being 40% wrong about any one does not reorder it — not to be believed as
  values.
- No wheel-drive alternative has been specified to a part number, only to a
  requirement (≥300 rpm, ≥0.20 N·m, ≤60 g).
