# 0006 — What the reward economics may and may not assume, going into the first evaluated arm

**Date:** 2026-07-30 · **Status:** accepted · **Robot:** hound

## The decision

A second independent refutation attacked `0005`'s successor reasoning and the
two `lin_vel_y` edits made under it. **This decision records what the reward
economics is now permitted to assume before the first evaluated Hound arm
launches.** Every figure below is printed by
`research/scripts/0006_reward_economics_refutation.py`; nothing else in the
refutation may be cited.

Five things are settled and one is reversed.

**1. The binding freeride is not a stander. It is point-and-park, and
`learnings/015` is not closed.** Under `heading_command=True` with
`heading_control_stiffness=0.5`, the scored yaw command is a feedback signal, so
a machine that yaws to the commanded heading and then holds still drives *its
own yaw command to zero* and collects `track_ang_vel_z_exp` at exactly 1.0. On
the inherited command set that behaviour earns **0.4630 of nominal maximum
against the plain stander's 0.2255 — 2.05× the fake `0005` B1 priced** — and,
net of `check_hound.py`'s own assumed penalty basket, it earns **103.40%** of a
competent driver's net, i.e. more than driving. On the committed `±0.3` range it
earns **1.628×** the stander and **64.95%–80.19%** of a driver's net. Defeating
it requires a change on the **lin/yaw weight axis** — or a change in what the
yaw channel scores — and cannot be reached from the contact term or the command
range, because the parker contacts the ground exactly as a driver does and buys
its whole advantage on the yaw channel.

The refutation's own headline — parker 0.018987/step at **92.42%** of a driver's
0.020545/step, margin **1.0821×** — is **reproduced to 1e-6** and is therefore
citable, but only with four assumptions attached, two of which are defects:
`c_y ≡ 0` (the collapsed range), `ρ = 0.46` as the driver's competence (the
defect the same refutation names in its own provenance list), a penalty basket
from a table that was never committed, and a static-stance floor of
0.000058/step. **The direction survives every correction; the 1.0821× does
not.** See *What we actually verified*.

**2. `0005`'s `lin_vel_y` premise is corrected by this decision, not edited into
it.** `0005` recorded lateral velocity as unachievable for this morphology and
put an unremovable ceiling of 0.4409 on the largest reward term. The
*impossibility* half is right: forward kinematics over 114,244 wheel
configurations spanning the whole joint range gives **max |axle_x| =
2.220e-16** — no joint steers. The *inference* is wrong. Rolling direction is
`axle × normal`, whose body-frame y-component is `sin(φ)·n_x`, and `n_x` is
nonzero whenever trunk pitch differs from the local ground plane — which twelve
leg joints set independently of the terrain. At the abduct limit φ = 0.8 rad on
a 20° pitch offset, **|v_y|/|v| = 0.3322**. And because the abduct axis is
`(1, 0, 0)` on all four legs, a common roll gives all four wheels the same
axle, so the no-lateral-slip constraint matrix has **rank 3 and a 3-dimensional
nullspace** containing the rolling direction itself: sustained lateral motion is
available by pure rolling, no stepping gait, no slip.

**3. The budget guard was loosened 1.5940× by our own edit, and it is the guard
that exists to catch `learnings/011`.**
`check_hound.py::check_reward_budget_against_011_and_015` divides every penalty
by "achievable income", and that denominator is a function of the `lin_vel_y`
range: **0.018821/step inherited, 0.030000/step at the `(0,0)` collapse,
0.027838/step at the committed `±0.3`**. The same unchanged penalty basket of
0.005299/step therefore reports as **28.15%, 17.66% and 19.03%** of income. No
weight moved between those three numbers. The 30% flag fires 1.5940× later
under a zero-width command range.

**4. The `(0,0)` collapse silently moved the terrain curriculum.** The demote bar
in `terrain_levels_vel` is proportional to `‖command_xy‖`, and `E‖c_xy‖` fell
**0.765196 → 0.500000, a 34.66% drop**, with the promote bar (`size[0]/2` =
4.0 m) unchanged. Any comparison spanning that edit is confounded on terrain
difficulty, not only on reward.

**5. "Self-pricing through physics" is refuted, and it was the elegant claim.**
The argument was that below-threshold leaning needs no penalty because weight on
linkage unloads the wheels and traction is proportional to normal load. Traction
is nowhere near binding. The friction cap is **3.1904 N·m per wheel**
(2.35 lbf·ft); demand at the measured 2.0 m/s² (6.6 ft/s²) saturation is
**0.7227 N·m, 22.65% of the cone**; steady cruise needs only the 0.3990 N·m
wheel frictionloss, **12.51%**. With a limb resting at 40 N (9.0 lbf) the cone
is still only 16.5% used at cruise. Top speed is set by a velocity limit and
thrust by a wheelie, not by force — and **no reward term prices wheel torque at
all**: `dof_torques_l2` is scoped to the legs, and the only wheel-scoped term,
`dof_acc_wheel_l2`, prices *acceleration* and is therefore **exactly zero at
constant speed**. A policy cruising with a limb leaning on the ground earns full
income at zero cost. The exploit is live.

**6. Two design choices survive.** The **count form over the graded form**:
`undesired_contacts` sums booleans, so its per-step charge is bounded by
`n_bodies · |w| · dt`, while `contact_forces` takes max-over-history and clips
only at min = 0 — a 5,000 N (1,124 lbf) solver impact spike costs 52.80% of
income and a 50,000 N spike 537.75%, unbounded in the peak. And **all of the
refutation's arithmetic reproduces to six digits** wherever its inputs are
stated.

Confidence, split by class of claim, because they are not equally strong:

| claim class | confidence | why |
|---|---|---|
| source-read facts (heading law, kernel forms, term scopes, absence of a termination reward) | **high, ~95%** | read from named files at named lines, all re-read for this decision |
| the axle impossibility (no joint steers) | **~98%** | analytic *and* reproduced by FK over the committed MJCF at 114,244 configurations |
| the guard-denominator gain, the demote-bar shift, the basket shares | **high, ~95%** | pure arithmetic over config literals, and the script reproduces `check_hound.py`'s own code path |
| point-and-park's income *ordering* (it beats the stander, by a lot) | **~90%** | follows from the heading law alone; the only way out is if the yaw transient costs more than the 0.006983/step it buys, which is unmeasured |
| point-and-park's *magnitudes* | **~70%** | the behaviour model is stated, not measured; no Hound policy exists, and the driver's penalty basket is `check_hound.py`'s own [ASSUMED] operating point |
| the lateral-rolling capability | **~75%** | the kinematics and the nullspace are exact; whether a policy can *hold* a pitched, abducted posture on a dune face while rolling is unmeasured |
| the wheel-torque gap | **~90%** | the cone arithmetic is static and simple; the "no term prices it" half is a source fact |

## Why we asked

`0005` closed with two gates and named `lin_vel_y` the cheapest one. Two commits
then answered it without a measurement: `db88770` collapsed the range to
`(0, 0)` and `b2e2634` reopened it at `±0.3`. Both are edits to the largest term
in the table, taken on an argument rather than a rollout, immediately before the
first multi-hour Hound run — and `0005`'s own "dangerous to act on" list opens
with *launching the reward table as designed*.

So the question this decision answers is not "what is the right reward" but
**"which of the things we now believe about this reward's economics are we
allowed to build a run on."** The refutation that produced it existed only in a
chat transcript and would have been lost.

## What we actually verified

All source re-read 2026-07-29/30. Every number is printed by
`research/scripts/0006_reward_economics_refutation.py`, which imports `Spec` and
reads the committed MJCF; figures the script prints as `claimed` are **not**
cited anywhere and are carried only so the record shows what moved.

### The heading law, and why a parker beats a stander

`velocity_command.py:184-194`, read verbatim:

```python
heading_error = math_utils.wrap_to_pi(
    self.heading_target[env_ids] - self.robot.data.heading_w.torch[env_ids]
)
self.vel_command_b[env_ids, 2] = torch.clip(
    self.cfg.heading_control_stiffness * heading_error,
    min=self.cfg.ranges.ang_vel_z[0],
    max=self.cfg.ranges.ang_vel_z[1],
)
```

With `rel_heading_envs = 1.0` this runs on **every** environment **every step**,
so the sampled `ang_vel_z` is always discarded and `ang_vel_z=(-1,1)` acts only
as the clip bound. Two consequences the record did not carry:

- A machine whose heading never changes keeps the heading error it drew, so its
  scored yaw command is `clip(0.5·U(-π, π), ±1)` and it earns a yaw kernel of
  **0.287431** (`0005` B1 published 0.2876 for the same quantity, computed
  independently).
- A machine that yaws once to `heading_target` and then stops has heading error
  zero, therefore **yaw command zero**, therefore — being still — yaw error zero
  and a yaw kernel of **exactly 1.0**, forever, for free.

The gap between those two is `0.5 · (1 − 0.287431) · dt = 0.006983/step`, and it
is independent of the linear command range because the lin term cancels.

### The freeride table

Model stated in full in the script's `freeride_table` docstring. Briefly: the
2% `rel_standing_envs` branch is priced separately; the stander and the parker
hold `v = 0`; the competent driver tracks `c_x` and `c_ω` exactly and does not
make `c_y` (the conservative reading, since the lateral capability is the
contested claim); the stander and parker pay no penalties and the driver pays
`check_hound.py`'s own [ASSUMED] basket. Per step:

| `lin_vel_y` | stander | **point-and-park** | driver, gross | driver, net | park/stander | park/driver net |
|---|---|---|---|---|---|---|
| inherited `±1.0` | 0.007229 | **0.014213** | 0.019044 | 0.013746 | 1.966× | **103.40%** |
| collapsed `(0,0)` | 0.012061 | **0.019044** | 0.030000 | 0.024701 | 1.579× | 77.10% |
| committed `±0.3` | 0.011127 | **0.018110** | 0.027882 | 0.022583 | 1.628× | 80.19% |

**Machinery check, and it is the reason to believe the rest.** With the 2%
standing branch dropped — which is how `0005` computed it — this script returns
the additive stander at **0.2255** of nominal maximum on the inherited command
set, reproducing `0005` B1's independently computed 0.2255 exactly. On the same
command set point-and-park gets **0.4630**.

### The refutation's own triple, reproduced exactly, and what it assumes

The transcript's triple does not reproduce on the committed range — matched
against all three, the stander and parker are out by −0.004774 (inherited),
+0.000058 (collapsed) and −0.000876 (committed). The near-match on the collapsed
range is the clue, and following it recovers the model completely:

| | computed | claimed | delta |
|---|---|---|---|
| stander | 0.012003 | 0.012003 | +4.38e−07 |
| **point-and-park** | **0.018987** | 0.018987 | −3.83e−07 |
| driver | 0.020545 | 0.020545 | −2.62e−07 |
| park / driver | **0.9242** | 0.9240 | +1.60e−04 |
| driver / park | **1.0821** | 1.08 | — |
| park / stander | 1.5818 | — | — |

**All three reproduce to 1e−6, so the figures are citable — provided four
assumptions travel with them.** Two are defects:

1. **`c_y ≡ 0`.** The y channel pays a still machine exactly 1.0, which is the
   collapsed range `b2e2634` replaced. On the committed `±0.3` a still machine's
   y factor is 0.8919.
2. **`ρ = 0.46` as the driver's competence — and the same refutation names this
   as a defect in its own provenance list.** `reward-composition.md`'s appended
   refutation established that `ρ = |0.271 − 0.50| / 0.50` is the relative error
   *of the fixed 0.271 m/s trot*. There is an identity here that makes it fatal
   rather than merely sloppy: at `ρ = 1` the relative-error mean kernel is
   **0.441041** and the stander's own lin factor is **0.441041** — `ρ = 1` *is*
   the stander on the speed channel. So the "driver" is parameterised on the same
   axis as the fake it is being compared to, which is exactly the circularity
   `reward-composition.md`'s refutation caught the first time.
3. **A penalty basket from a table that was never committed:** 0.004142/step
   against the committed table's 0.005299/step. It charges `dof_acc_l2` at −1e−7
   (committed: −2.5e−7), 8.0 N·m rms torque (`check_hound.py`: 5.0), and a
   contact term at 5% duty that the committed table does not have at all.
4. **A static-stance floor of 0.000058/step**, derived from
   `SPEC.static_torques()` — abduct +1.9386, hip −0.9920, knee −5.2425 N·m — plus
   the deterministic action-rate component. This one is a *correction to* the
   table earlier in this section, which charges a still machine nothing and
   therefore flatters the fake.

**So the 1.0821× margin is a margin between a parker priced on the collapsed
range and a driver priced at the fixed trot's own relative error under an
uncommitted table.** The direction survives all four corrections — the committed
table gives 64.95%–103.40% — but the margin itself must never be quoted without
them.

### The axle, and the rolling direction

The MJCF's own joint axes, read from `assets/hound16pd.xml`: `abduct` about
`(1, 0, 0)`; `hip`, `knee` and `wheel` all about `(0, 1, 0)`, on all four legs.
`R_y` leaves `(0, 1, 0)` invariant, so the axle in the trunk frame is
`R_x(φ)·(0,1,0) = (0, cos φ, sin φ)` for any hip/knee/wheel angle. Forward
kinematics over 114,244 wheel configurations spanning the full joint range gives
**max |axle_x| = 2.220e-16**, which is float64 noise.

But `d = axle × normal` gives `d_y = sin(φ)·n_x`, and `|d_y|/|d|`:

| φ [rad] | 10° | 18.4° | 20° | 38.8° | 45° |
|---|---|---|---|---|---|
| 0.20 | 3.5% | 6.4% | 6.9% | 12.6% | 14.2% |
| 0.40 | 7.3% | 13.2% | 14.3% | 25.6% | 28.6% |
| 0.60 | 11.8% | 21.1% | 22.8% | 39.4% | 43.5% |
| 0.80 | 17.6% | **30.9%** | **33.2%** | **54.2%** | 58.9% |

18.4° and 38.8° are `0005` B6's measured median tile slopes at terrain
difficulty 0.25 and 0.50, so the bolded cells are the ground this machine is
actually trained on. The pitch offset is *relative to the local ground plane*,
so it is reachable on flat ground too — twelve leg joints set trunk pitch
independently of the terrain.

The no-lateral-slip constraint `a_i · (v + ω × r_i) = 0` for four wheels is a
4×6 matrix. With a common abduct roll all four axles are identical, and the
matrix has **rank 3, nullspace dimension 3**; the rolling direction itself lies
in it to machine precision. Verified stable under a 2 cm (0.8 in) random
perturbation of the contact points, so the rank is not an artifact of exact
symmetry.

### What the y channel charges when `c_y ≡ 0`

Not a skid price. With the command pinned to zero, `e_y` is whatever lateral
velocity the machine has, and that velocity is what a pitched, abducted,
purely-rolling machine *produces*. Cost as a fraction of
`track_lin_vel_xy_exp`, at φ = 0.8 rad:

| speed | 18.4° | 20° | 38.8° |
|---|---|---|---|
| 0.20 m/s (0.66 ft/s) | 1.52% | 1.75% | 4.59% |
| 0.30 m/s (0.98 ft/s) | 3.38% | 3.89% | 10.04% |
| 0.50 m/s (1.64 ft/s) | 9.11% | 10.45% | 25.47% |
| 0.80 m/s (2.62 ft/s) | 21.70% | 24.61% | 52.88% |

0.20 m/s is the terrain curriculum's own promote threshold (4.0 m of
displacement over a 20 s episode), so the top row is the slowest speed at which
this machine is permitted to progress at all. **So `(0,0)` taxes
stance-widening while pitched — the posture that stabilises a wheeled machine on
a dune face — and calls it skid.** `±0.3` keeps the channel falsifiable at
0.027838/step of income, 1.4791× the inherited figure.

### Wheel-torque economics

| quantity | value | share of the cone |
|---|---|---|
| static load per wheel | 41.705 N (9.4 lbf) | — |
| friction cap, μ = 0.9 | 3.1904 N·m (2.35 lbf·ft) | 100% |
| `gear_wheel` | 3.0000 N·m | 94.0% |
| demand at measured saturation 2.0 m/s² | 0.7227 N·m | **22.65%** |
| demand at the CARD wheelie limit 5.22 m/s² | 1.8866 N·m | 59.13% |
| demand at steady cruise (frictionloss only) | 0.3990 N·m | **12.51%** |
| cruise, with a limb resting at 40 N (9.0 lbf) | 0.3990 N·m of 2.4254 | 16.5% |

**A disagreement inside our own record, recorded rather than resolved.**
`CARD.md`'s traction budget and `robots/hound/check.py:276` both say the cone is
"barely 5% used" at saturation; the static-load arithmetic above says 22.65%.
Both agree the cone does not bind — `check.py:321` asserts "the limit is
UNLOADING, not the friction cone" — but they are not the same number and neither
has an instrument recorded. This decision cites the computed one.

### The contact term

`undesired_contacts` (`isaaclab/envs/mdp/rewards.py:272-282`) sums booleans:
bounded at `n_bodies · |w| · dt`. `contact_forces` (`:296-306`) computes
`max(‖F‖, over history) − threshold` and clips **only at min = 0**, so an impact
spike enters unbounded — 52.80% of income at a 5,000 N peak, 537.75% at 50,000 N.
**The count form is the safer object and that choice survives.**

For one contacting body to cost at least half of achievable income,
`w · dt ≥ 0.5 · I` at `I = 0.027838/step` gives **w ≥ 0.6960**.

And `velocity_env_cfg.RewardsCfg` contains **no termination reward term** —
eleven terms, none of them `is_terminated` — so `V(terminate) = 0` exactly.

## The trigger to revisit

Reopen when **any** of these becomes true:

1. **A lateral-velocity rollout lands.** One scripted rollout on the Isaac
   desert — minutes, no training — commanding a pitched, common-abduct posture
   and reporting the largest sustained body-frame `|v_y|` the machine holds. That
   number replaces the 0.3322 kinematic bound with a measurement, sets the
   command range, and re-prices every income figure in this decision. **Still
   the cheapest gate, and now it has a specific posture to test rather than
   "push it sideways".**
2. **A point-and-park probe is run.** Two 1-seed arms at the same geometry,
   differing only in the lin/yaw weight ratio, judged on `vx_span_ratio` and a
   per-cell grid. If the shipped ratio produces a machine that yaws to heading
   and parks, claim 1's magnitudes become a scored prediction.
3. **The yaw transient is priced.** The whole parker advantage is 0.006983/step.
   If the cost of yawing to a fresh heading every 10 s exceeds that, the exploit
   closes itself and claim 1's confidence drops sharply. One rollout measures it.
4. **`check_hound.py`'s income denominator is made explicit.** The guard should
   report the *command range it priced against* on the same line as the
   percentage, so a future range edit cannot move the flag silently. Until then
   every "% of income" in the record needs its range attached.
5. **Any wheel-torque or wheel-power term is proposed.** Today nothing prices
   wheel torque and a resting limb is free. The moment a term is proposed, the
   22.65%/12.51% cone figures become the calibration for it.
6. **A Hound policy exists at all.** Every [ASSUMED] operating point in the
   penalty basket — 250 rad/s² of leg acceleration, 5.0 N·m rms torque, 0.15 m/s
   of vertical velocity — becomes measurable, and the driver column of the
   freeride table stops being a bracket.

## What we gave up

**We gave up the claim that the stander fence is the fence that matters.** `0005`
priced the additive form's freeride at 0.2255 and reasoned about whether command
geometry closes it. The number is right and the question is the wrong one: on
the same command set a parker gets 0.4630, and net of penalties it out-earns the
driver. Everything downstream of "the stander is the fake to beat" has to be
re-derived, including the parts of `0005` B1 and B2 that are otherwise sound.

**We gave up `0005`'s `lin_vel_y` premise, without editing it.** `0005` remains
`accepted` and is read as written; the ceiling of 0.4409 it records is the
ceiling *for a machine that cannot roll laterally*, and this decision holds that
such a machine is not this one. Decisions supersede.

**We gave up the elegance of self-pricing.** "Below-threshold leaning needs no
penalty because physics charges for it" was the nicest idea in the refuted pass
and it is simply false here: the cone is slack by a factor of four at the
machine's own measured saturation, and no term reads wheel torque. Protecting
the legs now needs a designed term with a stated weight — reserved work, its own
gate, not a scope change.

**We gave up comparing across the `lin_vel_y` edits.** Three things moved
together: the largest term's ceiling, the guard's denominator, and the terrain
curriculum's demote bar. Any run started before `b2e2634` is not comparable to
one started after it on income, on penalty share, or on terrain difficulty, and
no post-hoc correction recovers the comparison.

**We did not settle a reward table.** This decision says what may be assumed.
It does not choose weights, and deliberately: choosing them is new mathematics
with its own gate.

## How we would know this was wrong

- **The yaw transient costs more than 0.006983/step.** Then point-and-park is
  not a freeride, the whole headline collapses, and `0005`'s stander analysis was
  the right analysis after all. This is the cheapest way for this decision to be
  wrong and it is one rollout.
- **A trained policy neither parks nor tracks, but does something third.** The
  freeride table is an argument about *income optima*, and PPO with an entropy
  bonus on a terrain curriculum need not find one. If the first arm comes back
  driving with `vx_span_ratio ≥ 0.5`, the whole economics framing is a weaker
  predictor than this decision assumes — exactly as `0005` already allowed.
- **Sustained lateral rolling turns out to be unholdable.** The nullspace is
  exact and the kinematics are exact, but holding a pitched, common-abduct
  posture on a 38.8° dune face while rolling is a *control* problem this decision
  says nothing about. If a rollout cannot hold it for a second, `±0.3` is
  charging for something as unreachable as `±1.0` was, and `(0,0)` was right for
  the wrong reason.
- **The `(0,0)` curriculum shift turns out not to matter.** If the terrain level
  equilibrates near 0.25 regardless — which `0005` B6 predicts on slope
  statistics — then the demote bar was never the binding constraint and the
  34.66% shift is a confound that confounded nothing.
- **`check_hound.py`'s assumed basket is badly wrong once measured.** Every
  driver-column figure and every "% of income" here rides on 250 rad/s²,
  5.0 N·m, 0.15 m/s and 0.1 of action rate. If the real operating point is 2×
  off in either direction, the margins move by more than the effects this
  decision is arguing about.
- **The cone does bind somewhere we did not look.** The 22.65% figure is static
  load on level ground. On a 38.8° slope with the front pair unloaded by a
  wheelie the per-wheel normal force is much smaller, and the arithmetic here
  does not cover that case. If it binds there, "traction never prices anything"
  is too strong.

## Dangerous to act on as written

Recorded explicitly, because it is the most operationally useful part.

- **Re-enabling `undesired_contacts` at −1.0 over the twelve leg bodies. This
  makes self-termination optimal.** One body in contact costs **71.84%** of
  achievable income; twelve cost **0.2400/step = 862.12%**; thirteen (adding the
  trunk, as `robot_lab`'s `"^(?!.*_foot).*"` would resolve here) cost
  **933.96%**. And `V(terminate) = 0` **exactly**, because the inherited table
  has no termination reward term. So a fallen machine has strictly negative
  value and flopping the trunk down — which `terminations.base_contact` accepts —
  is the optimal action. Being down costs **8.62× per step what being up
  earns**, so a get-up only pays if it completes inside `R / 9.62` of the
  remaining episode time `R`: **2.08 s at 20 s left, 1.04 s at 10 s, 0.52 s at
  5 s**. Nothing in this repository has demonstrated a Hound get-up at all.
- **Citing "point-and-park collects 92.4% of the driver's net, margin 1.08×"
  bare.** The figures are exact (92.42% / 1.0821×) and citable, but only with all
  four assumptions above — `c_y ≡ 0`, `ρ = 0.46`, an uncommitted penalty basket,
  and the stance floor. Quoted alone they read as a statement about the committed
  table, where the computed figures are 64.95%–80.19% (committed range) and
  103.40% (inherited range). **The margin is the fragile part; the direction is
  not.**
- **Citing `ρ = 0.46` as a driver's competence.** `reward-composition.md`'s
  appended refutation already established that `ρ = |0.271 − 0.50| / 0.50` is the
  relative error *of the fixed 0.271 m/s trot itself*. Using it as a tracker's
  control error compares the fake against itself in a different
  parameterisation. A command-following policy's relative control error has still
  never been measured.
- **Citing "one tap per second is 4% of steps."** At `dt = 0.02 s` a second is
  50 steps, so one tap per second is **2%**.
- **Citing "2.0 m/s², wheelie-limited [VERIFIED]."** `CARD.md` gives the wheelie
  limit as 88.78 N → **5.22 m/s²** and reports **2.0 m/s²** separately as
  *measured saturation*. They are different numbers describing different things,
  and attaching `[VERIFIED]` to the second under the first's mechanism lends a
  measurement's authority to a mechanism the measurement contradicts.
- **Marking `init_std = 1.0` ASSUMED.** It is read from source
  (`anymal_c/agents/rsl_rl_ppo_cfg.py:23`,
  `GaussianDistributionCfg(init_std=1.0)`). Marking a source-read value ASSUMED
  is the mirror of the usual defect and just as costly — it invites a later cycle
  to spend a probe re-deriving a config literal.
- **Citing any "% of income" without naming the command range it was divided
  by.** 28.15%, 17.66% and 19.03% are the *same penalty basket*.

## Sources

- `research/scripts/0006_reward_economics_refutation.py` — every figure cited
  above. Pure arithmetic plus MuJoCo forward kinematics; no GPU, no Isaac Lab
  runtime, no simulation stepping.
- `src/bestiary/isaac/hound_desert_env_cfg.py` (the reward table and the
  `lin_vel_y` override), `src/bestiary/isaac/check_hound.py`
  (`check_reward_budget_against_011_and_015`),
  `src/bestiary/isaac/hound_cfg.py`, `src/bestiary/robots/hound/build.py`,
  `src/bestiary/robots/hound/check.py:255-330`,
  `src/bestiary/robots/hound/CARD.md` (traction budget),
  `assets/hound16pd.xml` (joint axes, read by FK).
- Installed Isaac Lab tree, read-only:
  `isaaclab/envs/mdp/commands/velocity_command.py:160-199`,
  `isaaclab/envs/mdp/rewards.py:272-306` and `:314-338`,
  `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py:140-151` and
  `:283-317`, `.../velocity/mdp/curriculums.py:42-54`,
  `.../config/anymal_c/agents/rsl_rl_ppo_cfg.py:23`.
- `research/decisions/0004`, `research/decisions/0005` (whose `lin_vel_y`
  premise this decision corrects),
  `research/learnings/011`, `research/learnings/015`,
  `docs/theory/reward-composition.md` including its two appended refutations.
- `research/anomalies.jsonl` rows 59–61, appended with this decision.
