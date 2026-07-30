# Reward composition: products of kernels, tolerance widths, and the yaw window

**Written:** 2026-07-29 · **Because:** the record now holds three measured
results about the command-tracking reward — the cost ratio of learning 011, the
heading tax of learning 012, and the one-trot optimum of learning 015 — and the
next reward decision has to be made *from* them. This note derives what the
reward's structure actually implies: what multiplying kernels does and does not
buy, what the tolerance width does to the one-gait exploit, and why the yaw
channel as currently specified is infeasible at measured noise levels.

---

> ## Status: DERIVED, NOT VALIDATED
>
> **No run has trained under any conclusion in this note.** Nothing here is a
> result. Every number is either read from a named primary source, read from
> this repo's committed measurements, or computed by numerical quadrature in
> the session that wrote this note — validated against the two env-computed
> values in `research/learnings/015` (`J(0.491) = 0.7643` against the record's
> 0.764, `J(0.271) = 0.4332` against 0.433, agreement to four decimals).
>
> **An independent refutation has NOT yet run against these conclusions.**
> Until it has, and until the F table below is reproduced by a committed
> script under `research/scripts/` (the number rule requires it before any
> figure here is cited downstream), cite this note as a proposal, never as
> evidence.
>
> Two external findings quoted in §1 (the thin lineage of multiplicative
> rewards, and the dm_control flooring convention with the Mysore et al.
> measurement) were verified from primary source by the parallel breadth
> review of the same date, **not re-verified in this session**. They are
> marked where they appear.

---

## What the production stacks actually do (verified from source, 2026-07-29)

Both reference implementations use **additive** tracking with a **fixed**
tolerance:

- `legged_gym` (`legged_gym/envs/base/legged_robot.py`, master):
  `exp(−‖v_cmd − v‖² / 0.25) · 1.0 + exp(−(ω_cmd − ω)² / 0.25) · 0.5`, with
  `tracking_sigma = 0.25` a constant in
  `legged_robot_config.py` — the divisor is σ², so the width is 0.5 m/s in
  this repo's `K(e; α) = exp(−(e/α)²)` convention.
- Isaac Lab (`source/isaaclab/isaaclab/envs/mdp/rewards.py`, identical at
  `main` and at tag `v3.0.0-beta2.patch1`):
  `torch.exp(-lin_vel_error / std**2)` with `std = math.sqrt(0.25)` — a
  constant config parameter, **command-independent**. Same kernel, same width.

One structural detail matters for everything below: Isaac's linear term sums
`e_x² + e_y²` *inside* the exponent. That is already a product of two kernels
(x and y), because for squared-exponential kernels a product of factors **is**
a single joint kernel:

    K_v · K_ω = exp(−(e_v/α_v)²) · exp(−(e_ω/α_ω)²) = exp(−‖ẽ‖²),
    ẽ = (e_v/α_v, e_ω/α_ω)     (each component unitless)

**Symbols:** `e_v` [m/s] planar velocity error in the heading frame; `e_ω`
[rad/s] yaw-rate error; `α_v` [m/s], `α_ω` [rad/s] tolerance widths; `ẽ` the
error vector with each channel divided by its own tolerance.

So "additive versus multiplicative" is not two philosophies. It is one
question: **where do you cut the error vector into separately-paid channels?**
Isaac cuts between linear and angular; this repo's product
(`src/bestiary/envs/hound_track_rel.py:335-374`, income at line 351) does not
cut at all.

## 1. What the product does to gradients, and the honest case against it

### 1a. The gradient identity

Differentiate the product with respect to one channel's error:

    ∂(K_v·K_ω)/∂e_v = −(2 e_v / α_v²) · K_v · K_ω

Every channel's gradient carries the **joint income** `K_v·K_ω` as a
prefactor. The additive form's gradient, `∂/∂e_v [w_v K_v] = −(2e_v/α_v²)·w_v
K_v`, carries only its own factor. In plain terms: under a product, being lost
in *any* channel silences the learning signal in *every* channel; under a sum,
each channel keeps its own signal alive regardless of the others.

That cuts both ways, and both edges have been measured in this repo:

- **The product's freeride immunity is structural.** There is no term a
  non-tracking policy can collect, and no coefficient rebalance can reopen
  one — learning 012's statement that "no coefficient outside the product can
  lift it" is a theorem about subtractions, and it equally means no
  coefficient can *leak* income to a free rider. The additive form's leak is
  computed in §1b.
- **The product's ceiling coupling is equally structural.** If one channel has
  an achievability ceiling `K_ω ≤ κ`, all income and every other channel's
  gradient is taxed by κ. Measured: κ = 0.41–0.67 while driving removed
  31.5–57.7% of tracking income (learning 012).

### 1b. The additive form's failure set, on the verified configs

Neither production stack has any yaw-error term outside the yaw tracking term
itself (`ang_vel_xy_l2` penalizes roll/pitch rates; the full reward lists were
checked term by term). So under Isaac's weights (1.0 linear, 0.5 yaw,
α = 0.5):

- a policy with perfect speed and an arbitrary spin retains 1.0/1.5 =
  **66.7%** of tracking income — nothing else in the stack charges yaw error;
- a standing machine under a straight-drive command `c` collects
  `exp(−(c/0.5)²) + 0.5`: **79.8%** of maximum at c = 0.3 m/s, **57.9%** at
  c = 0.5, **34.6%** at c = 1.0.

These holes are real, and production fences them outside the reward: linear
commands with norm below 0.2 m/s are zeroed (`legged_robot.py`,
`_resample_commands`), so standing is the *correct* answer to those draws; the
command range spans [−1, 1]² m/s so no single behaviour is near many commands
(§2); and the default `heading_command` mode turns the yaw command into a
feedback signal that saturates against a spinner (§5).

### 1c. The inversion at this repo's measured operating point

At the factor means measured in learning 011 (Cauchy kernels, this repo's
tolerances — policy Φ_v = 0.350, Φ_ω = 0.245; standing 0.240, 0.513):

| form | policy | standing | drive gain |
|---|---|---|---|
| additive, w = (1, 0.5) | 0.472 | 0.496 | −0.024 |
| additive, w = (1, 1) | 0.595 | 0.753 | −0.158 |
| product of the same means | 0.086 | 0.123 | −0.037 |

(Caveat carried openly: these are sums and products of *factor means*;
learning 011's exact accounting is a mean of per-step products over different
cell sets and gave the policy +0.0055/step. The rows are an operating-point
illustration, not a re-derivation of 011.)

The direction is the finding: **at this repo's tolerances and measured scores,
the additive form pays standing more than the product does**, because it hands
the stander its Φ_ω = 0.513 advantage at full weight, while the product gates
that advantage by the stander's tiny Φ_v. The framing "the additive form has a
hole the product closes" is true near the top of the kernels and inverts in
the incompetent regime. Physically: which composition wins depends on where on
the kernel surface the policy actually lives, and this project has measured
itself living far down the slope.

### 1d. The case against the product, carried honestly

Three findings from the parallel breadth review of the same date (primary
sources named there; **not re-verified in this session**), plus one finding
from this repo's own source, all of which cut against the product:

1. **The multiplicative convention has thin roots.** Heess et al. 2017
   (*Emergence of Locomotion Behaviours in Rich Environments*) is additive;
   Hwangbo et al. 2019 (ANYmal, Science Robotics) is a weighted sum, not a
   product; Kim et al. 2022 cites Heess as inspiration for a multiplicative
   form Heess does not use. `command-tracking-reward.md` reads as if
   multiplying is established practice. It is not: the verified production
   lineage is additive, and the product is this project's own design choice,
   to be defended on its own arguments (§1a) rather than by precedent.
2. **Where products are used successfully — dm_control — every factor is
   floored.** dm_control's composed rewards affinely rescale each factor into
   [lo, 1] with lo > 0 (forms like `(4+x)/5`, `(5x+1)/6`), so no single factor
   can zero the product and kill every gradient at once.
3. **Unfloored products have a measured failure in exactly this repo's
   configuration.** Mysore et al. (arXiv:2012.06656) measured Ant + SAC with a
   multiplicative reward at **−138.38 ± 39.44** against the additive form's
   **1299.70 ± 226.23**. This repo's MuJoCo track is SAC on a locomotion
   morphology — the same configuration.
4. **This repo's kernels are NOT floored — checked from source for this
   note.** The Cauchy kernel `Φ(u) = 1/(1+u²)`
   (`src/bestiary/envs/hound_track.py:138-153`) and the squared-exponential
   kernel `K(e; α) = exp(−(e/α)²)`
   (`src/bestiary/envs/hound_track_rel.py:230-239`) both go to zero as the
   error grows; neither has a dm_control-style affine floor, and the health
   gate multiplies in a hard 0/1 on top
   (`hound_track.py:311`, `hound_track_rel.py:351`). How dead the far field is
   depends on the tail (all values computed this session):

   | u = e/α | Cauchy K | Cauchy \|K′\| | Gauss-sq K | Gauss-sq \|K′\| |
   |---|---|---|---|---|
   | 1 | 0.5 | 0.5 | 0.368 | 0.736 |
   | 2 | 0.2 | 0.16 | 0.0183 | 0.0733 |
   | 3 | 0.1 | 0.06 | 1.23e−4 | 7.4e−4 |
   | 6 | 0.027 | 8.8e−3 | 2.3e−16 | 2.8e−15 |

   Beyond u ≈ 3 the squared-exponential factor is numerically dead (and
   underflows float64 outright at u ≥ 27.3), while the Cauchy tail still
   points home — at u = 3 its gradient is 81× larger, at u = 6, 3×10¹²×.
   The current env's mitigation for the dead far field is not a floor but the
   potential-based shaping term, which carries the old Cauchy product's
   gradient as a policy-invariant addition
   (`hound_track_rel.py:321-331` and the module docstring). That is a
   *different* repair for the same disease dm_control's floors repair: it is
   theoretically cleaner (it provably cannot change the optimal policy), but
   no completed run has validated it, whereas the one completed run under the
   unfloored **Cauchy** product did learn to drive (learning 011: Φ_v ×2.01
   over the control — one seed, a probe). The exposed configuration is
   specifically **squared-exponential product + SAC + no floor + no shaping**;
   that combination should never be launched, and the shaping term is what
   currently stands between this env and it.

**What this section means physically:** multiplying kernels is buying task
conjunction (speed AND heading, no partial credit) at the price of a gradient
that vanishes wherever competence is low in any channel. The price is real,
measured elsewhere, and this repo pays it with a shaping term instead of a
floor — a choice that is defensible but unvalidated, and that must be stated
as a choice, not as inherited practice.

## 2. The tolerance-width law, both ways

The one-gait exploit of learning 015 is governed by a single functional: the
income a **fixed** achieved speed collects in expectation over the command
distribution,

    J(v) = E_{c~U[a,b]} [ K(v − c; α(c)) ]        F = max_v J(v)

**Symbols:** `v` [m/s] a fixed achieved forward speed; `c` [m/s] the commanded
speed, uniform on [a, b]; `K` the tolerance kernel; `α(c)` its width; `F`
(unitless, ∈(0,1]) the **best-single-speed income fraction** — the share of
perfect-tracking income available to a policy that never reads the command.
Perfect tracking scores 1.0 by construction (v = c always).

### 2a. Fixed width: the span law

For constant α, the integral is exact:

    F = (√π · α / S) · erf( S / 2α ),   at v* = (a+b)/2,   S = b − a  [m/s]

F depends **only on S/α** — the command span in units of tolerance — and falls
toward zero like √π·α/S as the range widens. Physically: under a fixed-width
kernel, every extra metre-per-second of command range dilutes the fake's
income linearly, because the fake's kernel lobe has constant width and the
range grows past it.

### 2b. Command-scaled width: the ratio law and the floor

For α = β·|c| (this repo: β = 0.5, floor 0.15 m/s —
`hound_track_rel.py:242-249`), substitute c → v·s and the kernel becomes
`exp(−((1−s)/(βs))²)`, a function of the ratio c/v only. Therefore **F depends
only on b/a**, not on b − a (verified numerically: [0.30, 0.80] and
[0.60, 1.60], both ratio 2.67, both F = 0.7643). Two consequences:

- As c grows far above a fixed v, the relative error tends to 1/β, so the
  kernel tends to `exp(−1/β²)` = e⁻⁴ = 0.018 at β = 0.5 — a nonzero income for
  being arbitrarily wrong. This is the same scale-free leak that the
  `BETA_W = 0.75` incident found in the yaw channel
  (`hound_track_rel.py:157-186`), reappearing in the speed channel.
- The fake's kernel lobe has width ∝ β·v, so its income grows with v: the best
  fake rides near the top of the range (v*/b ≈ 0.575 across every ratio
  tested) and **F has a floor** — computed at ratios 16/32/64/128: F =
  0.5185/0.5018/0.4938/0.4899. Under β = 0.5, no achievable widening pushes
  the best single gait below roughly half the speed income.

### 2c. The F table (all values computed this session; range [0.30, 0.80] m/s unless noted)

| kernel | F | best fake v* [m/s] |
|---|---|---|
| this repo, α = max(0.15, 0.5·c) | **0.764** | 0.491 (reproduces learning 015) |
| fixed α = 0.50 (production width) | **0.923** | 0.550 |
| fixed α = 0.25 | 0.747 | 0.550 |
| fixed α = 0.15 (the old σ_v) | 0.522 | 0.550 |
| fixed α = 0.50 over production's own range [−1, 1] | **0.441** | **0.000 — the production recipe's best fake is standing still** |
| this repo's kernel over [0.30, 1.00] | 0.694 | 0.580 |
| this repo's kernel over [0.30, 1.20] | 0.648 | 0.691 |
| this repo's kernel over [0.30, 1.60] | 0.598 | 0.921 |
| β annealed to 0.3, [0.30, 0.80] | 0.606 | 0.530 |
| β = 0.3, [0.30, 1.00] | 0.509 | 0.669 |

Read the two bold rows together: the production **kernel** is more fakeable
than this repo's over any common range (0.923 vs 0.764), and the production
**recipe** is less fakeable than this repo's (0.441 vs 0.764) — because its
command range is four times wider, two-sided, and two-dimensional. F is not a
property of the kernel; it is a property of the kernel **relative to the
command geometry**. And every recipe has a best fake: production's is the
stander, which its command-zeroing, explicit stand draws, and air-time gate
then fence off.

**Why scaling is nonetheless forced here.** The two-sided argument in the
`hound_track_rel.py` module docstring is a theorem: the achieved error at
c = 0.8 (measured 0.400–0.491 m/s) *exceeds* a stander's error at c = 0.3
(0.335 m/s), so no function monotone in absolute error can pay the first and
starve the second — at any scale, in any family. Relative error reorders them
(stander at ρ ≈ 1.04 everywhere; policy at ρ ≈ 0.46–0.76). Fixed width is not
available to this robot while its achieved error grows with the command; the
faking floor of §2b is the price of that theorem.

**What this section means physically:** widening the command range always
makes faking harder, but under the scaled kernel the gain is logarithmic in
the range *ratio* with a floor near one half, so the sampler cannot close the
exploit. Something else has to — §3 says what.

## 3. The crossover: when honest tracking starts to out-earn the fake

An honest tracker at constant relative error ρ = e/|c| (unitless) earns, under
the scaled kernel with floor inactive,

    K = exp(−(ρ/β)²)      — constant in c, by construction.

It beats the best single-speed fake exactly when `exp(−(ρ/β)²) > F`, i.e. when

    ρ < ρ* = β · √ln(1/F)

With β = 0.5 and F = 0.7643: **ρ\* = 0.259**. With β = 0.3 (F = 0.606 on the
same range): ρ\* = 0.212.

Measured relative errors of `hound_track_rel_s1` (learning 015's table,
recomputed here): ρ = 0.46 at c = 0.5, 0.61 at c = 0.8, 0.76 at c = −0.3.

So at measured skill, honest tracking earns `exp(−(0.5/0.5)²)` ≈ **0.368** per
step of speed income while the fake at the observed trot earns J(0.271) =
**0.433**. **The one-trot policy of learning 015 was not collecting 76% of the
income for free — it was outright income-optimal.** 015's "rational optimum"
phrasing is exactly this inequality, now with the threshold attached: no
command-range decision changes the ordering while ρ > ρ*; only skill rising
(ρ falling below 0.26 at β = 0.5) or the tolerance narrowing (β annealed
downward as measured ρ falls, never below the noise-side bound) flips it.

**What this means physically:** the machine stops preferring one trot the day
its tracking error gets inside about a quarter of the commanded speed — and
not one day sooner, whatever the sampler does. The tolerance can follow skill
down (both F and ρ* shrink with β), but it cannot lead it.

## 4. The yaw window is empty as currently specified

The two-sided rule that sized every tolerance in this repo
(`command-tracking-reward.md` §2) is: noise side, α ≥ 3 × unremovable noise;
freeride side, an uncontrolled machine must score low. The published
derivation fed the **standing** yaw noise floor (0.0182 rad/s,
`research/measurements/tracking_noise.json`) into the noise side of a factor
that is scored **while driving**.

Learning 012's own inversion gives the driving-conditioned residual: the best
crash-free driving cell scored Φ_ω = 0.6052, and

    e = σ_ω · √(1/Φ − 1) = 0.10 · √(1/0.6052 − 1) = 0.0808 rad/s = 4.63 °/s

Redo both sides on the right conditional:

    noise side:    α_ω ≥ 3 × 0.0808              = 0.242 rad/s
    freeride side: exp(−(0.127/α_ω)²) ≤ 0.45  ⟹  α_ω ≤ 0.127/√ln(1/0.45) = 0.142 rad/s

**The window [0.242, 0.142] is empty.** The discrimination being asked of the
rate-domain kernel — well-steered driving (residual 0.081 rad/s) versus
unsteered driving (0.127 rad/s) — has a signal-to-noise ratio of 1.57, and no
kernel on |ω − ω_c| can separate the two while tolerating its own noise.

The measured ceiling confirms the arithmetic. For zero-mean Gaussian yaw error
of standard deviation s [rad/s] under the kernel of width α,

    E[K] = E[exp(−ω²/α²)] = 1 / √(1 + 2s²/α²)

At s = 0.0808, α = 0.10: E[K] = **0.659** — against the measured 0.6052 (the
gap is the bias component the Gaussian model omits; learning 015 measured a
fixed turning handedness). At the standing floor s = 0.0182: E[K] = 0.968,
which is the control arm's 0.9699. **The Φ_ω "ceiling" of learning 012 is the
noise ceiling of the kernel, computed; undertraining is not required to
explain it.** The window reopens only if a controller achieves a driving
residual below 0.142/3 = **0.047 rad/s** — possible, unmeasured, and not worth
betting the objective on. This defect in the published derivation is recorded
as `research/anomalies.jsonl` row 46.

**What this means physically:** the yaw-rate factor as specified demands that
a driving machine wobble less than the act of driving makes it wobble, and no
retune of α_ω can want both things at once. The quantity being scored has to
change.

## 5. The repair: close the loop through the command

The production stacks already contain the repair, verified from source. With
`heading_command = True` (the shipped default in both), legged_gym computes

    self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(heading_target − heading), −1., 1.)

(`legged_robot.py`, `_post_physics_step_callback`; Isaac Lab:
`heading_control_stiffness = 0.5` in the velocity command config). The scored
yaw command is a **feedback signal**: proportional to the accumulated heading
error, saturated. The rate kernel they score is closed-loop.

Adapted to this repo's command set (feedforward for turn-in-place commands,
feedback for heading hold):

    ψ_c(t) = ψ(t_resample) + ∫ ω_cmd dt                 [rad]  (commanded heading)
    ω_c(t) = clip( ω_cmd + k_ψ · wrap(ψ_c(t) − ψ(t)), ±ω_sat )   [rad/s]
    K_ω = exp( −((ω − ω_c)/α_ω)² ),   α_ω = 0.25 rad/s

**Symbols:** `ψ` [rad] trunk yaw (heading); `ψ_c` [rad] commanded heading,
integrating the commanded rate from the heading at resample; `k_ψ` [1/s]
feedback stiffness, 0.5 as in production; `ω_sat` [rad/s] the clip;
`wrap(·)` maps to (−π, π]; `α_ω = 0.25` = 3 × the measured driving residual
0.0808, satisfying the noise side by construction.

Why this dissolves the empty window rather than papering over it —
**integration separates bias from noise**:

- **Honest driver** (zero-mean wobble): integrating discrete white rate noise
  of standard deviation s [rad/s] at step dt [s] for T seconds gives a heading
  error of standard deviation σ_ψ = s·√(T·dt). At s = 0.081, dt = 0.05
  (20 Hz), T = 10 s: σ_ψ = 0.057 rad. Its rate error stays ≈ its wobble, and
  at α_ω = 0.25 it scores E[K] = 1/√(1 + 2·0.0808²/0.25²) = **0.91** — up from
  the 0.66 ceiling of §4.
- **Unsteered driver** (bias 0.127 rad/s): heading error grows linearly,
  1.27 rad in 10 s; ω_c ramps against it to saturation; the rate error grows
  far past any α; K → 0. The freeride cap is cleared with room instead of
  missed.
- **Stander under a turn command**: e_ψ = ω_cmd·t, so K falls from 0.039 at
  t = 1 s to 2.4e−6 at t = 2 s — against the constant-forever
  `exp(−1/β²)` = 0.018 of the scale-free relative-rate kernel.
- **Learning 015's fixed handedness** is a yaw *bias* by measurement — exactly
  the component integration amplifies (a 0.081 rad/s bias held 2 s scores
  0.657 and keeps falling) and the open-loop rate kernel tolerated.

Cost of the change: zero observation width (the ω command slot carries
`ω_c(t)`, one number, exactly the production trick); one new spec parameter
k_ψ hashed with the reward; the α_ω(c) scaling and its `exp(−1/β²)` leak are
deleted rather than retuned. It is a reward-**shape** change, so
`learnings/004` applies: its own arm, its own commit, the spec hash carrying
(k_ψ, α_ω, ω_sat).

**ASSUMED, with the cheapest settling experiment:** the white-noise model of
driving wobble. If the wobble has strong low-frequency content, σ_ψ grows
faster than √T and the 0.91 figure is optimistic. Settle by integrating the
yaw traces already in the run logs — one script, no GPU, no new run. Second
assumption: the 0.0808 residual is near the chassis floor rather than an
artifact of one undertrained seed (learning 012 marks it a probe); the same
script that measures the wobble spectrum on other checkpoints bounds this.

**What this means physically:** stop scoring how steadily the machine turns
and start scoring whether it ends up pointing where it was told, using the
turn-rate channel only as the messenger. Noise in the rate averages out of the
heading; a steering failure does not. The kernel then sees the signal instead
of the noise.

## Classification of every load-bearing claim

**VERIFIED (primary source, fetched 2026-07-29):** both production kernels,
widths, weights, and their command-independence; the absence of any yaw-error
penalty outside the tracking terms; the sub-0.2 m/s command zeroing; the
heading-feedback command law in both stacks (legged_gym master;
Isaac Lab main = tag v3.0.0-beta2.patch1); this repo's kernel forms, floors
(absent), gate, and shaping implementation at the file:line references given.

**DERIVED (computed this session; integrator validated against learning 015's
env-computed values to four decimals):** the product-of-kernels identity and
gradient prefactor; the additive failure-set shares; the §1c inversion table
(with its stated product-of-means caveat); the span law, ratio law, faking
floor, and the full F table; ρ* = β√ln(1/F) and the 0.368-vs-0.433 ordering;
the empty window [0.242, 0.142]; the noise-ceiling formula and its 0.659 and
0.968 values; every number in §5.

**ASSUMED (each with its cheapest experiment):** white-noise yaw wobble
(integrate existing logs); 0.0808 rad/s as a chassis-level residual rather
than one seed's (same script, other checkpoints); the moving-phase *speed*
noise floor, still unmeasured (`command-tracking-reward.md` §7.4's gap — the
same conditioning defect §4 found in yaw, one channel over); v_max ≈ 1.1 m/s
extrapolation gating any range widening (one scripted full-throttle rollout);
the three literature-lineage findings and the Mysore measurement in §1d
(breadth review's verification, not this session's).

## See also

- `command-tracking-reward.md` — the derivation this note extends and, in §2's
  yaw window, corrects; its status banner is the model for this note's.
- `research/learnings/011`, `012`, `015` — the three measured results every
  section here is answerable to.
- `src/bestiary/envs/hound_track_rel.py` — the implemented kernel, tolerances,
  and shaping this note analyses.
- `src/bestiary/envs/hound_track.py:138-153` — the Cauchy kernel.
- `research/anomalies.jsonl` row 46 — the mis-conditioned noise floor.

---

# Refutation — 2026-07-29, an independent Opus 5 pass

This note was committed (`30f3730`) stating that no refuter had run. One has now
run, and **it killed the central claim.** Per this record's convention a
falsified argument is superseded rather than deleted: everything above stands as
written, and what follows is what survives it.

## §4's empty α_w window is WITHDRAWN

The arithmetic reproduces exactly. **The error is which side of the inequality
the number was placed on.**

`command-tracking-reward.md` §2 had already considered and explicitly rejected
the substitution §4 makes:

> The trap arose from feeding the moving figure into the *noise-side* rule; but
> the moving figure is the sum of unremovable noise and exactly the error the
> policy exists to remove, so it belongs on the freeride side as a thing to
> *exclude*, never on the noise side as a thing to *tolerate*.

0.0808 rad/s is the **total driving yaw residual of one undertrained,
single-seed, non-converged probe** — `learnings/012` says so itself. It bounds
the unremovable noise floor from **above**, not below. So §4 declared a
published derivation defective for a mistake that derivation had anticipated in
writing.

**Corrected statement.** The window is `[3n, 0.0905]`, where `n` is the
driving-conditioned unremovable noise floor and **has never been measured**.
The noise side is **unpinned, not violated**. Measuring `n` is the whole
question.

Three compounding defects:

- **The 3× multiplier is this repo's own convention, not a law.** The window is
  empty *iff* that multiplier exceeds **1.760**. At a 20%-cost operating point
  it is non-empty. "Infeasible as specified" was really "these two arbitrary
  operating points are jointly infeasible, at a residual that is the wrong
  quantity."
- **The two sides were computed on different statistical arms** — the noise side
  inverts a *mean kernel*, the freeride side applies a *point kernel to an rms*.
  Both on the expectation arm, freeride becomes **α ≤ 0.0905**, not 0.142. This
  is precisely what `learnings/012` warns about: *when a falsifier quotes a
  threshold, it must quote the arm the threshold came from.*
- **`policy_yaw_err_rad_s` is not a measurement.** `heading_ceiling.py:235`
  computes it by the same point inversion. **No committed artifact holds a raw
  yaw trace**, so §5's "integrate the traces already in the run logs" experiment
  does not exist — it needs a re-roll with a trace hook.

**One attack that failed, and it favours the note:** Jensen. Φ_w is a mean of
per-step kernels and 1/(1+u²) is convex in u², so the point inversion is a
*lower* bound — the Gaussian-model rms reproducing E[Cauchy] = 0.6052 is
**0.1174 rad/s, 1.45× larger**. The window dies on the conditioning error, not
on the statistics. §4's figure should read "**at least** 0.0808".

## §4's 0.659 "confirmation" is wrong twice

`Φ_w = 0.6052` was measured under the **Cauchy** kernel
(`hound_track.py:138`); §4 computed a **Gaussian** expectation. On the correct
kernel, `E[Cauchy] at s = 0.0808, σ = 0.10` = **0.7203** — *further* from
0.6052, not closer. And the exercise is circular regardless, since `s` was
derived *from* 0.6052 by point inversion, so the round trip measures the Jensen
gap and nothing else.

## §3's ρ = 0.5 was never measured

A bare number-rule violation: `exp(−(0.5/0.5)²) = 0.368` appears in a headline
conclusion and no computation produced it. The measured values are 0.46 / 0.61 /
0.76.

At the actual measured ρ = 0.458, the honest tracker earns **0.4321** against
the fake's **0.4332** — **a tie to three decimals**, not 0.368 vs 0.433.

And the tie is an **identity**, not a coincidence: ρ = |0.271 − 0.50| / 0.50 is
the relative error *of the fixed 0.271 trot itself*, so §3 compares the fake
against **itself in a different parameterisation** and concludes the fake wins.
**A command-following policy's relative control error has never been measured.**
`learnings/015`'s "rational optimum" should stay unquantified until one exists.

## §1d's Mysore transfer does not apply

Their operator is **not a product**. Equation (8) is a **geometric mean with an
ε offset and a min-clamp**: `r = (∏ min(1, rₖ+ε))^(1/K)`. The K-th root and the
ε are both mitigations of the exact pathology §1d invokes — and **the ε is a
floor.** By this note's own §1d.4 taxonomy, Mysore is the *floored* case, which
defeats the §1d.3 inference. Seed count is never stated.

**Cite it as a caution about composing rewards multiplicatively; not as a
measurement of this repo's configuration.**

## §5's loop separates responsive from unresponsive, not bias from noise

Simulated at `k_ψ = 0.5`, `ω_sat = 1.0`, `α_ω = 0.25`, dt = 0.05, T = 20 s:

| driver | mean K | outcome |
|---|---|---|
| responsive, pure rate bias 0.0808 | 0.9008 | heading converges to a **bounded** −9.3° = −b/k_ψ; **rate error settles back at exactly b**; never saturates |
| responsive, zero-mean noise 0.0808 | 0.9238 | ‖ψ_e‖max 0.046 rad |
| **unresponsive**, bias 0.127 | 0.0834 | −2.54 rad, ω_c saturates, K → 0 ✓ |

A rate bias in a driver that *does* follow ω_c becomes a bounded heading offset
and **the scored rate error is unchanged**. So `learnings/015`'s 42.32-point
handedness — a bias by measurement — is **still not priced.**

The 0.91 is bought by widening α from 0.10 to 0.25, not by the loop: open-loop
at α = 0.25 already gives 0.9095. The loop's contribution is that the widening
now clears the freeride side.

Note also an internal inconsistency: §4 explains the 0.6052-vs-0.659 gap by
asserting the residual is *partly bias*, while §5 assumes it is *entirely
zero-mean noise*. The same quantity, modelled two contradictory ways in adjacent
sections.

**The saturation attack failed** — the law does not saturate on a biased honest
driver. That worry was unfounded.

## §2's F = 0.441 is a 1-D integral labelled as a 2-D command set

Production commands `lin_vel_x ∈ (−1,1)` **and** `lin_vel_y ∈ (−1,1)`, and the
kernel sums `e_x² + e_y²` inside the exponent. The true separable value is
**0.441² = 0.1945**.

And 0.441 vs our 0.764 is not like-for-like — a two-sided 2-D range against a
one-sided forward-only conditional. `learnings/015` already computed this repo's
best single speed over its own six-cell grid at **F = 0.411**, i.e. *lower* than
production's 1-D 0.441. The conclusion's direction survives, and only against
the true 2-D 0.1945.

Two smaller corrections: **"best fake is standing still" is an artifact of range
symmetry**, not of including zero — for fixed α the span law gives
v\* = (a+b)/2 exactly. And §2b's ratio law holds **only where the α floor is
inactive** (F = 0.9056 on [0.10, 0.267], where it is not).

## What survived

- **The F table's arithmetic** — all thirteen values, the analytic span law, the
  ratio-law twin and the four-point floor sequence, reproduced to four decimals
  by an independently written integrator, and it reproduces `learnings/015`'s
  env-computed 0.764 / 0.433. **Still no committed script**, so under the number
  rule these figures may not be cited downstream until one exists.
- **The gradient identity and the product-of-kernels identity** (§0, §1a).
- **The floor claim — measured, and stronger than this note claimed.** On 12
  random-action episodes (1677 healthy steps): max `u_w` **40.23 while
  healthy**, `k_w` **exactly 0.0 in float64 on 1.61%** of healthy steps, and
  `k_v·k_w < 1e-12` on **60.8%**. The Cauchy PBRS potential is **never** zero
  (min 1.88e-5). The underflow is reached in under two thousand steps. One
  caveat this note should carry: the Cauchy far-field gradient at u = 40 is
  ~3e-5 — nonzero, but "points home" is a statement about sign, not magnitude.

## Newly found, not previously recorded

**The incumbent `α_w = 0.10` fails its own freeride cap on the expectation
arm.** `hound_track_rel.py:155` comments that the unsteered freerider "scores
that freeride at 0.20, comfortably tighter than the 0.45 cap" — a *point*
kernel applied to an rms. On the expectation arm the same freerider scores
**0.4866** and **misses the cap.** The same arm defect as §4's, already latent
in shipped code.

**A stale docstring:** `hound_track_rel.py`'s module docstring states
`alpha_w(c) = max(0.10, 0.75·|w_cmd|)` while `BETA_W = 0.5` at line 186 — the
very constant whose 0.75 incident this note cites at `:157-186`.

---

# Second refutation — 2026-07-30, a second independent Opus 5 pass

The first refutation killed §4's empty window. A second pass, run against the
Isaac Hound stack rather than against the MuJoCo track, kills something this note
and its first refutation **share**: both of them price the freeride against a
**stander**, and on the production command geometry the stander is not the
binding fake. Everything above stands as written; what follows is what survives
it. Arithmetic in
`research/scripts/0006_reward_economics_refutation.py`; consequences in
`research/decisions/0006`.

## §1b's failure set and §2c's "best fake is standing still" both understate it

Production ships `heading_command = True`, and this note's own §5 quotes the law:
`ω_c = clip(k_ψ · wrap(ψ_target − ψ), ±1)`. §5 reads it as *the repair*. It is
also **a second freeride**, and a larger one.

A machine that yaws once to `ψ_target` and then **holds still** has heading error
zero, therefore its own scored yaw command is zero, therefore — being still — its
yaw error is zero and it collects the yaw kernel at **exactly 1.0, forever, for
free.** Call it **point-and-park**. A plain stander cannot do this: it keeps
whatever heading error it drew, and scores `E[exp(−(clip(0.5·U(−π,π), ±1)/0.5)²)]`
= **0.287431**.

Priced on the production command set (`lin_vel_x, lin_vel_y ∈ (−1,1)`, weights
1.0/0.5, α = 0.5), as a fraction of nominal maximum:

| behaviour | fraction of nominal max |
|---|---|
| stander (§1b's and §2c's fake) | **0.2255** |
| **point-and-park** | **0.4630** |

**2.05× the fake this note reasons about**, and it reproduces `decisions/0005`
B1's independently computed 0.2255 exactly, which is how the machinery is checked.
Net of a plausible penalty basket the parker earns **103.40%** of a competent
driver's net — it out-earns driving outright.

So §1b's holes (66.7%, 79.8%, 57.9%, 34.6%) are the *smaller* set, and §2c's
bolded conclusion — *"the production recipe's best fake is standing still"* —
is wrong in a way that matters: the production recipe's best fake **turns to face
the commanded heading and then stops**, which is the one behaviour the
command-zeroing, the stand draws and the air-time gate named in §1b do not
fence. The gap is `0.5 · (1 − 0.287431) · dt = 0.006983/step`, and it is
independent of the linear command range because the linear term cancels.

**This is not a criticism of the product form.** Under a product, `K_v · K_ω`
with `K_v` at the stander's value gates the parker's free `K_ω` exactly as it
gates everything else — the immunity §1a claims is real here too. It is a
criticism of every *additive* freeride figure in §1b, §2c and §3, and of the
first refutation for reproducing the same stander framing.

## §3's ρ is worse than circular — ρ = 1 *is* the stander

The first refutation established that `ρ = |0.271 − 0.50| / 0.50 = 0.458` is the
relative error *of the fixed trot*, so §3 compares the fake against itself. That
is right, and there is an identity underneath it that makes the defect
structural rather than incidental.

For a tracker at constant relative error ρ over `c ~ U(−L, L)`, the mean squared-
exponential kernel is

    E[K] = (√π/2) · (α / (ρL)) · erf(ρL/α)

At ρ = 1, L = 1, α = 0.5 this is **0.441041** — which is *exactly*
`E[exp(−(c/α)²)]` over the same uniform, i.e. **the stander's own speed-channel
factor.** ρ = 1 is not "a very bad driver": it is the stander, because a machine
with 100% relative error has zero achieved speed. So **any policy parameterised
by ρ is being compared to the fake along the same axis as the fake**, and a
break-even in ρ is a statement about the yaw channel wearing a speed-channel
disguise. `ρ = 0.46` has now been resurrected once as a driver's competence and
must not be again; a command-following policy's relative control error is still
unmeasured.

## §2's width law is a real law about F and a misleading one about a run

§2 closes: *"widening the command range always makes faking harder."* True of
`F`. **False of the experiment**, because `‖command_xy‖` is also read by the
terrain curriculum: `terrain_levels_vel`'s demote bar is
`distance < ‖c_xy‖ · max_episode_length_s · 0.5`, while the promote bar
(`size[0]/2`) is command-independent. `E‖c_xy‖` over the production square is
**0.765196** (analytically `(√2 + asinh 1)/3`); pin `c_y = 0` and it is
**0.500000** — the demote bar drops **34.66%** with the promote bar unchanged.

So a command-range edit made to move `F` also moves which ground the machine ends
up on, and the two effects are not separable after the fact. Any use of §2 as a
lever has to declare both.

## The first refutation's expectation-arm rule is now load-bearing twice

`E[K] = 1/√(1 + 2s²/α²)` for zero-mean Gaussian error is the arm the first
refutation insisted on, and the second pass needed it again — this time for a
driver's residuals rather than a freerider's cap. The rule is holding up:
**quote the arm the threshold came from, every time.** Two independent passes
have now each found a place where a point kernel was applied to an rms.

## What is now unbarred, and what is still barred

§2c's F table was barred pending a committed script. Two of its rows are now
computed by one:

- the fixed `α = 0.50` over production's own `[−1, 1]`: **0.441041** (the note
  says 0.441);
- its true 2-D value, which the first refutation corrected to 0.1945:
  **0.194517**.

Both are printed by `research/scripts/0006_reward_economics_refutation.py`
section 4 and may now be cited. **The other eleven rows — everything over this
repo's own command-scaled kernel on `[0.30, 0.80]` and its variants, including
`F = 0.764`, `J(0.271) = 0.433`, the ratio-law twin and the four-point floor
sequence — remain barred**, because that script computes the fixed-width kernel
only. `learnings/015`'s 0.764 and 0.433 are separately safe: they are
env-computed and printed by
`research/scripts/track_rel_command_independence.py`.

## What survived, again

- **The gradient identity and the product-of-kernels identity** (§0, §1a). Both
  passes attacked them and neither dented them.
- **The span law and the ratio law** as algebra.
- **The flooring measurement** from the first refutation, unchanged.
- **§1a's claim that the product's freeride immunity is structural.** The second
  pass is the strongest evidence yet *for* it: the additive form just turned out
  to have a freeride nobody had priced after two passes, and the product would
  have gated it automatically.
