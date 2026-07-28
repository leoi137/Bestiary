---
number: 012
title: The heading term is a tax on moving, not a tax on the terrain
date: 2026-07-28
from: research/scripts/heading_ceiling.py, research/measurements/heading_ceiling_s0.json, research/scripts/012_heading_tax_not_terrain.py
robot: hound
supersedes: none
extends: 011-the-crash-count-was-one-percent-of-the-gap.md
guard: heading-freeride-is-terrain-invariant — PROPOSED, not yet written; the exact assertion is in "Make it a guard" below
triggers:
  - reward_change
  - new_terrain
  - comparison
  - long_run
  - proposing more training steps to lift a tracking score
  - proposing flatter ground, a terrain curriculum, or an easier start to fix heading
  - widening sigma_w, or making the tracking reward additive instead of multiplicative
  - reading a per-step mean (Phi_v, Phi_w, track) from two arms whose episode lengths differ
last_confirmed: 2026-07-28
---

# 012 — The heading term is a tax on moving, not a tax on the terrain

**Date:** 2026-07-28 · **From:** `research/scripts/heading_ceiling.py` on
`hound_track_desert_s0` / `ant_sac.zip`, 16 cells × 10 episodes × 2 arms
**Robot:** hound

## What we believed before

[`011`](011-the-crash-count-was-one-percent-of-the-gap.md) found that the
command-tracking policy loses to a machine doing nothing because it earns
almost no *net* tracking reward: the reward is a **product** of a speed score
`Φ_v` and a heading score `Φ_w`, and the ×1.46 the policy gains in speed is
cancelled by the ×0.48 it loses in heading. It then wrote down, in plain
words, the story everybody in the room believed about *why* the heading
collapses:

> A machine that stands still holds its heading almost perfectly and scores
> nearly full credit on Φ_w for free. **A machine that drives on this terrain
> yaws**, and loses roughly half of it.

*On this terrain.* Read it again — the mechanism was attributed to the ground.
The hound drives over a heightfield with a 5.05 m elevation span; wheels climb
cell walls, one side catches before the other, the trunk twists. That is a
completely plausible physical story and it is the one `011` told, so `011`
listed it as its own second falsifier: *"If the same policy on flat ground
holds Φ_w near zero action's 0.513 while driving, then heading loss is a
terrain-contact problem, not a reward-balance problem."*

Everything downstream depended on which way that went, and the two answers
point in opposite directions and cost very differently:

- **Terrain-caused** → the objective is fine, the policy is undertrained and
  the ground is hard. `ep_rew_mean` was 43.24 and still climbing at 1.5M
  steps, so this was live. The fix is more steps, or a terrain curriculum, and
  retuning coefficients would be tuning away real signal.
- **Not terrain-caused** → `Φ_w` is depressed by locomotion itself, no amount
  of easier ground lifts it, and no rebalance of the *cost* coefficients
  touches it either, because the ceiling is inside the positive term.

`heading_ceiling.py` was written to settle it. This learning is the answer:
**the terrain is not the cause.**

## What happened

### Read this before the table

The sweep scales the heightfield's **elevation** by `α ∈ {0, 0.25, 0.5, 1.0}`
in memory — same model, same robot, same reward, same policy, same commands,
same seeds (3000–3009), spawn clearance held at 0.3684 m on every arm so
nothing free-falls. `α = 0` is exactly flat ground with an elevation span of
0.000 m.

**The `α = 0` row is the weakest row in the table and this learning does not
rest on it.** The policy crashes **40/40** at `α = 0`, at mean episode lengths
of 213–683 steps against zero action's 1000, and `Φ` is a per-step mean over
*the steps that actually happened* — so a short episode is averaged over an
earlier, easier window and is not comparable to a full one. The script's own
design note says the same thing from the other side: the policy was trained at
`α = 1`, so every lower `α` is off-distribution and a *drop* there is
ambiguous (distribution shift, not terrain). This design can cleanly show
"terrain was the binding constraint" and cannot cleanly show its negation from
the flat row alone.

**What carries the claim is the crash-free arms** — 8 of the 16 cells came back
0/10 crashed at a full 1000 steps every episode: `α = 0.25 drive_fast`, and
seven of the eight cells at `α ≥ 0.5` (all but the `α = 0.5` turn). Seven of
those eight are straight drives. They are length-matched against a control arm
that also ran 1000/1000, and they say the same thing the flat row says, without
the flat row's problem.

### The control arm does not move

This is the whole finding in one table. Zero action, no policy at all, at
every roughness, under a straight-drive command:

| α | elevation span | standing `Φ_w` | within-cell sd | yaw error |
|---|---|---|---|---|
| 0.00 | **0.000 m** | 0.9688 | 0.0115 | 1.03 °/s |
| 0.25 | 1.2625 m | 0.9690 | 0.0102 | 1.03 °/s |
| 0.50 | 2.5250 m | **0.9716** | 0.0051 | 0.98 °/s |
| 1.00 | 5.0500 m | 0.9699 | 0.0059 | 1.01 °/s |

Band **0.9688 … 0.9716**. Spread **0.0028 = 0.29%** of the mean, over an
elevation range from perfectly flat to 5.05 m of relief. It is **not monotone
in α**, and it is **1.09 × the mean within-cell standard error** — which is to
say the terrain's effect on a standing machine's heading is not distinguishable
from seed noise at n = 10. An independent earlier measurement
(`research/measurements/tracking_baseline_zero_action.json`, a different script
on a different day) puts the same quantity at 0.9705, agreeing to **0.06%**.

### The policy never gets near it, anywhere

| α | 0.30 m/s | 0.55 m/s | 0.80 m/s | standing |
|---|---|---|---|---|
| 0.00 | 0.8489 (10/10 crashed) | 0.7628 (10/10) | 0.6811 (10/10) | 0.9688 |
| 0.25 | 0.7302 (9/10) | 0.5349 (5/10) | **0.4099 (0/10)** | 0.9690 |
| 0.50 | **0.6656 (0/10)** | **0.4390 (0/10)** | **0.5967 (0/10)** | 0.9716 |
| 1.00 | **0.6052 (0/10)** | **0.4945 (0/10)** | **0.5300 (0/10)** | 0.9699 |

Bold cells are crash-free and full-length. Their `Φ_w` band is
**0.4099 … 0.6656**, a ratio of **0.4231 to 0.6850** against the matched
standing control — a deficit of **31.5% to 57.7%**. The policy's best value
*anywhere in the sweep*, including the crash-contaminated flat cells, is
**0.8489**, and it reaches **0.95 or better in 0 of 16 cells**.

So: **the ground costs a standing machine 0.29% of its heading score; driving
costs 31.5–57.7% of it.** The second effect is **109× to 199×** the first.
Easier ground narrows the gap a little and never closes it.

### And flat ground is where this policy dies

| α | elevation | policy crashes | zero-action crashes | policy mean steps |
|---|---|---|---|---|
| 0.00 | 0.000 m | **40/40** | 0/40 | 397.2 |
| 0.25 | 1.2625 m | 24/40 | 0/40 | 704.3 |
| 0.50 | 2.5250 m | 7/40 | 0/40 | 893.0 |
| 1.00 | 5.0500 m | **0/40** | 0/40 | 1000.0 |

The policy survives 40/40 on its own 5.05 m training terrain and 0/40 on
perfectly flat ground. **"Flat is easier" is false for this policy.** This is
recorded, not explained — the obvious candidate is that a gait tuned to
5.05 m of relief is doing something on smooth ground that a rough surface
would have damped, but nothing here measures that, and it stays an open
question rather than a mechanism.

## Why it happened

Because holding a heading and moving are not separate jobs for this machine,
and `σ_w = 0.1 rad/s` is a tolerance the *chassis* has to meet, not one the
*terrain* has to permit.

A body at rest has yaw rate ≈ 0 by definition. Nothing is pushing it, so
nothing is turning it, and it collects `Φ_w ≈ 0.97` for free — on flat ground
and on 5 m dunes alike, because a stationary wheel resting in a hollow is
resting just as still as one on a plane. That is why the control arm is flat
across α: **the terrain only acts on a body through motion**, and this body is
not moving.

A body that drives has sixteen actuators throwing torque into a trunk with
real inertia, four hub wheels whose contact patches make and break
independently, and a controller that is correcting its heading *after* the
error appears rather than before. Every one of those produces yaw, and the
*level* is set by the act of driving rather than by the ground under it.

It is tempting to fall back on "well, the terrain must still add *something*",
and the crash-free cells do not support even that. Holding the command fixed
and varying only α, the policy's heading deficit moves **+6.1 pp** rougher for
`drive_slow` (31.5% → 37.6%), **−5.8 pp** for `drive_mid` (54.8% → 49.0%), and
non-monotonically for `drive_fast` (57.7% → 38.6% → 45.4%). **The sign is not
consistent.** With one seed and n = 10 this is not evidence that terrain does
nothing at all to a driving machine — it is evidence that whatever it does is
small enough to be swamped, which is the same conclusion the control arm
reaches from the other direction.

### There is an older measurement that says the same thing, taken before the policy existed

`research/measurements/tracking_noise.json` was written in the cycle that
*derived* `σ_v` and `σ_w`, long before this policy was trained. It has two
arms, both open-loop constant wheel commands on `HoundPDDesert-v0` — no policy,
no controller, no steering at all:

| arm | achieved `v_x` | yaw rms | `Φ(rms)` |
|---|---|---|---|
| wheel command 0.0 (lying still) | −0.03553 m/s | 0.01823 rad/s = 1.04 °/s | 0.9678 |
| wheel command 0.3 (just rolling) | +0.15437 m/s | 0.12695 rad/s = 7.27 °/s | 0.3829 |

**A machine that merely rolls forward, with nothing steering it, yaws 6.96
times as fast as one lying still.** That number was on disk before this
question was asked. It is the mechanism stated in its purest form: motion
breaks heading, and no policy is needed to make it happen.

It also places the trained policy correctly. At α = 1.00 and 0.30 m/s it scores
`Φ_w = 0.6052`, sitting between open-loop rolling (0.3829) and lying still
(0.9699) — the controller **recovers 38%** of what driving costs and pays the
other 62% as the tax. It is not failing to steer; it is steering, and steering
is not enough.

*(Cross-instrument caveat: those two arms report an **rms** yaw rate, so
`Φ(rms)` is not the `E[Φ]` the sweep reports, and `Φ` is neither convex nor
concave over the range in play, so the two can differ either way. This is the
repo's existing convention — `guards/tracking_frame.py` assertion 5 uses it —
and it is quoted as an order-of-magnitude reference point, never as a cell of
the sweep's table. The 38% figure inherits that looseness.)*

The reason this matters so much more than a normal 40% shortfall is the
**product**. Section 1 of `docs/theory/command-tracking-reward.md` makes the
tracking reward multiplicative on purpose, so that a policy cannot buy a
perfect speed match by spinning: a zero on either factor is a zero on the
term. The cost of that design is symmetric. **A ceiling on either factor is a
ceiling on the whole positive term**, and no coefficient outside the product
can lift it. `w_ctrl`, `w_contact` and the termination penalty are all
*subtractions*; halving every one of them cannot make `Φ_v·Φ_w` bigger than
`Φ_v·Φ_w`.

### Two things the data said that we expected to say differently

Both are corrections to the story as it was framed before the numbers were
reduced, and both are in `research/scripts/012_heading_tax_not_terrain.py`
sections 5 and 7.

**`Φ_w` does not fall monotonically with commanded speed at every roughness.**
It was natural to expect it to. It does not: `slow > mid` holds at **4/4**
roughnesses and `slow > fast` holds at **4/4**, but `mid > fast` holds at only
**2/4**. At α = 0.5 the 0.80 m/s cell scores 0.5967 against the 0.55 m/s cell's
0.4390, and at α = 1.0 it is 0.5300 against 0.4945 — the *fast* command holds
heading better than the middle one. Strict monotonicity holds at exactly the
two roughnesses that are crash-contaminated (α = 0 and 0.25) and at **0 of 4**
crash-free ones. What survives is the weaker, fully supported statement:
**`Φ_w` at 0.55 and 0.80 m/s is below `Φ_w` at 0.30 m/s on every ground
tested.**

**The policy is not trading `Φ_w` for `Φ_v` along the command axis.** A trade
would show up as a negative correlation between the two factors across cells.
Measured across the 12 straight-drive cells, `r(Φ_v, Φ_w) = +0.80`; across the
7 crash-free ones, **+0.56**. Both factors get *worse together* as the command
gets harder — a faster command is simply a harder problem in both channels, not
a lever that moves score from one to the other.

The trade `011` identified is real, but it is between **moving and not
moving**, not between the two factors. That version replicates perfectly here:
at **12 of 12** straight-drive cells, the policy has *higher* `Φ_v` than
standing (×3.0 to ×12.7) and *lower* `Φ_w` than standing (×0.42 to ×0.88).
Every cell, every roughness, no exceptions.

## The math

The tracking term, per control step at 20 Hz (`envs/hound_track.py:302-311`):

    Φ(u) = 1 / (1 + u²)                        the Cauchy kernel
    u_v  = ‖v_heading − v_cmd‖ / σ_v           speed error, unitless
    u_w  = |ω − ω_cmd| / σ_w                   yaw error, unitless
    r_track = healthy · Φ(u_v) · Φ(u_w)

`v_heading` is the planar trunk velocity in the trunk's own yaw frame (m/s),
`ω` is the trunk yaw rate about body-z (rad/s), `σ_v = 0.15 m/s`,
`σ_w = 0.10 rad/s = 5.73 °/s`. `healthy` is 1 while the trunk is upright and 0
otherwise. Both factors are in `[0, 1]`.

**Reading a score back as a physical error.** `Φ` is a tolerance kernel, not a
quantity anyone has intuition for, so invert it:

    Φ = 1/(1+u²)  ⟹  u = √(1/Φ − 1)  ⟹  |ω − ω_cmd| = σ_w·√(1/Φ − 1)

On the committed terrain (α = 1.00, 5.05 m of relief), under `v_x = 0.30 m/s`:

    standing:  Φ_w = 0.9699 → |ω| = 0.10·√(1/0.9699 − 1) = 0.10·0.1763
                            = 0.01763 rad/s = 1.01 °/s
    policy:    Φ_w = 0.6052 → |ω| = 0.10·√(1/0.6052 − 1) = 0.10·0.8077
                            = 0.08077 rad/s = 4.63 °/s

**The machine wobbles 4.58 times as fast when it drives as when it lies there.**
And on perfectly flat ground the standing figure is 1.03 °/s — the terrain
buys the standing machine nothing, because it was already still.

**What the heading factor costs, in reward.** Take the same cell and ask what
the tracking term would be if the policy held heading as well as a standing
machine does, keeping its measured speed score:

    achieved      Φ_v · Φ_w      = 0.5028 × 0.6052 = 0.3043 per step
    counterfactual Φ_v · Φ_w^zero = 0.5028 × 0.9699 = 0.4877 per step
    removed by the heading factor = 0.1834 per step = 37.6%

Across the seven crash-free straight cells the heading factor removes
**31.5% to 57.7%** of the tracking reward the speed score alone would have
earned.

**One honesty note on that arithmetic, computed rather than waved at.** `track`
as reported is `mean_t[Φ_v·Φ_w]`, and `Φ_v`, `Φ_w` as reported are
`mean_t[Φ_v]` and `mean_t[Φ_w]`. These are not the same thing — the difference
is the per-step covariance. Measured on those seven cells the gap runs
**−0.0082 to +0.0270** (e.g. α=1.0 slow: `track = 0.3176` against a
product-of-means of `0.3043`). So the counterfactual above is accurate to about
that, which is small next to a 0.18 effect but is not zero, and the percentages
are factor-level statements rather than an exact per-step decomposition.

**Why no cost coefficient can fix this.** The full per-step reward is

    r = Φ_v·Φ_w − w_ctrl·Σaᵢ² − w_contact·Σ|f| − K·1[terminated]

The first term is bounded above by `Φ_v · max(Φ_w)`. Everything else is
subtracted. Setting `w_ctrl = w_contact = K = 0` outright would lift the return
by exactly the control and contact cost `011` measured (0.0689/step) and would
leave `Φ_v·Φ_w` untouched at 0.3043. **The heading ceiling is inside the term
no coefficient reaches.** Only three things move it: a better controller, a
wider `σ_w`, or not multiplying.

## What to do next time

1. **Stop proposing terrain as the lever for heading.** A flatter start, a
   roughness curriculum, or a terrain regen will not lift `Φ_w`, because the
   control arm is flat to 0.29% across the entire range those levers can move.
   This closes the "easier ground" branch that `011` left open. It says nothing
   about terrain as a lever for *crashes*, where the sweep shows a large and
   opposite effect.
2. **`Φ_w` is the binding factor, and it is a control problem.** 4.63 °/s of
   yaw wobble while driving at 0.30 m/s against a 5.73 °/s tolerance is a
   controller specification, not a reward-tuning one. Either the controller
   gets better at it (more training, better exploration, a yaw-rate feedback
   term the policy can actually use) or the objective stops demanding it.
3. **If the objective changes, change the structure, not the coefficients.**
   The two structural moves are widening `σ_w` and making the term additive.
   Both are reward-shape changes — [`004`](004-lock-the-reward-shape-not-just-the-weights.md)
   applies, `guards/reward_spec.py` will see them, and the multiplicative form
   exists for a documented reason (Section 1 of the theory note) that must be
   argued against, not quietly dropped. This is a decision for the mathematics,
   with its own derivation.
4. **Measure a "the environment caused X" claim against a policy-free control
   at the same environment setting.** That is the single move that made this
   sweep decisive. The zero-action arm costs nothing, is immune to distribution
   shift, and turns an ambiguous off-distribution comparison into a clean one:
   whatever the policy's number does across α, the control arm tells you how
   much of it the *ground* is responsible for.
5. **Never compare per-step means across arms of different episode length
   without saying so.** Half this table is length-biased and it took a
   `steps_mean` column to see it. This is the same trap that produced the ×1.47
   figure `011` had to retract in its own parenthesis. Two failures on the same
   mechanism is enough to make it a rule.

## What this does NOT show

- **One seed, one checkpoint.** `hound_track_desert_s0`, `ant_sac.zip` at 1.5M
  steps. Under the repo's seed rule this is a **probe**, not a finding about
  the reward *design*. It is decisive about what the terrain does — because the
  zero-action arm is policy-free and therefore seed-independent in the relevant
  sense — and it is provisional about everything concerning the policy.
- **It does not prove a ceiling.** The sweep shows *this* policy cannot exceed
  0.8489 anywhere and sits at 0.4099–0.6656 on the load-bearing cells. It does not
  show that no policy could do better; 1.5M steps did not converge. The
  supported claim is "easier ground will not lift it", not "nothing will".
- **The α = 0 and α = 0.25 rows are off-distribution and crash-contaminated.**
  They are shown because they are informative in the direction the design note
  says is safe, and because their crash counts are themselves a finding. Do not
  quote their `Φ` values as clean measurements of anything.
- **α scales elevation, it does not resample the terrain.** Every α is the same
  desert with the relief turned down, so slope statistics scale but the
  *pattern* — where the dunes are, how wide they are — is fixed. A genuinely
  different, gentler terrain was not tested.
- **The flat-ground crash result is unexplained.** Recorded, not diagnosed. See
  [`009`](009-the-cell-size-story-was-never-measured.md) for what happens when
  an unmeasured mechanism gets attached to a real observation.

## Make it a guard

**Yes, and the assertion is: a machine doing nothing holds its heading almost
perfectly, no matter what the ground looks like.** That single fact is what
every conclusion above rests on — it is what makes the control arm a valid
control, it is why `Φ_w` deficits can be attributed to locomotion, and it is
also the premise of the standing-freeride inequality that `σ_v` and `σ_w` were
*derived* from (`guards/tracking_frame.py` assertion 5, which currently reads
the standing yaw drift of **0.01823 rad/s** out of `tracking_noise.json` — a
number measured on exactly one heightfield). The proposed guard,
`heading-freeride-is-terrain-invariant`, would roll zero action on the
committed model at α = 0 and α = 1 in memory (nothing written to `assets/`,
same trick `heading_ceiling.py` uses, with the spawn clearance held) and assert
two things: that `Φ_w` ≥ 0.95 on both, and that the two differ by less than
0.01 — comfortably above the 0.0028 this sweep measured, so seed noise cannot
trip it. It fires on precisely the changes that would invalidate this learning
without anyone noticing: a terrain regeneration that makes standing yaw, a
`σ_w` retune, a spawn-pad change, or a regression in the body-frame yaw-rate
read. It is not a guard on the *policy*'s heading — that number is supposed to
be bad right now — it is a guard on the control arm the policy is judged
against, which is the thing that must never quietly stop being valid.

## How we would know this is wrong

- **A policy reaches `Φ_w` ≥ 0.95 while driving at any α.** This is the direct
  falsifier. Currently 0 of 16 cells clear it, the best full-length cell is
  0.6656 and the best cell of any kind is 0.8489. A longer run, a different
  algorithm, or a different gait that gets a driving machine to 0.95 makes the
  "tax on locomotion" framing wrong — it would mean the tax was a property of
  this controller, not of driving.
- **The zero-action control arm turns out to be terrain-dependent after all.**
  If a larger seed block, or a different terrain seed, shows standing `Φ_w`
  moving materially with α, the control is not a control and every attribution
  above collapses. Today the spread is 0.00281 against a mean within-cell SEM
  of 0.00259, so it is one seed block from being pure noise — this is the weakest
  quantitative leg and it is the one the proposed guard watches.
- **A gentler *pattern*, not just gentler relief, lifts `Φ_w`.** α scales
  amplitude only. If a terrain regenerated with longer-wavelength features at
  the same 5.05 m span lets a driving policy hold `Φ_w` near 0.97, then the
  binding constraint is terrain *geometry* rather than locomotion, and this
  learning is right about amplitude and wrong about the conclusion.
- **The crash-free cells are not representative.** All seven live at α ≥ 0.25
  and six of them at α ≥ 0.5. If a run that is crash-free at α = 0 shows
  `Φ_w` near the standing value there, the flat row's ambiguity was hiding the
  answer rather than being irrelevant to it.

### The falsifier this learning tested and did not fire

This sweep is the deliberate execution of
[`011`](011-the-crash-count-was-one-percent-of-the-gap.md)'s **falsifier 2**:
*"The Φ_w collapse is terrain-specific — if the same policy on flat ground
holds Φ_w near zero action's 0.513 while driving, then heading loss is a
terrain-contact problem."* **It did not fire.** `011` stands, and its mechanism
is now narrowed: the cancellation is real and it is not the ground's doing.

**One correction to how that falsifier was written, because it cannot be
applied as stated.** The `0.513` in `011` is not a straight-drive number — it
is the mean `Φ_w` over the six-cell drive grid, three of whose cells command a
nonzero yaw rate where a standing machine scores 0.055. Recomputed from
`tracking_baseline_zero_action.json`: the six-cell mean is **0.5128**, the
three straight cells alone are **0.9705**, the three yaw-commanded cells are
**0.0552**. So the flat-ground policy values in this sweep (0.6811–0.8489) do
sit *above* 0.513 — and that comparison is meaningless, because it puts a
straight-drive measurement against a mixture average. Against the matched
control at the same α and the same command, the policy is below standing at
**12 of 12** straight cells. **When a falsifier quotes a threshold, it must
quote the arm the threshold came from**; this one nearly fired itself on an
apples-to-oranges reading.

---

Every derived figure above is printed by
`research/scripts/012_heading_tax_not_terrain.py`, whose output is committed at
`research/measurements/012_heading_tax_not_terrain.txt`. Per-cell measurements
are `research/measurements/heading_ceiling_s0.json`, written by
`research/scripts/heading_ceiling.py`.
