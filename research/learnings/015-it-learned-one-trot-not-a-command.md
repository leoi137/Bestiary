---
triggers: [reward_change, comparison, metric_added, long_run]
guard: command-independence — src/bestiary/guards/command_independence.py
last_confirmed: 2026-07-28
---

# 015 — It learned one trot, not a command

**Date:** 2026-07-28 · **From:** `hound_track_rel_s1` (seed 1, 2,000,000 steps,
`HoundPDTrackRelDesert-v0`), measured at `research/measurements/track_rel_s1_best.json`
**Robot:** hound

Every number below is printed by
`research/scripts/track_rel_command_independence.py`, which reads only the
committed measurement JSONs and the env's own reward kernel.

## What we believed before

That the way to tell a command-*following* policy from a policy that has simply
learned to move is a **gain**: regress the forward speed the machine achieves
on the forward speed it was told to achieve, and demand a slope that is not
zero. `record/track_eval.py` calls `command_gain` "the cheapest lie-detector in
the whole design," this cycle pre-registered it as the falsifier before the run
started, and the bar was set at 0.05.

That was a reasonable instrument to trust. It is cheap, it is computed from the
same rollouts as everything else, and the failure it was written against —
a machine that has memorised one motion and ignores the command channel — is
a real and common one.

It came back at **0.382**, seven times the bar, and it was wrong.

The shape of the mistake, which is the transferable part: **a statistic fitted
across a sign change measures the sign, not the magnitude.** The eval grid
contains one backward-commanded cell, the machine creeps backward there
(−0.073 m/s) and trots forward everywhere else, and a straight line through
those points has a healthy slope no matter how flat the forward half is. The
detector answered *"does the command move the machine at all"* while the
question we thought we were asking was *"does a bigger number make it go
faster."*

## What happened

20 deterministic episodes per grid cell, seeds 1000–1019, the same seeds on
both arms, against the do-nothing control in
`research/measurements/track_rel_zero_action.json`.

**The achieved forward speed barely moves with the command.**

| commanded v\_x (m/s) | achieved v\_x (m/s) | achieved / commanded |
|---|---|---|
| +0.50 | +0.271 | 0.541 |
| +0.80 | +0.309 | 0.386 |
| +0.50, with yaw +0.4 | +0.243 | 0.486 |
| +0.50, with yaw −0.4 | +0.276 | 0.552 |
| −0.30 | −0.073 | 0.244 |

Across the four forward-commanded cells the achieved speed spans **0.243 to
0.309 m/s — a range of 0.066 m/s**. Asking for 0.3 m/s more bought 0.038 m/s
more. This is one ≈0.27 m/s trot, run under every command.

**Yaw is not followed at all.** On the three yaw-commanded cells the heading
score Φ\_w reads **0.141**, **0.178** and **0.040** out of a possible 1.0 —
absolutely small, whatever it is being compared to. (The comparison must be
made carefully: a standing machine scores only 0.019 on those cells, because
standing still is *also* the wrong answer to "turn," so the policy is nominally
7–9× the control there while still earning almost nothing. The absolute number
is the one that means something.) On the two straight cells the policy scores
Φ\_w **0.330** and **0.212** against the control's **0.969** — while driving
straight it holds heading *worse than standing still does*.

**The gait has a fixed handedness.** The two mirror cells, identical except for
the sign of the yaw command, return **−27.11** and **+15.21** — a **42.32
point** gap, with Φ\_v 0.151 against 0.388. A policy that steered would be
roughly symmetric under a sign flip; one with a single built-in turning bias is
not.

**The aggregate scored all of this as success.** `drive_grid_mean` is **19.73**
against the control's **3.91**, a ratio of **5.04**, clearing the ≥5× success
bar in `docs/theory/command-tracking-reward.md` §3 by 0.89%. But:

| grid cell | policy | zero action | gap |
|---|---|---|---|
| (0.5, 0, 0) | +93.63 | −0.18 | **+93.82** |
| (0.8, 0, 0) | +39.73 | +2.65 | +37.07 |
| (−0.3, 0, 0) | +25.86 | +31.36 | **−5.49** |
| (0.5, 0, +0.4) | −27.11 | −9.31 | **−17.80** |
| (0.5, 0, −0.4) | +15.21 | −9.31 | +24.52 |
| (0, 0, +0.45) | −28.92 | +8.26 | **−37.18** |

The grid mean gap is +15.82 points, and the single forward cell contributes
+15.64 of it — **98.8% of the win is one cell**, the one whose command happens
to be closest to the gait the machine has. **The policy loses to doing nothing
in 3 of the 6 cells.** Weighted by the command mixture the policy was actually
trained on (`config.json`: `drive` 0.8 / `turn` 0.1 / `stop` 0.1) rather than
by a flat grid, the margin over the control is **+11.8%**, not 5×. And stop
competence went *backwards*: **834.19** against the control's **898.24**.

## Why it happened

The tracking income is a **product**, deliberately:

    reward_track = Φ_v · Φ_w

`learnings/011` established on the previous, absolute-command arm what that
product does when a policy buys speed with heading: Φ\_v rose to ×1.46 of the
control while Φ\_w fell to ×0.48, the two cancelled, and driving earned
essentially nothing while paying full control cost. The lesson there was
*speed bought with heading cancels*.

**The same trade happened here — and this time it paid.** Φ\_w still collapsed
(0.969 → 0.330 and 0.212 on the straight cells), but the product came out
positive anyway, for two reasons that are both visible in the numbers. First,
the relative tolerance made Φ\_v much easier to earn at a modest speed than the
absolute one had been. Second, `w_ctrl` was halved from 0.01 to 0.005, so the
control cost stopped eating the difference: on the drive grid the policy earns
**+78.15** of tracking income and pays **−48.69** of control, i.e. control is
**62.3%** of income. In `learnings/011`'s arm the equivalent control cost was
**102.9%** of the entire policy-vs-control gap. Halving the coefficient moved
"driving does not pay" to "driving pays."

What it did **not** do is give the policy any reason to make its speed depend
on the command — and this is the mechanism worth carrying:

**Φ\_v is a tolerance band, not a gradient toward the exact commanded value.**
It scores *how close you are*, with a width that grows with the command, and
near its own peak it is flat. A single speed can therefore sit inside the band
for a whole range of different commands and collect most of the income
available across that range — at the cost of exactly one gait, learned once,
instead of a family of gaits that must each be discovered separately.

Worked over the forward-drive command range the policy was actually trained on
(`vx` uniform on [0.30, 0.80]): **one fixed speed of 0.491 m/s collects a mean
Φ\_v of 0.764**, against the 1.000 that perfect command-following collects.
Three quarters of all the speed income in the reward is available to a machine
that never reads the command at all.

That is why one trot is a *rational optimum here* rather than a training
failure. The policy solved the problem posed. The problem posed did not require
tracking.

Two honest complications, both from the same script:

- The observed trot is **0.271 m/s, not 0.491** — it earns mean Φ\_v 0.433, not
  0.764. So the band explains why varying speed is not *worth much*; it does not
  by itself explain why the machine settled slower than the best single speed.
  Control cost, terrain, and simply not having found a faster gait in 2M steps
  are all live candidates and none is measured.
- Over the **full eval grid**, which includes a backward and a zero-speed cell,
  no single speed is a good answer: the best is 0.570 m/s at mean Φ\_v 0.411.
  The single-gait strategy is optimal against the *training* distribution,
  which is 80% forward drive, and the flat grid is what let it look optimal
  against the eval too.

## The math

Per step the reward is

    r = Φ_v·Φ_w + F_shaping − w_ctrl·Σᵢaᵢ² − w_contact·Σ|f| − K·1[terminated]

with, from `src/bestiary/envs/hound_track_rel.py` and
`runs/hound_track_rel_s1/config.json`:

- **Φ\_v** = `exp(−(e_v/α_v(c))²)` — speed match, unitless, ∈(0,1]
- **e\_v** = ‖(v\_fwd, v\_left) − (c, 0)‖, the planar velocity error in the
  **heading frame**, m/s. Below it is written `v − c`, which is `e_v` when
  lateral drift is zero — the approximation the paper arithmetic makes and the
  measured Φ\_v does not
- **v** achieved forward velocity in the heading frame, m/s;
  **c** commanded forward velocity, m/s
- **α\_v(c)** = `max(0.15, 0.5·|c|)`, m/s — the tolerance *width*, which grows
  with the command. `ALPHA_V_MIN = 0.15`, `BETA_V = 0.5`
- **Φ\_w** = `exp(−((ω − c_ω)/α_w(c_ω))²)` — heading match, unitless, ∈(0,1],
  with `α_w = max(0.10, 0.5·|c_ω|)` rad/s. `BETA_W = 0.5` this run, down from
  0.75
- **F\_shaping** potential-based shaping, γ = 0.99, Cauchy potential, weight 1.0
- **a\_i** the 16 motor commands, each ∈[−1, 1], unitless;
  **w\_ctrl** = 0.005 (was 0.01); **w\_contact** = 0.0005; **K** = 10, once

Note the kernel is `exp(−(e/α)²)`, not the `exp(−(e/α)²/2)` of a Gaussian
density — the factor of two is absorbed into α, and every tolerance above is
derived against that form.

**Work the band at the observed trot.** With `v = 0.271` m/s held constant:

    c = 0.50:  α_v = max(0.15, 0.25) = 0.25;  Φ_v = exp(−((0.271−0.50)/0.25)²) = 0.432
    c = 0.80:  α_v = max(0.15, 0.40) = 0.40;  Φ_v = exp(−((0.271−0.80)/0.40)²) = 0.174

One unchanged gait scores 0.432 under one command and 0.174 under another, and
neither is zero. The *measured* Φ\_v on those cells is 0.378 and 0.341 — the
same order, differing because the machine's speed fluctuates step to step
around the mean and the kernel is nonlinear.

**Now the quantity that decides whether reading the command is worth anything.**
Average Φ\_v over the trained forward command distribution for a fixed achieved
speed v:

    J(v) = E_{c ~ U[0.30, 0.80]} [ exp(−((v − c)/max(0.15, 0.5c))²) ]

    J(0.491) = 0.764        best single fixed speed
    J(0.271) = 0.433        the speed this policy actually runs
    perfect tracking        = 1.000 by construction (v = c, so Φ_v = 1 always)

**In plain English: a machine that never looks at the command can still earn 76%
of all the speed credit the reward has to give, so the last 24% is the entire
prize for learning to track — and it must be paid for with a harder control
problem and more motor effort.** That is not a bug in the implementation; it is
what a bounded tolerance kernel means. The gradient toward *exact* tracking
exists but it is shallow and short, and it lives on top of a plateau the policy
can stand on for free.

## What to do next time

**A gain fitted across a sign change is not a magnitude test.** Report the
*span* — the range of achieved speeds across the range of commanded speeds,
divided — alongside any slope. That statistic is `vx_span_ratio` and it now
lives in the guard: it reads 1.0 for a perfect tracker, 0.0 for one fixed gait,
and **0.127** for this run.

**Report the yaw axis or state that you cannot.** `track_eval` records
`achieved_vx` and has no `achieved_wz`, so there is no yaw gain to compute from
any committed artifact — the yaw evidence here is Φ\_w and the mirror-cell
asymmetry, and both had to be read by hand. This is `anomalies.jsonl` row 38,
and the guard names the gap rather than papering over it.

**A flat grid mean is not the objective.** `drive_grid_mean` weights six cells
equally; training weighted them 0.8/0.1/0.1. Whenever those two disagree by
5.04× versus 1.12×, the flat number is measuring the grid, not the policy.
Report both.

**A per-cell table beats a ratio.** "Loses to doing nothing in 3 of 6 cells" and
"98.8% of the win is one cell" are both invisible in an aggregate and both
obvious in six rows.

**Do not attribute this to any one change.** See below — four variables moved.

## How we would know this is wrong

**Provisional — one seed.** `hound_track_rel_s1` is a **probe**, not an effect.
Under this repo's seed rule nothing here is a claim about the reward *design*;
it is a claim about what this run did, and about a mechanism that would explain
it.

**Four variables moved against the previous arm, not one, and no attribution to
any single one is available.** Compared with `hound_track_desert_s0`
(`learnings/011`): commands became relative rather than absolute; `BETA_W` went
0.75 → 0.5; `w_ctrl` went 0.01 → 0.005; and a `pbrs_shaping` term (weight 1.0,
γ = 0.99, Cauchy potential) is present in this reward and was absent from the
previous one. Any statement of the form "the halved control cost is what made
driving pay" is a hypothesis consistent with the arithmetic above, not a
measured effect.

**The decomposition used here is itself incomplete.** `track_eval` hardcodes
four reward terms and this reward has five, so the reported terms do not sum to
the return: residual **−1.19** on the policy arm and **−0.77** on the control
arm, the latter being **20%** of its 3.91 baseline. That is `anomalies.jsonl`
row 36. Every term-level number above (income 78.15, control −48.69, "62.3% of
income") is therefore a *four-of-five* accounting, and the missing term is
`pbrs_shaping`.

This learning is wrong if any of these is observed:

- **Other seeds track.** If seeds 2 and 3 come back with achieved v\_x that
  scales with the command — say **≥0.45 m/s at a 0.5 command and ≥0.65 m/s at
  0.8**, i.e. `vx_span_ratio ≥ 0.5` — together with **Φ\_w above ~0.5 on the
  yaw-commanded cells**, then the single command-independent trot is a property
  of seed 1's particular basin and not of the reward, and the band argument
  explains nothing that needed explaining.
- **The mirror asymmetry disappears.** If a second seed returns the two
  (0.5, 0, ±0.4) cells within ~10 points of each other instead of 42.32 apart,
  the fixed-handedness claim is seed noise.
- **Longer training closes it.** If a 4M-step continuation of this same run
  reaches `vx_span_ratio ≥ 0.5` with no reward change, then one trot was a
  waypoint on the way to tracking rather than an optimum, and the correct
  reading is "undertrained," not "the band pays for standing on it."
- **The band argument is arithmetically wrong.** J(0.491) = 0.764 is computed
  by calling the env's own `relative_kernel` and `velocity_tolerance`. If the
  reward the trainer actually paid differs from those functions — the failure
  `anomalies.jsonl` row 28 records for reward instruments generally — then the
  central mechanism here is computed against a kernel nothing was trained under.
- **The missing fifth term reverses the accounting.** If `pbrs_shaping` is
  recovered and turns out to be the dominant income on the drive grid, then
  "the halved control cost let the product pay" is the wrong story about where
  the return came from, even if the command-independence observation survives
  untouched.
