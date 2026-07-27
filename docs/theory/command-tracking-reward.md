# Command-tracking reward for the hound

**Written:** 2026-07-27 · **Because:** the hound's old reward paid for
existing rather than for doing the task, and the replacement had to be derived
— tolerances, command distribution, penalty scale and all — rather than picked.

This note is the full derivation behind the command-tracking reward, written
as the answer to six review comments (C1–C6) raised against the first sketch.
Every constant below is either read out of this robot's own measurements or
derived from an inequality stated here.

---

> ## Status: DESIGNED, NOT YET IMPLEMENTED, NOT YET VALIDATED
>
> **No run has trained under this reward.** Nothing here is a result. Every
> number in §5 is a *prediction* computed from the kernel and the command
> distribution, not a measurement of a policy.
>
> This derivation must survive **independent refutation** and **a training
> run** before any claim is made for it. Until then it is a design on paper:
> cite it as a proposal, never as evidence, and do not let a later note treat
> its predicted separations (8×, 2.8×) as observed ones.
>
> **The tolerances are terrain-specific.** σ_v and σ_w are derived from noise
> measurements taken on the **GRID=1024** heightfield that is committed today.
> A terrain regeneration invalidates them. This is not a soft caveat:
> `research/scripts/compare_terrain_grids.py` measures a correlation of
> **+0.061** between GRID=1024 and GRID=2048 at the *same seed* — changing the
> grid does not resample the same desert, it rerolls a different one. If the
> heightfield is regenerated, the noise floor must be remeasured and σ_v and
> σ_w re-derived before this reward means what §2 says it means.

---

## The problem it solves

The old reward paid the hound for being alive and upright, plus a small term
for forward velocity. A machine that stands perfectly still and does nothing
collects nearly the whole thing — the measured gap between a trained policy
and zero action was **1.128×**. That is not a policy that learned to walk
badly; it is a reward whose payment is decoupled from the task, so the
optimizer correctly discovered that existence pays and locomotion does not.
The replacement below removes the mechanism rather than repricing it: the
*only* positive term is a product of two tolerance kernels, one on linear
velocity and one on yaw rate, both measured against an explicitly sampled
command. There is no term a non-tracking policy can collect. What follows
derives the two tolerances, the command distribution they are measured
against, the termination penalty that keeps early training from preferring
death, and the numbers the design predicts.

## 1. The full reward

Per step, with `Φ(u) = 1/(1+u²)`:

```
v_b     = planar trunk velocity, expressed in the trunk's yaw (heading) frame   [m/s, 2D]
ω       = trunk yaw rate about the body z-axis                                   [rad/s]
u_v     = || v_b − (v_x_cmd, v_y_cmd) || / σ_v          σ_v = 0.15 m/s
u_w     = | ω − ω_cmd | / σ_w                           σ_w = 0.10 rad/s

r  =  1.0 · 1[healthy] · Φ(u_v) · Φ(u_w)        (track_cmd — the ONLY positive term)
    − 0.01   · Σ a_i²                            (ctrl_cost, unchanged)
    − 5e-4   · Σ |F_contact|                     (contact_cost, unchanged)
    − 10.0   · 1[unhealthy termination]          (termination, one-time)
```

Constants: σ_v and σ_w derived in §2; the tracking weight is 1.0 by
normalization (product ∈ [0,1], sets the return scale in §5); K = 10 derived
in §4; the two cost coefficients are carried over unchanged deliberately —
they were not implicated in the exploit, and changing them in the same commit
as the structural fix would confound the before/after comparison.

**Frame warning (this is load-bearing):** the velocity error must be computed
in the heading frame, not the world frame. Under a nonzero ω_cmd the
world-frame velocity of a correctly-driving body rotates continuously;
tracking a fixed world-frame v_cmd while turning is unsatisfiable and would cap
Φ_v at ~0.5 on every turning segment. The old reward's "v_x of the trunk"
convention must not be inherited silently.

**Why the product, not a weighted sum (C1, structural argument):** the only
positive term requires *simultaneously* matching both channels. With additive
channels `w_v·Φ_v + w_w·Φ_w`, a standing machine under (v_cmd=0.5, ω_cmd=0)
harvests the entire yaw term for free — it is not yawing — collecting
~0.97·w_w per step, i.e. ~32% of the maximum at w_w=0.5. That is the alive
bonus reborn in a smaller coat. Under the product, the same machine gets
0.069 × 0.968 ≈ 0.067. The exploit mechanism of the old reward was *payment
decoupled from the task*; the product removes the mechanism, it does not
reprice it: there exists no term a non-tracking policy can collect, and no
coefficient rebalance can reopen it. The standing policy's per-step supremum
is fixed by the kernel tail at the commanded offset — a property of the
command distribution, not of tunable weights.

**Honest bound on the structural claim:** Cauchy tails are polynomial, so
"near zero" means 0.03–0.16 on the easiest drive slice, not 1e-4. A Gaussian
kernel `exp(−u²)` would make standing leak ~e⁻⁵ ≈ 0.007, but its gradient is
numerically zero beyond ~3σ — and a from-scratch SAC policy spends its first
million steps entirely beyond 3σ. Cauchy's gradient `2u/(1+u²)²` decays only
as 1/u³, so the far field still points home. That is why Cauchy composes well
with a product: no factor ever hits hard zero and deadlocks the other's
gradient. We keep Cauchy and pay the quantified leak (§5: ~4% of a good
policy's return); this is a stated trade, reversible by kernel swap if the leak
ever measures larger than predicted.

Note under this project's observation rules: commands fill the 3
already-reserved obs slots, so observation width does not change — this is not
a one-way door.

## 2. σ_v and σ_w — the two-sided derivation (C3)

The 3× rule as it stands in the record is one inequality. Sizing σ needs two,
and stating both is what resolves the trap the review found:

- **Noise side (lower bound):** σ ≥ 3 × unremovable noise, so the floor costs
  ≤ 10% (Φ(1/3) = 0.9).
- **Freeride side (upper bound):** the score of an *uncontrolled* machine must
  stay low. The trap arose from feeding the moving figure into the
  *noise-side* rule; but the moving figure is the sum of unremovable noise and
  exactly the error the policy exists to remove, so it belongs on the freeride
  side as a thing to *exclude*, never on the noise side as a thing to
  *tolerate*.

**σ_w.** Floor: standing yaw std 0.0182 → lower bound 3 × 0.0182 = 0.055.
Freeride: the unsteered driving machine yaws at 0.127; we require that at least
half the attainable reward be contingent on active yaw stabilization, i.e.
Φ(0.127/σ_w) ≤ 0.45 → σ_w ≤ 0.127/1.106 ≈ 0.115. Window **[0.055, 0.115]**;
we take the top of it, **σ_w = 0.10**, buying maximum headroom for
moving-phase unremovable noise (tolerates up to 0.033 rad/s at the 10% level —
1.8× the standing floor) while the unsteered machine scores Φ(1.27) = 0.383.
Check the halves: steered straight driving earns yaw factor ≈ 0.968, unsteered
0.383; product against a legitimate v-factor 0.9: 0.871 vs 0.345 — 60% of the
reward rides on steering. At the reviewed σ_w = 0.30 that gap was 0.87 vs
0.848·0.9 = 0.76 — a 12% incentive, which is why it was a hole.

**σ_v.** Floor: the two figures in the record are mutually inconsistent (§8) —
rms speed 0.0361 cannot be below a 0.05 mean drift on the same rollouts.
Reconciled floor: 0.04 ± 0.01. Lower bound 3 × 0.04 = 0.12. Freeride: the
analogue of the unsteered machine is the standing machine under the *easiest*
drive command (min |v_cmd| = 0.3, ω_cmd = 0); we cap its product take at ~0.15.
Standing error under forward min-command = 0.3 + 0.04 drift = 0.34, so
Φ(0.34/σ_v)·0.968 ≤ 0.15 → σ_v ≤ 0.146. Window **[0.12, 0.146]**; we take
**σ_v = 0.15** (the rounded top, binding the freeride cap at equality:
Φ(2.27)·0.968 = 0.163·0.968 = 0.158). Robustness to the data inconsistency: if
the true floor is 0.036, σ_v = 4.2× floor (noise costs 5.5%); if it is 0.05,
σ_v = 3.0× floor (costs exactly 10%). Either resolution of the inconsistent
measurements leaves the choice inside the rule — that is the actual defense of
0.15 over 0.11.

Uncontrolled scores under the final numbers: standing under (0.5, 0, 0):
0.069 × 0.968 = **0.067**. Unsteered roller at its lucky command
(0.35, 0, 0): ≈ 0.9 × 0.383 = **0.345** vs 0.87 attained by steering. Standing
under a turn command (0, 0, 0.45): 0.934 × 0.047 = **0.044**.

## 3. The command distribution (C2)

Resampled at intervals of U[200, 300] steps (jittered — see failure mode 6),
held constant between, written raw into the three obs slots. Per draw:

```
p = 0.10   STOP    (0, 0, 0)
p = 0.10   TURN    v = 0,  ω_cmd = s · U[0.3, 0.6],  s = ±1 equiprobable
p = 0.80   DRIVE   v_x_cmd = s_v · U[0.3, 0.8],  P(s_v=+1) = 0.8
                   v_y_cmd = 0
                   ω_cmd   = 0 w.p. 0.5, else U[−0.6, 0.6]
```

Defense of each number:

- **min |v_x_cmd| = 0.3** = 2σ_v: keeps every drive command ≥ 2 kernel widths
  from the standing point, capping the standing v-factor at 0.163; also the one
  open-loop point measured so far (0.3449 m/s from wheel cmd 0.3), so every
  command in range is known-achievable at the bottom.
- **max 0.8** is provisional: 0.3 wheel command gave 0.3449 m/s, suggesting
  ~1.1 m/s at saturation if linear, but that extrapolation is unverified. One
  scripted full-throttle rollout (20 episodes, same harness as the existing
  measurements) pins the true max; cap commands at ~80% of it. Commanding the
  untrackable would recreate the punished-for-unfixable-error problem in
  reverse.
- **v_y_cmd ≡ 0.** Pushback on the reserved slot's implied ambition: nobody has
  measured whether this wheel configuration can hold a lateral velocity at all.
  Commanding a channel of unverified controllability injects unremovable error
  into u_v and hands the gradient to noise. The error term still *includes* v_y
  (the planar norm), so lateral drift is penalized against the commanded 0,
  which is correct. Widen only after a controllability measurement.
- **ω range ±0.6** = 6σ_w, and 4.7× the unsteered drift 0.127, so no passive
  behavior accidentally matches a nonzero yaw command; turn-in-place min 0.3 =
  2.4× the unsteered figure for the same reason.
- **ω_cmd = 0 point mass at 0.5 within DRIVE:** straight driving is the
  dominant real command and *must* be trained; the freeride analysis in §2
  shows the unsteered machine gets 0.383 vs 0.968 on these slices — a 0.59 gap
  is the signal, not a hole.
- **p_stop = 0.10:** large enough that stopping is learned (with ~4 draws per
  episode, a stop segment appears in ~34% of episodes), small enough that the
  always-stand policy's legitimate stop harvest is ~0.09/step in expectation,
  ~11% of a good policy's rate.

**The ill-posedness the review flagged, addressed directly:** "does the policy
beat zero-action?" is undefined over a mixture containing STOP, because
zero-action is near-optimal on STOP *by design*. The test must be conditioned
on commands. Fix an eval grid — {(0.5,0,0), (0.8,0,0), (−0.3,0,0), (0.5,0,±0.4),
(0,0,0.45), (0,0,0)} — run policy and zero-action on identical command
sequences paired by seed, and report the ratio **excluding the (0,0,0) cell**,
which is reported separately as the stop-competence check (where the policy
should *match* zero-action, not beat it). The exploit metric is dead only
relative to this stated protocol; publish the grid with the number, always.

## 4. The alive bonus (C4)

**Decision: b = 0.** Health enters only as the multiplicative gate on the
tracking product, plus a one-time termination penalty. Any additive b
re-creates the exploit at scale b × 1000 — it pays existence linearly in
episode length, decoupled from commands; shrinking it only shrinks the ratio,
it does not remove the mechanism. Survival is paid *implicitly*: a dead policy
collects zero tracking forever after. Costs stay **outside** the gate — gating
costs by health would make death erase pending costs, a small suicide subsidy.

(Today termination ≡ unhealthy, so the gate is redundant on all non-terminal
steps; it is written in so the reward stays correct if health and termination
ever decouple, e.g. a future recovery curriculum.)

**Termination penalty K = 10, derived:** the optimal policy never prefers death
for any K ≥ 0, because a strictly positive continuation is always available —
zero action yields ≈ +0.09/step net (§5) from anywhere upright. K exists to
shape the *learning path*: early SAC, with near-max-entropy actions
(E[a²] ≈ 1/3 per dim → ctrl cost ≈ 0.053) plus contact ≈ 0.05 and negligible
tracking, runs ≈ −0.10/step net, and for such a policy dying early is locally
attractive. The discounted value of escaping a −c/step stream is c/(1−γ). With
γ = 0.99 and c = 0.10:

```
K = c / (1−γ) = 0.10 / 0.01 = 10
```

so even a policy that believes it can never do better than worst observed
flailing is exactly indifferent to death, and anything better strictly prefers
living. (If γ ≠ 0.99 in the trainer, rescale K accordingly; if the cost
coefficients are ever enlarged, K must be re-derived — they are coupled.) K is
deliberately not larger: over-penalizing terminal proximity breeds timidity at
the uprightness margin, which on rough terrain is where the work is.

**The three returns (per-step nets from §5):**

| policy | return |
|---|---|
| (a) stands 1000 steps | 0.087 × 1000 ≈ **+87** |
| (b) tracks perfectly, 1000 steps | ≈ 0.70 × 1000 ≈ **+700** |
| (c) falls at step 200 | standing-quality: 0.087·200 − 10 ≈ **+7**; perfect-quality: 0.70·200 − 10 ≈ **+130** |

Suicide check: falling is worse than the same behavior surviving in every row
(7 < 87, 130 < 700); stand-then-fall < stand (7 < 87) ✓. Note 130 > 87 — a
tracker that dies at 200 outscores a permanent stander — and that ordering is
*correct*: 200 steps of the task beat 1000 steps of nothing, and the gradient
from 130 points at "keep tracking and survive" (700), not at standing.

## 5. Worked numbers

Command-mixture expectations, σ_v = 0.15, σ_w = 0.10, drift 0.04, 1000 steps.
(Standing drive-slice numbers are exact integrals of the Cauchy kernel over the
command range, e.g. E[Φ_v | forward drive] = (0.15/0.5)·[arctan(5.6) −
arctan(2.27)] = 0.072.)

**Standing (zero action):**

- STOP: 0.934 × 0.968 = 0.904 · TURN: 0.934 × 0.052 = 0.049 ·
  DRIVE: 0.077 × 0.601 = 0.046
- E[track] = 0.1(0.904) + 0.1(0.049) + 0.8(0.046) = **0.132/step** (68% of it
  earned legitimately on STOP; the true freeride is ~0.042/step ≈ 4% of a good
  policy's rate)
- costs: ctrl 0, contact ≈ 0.045 → net **0.087/step → ≈ 87/episode**. Under the
  drive-only eval grid: ≈ 0.046 − 0.045 ≈ **0 net**.

**Perfect tracker** (residuals: v = 0.06, i.e. 1.5× floor; yaw = 0.03):
Φ_v = 0.862, Φ_w = 0.917, product 0.79; ≈ 0.90 on STOP → E[track] ≈ 0.80/step;
costs ≈ 0.07–0.10 → net **≈ 0.70/step → ≈ 700/episode** (range 650–750
depending on true moving noise).

**Mediocre tracker** (both errors ≈ 1σ): DRIVE 0.5 × 0.5 = 0.25, STOP 0.85,
TURN 0.45 → E[track] = 0.33; costs 0.09 → net **≈ 0.24/step → ≈ 240/episode**.

Separations: perfect/standing ≈ **8×**, mediocre/standing ≈ **2.8×**, versus
the old 1.128×.

**C5 — comparability:** the review is right, the old 955–1078 numbers are dead
as comparators, twice over: the objective changed (existence-pay →
tracking-pay; the scales aren't even the same axis), and the old policies were
trained with zero-filled command slots, so evaluating them under nonzero
commands feeds them observations off their training manifold — undefined
behavior, not a baseline. The only number that carries over is the *zero-action
baseline re-measured under the new reward and stated command grid* (predicted
≈ 87 on the mixture, ≈ 0 on the drive grid). New headline metrics: the paired
ratio from §3 (predict ≥ 5× for a successful run), and the unitless mean
tracking score E[Φ_v·Φ_w] ∈ [0,1], which survives future cost-coefficient
retuning and episode-length changes.

## 6. Ordered term list for the spec hash (C6)

```
1. track_cmd      = 1.0  · healthy · Φ(u_v) · Φ(u_w)     params: σ_v=0.15, σ_w=0.10, Cauchy, heading-frame
2. ctrl_cost      = −0.01 · Σ a²
3. contact_cost   = −5e-4 · Σ|F|
4. termination    = −10 · 1[unhealthy terminal]
```

Two flags for the hashing machinery, both real holes:

- `track_cmd` is internally multiplicative and **must not** be decomposed into
  `track_lin` + `track_ang` for the hash — that list would collide with a
  genuinely additive two-term reward, which is a *different objective* (§1).
  One name, with σ_v, σ_w, kernel, and frame recorded as its parameters. A
  name-only hash cannot see a σ change, and a σ change *is* a reward change;
  hash (name, weight, params) tuples.
- **The command distribution is part of the objective but lives outside the
  reward function** (the sampler, not the reward, decides what u_v means in
  expectation). Two runs with identical term lists and different command
  mixtures are incomparable. Add a `cmd_dist` version string to the hashed
  spec. Without this, the exact class of silent-change bug the hash exists to
  catch walks straight past it.

## 7. Failure modes, ranked, with cheapest detections

1. **Stand-still local optimum (basin, not exploit).** The reward no longer
   *pays* standing, but standing (~0.13/step) is trivially reachable and
   tracking is not; SAC may park. Detect: Φ_v conditioned on DRIVE slices at
   each checkpoint; < 0.15 at 500k steps with total return ~120 = parked.
   Cheapest: the existing eval with per-command-type decomposition added.
   Pre-registered levers (decide now, not mid-run): p_stop → 0.05 or
   min |v_cmd| → 0.4.
2. **Yaw-freeride plateau.** "Match speed, ignore heading" earns 0.345 vs
   0.87 — a plateau SAC can sit on. Detect: Φ_w under ω_cmd = ±0.4 eval cells
   specifically; also variance of the steering-relevant action channels. If
   Φ_w|turn < 0.2 while Φ_v > 0.7 across 3 checkpoints, it is real; the lever
   is raising the P(ω_cmd ≠ 0 | DRIVE) from 0.5.
3. **Dead command input.** The policy may ignore the three obs slots entirely
   (especially if mode 1 bites first). Detect, very cheaply: regress achieved
   velocity on commanded velocity across eval episodes — the "command gain"
   slope should approach 1; slope ≈ 0 = dead input. One scatter plot from
   existing logs.
4. **σ_v undersized for true moving noise** (the review's crux — the standing
   floor is only a lower bound). Symptom: trained Φ_v saturates at 0.5–0.7 with
   velocity-error rms > σ_v/2 = 0.075, flat across checkpoints. Detect:
   residual error rms per checkpoint. Disambiguate incapacity from mis-sizing
   with one scripted steered rollout measuring achievable error before blaming
   σ. Revisit trigger, pre-registered: flat residual > 0.075 over 3 checkpoints
   → remeasure the floor while moving, re-derive σ_v; do not retune blind.
5. **Suicide spiral early in training.** Net-negative flailing plus K mis-sized
   to actual γ or actual costs. Detect: median episode length over the first
   300k steps; sustained collapse below ~400 = spiral. Already in SAC logs,
   zero cost.
6. **Frame bug** (world-frame velocity under yaw commands). Detect: Φ_v
   conditioned on |ω_cmd|: a systematic gap between turning and straight slices
   that no training closes is the signature. One conditional mean from logs.
7. **Resample-rhythm overfit.** The interval jitter U[200,300] is the
   mitigation; verify by evaluating with a shifted-phase, fixed-interval
   schedule and checking for a score drop.
8. **Contact-cost oddity, inherited:** −5e-4·Σ|F| charges ≈ 0.045/step for
   *supporting the robot's weight* — a negative alive bonus wearing a cost's
   clothes, and an incentive toward ballistic/contact-light gaits at the
   margin. Kept unchanged here to avoid confounding the structural fix, but the
   clean form is penalizing only exceedance, max(ΣF − 1.2·mg, 0). Detect drift
   toward contact-avoidance in flight-phase fraction from logs. Flagged as the
   next reward edit, separately committed.
9. **Cauchy tail credit for wrong-direction motion:** driving at −0.3 under
   +0.3 command scores Φ(4.3) ≈ 0.05. Quantified, accepted; listed so nobody
   rediscovers it as a surprise.

## 8. Where the prior design and the record are wrong, plainly

1. **The noise measurements contradict each other.** Standing planar speed rms
   0.0361 m/s and mean backward drift ~0.05 m/s cannot both describe the same
   rollouts: rms speed ≥ |mean velocity|, always. Either the drift figure is
   stale/rounded or the two were measured on different phases (settling vs
   settled). The old data leans against 0.05: a zero-action episode at drift
   0.05 loses 50 points of v_x, giving ≤ 950 before costs, below the measured
   955–960. Pull the per-term decomposition of one standing episode from logs
   and pin the floor; the σ_v = 0.15 chosen here is chosen to satisfy the 3×
   rule under *either* resolution (3.0× if floor = 0.05, 4.2× if 0.036), which
   is why the inconsistency doesn't block the design — but the record should
   not carry both numbers uncorrected.
2. **"The 3× rule is derived, not a convention" — half true.** Φ(1/3) = 0.9 is
   a theorem about the kernel; that unavoidable noise *should* cost exactly 10%
   is a chosen operating point (why not 5%? 20%?). It becomes principled only
   when paired with the freeride-side inequality, as in §2 — one inequality is
   a preference, two are a derivation. Own it as the pair.
3. **Arithmetic: 3 × 0.127 = 0.381, not 0.30.** The stated "3× rule on the
   moving figure gives σ_w = 0.30" doesn't reproduce; some other multiplier or
   rounding intervened. The trap's conclusion is unaffected, but the record
   shouldn't contain a derivation that doesn't check.
4. **C1's "near zero" is unachievable with the kernel proposed, and that's
   fine.** Cauchy tails leak 0.03–0.16 on the easiest slices, structurally. The
   defensible claim is "no payment without matching, leak quantified and
   bounded by the min-command choice, separation ≥ 5×" — not "zero". If a
   reviewer demands sub-1% leak, the answer is a Gaussian kernel and a dead
   far-field gradient for early SAC; we recommend against, but it is a trade to
   state, not to hide.
5. **The exploit *test*, not just the reward, was under-specified** — and C2
   half-saw this. "Beats zero-action" is meaningless without the command
   protocol attached, and the old torque-variant CI straddling zero shows the
   unpaired test also lacked power. The fix is both stated in §3: paired-by-
   command-sequence, stop-cell excluded and reported separately. The metric and
   its protocol should be hashed together with the reward spec.
6. **The reserved v_y slot smuggles in an unverified claim** — that lateral
   velocity is controllable on this platform. Keep the slot, command zero, and
   measure controllability before ever sampling it; otherwise the
   punished-for-unremovable-error problem just diagnosed gets rebuilt one
   channel over.

## See also

- `research/CORE_PLAN.md` — the reward and observation spec this replaces
- `research/scripts/measure_tracking_noise.py` — the yaw and linear noise
  floors σ_v and σ_w are derived from
- `research/scripts/compare_terrain_grids.py` — the +0.061 correlation that
  makes those floors terrain-specific
- `src/bestiary/envs/hound.py` — the env whose reward this changes
- `src/bestiary/envs/obs_spec.py` — the reserved command slots, and the
  one-way door this design deliberately does not open
