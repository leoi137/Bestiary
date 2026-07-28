# Episode 012 — the win was one cell

**Run:** `hound_track_rel_s1` · seed 1 · 2,000,000 steps · `HoundPDTrackRelDesert-v0`
**Date:** 2026-07-28 · **Ledger:** row 5 · **One seed, so a probe.**

---

## Thesis

Under the previous command-tracking reward, driving lost to standing still.
Episode 011's arm scored a drive-grid mean of **−6.48** against a do-nothing
control's **55.73**, and the control cost accounted for **102.9 %** of the gap.

This run changed the reward — relative commands instead of absolute, `BETA_W`
0.75 → 0.5, `w_ctrl` 0.01 → 0.005 — and asked one question: **does driving pay
yet?**

It does. And then the answer got much smaller under inspection.

## Diagnosis going in

The suspicion was that the objective was fine and the control cost was eating
it. The tracking reward is a **product**, `Φ_v · Φ_w` — speed tracking times
heading tracking — so a policy that buys speed by sacrificing heading gains
nothing, and on the previous arm exactly that happened: `Φ_v` rose to 1.46× the
control while `Φ_w` fell to 0.48×, and the product went nowhere while the
control cost decided the sign.

Halving `w_ctrl` tests whether the cost term was the binding constraint.

## What happened

Measured on a fixed grid of six drive commands plus a stop cell, 20 episodes
per cell, seeds 1000–1019, deterministic, paired against a zero-action control
on identical command sequences.

| | zero action | trained (`ant_sac_best.zip`) |
|---|---|---|
| drive-grid mean, six cells | 3.91 | **19.73** |
| drive-grid mean, five in-distribution cells | −1.58 | **18.51** |
| tracking income, `track_per_horizon` | 0.01397 | **0.07815** (5.59×) |
| reward terms — track / ctrl / contact / term | +13.97 / 0 / −9.28 / 0 | +78.15 / **−48.69** / −8.12 / −0.42 |
| stop-cell mean | 898.24 | 834.19 |

**The sign flipped.** Tracking income rose 5.6× on the length-comparable field,
and the control cost is no longer the whole story — 48.7 against an income of
78.1, where before it was 102.9 % of the entire gap.

Both headline numbers are quoted because one of the six commands, `(-0.3, 0, 0)`,
is off-manifold: the environment samples backward speeds from −0.8 to −0.4, so
a −0.3 command is one the policy never trained on, and a do-nothing control
collects most of the bar there. Neither headline is allowed to stand alone.

## Then the per-cell table

Policy minus control, cell by cell:

| cell | Δ | policy `Φ_w` |
|---|---|---|
| (0.5, 0, 0) | **+93.82** | 0.330 |
| (0.8, 0, 0) | +37.07 | 0.212 |
| (0.5, 0, −0.4) | +24.52 | 0.178 |
| (−0.3, 0, 0) | −5.49 | — |
| (0.5, 0, +0.4) | −17.80 | 0.141 |
| (0.0, 0, 0.45) | −37.18 | 0.040 |
| **sum** | **+94.93** | control: **0.969** |

**One cell is 93.82 of 94.93 — 98.8 % of the entire gap** (in grid-mean units,
+15.64 of +15.82). Remove it and the
five-cell means are 4.95 against 4.73: a dead heat. The policy **loses to doing
nothing in three of the six drive cells**. Dropping any single cell moves the
headline ratio across 1.05×, 3.15×, 3.78×, 4.44×, 9.69×, and one case where it
is undefined because the control's mean goes negative.

A summary statistic that ranges over an order of magnitude when one of six
equally weighted points is removed is not an estimate of anything.

## What the policy actually learned

Not command tracking. A **single forward trot**, run regardless of what was
asked:

- Commanded 0.5 → achieved **0.271** m/s. Commanded 0.8 → achieved **0.309**.
  Roughly one speed, 55 % of one command and 38 % of the other.
- **Yaw is not tracked.** `Φ_w` is 0.141, 0.178 and 0.040 on the three
  yaw-commanded cells — small in absolute terms, where 1.0 is perfect. Note the
  control is *also* wrong there (0.019), so the policy is nominally 7–9× it; the
  comparison that looks damning is not the one that proves the point. On the
  *straight* cells, where the control scores 0.969, the policy manages 0.330 and
  0.212 — it holds heading **worse than standing still does**.
- The gait has a **fixed handedness**: the mirror-image cells (0.5, 0, +0.4)
  and (0.5, 0, −0.4) differ by **42.3** return points. A policy that tracked yaw
  would be symmetric under the sign of the yaw command. **This, not the Φ_w
  magnitudes, is the evidence that there is no steering** — an asymmetry that
  large cannot be produced by a policy responding to the yaw command at all.

`command_gain`, the metric written to catch exactly this, read **0.382** and
raised no objection — because it regresses achieved *forward* velocity on
commanded, and it is large here only because the machine creeps backward on the
one backward cell while trotting forward on the rest. It is a sign detector, not
a magnitude tracker, and it has no yaw counterpart.

Weighted by how often each command actually occurs in training rather than by a
flat grid, the margin over doing nothing is **+11.8 %**, not 5×. The flat grid gives the stop cell zero weight; the
objective gives it a tenth. Stop competence **regressed**, 834.19 against 898.24.

## How the prediction did

Seven predictions, committed before the run landed and before any harvest
number existed. **Five correct.**

| | claim | p | |
|---|---|---|---|
| P1 | tracking income beats the control | 0.65 | ✅ 0.08098 |
| P2 | the sign flips | 0.35 | ✅ 19.73 |
| P3 | `command_gain` > 0.19 | 0.65 | ✅ 0.382 — and it meant nothing |
| P4 | the aggregates split by episode length | 0.65 | ❌ length barely moved |
| P5 | the checkpoints agree within 10 points | 0.75 | ❌ they differ by 28.5 |
| P6 | control cost in [10, 50] | 0.62 | ✅ 48.69 |
| P7 | reaches 2M and lands inside its ceiling | 0.78 | ✅ 25 min inside |

P2 at 0.35 was the lowest-confidence call and the one that mattered; it came in.
Running hit rate 48 % → **53 %**.

P5's failure is the more interesting one. The two checkpoints differ by 28.5
points — but that is **not** a selection effect. `ant_sac_best.zip` was written
at roughly 1.48M steps against the final checkpoint's 2M, on a run whose reward
was still climbing, so 520k steps of extra training are confounded into the
comparison. The checkpoint file records the score it was selected on but not the
step, which is what made this invisible until it was looked for.

## What this does and does not license

**Does:** two more seeds. The sign flipped and the cost term is no longer
binding, which is what the run existed to determine.

**Does not:** any claim about yaw, about tracking, or about *why*. Four
variables moved between this arm and the last — relative commands, `BETA_W`,
`w_ctrl`, and a potential-based shaping term present here and absent before.
No result here is attributable to any one of them. One seed makes every number
above provisional.

A defect found in the process: the reward decomposition quoted in the table is
**incomplete**. The instrument hardcodes four reward terms and this reward has
five, so the terms do not sum to the return — the residual is 0.77 on the
control arm, **20 % of its baseline**, and 1.19 on the policy. The previous episode's entire conclusion
was a decomposition argument, which is what makes a silently missing term worth
fixing before the next one is made.

## Open

- Is the single trot a property of the reward or a quirk of this seed? Two more
  seeds answer it. If all three settle at ~0.27 m/s regardless of command, the
  speed-tracking tolerance band is too wide to induce magnitude tracking, and
  that is a reward-design finding rather than a training one.
- Why does `Φ_w` fall *below* the do-nothing control on straight cells? The
  policy is paying a heading penalty it did not have to pay.
- Is an unweighted mean over signed cells the right headline at all? On this
  evidence, no.
