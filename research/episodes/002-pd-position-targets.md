# 002 — PD position targets: does changing the action space unstick the hound?

**Date:** 2026-07-25 · **Robot:** hound · **Run:** `hound_pd_desert_v0` (HoundPDDesert-v0)

Follows `001-hound-throughput.md`, which ranked PD position targets as the
single highest-leverage change available. This episode does that change and
tests it. Throughput (item 2 on that list) is deliberately untouched — one
variable at a time.

## Thesis

`hound_desert_v0` plateaued because under torque control the policy had to
learn *how to hold a pose* before it could learn where to put its feet. Every
one of the twelve leg motors had to be commanded, every step, just to keep
standing — and standing was worth 961 while moving was worth 1043, a margin
of 82 for a much harder behaviour.

PD position targets remove that burden. `tau = kp*(target - q) - kv*qdot` runs
in the simulator at 200 Hz, and a zero action now commands the standing
stance exactly. Standing is free. The policy's whole job becomes deciding
where to put the feet.

If the diagnosis in 001 was right, this should move the result. If it does
not, the bottleneck is exploration and sample count rather than the action
parameterization — which would strengthen the MJX argument in decision 0001.

## What was measured before training

On `HoundPDDesert-v0`, legs held at stance, wheels commanded open-loop:

| wheel cmd | return | steps survived | x (m) | v (m/s) |
|---|---|---|---|---|
| 0.0 | 954.1 | 1000 | −1.82 | −0.036 |
| 0.1 | 962.5 | 1000 | −1.39 | −0.028 |
| 0.2 | 538.6 | 442 | 5.10 | 0.231 |
| **0.3** | **1095.5** | **1000** | **5.41** | **0.108** |
| 0.5 | 217.8 | 102 | 5.93 | 1.164 |
| 0.7 | 214.6 | 81 | 1.688 | 1.688 |
| 1.0 | 312.1 | 119 | 9.97 | 1.675 |

Two things matter here.

**The standing check passes.** 1095.5 rolling beats 954.1 standing, a margin
of 141 — wider than the torque model's 82. The reward is not instructing the
machine to stand still.

**The survivable band is a knife-edge, and it is not monotonic.** 0.3 survives
a full episode; 0.2 and 0.5 both die inside 450 steps. A policy exploring
wheel torque will be punished on both sides of a narrow ridge. That is the
risk this run actually tests.

Also confirmed before starting: observation width unchanged at 169, so
CORE_PLAN's observation contract is untouched; steady-state joint error when
commanded to its own stance is 0.4°, so the parallel springs and the servo
are not fighting; the torque model still passes 38/38.

## Prediction

Written 2026-07-25, before the run started. Baselines to beat: **954**
(standing) and **1096** (open-loop rolling at 0.3). The torque run plateaued
at ~1010 after 3.75M steps.

- **55%** — `ep_rew_mean` clears **1096** by 1M steps. That is the bar that
  means the policy learned something better than a fixed open-loop roll.
- **Most likely single outcome:** `ep_rew_mean` lands in **1000–1400**,
  `ep_len_mean` near **1000** (survival is nearly free now — standing costs
  nothing), forward distance **5–20 m**, still on the flat basin, still no
  dune reached.
- **25%** — it parks at standing. `ep_rew_mean` pinned in **950–1000** with
  `ep_len_mean` at 1000 and almost no forward motion. This is the failure mode
  I expect most if it fails: standing is now *trivially* reachable (the zero
  action), and the crash penalty either side of the 0.3 ridge makes exploring
  speed expensive. PD could make the local optimum easier to fall into, not
  harder.
- **<10%** — it exceeds 2000, i.e. a real gait rather than a roll.
- `ent_coef` collapses below **0.02 by 60k steps**, as it did in both previous
  hound runs (0.011 by 57k on the throwaway, 0.0218 final on the 3.75M).

**What would prove the 001 diagnosis wrong:** `ep_rew_mean` plateauing at
~1010 again, the same number as the torque run, from a materially different
action space. That would say the control parameterization was never the
binding constraint.

**What would prove it right:** clearing 1096 with `ep_len_mean` at 1000, at a
step count well under the 3.75M the torque run needed.

## Run parameters

`HoundPDDesert-v0`, SAC, seed 0, 1,000,000 steps, no wrapper, default
hyperparameters. Sized to finish inside ~2.5 hours at the ~129 steps/s the
previous run sustained, so the system can be adjusted before committing to a
longer one.

## Open questions inherited by 003

- **Throughput is still untouched.** Whatever this run says, four experiments
  a week remains the real limit. Decision 0001 chose MJX; nothing here changes
  that.
- **The command spec is still undecided.** 001 asked whether to settle the
  velocity-command interface at the same time as the action space. The action
  space has now changed without settling it, so the 3 reserved command slots
  in the observation are still zeros. That door is still open, and closing it
  is still a from-scratch retrain.
- **`ctrl_cost_weight` is unexamined under PD.** It is still 0.01, sized for
  torque actions where the action *was* the torque. Under PD the action is an
  offset from stance, so the term now penalizes moving the legs away from
  standing — a different thing wearing the same name. It may be an unintended
  crouch-and-freeze prior.
- **Why is the survivable band non-monotonic?** 0.2 crashing while 0.3
  survives is not explained. Worth a look if this run underperforms.
