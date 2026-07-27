---
triggers: [reward_change, comparison, metric_added, long_run]
guard: parked-detector — src/bestiary/guards/parked_detector.py
last_confirmed: 2026-07-27
---

# 011 — The crash count was 0.9% of the gap

**Date:** 2026-07-27 · **From:** `hound_track_desert_s0`, the first run under
the command-tracking reward
**Robot:** hound

## What we believed before

That a policy failing a locomotion reward fails in one of two recognisable
ways: it is **parked** (it discovered that standing still pays best — this is
`learnings/001` and `learnings/005`, and it has bitten this project twice), or
it is **broken** (it falls over and the termination penalty eats the return).

That belief was reasonable and it was built from evidence. Both prior hound
failures really were the parked kind, and the command-tracking reward was
designed specifically to close that hole — `docs/theory/command-tracking-reward.md`
enumerates nine failure modes and puts "parked in the standing basin" first.
So when the run came back at **−6.48** on the drive grid against zero action's
**55.73**, with **71 of 120** drive episodes ending in a crash at the 950k
checkpoint, the two candidate stories were already in mind and one of them fit
the visible evidence perfectly.

The mistake was not choosing the wrong story. It was that **a return is a sum
of terms and we read it as one number**, then reached for the most visible
correlate of a bad number rather than splitting it.

## What happened

The claim written was: *the negative return is caused by falling over, so the
pre-registered lever for a parked policy is the wrong response.* An independent
refutation killed the causal half of it in one table, by doing the thing that
should have been done first — decomposing the return.

`HoundPDTrackDesert-v0` already reports all four reward terms in its `info`
dict (`src/bestiary/envs/hound_track.py:328-331`), so no re-derivation was
needed. Measured on the unselected final checkpoint at 20 episodes per cell
(`research/scripts/track_return_decomposition.py`, residual 2.8e−14 — the split
is exact, not approximate):

| term | policy | zero action | contribution to the gap | share |
|---|---|---|---|---|
| `reward_track` | 67.09 | 65.01 | **+2.09** | −3.4% |
| `reward_ctrl` | −65.59 | 0.00 | **−65.59** | **105.5%** |
| `reward_contact` | −7.39 | −9.28 | +1.89 | −3.0% |
| `reward_termination` | −0.58 | 0.00 | **−0.58** | **0.9%** |
| **return** | **−6.48** | **55.73** | **−62.20** | |

The crashing that the whole diagnosis rested on is **0.9%** of the gap. Control
cost is **105.5%** — more than all of it, offset slightly by the policy
actually contacting the ground *less* than a machine lying still does.

Then a natural experiment arrived by accident. `ant_sac_best.zip` was
overwritten in place during the review (an anomaly in its own right, now
recorded). The newer checkpoint crashes **1 time in 120** instead of 71 — and
its drive-grid return is **−4.87** against the old **−4.87/−6.48** band.
**Remove essentially all of the falling over and the number does not move.**

## Why it happened

The policy is not parked and it is not broken. It drives, and driving is not
worth what it costs.

Two things had to be true at once, and each is measured:

**One — it really does read the command.** Mean Φ_v over the five
nonzero-speed cells is **0.1981** against zero action's **0.0985**, i.e. ×2.01.
It moves in the commanded direction at roughly half the commanded magnitude.

**Two — it pays for that motion in heading, and the reward multiplies.** The
tracking term is a product, deliberately (`docs/lessons/003`):

| | zero action | policy | ratio |
|---|---|---|---|
| Φ_v (speed match) | 0.240 | 0.350 | **×1.46** |
| Φ_w (heading hold) | 0.513 | 0.245 | **×0.48** |
| Φ_v·Φ_w | 0.0650 | 0.0955 | ×1.47 raw |

A machine that stands still holds its heading almost perfectly and scores
nearly full credit on Φ_w for free. A machine that drives on this terrain
yaws, and loses roughly half of it. Because the two factors multiply, the ×1.46
it *gains* in speed is cancelled by the ×0.48 it *loses* in heading. Net
tracking earned: **essentially the same as doing nothing.**

And then it pays the control cost that doing nothing does not pay, and that is
the entire return.

*(The ×1.47 raw figure above is length-biased — it averages a per-step score
over 952-step episodes against 1000-step ones. Length-normalised it is ≈0.99.
This does not change the mechanism, which is the per-term decomposition above,
but the raw ratio must not be quoted as "the policy beats zero action on
tracking." It does not.)*

## The math

Per step, the reward is

    r = Φ_v·Φ_w − w_ctrl·Σaᵢ² − w_contact·Σ|f| − K·1[terminated]

**Φ_v** speed-match score, unitless, ∈[0,1]; **Φ_w** heading-match score,
unitless, ∈[0,1]; **aᵢ** the 16 motor commands, each ∈[−1,1], unitless;
**w_ctrl** = 0.01, unitless; **w_contact** = 0.0005; **K** = 10, one-time.

Divide the measured per-episode totals by the measured mean episode length
(952.2 steps for the policy, 1000.0 for zero action) to get the per-step rates
that decide whether motion is worth buying:

    policy   track:  67.09 / 952.2  = 0.07046 per step
    zero     track:  65.01 / 1000.0 = 0.06501 per step
    gain from driving                = 0.00545 per step

    policy   ctrl:   65.59 / 952.2  = 0.06889 per step
    zero     ctrl:    0.00 / 1000.0 = 0.00000 per step
    cost of driving                  = 0.06889 per step

    ratio = 0.06889 / 0.00545 = 12.6

**In plain terms: after 1.5 million steps of training, the machine earns
0.0055 of extra reward per step for driving, and pays 0.0689 per step to do
it. It pays about 12.6 times what it earns.** A policy that maximises return
should stop driving — and the fact that it has not is a statement about
optimisation being unfinished at 1.5M steps, not about the reward being
gameable.

This is the same arithmetic shape as `learnings/001`, where the flat-tuned
reward had the hound paying 0.57 per step to earn 0.29. That one was caught
because the machine visibly stood still. This one hid because the machine
visibly *moved*.

## What to do next time

**Decompose the return before naming a failure mode.** It cost two minutes here
and it overturned the diagnosis completely. Any reward that is a sum of terms
can and should be reported term by term, against the control arm, with a
residual check — a decomposition whose residual is not ~0 is not a
decomposition.

**Do not apply the pre-registered lever.** `p_stop → 0.05` and
`min |v_cmd| → 0.4` both push *more* driving at a machine whose problem is that
driving does not pay. Both would make the return more negative. The
pre-registration did its job — it forced the gate to be honest — but a
pre-registered *response* is only as good as the failure mode it assumed, and
this cycle observed a mode the design did not enumerate.

**The live question is the ratio, not the crash rate:** either the achievable
Φ_v·Φ_w has to rise (heading control is the limiting factor, not speed), or
`w_ctrl` has to fall far enough that 0.0055/step of gain is worth buying.
Choosing between those is a reward-design decision and belongs with the
mathematics, not with a coefficient nudge.

## How we would know this is wrong

**Provisional — one seed.** This is a probe. It cannot support a claim about
the reward *design*, only about what this run did.

Concretely, this learning is wrong if any of these is observed:

- **A longer run closes the gap on its own.** `ep_rew_mean` was **43.24** and
  still climbing at the step budget (29 at 1.02M), so 1.5M did not converge. If
  a 4M-step run reaches a positive drive-grid return with `w_ctrl` unchanged,
  then the reward was fine and the run was simply short — the claim that
  driving cannot pay would be false.
- **The Φ_w collapse is terrain-specific.** If the same policy on flat ground
  holds Φ_w near zero action's 0.513 while driving, then heading loss is a
  terrain-contact problem, not a reward-balance problem, and the cost ratio is
  a symptom rather than the cause.
- **Another seed decomposes differently.** If seed 1 comes back with the
  termination penalty as a large share rather than ~1%, then 0.9% was a
  property of this seed's particular gait and not of the reward.
- **The control-cost measurement is an artifact of the deterministic policy.**
  All numbers here are from `deterministic=True` rollouts. If the stochastic
  policy the return was actually optimised under pays materially less control
  cost, the ratio computed above is not the one training saw.
