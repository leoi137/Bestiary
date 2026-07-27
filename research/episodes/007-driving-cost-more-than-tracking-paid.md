# 007 — Driving cost more than tracking paid

**Date:** 2026-07-27 · **Run:** `hound_track_desert_s0` (1,500,000 steps, seed 0,
`HoundPDTrackDesert-v0`) · **Robot:** hound

## Thesis

The command-tracking reward was built to close a specific hole: under the old
forward-velocity reward the hound was paid for staying alive, and standing still
was competitive with moving. Episode 006 replaced the forward term with a
tracking term that pays only for matching a commanded velocity, measured that
standing earns just 0.0652 of a possible 1.0 on driving commands, and launched
the first run under it with a gate registered in advance.

**The gate: by 800k steps the drive-grid mean must exceed 2× zero action, i.e.
111.5.** Below that, the run is called *parked* — the machine found the
standing basin again — and two levers were pre-registered as the response.

The gate failed. **The failure mode it named did not happen**, and the response
it pre-registered would have made things worse.

## Diagnosis

At 1.5M steps, measured on the unselected checkpoint at 20 episodes per grid
cell against a zero-action arm on identical seeds:

| | zero action | policy |
|---|---|---|
| drive-grid mean return | **55.73** | **−6.48** |
| stop-cell return | 909.35 | 755.10 |
| mean Φ_v, 5 nonzero-speed cells | 0.0985 | 0.1981 |
| command gain (OLS slope) | 0.0000 | 0.1935 |
| crashes, 120 drive episodes | 0 | 7 |

Read the first row alone and the story writes itself: −6.48 against 55.73, and
at the 950k checkpoint **71 of 120** drive episodes ended in a crash. Falling
over is obviously the problem.

It is not. It is **0.9%** of the problem.

`HoundPDTrackDesert-v0` reports all four reward terms in its `info` dict, so the
return can be split rather than argued about. Decomposed on the final
checkpoint, residual 2.8e−14:

| term | policy | zero action | contribution to the gap | share |
|---|---|---|---|---|
| `reward_track` | 67.09 | 65.01 | **+2.09** | −3.4% |
| `reward_ctrl` | −65.59 | 0.00 | **−65.59** | **105.5%** |
| `reward_contact` | −7.39 | −9.28 | +1.89 | −3.0% |
| `reward_termination` | −0.58 | 0.00 | **−0.58** | **0.9%** |
| **return** | **−6.48** | **55.73** | **−62.20** | |

Control cost is more than the whole gap. The crashing that the first diagnosis
rested on is under one percent of it.

Per step, dividing by measured mean episode lengths of 952.2 and 1000.0:

    policy earns  67.09 / 952.2  = 0.07046 of tracking per step
    zero   earns  65.01 / 1000.0 = 0.06501
    gain from driving             = 0.00545 per step

    policy pays   65.59 / 952.2  = 0.06889 of control cost per step
    zero   pays    0.00           = 0.00000
    cost of driving               = 0.06889 per step

**After 1.5 million steps the machine earns 0.0055 of extra reward per step for
driving and pays 0.0689 to do it — about 12.6× what it earns.**

### Why tracking pays a driving machine no more than a standing one

The tracking term is a *product* of two scores, deliberately: Φ_v for matching
commanded speed, Φ_w for holding commanded heading.

| | zero action | policy | ratio |
|---|---|---|---|
| Φ_v (speed) | 0.240 | 0.350 | **×1.46** |
| Φ_w (heading) | 0.513 | 0.245 | **×0.48** |

The policy genuinely drives — Φ_v up 46%, command gain 0.1935 against zero
action's algebraic 0.0, achieved forward velocity moving monotonically with the
commanded value. It pays for that in heading: driving on this terrain makes it
yaw, and a standing machine holds heading almost perfectly for free. Because the
two multiply, the gain and the loss cancel, and the control cost decides the
sign.

**This is not the parked failure and it is not the broken failure.** The machine
drives, stays upright most of the time, and keeps 83% of zero action's stop-cell
score. It simply cannot earn enough to cover what earning it costs.

## What happened

The run trained 1,500,000 steps in 3h51m47s at ~108 steps/s and finished on its
own, 22 minutes inside its declared 4.25h ceiling.

Something worth recording happened to the checkpoints. Between 950k and 1.5M the
policy **traded command-following for uprightness**: crashes over 120 drive
episodes fell 71 → 7, while command gain fell 0.482 → 0.1935. Both directions
are real and they moved together.

The `ant_sac_best.zip` checkpoint was also overwritten in place mid-analysis,
which produced an accidental natural experiment. The newer checkpoint crashes
**1 time in 120** instead of 71 — and its drive-grid return is **−4.87**,
statistically indistinguishable from −4.19. Remove essentially all the falling
over and the number does not move. That is the cleanest single piece of evidence
that crashing was never the cause.

Both checkpoints were measured, as this project's own rule requires. The
selection delta on the drive-grid mean is **1.60** — against the 98.42 points
checkpoint selection moved on the previous arm. `*_best.zip` is chosen on
single-episode *mixture* draws, which barely correlate with a 20-episode
per-cell grid mean, so this instrument is structurally more robust to that bias
than the previous one was.

## Measurements

Everything above is computed, not asserted. Committed alongside:

- `research/measurements/hound_track_desert_s0_final_sac.json` — the unselected
  checkpoint, 20 episodes/cell, seeds 1000–1019
- `research/measurements/hound_track_desert_s0_final_best.json` — the selected one
- `research/measurements/hound_track_desert_s0_final_decomposition.json` — the term split
- `research/measurements/hound_track_desert_s0_midrun_950k.json` — the 950k
  reading, carrying a note that its checkpoint no longer exists
- `research/scripts/track_return_decomposition.py` — the decomposition
- `research/scripts/entropy_lesson_math.py` — the entropy figures

## How the prediction did

Ten claims were registered before any measurement — five inherited from episode
006 and five written this cycle. **Four true, six false.** Across all 21
resolved predictions the hit rate falls 73% → **57%** and the Brier score
worsens 0.1421 → **0.1650**.

| # | claim | p | outcome |
|---|---|---|---|
| 12 | drive-grid ≥ 5× zero action (279) | 0.50 | **false** — −6.48 |
| 13 | mean Φ_v ≥ 0.35 | 0.50 | **false** — 0.1981 |
| 14 | command gain ≥ 0.5 | 0.40 | **false** — 0.1935 |
| 15 | stop competence ≥ 700 retained | 0.75 | **true** — 755.10 |
| 16 | mid-run gate > 111.5 by 800k | 0.55 | **false** — −4.19 |
| C1 | both checkpoints agree in direction | 0.75 | **true** |
| C2 | selection delta ≥ 20 points | 0.60 | **false** — 1.60 |
| C3 | stop competence lost (< 700 both) | 0.65 | **false** — 755, 723 |
| C4 | `standing-control` hard-floor failure | 0.85 | **true** — ×0.43 |
| C5 | reaches 1.5M inside its ceiling | 0.90 | **true** |

**The most useful wrong prediction was C3.** The reasoning behind it was
arithmetic and it was checkable: the command mixture is 10% STOP, zero action
scores 909 there, so merely holding still would bank ~90 of the mixture — and
the policy was banking 29 in total, so it must have lost stop competence. It had
not. It kept 755 of 909. The error was assuming the deficit had to come from
somewhere *visible*; it came from a per-step cost spread evenly across every
step of every episode, which no single cell reveals.

**And the reliability table moved against this cycle's own reasoning.** The
40–60% band read *under-confident, 100% actual* on a single sample when this
cycle read it, and this cycle raised its confidences citing that. With five
samples it now reads **over-confident, 20% actual**. The correction was applied
in the right spirit off a sample far too small to carry it.

## Ranked actions

1. **Do not apply the pre-registered levers.** `p_stop → 0.05` and
   `min |v_cmd| → 0.4` both push more driving at a machine whose problem is that
   driving does not pay. The theory note has been corrected in place to say so.
2. **The live question is the ratio.** Either achievable Φ_v·Φ_w rises — and the
   limiting factor is *heading*, not speed — or the control cost falls far
   enough that 0.0055/step of gain is worth buying. That is a reward-design
   decision and belongs with the mathematics.
3. **Seeds.** One seed is a probe. Nothing here supports a claim about the
   reward design itself.
4. **A longer run is a real alternative explanation** and it is cheap to test:
   `ep_rew_mean` was 43.24 and still climbing at the step budget, up from 29 at
   1.02M. 1.5M did not converge.

## Open questions

- **Is the Φ_w collapse terrain-specific?** If the same policy holds heading on
  flat ground while driving, this is a contact problem, not a reward-balance one.
- **Does the deterministic rollout misrepresent what training optimised?** Every
  number here is from `deterministic=True`. If the stochastic policy pays
  materially less control cost, the 12.6× ratio is not the one SAC saw.
- **Why did command-following fall as uprightness rose?** 0.482 → 0.1935 while
  crashes went 71 → 7. Staying upright and following commands should not be in
  tension, and if they are, that is a statement about the machine rather than
  the reward.
