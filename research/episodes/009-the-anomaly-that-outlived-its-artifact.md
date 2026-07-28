# 009 — The anomaly that outlived its artifact

**Date:** 2026-07-28
**Run under measurement:** `hound_track_desert_s0` (already in the ledger, row 4)
**Arithmetic:** `research/scripts/track_length_bias_math.py`
**Measurements:** `research/measurements/track_length_bias_s0.json`,
`research/measurements/track_length_bias_s0_best.json`

---

## Thesis

`track_eval`'s `mean_track` is length-biased, and the record's own note about
that bias overstated it by a factor that no longer exists on disk. Both halves
matter, and the second one is the finding.

---

## Diagnosis going in

`drive_grid_track` is the number this record explicitly tells readers to trust
whenever `drive_grid_ratio` becomes unreadable — which is often, because
returns here can go negative and a ratio of two negative numbers is not a
performance ordering. It was the designated safe metric.

It carried an undocumented bias. `mean_track` is a mean **over episodes** of
each episode's **own per-step mean**, so an episode that ends at step 300
carries exactly the same weight as one that runs the full 1000. The reward is
not that quantity: it is a per-step integral over a fixed horizon. A policy
that crashes early therefore reads as competent as one that survives, while
banking a fraction of the income.

The module docstring asserted the opposite in as many words — "survives
cost-coefficient retuning **and episode-length changes**". `anomalies.jsonl`
row 20 caught the false claim on 2026-07-27 and recorded the damage:

| | policy | zero action | ratio |
|---|---|---|---|
| raw `mean_track` | 0.0955 | 0.0650 | **×1.47** |
| normalised to 1000 steps | 0.0644 | 0.0650 | **×0.992** |

Read straight, that reverses a published verdict: length-normalised, the policy
does not beat doing nothing on tracking *at all*.

---

## What was done

Three separately-named aggregates now exist, because the old single name was
answering three different questions at once:

| field | question | length-biased? |
|---|---|---|
| `mean_track` | while it is up, how well does it track? | yes, by design |
| `mean_track_stepw` | per step actually taken, how well? | no |
| `track_per_horizon` | how much did it **bank** per unit of horizon? | no |

`track_per_horizon` is the tracking analogue of the return. The horizon is read
from the env's own `max_episode_steps`, never assumed, and each cell records the
divisor it used so a reader can recover it from the committed JSON
(`mean_track_stepw × mean_steps / track_per_horizon`) rather than trust the code.

Then the run was re-measured under the identical protocol — n=20 per cell, seeds
1000–1019, deterministic, both checkpoints.

---

## What happened

**Zero action reproduced bit-for-bit: `0.06500765521917337`.** That is the
control that matters. It proves the env, the grid, the seeds and the protocol
did not move between 27 and 28 July — which had to be established first,
because an env change did land in between (the backward command floor moved
0.3 → 0.4 on 2026-07-27). That change touches only `_resample_command`, and the
eval protocol pins the command and pushes the resample past the horizon, so the
sampler is never called. Zero action's unchanged number is the proof, not the
argument.

**The policy did not reproduce.**

| | row 20 | today, `ant_sac.zip` | today, `ant_sac_best.zip` |
|---|---|---|---|
| raw `mean_track` | 0.0955 | 0.0718 | 0.0695 |
| `track_per_horizon` | 0.0644 | 0.0671 | 0.0678 |
| raw ratio | ×1.47 | **×1.104** | **×1.069** |
| normalised ratio | ×0.992 | **×1.032** | **×1.043** |

The reversal is gone. Length-normalised, the policy still *marginally beats*
zero action on tracking — replicated at ×1.027 on an independent seed block
(seeds 5000–5009).

**Where row 20's number came from.** It is
`research/measurements/hound_track_desert_s0_midrun_950k.json`: `ant_sac_best.zip`
at 950k steps, **71 crashes**, committed 10:58:13 on 27 July. That checkpoint
file's mtime is **11:07:01.732** — the training run's own best-eval saving
overwrote it nine minutes after the measurement was committed.

So ×1.47 was a property of one vanished checkpoint's crash rate, not of the
metric.

**The bias is one cell, not a diffuse effect.**

| checkpoint | bias | carried by | non-crashing cells |
|---|---|---|---|
| `ant_sac.zip` | ×1.0694 (+6.9%) | (0.5, 0, 0.4) — 7/20 crashes, 713 steps | **exactly 1.000000** |
| `ant_sac_best.zip` | ×1.0242 (+2.4%) | (0.8, 0, 0) — 1/20 crashes, 959.5 steps | **exactly 1.000000** |

Over the five cells where every episode runs the full horizon the bias is
exactly 1, because there is no bias to have. Which cell carries it *differs by
checkpoint*, so naming one of them as "the" crashing cell is wrong.

Note the shape of that: **the bias magnitude is a crash-rate proxy.** It was
~×1.48 at 71 crashes and is ×1.07 at 7 and ×1.02 at 1. It is not tempting-but-
false to call it `1/(fraction of horizon lived)` — that identity does not hold
here. The crashing cell's bias is 0.05087/0.02292 = 2.219 while 1000/713 =
1.4025, because the episodes that crash also track at a different rate from the
ones that survive.

---

## How the prediction did

Committed before measuring, in `research/calibration.jsonl`.

| p | claim | outcome |
|---|---|---|
| 0.85 | zero action's two aggregates agree within 0.0005 | **TRUE** — delta exactly 0.000000 |
| 0.70 | policy's `track_per_horizon` in [0.060, 0.069] | **TRUE** — 0.067094 |
| 0.70 | raw ratio > 1.4 **and** normalised < 1.05 | **FALSE** |

The false one is the finding. The normalised half held (1.032 < 1.05); the raw
half did not, and could not, because the artifact that produced 1.47 was gone
before the prediction was written. **The diagnosis this missed** is that a
measurement can fail to reproduce for reasons that have nothing to do with the
measurement — and that the record gave no signal at all, because the JSON still
names `ant_sac_best.zip` and still looks fully sourced.

The second prediction is scored TRUE with a caveat that costs it most of its
value: row 20's 0.0644 is itself unverifiable. That JSON predates the
`mean_steps`, `track_income` and `reward_track` fields, so 0.0644 was
hand-derived and cannot be recomputed from anything committed.

---

## What the refutation changed

An independent review was given the claim and told to kill it. It could not,
but it corrected four things and found two real defects that this analysis had
missed:

1. **The crashing cell was mis-named** for `ant_sac_best.zip` — (0.8, 0, 0),
   not (0.5, 0, 0.4). The per-checkpoint table above is the corrected version.
2. **"1.0688/1.0435" was a ratio of ratios**, valid only because zero action's
   own bias is exactly 1.0. It is now quoted as a bias directly, ×1.0242.
3. **`track_income` over-counted.** It recomputed `phi_v*phi_w`, but the env
   pays `phi_v*phi_w if healthy else 0.0` — so the terminal unhealthy step of a
   crashing episode was scored when the env refused to pay it. On exactly the
   episodes the whole correction is about. It now reads the env's own
   `reward_track`, making it the reward integral exactly rather than to four
   significant figures. Measured error was 8.3e-4 absolute: immaterial to every
   conclusion here, and wrong.
4. **The horizon lookup silently degraded.** It fell back to the episode's own
   length when no `TimeLimit` was found, which turns `track_per_horizon` into
   `mean_track_stepw` with every guard green and no trace in the output. It now
   raises.

It also validated `track_per_horizon` in a way this analysis had not thought to:
`reward_track`, accumulated from the env's `info` by a completely separate code
path, equals `track_per_horizon × 1000` to 1e-14 on every non-crashing cell.

---

## What is now enforced

New guard **`track-length-bias`**, fast tier, 7 assertions, gating every launch.
It runs on synthetic episodes whose right answers hold by construction, and it
calls the real `aggregate_cell` rather than a copy of the formula — a guard that
reimplements what it checks bounds nothing.

Verified in both directions: restoring the old conflation reddens 3 of the 4
arithmetic assertions (the equal-length case correctly still passes, because a
length correction with nothing to correct is the identity); restoring the false
docstring sentence reddens the fifth on its own; restoring the silent horizon
fallback reddens the seventh, as does a function that always raises.

**What is not enforced:** the artifact half. Nothing yet stops a mutable
checkpoint from underwriting a published number. That fix — copy to a
content-addressed path before measuring, record the sha256 in the JSON — is an
open anomaly dated 2026-07-28, not built. Cycle 007 had already recorded
`*_best.zip is overwritten in place` as known-broken and had already observed
that hashing before and after is insufficient by construction. The gap was
never that nobody knew.

---

## What this changes

- **`drive_grid_track` is still usable, and now says which question it answers.**
  Read `track_per_horizon` against a return, `mean_track` against competence.
- **The tracking verdict on `hound_track_desert_s0` stands as row 4 recorded it.**
  Length-normalising does not rescue it and does not damn it further: the policy
  wins tracking by 3% and loses the drive grid by 62 return points, which is
  learning 011's finding — the machine drives and cannot afford to — unchanged.
- **A measurement is only as durable as its artifact**, and this record has now
  paid for that twice.
