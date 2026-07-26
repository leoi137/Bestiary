# 003 — PD made the plateau 5× cheaper, not higher

**Date:** 2026-07-25 · **Robot:** hound · **Run:** `hound_pd_desert_v0` (finished)

Scores the prediction made in `002-pd-position-targets.md`, which was written
and committed before the run started.

## How the prediction did

| claim | called | actual | verdict |
|---|---|---|---|
| `ep_rew_mean` clears 1096 by 1M | 55% | **1099.63** | **hit, by 3.6 points** |
| most likely 1000–1400, `ep_len` near 1000 | — | 1099.6 / 930.8 | **hit** |
| parks at standing (950–1000) | 25% | no | correctly not taken |
| exceeds 2000 (a real gait) | <10% | no | correctly not taken |
| `ent_coef` < 0.02 by 60k | — | 0.0114 at 61k | **hit** |

The headline call was right, and by a margin thin enough (0.3%) that calling
it a clean win would be overselling it. The distribution was well calibrated:
the outcome landed in the band called most likely, and neither tail fired.

One detail the prediction got wrong in character rather than in number.
`ent_coef` was described as "collapsing". It dipped to 0.0114 at 61k, then
**rose** to 0.0343 by 350k before annealing back to 0.0181. SAC's automatic
temperature bought exploration *back* once the policy stopped dying instantly.
The threshold was met; the dynamic was not what the word "collapse" implied.

## The actual result

Against `hound_desert_v0` (torque, 3.75M steps):

| | torque | PD | |
|---|---|---|---|
| steps | 3,750,000 | 1,000,000 | |
| wall clock | 8 h 05 m | **2 h 35 m** | |
| **first eval ≥ 1100** | **1,502,322** | **300,809** | **5× fewer samples** |
| peak eval | **1218.3** | 1176.7 | torque slightly higher |
| mean eval after 400k | 887.5 | **1113.1** | PD far more stable |
| final `ep_rew_mean` | 1009.4 | **1099.6** | |
| final `ep_len_mean` | 794.2 | **930.8** | |

**PD position targets did not raise the ceiling. They made reaching it five
times cheaper, and made staying there reliable.**

The stability difference is the one that does not show up in a peak-score
comparison and matters more than it looks. The torque run's evals after 400k
average **887.5** because it kept falling over — individual evals of 390, 522,
955 scattered through 3.75M steps. Its *best* number, 1218.3, is a number it
visited, not a number it held. PD's evals after 400k average **1113.1** and
sit consistently near 1170, with two crashes in nineteen.

A policy that reliably scores 1170 is worth more than one that occasionally
scores 1218 and often scores 500 — and the ledger's `best_eval_return` field
alone would have hidden that entirely.

## Was the diagnosis in 001 right?

Yes, partially, and the falsifier did not fire.

Episode 002 pre-registered: *"plateauing at ~1010 again from a materially
different action space would say the control parameterization was never the
binding constraint."* That did not happen — PD ended higher (1099.6 vs
1009.4), survived longer (930.8 vs 794.2), and got there 5× faster.

But episode 001's stronger implication — that pose-holding was what kept this
robot from walking — is **not** supported. Both control schemes plateau in the
same 1150–1220 band. Removing the pose-holding burden did not produce a
qualitatively different machine. It produced the same machine, sooner and more
reliably.

The honest summary: **PD was a real win on cost and reliability, and not a win
on capability.**

## What the machine is actually doing

Still not a gait. `eval/episode_length` is a clean 1000 from 400k onward, so it
survives; the reward is dominated by the alive bonus (1.0/step × ~930 = 930 of
the 1099.6). Forward motion contributes roughly 170 over a 46-second episode.

`eval/mean_idle_legs` reads 0.00 for the whole run — and for the torque run
too. **That metric is dead**: it is populated from `info["shaping/idle_legs"]`,
which only exists when a shaping wrapper is active, and neither run used one.
It has been silently logging zeros in every hound run to date and tells us
nothing about whether the legs are being used. Fixing or removing it is a
prerequisite for answering the question this episode most wants answered.

## The run did not converge

`ep_rew_mean` finished at 1099.63, which is also its **maximum** — it was still
climbing when the step budget ran out. The last five logged values were
1077, 1090, 1090, 1100. This run was cut short, not finished.

That was the right call for a first dry run under a 3-hour ceiling, but it
means the comparison above understates PD: torque got 3.75M steps to find its
peak and PD got 1M.

## Ranked actions for 004

1. **Fix or delete `eval/mean_idle_legs`.** We cannot answer "is it walking or
   rolling?" without it, and that is the central question for a wheel-legged
   robot on terrain. Cheap, and it unblocks everything else.
2. **Re-run PD to 2M steps.** It had not converged. This is the cheapest test
   of whether the ceiling is real or just where the budget ran out — and at
   108 fps it is ~5 hours, so it needs the throughput work or an overnight.
3. **Examine `ctrl_cost_weight` under PD.** Still 0.01, still penalizing
   `||action||²` where the action is now an offset from stance rather than a
   torque. It is a crouch-and-freeze prior wearing the name of an effort
   penalty, and it is a candidate explanation for why the legs stay still.
4. **Throughput remains untouched and remains the real limit.** Decision 0001
   stands; nothing here challenges it.

## Open questions inherited by 004

- Why is the open-loop survivable wheel band non-monotonic (0.3 survives,
  0.2 and 0.5 crash)? Still unexplained, still unexamined.
- The command spec is still undecided; the 3 reserved observation slots are
  still zeros. Two episodes have now passed without closing that door.
- Does the PD policy use its twelve leg joints at all? Blocked on item 1.
