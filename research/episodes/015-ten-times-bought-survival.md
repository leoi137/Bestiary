# 015 — Ten times the training bought survival, not tracking

**Date:** 2026-08-07 · **Robot:** Spyder-12 on Isaac Lab, PPO ·
**Runs:** `spyder_ladder_s1` (three arms × 1500 iterations),
`spyder_overnight_s1` (15,000 iterations)

**One seed per arm, everywhere in this episode. Every number below is a probe,
not a finding** — the seed rule applies with no exceptions, and the ladder's
margins in particular are far too small for one seed to rank.

Every figure here is printed by
`research/scripts/018_ladder_overnight_reads.py`, which reads the four runs'
own TensorBoard event files, their launch-time config dumps and their console
logs. Nothing in this file was typed from a terminal.

## The two questions, and why they are one episode

Episode 014 deleted ten of the gentle task's eleven reward terms, kept
base-frame `v_x`, and got a machine that ran 147 ft (44.7 m) per 20 s episode
and could not be steered. Its third reading named the gap: *the span between
one term and eleven is unexplored; add one term at a time and watch which
single term buys the most survivable gait.*

That produced two runs, three hours apart, and they belong in one episode
because the second one **spends the first one's answer**. The ladder is the
instrument; the overnight run is what was built on its reading. Splitting them
would let the second look like an independent result, and it is not.

- **The ladder** (`spyder_ladder_env_cfg.py`) pays the full command-tracking
  income on every rung — both of the gentle task's kernels at the gentle
  weights — and adds **at most one penalty**:

      bare        income only                       the control
      actionrate  income + action_rate_l2  (-0.01)
      tilt        income + ang_vel_xy_l2   (-0.05)

  The correction to 014's sentence is deliberate: `v_x` is not a task, so a
  gait ranked on top of it would be un-steerable whatever it scored. Every
  rung here is drivable by construction.

- **The overnight run** (`spyder_overnight_env_cfg.py`) is the ladder's winner
  plus two gait-shaping terms — `feet_air_time` (+0.125) and `lin_vel_z_l2`
  (−2.0) — trained 10× longer. **Two terms were added at once, so this run
  cannot say which of them did anything**, and its own module docstring says so
  before any result existed. It is a production run, not an experiment.

Both open the lateral command range to ±0.89 mph (±0.4 m/s). Strafe is
commanded here for the first time in this project.

## The setup, verified from what launched

Read out of each run's own `params/env.yaml` and `params/agent.yaml`, not from
the source:

| run | seed | envs | episode | iterations | live reward terms |
|---|---|---|---|---|---|
| ladder/bare | 1 | 4096 | 20 s | 1500 | 2 |
| ladder/actionrate | 1 | 4096 | 20 s | 1500 | 3 |
| ladder/tilt | 1 | 4096 | 20 s | 1500 | 3 |
| overnight | 1 | 4096 | 20 s | 15000 | 5 |

Each ladder arm collected 147,456,000 samples in 42–46 min on a rented 5090 at
54–58 k steps/s; the overnight run collected 1,474,560,000 in 7 h 39 min at
53.7 k steps/s. The derived sample count (iterations × envs ×
`num_steps_per_env`) matches the console's own `Total steps` exactly in all
four runs, and the script raises rather than prints if it ever does not.

TensorBoard and the terminal are written by different code paths in rsl_rl, so
they are an independent transcription of the same run. They agree to 0.0016 or
better on every arm's final reward and to the last printed decimal on every
episode length.

## What the ladder measured

Final iteration (1499), with the mean of the last ten iterations beside it so
nothing here is a peak read against a mean:

| rung | final reward | last-10 | final ep length | last-10 | error_vel_xy | terrain level |
|---|---|---|---|---|---|---|
| bare | 17.4516 | 16.7701 | 866.43 | 818.85 | 0.2831 | 2.9958 |
| **actionrate** | **18.0086** | **17.3306** | **869.66** | **847.92** | **0.2317** | 2.9332 |
| tilt | 17.1316 | 16.6525 | 847.83 | 832.05 | 0.2656 | 2.8830 |

All three learn a driveable gait. The ordering is the same on the final
iteration and on the last-ten mean, which rules out one kind of artefact and
exactly one.

**It does not rule out the kind that matters.** The winner beats the control
by **0.557 of reward (3.2%) and 3.23 steps of episode length (0.4%)**. One seed
per arm cannot rank margins that size. Between-seed spread on this stack has
never been measured — not once, on any robot, in this repository — so the
honest statement is not "actionrate is 3.2% better" but "**this ladder did not
establish an ordering, and was never going to at one seed**". The overnight
run inherited `action_rate_l2` on that basis, which is the best evidence in
hand and is not the same thing as evidence.

There is one structural observation the seed problem does not fully erase, and
it is worth stating with its caveat attached. rsl_rl logs each reward term as
an episode sum normalised by the maximum episode length, so per-episode income
is the logged figure × 20 s:

| rung | income/s | income/episode | tax/episode | net |
|---|---|---|---|---|
| bare | 0.8583 | 17.17 | 0.00 | 17.17 |
| actionrate | 0.9853 | 19.71 | −1.77 | 17.93 |
| tilt | 0.8601 | 17.20 | −0.48 | 16.72 |

The income ceiling is (1.0 + 0.5) × 20 s = 30.0 per episode. `actionrate`
earns **14.8% more tracking income than the control while paying a tax the
control does not pay**, so its 3.2% margin on net reward *understates* its
margin on the thing the task is actually for. That is a coherent story rather
than a demonstrated effect: it is still one seed, and a single seed can
produce a coherent story by accident.

## What the overnight run measured

| iteration | mean reward | mean episode length |
|---|---|---|
| 1707 | 15.6238 | 875.89 |
| 4023 | 15.4594 | 836.79 |
| 7635 | 18.9397 | 959.27 |
| 11517 | 17.7448 | 920.88 |
| 14193 | 17.9555 | 935.32 |
| **14999 (final)** | **18.3910** | **934.48** |

Peak reward was 20.5450 at iteration 14883; peak episode length 999.05 at
8136. The final term decomposition, per episode: income 21.60 (72.0% of the
30.0 ceiling), `action_rate_l2` −1.78, `lin_vel_z_l2` −1.31, `feet_air_time`
−0.05. Terminations at the end are 92.2% timeout, 7.8% base contact.

**`feet_air_time` finished NEGATIVE.** It was added as the only term in the
table that reads foot contact timing, and the config's own docstring computed
before launch that at upstream's 0.5 s threshold it pays zero at 1 Hz per foot
and goes negative above it. It went negative. The term intended to ask for a
step to be a step ended the run as a small net tax on the cadence the policy
chose. Its magnitude is trivial (−0.05 of 18.39), so this changes nothing
about the run; it is recorded because a prediction made in a docstring before a
run is still a prediction, and this one held.

### The headline: what 10× the training actually bought

Comparing the winning rung at iteration 1499 to the overnight run at 14999.
The two pay **different reward tables**, so net reward is not like-for-like and
income is:

| metric | rung @1499 | overnight @14999 | change |
|---|---|---|---|
| mean reward (net) | 18.0086 | 18.3910 | +2.1% |
| income per second | 0.9853 | 1.0798 | +9.6% |
| mean episode length | 869.66 | 934.48 | +7.5% |
| base-contact termination share | 0.1793 | 0.0782 | **−56.4%** |
| error_vel_xy | 0.2317 | 0.2245 | −3.1% |
| error_vel_yaw | 0.2020 | 0.2034 | +0.7% (worse) |
| terrain level | 2.9332 | 2.9177 | −0.5% |

Ten times the compute, ten times the samples, and **linear-velocity tracking
error improved by 3.1% while yaw tracking got very slightly worse**. What
actually moved is falling: the machine falls **less than half as often**. The
overnight run first matched the rung's final reward at iteration 963 — before
the ladder's arms had finished — and spent 43.7% of its 15,000 iterations at
or above it.

That is the reading of this episode. The long run bought survival. It did not
buy obedience, and obedience is the capability the whole Spyder track exists
to reach.

### The curriculum never promoted, and nobody predicted that

Terrain level **started at 3.4758, and 3.4758 is its maximum for the entire
run**. It ended at 2.9177. Over 1.47 billion samples the population sat below
the level it was initialised at and never climbed back. The three ladder arms
finished in the same band (2.88–3.00) after a tenth of the training.

This was not predicted, is not explained, and is the single most interesting
number in the run. Whatever else 10× the compute was doing, it was not walking
the machine onto harder ground. The arc-corrected demote bar in
`curriculums.terrain_levels_vel_arc` was built specifically so that a good
turner would stop being demoted for tracking a curve; that fix is in force here
and the population still did not promote. Either the bar is still wrong, or the
gentle terrain's levels are not ordered by difficulty the way the curriculum
assumes, or the commanded distribution makes promotion unreachable. Nothing in
these artifacts distinguishes those.

## The six predictions, scored

All six were written into `research/calibration.jsonl` before the runs they
describe. Four resolve; **two cannot be resolved at all, and why is the most
transferable thing in this episode.**

| id | claim | p | outcome |
|---|---|---|---|
| P6 | all three rungs reach mean reward ≥ 12 by iteration 1500 | 0.75 | **TRUE** — weakest is tilt at 17.1316, +5.13 over the bar |
| P7 | actionrate's action-rate metric ≥30% below bare's | 0.70 | **VOID** — see below |
| P8 | bare is the wildest in playback, by base_ang_vel_xy rms | 0.60 | **NOT SCOREABLE** — see below |
| P9 | final mean reward ≥ 20 at iteration 15000 | 0.65 | **FALSE** — 18.3910, missing by 1.61 |
| P10 | final mean episode length ≥ 900 | 0.60 | **TRUE** — 934.48, +34.48 |
| P11 | final error_vel_xy ≤ 0.35 | 0.55 | **TRUE** — 0.2245, 0.126 of margin |

**P9 is the one to be careful with.** The run's peak reward, 20.5450 at
iteration 14883, clears 20. The claim says *final mean reward at iteration
15000*. Scoring it on the peak would be precisely the peak-versus-final
substitution this record exists to make impossible, so it is scored FALSE on
the final. The last-10 mean, 18.2796, misses too.

**P11 was priced badly in a way worth naming.** Its own note admits the
threshold was a guess because "no ladder-final baseline for this metric was
recorded". That baseline was on disk: the actionrate rung finished at 0.2317.
The bar was set 51% looser than a number three finished runs had already
measured, and a claim that should have been near-certain was stated at 0.55.
The failure was in the search, not the forecast — read the arms you have
already run before weighing the next claim.

### P7: a comparison whose instrument existed on only one arm

The claim compares `Episode_Reward/action_rate_l2` between the actionrate rung
and the bare rung. **The bare rung never logs that tag.** A reward term absent
from a task's table is never evaluated by the reward manager, so it is logged
by exactly the arm that pays it and by no other. bare's entire `Episode_Reward`
inventory is `track_lin_vel_xy_exp` and `track_ang_vel_z_exp`.

What exists is one half of a ratio: actionrate at −0.088603 per second at
iteration 1499, which at weight −0.01 is 8.8603 sum-of-squares per second, or
−1.7721 per episode, 8.99% of that arm's own income. There is nothing to divide
it by, and there never will be from these artifacts.

The prediction was therefore **unresolvable the moment it was written** — the
config that made it unresolvable was authored alongside it, and neither noticed
the other. The generalisation is cheap and applies to every reward ablation
this project will ever run:

> **Never pre-register a cross-arm comparison on a metric that only one arm
> instruments.** If a quantity has to be compared across arms that differ in
> whether they price it, it must be logged as a *metric* — computed for every
> arm regardless of the reward table — not read off a reward line.

That is a fix worth making in the config, not a lesson worth remembering:
`action_rate_l2` and `ang_vel_xy_l2` as zero-weight-logged observations, or an
explicit metrics block every rung carries. It is the cheapest thing that makes
this class of dead prediction impossible rather than merely noticed.

### P8: an instrument that was never built

P8 operationalised itself as "base_ang_vel_xy rms over a fixed 8 s
straight-drive script, computed by a committed measurement snippet". The
snippet was never written. No training log carries a roll/pitch-rate metric at
all — all three rungs log the same 20 scalars and the only angular ones are the
yaw-tracking reward and `error_vel_yaw`, neither of which is torso roll or
pitch rate. `research/measurements/` holds no ladder file.

What the ladder run directory holds is three drive videos of about 2.4 MB each.
A video supports an impression; the claim asks for an rms. Deciding P8 by eye
from footage would be worth less than leaving it open, so it is left open.

Unlike P7 this is **pending, not void**: all three `model_1499.pt` checkpoints
are on disk, so the measurement is still makeable the next time a GPU can
replay them. Both rows keep `"outcome": null` so neither can contaminate the
Brier score, with the reason recorded in the row's `resolved` field.

After these four resolutions the record stands at 42 resolved predictions,
55% hit rate, Brier 0.2332 (`venv/bin/python -m bestiary.record.calibration`).
Every band above 40% still reads over-confident.

## Strafe: commanded, and completely unmeasured

All four runs launched with ±0.89 mph (`lin_vel_y = (−0.4, 0.4)` m/s), verified
from each run's own config dump. That is the first lateral command in this
project's history, and the observation width did not move to get it — exactly
what `spyder_gentle_env_cfg.py` reserved the slot for.

**Nothing here says whether the machine can strafe.** The only linear tracking
metric is `error_vel_xy`, the norm of a 2-D error: a policy that tracks `v_x`
perfectly and simply never sidesteps, and one that splits its error evenly
between the axes, produce the same number. There is no per-axis metric in any
of the four runs' 20 logged scalars.

So the honest summary of the strafe change is: **a lateral command was
introduced into four runs and its effect was not measured once.** The commanded
box's corner is now √(0.6² + 0.4²) = 1.61 mph (0.721 m/s), 20% past a forward
ceiling of 1.34 mph (0.6 m/s) that was itself never verified — so a fifth of
the command distribution may be unreachable, and no reading distinguishes
"cannot strafe" from "will not strafe" from "was never asked in a way that
shows".

## Reading, provisional

1. **All three rungs learn to walk and be driven.** Income-only is enough; the
   nine deleted taxes are not load-bearing for locomotion emerging on this
   stack, which reproduces 014's conclusion under a real task rather than under
   `v_x`.
2. **The ladder did not rank its rungs.** 3.2% on reward and 0.4% on episode
   length at one seed per arm is not an ordering. `action_rate_l2` was carried
   forward as the winner of a single-seed probe, and if a multi-seed ladder
   reverses it, the overnight recipe is what has to change.
3. **10× the training bought survival, not tracking.** Falls more than halved;
   linear tracking error moved 3.1% and yaw error moved the wrong way. If the
   goal is a machine that obeys, this run bought the cheaper half of it.
4. **The terrain curriculum ran backwards from its starting level for 1.47
   billion samples** and no artifact explains why.
5. **Two of six predictions were unresolvable, both for instrument reasons.**
   That is a 33% waste rate on a cycle's forecasting, and neither failure was
   about being wrong — they were about measuring. The prediction discipline is
   working; the instrumentation discipline is not keeping up with it.

## What the next episode inherits

- **A metrics block that does not depend on the reward table.** P7's failure
  is a config bug with a prediction attached. Until every arm logs the
  quantities an ablation compares, reward-ablation predictions will keep
  arriving dead.
- **Per-axis velocity tracking**, or the strafe change is unfalsifiable and
  stays that way however many runs command it.
- **Why the curriculum demotes.** Start 3.4758, end 2.9177, over 15,000
  iterations, with the arc-corrected bar already in force.
- **P8's rms**, which needs one rollout script and a GPU, and would then score
  a prediction that is otherwise stranded.
- **Seeds 2 and 3 of the ladder**, if its ordering is ever to be cited. Right
  now it may not be.

No new calibration rows are registered by this episode. The next prediction
should be written against an instrument that is known to exist for every arm
it names — which is the whole content of P7's failure, and it would be a poor
joke to repeat it in the act of recording it.
