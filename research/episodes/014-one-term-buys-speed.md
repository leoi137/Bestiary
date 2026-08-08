# 014 — One term buys speed; the other ten buy grace

**Single seed. Everything below is a probe, not a finding** — the seed rule
applies to this episode with no exceptions, and nothing here may be cited as
an effect until two more seeds exist.

## The question

The Spyder's Isaac Lab task pays eleven reward terms: two tracking kernels as
income and nine shaping terms as taxes. The proven MuJoCo/SAC spyder walked
off five, and its ancestor — the 2016-era gym locomotion stack — off
essentially one: forward speed. So which part of the eleven is load-bearing
for *locomotion emerging at all*, and which part is polish? The cheapest
possible test: delete everything except forward speed and see whether the
machine still learns to walk.

## The setup, one variable

`Bestiary-Forward-Spyder-v0` subclasses the gentle task and replaces the
whole reward table with a single term:

    r = v_x            (base-frame forward velocity, weight 1.0)

Per step that pays `v_x · dt`, so an episode's return reads directly as
metres travelled forward. Terminations (base contact, timeout) survive —
they are resets, not rewards. Everything else is byte-identical to the
gentle task and provably so: a config diff shows `rewards` as the only
differing section, and `check_spyder`'s `forward-variant-is-reward-only`
check (mutation-tested, six sabotages, six catches) pins it that way.

Training: seed 1, 4096 envs, 1500 iterations, 41 min on a rented 5090,
oracle 25/25 on the box before launch. Artifacts under
`runs/spyder_forward_s1/` (checkpoints, tensorboard, `box_train_console.log`
— every number below is grep-able from that log).

## What happened

| iteration | mean reward (m/episode) | mean episode length (steps of 1000) |
|---|---|---|
| ~30 | 0.02 | collapsing |
| 157 | 18.03 | 450.47 |
| 514 | 28.89 | **395.29** |
| 1053 | 39.93 | 521.78 |
| 1417 | 34.72 | 589.10 |
| 1499 | 44.72 | 572.52 |

Locomotion emerged fast and unambiguously: 59 ft (18 m) per episode within
157 iterations. The interesting structure is the iteration-514 row: reward
rose while episode length *fell* — a reckless-sprint phase where dying young
at high speed out-earned living carefully. It resolved on its own; by the
end, 47% of episodes ran the full 20 s (termination mix at iteration 920:
time_out 0.4683, base_contact 0.5319) while the terrain curriculum promoted
the population to level ~6.

Played back deterministically (`play_spyder`, seed 1000), the checkpoint
holds 9.4–12.1 mph (4.2–5.4 m/s) — ~13 body-lengths per second — as a
bounding, airborne, visibly violent gait, covers ~98 ft (30 m) in 8 s, and
then sprints off the southern edge of the play terrain and free-falls: an
edge is a thing it never once saw in training. It is deaf to commands by
construction; W changes nothing. Video: `runs/spyder_forward_s1/forward_sprint.mp4`.

## Reading, provisional

1. **Pure forward-speed reward produces strong forward locomotion on this
   Isaac/PPO stack.** The eleven-term table is not what makes walking
   possible; the stack reproduces the 2016 result. Whatever ailed earlier
   ports, it was never "PPO needs shaping to locomote."
2. **The shaping terms' observable job is the sprint-versus-survive trade**,
   visible as the iteration-514 dip and as the gait's violence: nothing
   priced flailing, slamming, or airtime, so the optimum bought speed with
   all three. Speed and controllability were traded exactly as the reward
   table said they would be — the gentle policy is slow, steerable and
   gentle; this one is fast, deaf and reckless. Both checkpoints exist,
   which makes the pair a useful bracketing exhibit for every future reward
   discussion.
3. **What this does not show:** that eleven terms are *needed* for grace.
   The gap between 1 and 11 is unexplored; the cheapest next experiment is
   a ladder — add one term at a time to `r = v_x` and watch which single
   term buys the most survivable gait.

## Deviations, stated

- No calibration rows were written before launch. Two informal in-session
  predictions were made ("verdict visible by ~iteration 500" — held, 18 m
  per episode by 157; "should reproduce the SAC result" — held
  directionally) but neither carries a stated probability, so neither
  enters `calibration.jsonl`. A same-day diagnostic is not an excuse the
  record accepts twice.
- One seed, and seeds 2–3 are not planned: this arm exists to answer a
  yes/no about the stack, not to measure an effect size. If any number
  from this episode is ever needed as evidence, it must be re-run ≥3×.
