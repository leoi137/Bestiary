# 016 — Wider dials, same brain: a fine-tune instead of a retrain

**Single seed, and a fine-tune besides — nothing here is a finding, and the
run is not even a clean arm:** it inherits every gradient of the overnight
run it resumed from. What it is: the cheapest possible answer to "can the
overnight policy learn a faster envelope without forgetting how to walk."

## The change, exactly one section

`Bestiary-Fast-Spyder-v0` subclasses the overnight task and moves three
numbers: command ranges to ±1.5 m/s forward (was ±0.6), ±0.6 m/s lateral
(was ±0.4), ±1.5 rad/s yaw (was ±0.8). Reward table, kernel widths,
terminations, observations, terrain — provably untouched (oracle check
`fast-task-widens-only-commands`, which pins all six range numbers rather
than importing them, so editing the constant cannot stay self-consistently
green). The kernel σ/v_max ratio therefore falls 0.5 → 0.2: tracking at
speed is graded *relatively harder*, deliberately, because restoring the
ratio would reward keeping the old speed.

Resume required new plumbing: upstream's checkpoint resolver never leaves
the experiment directory, so a fine-tune that reads `spyder_overnight/` and
writes `spyder_fast/` is impossible with stock flags. `train_desert
--from-checkpoint` patches the resolver behind a sentinel and exits nonzero
if the resolver is never consulted — a fine-tune that silently trained from
random weights is the failure mode it exists to make loud.

## What happened (numbers from `runs/spyder_fast_s1/box_console.log`)

6000 additional iterations, 14999 → 20998, 3.0 h, from the overnight
checkpoint (load line verified in the console log before iteration 0).

| iteration | mean reward | mean ep length |
|---|---|---|
| 16760 | 11.86 | 890.19 |
| 18116 | 12.70 | 886.52 |
| 19796 | 12.74 | 900.74 |
| 20804 | 13.04 | 935.14 |
| 20998 (final) | 13.08 | 933.22 |

The shape is the story: reward *fell* from the overnight's 18.39 to ~12 at
the start — the same kernel now grades commands up to 2.2× larger — then
recovered steadily while survival dipped only 934 → 887 and climbed back to
933. That is adaptation without forgetting, which was the whole bet.

Final block, and what each number means against the overnight run:

    mean episode length    933.22   (overnight 934.48 — survival unchanged)
    error_vel_xy           0.4829   (overnight 0.2245 — but the command
                                     corner is 1.616 m/s vs 0.721: the
                                     RELATIVE error is ~0.30 both times)
    error_vel_yaw          0.3883   (overnight 0.2034; yaw range ×1.88)
    base_contact share     0.1209   (overnight 0.0782 — falls up ~half,
                                     at 2.5× the speed)
    terrain_levels         3.4719   (overnight 2.9177 — see below)

Playback (deterministic, `runs/spyder_fast_s1/film_console.log`): commanded
+1.50 m/s, achieved +1.44…+1.52 across five seconds, including over a hill
crest. The envelope is real, not a config artifact.

## Predictions, scored

- **P12 TRUE** — episode length 933.22 ≥ 850.
- **P13 FALSE, by 0.033** — error_vel_xy 0.4829 vs the 0.45 bar. The miss
  is instructive: the bar was set on absolute error without first dividing
  by the command scale; the relative ratio came out *unchanged* (~0.30).
  Set relative bars for relative quantities.
- **P14 FALSE at p 0.7, sign wrong** — terrain levels ROSE (2.92 → 3.47)
  despite the demote bar hardening 6 m → 15 m. The half of the arithmetic
  the prediction ignored: faster commands displace farther, so promotions
  outpaced demotions. A directional claim derived from one bar of a
  two-bar mechanism. Related open question from episode 015 — the
  overnight run's curriculum never promoted at all — now has a contrast
  case: the fast run promoted plenty. Both facts await one explanation.

## Still unmeasured, carried forward honestly

Strafe quality (a lateral-axis metric exists in no run's logs — episode
015's gap, still open); the roll/pitch-rate rms instrument P8 waits on; and
whether 0.12 base-contact at 2.5× speed is the fine-tune's floor or just
where 6000 iterations stopped. All three need either a GPU replay script or
another run, and none blocks driving the machine.
