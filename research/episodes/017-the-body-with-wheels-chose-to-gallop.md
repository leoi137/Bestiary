# 017 — The body with wheels chose to gallop

**Date:** 2026-08-08 · **Robots:** Spyder-12, Hound-16 · **Runs:**
`spyder_forward_v5_s1`, `hound_forward_v5_s1` (two run directories: the scratch
run and the fine-tune that resumed it)

**Single seed everywhere, and two of the three runs are fine-tunes besides —
these are probes, not findings.** A fine-tune is not even a clean arm: it
inherits every gradient of the run it resumed, so nothing below may be quoted
as an effect until it has been re-run with at least three seeds. Every number
here is printed by `research/scripts/019_forward_v5_paired_reads.py`, which
reads the event files, the launch-time config dumps and
`runs/hound_forward_v5_s1/box_console.log` and refuses to run if any of them is
missing — with **two exceptions that are observed telemetry rather than
computed**: the playback speeds and the demo-strip positions. Those come off
the player's terminal, only the video was kept, and both are flagged again
where they appear.

## The question

Episode 014 deleted the Spyder's whole eleven-term reward table, paid one term
— base-frame `v_x` at weight 1.0 — and got a walker. That answered a question
about a *stack*. It left a sharper one about a *body*.

The Hound has driven hub wheels where the Spyder has feet, so `r = v_x` on the
Hound is genuinely ambiguous in a way it never was on the Spyder. There are two
ways to earn metres and the reward cannot tell them apart:

* **Roll.** Spin the four wheels. The velocity drive saturates at 3.0 N·m /
  0.28128 N·m per rad/s = 10.666 rad/s, which on an 0.08496 m rim is
  **2.0 mph (0.906 m/s)**. Over a 20 s episode, pure rolling earns at most
  **59 ft (18.12 m)**.
* **Gallop.** Use the twelve leg joints and bound, with the wheels along for
  the ride.

Every reward coefficient that expressed a preference between those — leg-only
torque and acceleration taxes, a 100× weaker wheel acceleration penalty, a
`lin_vel_z_l2` term at −2.0 that charges for exactly the vertical bouncing a
gallop is made of — is gone with the table that carried it. **That ambiguity is
the experiment**, and it was pre-registered as P16: above 18.1 m per episode
requires legs.

Both robots ran the same one-term reward on the same v5 ground
(`decisions/0007`), sequentially, 4096 envs, seed 1, 147,456,000 samples each —
37 min for the Spyder fine-tune, 65 min for the Hound from scratch.

## The Spyder half: the gain is survival, not the headline

The Spyder run was a **fine-tune** of the v4 forward probe's `model_1499`, 1500
further iterations (1499 → 2998) on the new ground. Two variables moved at once
— the terrain *and* the extra training — so this table describes what came out
and attributes nothing to either.

| metric | v4 @1499 | v5 @2998 | change |
|---|---|---|---|
| mean reward, m/episode | 44.7176 | 44.0358 | **−1.5%** |
| mean episode length, of 1000 | 572.52 | **629.10** | **+9.9%** |
| `forward_velocity` per second | 1.9033 | 2.0922 | +9.9% |
| `time_out` share | 0.5081 | 0.5503 | +8.3% |
| `base_contact` share | 0.4921 | **0.4497** | **−8.6%** |

The headline number did not move. Everything about *staying up* did: an extra
57 steps of survival per episode, and about four episodes in every hundred
moved out of the crash column into the timeout column. Played back
deterministically at seed 1000 the checkpoint holds **12.3–14.5 mph
(5.5–6.5 m/s)** against the parent's **9.4–12.1 mph (4.2–5.4 m/s)** — observed
telemetry, not an instrument, and see the speed section below for why that
distinction turns out to matter.

**The peak-versus-final discipline, third time on this record — and this time
it cuts the other way.** The first two were episode 015: its ladder table
printed the last-10 mean beside every final so nothing was a peak read against
a mean, and P9 was scored FALSE on a final of 18.39 while the run's peak of
20.55 cleared the bar. Here the same discipline *costs* the Spyder run
something. Its final iteration, 44.04, sits **below** its own trailing means:

| window | Spyder v5 reward |
|---|---|
| final iteration (2998) | 44.0358 |
| last-10 mean | 52.4571 |
| last-100 mean | 48.6210 |
| best 100-iteration window | 52.3566 (at iteration 2636) |
| single best iteration | 79.9815 (at iteration 2859) |

A single rsl_rl iteration averages over whatever episodes happened to end
inside it, so the per-iteration series is very noisy — the single-iteration
peak is 1.82× the final, which is a statement about sampling, not about the
policy. The run was living at roughly 48–52 and its last draw was a low one.
P15 is still scored on 44.04, because 44.04 is what P15 named. The rule is not
"report the smaller number"; it is "report the number the claim named, and
print the others next to it."

## The Hound half: the fork resolved, and not narrowly

From scratch, 1500 iterations, 65 min.

    final, iteration 1499     203.5588 m/episode      606.28 steps of 1000

**668 ft (203.56 m) per episode against a pure-rolling ceiling of 59 ft
(18.12 m): ×11.25.** The machine was above the rolling ceiling by iteration
**111**, and 92.3% of all 1500 iterations sit above it. Played back
deterministically it holds **17.9–22.4 mph (8–10 m/s)** — observed telemetry
again, and roughly ten times the wheel drive's 2.0 mph (0.906 m/s) saturation.
It gallops. Nothing in the reward asked it to.

What that rules out is exactly one thing, and it is worth being precise about
which: **pure rolling cannot account for the metres.** The drive physically
stops commanding speed at 0.906 m/s; past that the wheel is a torque source at
the traction limit, so 18.12 m is a soft wall rather than a hard one — but not
a soft wall with a factor of eleven behind it. Legs are doing the propelling.
It does **not** rule out the wheels contributing; nothing here instruments
wheel torque or contact, and anomaly 61 already records that no reward term
prices wheel effort at all.

The reading that follows, and it is a reading rather than a measurement: every
wheel-preferring coefficient in the desert reward table was weighting a mode
the unshaped objective never wanted. `decisions/0004` Part B put a deliberate
thumb on the scale for rolling. Given a body that can do both and an objective
that says only "go forward", the optimum picked the other one.

## The fine-tune, and the number that flatters it

A further 800 iterations from that checkpoint (1499 → 2298, 32.7 min of
wall clock read off the event file — this run has **no console log at all**, so
its numbers have one transcription and no cross-check, which is stated here
rather than hidden).

| metric | scratch @1499 | fine-tune @2298 | change |
|---|---|---|---|
| mean reward, m/episode | 203.5588 | 297.5872 | **+46.2%** |
| mean episode length, of 1000 | 606.28 | **815.97** | +34.6% |
| `base_contact` share | 0.4097 | 0.3696 | −9.8% |

That +46.2% is a final-versus-final comparison and it is the most flattering
one available. The fine-tune's final iteration sits *above* its own trailing
mean while the scratch run's sits *below* its own, so the two errors compound
in the same direction:

| window | scratch | fine-tune | change |
|---|---|---|---|
| final iteration only | 203.5588 | 297.5872 | +46.2% |
| last-10 mean | 223.7357 | 255.9269 | +14.4% |
| last-100 mean | 223.4418 | 236.4954 | **+5.8%** |

**The honest sentence is +5.8%, not +46%.** The survival gain — 606 → 816
steps, 16.3 s of a 20 s episode — is the part that does not depend on which
window you pick, and it is the same shape as the Spyder half: the second pass
buys steadiness.

## The number that does not add up

`r = v_x` × the control period means an episode's return **is** metres of
forward travel, so mean return over mean episode duration is a time-weighted
mean forward speed:

| run | m/ep | sec | implied | playback |
|---|---|---|---|---|
| Spyder v4 | 44.72 | 11.45 | 8.7 mph (3.905 m/s) | 9.4–12.1 mph |
| Spyder v5 | 44.04 | 12.58 | 7.8 mph (3.500 m/s) | 12.3–14.5 mph |
| Hound scratch | 203.56 | 12.13 | **37.6 mph (16.788 m/s)** | **17.9–22.4 mph** |
| Hound fine-tune | 297.59 | 16.32 | 40.8 mph (18.235 m/s) | not measured |

The Spyder rows are consistent and in the expected direction: a deterministic
single-robot rollout with no action noise, no pushes and no early falls is
*faster* than the population's time-weighted mean. **The Hound row is the wrong
way round** — on the *same* checkpoint, the training log implies roughly double
what the playback showed. Three candidates, none tested: base-frame `v_x` is a projection onto a
nose axis that swings on a pitching gallop (but `|v_x^b| ≤ |v|` holds
instantaneously, so that direction alone cannot produce an *excess*); the
4096-env population is heterogeneous in a way one seeded rollout is not, and
time-weighting *up*-weights an episode that tumbles down a v5 slope earning
metres fast before it dies; or the playback figure is eyeballed telemetry
rather than an instrument. Filed as anomaly 63, unexplained. Until it is
closed, any Hound speed quoted anywhere has to say which of the two numbers it
is.

## The demo strip: 84%, then the steep face

Deterministic playback on the one-piece demo strip, seed 1000 — **256 ft (78 m)
end to end**, running x = −39 to +39 m with the spawn pinned at x = −34, flat at
the low end and hardest at the summit.

| checkpoint | reached | travelled | % of strip | left to summit |
|---|---|---|---|---|
| `model_1499.pt` | x = +20.0 | 177 ft (54.0 m) | 75.6% | 62 ft (19.0 m) |
| `model_2298.pt` | x = +26.4 | 198 ft (60.4 m) | **83.8%** | 41 ft (12.6 m) |

The second attempt stalled on the steep face for about two and a half seconds
before falling. Repeat attempts wander off the *side* of the strip rather than
failing at the same place, which is not a surprise and should not be reported
as one: `spyder_demo_env_cfg`'s docstring stated the caveat before any of this
was filmed — the forward-only diagnostic does not read commands at all, so
pinning its spawn aims it up the ramp but nothing keeps it there, and the strip
was made as wide as it is long for exactly this reason. A caveat written in
advance and then observed is the cheapest kind of prediction there is. **A
command-following checkpoint is what will run the strip end to end**, and no
amount of further fine-tuning of a command-deaf policy will do it.

## Anomaly 62 is resolved, by the experiment it named

`anomalies.jsonl` row 62 recorded that the offscreen `--video` render camera is
frozen at boot on this install: four separate camera-move mechanisms — the
viewer's `origin_type='asset_root'`, `sim.set_camera_view()` per frame, writing
every stage `UsdGeom.Camera`'s USD transform, and mirroring the same pose into
Fabric via USDRT — are all silent no-ops on the recorded frames while the run
continues normally. It also named the cheapest experiment that would settle it:
author a *fresh* camera prim post-boot, bind our *own* replicator render
product to it, and see whether that camera's transform tracks.

It tracks. A post-boot-authored `UsdGeom.Camera` with its own render product
follows the robot per frame, and the freeze is therefore a property of **Kit's
boot cameras** — the ones the render pipeline was born with — and not of USD
writes, of Fabric, or of the offscreen path in general. The fix was the ~20
lines the anomaly predicted it would be if the hypothesis held; it is the
chase-cam block in `play_spyder.py`, active under `--follow --video`, and it is
what made a following shot of the strip attempts possible at all.

Two things this buys beyond the clips. The shipped workaround it replaces was a
fixed camera parked beside the policy's path, which only worked because
seed-pinned playback repeats the same trajectory — a stochastic or driven
policy could not be filmed at all. And a two-day-old anomaly was closed by
running its own `cheapest_next_step` verbatim, which is the argument for
writing that field carefully in the first place.

Row 62 is updated in place to `status: "resolved"` with a `resolution` field,
following rows 5, 7, 8, 12–15 and 20. `anomalies.jsonl` is append-only for
rows; `status` is the one field the file's own contract says may change.

## Predictions, scored

| | claim | p | outcome |
|---|---|---|---|
| **P15** | Spyder v5 final mean reward ≥ 35 at iteration 1500 | 0.70 | **TRUE** — 44.0358, +9.04 of margin |
| **P16** | Hound final mean reward > 18.1 at iteration 1500 | 0.60 | **TRUE** — 203.5588, ×11.25 |
| **P17** | Hound mean reward ≥ 10 by iteration 500 | 0.70 | **TRUE** — 108.3448 at iteration 500; the bar was cleared at iteration **76** |

P17 deserves its margin stated rather than a bare TRUE. It was written to test
whether the "verdict visible by ~iteration 500" heuristic — which held on both
Spyder forward probes — transfers to a heavier, higher-DoF body. It transfers
with room to spare: the Hound crossed 10 m per episode 424 iterations early and
was at ×10.8 the bar by the deadline. **Future Hound probes can be budgeted on
the same early-verdict rule as Spyder probes**, which was the whole point of
asking.

All three were scored on the reading the claim named, with peaks and trailing
means printed beside them. Three TRUE at stated probabilities of 0.70, 0.60 and
0.70 is a set of bars that were too easy, not a set of predictions that were
good; P16 in particular was stated at 0.60 for an outcome that was visible by
iteration 111.

## Open questions the next episode inherits

1. **The speed discrepancy (anomaly 63).** Two instruments disagree by ~2× on
   the same robot and the same checkpoint. Cheapest resolution: log world-frame
   ground track alongside base-frame `v_x` on one deterministic rollout.
2. **What the wheels actually do.** "Legs propel it" is established; "the
   wheels are passive" is not, and nothing measures wheel torque. Anomaly 61's
   proposed instrument — per-wheel applied torque and per-leg contact normal
   force on a rollout — answers this one too.
3. **Is +5.8% real?** One seed, one fine-tune, and a trailing-window gain of
   that size is inside what between-seed variance has produced before on this
   record.
4. **Heading.** Every result above is from a policy that cannot be steered.
   The strip, and everything past it, needs the command interface back.
5. **Seeds.** All of it. Three seeds per arm, one variable, or none of this is
   an effect.
