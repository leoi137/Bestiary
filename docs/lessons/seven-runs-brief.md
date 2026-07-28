# Seven runs, plainly

**A briefing, not a lesson — a dated snapshot of the first seven runs.**
Written 2026-07-28, and deliberately not updated as runs eight and onward
land. It is a photograph of what the ledger said that day, kept because the
story it tells only reads as one story in order; the current numbers always
live in `research/ledger.jsonl`, never here.

Every number here is pulled from `research/ledger.jsonl`, the run `config.json`
files, `research/retired_runs.jsonl`, or learnings 001–011. Nothing is
reconstructed from memory. Where the record does not contain a number, it says
so instead of guessing.

Ordered oldest → newest, because it reads as one story that way.

---

## The one-paragraph version

Two robots, seven runs, nine weeks. The first five runs were all the same bug
wearing different clothes: **a reward function that paid the robot less to move
than it cost to move**, so the best available strategy was to stand still. It
took two robots and two training setups to prove the bug was in the reward and
not in either body. The last two runs were a different kind of failure and a
more expensive one — **the measuring instrument was wrong**. We compared two
policies through a checkpoint we had already proved was cherry-picked, and we
diagnosed a failure by looking at the most visible symptom instead of splitting
the number into its parts. Both times, careful work produced a confident wrong
answer.

---

## 1 · `spyder_walk_v3` — the one that worked

**2026-07-20 · `Spyder-v0` (flat ground) · SAC · 3.75M steps · seed 0**

### What changed
Nothing clever. A 12-legged spider robot, flat ground, the standard reward:
get paid for forward speed, pay a small cost for effort.

### What came out
Best evaluation: **7392.09**. It walked. This is the only unambiguous success
in the seven, and it matters mostly as the control arm for everything after it.

### What we learned
That the setup works when the ground is flat. That sounds trivial and it is
not — it means every later failure is about *terrain*, not about the algorithm,
the robot, or the code.

### The math
On flat ground, per step:

```
forward reward   +7.05
effort cost      −0.81
ratio             8.7 : 1
```

The robot earns nearly nine times what moving costs it. At that ratio the
effort cost is basically noise, and you can ignore it. **Remember this ratio —
the next six runs are all about what happens when it flips.**

---

## 2 · `spyder_desert_v0` — the reward inverted

**2026-07-24 · `SpyderDesert-v0` · warm-started from run 1 · ~5.75M total steps**

### What changed
Same spider, same reward, rough desert ground instead of flat. Rather than
starting over, we loaded the trained flat-ground policy and continued from it
with a fresh replay buffer.

### What came out
Best evaluation **832.36**, typical **509**. Then we ran the comparison that
should have been run first — roll the policy that does *literally nothing*,
motors at zero:

| policy | reward | speed |
|---|---|---|
| `spyder_desert_v0` | 509 | 0.37 m/s |
| **doing nothing** | **987** | 0 m/s |

**Doing nothing won, by nearly 2×.**

### What we learned
Two separate lessons, and this run is the only one that produced two.

**(a) A reward tuned on flat ground breaks on terrain** (`learnings/001`). The
effort cost did not rise. *The payoff collapsed.*

**(b) You cannot warm-start a critic across a reward change** (`learnings/002`).
SAC has two parts: an *actor* that picks actions and a *critic* that predicts
how much total reward a situation is worth. The actor survived. The critic did
not — it had learned "this state is worth ~7000" and reality was now ~500. Every
prediction was wrong by a factor of fourteen, the error compounded, and it
dragged the actor down with it. `ep_rew_mean` fell 6331 → 146 within a few
thousand steps and never recovered over 2M more.

There was a third problem stacked on top: the inherited policy had **no
exploration left**. SAC's entropy coefficient α is what pays a policy to stay
a bit random. A fresh run starts near α ≈ 1.0. This one entered the desert
already annealed at 0.136, fell 17× to 0.015 within 120k steps, and flatlined
at 0.008. Even if the reward had pointed the right way, there was no
exploration pressure left to follow it.

### The math
Same reward function, both grounds, per step:

|  | flat | desert |
|---|---|---|
| forward reward | 7.05 | **0.29** |
| effort cost | −0.81 | −0.57 |
| **ratio** | **8.7 : 1** | **0.51 : 1** |

Speed fell **24×**. Effort only fell **1.4×** (creeping slowly means smaller
actions, so the cost actually went *down*). Over a full episode: **+294 earned
for moving, −571 paid in effort — moving was a net loss of −293**, against a
1000-point bonus just for staying alive.

A cost weight calibrated against 7 m/s payoffs is ruinous at 0.3 m/s. The
reward was telling the spider to stand still. It half-listened, which is why it
looked like slow progress instead of obvious failure.

---

## 3 · `hound_desert_test150k` — proving it was the reward

**2026-07-25 · `HoundDesert-v0` · 150k steps · from scratch · seed 0**

### What changed
Everything except the terrain and the reward. New robot — Hound, a 16-DoF
wheel-legged dog, completely different body. **From scratch**, no warm start, no
inherited critic, no annealed entropy. Deliberately short.

### What came out
Best evaluation **109.49**, dying at 21–141 steps into a 1000-step episode.
Zero action: **961**, surviving all 1000.

**Doing nothing scored 9× better.**

### What we learned
This is the run that made the previous one interpretable. Spyder's failure had
*two* candidate causes — a bad reward, and a wrecked critic from warm-starting
— and there was no way to tell how much each contributed.

Hound had none of the warm-start problems and **failed the same way**. So the
reward pathology is *sufficient on its own*. The warm-start only made Spyder's
version worse.

Two robots, two morphologies, two training setups, one bug — and it is in the
reward, not in either body.

This run has since been retired from active use (`retired_runs.jsonl`) because
the observation vector later widened from 141 to 169 numbers, which makes its
checkpoints unloadable. It is kept anyway, because it is the evidence behind
the whole plan.

### The math
Best episode, forward vs effort: **+88.7 vs −67.2**, a ratio of **1.3 : 1**
against flat ground's 8.7 : 1. Same collapse as Spyder, different body.

**Second finding, and it matters later:** with all motors at zero the Hound
*drifts backwards* **−1.5 m per episode** on this terrain. Not a policy
behaviour — the physics does it. Note this; run 7 depends on it.

### The rule that came out of it
> **Run the standing check on every new robot and every new terrain, in the
> first 30k steps.** Roll a do-nothing policy. If nothing beats something, the
> reward is wrong. It takes two minutes and it has now caught the same bug
> twice.

---

## 4 · `hound_desert_v0` — the long torque run

**2026-07-25 · `HoundDesert-v0` · 3.75M steps · seed 0 · 8h05m**

### What changed
The fixed reward, and a full-length run — 3.75M steps, the same budget that
worked for Spyder on flat ground.

### What came out
Best evaluation **1218.32**. Training reward settled at **~1010**. Verdict:
**plateau**.

The standing check brackets it precisely: **961** standing still, **1043**
moving at speed. Settling at 1010 means the policy landed *between the two* —
it mostly holds pose and creeps. Better than standing, nowhere near driving.

### What we learned
That the reward was no longer actively inverted, but the run had plateaued
short. And — only visible later, by comparison with run 5 — that this policy
was **unreliable**: it kept falling over for the entire run, still producing
evaluations of 390 at 3.5M steps.

### The math
This is the number that later got us in trouble, so it is worth stating
carefully. `best_eval_return` is a **maximum over a noisy sequence**, and a
maximum only ever goes up. Its expected value is roughly:

```
E[Mₙ] ≈ μ + σ · aₙ        where aₙ grows like √(2 ln n)
```

- `μ` = the policy's true mean return
- `σ` = its spread (how unreliable it is)
- `n` = how many times you evaluated it

**Both terms push the peak up.** A policy can raise its peak by getting
*better* (μ↑) **or by getting less reliable** (σ↑). The statistic cannot tell
those apart — and it also grows with `n`, so a longer run scores higher for
free.

For this run: peak **1218.3**, mean after 400k **887.5**. The gap of **330.8**
is a direct read-out of instability.

---

## 5 · `hound_pd_desert_v0` — PD control, and the peak that lied

**2026-07-25 · `HoundPDDesert-v0` · 1.0M steps · seed 0 · 2h35m**

### What changed
How the policy commands its legs. Before: the network outputs raw **torque**
— "push this hard." Now: it outputs a **target angle** and a small built-in
controller (a PD controller) generates the torque to get there. The network
says *where*, not *how hard*.

### What came out
Verdict: **improved** — the only `improved` in the ledger.

| | torque (3.75M) | PD (1.0M) |
|---|---|---|
| peak eval | **1218.3** | 1176.7 |
| **mean eval after 400k** | 887.5 | **1113.1** |
| steps to reach eval ≥1100 | 1,502,322 | **300,809** |
| crashes (eval < 600) | frequent, throughout | 2 of 19 |

**5× fewer samples to reach the same band**, and far steadier once there.

### What we learned
Two things, and the second is the important one.

**(a) PD lowered the cost of reaching the plateau without raising the ceiling.**
Peak actually went *down* slightly (1176.7 vs 1218.3). The action space was the
bottleneck for *learning speed*, not for *how good the policy can get*.

**(b) The number we had been reporting as "the result" ranked these two runs
backwards** (`learnings/007`). Read peaks alone and torque wins. Read means and
PD wins by 225.6. A policy that reliably scores 1170 is worth more than one
that occasionally hits 1218 and often scores 500.

There was also a software lesson here (`learnings/006`): a big refactor passed
38/38 robot assertions, bit-identical physics hashes, and a `--help` smoke test
— then training died instantly on an `UnboundLocalError`. The test suite was a
*robot* oracle being quietly treated as a *repository* oracle. `--help` is worse
than no test, because `argparse` exits before the broken line and reports
success on the way past it.

### The math
The two peaks were not even drawn under the same conditions:

```
torque:  μ = 887.5    peak = 1218.3    peak − μ = 330.8    n = 14 evals
PD:      μ = 1113.1   peak = 1176.7    peak − μ =  63.6    n = 19 evals
```

Torque's peak sits **330.8** above its own mean; PD's sits **63.6**. That gap
is σ, read directly — torque's spread is about **five times wider**. Ranking
flips depending on which statistic you pick:

```
by peak:  torque 1218.3  >  PD 1176.7     (torque by 41.6)
by mean:  PD     1113.1  >  torque 887.5  (PD by 225.6)
```

**The torque policy is not better. It is noisier, and the statistic we were
using pays for noise.**

---

## 6 · `hound_pd_desert_s1` — the second seed, and the instrument breaks

**2026-07-27 · `HoundPDDesert-v0` · 997,379 steps · seed 1 · 2h35m**

### What changed
**Only the random seed.** Same environment, same steps, same hyperparameters as
run 5. The first time any arm in this project had more than one seed — which is
the minimum for calling anything a result rather than an anecdote.

### What came out
Verdict: **plateau**. And, initially, an exciting number: the two seeds
appeared to differ by **91.55 return points**, with seed 1 becoming the first
Hound policy to clear the project's ×1.18 advisory margin, at ×1.1937.

That was wrong.

### What we learned
**We compared two seeds through a checkpoint we had already proved was
cherry-picked** (`learnings/010`, extending `learnings/008`).

The measurement itself was flawless: one instrument, one protocol, 60
deterministic episodes, the *same* 60 seeds on both arms, a live zero-action
control, repeated to confirm the tool was deterministic — it returned
byte-identical results. **Every methodological rule the project had written
down was followed.** What nobody re-examined was *the object being compared*.

Both readings came from `ant_sac_best.zip`. That file is saved by `argmax` over
**single-episode** evaluations — so on a policy that either completes an episode
or falls over early, it is reliably a snapshot caught in its good mode. It is
the luckiest episode, not the better policy.

Re-measured on the *unselected* final checkpoint, same instrument, same 60
seeds:

| checkpoint | seed 0 | seed 1 | spread | clears ×1.18? |
|---|---|---|---|---|
| `ant_sac_best.zip` | 1049.10 | 1140.65 | **+91.55** | seed 1 only |
| `ant_sac.zip` (final) | 1089.05 | 1082.18 | **−6.87** | **neither** |

The sign flips, the magnitude collapses, the ×1.18 clearance vanishes.
**Checkpoint selection alone moved the thing we were calling "the seed spread"
by 98.42 points — more than the spread itself.**

Why it slipped through: `learnings/008` was filed under *how to read one
policy's score*. This was a *comparison* question, which felt like a different
topic. The trigger list didn't name it, the guard enforcing 008 checked sample
size but not which checkpoint, and the eval tool defaulted to `_best.zip`
silently. Every mechanism that existed to carry the lesson forward was pointed
at the wrong moment.

### The math
These policies are **bimodal** — they either finish the 1000-step episode
(~1170 points) or crash early (~284). So the mean is not really a performance
score. It is a crash counter:

```
mean = ( (n − c)·g + c·b ) / n
```

with `n` = 60 episodes, `c` = crashes, `g` = 1170.20 (mean of the 58 good
episodes), `b` = 283.54 (mean of the 2 crashes). Differentiate to find what one
crash is worth:

```
d(mean)/dc = (b − g) / n = (283.54 − 1170.20) / 60 = −14.78 per crash
```

**One extra crash in 60 episodes moves the reported mean by 14.78 points.** So
the celebrated 91.55-point seed spread is:

```
91.55 / 14.78 = 6.2 crashes' worth
```

And the actual crash difference under `_best.zip` was exactly **6** (8 vs 2).
Under the final checkpoint it was **1, in the other direction** (6 vs 7).

**The spread is not partly explained by the crash difference. It *is* the crash
difference.** We were not measuring how well two policies walk. We were
measuring how often each fell over — on a checkpoint chosen for not having
fallen over.

---

## 7 · `hound_track_desert_s0` — driving cost 12.6× what it paid

**2026-07-27 · `HoundPDTrackDesert-v0` · 1,499,378 steps · seed 0 · 3h51m**

### What changed
A completely new reward: **command tracking**. Instead of "go forward, get
paid," the robot is given a commanded velocity and heading each episode and
paid for *matching the command* — including a command of zero, where the right
answer is to hold still. This was designed specifically to close the
stand-still hole that had bitten twice (runs 2 and 3).

### What came out
A pre-registered gate — a pass/fail bar written down *before* the run, so the
result couldn't be rationalised afterwards — and it **failed, badly**. Drive-grid
score **−6.48** against a bar of 111.5, and against zero action's own **55.73**.
Verdict: **inconclusive**.

The obvious diagnosis: 71 of 120 drive episodes ended in a crash. It's falling
over.

**That diagnosis was wrong, and an independent refutation killed it.**

### What we learned
`learnings/011`. The reward is a *sum of four terms* and we read it as one
number, then reached for the most visible correlate of a bad number instead of
splitting it. The environment already reported all four terms; nobody looked.

Decomposed on the final checkpoint, 20 episodes per cell (residual 2.8e−14 —
the split is exact, not approximate):

| term | policy | zero action | contributes to the gap | share |
|---|---|---|---|---|
| tracking | 67.09 | 65.01 | +2.09 | −3.4% |
| **control cost** | −65.59 | 0.00 | **−65.59** | **105.5%** |
| contact | −7.39 | −9.28 | +1.89 | −3.0% |
| **crashing** | −0.58 | 0.00 | **−0.58** | **0.9%** |
| **total** | **−6.48** | **55.73** | **−62.20** | |

**The crashing that the entire diagnosis rested on is 0.9% of the gap. Control
cost is 105.5% — more than all of it.**

Then a natural experiment arrived by accident: a checkpoint was overwritten
mid-review. The replacement crashes **1 time in 120** instead of 71 — and
scores **−4.87**, statistically identical. *Remove essentially all the falling
over and the number does not move.*

The policy is not parked and it is not broken. **It drives, and driving is not
worth what it costs.**

### The math
The tracking reward is a **product**, deliberately (`docs/lessons/003`) — you
must match speed *and* heading, and doing one without the other earns nothing:

```
r = Φ_v · Φ_w − w_ctrl·Σaᵢ² − w_contact·Σ|f| − K·1[terminated]
```

where Φ_v is the speed-match score and Φ_w the heading-match score, both
between 0 and 1.

|  | zero action | policy | ratio |
|---|---|---|---|
| Φ_v (speed match) | 0.240 | 0.350 | **×1.46** |
| Φ_w (heading hold) | 0.513 | 0.245 | **×0.48** |

A machine standing still holds its heading perfectly and gets near-full credit
on Φ_w **for free**. A machine driving on rough desert yaws and loses about
half of it. Because the terms multiply, the ×1.46 gained in speed is cancelled
by the ×0.48 lost in heading. **Net tracking earned by driving: essentially the
same as doing nothing.** And then it pays a control cost that doing nothing
does not pay.

Per step, from the measured totals:

```
tracking gained by driving   0.07046 − 0.06501 = 0.00545 per step
control cost of driving      0.06889 − 0.00000 = 0.06889 per step

ratio = 0.06889 / 0.00545 = 12.6
```

**After 1.5 million steps, the machine earns 0.0055 reward per step for driving
and pays 0.0689 per step to do it. It pays about 12.6× what it earns.**

This is the *same arithmetic shape* as run 2, where the hound paid 0.57 to earn
0.29. **That one was caught because the machine visibly stood still. This one
hid because the machine visibly moved.**

### The trap that was avoided
The pre-registered response to a failed gate was "push it to drive more"
(`p_stop → 0.05`, `min|v_cmd| → 0.4`). Those levers assume the policy is parked.
This one isn't — it's driving at a loss, and both levers would have made the
return *more* negative. The pre-registration did its job by forcing honesty
about the failure; but **a pre-registered response is only as good as the
failure mode it assumed**, and this run showed one the design never enumerated.

---

## What actually carries forward

**1 · Always run the do-nothing control, on the same environment, first.**
Two minutes. Caught the same bug twice (runs 2 and 3) and would have saved
about 2M wasted steps each time. Never compare rewards *across* environments —
flat scored 7392 and desert 832, but that gap is mostly "7 m/s × 1000 steps,"
not a ranking.

**2 · Never trust a maximum.** `best_eval_return` and every `*_best.zip` is an
argmax over single noisy episodes. It rises with instability and with run
length, both for free. It is fine for *selecting* a checkpoint and useless for
*comparing* two runs. Those are different jobs and they need different
statistics.

**3 · Decompose the return before naming a failure.** Any reward that is a sum
of terms can be reported term by term against the control arm, with a residual
check. It cost two minutes in run 7 and reversed the conclusion completely.

**4 · The actor survives a reward change; the critic does not.** If you must
warm-start across a reward change: keep the actor, **reset the critic**,
relabel the buffer, lower the learning rate.

**5 · Know what your tests actually cover.** A green suite means "the thing the
suite tests is fine," never "the change is fine." The gap between those is
where an overnight run dies having reported success on the way in.

**6 · The recurring failure is not the robot. It is the measurement.** Runs 2
and 3 were a bad reward. Runs 6 and 7 were a *correct measurement of the wrong
object*. Both times the work was careful, the protocol was followed, and the
answer was confidently wrong. That is the failure mode worth watching for —
because it does not feel like a mistake while you are making it.

---

## Where the real versions live

| topic | file |
|---|---|
| all seven, one row each | `research/ledger.jsonl` (rows 1–4 only; 5–7 predate it) |
| the retired two | `research/retired_runs.jsonl` |
| flat reward breaks on terrain | `research/learnings/001` |
| no warm-start across a reward change | `research/learnings/002` |
| standing check on a second robot | `research/learnings/005` |
| the test oracle gap | `research/learnings/006` |
| peaks hide unreliable policies | `research/learnings/007` |
| best checkpoint = luckiest episode | `research/learnings/008` |
| the creep story was never measured | `research/learnings/009` |
| we compared two lucky draws | `research/learnings/010` |
| crashing was 0.9% of the gap | `research/learnings/011` |
| why rewards multiply | `docs/lessons/003-add-or-multiply.md` |
| why one seed is not a result | `docs/lessons/002-why-one-seed-is-not-a-result.md` |

**One thing this brief does not cover:** `learnings/009` came from a
measurement script rather than a training run, so it has no run of its own
above. Short version — the Hound's backward drift on desert (run 3's −1.5 m per
episode) was blamed on the terrain's cell size being close to the wheel radius,
and a fix was proposed and repeated in four places in the repo. Measured:
halving the cell size closes **0.7%** of the gap (+12 mm of the +1776 mm
needed). The drift is real; the explanation for it was never measured, and the
proposed fix does nothing. **The creep is currently unexplained.**
