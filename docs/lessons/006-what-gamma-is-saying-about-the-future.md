# 006 — What γ = 0.99 is really saying about the future

**One sentence:** γ = 0.99 is not a dial for *how much we care about the
future* — it is a **horizon of 100 steps**, and our episodes are 1000 steps
long, so the robot cannot see the end of its own episode.

## The idea

The robot never maximises this step's reward. It maximises a discounted sum:
this step, plus 0.99 × next step, plus 0.99² × the one after, and so on. **γ**
(gamma) is that per-step shrink factor. Discounting keeps the sum finite and
makes learning stable — and the side effect is the whole lesson.

At γ = 0.99 a reward **100** steps away is worth **0.366** of the same reward
now. A reward **1000** steps away is worth **4.32e−05** — nothing. Our episodes
are exactly 1000 steps (50 s at 20 Hz).

So the hound is optimising roughly **the next five seconds**. *"This will pay
off later in the episode"* is not an argument it can act on, because later,
arithmetically, is not there. That is why per-step economics decide behaviour:
what an action earns **this** step against what it costs **this** step.

γ picks a timescale and does nothing else:

| γ | horizon 1/(1−γ) | at 20 Hz |
|---|---|---|
| 0.95 | 20 steps | 1.0 s |
| **0.99** | **100 steps** | **5.0 s** |
| 0.999 | 1000 steps | 50 s |

We never chose 0.99. `train/train.py` builds SAC without naming γ, so it is
Stable-Baselines3's default.

## The math

A stream of `c` per step, forever, discounted, is a geometric series:

    Σ(t=0→∞) γᵗ·c  =  c / (1 − γ)  =  c × horizon

**c** the per-step rate, in reward units per step; **γ** unitless; **1/(1−γ)**
= 100 steps. Truncating at the episode's real 1000 steps gives 9.999568 instead
of 10.0 — the infinite sum is honest here because the horizon is 10× shorter
than the episode.

**A one-time number is a per-step rate times the horizon.** That is the entire
content of the formula, and this repo used it to derive a constant:

    src/bestiary/envs/hound_track.py:127
    TERMINATION_PENALTY = 10.0   # = c/(1-gamma) = 0.10/0.01 at gamma=0.99

Ten is not a round number somebody liked. An early, flailing policy pays motor
and contact costs and earns almost no tracking, so its net rate is about
**−0.10/step**; dying escapes that stream, and escaping is worth 0.10/0.01 =
**10**. The penalty cancels it exactly, so early SAC is *indifferent* to falling
over rather than attracted to it.

## Where it bites here

**c = 0.10 was an estimate.** Cycle 007 measured the rates — 20 episodes on each
of six drive commands, `research/measurements/hound_track_desert_s0_final_decomposition.json`:

| per step, drive grid | trained | does nothing |
|---|---|---|
| tracking reward | +0.07046 | +0.06501 |
| control cost | −0.06889 | 0 |
| contact cost | −0.00776 | −0.00928 |
| **net** | **−0.00619** | **+0.05573** |

The contact cost the derivation assumed — 0.045/step "for supporting the
robot's weight" — measures **0.00928/step**, 4.85× smaller. And the converged
policy's net rate is −0.00619/step, not −0.10. Same formula, measured rate:
K = 0.00619/0.01 = **0.62**. The shipped penalty is **16.2×** the stream it was
derived to price.

Plainly: **a constant derived from an assumed rate is only as good as the rate,
and 1/(1−γ) multiplies the error by 100.**

Read the sum the other way and it stings again. A policy doing *literally
nothing* collects **92.3%** of the tracking reward the trained one collects, so
a whole discounted life of trying is worth **7.05** against **6.50** for
standing there — a gap of 0.55, one eighteenth of the death penalty.

No damage this time: terminations were rare (7 crashes in 120 drive-grid
episodes) and `research/learnings/011` measured the termination term at 0.9% of
the gap to zero action. The same shape of error sits in another constant,
though — the tolerance σ_v = 0.15 was derived from a *standing* machine's noise
and then used to score a moving one. Different number, same mistake; that is
its own lesson.

Nothing here argues for changing γ. It argues that everything derived *from* γ
inherits whatever you assumed about the machine.

Arithmetic: `scripts/006_discounting_math.py`.

## If you want to go deeper

[`../theory/command-tracking-reward.md`](../theory/command-tracking-reward.md)
§4 — the full termination-penalty derivation, including why K is deliberately
not larger (over-penalising terminal proximity breeds timidity at the
uprightness margin, which on rough terrain is where the work is).

*Assumes [001 — what a reward function is](001-what-a-reward-function-is.md).*
