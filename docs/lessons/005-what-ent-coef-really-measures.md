# 005 — What `ent_coef` really measures

**One sentence:** `ent_coef` is not how random the robot is — it is the *price*
SAC currently pays for randomness, and it moves in the **opposite** direction to
the randomness itself.

## The idea

SAC does not just maximise reward. It maximises reward *plus* a bonus for
keeping the policy unpredictable, because a policy that commits too early stops
discovering things. The size of that bonus is a number called α (the code logs
it as `ent_coef`).

You do not choose α. SAC tunes it automatically against a fixed goal called the
**target entropy**, written H̄. Entropy here means one thing: how spread out the
robot's action choices are. High entropy = it might do many different things
next. Low entropy = it always does nearly the same thing.

The controller is a thermostat, and this is the whole lesson:

- policy **more** random than the target → randomness is cheap → **α falls**
- policy **less** random than the target → randomness is scarce → **α rises**

So reading "`ent_coef` collapsed to 0.0002, the policy has gone deterministic"
is backwards. α collapsing means the policy was *more* random than asked, for
long enough that SAC stopped paying for it.

## The math

SAC adjusts α by descending this loss (Haarnoja et al. 2018, §5):

    J(α) = E[ −α·log π(a|s) − α·H̄ ]

Differentiate with respect to α. Since entropy is H = E[−log π]:

    ∂J/∂α = H − H̄        so      α ← α − λ·(H − H̄)

Every symbol: **α** the entropy price, unitless; **H** the policy's current
entropy in *nats*; **H̄** the target entropy in nats; **λ** the learning rate.
The sign is the lesson — α is driven by the *gap* `H − H̄`, not by H.

The default target is `H̄ = −dim(A)`, one negative nat per actuator. The hound
has 16 actuators (4 wheels + 12 leg joints), so **H̄ = −16 nats**. What does
−16 nats look like? For a Gaussian with the same spread σ on each actuator:

    H = dim · ½·ln(2πe·σ²)   ⇒   σ = √( e^(2H/dim) / (2πe) )

Worked at H = −16, dim = 16:  σ = √( e^(−2) / (2πe) ) = **0.089016**

Check: 16 × ½ × ln(2πe × 0.089016²) = −16.0000 nats. ✓

**Physically:** SAC is aiming for a robot whose motor commands wobble by about
**±0.089** on a [−1, 1] scale — roughly 4.5% jitter per actuator. Above that
wobble α gets cut; below it α gets raised.

Now the two hound runs, same robot, same 16 actuators, same target — only the
reward differs:

| steps | forward-velocity reward | command-tracking reward |
|---|---|---|
| 50k | 1.14e−2 | 4.60e−4 |
| 100k | 1.53e−2 | 2.16e−4 |
| **minimum** | 1.03e−2 @ 56k | **1.76e−4 @ 125k** |
| 400k | 2.74e−2 | 1.64e−3 |
| 1.0M | 1.84e−2 (final) | 2.32e−3 |
| 1.14M | — | 2.63e−3 |

The tracking run sits **~100× lower** (0.0184 / 1.76e−4 = 104.5×) — then climbs
**15×** off its own floor. Both halves are the thermostat working: early on the
tracking reward is near zero and almost flat, so the actor has no strong reason
to sharpen, entropy stays above target, and α is cut. As the policy finally
commits to a gait, entropy falls through the target and α is pushed back up.

*(The "why" in that last sentence is the plausible reading, not a measurement —
policy entropy itself is not logged. The direction of the α update is not a
reading; it is the algorithm.)*

## Where it bites here

`research/anomalies.jsonl` (2026-07-27) flags the tracking run's collapse as
*"the entropy constraint has stopped applying pressure."* Half right: the
**bonus** vanished, but the **constraint** was being satisfied with room to
spare — and it proved it by raising α 15× the moment that stopped. The row was
correctly filed as *not yet evidence of anything*, and it still isn't. Watch the
gap `H − H̄`, never α alone.

## If you want to go deeper

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905),
Haarnoja et al. 2018 — §5 derives the temperature update above.

*Assumes [001 — what a reward function is](001-what-a-reward-function-is.md).*
