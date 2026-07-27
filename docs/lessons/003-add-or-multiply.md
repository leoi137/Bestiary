# 003 — Why two rewards should be multiplied, not added

**One sentence:** If you pay a robot for doing two things and you *add* the two
payments, it will find the easier one and collect that half forever — but if
you *multiply* them, neither half is worth anything without the other.

Assumes [001 — What a reward function is](001-what-a-reward-function-is.md).

## The idea

Our hound is meant to drive at a commanded speed *and* turn at a commanded
rate. So the reward has two jobs: match the speed, match the turn. The obvious
thing is to score each job from 0 to 1 and add them together, with weights that
say how much each matters.

The problem is what "add" means to something searching for the highest number.
Addition makes the two jobs **independent offers**. A robot that stands
perfectly still is, technically, turning at exactly the rate you asked for
whenever you asked it not to turn — so it collects the entire turning payment
for doing nothing at all, and never has to attempt the hard half.

This is not hypothetical. It is the same shape as the bug that produced
[lesson 001](001-what-a-reward-function-is.md): a term that pays out no matter
what the robot does becomes the whole objective. Adding a second good-looking
term next to a free one does not fix a free term. It just hides it.

Multiplication removes the offer. A product is only large when **every** factor
is large — score zero on either job and the whole step pays zero, no matter how
perfect the other half was.

## The math

Score each job with a kernel that is 1 when perfect and falls off as the error
grows:

    Φ(u) = 1 / (1 + u²)

where `u` is the error divided by a tolerance — dimensionless. For speed,
`u_v = ‖v − v_cmd‖ / σ_v` with `σ_v = 0.15 m/s`; for turning,
`u_ω = |ω − ω_cmd| / σ_ω` with `σ_ω = 0.10 rad/s`.

Now take a robot standing still, commanded to drive forward at 0.5 m/s and not
to turn. Its speed error is the full 0.5 m/s; its turn error is ≈ 0.

    u_v = 0.5 / 0.15   = 3.33   →   Φ_v = 1/(1 + 11.11)  = 0.083
    u_ω = 0.0182 / 0.10 = 0.18  →   Φ_ω = 1/(1 + 0.033)  = 0.968

The 0.0182 rad/s is not a guess: it is this robot's measured yaw noise standing
on this terrain (`research/measurements/tracking_noise.json`). A standing
machine is not perfectly still, but it is nearly perfectly *not turning*.

**Added**, with equal weights of 0.5 each:

    r = 0.5(0.083) + 0.5(0.968) = 0.525   ← better than half marks, for standing still

**Multiplied:**

    r = 0.083 × 0.968 = 0.080   ← about 8% of a step, and it must move to earn more

Physically: added, the robot gets paid 0.525 per step for succeeding at the one
job it was already doing by accident — more than half the maximum, for nothing.
Multiplied, its excellent turning is worth almost nothing because it is being
multiplied by a terrible speed score. The only way to raise a product is to fix
its *worst* factor, which is exactly the behaviour we wanted.

## Where it bites here

`src/bestiary/envs/hound.py` currently adds four terms, one of which is a flat
`+1.0` for staying upright — 1000 free points per episode, which is why standing
still scored 955 against a trained policy's 1078. The replacement in
[`../theory/command-tracking-reward.md`](../theory/command-tracking-reward.md)
makes the tracking term the *only* positive one and multiplies its two channels
together, so there is no term left for a non-moving policy to collect. That
design is derived but **not yet validated** — no run has trained under it.

On the numbers above, the additive form pays **6.6×** more than the product for
standing still. Arithmetic: `scripts/003_add_or_multiply.py`.

## If you want to go deeper

[`../theory/command-tracking-reward.md`](../theory/command-tracking-reward.md) —
the full derivation, including where σ_v and σ_ω come from and why a tolerance
has to be bounded from *both* sides rather than just set to three times the
noise.
