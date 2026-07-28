# 008 — What a policy is, and why it is a neural network

**One sentence:** A policy is just a **function from what the robot senses to
what its motors do** — and it is a neural network only because that function
has 169 inputs, which is far too many to write down any other way.

## The idea

Every control step (20 times a second) the hound assembles one list of 169
numbers: joint angles, joint speeds, body orientation, the drive command, and
so on. It must reply with 16 numbers, one per actuator — 4 wheels, 12 leg
joints — each in [−1, 1]. **The policy is whatever turns the first list into
the second.** That is the entire definition. Nothing about it requires a
network.

The obvious way to store a function is a **lookup table**: one row per possible
input, holding the answer. Split each of the 169 inputs into just **10** bins —
absurdly coarse, you could not balance on it — and the table has 10¹⁶⁹ rows.
The observable universe holds roughly 10⁸⁰ atoms. The table is **10⁸⁹ times
larger than that**. Even a *binary* table, each input reduced to "high or low",
needs 2¹⁶⁹ ≈ 10⁵⁰·⁹ rows.

So a table is not merely expensive; it is impossible, and by a margin no
hardware ever closes. A network is the alternative: a fixed, small stack of
matrix multiplications with adjustable numbers in it, that **computes** the
answer instead of remembering it, and generalises to the vast majority of
inputs it will never have seen.

## The math

The hound's actor, read straight out of a real checkpoint
(`runs/hound_pd_desert_v0/ant_sac.zip`):

    a = tanh( W₃ · relu(W₂ · relu(W₁·o + b₁) + b₂) + b₃ )

**o** the observation, 169 real numbers; **W₁** a 256×169 matrix; **W₂**
256×256; **W₃** 16×256; **b** the bias vectors; **relu(x) = max(0, x)**, the
bend that stops three matrices collapsing into one; **a** the 16 actions,
unitless in [−1, 1].

Counting parameters:

| tensor | shape | count |
|---|---|---|
| `latent_pi.0` (weight + bias) | 256×169, 256 | 43,520 |
| `latent_pi.2` | 256×256, 256 | 65,792 |
| `mu` | 16×256, 16 | 4,112 |
| `log_std` | 16×256, 16 | 4,112 |
| **total** | | **117,536** |

**117,536 adjustable numbers replace 10¹⁶⁹ table rows** — about 10¹⁶³·⁹ rows
per parameter. That compression *is* the reason for the network.

One wrinkle the table story hides: the last layer has **two** heads, `mu` and
`log_std`, 16 numbers each. SAC's policy does not output an action; it outputs
a **probability distribution** over actions — a mean and a spread per actuator
— and training samples from it, which is where exploration comes from. At
evaluation we pass `deterministic=True`, and Stable-Baselines3 then returns the
distribution's mode, `tanh(mu)`, discarding the spread entirely
(`SquashedDiagGaussianDistribution.mode`).

**Physically:** the trained hound is 117,536 numbers that, 20 times a second,
turn 169 sensed values into 16 motor targets — and during training it is really
proposing a *range* of motor targets and rolling the dice inside it.

## Where it bites here

`W₁` is `Linear(169, 256)`, so its first dimension **is** the observation
width. Change the observation list by one term and every existing checkpoint
does not degrade — it **fails to load**. That is why the width is a one-way
door, and why `src/bestiary/envs/obs_spec.py` hashes the term list and
`train.py` refuses a resume whose hash moved: a reordered observation of
identical width loads cleanly and feeds the policy a permuted world.

Arithmetic: `../../research/scripts/policy_lesson_math.py`.

## If you want to go deeper

[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905),
Haarnoja et al. 2018 — §4 defines the squashed Gaussian policy the `mu`/
`log_std` pair parameterises.

*Assumes [001 — what a reward function is](001-what-a-reward-function-is.md).*
