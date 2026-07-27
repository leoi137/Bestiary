# 002 — Why one training run is not a result

**One sentence:** Training a robot twice from the same code with a different
random seed gives two different robots, so a single run is one draw from a
distribution — and no amount of testing that one robot tells you where the
next one would land.

## The idea

A **seed** is the number that initializes every random choice in a training
run: the network's starting weights, the exploration noise, the order
experience is replayed. Change only the seed and everything else — the robot,
the reward, the terrain, the hyperparameters — is identical. The policy you
get at the end is still different, and often *very* different. Reinforcement
learning is not a calculation that converges to one answer; it is a search
that finds one of many.

This creates a trap, and it is worth naming because it feels like rigor from
the inside. There are two separate sample sizes:

- **Episodes** — how many rollouts you average to score *one* trained policy.
  Cheap. Each one takes seconds.
- **Seeds** — how many times you repeat the *whole training run*. Expensive.
  Each one takes hours.

Measuring more episodes makes your estimate of *that policy* sharper. It does
nothing at all for your estimate of the **spread between training runs**,
because there is still only one training run. Averaging ten thousand episodes
of one seed gives you a beautifully precise measurement of a single sample.

## The math

The honest summary of a comparison is a **confidence interval** — a range that
probably contains the truth. For a paired difference between two arms measured
over `n` episodes:

    CI₉₅ = d̄ ± 1.96 · SE ,   SE = s / √n

- `d̄` — mean per-episode difference (trained policy − doing nothing), in
  return points
- `s` — standard deviation of that difference across episodes
- `n` — number of episodes
- `1.96` — the multiplier that makes it a 95% interval

For our torque-driven Hound against a do-nothing control, at `n = 60`:

    CI₉₅ = [−52.1, +132.9]  →  d̄ = +40.4 ,  SE = 47.2

**That interval contains zero.** The trained policy might be 133 points better
than standing still, or 52 points *worse*, and the data does not distinguish
those. Physically: after 3.75 million steps of training, we cannot say the
robot learned to do anything more useful than not falling over.

Shrinking the interval below the effect needs `n` to grow by the square of the
ratio, since `SE` falls as `1/√n`:

    (92.5 / 40.4)² = 5.2×  →  315 episodes, up from 60

And that only settles it for *this one policy*. The spread between seeds is a
different quantity, and with one seed per arm there is nothing to compute it
from — one draw against one draw, however many episodes back each.

## Where it bites here

`research/ledger.jsonl` has one training seed per arm, so **every comparison
in it is provisional** — including the headline that PD control reached the
same reward band in 5× fewer samples than torque. `CLAUDE.md`'s seed rule
(≥3 seeds per arm, exactly one changed variable) exists for this, and
`record/ledger.py` refuses to write a single-seed row unless it is marked
`provisional`. Cycle 005 launched `hound_pd_desert_s1` — identical to
`hound_pd_desert_v0` except `seed=1` — for no other purpose than to turn n=1
into n=2.

Arithmetic above: `../../research/scripts/seed_variance_math.py`.

## If you want to go deeper

[`../../research/learnings/008-best-checkpoint-is-the-luckiest-episode.md`](../../research/learnings/008-best-checkpoint-is-the-luckiest-episode.md)
— the same sample-size mistake one level down, where picking the best of many
noisy evaluations selects the luckiest episode rather than the best policy.
