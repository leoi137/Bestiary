# 012 — When an average hides a single winner

**One sentence:** A mean over N conditions tells you the *total* divided by N,
never how that total was spread, so a policy that wins enormously in one
condition and loses in three others reports the same headline as one that is
competent everywhere.

## The idea

To score a policy you give it a set of jobs — here six velocity commands, 20
episodes each — and average the returns. A **return** is the total reward
collected in one episode; a **cell** is one command, scored on its own. The
average is one number, so it is the number that ends up in the write-up.

The trouble is that returns are **signed**: a cell can pay −29 as easily as
+94. When you average signed numbers whose spread is much larger than the
average itself, the result is a small difference of large opposite-signed
quantities. Such a difference is arbitrarily sensitive to any one term — move
one cell and the answer moves by more than the answer.

And the thing actually reported was not the mean but a **ratio**: the policy's
mean over the control's mean, against a pre-registered bar of "≥5x is success".
A ratio inherits every problem of its numerator and adds one of its own. Its
denominator is a signed average too, so it can pass through zero, and a
quantity divided by something that can be zero has no stable meaning — near the
crossing it takes any value you like, and past it, it flips sign while the
policy has not changed at all.

## The math

The mean of N cell gaps `g_i` (policy return minus control return, in units of
reward per episode) is `ḡ = (1/N)·Σ g_i`. Measured, N = 6:

    (0.5, 0, 0)      +93.82
    (0.8, 0, 0)      +37.07
    (0.5, 0, -0.4)   +24.52
    (-0.3, 0, 0)      -5.49
    (0.5, 0, +0.4)   -17.80
    (0.0, 0, 0.45)   -37.18
    ---------------------------
    Σ                +94.93     ḡ = 94.93/6 = +15.82

One cell is **93.82/94.93 = 98.8%** of the entire total. The policy loses to
doing nothing in **3 of 6** cells. The headline was policy 19.73 versus control
3.91 — a **5.04x** ratio, which cleared the bar.

Now the diagnostic: **leave-one-out**. Drop cell i, recompute the headline, and
see how far it moves.

    dropped            policy  control     ratio
    (0.5, 0, 0)          4.95     4.73     1.05x      <- the whole result
    (0.8, 0, 0)         15.73     4.16     3.78x
    (-0.3, 0, 0)        18.51    -1.58     UNDEFINED  <- denominator changed sign
    (0.5, 0, +0.4)      29.10     6.56     4.44x
    (0.5, 0, -0.4)      20.64     6.56     3.15x
    (0.0, 0, 0.45)      29.46     3.04     9.69x

Remove one of six points and 5.04x becomes 1.05x — a dead heat. Remove a
different one and the control's mean is **−1.58**, and "18.51 versus −1.58" is a
pair of numbers a reader will divide out of habit into a ratio that does not
exist.

**What it means:** the experiment measured one command, not six. The machine
learned to drive forward at 0.5 m/s and did not learn the other five.

All figures from
[`scripts/012_leave_one_out.py`](scripts/012_leave_one_out.py), which reads
`research/measurements/track_rel_s1_best.json` and
`track_rel_zero_action.json`.

## The reflex to build

**Look at the per-cell table before believing any aggregate, and always compute
leave-one-out.** If dropping one of N points moves the headline by an order of
magnitude or flips its sign, you have measured one point.

Better summaries than a mean ratio, cheapest first: publish the per-cell table;
report **fraction of cells won** (3/6 — which alone would have stopped this);
use a median, which one outlier cannot move; or weight each cell by how often
that command actually occurs in training. That last one is available here for
free from the command sampler's own probabilities, and it puts the policy
**+12.6%** over doing nothing rather than 5.04x.

## Where it bites here

`src/bestiary/record/track_eval.py` computes `drive_grid_mean`, and that single
field was compared against a pre-registered success bar. The per-cell dictionary
sat in the same JSON, unread. A success verdict was one commit away from
entering the ledger for a policy that is worse than a switched-off robot on half
its job.

## If you want to go deeper

[`002 — Why one training run is not a result`](002-why-one-seed-is-not-a-result.md)
— the same failure one level up: a mean over seeds hides which seed carried it.
