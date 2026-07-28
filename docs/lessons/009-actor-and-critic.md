# 009 — Why one network is not enough

**One sentence:** The **actor** picks the move and the **critic** guesses how
much reward will follow — and the actor needs that guess, because the reward
for the current step cannot tell a slow step that leads somewhere good from a
fast step that leads off a cliff.

Assumes [008 — what a policy is](008-what-a-policy-is.md).

## The idea

Lesson 008 showed the policy as a matrix that turns observations into motor
commands. That network is the **actor**. If it were the only one, training it
would need an answer to "was that a good action?" — and the only thing arriving
each step is that step's reward. A step that scores 0.3 now and puts the robot
on its face in two seconds looks identical to a step that scores 0.3 now and
sets up a clean stride.

So SAC trains a second network, the **critic**, whose only job is to output one
number: the total future reward expected if you take action *a* in state *s* and
behave normally afterwards. That number is called **Q(s, a)**. The actor is then
trained to pick actions the critic scores highly. Neither can do the other's job:
the actor cannot see the future, and the critic never chooses anything.

The giveaway is what each one is allowed to look at. Open the checkpoint this
project froze — 5,106,759 bytes of `hound_track_rel_s1` — and the first layers
are different shapes:

    actor  first layer   Linear(169, 256)     <- 169 numbers: the observation
    critic first layer   Linear(185, 256)     <- 185 = 169 + 16

**185 = 169 + 16.** The critic's input is the observation *plus the 16 motor
commands*, because it is grading a pair. The actor's is the observation alone,
because it is producing those 16 commands and cannot be handed its own answer.
That single difference of 16 columns is the whole architecture.

The critic's output layer is `Linear(256, 1)` — 256 numbers in, **one** out.
All that machinery to produce a single scalar guess.

## The math

The critic is trained to satisfy the **Bellman equation** — a consistency rule,
not a formula you evaluate:

$$Q(s_t, a_t) \;=\; r_t \;+\; \gamma\, Q(s_{t+1}, a_{t+1})$$

- $Q(s,a)$ — expected total future reward from state $s$ after action $a$, in
  reward units
- $r_t$ — the reward actually paid this step, reward units
- $\gamma = 0.99$ — the discount, dimensionless ([lesson 006](006-what-gamma-is-saying-about-the-future.md))
- $s_t, a_t$ — the 169-number observation and the 16-number action at step $t$

It says: *my guess now must equal what I actually got, plus my guess from the
next state.* The critic never sees the true future — it is trained to be
consistent with its own next guess, one step at a time, and the real reward
$r_t$ leaking in at every step is what stops that circularity from being empty.

How far ahead does the guess reach? Weights fall off as $\gamma^k$, so the
effective horizon is

$$\frac{1}{1-\gamma} \;=\; \frac{1}{1-0.99} \;=\; 100 \text{ steps}$$

A step here is `frame_skip = 10` physics steps of 0.005 s, so the control rate is
20 Hz and one step is 0.05 s:

$$H \times \Delta t \;=\; 100 \text{ steps} \times 0.05\,\mathrm{s} \;=\; 5.0\,\mathrm{s}$$

- $H$ — the horizon in steps, dimensionless
- $\Delta t$ — one control step, 0.05 s

Physically: the critic is what lets the hound accept a worse step now for a
better **five seconds**. That is roughly the length of the manoeuvre this
project cares about — turning onto a commanded heading and settling — which is
why $\gamma$ is 0.99 and not 0.9, whose horizon of 0.5 s is too short to see a
turn finish.

That job is harder than the actor's, and the checkpoint shows the field paying
for it. From `docs/lessons/scripts/actor_critic_math.py`:

| | parameters | share |
|---|---|---|
| critic | 227,330 | 39.7% |
| `critic_target` | 227,330 | 39.7% |
| actor | **117,536** | 20.5% |
| total | 572,196 | |

$$\frac{227{,}330}{117{,}536} = 1.93\times$$

The critic carries **1.93×** the actor's parameters, and its optimizer state in
the zip is 1,828,586 bytes against 947,150 — the same 1.93×. Four fifths of this
checkpoint is machinery for *judging* actions; one fifth chooses them.

Two details that fall out of the table. The critic is **two** networks, `qf0`
and `qf1`, trained on the same data and always read as the smaller of the pair —
a single critic learns to be optimistic, and the actor then chases actions that
are only good on paper. And `critic_target` is a slow-moving copy, held back so
the right-hand side of the Bellman equation does not move every time the
left-hand side is updated: without it the critic chases a target it is itself
shifting.

## Where it bites here

The critic is why changing a reward mid-run is not recoverable
([lesson 004](004-why-a-reward-change-poisons-the-buffer.md)). Its 227,330
parameters encode predictions of *the old* reward; move the reward and every one
of them is a confident estimate of a quantity that no longer exists. This is not
theoretical here — it is `research/nulls.jsonl` row 1 and `learnings/002`, and it
is why `train.py` writes `reward_spec` into `config.json` and refuses a resume
across a changed reward rather than trusting anyone to remember.

It also sets the price of a one-way door. The actor's first layer is
`Linear(169, 256)`; the critic's is `Linear(185, 256)`. Change the observation
width and **both** die, along with both optimizer states — 4.9 MB of the 5.1 MB
checkpoint.

## If you want to go deeper

Haarnoja et al., *Soft Actor-Critic* (2018), [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
— §4.2 is the twin-critic argument, and it is short.
