# 001 — What a reward function is, and how ours told the robot to stand still

**One sentence:** A reward function is a single number handed to the robot
every control step saying how good that step was, and the robot will maximize
*that number* — not what you meant by it.

## The idea

Reinforcement learning has no notion of "walk properly". It has a number. The
simulator steps, the reward function looks at what happened, returns one
scalar, and the learning algorithm adjusts the network to make the sum of
those scalars larger. Every intention you have about the robot's behaviour has
to be expressed inside that one number, or it does not exist.

Ours was built out of four terms added together each step:

- **forward** — reward for moving in +x, so it goes somewhere
- **control cost** — a penalty for large motor commands, so it does not
  thrash
- **contact cost** — a small penalty for violent collisions
- **healthy bonus** — a flat +1 for every step it has not fallen over

That is a reasonable-sounding list, and on flat ground it worked: the spider
learned to run at 7 m/s. On the desert heightfield the same four terms
produced a robot that braced itself and crept at 0.37 m/s — and when we
measured a policy that does *literally nothing*, it scored **higher**.

## The math

Every step, the reward is

    r = w_fwd · v_x  −  w_ctrl · Σaᵢ²  −  contact  +  healthy

- `v_x` — forward velocity, m/s
- `a` — the 12 motor commands, each dimensionless in [−1, 1]; `Σaᵢ²` is total
  effort
- `w_fwd = 1.0`, `w_ctrl = 0.1` — the weights we chose
- `healthy = 1.0` per step survived

Measured over 5 episodes of `spyder_desert_v0` on the desert, per step:

| | walking | standing still |
|---|---|---|
| forward | +0.294 | 0.000 |
| control | **−0.571** | **0.000** |
| contact | −0.016 | −0.013 |
| healthy | +1.000 | +1.000 |
| **per episode (×1000)** | **707** | **987** |

Standing wins by **280 points**. And notice *why*: standing still means
`a ≈ 0`, so `Σaᵢ² ≈ 0`, so the control cost is **exactly zero**. Doing nothing
is the one behaviour that pays no effort penalty at all.

The rough terrain is what broke it. On flat ground the spider moved fast
enough that forward reward beat control cost about 8.7 : 1. On the desert it
could only manage 0.294 against 0.571 — **0.51 : 1**. The cost never rose; the
payoff collapsed.

**In plain terms: we were paying the robot 29 cents to do 57 cents of work,
and offering a guaranteed dollar for sitting still. It sat still, and it was
right to.**

The fix is not to punish standing — it is to make moving affordable. Dropping
`w_ctrl` from 0.1 to 0.02 rescores the exact same gait at **1164** against
standing's 987, without changing anything the robot does.

## Where it bites here

`src/bestiary/envs/spyder.py` — the weights, and a comment block that still
argues for the old value. `research/CORE_PLAN.md` holds the worked fix, and
`research/learnings/001-flat-reward-breaks-on-terrain.md` is the postmortem.

The habit this leaves behind: **before trusting any reward, roll a
zero-action policy against it.** If doing nothing scores higher than the
trained policy, the reward is wrong — no matter how sensible its terms look
written down. That two-minute check caught the same bug independently on a
second robot.

Arithmetic above: `scripts/001_reward_arithmetic.py`.

## If you want to go deeper

`../../research/CORE_PLAN.md` — the full reward and observation spec, including
why the fix replaces "go fast" with "hit the commanded speed".
