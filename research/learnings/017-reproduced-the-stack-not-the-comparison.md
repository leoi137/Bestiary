---
triggers: [new_machine, comparison, new_robot, long_run]
guard: none — the reproduction itself is not assertable, and the one assertable
       piece (that a run's critic sees a privileged observation group when the
       config claims asymmetry) is filed as anomaly 57 rather than guessed at here.
last_confirmed: 2026-07-30
---

# 017 — The second machine reproduced the stack, not the comparison

**Date:** 2026-07-30 · **From:** the first Isaac Hound run, on a second machine
with a Blackwell-class GPU
**Robot:** hound

Every derived number below is printed by
`research/scripts/017_blackwell_reproduction_arithmetic.py`. The measurements it
consumes were reported from the other machine and are named as such in the
script's input block; nothing in this lesson recomputes them from a log this
repository holds, which is itself one of the caveats.

**Written separately from `learnings/016` rather than as a section of it, on
purpose.** The two findings came out of the same day on the same host and share
nothing else. 016's trigger is *someone is about to add a launch path*; this
one's is *someone is about to compare two numbers*. 016's falsifier is about
thread pools; this one's is about confounds. The format gives a lesson one
`guard:` field and one *How we would know this is wrong* section, and forcing two
unrelated mechanisms through one of each would blunt both. Splitting is what the
folder's own rule says to do when torn.

## What we believed before

That the hard part of putting this project on a bigger GPU was **getting it to
run**, and that once it ran the throughput number would tell us how much faster
the second machine is.

Half of that was reasonable and recently earned. `research/decisions/0001`
deferred Isaac Lab on a hardware argument — half the minimum VRAM, an
architecture no longer listed as supported — and that premise was refuted on
2026-07-29 when the stack ran on this machine's RTX 2080 anyway. So "will it
run at all" was, correctly, the live question, and there was a real chance the
answer on brand-new silicon was "not without the vendor's container."

The other half was the error. **A number that comes back from a machine you have
just stood up is not automatically comparable to the number already in the
ledger**, and this project has written that lesson down twice already —
`learnings/010` compared two seeds through a checkpoint it had already proven not
to trust, and `learnings/013` established that a number is only as durable as the
artifact it was computed from. Both are about the same failure and neither had
anything to do with hardware, which is exactly why the hardware case did not feel
like an instance of it.

## What happened

**The stack reproduced.** Reported: Isaac Lab **3.0.0-beta2** at commit
`af1bab4dc173ba69b08fab779c14ead61d13fd33` on a **Blackwell-class GPU, compute
capability 12.0**, 32,607 MiB of VRAM, driver 580.95.05, CUDA 12.8, on a 48-core
host, installed **from a bare CUDA image via pip — no vendor container**. Both
oracles came back identical to local:

| oracle | result |
|---|---|
| `bestiary.isaac.check_desert_terrain` | **8/8** |
| `bestiary.isaac.check_hound` | **22/22** |

Those two totals are verifiable here without a GPU: `check_desert_terrain.py`'s
`CHECKS` tuple holds 8 entries, and `check_hound.py` has `FILE_CHECKS` of 9 plus
`SIM_CHECKS` of 13, which is 22. A matching count is therefore a real claim about
the assets and the articulation, not a coincidence of output formatting.

**And Hound trained.** 2048 environments, `num_steps_per_env` 24, 30 iterations
clean, one seed:

| quantity | reported |
|---|---|
| throughput | **22,241 steps/s** |
| iteration time | **2.21 s** (collection 2.07 s, learning 0.146 s) |
| peak VRAM | **5,718 MiB** of 32,607 |

The reported figures are internally consistent, which is worth checking before
anything is built on them:

    2048 envs x 24 steps / 2.21 s = 22,240.7 steps/s   vs 22,241 reported   (0.00%)
    2.07 s + 0.146 s = 2.216 s                          vs 2.21  reported   (0.27%)

Collection is **93.7%** of the iteration and learning is **6.6%**. The run is
simulation-bound, not optimisation-bound — which means a bigger network is close
to free here and a faster simulator is not.

**Then the comparison fell apart.** The record's local figure is **7,630
steps/s at 1024 environments** on the RTX 2080
(`research/decisions/0003`, line 31). The ratio is right there and it is
worthless:

| ratio | value | why it is not a speedup |
|---|---|---|
| raw steps/s | **2.915×** | four variables moved |
| per-environment steps/s | **1.457×** | removes one of the four |

The four, all moving at once: the **robot** (Hound, 16 DoF / 17 bodies / 4
rolling cylinder contacts, against ANYmal-C's 12 DoF / 13 bodies / 4 point feet
carrying an LSTM actuator network), the **environment count** (2048 vs 1024), the
**GPU** (Blackwell vs Turing), and the **host core count** (48 vs 16). Under this
repository's seed rule a comparison needs at least three seeds per arm and
exactly one variable changed. This has one seed per arm and four.

**A third finding, from reading the config rather than the run.** The run was
**not** asymmetric actor-critic, despite rsl_rl supporting it. `obs_groups` is
declared `MISSING` in the base runner config
(`isaaclab_rl/rsl_rl/rl_cfg.py:252`) and `AnymalCRoughPPORunnerCfg` never sets
it — while the sibling `AnymalDRoughPPORunnerCfg` does, at
`.../anymal_d/agents/rsl_rl_ppo_cfg.py:25`. And the shipped observation
configuration has nowhere for a critic to look anyway: `ObservationsCfg` in
`velocity_env_cfg.py` defines exactly one group, `policy`, at line 192. There is
no privileged group in the environment, so the critic sees the same vector the
actor does no matter how `obs_groups` is filled in.

## Why it happened

**Why the reproduction was easier than expected.** Decision 0001's hardware
argument rested on Isaac *Sim*'s published minimum specification, and that
specification is justified in the vendor's own documentation by *rendering*
requirements. Headless RL training does not render. The 2080 result already
showed the floor was lower than the spec sheet; the Blackwell result shows the
ceiling is not gated by a container either, because what a from-scratch
locomotion run needs from the platform is PhysX plus a working CUDA runtime, and
pip wheels supply both.

**Why the throughput ratio is not a measurement.** The two arms differ in the
two things that dominate the cost of the part of the iteration that takes 93.7%
of it. Contact count is one: four continuously-rolling cylinder-versus-mesh
contacts do not cost what four point feet cost, and the solver work scales with
the contact set, not with the body count. Environment count is the other, and it
does not cancel by division — a GPU simulator has fixed overhead per iteration
and per-environment cost on top, so doubling environments changes throughput
without telling you anything about the hardware.

Dividing by environment count moves the number from 2.915× to 1.457× and that
movement is the whole point: **half of the apparent speedup was the environment
count, and the remaining half still contains the robot and the GPU
inseparably.** Reporting 2.915× would have credited the card with work the batch
size did.

**Why the critic gap went unnoticed.** Asymmetric actor-critic — a critic that
sees privileged simulator state the actor cannot have — is standard in this
literature and rsl_rl implements it, so "the framework supports it" reads as "we
have it." It was inherited, not chosen, and the inheritance is silent in both
directions: nothing warns that `obs_groups` is unset, and nothing warns that the
environment publishes no group a critic could have been given. A prior refutation
pass had already noted the shipped ANYmal config carries this gap; this run is
the first time it applied to our own training.

## The math

**Internal consistency.** For a synchronous on-policy trainer, one iteration
collects a fixed batch and then learns from it, so throughput is not an
independent measurement:

    steps/s = N · T / t_iter

with `N` the number of parallel environments (dimensionless count), `T =
num_steps_per_env` the rollout length in control steps per environment, and
`t_iter` the wall-clock seconds per iteration. Worked on the reported numbers:

    2048 · 24 / 2.21 s = 49,152 / 2.21 = 22,240.7 steps/s

against 22,241 reported — agreement to 0.00%. Two reported numbers that must
agree and do are worth more than either alone.

**Wall clock, which is what a budget is denominated in.** The shipped
`max_iterations` is 1500 (verified in `AnymalCRoughPPORunnerCfg`):

    1500 · 2.21 s = 3,315 s = 0.921 h per seed
    x 3 seeds (the seed rule) = 2.762 h per arm
    samples seen = 1500 · 49,152 = 73,728,000

The same 1500 iterations on this machine, at 1024 environments and 7,630
steps/s, is 1.342 h. So: **the same iteration count, at twice the parallelism, in
69% of the wall clock** — a statement that holds iteration count fixed and lets
sample count double, and says so.

**Re-pricing the two published sample budgets from `decisions/0004`.** These are
projections at a measured rate, and the second row is the one that changes a
decision:

| budget | samples | at the local rate | at the remote rate |
|---|---|---|---|
| Rudin et al. operating point | 1.47e8 | 5.4 h | **1.8 h** |
| published Go2-W recipe | 1.97e9 | 71.7 h | **24.6 h** |

Decision 0004 priced the wheeled-legged budget at 72 hours using an ANYmal rate,
because an ANYmal rate was the only one we had. The remote rate was measured on
**Hound**, the wheeled-legged robot that budget is about, so 24.6 h is the first
projection of it at a rate from the right robot. It is still a projection.

**The VRAM figure, and one thing it does say.** Against the second machine's card
the footprint is unremarkable:

    5,718 / 32,607 = 17.5% of the card

Against **this** machine's ceiling it is not:

    5,718 / 6,000 = 95.3%,  leaving 282 MiB spare

That is the decision-relevant number, and it points the opposite way from
`decisions/0004`'s "4096 envs is out of reach": 2048 environments of *Hound*
would fit under the local 6,000 MiB ceiling with a thin margin — **if** the two
cards allocate alike, which nothing here establishes.

And a flag, not a finding:

    remote 5,718 / 2048 = 2.792 MiB per environment
    local  4,649 / 1024 = 4.540 MiB per environment
    ratio  0.615

The **bigger** robot at **twice** the environment count reports **39% less**
memory per environment. Either those two "peak" figures were sampled by different
instruments or one of them is not a peak. `anomalies.jsonl` row 55 already
records that the local 4,649 MiB has no recorded instrument; that caveat now
extends to the remote 5,718 MiB, and until one of them has an instrument neither
belongs in a sizing argument.

## What to do next time

**Say what moved.** A throughput number carries the robot, the environment count,
the rollout length, the GPU and the host with it. Publish those five alongside
it or publish no ratio. A ratio with unnamed confounds outlives its caveats.

**The reproduction is the finding; the number is not.** What this run established
is that the stack runs on Blackwell from pip, that both oracles pass at full
count on a machine that has never seen the assets, and that Hound's articulation
loads and trains. All three are real and none of them is a speedup.

**When a comparison is uncontrolled, publish the normalisation anyway and say
what it removes.** 2.915× raw and 1.457× per-environment, printed together, tell
a later reader that half the apparent gain was batch size. Printing only the
uncontrolled figure would have hidden that; printing only the normalised one
would have hidden that it is still uncontrolled.

**Check `obs_groups` before claiming asymmetry, and check the environment for a
privileged group before checking `obs_groups`.** The config field is downstream:
an `obs_groups` naming a group the environment does not publish is a different
error, and an environment with one group makes the field moot. When Hound gets
its own agent config, either declare the asymmetry and add the privileged
observation group, or state plainly that the critic is symmetric. Filed as
`anomalies.jsonl` row 57 rather than fixed here.

**A second machine does not inherit the eight preflight conditions.** Those
conditions describe this machine's card, disk, RAM and clock. A remote run needs
its own preflight, and the fact that it needed writing at all is a governance
question rather than a physics one — recorded on the private side.

## How we would know this is wrong

**One seed, one host, 30 iterations, and no artifact in this repository.** That
last clause is the sharpest limitation and it is the one `learnings/013` warns
about: every number in this lesson was transcribed from a session rather than
computed from a log, checkpoint or event file that this repository holds. The
internal-consistency check above is the only defence against a transcription
error, and it can only catch an inconsistent one.

This learning is wrong if any of these is observed:

- **A controlled arm shows a different ratio.** Run ANYmal-C at 1024
  environments on the second machine — one variable, the GPU. If that comes back
  near 7,630 steps/s, then essentially none of the 2.915× was the card and the
  gain was robot and batch size, which would make even the 1.457× figure a
  misleading thing to have printed.
- **The VRAM figures reconcile.** If both peaks are re-measured with one named
  instrument and the per-environment footprint comes out *higher* for Hound as it
  should, then the 0.615 ratio was an instrument artifact and the "flag, not a
  finding" paragraph resolves rather than deepens. Anomaly 55's cheapest next
  step is the same measurement.
- **2048 environments of Hound does not fit locally.** The 95.3%-of-ceiling
  arithmetic assumes the footprint transfers between a Blackwell card and a
  Turing one. If a local attempt exceeds 6,000 MiB, the assumption is wrong and
  the interesting consequence — that `decisions/0004`'s "4096 envs is out of
  reach" may have been reasoning from an unmeasured base — does not follow.
- **The oracles pass for the wrong reason.** 8/8 and 22/22 match local exactly.
  If any of those checks turns out to be vacuous on a fresh host — an asset it
  silently skips, a path that resolves to nothing — then a matching count proves
  the checks ran, not that the machine is the same. `learnings/014` is the
  precedent: a guard can be green, fully tested, and check nothing at all, and no
  count distinguishes that case.
- **Collection is not 93.7% at scale.** The simulation-bound conclusion, and
  therefore "a bigger network is close to free," comes from a 30-iteration
  average at 2048 environments. If the learning share grows materially at 4096
  environments or with a wider observation, the conclusion inverts and network
  size stops being free.
