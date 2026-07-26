# Core Plan: One Model We Keep Training

How we go from "retrain every time something changes" to "one core policy that
keeps improving." Written after `runs/spyder_desert_v0` failed. Companion to
`ROADMAP.md` (this is the *how*; the roadmap is the *what*).

Lessons that came out of failed runs live in [`learnings/`](learnings/README.md).
That folder is the thing that carries over when we retrain — add to it whenever a
run surprises us.

## What went wrong (the 30-second version)

`spyder_desert_v0` fine-tuned the flat-ground spider on the desert. Measured
over 5 greedy episodes each:

| policy | env | reward | speed |
|---|---|---|---|
| `spyder_walk_v3` | flat | 7224 | 7.05 m/s |
| `spyder_walk_v3` | desert (zero-shot) | 83 | dies at 77 steps |
| `spyder_desert_v0` | desert | 509 (best eval 832) | 0.37 m/s |
| **zero action (do nothing)** | desert | **987** | 0 m/s |

**Doing nothing beats the trained policy.** On rough ground the spider can only
reach ~0.4 m/s, but `ctrl_cost_weight=0.1` still charges it full price for the
leg torque. So a full episode earns +294 for moving and pays −571 for the
effort. Moving is a net loss against the 1000-point alive bonus. The reward was
telling it to stand still, and it half-listened.

Two things caused this:

1. **Reward built for flat ground.** On flat, forward reward beats control cost
   ~8.7 : 1, so the cost never mattered. On terrain the ratio flips to 0.5 : 1.
   The cost did not rise — the *payoff collapsed*. Speed fell 24× (7.05 → 0.29
   per step), effort only 1.4× (0.81 → 0.57).
2. **Warm-start across a reward change.** The critic was fit on ~7000-return
   flat episodes and suddenly saw ~500-return desert ones. Huge errors, and the
   gait was destroyed in the first few thousand steps (6331 → 146).

### Confirmed on a second robot, from scratch

`runs/hound_desert_test150k` — the 16-DoF wheel-legged Hound, 150k steps, **no
warm-start**:

| HoundDesert-v0 | reward | episode length |
|---|---|---|
| trained best (150k) | 109 (probe: 8–161) | dies at 21–141 steps |
| **zero action** | **961** | **survives all 1000** |

Doing nothing scores **9× better** than the trained policy. Forward vs effort on
its best episode: +88.7 vs −67.2 (1.3 : 1).

**This isolates the variable.** Spyder's v0 had two candidate causes — bad reward
*and* a warm-start. Hound had no warm-start, no stale critic, no annealed α, and
landed in the same place. So cause #1 is sufficient on its own; #2 made it worse.
Two robots, two setups, one bug — and it is in the reward, not in either body.

## Why from scratch — once

The actor can usually survive a reward change. The critic can't: it is
bootstrapped over 1000 steps, so a reward rescale is wrong everywhere and the
error compounds. Normally you'd fix that by keeping the actor, resetting the
critic, and relabelling the buffer with the new reward.

Three reasons that recipe doesn't rescue us here:

1. **The buffer can't be relabelled.** `envs/spyder_env.py:289` drops world x,y
   from the observation, so the stored transitions don't contain the
   x-displacement the forward reward needs.
2. **The old actor is the problem.** Its habit is brace-and-creep at 0.37 m/s —
   the exact prior we want gone.
3. **It has no exploration left.** SAC's entropy coefficient α entered the
   desert already annealed (0.136), fell 17× in 120k steps, and flatlined at
   0.008. A fresh run starts near 1.0 and explores widely first. See
   [`learnings/002`](learnings/002-no-warm-start-across-reward-change.md).

So: from scratch. Once.

## The five steps

### 1. Start one fresh model

A SAC model is three parts, and **all three start blank**:

| part | what it is | file |
|---|---|---|
| **Actor** | the brain — obs in, 12 torques out. The thing we keep. | `ant_sac.zip` |
| **Critic** | predicts future reward so the actor knows what was good. Training-only, discarded at deploy. | also in the zip |
| **Replay buffer** | memory of past steps, rewards baked in at collection time. | `ant_buffer.pkl` |

v0 loaded actor *and* critic but started with an empty buffer — a stale critic
insisting "7000" while the buffer filled with 500. That mismatch is the failure.

Nothing to build: `train.py` decides fresh-vs-resume by whether `config.json`
exists in the run dir, so **a new `--run-name` is already a fresh model.**
Budget 4–5M steps (~11–14 h at the current 370k steps/hour). The first ~10–20k
steps are random flailing — normal.

### 2. Lock three things before starting

These three are the *only* things that force another from-scratch redo. Lock
them now and we stop redoing.

- **Reward shape *and* numbers.** Not just the weights — the *terms*. See below.
- **Observation list.** Add the command slots *and* the terrain height-sample
  slots **now**, filled with zeros when unused. Adding them later changes the
  obs shape and orphans every checkpoint — the trap already flagged in the
  `spyder_env.py` docstring.
- **Terrain set.** Decide the full list up front (dunes, rocks, slopes, stairs,
  flat). Unused types can stay at zero difficulty.

**These three are not equally binding.** If we only get one right, get the obs
list right:

| lock | how reversible? |
|---|---|
| **Obs list** | **Truly one-way.** Change it and every checkpoint dies — not degraded, incompatible. |
| Reward shape (terms) | Semi, but painful. The checkpoint still loads; the critic is worthless. |
| Reward numbers (weights) | Semi. Keep actor, reset critic, relabel buffer — *if* the obs supports it. |
| Terrain set | Mostly reversible. Matters only because new terrain may demand new obs. |

**The obs arithmetic, since this is the one-way door.** Spyder's observation is
113 numbers today:

```
qpos[2:]   17   joint angles + torso quaternion (world x,y dropped)
qvel       18   joint + torso velocities
cfrc_ext   78   contact forces, 13 bodies × 6
         ----
          113
```

With the slack added: `+3` command `(vx, vy, ω_yaw)` and `+25` height samples
(5×5 grid) → **141**. The actor's first layer is `Linear(obs, 256)`, so:

- today: `113 × 256 = 28,928` weights
- after: `141 × 256 = 36,096` weights

Different shape → **`SAC.load()` raises**. Not degraded — it will not load. A
zero-filled slot contributes `0 × w = 0`, so the network behaves identically today
and switching the slots on later is a config flag. Cost of the slack: 7,168
weights, ~0.7% of the actor. Free.

#### The reward, locked (this replaces Step 4's reward change)

Two changes, both on day one of the fresh run.

**(a) Make movement affordable.** `ctrl_cost_weight` 0.1 → **0.02**,
`healthy_reward` 1.0 → **0.5**. Over a 1000-step desert episode:

| | walk | stand still |
|---|---|---|
| today (`0.1` / `1.0`) | 1000 + 294 − 571 − 16 = **707** | 1000 + 0 − 0 − 13 = **987** ← wins |
| effort fixed (`0.02` / `1.0`) | 1000 + 294 − 114 − 16 = **1164** ← wins | **987** (1.18×) |
| both fixed (`0.02` / `0.5`) | 500 + 294 − 114 − 16 = **664** ← wins | 500 + 0 − 0 − 13 = **487** (1.36×) |

Standing pays *zero* effort cost (`a = 0` → `Σa² = 0`), which is the whole trap.
Note we never lower standing's score — we make walking affordable. The
`healthy_reward` cut is signal-to-noise, not correctness: the absolute gap stays
177 either way, but the critic resolves 177-on-487 far more easily than
177-on-987. **The `ctrl_cost_weight` change is the one that matters.**

**(b) Reward the command, not the direction.** Replace the raw forward term with
command tracking:

```
today:    reward += 1.0 · v_x                        "go fast"
locked:   reward += exp(−|v_actual − v_command|² / σ) "hit the commanded speed"
```

Then command `(1, 0, 0)` rewards walking and punishes standing, while command
`(0, 0, 0)` rewards standing and punishes drift. **Standing is not punished —
standing *when told to stand* is rewarded.**

This is why it belongs here and not in Step 4: it is a change to the reward
*terms*, so deferring it would mean another reset-critic-and-relabel cycle.
Sample commands from day one, just biased toward forward at first.

`σ` sets how tight "hitting the command" has to be, and it trades off directly
against the effort cost — too tight and the policy cannot earn it, too loose and
standing satisfies a walk command. **Unset; start at σ ≈ 0.25 m/s and verify with
the standing check under a `(0,0,0)` command.** This is a guess until measured.

It also kills the do-nothing exploit better than (a) does. Once commands are
sampled, no single behaviour scores well across all of them — freezing fails
every nonzero command. There is nothing left to farm.

### 3. Train on all terrains mixed, random every reset

Not desert, then sand, then water. One policy, all terrain types at once,
randomised per reset, with difficulty that ramps up as the policy succeeds.
This is the standard legged-robotics recipe (`legged_gym`). Terrains are not
separate tasks — they are one randomised parameter of one task.

### 4. Widen the command distribution

The command slots and the tracking reward are already in from step 2, so there is
**no reward change here** — this step is just sampling a wider range of commands.

Start biased toward forward. Then widen to sideways, turning, and `(0, 0, 0)`.
"Turn left" and "stand still" become *command values*, not separate models. This
is `ROADMAP.md` Step 1, and it is free once step 2 is locked — which is exactly
why step 2 has to include the reward *shape*, not just the weights.

### 5. Move to MJX or Isaac Lab

Today a run is ~370k steps/hour — 11 hours for 4M steps. At ~4096 parallel envs
it is minutes. **This is the real answer to "we can't retrain every time":
don't make weights precious, make retraining cheap.**

That is what the industry actually did, and it splits cleanly in two:

- **Legs and balance (RL) → trained from scratch in sim.** Boston Dynamics' Spot
  locomotion was trained this way in Isaac Lab, zero-shot onto hardware at
  5.2 m/s. Unitree ships the same recipe (`unitree_rl_gym`). Nobody keeps one
  eternal locomotion policy and fine-tunes it forever.
- **Hands and tasks (imitation) → one core model, fine-tuned forever.** NVIDIA
  GR00T N1 (~50k H100-hours to pretrain, then fine-tunes on one GPU); Boston
  Dynamics + Toyota's Atlas Large Behavior Model.

"One core model forever" is real — but it is how *manipulation* works, because it
learns by copying humans (supervised). Locomotion learns by trial and error, and
RL weights go stale when the reward changes. We are doing locomotion.

## The standing check (run this every time)

Before trusting any terrain reward, roll a zero-action policy on it. **If doing
nothing scores higher than the trained policy, the reward is wrong.** Two
minutes, and it would have caught this at 4M steps instead of 5.75M.

## What we carry forward vs. throw away

**Carry forward (the real asset):** the env suite, the reward spec, the command
spec, the eval battery, the hyperparameters, the terrain generator.

**Throw away freely:** the `.zip`. Weights in RL are a cache of a reward
function crossed with a dynamics distribution. Change either input and the
cache is stale.

## After this

Adding a new skill (water, jumping) no longer means from scratch:

- **Same reward and obs?** Add the terrain type to the mix, keep training the
  same model.
- **Too different to mix?** Train a small specialist, then distil the old policy
  and the specialist into one student. The student trains *supervised* against
  teacher actions, which is the stable regime. Teacher-student distillation is
  already standard in this field.
- **Reward tweak later?** Keep the actor, reset the critic, relabel the buffer.
  That works once the obs list is locked — which is step 2's whole point.

Expect one, maybe two more from-scratch runs ever. Not one per skill.

**Possible Step 3.5 — Adversarial Motion Priors (AMP).** A real discriminator
judges "does this gait look like the reference mocap?" and that score joins the
reward. It is how people get natural-looking gaits instead of the twitchy 7 m/s
thing the flat run learned. Only worth it once commands and terrain are locked.

## The Spyder edits — ⚠️ NOT APPLIED YET

Nothing below has been done. No code has been touched. This is the worked-out
edit list for when we act on step 2, recorded so it does not have to be
re-derived.

**Decision: register new env ids, do not mutate `SpyderEnv` in place.** Changing
the observation space breaks `watch.py` on `spyder_walk_v3` and
`spyder_desert_v0` — those checkpoints will not load at all. So the locked spec
lands as **`Spyder-v1` / `SpyderDesert-v1`** in `envs/__init__.py`, and the v0 ids
keep working. Costs a few lines; keeps every existing run watchable.

Ten edits, all in `envs/spyder_env.py` unless noted:

| # | where | change |
|---|---|---|
| 1 | `:122`, `:124` | `ctrl_cost_weight` 0.1 → **0.02**; `healthy_reward` 1.0 → **0.5** |
| 2 | `:115-121` | **Rewrite the comment block.** It currently argues *why* 0.1 is right ("walking (+~1.9) > standing (+1.0) > dying") — reasoning the desert measurements disproved. A stale justification above a changed number is worse than no comment. |
| 3 | `__init__` | New params: `tracking_reward_weight=1.5`, `command_sigma=0.25`, `command_ranges=((0.3, 1.0), (0.0, 0.0), (0.0, 0.0))` (forward-biased to start), `height_grid=(5, 5)`, `height_spacing=0.15`, `height_enabled=False` |
| 4 | `:145-159` | Add all of those to the `EzPickle.__init__` list — miss this and resume/pickling silently desyncs |
| 5 | `:183-187` | Obs size `+3` command `+25` height → **113 → 141** |
| 6 | `:288-292` | `_get_obs`: append the command block (live) and height block (zeros) |
| 7 | `:260`, `:265` | Replace the forward term with command tracking (below) |
| 8 | `:294` | Sample a fresh command per reset into `self._command` |
| 9 | `:269` | Add `reward_tracking` and `command` to `info` so `VideoEvalCallback` can log tracking error — same pattern the shaping wrappers use |
| 10 | new helper | `_height_samples()`: 5×5 grid in the torso frame via the existing `ground_height_at`, returned relative to torso height. Wire it up, leave disabled. |

Edit 7 in full:

```python
# was: forward_reward = self._forward_reward_weight * x_velocity
err = np.array([x_velocity, y_velocity, self.data.qvel[5]]) - self._command
tracking_reward = self._tracking_weight * np.exp(-np.sum(err**2) / self._sigma**2)
```

`qvel[5]` is the free joint's yaw rate — already available, no new plumbing.

**Command slots go live on day one; only height slots start zeroed.** With
commands enabled and ranges biased forward, the tracking reward is well-defined
immediately. If commands stayed at `(0,0,0)` the tracking term would reward
*standing still* — the exact bug being fixed.

`train.py` needs no changes. `watch.py` will eventually want a way to send
commands for demos — that is step 4 work.

## Which robots this covers

**Shared recipe, separate policies.** The reward shape, command spec, terrain set
and standing check are identical for Spyder and Hound — the bug they fix is in the
reward, which is why Hound reproduced it independently. But one *network* driving
both bodies is a different, harder thing: the shapes differ (Spyder 113 obs / 12
act, Hound 141 / 16), and padded action slots are *meaningless* rather than merely
zero — Spyder has no hub wheels, so the net must infer "these 4 outputs do nothing
on this body." That is morphology-aware architecture work (UniLegs,
multi-embodiment policies), a research project, not a config change.

Revisit one-network-two-bodies after both work separately — by then we will be on
Isaac Lab, where it is the normal way to do it.

**Blocker before any serious Hound run:** zero action drifts **−1.5 m per
episode** (~−3 cm/s) on the desert, because the terrain's 7.82 cm cells match the
8.5 cm wheels. Under command tracking, "commanded zero, actually drifting
backward" is exactly what the reward measures — so this corrupts the reward
directly. Regenerate `make_terrain.py` at `GRID=2048` first. Note this changes a
shared asset that Spyder does not need.

## Open decisions (blocking Step 2)

**Both are UNDECIDED as of 2026-07-25.** The numbers used elsewhere in this
document (25 samples, 141 obs) are placeholders reflecting a recommendation, not a
choice that has been made. Both land in the observation list, so both must be
settled *once*, before the fresh run starts.

### 1. How many terrain height samples? — undecided

**What a height sample is.** The policy is blind today: it feels the ground only
through foot contact. Height samples let it *see* — measure the ground height at a
grid of points around the body and feed those numbers into the observation:

```
· · · · ·     each · = "how high is the ground here,
· · · · ·      relative to my torso?"
· · ▲ · ·     ▲ = the robot
· · · · ·     5×5 grid = 25 numbers
· · · · ·
```

A dune ahead shows up as positive values before a foot ever touches it, so the
policy can prepare instead of stumbling.

**Not lidar — a lookup.** We already own the terrain as a heightfield array, so
there is no ray casting and no rendering: for each grid point, compute its world
(x, y) and call the existing `ground_height_at()` in `envs/terrain.py` — the same
function the terrain-aware health check already uses. 25 array reads per step.

That is **privileged information**: data the sim has that a real robot could not
get for free. The standard way to cash it in is teacher-student — a teacher trains
with the cheat lookups, then a student learns to reproduce its actions from what
real sensors (depth camera, lidar) actually measure, noise and blind spots
included. The student trains *supervised*, which is the stable regime from
[`learnings/002`](learnings/002-no-warm-start-across-reward-change.md).

**The decision:** `legged_gym` uses ~187 points (denser foresight, bigger network,
slower). Suggest 25 as a light first pass. Whatever we pick sets the obs width
permanently, so picking 187 later would orphan every checkpoint trained at 25.

Note the first fresh run stays **blind** regardless (`height_enabled=False`) — the
slots are reserved so switching sight on later costs a config flag, not a retrain.

### 2. Do we pad Spyder and Hound to one shared obs width? — undecided

Recommendation above is no — lock them separately (Spyder 141, Hound 169) and
revisit on Isaac Lab.
