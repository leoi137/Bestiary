# Bestiary — legged robots built, trained, and documented in simulation

> 🚧 **Work in progress.** Functional and reproducible today, actively being
> polished. Expect frequent updates — issues and PRs welcome.

Legged robots authored as **code** — MJCF and URDF emitted by generator
scripts, never hand-written XML — and driven three different ways:

| | |
| --- | --- |
| **SAC from scratch** | one MuJoCo env, CPU-bound, millions of steps. The trainer in this repo. |
| **PPO at scale** | thousands of parallel envs in Isaac Lab, billions of steps, one GPU. |
| **Supervised imitation** | no reward and no environment — a causal transformer predicting the next entry of a recorded tape. |

Three machines of our own, one borrowed, and the stock Gymnasium benchmarks as
controls. Everything is written up as it happens, failed runs included.

| | |
| --- | --- |
| [**`research/`**](research/) | what each run taught us, which decisions are settled, and what would reverse them. **The weights are disposable; that folder is not.** |
| [**`docs/lessons/`**](docs/lessons/README.md) | start here if you are learning the field rather than following the project. One idea per page, from scratch, with the equation worked on a number this repo produced. |
| [**`ROADMAP.md`**](ROADMAP.md) | where this is going next |

---

## The same six commands, two different ways

<p align="center">
  <img src="assets/spyder_command_tour.gif" alt="Command tour: the velocity-commanded PPO policy driven through each command in isolation — forward, backward, side-steps, turns — with a full stop between each" width="396"/>
  <img src="assets/spot_ntp_tour.gif" alt="Command tour: the from-scratch causal transformer driving the quadruped through each command in isolation" width="396"/>
</p>

<p align="center">
  <em><strong>Left — Spyder-12, reinforcement learning.</strong> A 287K-parameter
  policy found by PPO over 2.06B simulated steps.<br/>
  <strong>Right — Spot, imitation.</strong> A 25.3M-parameter causal transformer
  trained for 11.5 minutes on a tape of somebody else's policy, with no reward
  and no environment in the loop.<br/>
  Same six commands, one at a time, with a commanded full stop between each:
  FORWARD · BACKWARD · SIDE-STEP LEFT/RIGHT · TURN LEFT/RIGHT · STOP.</em>
</p>

These two clips are the point of the repo. **They cost three orders of
magnitude apart and they look about the same.**

The left-hand policy was *searched for*: 2.06 billion steps of trial and error
against a reward table, ~10.7 hours on one GPU, and it is 287K parameters
because that is all a walking controller needs. The right-hand policy was
*copied*: 1,038 episodes of a pretrained Isaac Sim walker were recorded as
3.2 hours of `(observation, action)` tape at 50 Hz, rewritten as an interleaved
diary `o₀, a₀, o₁, a₁, …`, and a transformer trained from scratch to predict the
next entry — plain supervised learning to a best validation loss of **0.0013**.
Blind, both of them: proprioception in, joint targets out, nothing else.

On 12 held-out command scripts the transformer survives **12/12** and covers
**7.223 m** against the teacher's **7.215 m**; the same architecture with random
weights falls within two seconds. An adversarial refutation pass bounds what
that shows — *it walks from tapes*, not yet *it matches the teacher's gait*.

**Read more:** [the imitation method on one page](research/NTP_STAGE1_METHOD.md) ·
[the dataset contract](research/SPOT_ROLLOUTS_SPEC.md) ·
[lesson 014, the anatomy of the Spyder policy](docs/lessons/014-anatomy-of-the-spyder-policy.md)

---

## The bestiary

| Machine | DoF | Where it lives | Driven by | What it is for |
| --- | --- | --- | --- | --- |
| [**Spyder-12**](#spyder-12--the-12-dof-spider) | 12 | MuJoCo · Isaac Lab | SAC, then PPO | the one that gets trained — every algorithm lands here first |
| [**Hound-16**](#hound-16--the-16-dof-wheel-legged-dog) | 16 | MuJoCo · Isaac Lab | SAC, then PPO | the one that taught us how to measure — 11 of the 17 learnings came off it |
| [**Whelp-16**](#whelp-16--the-one-that-has-to-survive-a-floor) | 16 | a print bed | nothing yet | the one that answers what simulation cannot: *what actually breaks* |
| [**Spot**](#the-same-six-commands-two-different-ways) *(borrowed)* | 12 | Isaac Sim | supervised imitation | not ours — the machine the transformer learned to drive |
| [**Controls**](#controls--ant-v5-walker2d-v5-humanoid-v5) | — | MuJoCo | SAC | stock Gymnasium benchmarks; the calibration set, not robots |

---

### Spyder-12 — the 12-DoF spider

<p align="center">
  <img src="assets/spyder_walk_v3.gif" alt="SAC policy on the custom Spyder-v0 spider environment in MuJoCo" width="210"/>
  <img src="assets/spyder_isaac_forward.gif" alt="Spyder-12 crossing the demo ramp in Isaac Lab under a forward-velocity-only reward" width="374"/>
</p>

<p align="center">
  <em><strong>Left:</strong> SAC in MuJoCo, 3.75M steps, eval 7,392 — a ~6.5 m/s
  four-legged bound. The floor's checker texture is only drawn over an 80×80 m
  patch, so <strong>this</strong> spider outruns it; it is on the ground
  throughout.<br/>
  <strong>Right:</strong> PPO in Isaac Lab on the demo ramp, under a reward that
  is forward velocity and nothing else.</em>
</p>

This repo's own environment, and the machine every method gets tried on first.

**SAC, from scratch.** Reward-hacked twice — a jump-to-termination exploit, then
a cartwheel — and with both loopholes closed it walks upright at 3.2 m/s by
400K steps and bounds at ~6.5 m/s by 3.75M.

**PPO, at scale.** Ported to Isaac Lab it reaches 4–6 m/s in 1,500 iterations
(147M steps, 39 minutes) on a reward that is **forward velocity and nothing
else** — no shaping, so what it does is attributable to the stack rather than to
a reward table. That policy reads no command and holds no heading. Steering was
a separate arm, and it is the clip [at the top of this page](#the-same-six-commands-two-different-ways):
command-tracking reward plus three shaping terms, then fine-tuned from its own
checkpoint to a 2.5× wider speed envelope.

<p align="center">
  <img src="assets/spyder_shell_turntable.gif" alt="Turntable of the Blender-authored visual shell on Spyder-v0" width="250"/>
</p>

<p align="center">
  <em>Not a result — the Blender-authored visual shell. Decorative only; the
  collision geometry and the physics are unchanged.</em>
</p>

**Read more:** [`envs/spyder.py`](src/bestiary/envs/spyder.py) — the
reward-hacking postmortem is in its docstring ·
[episode 014, one term buys speed](research/episodes/014-one-term-buys-speed.md) ·
[episode 015, ten times the training bought survival](research/episodes/015-ten-times-bought-survival.md) ·
[episode 016, wider dials, same brain](research/episodes/016-wider-dials-same-brain.md)

---

### Hound-16 — the 16-DoF wheel-legged dog

<p align="center">
  <img src="assets/hound/preview.png" alt="HOUND-16 on the flat plane" width="390"/>
  <img src="assets/hound/desert.png" alt="HOUND-16 on the desert heightfield" width="390"/>
</p>

<p align="center">
  <em><code>Hound-v0</code> on the plane, and the identical robot on the
  <code>HoundDesert-v0</code> heightfield.</em>
</p>

Four legs of four joints each — abduction, hip, knee, and **a driven wheel where
the foot would be**. Link lengths, masses and torque limits are Unitree Go2's,
read off [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
so the mass distribution describes a machine that could exist; the wheel is ours,
since no vendor ships a wheel-legged MJCF. 17.0 kg, stands at 0.363 m, 169-dim
observation, 3.0 N·m at each wheel, and a 38-assertion oracle that fails if any
of that moves.

**It has no headline result, and that is worth saying plainly.** Five runs are in
the ledger and not one of them cleared this repo's ≥3-seed bar for a claim. What
it produced instead is the record: **11 of the 17 learnings came off this
robot** — that a peak score hides an unreliable policy, that the best checkpoint
is the luckiest episode, that a crash count was 0.9% of a gap we had already
attributed to crashes, that a policy had learned one trot rather than a command.
Every measurement discipline the rest of the repo now runs on was paid for here.

**Its first Isaac Lab run chose to gallop.** Given a reward with one term in it
and no instruction about how to move, the Hound ran on its legs and left its
wheels out of it. One seed, so this is a probe and not a finding — but it is the
first time this machine has been fast.

<p align="center">
  <img src="assets/hound_strip_chase.gif" alt="Hound-16 galloping up the demo strip in Isaac Lab under a forward-velocity-only reward, chase camera" width="620"/>
</p>

<p align="center">
  <em>The Hound's first Isaac Lab run — about 65 minutes of training from
  scratch, under a reward that is forward velocity and nothing else. It chose to
  <strong>gallop</strong> at 18–22 mph (8–10 m/s) rather than use its wheels,
  whose drives saturate at 2.0 mph (0.906 m/s) of rim speed. Shown charging 84%
  of the demo strip's difficulty gradient before the steepest face wins. Chase
  camera.</em>
</p>

**Read more:** [`CARD.md`](src/bestiary/robots/hound/CARD.md) — every dimension,
the solved stance, the spring sizing, and the traction budget that explains why
the hub motors are small ·
[learning 015, it learned one trot, not a command](research/learnings/015-it-learned-one-trot-not-a-command.md) ·
[learning 017, the second machine reproduced the stack, not the comparison](research/learnings/017-reproduced-the-stack-not-the-comparison.md)

---

### Whelp-16 — the one that has to survive a floor

<p align="center">
  <img src="assets/whelp/whelp16.png" alt="WHELP-16 skeleton at the solved stance, with its derived limits" width="720"/>
</p>

<p align="center">
  <em>The skeleton at its solved stance, annotated with the limits every number
  below is derived from — not a render, and not trained: a design.</em>
</p>

Hound's printable sibling: the same topology in 2.21 kg of PETG and brass,
229 mm standing, on twelve hobby serial-bus servos. It exists to answer the
question simulation cannot — *what actually breaks*. Every number is derived
rather than estimated, and the derivations are unkind: 0.21 m/s top speed,
4.7 rad/s of joint rate against the ~30 rad/s published legged-RL configs assume,
and a drop envelope measured in millimetres because a 1:345 gearbox is a solid
block on a 40 ms impact.

**Read more:** [`CARD.md`](src/bestiary/robots/whelp/CARD.md) — the yield chain,
the material choice, the print rules, and the sim-to-real settings that fail
silently

---

### Controls — Ant-v5, Walker2d-v5, Humanoid-v5

<p align="center">
  <img src="assets/baseline_2leg.gif" alt="Baseline SAC policy on Ant-v5" width="190"/>
  <img src="assets/foot_contact_v1.gif" alt="Foot-contact-shaped SAC policy on Ant-v5" width="190"/>
  <img src="assets/walker_baseline.gif" alt="Baseline SAC policy on Walker2d-v5" width="190"/>
  <img src="assets/humanoid_baseline.gif" alt="Baseline SAC policy on Humanoid-v5" width="190"/>
</p>

<p align="center">
  <em>Ant-v5 unshaped · Ant-v5 with the foot-contact term · Walker2d-v5 ·
  Humanoid-v5. All four are SAC on stock Gymnasium environments — the same
  trainer as Spyder's first arm, on problems with known answers.</em>
</p>

The Ant pair is the point of this row. Nothing in the stock reward says "use all
four legs", so SAC found a two-legged hop; `FootContactRewardWrapper`
([`rewards/shaping.py`](src/bestiary/rewards/shaping.py)) adds one term — count
the ankles that touched ground in the last 50 steps, penalise each idle leg — and
trades a little velocity for a gait that is actually quadrupedal. Walker2d and
Humanoid need no shaping: a biped has no degenerate gait to shape away.

---

## Every trained policy

Grouped by trainer, because **the scores are only comparable inside a group.**
A SAC eval return, a PPO iteration count and a validation loss do not rank
against each other.

### SAC — single MuJoCo env, this repo's trainer

| Run | Env | Reward | Steps | Best eval | Gait |
| --- | --- | --- | --- | --- | --- |
| `spyder_walk_v3` | `Spyder-v0` | default (Ant-style + upright termination) | 3.75M | 7,392 | ~6.5 m/s four-legged bound |
| `baseline_2leg` | `Ant-v5` | default | 3.75M | 6,657 (unshaped) | two legs only |
| `foot_contact_v1` | `Ant-v5` | default + foot-contact penalty | 3.75M | 5,647 (shaped) | uses all four |
| `walker_baseline` | `Walker2d-v5` | default | 3.75M | 5,944 | upright 2-legged walk |
| `humanoid_baseline` | `Humanoid-v5` | default | 4M | 6,458 | upright 3D bipedal walk |

```bash
venv/bin/python -m bestiary.train.watch --run <name>   # add --latest for the newest checkpoint
```

### PPO — Isaac Lab, thousands of parallel envs

| Run | Robot | Reward | Budget | What it showed |
| --- | --- | --- | --- | --- |
| `spyder_forward_s1` | Spyder-12 | forward velocity, one term | 1,500 iters · 147M steps · 39 min | locomotion emerges with no shaping at all — [ep. 014](research/episodes/014-one-term-buys-speed.md) |
| `spyder_ladder_s1` | Spyder-12 | one term added per arm, ×3 arms | 3 × 1,500 iters | which single shaping term buys the most gait — [ep. 015](research/episodes/015-ten-times-bought-survival.md) |
| `spyder_overnight_s1` | Spyder-12 | full command-tracking table | 15,000 iters | 10× the training bought survival, not tracking — [ep. 015](research/episodes/015-ten-times-bought-survival.md) |
| `spyder_fast_s1` | Spyder-12 | same table, 2.5× wider command envelope | +6,000 iters · 3.0 h (fine-tune) | **the command tour at the top of this page** — [ep. 016](research/episodes/016-wider-dials-same-brain.md) |
| `isaac_hound_arm1_s{1,2,3}` | Hound-16 | command tracking | 3 seeds × 1,500 iters | the stack reproduced; the comparison did not — [learning 017](research/learnings/017-reproduced-the-stack-not-the-comparison.md) |

**Single seed unless the run name says otherwise.** Under this repo's seed rule
a one-seed arm is a *probe*, never a finding — the episodes say so on their own
first line, and so does this table.

### Supervised imitation — no reward, no environment

| Run | Teacher | Data | Budget | Result |
| --- | --- | --- | --- | --- |
| `ntp_spot_s0` | pretrained Isaac Sim flat-terrain walker | 1,038 episodes · 3.2 h of tape @ 50 Hz | 11.5 min, 25.3M params | val loss 0.0013 · 12/12 held-out scripts · 7.223 m vs teacher 7.215 m |

### The real record

These tables are a showcase. The append-only ledger is the record:
[`research/ledger.jsonl`](research/ledger.jsonl) — one row per finished run,
every number computed from that run's own event files, with a verdict.
It currently carries five Hound rows that never made it to a headline, which is
[the point](#hound-16--the-16-dof-wheel-legged-dog).

---

## Install

```bash
python3.13 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e . --no-deps      # makes `bestiary` importable from anywhere
```

Python 3.13 · PyTorch 2.5.1 + CUDA 12.1 · Gymnasium 1.2 with MuJoCo 3.8 ·
Stable-Baselines3 2.8. GPU: NVIDIA GeForce RTX 2080.

> For CPU-only or another CUDA version, drop the `--extra-index-url` line from
> `requirements.txt` and follow <https://pytorch.org/get-started/locally/>.
> `--no-deps` is deliberate: re-resolving pinned versions can silently move one
> out from under a reproducible run. Call `venv/bin/python` directly — this venv
> was created elsewhere and moved, so `source venv/bin/activate` exports a stale
> `VIRTUAL_ENV` and leaves no `python` on `PATH`.

## Quick start

`--env` is given **once**, at creation, and pinned in that run's `config.json`
(default `Ant-v5`). `--steps` is per-invocation, not a cumulative target.

```bash
# fresh run — --env once, then never again
venv/bin/python -m bestiary.train.train --run-name walker_baseline --env Walker2d-v5 --seed 0 --steps 1_000_000

# resume — env, wrapper and seed come back from config.json, which WINS on conflict
venv/bin/python -m bestiary.train.train --run-name walker_baseline --steps 2_000_000

# reward shaping (foot_contact is Ant-only)
venv/bin/python -m bestiary.train.train --run-name my_shaped --seed 0 --steps 1_000_000 \
    --wrapper foot_contact --wrapper-kwargs '{"penalty": 1.0, "window": 50, "contact_threshold": 1.0}'
```

## Repo layout

An installable package under `src/`, so nothing depends on the current working
directory and there are no `sys.path` games.

| Path | Purpose |
| --- | --- |
| `src/bestiary/paths.py` | **every** filesystem path in the project resolves from here |
| `src/bestiary/train/` | `train.py` (train / resume SAC on any env), `watch.py` (render a run's policy) |
| `src/bestiary/envs/` | custom Gymnasium envs (`Spyder-v0`, `Hound-v0`, their `*Desert-v0` variants); importing registers them |
| `src/bestiary/ntp/` | next-token imitation — data, model, training loop |
| `src/bestiary/isaac/` | Isaac Lab / Isaac Sim tasks, env configs, recording and playback |
| `src/bestiary/robots/<name>/` | `build.py` (MJCF generator), `check.py` (assertions), `CARD.md` |
| `src/bestiary/rewards/`, `terrain/`, `guards/` | shaping wrappers · heightfield generate/read/hash · the lessons already paid for, as assertions |
| `research/`, `docs/` | the record; the teaching track, and the math when it becomes load-bearing |
| `assets/` | **generated** output — model XMLs, meshes, terrain, figures, README media. They stay here: MuJoCo resolves `<mesh>` and `<hfield>` paths relative to the XML's own directory |
| `runs/<name>/` | one self-contained experiment; gitignored, tens of GB |

---

## Appendix — reading a SAC run

`runs/<name>/` holds the latest and best-eval checkpoints, the replay buffer
(~2.6 GB, resume-only), TensorBoard events, one eval MP4 per `--video-every`
steps, and `config.json` — which **overrides** any conflicting
`--env`/`--wrapper`/`--seed` on resume, so reward semantics cannot change mid-run
and contaminate a buffer filled under different dynamics. The full set of
invariants is in [`CLAUDE.md`](CLAUDE.md).

```bash
venv/bin/tensorboard --logdir runs/     # then open http://localhost:6006
```

| Tag | What it means |
| --- | --- |
| `rollout/ep_rew_mean` | average episode return — the headline training curve |
| `rollout/ep_len_mean` | episode length; rises toward 1000 as the robot stops falling |
| `eval/mean_reward` | shaped return on the deterministic eval episode |
| `eval/base_reward` | **unshaped** reward — for apples-to-apples comparison across runs |
| `eval/mean_idle_legs` | avg legs with no recent ground contact (lower = better gait) |
| `train/actor_loss`, `train/critic_loss` | SAC actor and critic (Q) losses |
| `train/ent_coef` | auto-tuned entropy temperature α |

`eval/base_reward` and `eval/mean_idle_legs` only have data for runs that used a
wrapper — the baseline predates them.

**Roughly what SAC on `Ant-v5` does**, default reward: flailing to 50k steps
(returns near 0); standing, then shuffling, to 150k (500–1500); a recognisable
gait by 300k (2000–3500); smoother by 500k (3500–5500); "solved" past 1M
(~6000+). Foot-contact shaping tracks the same curve with base reward growing a
little slower — the policy has to explore four-legged gaits first.
