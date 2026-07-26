# 001 — Hound: what to fix next

Written 2026-07-25, while `hound_desert_v0` (3.75M, HoundDesert-v0) was running
overnight. Companion to `ROADMAP.md` (the *what*) and `CORE_PLAN.md` (the *how*).
This one is narrower: **what is actually blocking this robot right now.**

> **Editor's note, added 2026-07-25 after the repository refactor.** The body
> below is left exactly as written, per the rule in `README.md` that episodes
> are snapshots and are never edited to match how things turned out. Only the
> file paths it mentions have moved:
>
> | Then | Now |
> |---|---|
> | `make_hound.py` | `src/bestiary/robots/hound/build.py` |
> | `make_terrain.py` | `src/bestiary/terrain/generate.py` |
> | `envs/hound_env.py` | `src/bestiary/envs/hound.py` |
> | `CORE_PLAN.md` | `research/CORE_PLAN.md` |
>
> **Outcome:** the run finished at 3,750,000 steps in 8 h 05 m (129 steps/s),
> best eval 1218.3, final `ep_rew_mean` ≈ 1010, `ep_len_mean` ≈ 799. The
> standing check brackets that: 961 standing still, 1043 at speed. Landing
> between them is consistent with the ~75% prediction below — a survivor that
> mostly holds pose — not the 25% one. See `../ledger.jsonl`. The Isaac Lab
> vs. MJX question raised here is now settled in
> `../decisions/0001-defer-isaac-lab.md`.

## Thesis

The model is good. **The pipeline is the bottleneck.** No amount of further
reward tuning changes that, so the next work should be tooling, not terms.

## Diagnosis

**1. Throughput.** We run 124 steps/s in one env — 8.5 h for 3.75M, so roughly
four experiments a week. Unitree trains the machine in the reference video with
PPO at ~4096 parallel envs, order 1e9 samples in under an hour on one GPU. That
is a ~1000x gap in samples per wall-clock hour. Experiment *count*, not reward
quality, is what limits us.

**2. Torque control.** Our actions are raw joint torques, inherited from
Ant-v5's convention rather than chosen. Every production legged-RL stack —
`legged_gym`, `unitree_rl_gym`, Isaac Lab — has the policy emit **PD position
targets** with a stiff PD loop underneath at 200–1000 Hz. That is not cosmetic:
it means the policy never has to learn *how to hold a pose*, only where to put
the feet. Probably the single highest-leverage change available to us.

**3. Torque-to-weight.** 2.17 N·m/N, against the spider's 45.1 and Ant's ~8.8.
Go2's real masses AND Go2's real torque limits were both adopted for realism.
Realistic, but it makes the MVP ("just move forward") much harder than it needed
to be. Defensible for a portfolio piece; worth revisiting if it stalls.

## Ranked actions

| # | Change | Why | Cost |
|---|---|---|---|
| 1 | **PD position targets** instead of torques | Removes pose-holding from what the policy must learn; matches every production stack | Changes the action space → from scratch (we are anyway) |
| 2 | **MJX or Isaac Lab** | A night becomes minutes; makes weights genuinely disposable, which is CORE_PLAN's actual thesis | Real port; the biggest single investment here |
| 3 | Revisit torque-to-weight | Only if 1 and 2 do not close it | One number in `make_hound.py` |

Do **not** spend the next cycle on more reward iteration. The reward now passes
the standing check and shows a clean monotone gradient (961 → 1043 as speed
rises). It is not what is holding us back.

## Prediction to check against (falsifiable, on purpose)

Written before the 3.75M result was known:

- **~25%** that it is clearly moving forward across varied terrain.
- More likely: a crouching roller surviving 300–600 steps, covering 5–15 m on
  the flat basin, never reaching a dune.
- Expect a plateau somewhere in 500k–1.5M, with `ent_coef` collapsing early
  (it hit 0.011 by 57k, same as the 150k throwaway).

If the plateau shows up on schedule, that is evidence for the diagnosis above,
not against the reward. If it does not, revise this file.

## Open, from CORE_PLAN

- Does Hound share the spider's observation layout? Still undecided. The 28
  reserved slots (3 command + 25 height) are cut on the hound already, so the
  door is open either way.
- If we go to PD targets, the action space changes too — worth deciding the
  command spec at the same time so both robots move once, not twice.
