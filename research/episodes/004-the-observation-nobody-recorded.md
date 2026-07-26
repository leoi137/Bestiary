# 004 — The observation nobody recorded

**Date:** 2026-07-26 · **Robot:** hound + spyder · **Run:** none (no training)

## Thesis

The observation list is documented here as the one truly one-way door: change
it and every existing checkpoint does not degrade, it fails to load. That
lesson is written down as [`learnings/003`](../learnings/003-obs-list-is-a-one-way-door.md)
and enforced by a guard. The door was walked through anyway.

The interesting part is not that it happened. It is that **finding out** took
git archaeology — because no run in this repository records which observation
it was trained against.

## Diagnosis

`checkpoint-width` reports that `hound_desert_test150k`'s checkpoints are
141 obs against a 169-obs `HoundDesert-v0`. The recorded explanation was that
"the hound obs went 141 → 169" after that run.

That is not what happened.

```
git log --oneline --all -S "_n_command"
  7bd83ff Add Hound-v0 / HoundDesert-v0: wheel-aware observation and reward

git log -1 --format='%h %ad' --date=iso 7bd83ff
  7bd83ff 2026-07-25 03:22:23 -0500

stat -c '%y %n' runs/hound_desert_test150k/ant_sac.zip
  2026-07-25 03:02:03 ...
```

There is exactly one commit in which the hound env exists, and it *created* the
env already carrying `_n_command = 3` and `_n_height = 25`. The run's last
checkpoint was written **twenty minutes earlier**. The env was never committed
at 141. The width did not change under the run; the run predates the env's only
commit, and both states were squashed into it.

So the checkpoint is permanently unloadable and its spec is unrecoverable —
there is no commit to restore it from.

The same archaeology settles the second orphan. `runs/ant12` declares
`env_id: "Ant12-v0"`, and:

```
git log --oneline --all -S "Ant12"
  63d9202 Add the calibration, null-result, and anomaly records
```

The only place `Ant12` has ever appeared in 80 commits is the anomaly record
describing it. The id was never registered here. The run is from 2026-06-22,
predating the package restructure, and its checkpoint is 89 obs / 12 act — a
morphology matching no XML in `assets/`. There is nothing to restore.

**What generalizes past these two runs.** `config.json` stores `env_id`,
`algo`, `wrapper`, hyperparameters and `seed`. It stores no width, no term
list, no hash. The only surviving evidence of a run's observation is the
pickled space width inside its checkpoint, so the only way to detect a change
is to attempt a load and catch the exception. That is an autopsy, not an
instrument.

And it is blind in the direction that matters most. A width change at least
fails loudly at `SAC.load()`. A change that *preserves* the width — reordering
two terms, redefining what a term means — loads cleanly and silently feeds the
policy a permuted world. Nothing in this repository could see that. It was
structurally invisible, because each env stated its observation twice: a width
formula in `__init__` and a `np.concatenate` in `_get_obs`, tied together by
nothing, with MuJoCo's base class only *warning* when the built vector
disagrees with the declared space.

## What happened

**One declaration per env.** `envs/obs_spec.py` holds an ordered list of named,
sized terms. It sizes the `Box`, `_get_obs` validates against it and **raises**
with the per-term breakdown rather than warning, and it hashes.

The hash is the new capability. Verified behaviour, all at identical width:

| change | hash |
|---|---|
| baseline | `b00c044a4940d4cc` |
| two terms reordered | **differs** |
| a term renamed | **differs** |
| one 78-wide block split into 40 + 38 | **differs** |
| a term's comment reworded | **same** |

That last row is deliberate. A hash that moves when prose moves is a hash
people learn to ignore.

Hound's 28 reserved slots are declared as **two** terms — `command_reserved(3)`
and `height_reserved(25)` — rather than one block of 28, so that changing the
split (say, to `legged_gym`'s 187 height samples) moves the hash even though a
single 28-wide term would have concealed it.

**The spec is now recorded and enforced at the boundary.** A fresh run pins its
spec into `config.json` before the first step; a resume compares and raises,
printing both specs and both widths. Verified end to end: a 2000-step run
pinned `11093686ef09fe13`, an honest resume matched, and a config edited to
mimic a changed observation list **exited 1 before a single step was taken**.

**`checkpoint-width` now asserts the recorded hash too**, and reports the nine
runs that predate the record instead of skipping them — a silent skip is how a
guard claims coverage it does not have. Those runs are not back-filled;
inventing a spec for a finished run from today's code is exactly the false
provenance this exists to prevent.

Nothing was softened. The suite still reports **exactly 12 failing assertions**,
unchanged.

## Measurements

A second instrument landed: `record/greedy_eval.py`, which measures a trained
policy the same way the do-nothing control is measured — same env, same episode
count, the same seeds, deterministic on both sides. `guards/standing.py`
compares a *training* rollout mean (exploration noise on, 799-step episodes)
against a *deterministic* 1000-step zero-action rollout, and never states that
it chose.

5 episodes, seeds 0–4, `ant_sac_best.zip`:

| run | greedy mean | sd | zero-action | ratio | below standing | died |
|---|---|---|---|---|---|---|
| `spyder_desert_v0` | 540.0 | 245.5 | 985.9 | **0.548** | 5/5 | 2/5 |
| `hound_desert_v0` (torque, 3.75M) | 1214.8 | **5.4** | 960.2 | **1.265** | 0/5 | 0/5 |
| `hound_pd_desert_v0` (PD, 1M) | 977.2 | **441.0** | 955.3 | **1.023** | 1/5 | 1/5 |

Two things follow, and they point in opposite directions from the record.

**Doing nothing beats the trained policy on Spyder, not on the hound.** The
claim as carried in the ladder — "on both robots" — came from
`hound_desert_test150k` at `ctrl_cost_weight` 0.1 and was never re-measured
after the weight dropped to 0.01. Spyder is genuinely inverted at 0.548.

**The zero-action baseline is the most precise number in this repository** —
sd 2.4, 0.9 and 6.1, surviving 1000 of 1000 steps every seed. It is not the
source of any disagreement.

**The failures are bimodal, not a uniform creep.** Spyder scores ~700 or dies
at 66–211 steps. The PD hound scores ~1175 four times and 188.3 once, dying at
61 steps. A mean alone describes neither.

## How the prediction did

Committed before the work, in
[`calibration.jsonl`](../calibration.jsonl):

| # | claim | p | outcome |
|---|---|---|---|
| C1 | Spyder greedy still loses to zero action (ratio < 1.0) | 0.75 | **true** — 0.548, inside all three stated bands |
| C2 | refactor changes no width (113 / 169) | 0.97 | **true** — all six env ids, oracles 38/38 and bit-for-bit |
| C3 | suite still reports exactly 12 failures | 0.70 | **true** — exactly 12 |
| C4 | the manifest exposes a latent inconsistency | 0.35 | **false** |

C4 is the useful one. The two definition sites were **duplicated, not
divergent** — both envs agreed exactly, and both module docstrings' index
ranges check out (Spyder 17/18/78; Hound 17/22/102, giving the documented
`[39:141]` contact block). The structural hazard was real and the drift had not
happened yet. Called at 0.35 and correctly not taken.

C1 was right about the number and incomplete about the mechanism: nothing in
the prediction anticipated that Spyder *dies* in 2 of 5 episodes.

Brier over 7 resolved rows: 0.0787, from 0.0917. Still too few to read as skill.

## Ranked actions

1. **A `toolchain` guard.** This venv was created as `GymMuJoCo/venv` and
   moved, so `source venv/bin/activate` exports a `VIRTUAL_ENV` outside the
   repository and leaves no `python` on `PATH`. Every documented command was
   corrected this cycle — including the lint line, which is marked REQUIRED
   after any refactor and exited 127 — but nothing stops it regressing. ~40
   lines, no GPU.
2. **Stop defaulting missing metrics to 0.0.** `eval/mean_idle_legs` reads
   exactly 0.0 in 7 runs, and the same bug makes `eval/base_reward` a perfect
   duplicate of `eval/mean_reward` in 8 of 9 — a signature `metric-liveness`
   cannot see, because it tests for constants.
3. **Give the reward the same treatment the observation just got.** `config.json`
   records no reward parameters, so a ledger row's numbers cannot be attributed
   to a reward. The machinery landed this cycle.
4. **Re-derive the 1.18 healthy margin from a measurement, or mark it
   unverified.** Its cited source is a hand-computed table about a different
   robot at a `ctrl_cost_weight` no env here uses.

## Open questions

- **Nothing reads `nulls.jsonl`.** No path constant, no schema, no reader. Its
  row 2 says *do not repeat unless `ctrl_cost_weight` is lowered to 0.02 **and**
  the forward term is replaced by command tracking*. Both hound ledger runs
  lowered the weight and neither touched the forward term. A recorded dead end
  was re-entered at half its condition, twice, for ~10.7 GPU-hours. What is the
  cheapest check that would have fired?
- **Is the PD run's catastrophic episode reproducible or a one-off?** One
  episode in five, on one policy, at a different step budget from its
  comparison arm.
- **`Spyder` has no reserved slots and Hound has 28.** One robot's observation
  is locked and the other's is not.
