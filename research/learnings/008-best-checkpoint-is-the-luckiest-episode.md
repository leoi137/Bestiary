---
number: 008
title: The best checkpoint is the luckiest episode, not the best policy
date: 2026-07-26
from: runs/hound_desert_v0, runs/hound_pd_desert_v0, runs/spyder_desert_v0
supersedes: none
extends: 007-peak-score-hides-an-unreliable-policy.md
guard: src/bestiary/guards/eval_sampling.py
triggers:
  - comparing two runs by best_eval_return, or by any *_best.zip
  - writing a ledger row's best_eval_return
  - deciding a policy is "stable" or "reliable"
  - choosing which checkpoint to watch, evaluate, or deploy
  - reporting a result from fewer than ~20 evaluation episodes
---

# The best checkpoint is the luckiest episode, not the best policy

## The belief this overturned

[`Learning 007`](007-peak-score-hides-an-unreliable-policy.md) established that
a peak eval score hides an unreliable policy, and the response was to stop
*quoting* peaks. That treated it as a reporting problem.

It is not only a reporting problem. The peak is not merely written down — it is
used to **select the artifact we keep**. Every `ant_sac_best.zip` in this
repository is the snapshot that produced the single highest-scoring evaluation
episode, and "highest-scoring single episode" is very nearly a synonym for
"luckiest".

## The mechanism

`train/train.py`, `VideoEvalCallback._record_one_episode`, does exactly what its
name says:

```python
def _record_one_episode(self) -> None:
    obs, _ = self.eval_env.reset()
    ...
    while not (terminated or truncated):
        action, _ = self.model.predict(obs, deterministic=True)
    ...
    self.logger.record("eval/mean_reward", total_reward)
    ...
    if total_reward > self.best_eval_reward:
        self.model.save(self.best_model_path)
```

Three things are wrong and they compound:

1. **One episode per evaluation.** No seed is passed, no averaging is done.
2. **That single return is logged as `eval/mean_reward`.** It is not a mean of
   anything. The name asserts a sample size the code does not take.
3. **`best_model` is `argmax` over those single draws.** Saved whenever one
   episode beats the stored record.

So the checkpoint we keep, watch, and evaluate is selected by a maximum over a
noisy one-sample statistic — and the noise is not small.

## The math, on this repo's numbers

These policies are **bimodal**: they either complete the 1000-step episode or
fail early. Measured over 60 deterministic episodes each (three disjoint seed
blocks, `record/greedy_eval.py`; arithmetic in
`research/scripts/learning_008_math.py`):

| policy | failures / 60 | rate |
|---|---|---|
| `hound_desert_v0` (torque) | 16 | **26.7%** |
| `hound_pd_desert_v0` (PD) | 6 | **10.0%** |
| `spyder_desert_v0` | 57 below standing | **95.0%** |

Take the torque hound. Each evaluation is one Bernoulli draw with failure
probability *p* = 0.267. The chance a *given* evaluation lands in the good mode
is 0.733. Its run evaluated 14 times, and the saved `best` is the maximum over
those 14 draws — so the probability that at least one landed in the good mode,
and therefore that `best` is a good-mode snapshot, is

```
P(at least one good draw in 14) = 1 − 0.267**14 = 1 − 9.195e-09 ≈ 1.000000
```

The selection is not merely biased; at 14 evaluations it is **guaranteed** to
pick a good-mode episode. `ant_sac_best.zip` therefore reports the policy's
*best mode*, never its mixture, no matter how often the policy actually fails.

The same arithmetic is why small samples mislead downstream. Five episodes of a
26.7%-failure policy come back clean

```
P(0 failures in 5) = 0.733**5 = 0.2121
```

— once in five tries. And the specific misleading picture that prompted this
lesson (torque clean at 0/5, PD showing its failure at ≥1/5) had probability

```
0.7333**5 × (1 − 0.9000**5) = 0.2121 × 0.4095 = 0.0869
```

about **one in twelve**. That draw was measured, believed, and written up as
overturning a published result, and an independent refutation at n=60 reversed
it: torque's true ratio is x1.042 with 16/60 failures, PD's x1.128 with 6/60,
Fisher exact p = 0.032 — which is the *original* ranking in
[`episodes/003`](../episodes/003-pd-result-cheaper-not-higher.md), confirmed.

## Why this was reasonable and still wrong

Saving on improvement is the standard pattern and it is correct when the
evaluation is low-variance. `VideoEvalCallback` exists to record a *video*, and
one episode is the right amount of video. Reusing that same rollout to select a
checkpoint was free, obvious, and never revisited — the cost only appears once
a policy becomes bimodal, which is exactly what happens on terrain.

## What this changes

- `best_eval_return` in a ledger row is a **maximum over single-episode draws**,
  not a policy quality. It cannot rank two runs, and it grows with the number of
  evaluations — so a longer run scores higher for free.
- `*_best.zip` is not "the best policy". Evaluating one measures a policy's good
  mode. `--latest` is the unbiased snapshot, and the honest comparison uses both.
- A reliability claim needs a **crash rate over ~20+ episodes**, reported with
  its n. A mean alone describes neither mode of a bimodal policy.

## How we would know this is wrong

- **If the policies are not bimodal on some future env**, the selection bias
  collapses to ordinary noise and the effect stops mattering. Test: episode
  lengths cluster at 1000 with no early-termination mode.
- **If `best` and `latest` checkpoints score the same** over 20+ episodes, then
  argmax selection is not buying a good-mode snapshot and this lesson is
  overstated. The refuter already saw movement in this direction: on `--latest`
  the torque/PD failure gap nearly vanished (4/20 vs 3/20).
- **If a run's evaluation count stops predicting its `best_eval_return`** across
  several runs, the "longer run scores higher for free" claim is wrong.

## The guard

`guards/eval_sampling.py` asserts that any run recording a `best_eval_return`
also records how many episodes that number came from, and that the count is
above 1. It cannot retroactively fix the nine existing runs — it names them,
the same way `checkpoint-width` names the runs predating the observation spec.
