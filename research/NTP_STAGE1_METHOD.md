# Stage-1 next-token imitation — the method, on one page

*2026-08-07. Code: `src/bestiary/ntp/`. Data contract:
`research/SPOT_ROLLOUTS_SPEC.md`. This page is the interface card; change it
when the module changes.*

Supervised imitation of a recorded walking teacher. The robot's rollout is
written as an interleaved diary `o_0, a_0, o_1, a_1, …` and a causal
transformer learns to fill in the next entry. There is **no reward, no
environment interaction, and no exploration during training** — every
timestep already carries its correct answer.

## The RL-to-supervised dictionary

| RL habit | what it is here |
|---|---|
| observation space | **model input** — 48-dim proprioceptive vector per step (layout in the dataset spec: body-frame velocities, gravity, command, joint state, previous action), plus the 12-dim actions, interleaved over a 32-step context (0.64 s at 50 Hz) |
| action space | **model output / training target** — 12 continuous joint-position offsets, the teacher's raw network output; the controller applies `target = default_pose + 0.2 × action` (rad) |
| reward function | **does not exist.** Its role is played by the loss: `MSE(action) + 0.5 × MSE(next obs)`, on normalized tensors. The teacher's reward already shaped the *data*; this model never sees one |
| policy | the **act head at obs positions** — given the diary so far, emit the next action |
| dynamics model | the **obs head at act positions** — predict what the world does next (arXiv:2402.19469's observation-prediction trick; costs one linear layer) |
| rollout / episode | a **tape** from `runs/spot_rollouts/` (1,038 train-dir episodes, ~519k pairs) |
| eval | closed-loop: the transformer replaces the teacher as controller on held-out episodes — the only number that matters |

## The model, verbatim (`print(NTPModel(NTPConfig()))`)

```
NTPModel(
  (obs_in): Linear(in_features=48, out_features=512, bias=True)
  (act_in): Linear(in_features=12, out_features=512, bias=True)
  (pos): Embedding(64, 512)
  (blocks): ModuleList(
    (0-7): 8 x Block(
      (ln1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (qkv): Linear(in_features=512, out_features=1536, bias=True)
      (proj): Linear(in_features=512, out_features=512, bias=True)
      (ln2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (mlp): Sequential(
        (0): Linear(in_features=512, out_features=2048, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=2048, out_features=512, bias=True)
      )
    )
  )
  (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (act_head): Linear(in_features=512, out_features=12, bias=True)
  (obs_head): Linear(in_features=512, out_features=48, bias=True)
)
total parameters: 25,315,388
```

25.3M parameters — inside the 10–50M from-scratch envelope. GPT-style
pre-LN blocks, causal attention (`scaled_dot_product_attention`,
`is_causal=True`), learned positions over 64 tokens (32 timesteps × 2
modalities).

## Splits, restated once

`holdout/` (record-time, seed % 10 == 0) — closed-loop eval only, never
opened by training. Validation (seed % 10 == 1) — whole episodes, so val
loss measures cross-episode generalization. Normalization statistics — fit
episodes only, saved as the run's `stats.json`, read back at deployment.

## Result — run `ntp_spot_s0`, 2026-08-07 (single seed: a probe, not a finding)

Trained 20,000 steps in 11.5 min on one RTX 5070 Ti. Best validation loss
**0.0013** against a predict-the-mean baseline of 1.5 (exact by construction
of the normalization) — pre-registered bar was ≤ 0.30.

Closed-loop (`bestiary.isaac.play_ntp`), teacher vs transformer on 12
byte-identical holdout command scripts neither had seen:

| | teacher | transformer |
|---|---|---|
| survived | 12/12 | 12/12 |
| mean distance | 7.215 m | 7.223 m |

Per-episode |Δdistance| mean 0.065 m, max 0.179 m; transformer mean
\|v_x − cmd\| = 0.314 m/s. Numbers computed from
`runs/ntp_spot_s0/eval_{teacher,ntp}/results.jsonl` by the comparison
snippet in the session log; the demo clip is `assets/spot_ntp_drive.gif`.
Adversarial refutation of this result is still owed before any learning is
written from it — until then it is a probe.

## Refuted, 2026-08-07 — the result section above overstates, and here is how

The adversarial refutation pass ran and killed the headline claim. The
section above is left as written (the record does not silently rewrite);
this section names its errors.

1. **"Reproduces the teacher" is not established.** The discriminating
   check — teacher's own tracking error, computed from the holdout tapes —
   comes out 0.3168 m/s vs the transformer's 0.3137 m/s with per-episode
   r = 0.998: that metric is 99.7% determined by *which script ran* and
   0.3% by which controller drove. Distance is straight-line displacement,
   not path length, and survival has zero variance across both arms (the
   teacher's fall rate on flat ground is <0.26% over 1,154 episodes — an
   easy bar). The experiment separates *walks* from *falls over*; it cannot
   separate "the teacher's gait" from "any competent flat-ground walker."
2. **The baseline was wrong twice.** Measured predict-the-mean val loss is
   1.4697, not the "exact 1.5" (stats fit on the fit split, loss on val).
   And the honest non-learning baseline is *copy the previous action* —
   which sits verbatim in obs[36:48] — at **0.2257**: the true ratio is
   **169×**, not ~1100×. Still large; not what was printed.
3. **Data accounting:** the model trained on the fit split, 519,352 pairs =
   2.885 h, not the 3.2 h train directory; "1,038 episodes" belongs to the
   directory, "~519k pairs" to its 922-episode fit subset.
4. **The random control was weaker than it read:** 4 episodes vs 12, its
   checkpoint was not regenerated by committed code, and two of its four
   falls happened during the 1 s settle — 0.04 s of driving, not 2.1 s.
5. 12 episodes bound survival only at ≥0.78 (95%); the pre-registered
   prediction named all 116 holdout episodes.

**What stands, provisional (single seed):** a 25.3M from-scratch causal
transformer trained on ~2.9 h of tapes walks Spot closed-loop on flat
ground under velocity commands, and the identical architecture with random
weights falls immediately in the same harness. That, and nothing stronger.

**What would settle the strong claim:** per-step action agreement against
the recorded holdout tapes for both arms, path length instead of
displacement, all 116 holdout episodes, ≥3 training seeds, and the random
checkpoint generated by a committed script.

## Training defaults (`bestiary.ntp.train`)

AdamW, lr 3e-4, 500-step warmup then cosine, batch 64, 20k steps, grad-clip
1.0, weight decay 0.01. Fresh runs only; `config.json` + data fingerprint
written before step 0. Pure PyTorch — no simulator in the loop, so the
training box needs torch and 137 MB of tapes, nothing else.
