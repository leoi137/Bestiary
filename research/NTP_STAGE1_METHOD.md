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

## Training defaults (`bestiary.ntp.train`)

AdamW, lr 3e-4, 500-step warmup then cosine, batch 64, 20k steps, grad-clip
1.0, weight decay 0.01. Fresh runs only; `config.json` + data fingerprint
written before step 0. Pure PyTorch — no simulator in the loop, so the
training box needs torch and 137 MB of tapes, nothing else.
