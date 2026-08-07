"""Train the next-token imitator on recorded rollout tapes.

    venv/bin/python -m bestiary.ntp.train --run-name ntp_spot_s0 --seed 0
    venv/bin/python -m bestiary.ntp.train --run-name smoke --steps 25 --device cpu

Fresh runs only — this trainer refuses a run directory that already has a
config.json, because silently resuming with different arguments is how two
half-runs get averaged into one unexplainable checkpoint. Delete the
directory or pick a new name.

What lands in runs/<run-name>/:
    config.json   every argument + the data fingerprint, written BEFORE step 0
    stats.json    normalization statistics (fit split only) — part of the model
    log.csv       step, losses, val, lr, wall-clock; the ledger reads this
    ntp_best.pt   lowest-validation-loss checkpoint (weights + config)
    ntp_latest.pt overwritten at every validation

Loss = action MSE + OBS_LOSS_WEIGHT * observation MSE, both on normalized
tensors. The action term is the controller; the observation term is
arXiv:2402.19469's dynamics trick. Validation is whole-episode-held-out
(seed % 10 == 1), so a fall in val loss means generalization across
episodes, not across adjacent timesteps.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bestiary.paths import RUNS

from .data import WindowDataset, compute_stats, load_episodes, split_fit_val
from .model import NTPConfig, NTPModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--data", type=Path, default=RUNS / "spot_rollouts")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--context", type=int, default=32, help="timesteps of history (0.64 s at 50 Hz)")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--obs-loss-weight", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=50)
    return p.parse_args()


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine to zero — the boring default that works."""
    if step < warmup:
        return base * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * frac))


def main() -> None:
    args = parse_args()
    run_dir = RUNS / args.run_name
    if (run_dir / "config.json").exists():
        raise SystemExit(f"{run_dir} already has a config.json — fresh runs only; pick a new name")
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    episodes = load_episodes(args.data)
    fit, val = split_fit_val(episodes)
    stats = compute_stats(fit)
    fit_ds = WindowDataset(fit, stats, args.context)
    val_ds = WindowDataset(val, stats, args.context)
    print(
        f"data: {len(fit)} fit / {len(val)} val episodes, "
        f"{stats['fit_pairs']} fit pairs, {len(fit_ds)} fit windows"
    )

    cfg = NTPConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, context=args.context
    )
    model = NTPModel(cfg).to(args.device)
    print(f"model: {model.n_params() / 1e6:.1f}M params, context {args.context} steps")

    (run_dir / "stats.json").write_text(json.dumps(stats))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "n_params": model.n_params(),
                "data_fingerprint": {
                    "episodes": len(episodes),
                    "fit_pairs": stats["fit_pairs"],
                    "seeds": [min(e.seed for e in episodes), max(e.seed for e in episodes)],
                },
            },
            indent=2,
        )
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loader = DataLoader(
        fit_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, drop_last=True)

    def batch_loss(obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        obs, act = obs.to(args.device), act.to(args.device)
        act_pred, obs_pred = model(obs, act)
        l_act = torch.nn.functional.mse_loss(act_pred, act)
        l_obs = torch.nn.functional.mse_loss(obs_pred, obs[:, 1:])
        return l_act + args.obs_loss_weight * l_obs, l_act.item(), l_obs.item()

    log = open(run_dir / "log.csv", "w")
    log.write("step,loss,act_loss,obs_loss,val_loss,lr,wall_s\n")
    best_val = float("inf")
    t0 = time.monotonic()
    step, data_iter = 0, iter(loader)

    while step < args.steps:
        try:
            obs, act = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            continue
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        loss, l_act, l_obs = batch_loss(obs, act)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: act {l_act}, obs {l_obs}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        step += 1

        val_loss = ""
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vl = [batch_loss(o, a)[0].item() for (o, a), _ in zip(val_loader, range(args.val_batches))]
            model.train()
            val_loss = float(np.mean(vl))
            payload = {"model": model.state_dict(), "config": vars(cfg), "step": step, "val_loss": val_loss}
            torch.save(payload, run_dir / "ntp_latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(payload, run_dir / "ntp_best.pt")
            print(f"step {step:>6}  loss {loss.item():.4f}  val {val_loss:.4f}  best {best_val:.4f}")
        elif step % 100 == 0:
            print(f"step {step:>6}  loss {loss.item():.4f}  act {l_act:.4f}  obs {l_obs:.4f}")
        log.write(f"{step},{loss.item():.6f},{l_act:.6f},{l_obs:.6f},{val_loss},{lr:.2e},{time.monotonic() - t0:.1f}\n")

    log.close()
    print(f"done: {step} steps in {(time.monotonic() - t0) / 60:.1f} min, best val {best_val:.4f} -> {run_dir}")


if __name__ == "__main__":
    main()
