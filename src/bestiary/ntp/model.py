"""A small causal transformer over interleaved (obs, act) tokens.

The sequence is the robot's diary: `o_0, a_0, o_1, a_1, ...`, one linear
embedding per modality, causal attention over the lot. Two heads read the
hidden states:

- at an **obs** position (the model has just "sensed"): predict the action
  the teacher took — this head IS the controller at deployment;
- at an **act** position (the model has just "acted"): predict the next
  observation — the dynamics loss, arXiv:2402.19469's observation-prediction
  trick, which costs one linear layer and teaches the model what walking
  does to the world.

Sizing: d=512, 8 layers, 8 heads ≈ 25.4M parameters — inside the 10–50M
stage-1 envelope, trained from scratch, no pretrained anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NTPConfig:
    obs_dim: int = 48
    act_dim: int = 12
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    context: int = 32          # timesteps; token length is 2*context
    dropout: float = 0.0       # 630k pairs / 25M params underfits before it overfits


class Block(nn.Module):
    """Pre-LN transformer block; SDPA supplies the causal mask."""

    def __init__(self, cfg: NTPConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class NTPModel(nn.Module):
    def __init__(self, cfg: NTPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_in = nn.Linear(cfg.obs_dim, cfg.d_model)
        self.act_in = nn.Linear(cfg.act_dim, cfg.d_model)
        self.pos = nn.Embedding(2 * cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.act_head = nn.Linear(cfg.d_model, cfg.act_dim)
        self.obs_head = nn.Linear(cfg.d_model, cfg.obs_dim)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _trunk(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Interleave obs (B,t,48) and act (B,t or t-1,12) → hidden (B,tokens,d)."""
        b, t, _ = obs.shape
        tok_obs = self.obs_in(obs)
        tok_act = self.act_in(act)
        n_tok = t + act.shape[1]
        x = torch.empty(b, n_tok, self.cfg.d_model, device=obs.device, dtype=tok_obs.dtype)
        x[:, 0::2] = tok_obs
        x[:, 1::2] = tok_act
        if n_tok > self.pos.num_embeddings:
            raise ValueError(f"{n_tok} tokens > positional table {self.pos.num_embeddings}; grow context")
        x = x + self.pos.weight[:n_tok]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training pass.

        obs is (B, K+1, 48) — the extra row is the last dynamics target —
        act is (B, K, 12). Returns (act_pred (B,K,12), obs_pred (B,K,48)):
        act_pred[t] from the hidden at o_t, target act[t]; obs_pred[t] from
        the hidden at a_t, target obs[t+1].
        """
        k = act.shape[1]
        if obs.shape[1] != k + 1:
            raise ValueError(f"obs has {obs.shape[1]} rows for {k} actions; want K+1={k + 1}")
        h = self._trunk(obs[:, :k], act)
        return self.act_head(h[:, 0::2]), self.obs_head(h[:, 1::2])

    @torch.no_grad()
    def predict_next_action(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        """Deployment: t observations, t-1 actions (all normalized) → a_t (normalized).

        The eval harness owns normalization (the run's stats.json); this
        method is pure sequence-in, action-out.
        """
        t = obs_hist.shape[1]
        if act_hist.shape[1] != t - 1:
            raise ValueError(f"{t} obs need {t - 1} acts, got {act_hist.shape[1]}")
        h = self._trunk(obs_hist, act_hist)
        return self.act_head(h[:, -1])
