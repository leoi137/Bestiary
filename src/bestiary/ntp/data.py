"""Load recorded rollout tapes into training windows.

The on-disk contract is `research/SPOT_ROLLOUTS_SPEC.md`; this module holds
the three decisions that sit between those files and a batch:

**Holdout stays on disk.** Only `<data>/train/` is ever read here. The
`holdout/` directory belongs to the closed-loop eval and is not opened by the
training path — not even for normalization statistics.

**Validation is split by seed, not by window.** Episodes with
`seed % 10 == 1` are the validation set (holdout already took `% 10 == 0` at
record time). Splitting at the window level would put timestep t of an
episode in train and t+1 in val, which measures memorization, not learning.

**Normalization statistics come from the fit episodes only** and are saved
next to the checkpoint. A model's statistics are part of the model: the
classic silent killer in imitation is training on normalized tensors and
deploying on raw ones (or on statistics recomputed over different data), so
the stats travel with the run, and the eval harness must read them back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

#: Interface contract, asserted against every episode's own metadata.
#: From research/SPOT_ROLLOUTS_SPEC.md (which the recorder asserts at boot).
OBS_DIM = 48
ACT_DIM = 12
POLICY_RATE_HZ = 50.0

#: Validation split: seeds ending in this digit. Holdout took 0 at record
#: time; 1 gives ~10% of what remains, deterministically, with no overlap.
VAL_SEED_DIGIT = 1

#: Standard-deviation floor. A constant obs dimension (e.g. gravity z on
#: flat ground barely moves) would otherwise divide by ~0 and explode.
STD_FLOOR = 1e-6


@dataclass
class Episode:
    seed: int
    obs: np.ndarray  # (T, 48) float32, raw SI
    act: np.ndarray  # (T, 12) float32, raw net output


def load_episodes(data_dir: Path) -> list[Episode]:
    """Read every training tape, asserting the contract per file."""
    train_dir = data_dir / "train"
    paths = sorted(train_dir.glob("ep_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episodes under {train_dir} — record first")

    episodes: list[Episode] = []
    engines: set[str] = set()
    for p in paths:
        with np.load(p) as d:
            meta = json.loads(str(d["meta"]))
            obs, act = d["obs"], d["act"]
        if obs.shape[1] != OBS_DIM or act.shape[1] != ACT_DIM:
            raise ValueError(f"{p}: obs {obs.shape} act {act.shape}, expected (*, {OBS_DIM})/(*, {ACT_DIM})")
        if abs(meta["policy_rate_hz"] - POLICY_RATE_HZ) > 1e-6:
            raise ValueError(f"{p}: rate {meta['policy_rate_hz']} Hz, expected {POLICY_RATE_HZ}")
        if not (np.isfinite(obs).all() and np.isfinite(act).all()):
            raise ValueError(f"{p}: non-finite values in tape")
        engines.add(meta["engine"])
        episodes.append(Episode(seed=int(meta["seed"]), obs=obs, act=act))
    if len(engines) != 1:
        # Spec rule 4: PhysX and Newton are different dynamics, never mixed silently.
        raise ValueError(f"episodes span physics engines {sorted(engines)} — split them, do not mix")
    return episodes


def split_fit_val(episodes: list[Episode]) -> tuple[list[Episode], list[Episode]]:
    fit = [e for e in episodes if e.seed % 10 != VAL_SEED_DIGIT]
    val = [e for e in episodes if e.seed % 10 == VAL_SEED_DIGIT]
    if not fit or not val:
        raise ValueError(f"degenerate split: {len(fit)} fit / {len(val)} val episodes")
    return fit, val


def compute_stats(fit: list[Episode]) -> dict:
    obs = np.concatenate([e.obs for e in fit])
    act = np.concatenate([e.act for e in fit])
    return {
        "obs_mean": obs.mean(0).tolist(),
        "obs_std": np.maximum(obs.std(0), STD_FLOOR).tolist(),
        "act_mean": act.mean(0).tolist(),
        "act_std": np.maximum(act.std(0), STD_FLOOR).tolist(),
        "fit_episodes": len(fit),
        "fit_pairs": int(sum(len(e.act) for e in fit)),
        "val_seed_digit": VAL_SEED_DIGIT,
    }


class WindowDataset(Dataset):
    """Fixed-length windows that never cross an episode boundary.

    A window at (episode, start s) yields normalized
    `obs[s : s+K+1]` (K+1 rows) and `act[s : s+K]` (K rows): the model
    consumes obs[:K] and act[:K] interleaved, predicts act[t] at obs
    positions (the policy) and obs[t+1] at act positions (the dynamics) —
    the K+1th obs row exists so every act position has a target.
    """

    def __init__(self, episodes: list[Episode], stats: dict, context: int) -> None:
        self.context = context
        self.obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(stats["obs_std"], dtype=np.float32)
        self.act_mean = np.asarray(stats["act_mean"], dtype=np.float32)
        self.act_std = np.asarray(stats["act_std"], dtype=np.float32)
        self.episodes = [e for e in episodes if len(e.act) >= context + 1]
        dropped = len(episodes) - len(self.episodes)
        if not self.episodes:
            raise ValueError(f"no episode is >= {context + 1} steps; shorten --context")
        if dropped:
            print(f"windowing: dropped {dropped}/{len(episodes)} episodes shorter than {context + 1} steps")
        self._index: list[tuple[int, int]] = [
            (i, s) for i, e in enumerate(self.episodes) for s in range(len(e.act) - context)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, s = self._index[idx]
        e, k = self.episodes[i], self.context
        obs = (e.obs[s : s + k + 1] - self.obs_mean) / self.obs_std
        act = (e.act[s : s + k] - self.act_mean) / self.act_std
        return torch.from_numpy(obs), torch.from_numpy(act)
