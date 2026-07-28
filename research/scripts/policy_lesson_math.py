"""Arithmetic behind docs/lessons/008 — what a policy is, and why it is a network.

Every number the lesson quotes is computed here. Two sources:

  1. a real trained checkpoint (runs/hound_pd_desert_v0/ant_sac.zip), read
     layer by layer for the actor's actual parameter count;
  2. the lookup-table comparison, which is pure arithmetic on the observation
     width.

Run:  venv/bin/python research/scripts/policy_lesson_math.py
"""

from __future__ import annotations

import zipfile
from decimal import Decimal
from math import log10

import torch

from bestiary.paths import RUNS

CKPT = RUNS / "hound_pd_desert_v0" / "ant_sac.zip"

# Estimated number of atoms in the observable universe, order of magnitude.
# Source: baryon count ~1e80, the standard textbook figure.
ATOMS_OBSERVABLE_UNIVERSE = Decimal(10) ** 80


def actor_parameters() -> dict:
    """Count the actor's parameters directly from the saved checkpoint."""
    with zipfile.ZipFile(CKPT) as z:
        with z.open("policy.pth") as f:
            state = torch.load(f, map_location="cpu", weights_only=False)

    actor = {k: v for k, v in state.items() if k.startswith("actor.")}
    if not actor:
        raise SystemExit(f"no actor.* tensors in {CKPT}")

    rows = [(k, tuple(v.shape), v.numel()) for k, v in actor.items()]
    total = sum(n for _, _, n in rows)

    obs_dim = actor["actor.latent_pi.0.weight"].shape[1]
    hidden = actor["actor.latent_pi.0.weight"].shape[0]
    act_dim = actor["actor.mu.weight"].shape[0]
    return {
        "rows": rows,
        "total": total,
        "obs_dim": obs_dim,
        "hidden": hidden,
        "act_dim": act_dim,
        "mu_out": act_dim,
        "log_std_out": actor["actor.log_std.weight"].shape[0],
    }


def lookup_table_cells(obs_dim: int, bins: int) -> Decimal:
    """Cells in a table that discretises each of obs_dim inputs into `bins`."""
    return Decimal(bins) ** obs_dim


def main() -> None:
    a = actor_parameters()

    print(f"checkpoint: {CKPT}")
    print(f"observation width : {a['obs_dim']}")
    print(f"actuators (mu out): {a['act_dim']}   log_std out: {a['log_std_out']}")
    print(f"hidden width      : {a['hidden']}")
    print()
    print("actor tensors:")
    for k, shape, n in a["rows"]:
        print(f"  {k:34s} {str(shape):14s} {n:>8,}")
    print(f"  {'TOTAL':34s} {'':14s} {a['total']:>8,}")
    print()

    # The lookup table the network replaces.
    for bins in (2, 10):
        cells = lookup_table_cells(a["obs_dim"], bins)
        print(
            f"lookup table, {bins:2d} bins per input: "
            f"{bins}^{a['obs_dim']} = 1e{log10(cells):.1f} cells, "
            f"1e{log10(cells / ATOMS_OBSERVABLE_UNIVERSE):+.1f} x the "
            f"~1e80 atoms in the observable universe"
        )
    print()

    cells10 = lookup_table_cells(a["obs_dim"], 10)
    print(
        "cells (10 bins) per actor parameter: "
        f"1e{log10(cells10 / Decimal(a['total'])):.1f}"
    )

    # What the network outputs: mean + log-std per actuator.
    print()
    print(f"outputs per step  : {a['mu_out']} means + {a['log_std_out']} log-stds "
          f"= {a['mu_out'] + a['log_std_out']} numbers")
    print("eval action       : tanh(mu)   (SquashedDiagGaussianDistribution.mode)")


if __name__ == "__main__":
    main()
