"""Guard: doing nothing must not outscore the trained policy.

Enforces `research/learnings/001` (a reward tuned on flat ground breaks on
terrain) and `research/learnings/005` (the standing check caught it again, on a
different robot, from scratch).

This is the cheapest real check this project owns, and it has already caught a
fatal reward bug twice — on the spider after 5.75M steps, and independently on
the hound from scratch. It rolls a zero-action policy and compares the return
against what the ledger says the trained policy achieved. If standing still
wins, the reward is wrong and no amount of training will fix it.

No torch and no GPU: it steps the env with `action = 0`, which is the whole
point — the control cost of doing nothing is exactly zero, and that is the
loophole being tested for.
"""
from __future__ import annotations

import json

import numpy as np

from bestiary import paths
from bestiary.guards import Finding

EPISODES = 3
SEED = 0

# Two thresholds, and neither is invented.
#
# Beating standing at all is the hard floor: below it the reward is literally
# inverted, which is what learnings 001 and 005 record.
#
# The advisory margin is 1.18, taken from CORE_PLAN.md's own arithmetic rather
# than chosen: with the control cost corrected to 0.02, the same desert gait
# scores 1164 against standing's 987, i.e. 1.18x. A policy under that ratio is
# not obviously inverted, but it is not being paid for locomotion either — it
# is being paid for staying alive, and it will behave accordingly.
HEALTHY_MARGIN = 1.18


def _ledger_rows() -> dict[str, dict]:
    if not paths.LEDGER.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in paths.LEDGER.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["run"])] = row
    return rows


def _zero_action_return(env_id: str) -> tuple[float, float]:
    """Mean return and mean episode length of the do-nothing policy."""
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids

    env = gym.make(env_id)
    try:
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        returns, lengths = [], []
        for episode in range(EPISODES):
            _, _ = env.reset(seed=SEED + episode)
            total, steps = 0.0, 0
            while True:
                _, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(steps)
        return float(np.mean(returns)), float(np.mean(lengths))
    finally:
        env.close()


def run() -> list[Finding]:
    rows = _ledger_rows()
    if not rows:
        return [Finding("ledger has rows to check", True, "empty ledger — nothing to do")]

    findings: list[Finding] = []
    for name, row in rows.items():
        env_id = str(row.get("env_id", ""))
        trained = row.get("final_ep_rew_mean")
        if not env_id or not isinstance(trained, (int, float)):
            findings.append(
                Finding(f"{name}: row has env_id and final_ep_rew_mean", False, str(row.get("env_id")))
            )
            continue

        try:
            standing, length = _zero_action_return(env_id)
        except Exception as exc:
            findings.append(
                Finding(f"{name}: {env_id} rolls out", False, f"{type(exc).__name__}: {exc}")
            )
            continue

        ratio = float(trained) / standing if standing else float("inf")
        measured = (
            f"trained {float(trained):.1f} vs standing {standing:.1f} (x{ratio:.2f}, "
            f"standing survived {length:.0f} of 1000 steps)"
        )

        findings.append(
            Finding(
                f"{name}: trained policy beats doing nothing on {env_id}",
                ratio > 1.0,
                measured + ("" if ratio > 1.0 else "  <- the reward is wrong, not the policy"),
            )
        )
        findings.append(
            Finding(
                f"{name}: the margin over doing nothing is worth training for",
                ratio >= HEALTHY_MARGIN,
                measured
                + (
                    ""
                    if ratio >= HEALTHY_MARGIN
                    else f"  <- under {HEALTHY_MARGIN}x, the reward is paying for"
                    " survival rather than locomotion (CORE_PLAN.md)"
                ),
            )
        )
    return findings
