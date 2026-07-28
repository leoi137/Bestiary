"""The arithmetic behind anomalies.jsonl row 39, and the check that it is closed.

Row 39: `record/track_eval.py` decomposed a tracking run's return into a
HARDCODED four-term tuple, and `HoundPDTrackRelDesert-v0` pays five -- it
gained `reward_shaping` when the PBRS term landed. So every
`drive_grid_reward_*` field it printed was a partial account of the return,
presented as a complete one, and nothing in the output said so.

This script does two things, and both are the point:

1. **Sizes the error that was there**, per env, by summing the OLD four-term
   list against the return the env actually paid. On the 4-term env the
   residual must be exactly zero -- which is what says no previously published
   4-term number moves because of this fix.
2. **Shows the new assertion can go red.** A guard that cannot fail bounds
   nothing (`learnings/014`). `assert_decomposition_complete` is handed a
   deliberately truncated term list and must raise.

    venv/bin/python research/scripts/decomposition_completeness.py

Zero-action rollouts, so it needs no checkpoint and takes a few seconds.
"""
from __future__ import annotations

import gymnasium as gym

import bestiary.envs  # noqa: F401 -- registers the env ids
from bestiary.record.track_eval import (
    assert_decomposition_complete,
    discover_terms,
    rollout,
)

# The tuple that was hardcoded in `record/track_eval.py` and duplicated in
# `research/scripts/track_return_decomposition.py` until 2026-07-28.
OLD_HARDCODED_TERMS = (
    "reward_track", "reward_ctrl", "reward_contact", "reward_termination",
)

# Both envs the tracking protocol is defined for, with the term count each one
# actually pays. Asserted, not printed: the whole failure was a count nobody
# checked.
ENVS = (("HoundPDTrackRelDesert-v0", 5), ("HoundPDTrackDesert-v0", 4))

CELL = (0.5, 0.0, 0.0)
SEED = 1000


def measure(env_id: str) -> dict:
    """One zero-action episode; the new residual and the old one."""
    env = gym.make(env_id)
    try:
        ep = rollout(env, seed=SEED, policy=None, forced_cmd=CELL)
    finally:
        env.close()

    terms = tuple(ep["terms"])
    ret = float(ep["return"])
    new_sum = sum(float(ep[t]) for t in terms)
    old_sum = sum(float(ep[t]) for t in OLD_HARDCODED_TERMS if t in ep)
    return {
        "env": env_id,
        "terms": terms,
        "n_terms": len(terms),
        "return": ret,
        "new_sum": new_sum,
        "new_residual": new_sum - ret,
        "old_sum": old_sum,
        "old_residual": old_sum - ret,
    }


def assertion_fires() -> str:
    """Hand the checker a truncated term list; it must raise. Returns the message."""
    info = {
        "reward_track": 0.9, "reward_shaping": -0.02, "reward_ctrl": -0.1,
        "reward_contact": 0.0, "reward_termination": 0.0, "track_phi_v": 0.5,
    }
    reward = 0.78
    full = discover_terms(info)
    # The complete list must PASS, or the check is just always-red.
    assert_decomposition_complete(full, info, reward, 0)

    truncated = tuple(t for t in full if t != "reward_shaping")
    try:
        assert_decomposition_complete(truncated, info, reward, 17)
    except ValueError as exc:
        return str(exc).splitlines()[0]
    raise SystemExit(
        "assert_decomposition_complete did NOT raise on a term list missing "
        "reward_shaping. The check is decorative and row 39 is not closed."
    )


def main() -> None:
    print("anomalies.jsonl row 39 -- reward decomposition completeness\n")
    rows = [measure(env_id) for env_id, _ in ENVS]

    for r, (_, expected_n) in zip(rows, ENVS):
        pct = abs(r["old_residual"]) / abs(r["return"]) * 100 if r["return"] else 0.0
        print(f"{r['env']}  ({r['n_terms']} terms, expected {expected_n})")
        print(f"    {list(r['terms'])}")
        print(f"    return                {r['return']:+.9f}")
        print(f"    sum of ALL terms      {r['new_sum']:+.9f}   "
              f"residual {r['new_residual']:+.3e}")
        print(f"    sum of the OLD 4      {r['old_sum']:+.9f}   "
              f"residual {r['old_residual']:+.6f}  ({pct:.1f}% of the return)")
        print()
        if r["n_terms"] != expected_n:
            raise SystemExit(
                f"{r['env']} pays {r['n_terms']} terms, expected {expected_n}"
            )
        # The new decomposition must close to float noise on every env.
        if abs(r["new_residual"]) > 1e-9 * max(1.0, abs(r["return"])):
            raise SystemExit(f"{r['env']}: new decomposition does not close")

    # The 4-term env must be BIT-IDENTICAL under old and new. This is the
    # claim that no previously published 4-term figure is invalidated.
    four = next(r for r in rows if r["n_terms"] == 4)
    if four["old_sum"] != four["new_sum"]:
        raise SystemExit(
            f"{four['env']}: old and new decompositions differ "
            f"({four['old_sum']!r} vs {four['new_sum']!r}) -- published 4-term "
            f"numbers for this env WOULD move, which contradicts the writeup."
        )
    print(f"{four['env']}: old and new sums are bit-identical "
          f"({four['new_sum']!r}) -- no published 4-term number moves.\n")

    print("the assertion can go red:")
    print("   ", assertion_fires())
    print("\nrow 39 closed.")


if __name__ == "__main__":
    main()
