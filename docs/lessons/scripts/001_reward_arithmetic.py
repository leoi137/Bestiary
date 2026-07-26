"""Arithmetic for lesson 001 — why standing still outscored walking.

The number rule: no number enters the record unless code computed it. This is
that code for `../001-what-a-reward-function-is.md`.

Inputs are the per-episode totals measured over 5 greedy episodes of
`spyder_desert_v0` on SpyderDesert-v0, recorded in research/CORE_PLAN.md.
Nothing here is fitted or estimated — it is division and addition over
measured totals, done in code so the lesson cannot quietly contain a typo.

    python docs/lessons/scripts/001_reward_arithmetic.py
"""

STEPS = 1000  # a full episode at 20 Hz control = 50 s of simulated time

# Measured episode totals, spyder_desert_v0 on the desert heightfield.
WALK = {"forward": 294.0, "ctrl": -571.0, "contact": -16.0, "healthy": 1000.0}
STAND = {"forward": 0.0, "ctrl": 0.0, "contact": -13.0, "healthy": 1000.0}

CTRL_COST_WEIGHT = 0.1  # envs/spyder.py, the weight the desert run used


def total(terms: dict[str, float]) -> float:
    return sum(terms.values())


def per_step(terms: dict[str, float]) -> dict[str, float]:
    return {k: v / STEPS for k, v in terms.items()}


def main() -> None:
    walk_total, stand_total = total(WALK), total(STAND)
    w, s = per_step(WALK), per_step(STAND)

    print(f"{'':10} {'walk':>10} {'stand':>10}")
    for k in WALK:
        print(f"{k:10} {w[k]:>10.3f} {s[k]:>10.3f}   (per step)")
    print(f"{'TOTAL':10} {walk_total:>10.1f} {stand_total:>10.1f}   (per episode)")
    print()

    gap = stand_total - walk_total
    print(f"Standing beats walking by {gap:.0f} points per episode "
          f"({gap / STEPS:.3f} per step).")

    # What the control cost implies about effort: ctrl_cost = w * sum(a^2)
    sum_a2 = -w["ctrl"] / CTRL_COST_WEIGHT
    print(f"Mean effort while walking: sum(a^2) = {sum_a2:.2f} "
          f"(from {-w['ctrl']:.3f} / {CTRL_COST_WEIGHT})")

    # The ratio that flipped: payoff per unit of effort paid.
    print(f"Forward reward per unit of control cost: "
          f"{w['forward'] / -w['ctrl']:.2f} : 1")

    # What the fix in CORE_PLAN does to the same episode.
    fixed_ctrl = w["ctrl"] * (0.02 / CTRL_COST_WEIGHT) * STEPS
    fixed_total = WALK["forward"] + fixed_ctrl + WALK["contact"] + WALK["healthy"]
    print(f"\nWith ctrl_cost_weight 0.1 -> 0.02, the same gait scores "
          f"{fixed_total:.0f} against standing's {stand_total:.0f}.")


if __name__ == "__main__":
    main()
