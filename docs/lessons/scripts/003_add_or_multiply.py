"""Arithmetic for lesson 003 — add or multiply two tracking scores.

The number rule: no number enters the record unless code computed it. This is
that code for `../003-add-or-multiply.md`.

The scenario is a hound standing perfectly still while commanded to drive
forward at 0.5 m/s and not to turn. Its speed error is therefore the whole
commanded speed, and its yaw error is the measured standing yaw noise from
`research/measurements/tracking_noise.json` — the machine is not still, but it
is very nearly not turning, which is precisely why the turning term is free.

Tolerances are the ones derived in `../../theory/command-tracking-reward.md`.

    venv/bin/python docs/lessons/scripts/003_add_or_multiply.py
"""

SIGMA_V = 0.15      # m/s,  derived in docs/theory/command-tracking-reward.md
SIGMA_W = 0.10      # rad/s, same
V_CMD = 0.5         # m/s,  a mid-range drive command
YAW_NOISE = 0.0182  # rad/s, measured: research/measurements/tracking_noise.json
WEIGHT = 0.5        # equal weights, the natural choice for an additive form


def phi(u: float) -> float:
    """Cauchy kernel: 1 at zero error, falling as the error grows."""
    return 1.0 / (1.0 + u * u)


def main() -> int:
    u_v = V_CMD / SIGMA_V            # standing => the error IS the command
    u_w = YAW_NOISE / SIGMA_W        # standing => nearly perfect on yaw
    phi_v, phi_w = phi(u_v), phi(u_w)

    added = WEIGHT * phi_v + WEIGHT * phi_w
    product = phi_v * phi_w

    print(f"standing, commanded ({V_CMD} m/s forward, 0 rad/s yaw)")
    print(f"  u_v = {V_CMD}/{SIGMA_V} = {u_v:.4f}   ->  Phi_v = {phi_v:.4f}")
    print(f"  u_w = {YAW_NOISE}/{SIGMA_W} = {u_w:.4f}  ->  Phi_w = {phi_w:.4f}")
    print()
    print(f"  added    {WEIGHT}*{phi_v:.3f} + {WEIGHT}*{phi_w:.3f} = {added:.4f}")
    print(f"  product  {phi_v:.3f} * {phi_w:.3f}         = {product:.4f}")
    print()
    print(f"  the additive form pays {added / product:.1f}x more for standing still")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
