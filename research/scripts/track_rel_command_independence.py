"""Arithmetic behind research/learnings/015.

Every number quoted in learning 015 is printed here, from the committed
measurement JSONs, so that none of it is prose arithmetic.

    venv/bin/python research/scripts/track_rel_command_independence.py

Sources (both committed):
    research/measurements/track_rel_s1_best.json    policy + zero-action arms
    research/measurements/track_rel_zero_action.json  the control arm alone

The question the script answers: did `hound_track_rel_s1` learn to *track a
command*, or did it learn one command-independent gait that a flat grid mean
scored as tracking?
"""
from __future__ import annotations

import json

from bestiary.paths import REPO_ROOT

BEST = REPO_ROOT / "research/measurements/track_rel_s1_best.json"

# The training command mixture, from runs/hound_track_rel_s1/config.json:
#   "stop0.1|turn0.1@0.3-0.6|drive vx0.3-0.8p+0.8,w0p0.5else+-0.6,vy0"
# The `drive` family samples a yaw command half the time, so the two
# (0.5, 0, +-0.4) cells belong to drive, not to turn. `turn` is the
# zero-forward-speed spin, which the grid represents with (0, 0, 0.45).
DRIVE_FAMILY = ("(0.5, 0.0, 0.0)", "(0.8, 0.0, 0.0)", "(-0.3, 0.0, 0.0)",
                "(0.5, 0.0, 0.4)", "(0.5, 0.0, -0.4)")
TURN_FAMILY = ("(0.0, 0.0, 0.45)",)
STOP_CELL = "(0.0, 0.0, 0.0)"
MIX = {"drive": 0.8, "turn": 0.1, "stop": 0.1}

# The eval grid's six non-stop cells, which is what drive_grid_mean averages.
GRID = DRIVE_FAMILY + TURN_FAMILY


def _mean(cells: dict, keys, field: str = "mean") -> float:
    return sum(cells[k][field] for k in keys) / len(keys)


def _mixture_weighted(cells: dict) -> float:
    return (MIX["drive"] * _mean(cells, DRIVE_FAMILY)
            + MIX["turn"] * _mean(cells, TURN_FAMILY)
            + MIX["stop"] * cells[STOP_CELL]["mean"])


def main() -> None:
    d = json.loads(BEST.read_text())
    pol, zero = d["trained"], d["zero_action"]
    pc, zc = pol["cells"], zero["cells"]

    print(f"run={d['run']}  checkpoint={d['checkpoint']}  "
          f"episodes/cell={d['episodes_per_cell']}  seed0={d['seed0']}")

    print("\n1. ACHIEVED FORWARD VELOCITY vs COMMANDED (m/s)")
    for k in GRID:
        c = pc[k]
        frac = c["achieved_vx"] / c["commanded_vx"] if c["commanded_vx"] else float("nan")
        print(f"   {k:<20} cmd_vx={c['commanded_vx']:+.2f}  "
              f"achieved={c['achieved_vx']:+.3f}  achieved/cmd={frac:6.3f}")
    fwd = [pc[k]["achieved_vx"] for k in GRID if pc[k]["commanded_vx"] > 0]
    print(f"   spread over the {len(fwd)} forward-commanded cells: "
          f"min={min(fwd):.3f} max={max(fwd):.3f} range={max(fwd) - min(fwd):.3f} m/s")
    print(f"   command_gain (fwd slope) policy={pol['command_gain']:.4f}  "
          f"zero_action={zero['command_gain']:.2e}")

    print("\n2. HEADING HOLD (mean_phi_w), policy vs the do-nothing control")
    for k in GRID + (STOP_CELL,):
        print(f"   {k:<20} policy={pc[k]['mean_phi_w']:.3f}  "
              f"zero={zc[k]['mean_phi_w']:.3f}  ratio={pc[k]['mean_phi_w'] / zc[k]['mean_phi_w']:.3f}")

    print("\n3. MIRROR CELLS — a yaw tracker would be symmetric in sign(w_cmd)")
    a, b = pc["(0.5, 0.0, 0.4)"], pc["(0.5, 0.0, -0.4)"]
    print(f"   (0.5,0,+0.4) return={a['mean']:+.2f}  phi_v={a['mean_phi_v']:.3f}  phi_w={a['mean_phi_w']:.3f}")
    print(f"   (0.5,0,-0.4) return={b['mean']:+.2f}  phi_v={b['mean_phi_v']:.3f}  phi_w={b['mean_phi_w']:.3f}")
    print(f"   asymmetry: {abs(a['mean'] - b['mean']):.2f} return points, "
          f"phi_v {a['mean_phi_v']:.3f} vs {b['mean_phi_v']:.3f}")

    print("\n4. PER-CELL WIN/LOSS AGAINST DOING NOTHING")
    gap_total = pol["drive_grid_mean"] - zero["drive_grid_mean"]
    losses = 0
    for k in GRID:
        g = pc[k]["mean"] - zc[k]["mean"]
        losses += g < 0
        print(f"   {k:<20} policy={pc[k]['mean']:+9.2f}  zero={zc[k]['mean']:+9.2f}  "
              f"gap={g:+9.2f}  share of grid gap={g / len(GRID) / gap_total * 100:6.1f}%")
    g_fwd = pc["(0.5, 0.0, 0.0)"]["mean"] - zc["(0.5, 0.0, 0.0)"]["mean"]
    print(f"   drive_grid_mean policy={pol['drive_grid_mean']:.2f}  "
          f"zero={zero['drive_grid_mean']:.2f}  ratio={d['drive_grid_ratio']:.3f}  "
          f"(theory bar >= 5x, cleared by {(d['drive_grid_ratio'] / 5 - 1) * 100:.2f}%)")
    print(f"   grid gap = {gap_total:+.2f} points; the single (0.5,0,0) cell "
          f"contributes {g_fwd / len(GRID):+.2f} of it = "
          f"{g_fwd / len(GRID) / gap_total * 100:.1f}%")
    print(f"   the policy LOSES to doing nothing in {losses} of {len(GRID)} cells")
    print(f"   stop cell: policy={pol['stop_cell_mean']:.2f}  "
          f"zero={zero['stop_cell_mean']:.2f}  ratio={d['stop_cell_ratio']:.4f}")

    print("\n5. FLAT GRID vs THE TRAINING COMMAND MIXTURE "
          f"({MIX['drive']:.0%} drive / {MIX['turn']:.0%} turn / {MIX['stop']:.0%} stop)")
    wp, wz = _mixture_weighted(pc), _mixture_weighted(zc)
    print(f"   flat 6-cell grid : policy={pol['drive_grid_mean']:.2f} "
          f"zero={zero['drive_grid_mean']:.2f}  ratio={pol['drive_grid_mean'] / zero['drive_grid_mean']:.3f}x")
    print(f"   mixture-weighted : policy={wp:.2f} zero={wz:.2f}  "
          f"ratio={wp / wz:.4f}x  (margin {(wp / wz - 1) * 100:.1f}%)")

    print("\n6. INCOME vs CONTROL COST ON THE DRIVE GRID (per episode)")
    inc, ctrl = pol["drive_grid_reward_track"], pol["drive_grid_reward_ctrl"]
    print(f"   reward_track={inc:+.2f}  reward_ctrl={ctrl:+.2f}  "
          f"ctrl as share of income={abs(ctrl) / inc * 100:.1f}%")
    print(f"   income above the control arm: "
          f"{inc - zero['drive_grid_reward_track']:+.2f}")

    print("\n7. DECOMPOSITION RESIDUAL — anomalies row 39 (the four reported "
          "terms do not sum to the return)")
    for name, arm in (("policy", pol), ("zero_action", zero)):
        s = sum(arm[f"drive_grid_reward_{t}"]
                for t in ("track", "ctrl", "contact", "termination"))
        r = arm["drive_grid_mean"] - s
        print(f"   {name:<11} sum_of_4_terms={s:+.2f}  return={arm['drive_grid_mean']:+.2f}  "
              f"residual={r:+.2f} ({abs(r) / abs(arm['drive_grid_mean']) * 100:.0f}% of the return)")


    print("\n8. WHY ONE SPEED IS A RATIONAL OPTIMUM: Phi_v is a tolerance BAND")
    print("   Phi_v(v; c) = exp(-((v - c) / alpha_v(c))^2),  "
          "alpha_v(c) = max(0.15, 0.5*|c|)   [env constants]")
    grid_cmds = sorted({pc[k]["commanded_vx"] for k in GRID})
    for v in (0.271, 0.309, 0.0):
        per = [(c, kernel(v, c)) for c in grid_cmds]
        print(f"   one fixed speed v={v:.3f} m/s scores  "
              + "  ".join(f"Phi_v({c:+.2f})={p:.3f}" for c, p in per)
              + f"   mean={sum(p for _, p in per) / len(per):.3f}")
    best_v = max((sum(kernel(v / 1000, c) for c in grid_cmds) / len(grid_cmds), v / 1000)
                 for v in range(-400, 901))
    print(f"   best single speed over these {len(grid_cmds)} commands: "
          f"v={best_v[1]:.3f} m/s, mean Phi_v={best_v[0]:.3f}  "
          f"(perfect command-following scores 1.000)")
    # Over the command distribution the policy was actually trained on, not the
    # eval grid: the `drive` family samples vx uniform in [0.3, 0.8], forward
    # with p=0.8 (config.json cmd_dist). This is the number that says how much
    # of the available speed income one fixed gait can collect.
    fwd = [0.3 + 0.5 * i / 500 for i in range(501)]
    best_fwd = max((sum(kernel(v / 1000, c) for c in fwd) / len(fwd), v / 1000)
                   for v in range(0, 901))
    at_trot = sum(kernel(0.271, c) for c in fwd) / len(fwd)
    print(f"   over the TRAINED forward-drive range vx ~ U[0.30, 0.80]: "
          f"best single speed v={best_fwd[1]:.3f} m/s earns mean Phi_v="
          f"{best_fwd[0]:.3f} of the 1.000 a perfect tracker earns; "
          f"the observed 0.271 m/s trot earns {at_trot:.3f}")
    print("   the band is flat near its own optimum: mean Phi_v over "
          + ", ".join(f"v={v:.2f}->{sum(kernel(v, c) for c in grid_cmds) / len(grid_cmds):.3f}"
                      for v in (0.20, 0.25, 0.30, 0.35)))


def kernel(v: float, c: float) -> float:
    """Phi_v for an achieved speed v under command c.

    Calls the ENV's own kernel and tolerance rather than restating them, so
    this script cannot drift from the reward the policy was actually paid by.
    """
    from bestiary.envs.hound_track_rel import relative_kernel, velocity_tolerance

    return relative_kernel(v - c, velocity_tolerance(c))


if __name__ == "__main__":
    main()
