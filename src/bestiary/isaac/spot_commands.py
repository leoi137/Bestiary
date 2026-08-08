"""The Spot command schedule — shared by the recorder and the closed-loop eval.

Pure numpy, no kit imports, importable before (or without) SimulationApp.
Extracted from `record_spot` so that `play_ntp` replays byte-identical
schedules: prediction P16 compares transformer vs teacher "on identical
scripts", and identical is only provable if both draw from this one function.
"""
from __future__ import annotations

#: The trained command distribution's edges, verbatim from the policy's own
#: env config (spot_env.yaml on the NVIDIA asset server, fetched 2026-08-06:
#: lin_vel_x [-2.0, 3.0] m/s, lin_vel_y [-1.5, 1.5] m/s, ang_vel_z
#: [-2.0, 2.0] rad/s). Commands are sampled INSIDE these; outside them the
#: policy is being asked a question it never trained on (play_spyder's rule).
VX_RANGE = (-2.0, 3.0)
VY_RANGE = (-1.5, 1.5)
WZ_RANGE = (-2.0, 2.0)

#: A Spot that has fallen: torso below this height. Default stand is ~0.55 m
#: (spawn at 0.8 m settles to stance); 0.3 m is unambiguous collapse.
FALL_HEIGHT_M = 0.3

#: Schedule shape per episode (seeded, reproducible): a stand phase, then
#: 2-4 driving phases of 2-4 s each, then a stop phase. One phase in every
#: episode is forced pure-forward — stage 1's target behaviour must appear
#: in every tape, not merely with sampling luck.
STAND_S = 1.0
PHASES = (2, 4)
PHASE_S = (2.0, 4.0)
STOP_S = 1.0


#: The command tour: a fixed, seedless schedule for filming. One command at a
#: time, held long enough to read, with a full stop between every pair so the
#: transformer is seen starting and stopping rather than blending one command
#: into the next. Every value sits well inside the trained ranges above.
#: Seedless on purpose: a title burned in at a computed timestamp stays
#: correct across re-films only if the tape is the same tape every time.
TOUR_ACTION_S = 4.5
TOUR_STOP_S = 2.5
TOUR_ACTIONS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("FORWARD", (1.5, 0.0, 0.0)),
    ("BACKWARD", (-1.0, 0.0, 0.0)),
    ("SIDE-STEP LEFT", (0.0, 1.0, 0.0)),
    ("SIDE-STEP RIGHT", (0.0, -1.0, 0.0)),
    ("TURN LEFT", (0.0, 0.0, 1.5)),
    ("TURN RIGHT", (0.0, 0.0, -1.5)),
)


def tour_schedule() -> list[tuple[float, tuple[float, float, float], str]]:
    """The filming script: [(duration_s, (vx, vy, wz), label), ...], ending stopped."""
    out: list[tuple[float, tuple[float, float, float], str]] = []
    for label, cmd in TOUR_ACTIONS:
        out.append((TOUR_ACTION_S, cmd, label))
        out.append((TOUR_STOP_S, (0.0, 0.0, 0.0), "STOP"))
    return out


def phase_schedule(rng) -> list[tuple[float, tuple[float, float, float]]]:
    """The episode's command script: [(duration_s, (vx, vy, wz)), ...]."""
    phases: list[tuple[float, tuple[float, float, float]]] = [(STAND_S, (0.0, 0.0, 0.0))]
    n = int(rng.integers(PHASES[0], PHASES[1] + 1))
    forced_forward = int(rng.integers(0, n))  # which driving phase is pure +vx
    for k in range(n):
        dur = float(rng.uniform(*PHASE_S))
        if k == forced_forward:
            cmd = (float(rng.uniform(0.5, VX_RANGE[1])), 0.0, 0.0)
        else:
            cmd = (
                float(rng.uniform(*VX_RANGE)),
                float(rng.uniform(*VY_RANGE)),
                float(rng.uniform(*WZ_RANGE)),
            )
        phases.append((dur, cmd))
    phases.append((STOP_S, (0.0, 0.0, 0.0)))
    return phases
