"""Command sampling and language labels for the Spyder camera tapes.

Pure numpy, no kit imports, importable before (or without) SimulationApp —
the same contract as `spot_commands.py`, for the same reason: the closed-loop
eval (blind-vs-sighted, stage 2's success bar) must replay byte-identical
command schedules, and identical is only provable if the recorder and the
eval both draw from this one module.

ONE COMMAND PER EPISODE, deliberately — the Spot tapes ran multi-phase
schedules because next-token imitation wants transitions, but these tapes are
instruction-conditioned training data and the instruction must be TRUE of the
frames it captions. A tape captioned "walk forward" whose tail is commanded
stop is a mislabeled chunk for every training window that straddles the
switch. So an episode is: settle (untaped) -> STAND_S of commanded stand ->
one driving command held to the end. Pure standing episodes are their own
label, at the same 1-in-10 rate the teacher trained under
(`spyder_gentle_env_cfg.REL_STANDING = 0.1`).

The sampler mirrors `bestiary.isaac.commands.DeadZoneVelocityCommand`
semantics — the distribution the teacher was TRAINED on — taking the ranges
from the caller (the recorder reads them off the live env config) so a
widened task cannot silently disagree with this module:

  * v_x: sign * U(vx_min, vx_max) — the ambiguous near-zero band is never
    commanded, matching the dead zone.
  * w_z: U(-wz_max, wz_max), snapped to exactly 0 below wz_min — ~25% of
    driving episodes are straight drivers, as in training.
  * v_y: passed through the same shape for future wide-envelope teachers;
    the gentle task's range is (0, 0) so it stays exactly 0 there.
"""
from __future__ import annotations

#: Episode shape, seconds. Settle is simulated but never taped (the robot is
#: recovering from reset scatter); stand-in is taped (an instruction-true
#: stationary prefix teaches starting); the drive holds to the end — no stop
#: tail, per the module docstring.
SETTLE_S = 1.0
STAND_S = 1.0
DRIVE_S = 10.0

#: Fraction of episodes that are pure standing, commanded zero throughout and
#: labeled "stand still". Matches the teacher's trained standing fraction.
P_STAND = 0.1


def sample_command(
    rng,
    vx_range: tuple[float, float],
    vx_min: float,
    vy_range: tuple[float, float],
    wz_range: tuple[float, float],
    wz_min: float,
) -> tuple[float, float, float]:
    """One episode's held command, drawn from the teacher's trained regimes.

    `rng` is a seeded `numpy.random.Generator`; the schedule is a pure
    function of the episode seed. Returns (vx, vy, wz) with exact zeros where
    the dead zone dictates them — labels key on exact zeros, never epsilons.
    """
    if float(rng.uniform()) < P_STAND:
        return (0.0, 0.0, 0.0)
    sign = 1.0 if float(rng.uniform()) < 0.5 else -1.0
    vx = sign * float(rng.uniform(vx_min, vx_range[1]))
    vy = float(rng.uniform(*vy_range)) if vy_range[1] > vy_range[0] else 0.0
    wz = float(rng.uniform(*wz_range))
    if abs(wz) < wz_min:
        wz = 0.0
    return (vx, vy, wz)


def command_text(vx: float, vy: float, wz: float) -> str:
    """The mechanical caption: one deterministic phrasing per command class.

    Deterministic on purpose — paraphrase augmentation belongs to the
    training-side converter, where it can be seeded and recorded. Signs:
    +v_x forward, +v_y left (the FPS layout in `play_spyder`), +w_z is CCW
    from above = a left turn.
    """
    if vx == 0.0 and vy == 0.0 and wz == 0.0:
        return "stand still"
    parts: list[str] = []
    if vx > 0.0:
        parts.append("walk forward")
    elif vx < 0.0:
        parts.append("walk backward")
    if vy > 0.0:
        parts.append("side-step left")
    elif vy < 0.0:
        parts.append("side-step right")
    if wz != 0.0:
        turn = "turning left" if wz > 0.0 else "turning right"
        parts.append(turn if parts else ("turn left in place" if wz > 0.0 else "turn right in place"))
    return ", ".join(parts)
