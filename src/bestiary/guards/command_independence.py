"""Guard: a policy that wins the drive grid must move differently for different commands.

Enforces `research/learnings/015` and `research/anomalies.jsonl` row 38.

WHAT WENT WRONG

`hound_track_rel_s1` cleared the theory note's success bar — `drive_grid_ratio`
5.04 against a required 5.0 — while achieving 0.271 m/s when 0.5 was commanded
and 0.309 m/s when 0.8 was. A 0.30 m/s change in the command bought a 0.038 m/s
change in the machine. It had learned one forward trot and was running it under
every command.

The pre-registered lie-detector did not see it. `command_gain` regresses
achieved forward velocity on commanded velocity across the drive cells, and it
read 0.382 against a bar of 0.05 — large only because the machine creeps
*backward* (−0.073 m/s) on the one backward-commanded cell while trotting
forward on all the others. A slope fitted across a sign change is a sign
detector; it says nothing about whether 0.8 is faster than 0.5.

WHAT THIS GUARD ASSERTS, AND WHY IT IS SHAPED THIS WAY

One statistic, `vx_span_ratio`: over the cells that command a *positive forward
speed and no yaw*, the spread of achieved forward velocity divided by the
spread of commanded forward velocity.

    vx_span_ratio = (max achieved_vx − min achieved_vx)
                  / (max commanded_vx − min commanded_vx)

A machine that tracks perfectly scores 1.0. A machine running one gait
regardless of the command scores 0.0. It is deliberately *not* a regression
slope: no fit, no sign change to hide behind, and it is undefined rather than
misleading when fewer than two distinct forward commands were measured.

The bar is `MIN_SPAN_RATIO = 0.25` and it is loose on purpose. A policy tracking
at half gain — the honest half-speed failure this project has seen repeatedly —
scores 0.5 and passes comfortably. `hound_track_rel_s1` scores 0.127. The bar is
set to separate *a weak tracker* from *no tracker*, not to grade tracking.

It applies only where a win is being claimed: `drive_grid_ratio >=
WIN_RATIO = 5.0`, the success bar in `docs/theory/command-tracking-reward.md`
§3. A measurement of a policy that is losing anyway needs no command-
independence check, and failing there would only add noise to a run already
known to be bad.

WHAT THIS GUARD DOES NOT ASSERT

**The yaw half, because the artifact cannot support it.** `track_eval` records
`achieved_vx` per cell and has no yaw twin — no `achieved_wz` — so there is
nothing on disk to regress a yaw command against (`anomalies.jsonl` row 38).
The yaw evidence in learnings/015 is `mean_phi_w` and the mirror-cell
asymmetry, and both are *reported* below rather than asserted: a threshold on
Φ_w would be a threshold on terrain and gait as much as on command-following.
When `achieved_wz` exists, the second assertion goes here.
"""
from __future__ import annotations

import json
from pathlib import Path

from bestiary import paths
from bestiary.guards import Finding

# Achieved-speed spread as a fraction of commanded spread. See the module
# docstring: 1.0 is perfect tracking, 0.0 is one command-independent gait, and
# a half-gain tracker scores 0.5. 0.25 separates the last from the first two.
MIN_SPAN_RATIO = 0.25

# docs/theory/command-tracking-reward.md §3: a successful run returns >= 5x the
# do-nothing control on the drive grid. Below it, nothing is being claimed.
WIN_RATIO = 5.0

# The measurements that ARE learnings/015 — the evidence the lesson was written
# from, committed and kept. They are not exempt from the finding; they are the
# finding, and re-failing on them forever would gate every future launch behind
# a run that is already understood and already written up.
#
# Same discipline as `measurement_provenance`'s grandfather list: explicit,
# committed, and it may only ever get shorter. A new name added here is a guard
# being defeated rather than satisfied.
GRANDFATHERED: frozenset[str] = frozenset({
    "track_rel_s1_best.json",
    "track_rel_s1_latest.json",
})


def span_ratio(cells: dict) -> tuple[float | None, int, str]:
    """(vx_span_ratio, cells used, one-line detail) over forward-only cells.

    Returns `None` when fewer than two distinct positive forward commands with
    no yaw were measured — the statistic is undefined, and reporting 0.0 there
    would fail a grid that simply never asked the question.
    """
    pts = [(c["commanded_vx"], c["achieved_vx"])
           for c in cells.values()
           if c["commanded_vx"] > 0 and c["command"][1] == 0 and c["command"][2] == 0]
    if len({round(cmd, 6) for cmd, _ in pts}) < 2:
        return None, len(pts), f"only {len(pts)} forward-only cell(s); undefined"
    cmd_span = max(c for c, _ in pts) - min(c for c, _ in pts)
    ach_span = max(a for _, a in pts) - min(a for _, a in pts)
    return (ach_span / cmd_span, len(pts),
            f"achieved spread {ach_span:.3f} m/s over commanded spread "
            f"{cmd_span:.3f} m/s")


def _mirror_asymmetry(cells: dict) -> str:
    """Reported, never asserted: the +w and -w cells of the same speed."""
    out = []
    for key, cell in cells.items():
        vx, vy, wz = cell["command"]
        if wz <= 0 or vy != 0:
            continue
        twin = next((c for c in cells.values()
                     if c["command"] == [vx, vy, -wz]), None)
        if twin is not None:
            out.append(f"{key} vs its mirror: return {cell['mean']:+.2f} vs "
                       f"{twin['mean']:+.2f} (|delta| {abs(cell['mean'] - twin['mean']):.2f}), "
                       f"phi_w {cell['mean_phi_w']:.3f} vs {twin['mean_phi_w']:.3f}")
    return "; ".join(out) if out else "no mirrored yaw cells in this grid"


def _measurements() -> list[Path]:
    return sorted((paths.REPO_ROOT / "research/measurements").glob("*.json"))


def run() -> list[Finding]:
    findings: list[Finding] = []

    # 1. The statistic's own arithmetic, on two cell-sets whose right answers
    #    are known by construction. Without this the guard would be vacuous on a
    #    clone with no qualifying measurement, and — worse — a formula nobody
    #    ever saw produce a known value would be gating launches.
    def _cells(pairs) -> dict:
        return {str(i): {"command": [c, 0.0, 0.0], "commanded_vx": c,
                         "achieved_vx": a, "mean": 0.0, "mean_phi_w": 0.0}
                for i, (c, a) in enumerate(pairs)}

    perfect, _, _ = span_ratio(_cells([(0.5, 0.5), (0.8, 0.8)]))
    frozen, _, _ = span_ratio(_cells([(0.5, 0.27), (0.8, 0.27)]))
    findings.append(Finding(
        "vx_span_ratio reads 1.0 for a perfect tracker and 0.0 for one fixed gait",
        perfect is not None and abs(perfect - 1.0) < 1e-12
        and frozen is not None and abs(frozen) < 1e-12,
        f"synthetic tracker (0.5->0.5, 0.8->0.8) scores {perfect}; synthetic "
        f"fixed 0.27 m/s trot under both commands scores {frozen}. These bound "
        f"the scale the {MIN_SPAN_RATIO} bar is read on",
        n=2,
    ))

    # 2. THE assertion, over every committed measurement that claims a win.
    checked = 0
    bad: list[str] = []
    lines: list[str] = []
    for path in _measurements():
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:      # loud, with the file named
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        trained = doc.get("trained")
        ratio = doc.get("drive_grid_ratio")
        if not isinstance(trained, dict) or "cells" not in trained or ratio is None:
            continue
        if ratio < WIN_RATIO:
            continue
        value, n_cells, why = span_ratio(trained["cells"])
        if value is None:
            continue
        checked += 1
        tag = "" if path.name not in GRANDFATHERED else "  [grandfathered: learnings/015]"
        lines.append(f"{path.name}: drive_grid_ratio={ratio:.2f}, "
                     f"vx_span_ratio={value:.3f} over {n_cells} cells ({why}); "
                     f"{_mirror_asymmetry(trained['cells'])}{tag}")
        if value < MIN_SPAN_RATIO and path.name not in GRANDFATHERED:
            bad.append(f"{path.name}: vx_span_ratio={value:.3f} < {MIN_SPAN_RATIO} "
                       f"while claiming drive_grid_ratio={ratio:.2f}")

    detail = (f"{checked} measurement(s) claim drive_grid_ratio >= {WIN_RATIO} "
              f"and measure >=2 distinct forward-only commands. ")
    detail += ("; ".join(lines) if lines else "none on disk. ")
    if bad:
        detail += ("  <- A DRIVE-GRID WIN IS BEING CLAIMED BY A POLICY THAT MOVES "
                   "THE SAME WAY UNDER EVERY COMMAND. learnings/015: the reward is "
                   "a product of tolerance bands, so one speed sitting inside the "
                   "band for several commands collects most of the income at one "
                   "gait's control cost. " + "; ".join(bad))
    findings.append(Finding(
        "a measurement claiming a drive-grid win shows achieved speed varying "
        "with commanded speed",
        not bad,
        detail,
        n=checked,
    ))

    return findings
