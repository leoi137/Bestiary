"""Guard: a "parked" detector must be able to flag the policy that does nothing.

Enforces `research/learnings/011`.

`docs/theory/command-tracking-reward.md` failure mode 1 calls a policy *parked
in the standing basin* when its mean Φ_v on the drive slices falls below
**0.15**. The whole point of that number is to catch a machine that has
rediscovered standing still.

It cannot. Measured over the six drive cells as `record/track_eval.py` defines
them, **zero action itself reads Φ_v = 0.2399** — above its own parked
threshold. The do-nothing policy, which is the exact thing the detector exists
to flag, passes.

The cause is one cell. The grid's `(0.0, 0.0, 0.45)` entry is a *pure turn*:
commanded forward speed is zero, so a machine that correctly does not move
forward scores Φ_v = 0.946 there. Averaging that into an "is it driving?"
statistic lets one free cell carry the mean over the line.

Over the five cells with genuinely nonzero commanded speed, zero action reads
**0.0985**, and a 0.15 threshold separates properly.

So the assertion is not "0.15 is the right number" — it is the weaker,
checkable thing that must hold for *any* threshold to mean anything:

    a detector's threshold must lie strictly above the control arm's own
    reading on the same cells it is evaluated over

Both numbers come from `research/measurements/tracking_baseline_zero_action.json`,
which was measured and committed before the run it judges. No physics is
stepped here — it reads one committed file — so this is fast tier and gates
launches.

`learnings/011` is the cost of not having this: a run's failure mode was named
from a detector nobody had checked against the null policy.
"""
from __future__ import annotations

import json

from bestiary import paths
from bestiary.guards import Finding

BASELINE = paths.RESEARCH / "measurements" / "tracking_baseline_zero_action.json"

# docs/theory/command-tracking-reward.md, failure mode 1.
PARKED_THRESHOLD = 0.15

STOP_CELL = (0.0, 0.0, 0.0)


def _zero_action_phi_v() -> tuple[float, float, int, int]:
    """(phi_v over all drive cells, phi_v over nonzero-speed cells, n_all, n_nonzero)."""
    grid = json.loads(BASELINE.read_text())["grid"]
    drive = [c for c in grid.values() if tuple(c["command"]) != STOP_CELL]
    nonzero = [c for c in drive if c["command"][0] != 0.0]
    mean = lambda cs: sum(c["mean_phi_v"] for c in cs) / len(cs)  # noqa: E731
    return mean(drive), mean(nonzero), len(drive), len(nonzero)


def run() -> list[Finding]:
    if not BASELINE.exists():
        return [
            Finding(
                "the zero-action tracking baseline exists",
                False,
                f"{BASELINE} is missing — the parked detector has no control arm to "
                f"be checked against, so its threshold asserts nothing (learnings/011)",
                n=0,   # no grid, so no cells examined; a FAIL is never vacuous
            )
        ]

    all_drive, nonzero, n_all, n_nonzero = _zero_action_phi_v()

    return [
        # THE assertion, and it is unconditional: whatever cell set the detector
        # is defined over, the control arm must fall below the threshold on it.
        # A detector that cannot flag the do-nothing policy is not a detector.
        Finding(
            "the parked detector can flag zero action on its own denominator",
            PARKED_THRESHOLD > nonzero,
            f"zero action reads phi_v = {nonzero:.4f} over the {n_nonzero} cells with "
            f"nonzero commanded speed vs a parked threshold of {PARKED_THRESHOLD} "
            f"(margin {PARKED_THRESHOLD - nonzero:+.4f})"
            + (
                ""
                if PARKED_THRESHOLD > nonzero
                else "  <- the do-nothing policy is ABOVE the threshold meant to catch"
                " it, so failure mode 1 cannot fire (learnings/011)"
            ),
            n=n_nonzero,   # the detector's own denominator
        ),
        # Why the denominator excludes the pure-turn cell. Reported, not
        # asserted-against: this is the measurement that forced the definition,
        # and it must stay visible or the next person re-derives the bug.
        Finding(
            "the zero-speed cell is excluded, and here is why",
            True,
            f"including it, zero action reads phi_v = {all_drive:.4f} over {n_all} "
            f"cells — ABOVE the {PARKED_THRESHOLD} threshold, because the pure-turn "
            f"cell commands v_x = 0 where a standing machine scores ~0.95 for "
            f"correctly not moving. That is the definition learnings/011 was written "
            f"about; track_eval's drive_grid_* still averages all {n_all}.",
            n=n_all,
        ),
        # A future grid edit that adds another zero-speed cell, or removes the
        # turn cell, changes the denominator silently. Make it visible.
        Finding(
            "the two cell sets still differ, so the exclusion still does something",
            n_nonzero < n_all,
            f"{n_all - n_nonzero} of {n_all} drive cell(s) command zero forward speed; "
            f"if this reaches 0 the distinction has quietly stopped mattering and the "
            f"threshold should be re-derived",
            # The set whose partition is being asserted. An empty grid makes this
            # VACUOUS rather than a green "the exclusion still does something",
            # which is the same self-monitoring the detail string already does.
            n=n_all,
        ),
    ]
