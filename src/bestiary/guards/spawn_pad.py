"""Guard: the spawn pad is EXACTLY flat, which is what `learnings/009` rests on.

`research/learnings/009` establishes that the hound's passive backward creep is
not caused by the wheel/cell scale collision, and that regenerating the terrain
at `GRID=2048` does not fix it. The whole argument is one measurement:

    max|h| = 0.000e+00 m  and  peak-to-peak = 0.000e+00 m
    over every cell within 2.5 m of the origin

On exactly flat ground every heightfield prism's normal is +z by construction,
so the "wheel straddles a boundary between two prisms whose normals disagree"
mechanism cannot occur there — at any cell size. Since every contact point in
that measurement landed within 2.034 m, the creep is generated entirely on
ground where the proposed mechanism is unavailable. That is the finding.

**The finding is only as durable as the flatness.** `terrain/generate.py`
produces the pad by multiplying the composed field by a blend that is exactly
zero inside `FLAT_RADIUS_M`. That is a design choice sitting in the middle of a
procedural generator, several layers deep, and nothing anywhere records that a
learning depends on it. Change the blend, the falloff radius, or the order of
composition, and the pad acquires relief — at which point learnings/009 becomes
quietly wrong and the GRID=2048 proposal it retired becomes live again, with
nothing on disk to say so.

This is the falsifier from that learning's "How we would know this is wrong",
written as an assertion instead of as a sentence someone has to remember to
re-check. It is the cheapest thing that makes the class of bug impossible
rather than merely fixed: a lesson in prose depends on a reader arriving at the
right moment, and this one's moment is "somebody edits a terrain generator",
which is exactly when nobody is reading research/learnings/.

Deliberately checks the METRIC FIELD rather than a compiled MuJoCo model. The
question is whether `generate.py` still produces a flat pad, and that is a
property of the generator; going through a compiled model would add a MuJoCo
dependency to the fast tier for no extra coverage, and `terrain-spec` already
watches the compiled side.
"""
from __future__ import annotations

import numpy as np

from bestiary.guards import Finding

# The radius generate.py flattens, and the radius learnings/009's argument
# claims. Not independently chosen here: if these two ever disagree, the
# learning is describing a pad that no longer exists, which is the whole thing
# this guard is for.
FLAT_RADIUS_M = 2.5

# generate.py's own default seed. The pad is a construction of the blend and is
# seed-independent by design -- which is itself worth asserting, so a second
# seed is checked rather than assumed.
SEEDS = (7, 12)

# "Exactly flat" is meant literally. The blend multiplies the field by zero, so
# the cells hold bit-identical values and the spread is 0.0, not merely small.
# A tolerance here would let a real change hide: 1e-9 m of relief is still a
# surface with disagreeing normals, and the mechanism learnings/009 rules out
# needs only that the normals differ, not that they differ by much.
EXACT = 0.0


def _pad_spread(height_m: np.ndarray, half_extent_m: float, radius_m: float) -> tuple[float, float, int]:
    """(max|h|, peak-to-peak, cell count) over the disk of `radius_m`.

    `height_m` is the composed terrain in meters with the spawn surface at 0,
    row axis = y, col axis = x, as `generate.build_height_m` returns it.
    """
    n = height_m.shape[0]
    coords = np.linspace(-half_extent_m, half_extent_m, n)
    x, y = coords[None, :], coords[:, None]
    inside = (x * x + y * y) <= radius_m * radius_m
    pad = height_m[inside]
    if pad.size == 0:
        raise ValueError(
            f"no cells within {radius_m} m of the origin on a {n}x{n} grid over "
            f"+/-{half_extent_m} m; the geometry assumption in this guard is wrong"
        )
    return float(np.abs(pad).max()), float(pad.max() - pad.min()), int(pad.size)


def run() -> list[Finding]:
    from bestiary.terrain import generate

    findings: list[Finding] = []

    for seed in SEEDS:
        try:
            height = generate.build_height_m(seed)
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            findings.append(
                Finding(
                    f"seed {seed}: terrain builds",
                    False,
                    f"{type(exc).__name__}: {exc}. learnings/009 cannot be checked, "
                    f"so treat its conclusion about GRID=2048 as unverified",
                    n=0,   # the terrain never built, so no cell was examined
                )
            )
            continue

        max_abs, ptp, cells = _pad_spread(height, generate.HALF_EXTENT, FLAT_RADIUS_M)
        flat = max_abs == EXACT and ptp == EXACT
        findings.append(
            Finding(
                f"seed {seed}: spawn pad is exactly flat within {FLAT_RADIUS_M} m",
                flat,
                f"max|h| = {max_abs:.3e} m, peak-to-peak = {ptp:.3e} m over {cells} cells"
                + (
                    ""
                    if flat
                    else (
                        f" -- expected exactly {EXACT}. The pad has acquired relief, so "
                        f"heightfield prisms under the robot no longer all share a +z "
                        f"normal. research/learnings/009 concluded that the hound's creep "
                        f"is NOT caused by the wheel/cell scale collision precisely "
                        f"because the mechanism was unavailable on a flat pad. That "
                        f"argument no longer holds: re-run research/scripts/creep_vs_grid.py "
                        f"and supersede learnings/009 before relying on it, and before "
                        f"treating GRID=2048 as retired"
                    )
                ),
                # The quantification really is over cells: "every cell within
                # 2.5 m is at exactly 0". `_pad_spread` raises rather than
                # returning 0 cells, so this cannot silently go vacuous.
                n=cells,
            )
        )

    return findings
