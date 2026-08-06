"""The gentle terrain: the desert's texture without the desert's relief.

    venv/bin/python -m bestiary.terrain.gentle            # writes the asset
    venv/bin/python -m bestiary.terrain.gentle --stats    # numbers only, writes nothing

Writes `assets/terrain/gentle_hfield.bin` (+ a baked-albedo preview PNG) in the
same MuJoCo custom-heightfield format as the desert, over the same 80 x 80 m
footprint at the same 7.8 cm sampling — so `terrain/isaac_hf.py` serves it to
Isaac Lab through the exact machinery the desert already uses, and nothing
downstream learns a second file format.

WHY THIS ASSET EXISTS
---------------------
The desert spans 5.05 m of elevation; its mountains are terrain for a machine
that already walks. This is the terrain for a machine that is *learning* to
walk: the same three-layer composition — broad forms, quasi-regular ripples,
fine rubble — with the mountain layer replaced by low hills and every
wavelength kept, so the ground is genuinely irregular at the scale of a
footstep without ever being a wall at the scale of a body. Hills stay; summits
go. The operator asked for exactly that, 2026-08-05.

It is a NEW file, not a regenerated desert. The desert's bytes are pinned by
hash into the provenance of every run that stood on them (`terrain/spec.py`;
the terrain invariant in CLAUDE.md calls a moved ground the quietest failure in
the project). Assets are append-only the same way the ledger is.

THE SPAN IS DECLARED, NOT EMERGENT
----------------------------------
`Z_SPAN_M` below is exact by construction: the composed field is rescaled so
its min-to-max span equals the constant. The `.bin` format stores samples
normalised to [0, 1] and cannot carry metres, so every reader must be *told*
the span — the desert reads its 5.05 from an `<hfield size=...>` attribute in a
committed XML, but this asset has no MuJoCo XML yet, so the constant here is
the single source of truth and the rescale is what makes it true rather than
approximately true. The pre-rescale span is printed at generation so the
factor is visible (a factor far from 1.0 means the layer amplitudes below no
longer mean what they say).

Layer ratios are preserved by the rescale, so the amplitudes below are exact
only relative to each other; the printed post-rescale stds are the real ones.

WHAT "GENTLE BUT NOT BASIC" MEANS, IN NUMBERS
---------------------------------------------
The scale that matters is relative to the robot, not absolute. Spyder stands
~0.35 m tall on ~0.6 m legs; terrain that is hard *for it* is decimetre relief
inside a stride, not metre relief across a dune. So against the desert:

    layer      desert                      gentle
    broad      mountains, 14-80 m, ~3.3    hills, 6-20 m, no ridging — SHORT
                                           so a crest and a base fit inside
                                           an 8 m training tile (the
                                           amplitude constants' note is the
                                           full argument)
    ripples    dunes 14-30 m, 0.8          ripples 4-10 m, same reason
    rubble     0.04 x (0.5-3.0 m)          0.035 x (1.5-3.5 m) — relative to
                                           its hills this is ~8x the desert's
                                           rubble-to-broad ratio, which is the
                                           "more roughness" the operator asked
                                           for without the slopes that come
                                           with sub-metre wavelengths

The rubble band's low end moved 0.5 -> 1.5 m deliberately: training resamples
to a 0.1 m grid (`TRAIN_CELL_M` in the env cfg), so wavelengths under ~8 cells
arrive aliased; and short-wavelength height IS slope — at 0.8 m the first
composition measured P99 slopes of 50 degrees, accidental mountains at gravel
scale. The sweep note on the amplitude constants has the numbers.

`--stats` prints slope percentiles and per-tile relief because those are the
two numbers that say "walkable but rough": slope P99 should sit well under
atan(0.75) = 36.9 degrees — the rise/run ratio Isaac Lab's own
`slope_threshold=0.75` treats as a wall and converts to vertical — and median
8 m-tile relief should be a fraction of standing height, not a multiple.
"""

from __future__ import annotations

import argparse

import numpy as np

from bestiary import paths

# The spectral machinery is imported from the desert generator rather than
# copied: `_spectral_field` reads generate.py's module-level CELL, which this
# asset shares by USING THE SAME GRID GEOMETRY (below). If gentle ever needs a
# different extent or grid, the helper must grow a `cell` parameter first —
# importing it while the constants disagree would band-pass at the wrong
# wavelengths and raise nothing.
from bestiary.terrain.generate import (
    CELL,
    GRID,
    HALF_EXTENT,
    _spectral_field,
    save_hfield_bin,
    save_texture,
)

#: Total elevation span, metres, EXACT by rescale (see module docstring).
#: 1.0 m over an 80 m world against the desert's 5.05 m: hills a spider walks
#: over, not mountains it summits. 1.0 m = 3.3 ft.
Z_SPAN_M = 1.0

#: Generation seed. Arbitrary, fixed, and deliberately NOT the desert's 7 —
#: sharing a seed would correlate the two worlds' large-scale forms, and the
#: point of a second asset is a second world.
SEED = 11

#: Pre-rescale layer amplitudes (each multiplies a unit-std field). Only their
#: RATIOS survive the exact-span rescale; the printed post-rescale stds are the
#: authoritative numbers.
#:
#: THIRD composition, and each predecessor died on a measurement:
#:
#: 1. (0.26/0.16/0.08, rubble floor 0.8 m, ripple ^1.5, span 1.2): slope
#:    P90/P99 = 36.7/50.4 deg — accidental mountains at rubble scale.
#:    Short-wavelength height IS slope.
#: 2. (0.34/0.09/0.035, hills in a 15-40 m band): slopes fine — and the
#:    operator looked at it in the viewer and saw NO HILLS, because there
#:    were none to see: training and viewing happen on 8 m tiles, each with
#:    its own baseline subtracted, and a 15-40 m wavelength cannot fit a
#:    crest and a base inside an 8 m window. Measured: of that layer's
#:    0.283 m median per-tile contribution, only 0.161 m survived removing
#:    the best-fit plane — the "hills" were tile-scale TILT. The desert's
#:    mountains survive its tiling only because 5.05 m of relief makes even
#:    a patch of mountainside steep; scale the amplitude down and keep the
#:    wavelength, and tiling erases the layer. A HILL THE TILING PRESERVES
#:    MUST FIT INSIDE A TILE.
#:
#: So the hills moved down-band to 6-20 m — crest AND base inside 8 m —
#: staying the SMOOTH field, not the desert's ridged construction: 1-|f|
#: folding puts crease lines carrying the raw field's full gradient
#: everywhere, tolerable across a 40 m mountainside, measured at P99 52-61
#: deg when the wavelength is 10 m. Swept at (6, 20): slope P50/P90/P99 =
#: 8.3/19.5/29.8 deg (under the 36.9 deg wall), median 8 m-tile relief
#: 0.615 m of which 0.518 m survives de-tilting — a genuine hill about 1.5
#: standing heights tall in a typical tile, on every tile. `--stats`
#: reproduces the slope and relief numbers whenever they are doubted.
HILL_AMP = 0.16
RIPPLE_AMP = 0.05
RUBBLE_AMP = 0.02

#: Spawn pad, metres: flat inside r=2.0, cosine-blended to full terrain by
#: r=5.0. Same mechanism as the desert's (2.5, 6.0), slightly tighter because
#: there is less relief to blend away.
PAD_FLAT_M, PAD_BLEND_M = 2.0, 5.0


def build_height_m(seed: int) -> np.ndarray:
    """The composed gentle terrain in metres, spawn surface = 0.

    Same conventions as `generate.build_height_m`: returns (GRID, GRID),
    row axis = y, col axis = x, and the caller normalises for storage.
    """
    rng = np.random.default_rng(seed)
    n = GRID

    coords = np.linspace(-HALF_EXTENT, HALF_EXTENT, n)
    x, y = coords[None, :], coords[:, None]
    dist = np.sqrt(x * x + y * y)

    # -- Low hills -----------------------------------------------------------
    # A plain smooth field, NOT the desert's ridged-and-warped multifractal:
    # ridging folds crease lines into every wavelength, measured at P99
    # slopes of 52-61 deg at this band. beta=2.5 in a 6-20 m band puts a
    # crest AND a base inside every 8 m training tile — the constants' note
    # records why hills that tiling cannot see are not hills.
    hills = HILL_AMP * _spectral_field(rng, n, beta=2.5, band=(6.0, 20.0))

    # -- Ripples -------------------------------------------------------------
    # The desert's transverse-dune construction (band-passed anisotropic field,
    # 1-|f| crests, ^1.5 sharpening) at a 4-10 m wavelength instead of 14-30 m,
    # so several crests land inside one 8 m training tile — at the desert's
    # spacing a tile sees at most one, and the "quasi-regular ridges" character
    # is invisible at training scale.
    f_rip = _spectral_field(rng, n, beta=2.0, stretch=(1.0, 0.4), band=(4.0, 10.0))
    # ^1.2, not the desert's ^1.5: sharper crests measured as P99 slope past
    # the wall threshold at this wavelength. See the sweep note on the
    # amplitude constants.
    crest = np.clip(1.0 - np.abs(f_rip), 0.0, None) ** 1.2
    ripples = RIPPLE_AMP * (crest - crest.mean()) / crest.std()

    # -- Rubble --------------------------------------------------------------
    # The roughness knob. Band floor at 1.5 m (not the desert's 0.5) for two
    # measured reasons: the 0.1 m training resample cannot carry wavelengths
    # under ~0.8 m honestly, and even 0.8 m rubble at this amplitude pushed
    # slope P99 to 50 deg — short-wavelength height is slope, and slope is
    # what "gentle" bounds.
    rubble = RUBBLE_AMP * _spectral_field(rng, n, beta=1.5, band=(1.5, 3.5))

    h = hills + ripples + rubble

    # -- Spawn pad: flatten a disk at the origin, cosine blend outward -------
    w = np.clip((dist - PAD_FLAT_M) / (PAD_BLEND_M - PAD_FLAT_M), 0.0, 1.0)
    h *= 0.5 - 0.5 * np.cos(np.pi * w)

    # -- Exact declared span (see module docstring) --------------------------
    raw_span = float(h.max() - h.min())
    if raw_span <= 0.0:
        raise ValueError(f"composed field is flat (span {raw_span}); a layer amplitude is broken")
    h *= Z_SPAN_M / raw_span
    # Stash the factor for main()'s printout without changing the return type.
    build_height_m.last_rescale = Z_SPAN_M / raw_span  # type: ignore[attr-defined]
    return h


def _stats(h: np.ndarray) -> dict[str, float]:
    """The numbers that decide walkable-but-rough. Computed, never asserted."""
    gy, gx = np.gradient(h, CELL)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    tile = int(round(8.0 / CELL))  # cells per 8 m training tile
    reliefs = [
        float(np.ptp(h[i : i + tile, j : j + tile]))
        for i in range(0, GRID - tile + 1, tile)
        for j in range(0, GRID - tile + 1, tile)
    ]
    return {
        "span_m": float(np.ptp(h)),
        "std_m": float(h.std()),
        "slope_p50_deg": float(np.percentile(slope_deg, 50)),
        "slope_p90_deg": float(np.percentile(slope_deg, 90)),
        "slope_p99_deg": float(np.percentile(slope_deg, 99)),
        "slope_max_deg": float(slope_deg.max()),
        "tile8_relief_min_m": float(min(reliefs)),
        "tile8_relief_med_m": float(np.median(reliefs)),
        "tile8_relief_max_m": float(max(reliefs)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--stats", action="store_true", help="print numbers, write nothing")
    args = p.parse_args()

    h = build_height_m(args.seed)
    rescale = build_height_m.last_rescale  # type: ignore[attr-defined]

    print(f"grid {GRID}x{GRID} over {2 * HALF_EXTENT:.0f}x{2 * HALF_EXTENT:.0f} m "
          f"({CELL * 100:.1f} cm/cell), seed {args.seed}")
    print(f"declared span {Z_SPAN_M} m, rescale factor {rescale:.3f} "
          f"(pre-rescale span {Z_SPAN_M / rescale:.2f} m)")
    for name, amp in (("hills", HILL_AMP), ("ripples", RIPPLE_AMP), ("rubble", RUBBLE_AMP)):
        print(f"  {name:<8} authored {amp:.3f} -> effective std {amp * rescale:.3f} m")
    for k, v in _stats(h).items():
        print(f"  {k:<22} {v:8.3f}")

    if args.stats:
        return

    paths.TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    h_min, h_max = save_hfield_bin(h, paths.GENTLE_HFIELD)
    save_texture(h, paths.GENTLE_TEXTURE, args.seed)
    print(f"wrote {paths.GENTLE_HFIELD}")
    print(f"wrote {paths.GENTLE_TEXTURE}")
    # If this asset ever gets a MuJoCo XML, these are the numbers it must carry:
    print(f"XML check: <hfield ... size=\"{HALF_EXTENT:.0f} {HALF_EXTENT:.0f} "
          f"{h_max - h_min:.2f} 1.0\"/> and hfield geom pos = \"0 0 {h_min:.2f}\"")


if __name__ == "__main__":
    main()
