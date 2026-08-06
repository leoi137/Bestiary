"""The gentle terrain: the desert's own recipe with the mountains turned down.

    venv/bin/python -m bestiary.terrain.gentle            # writes the asset
    venv/bin/python -m bestiary.terrain.gentle --stats    # numbers only, writes nothing

Writes `assets/terrain/gentle_hfield.bin` (+ a baked-albedo preview PNG) in the
same MuJoCo custom-heightfield format as the desert, over the same 80 x 80 m
footprint at the same 7.8 cm sampling — so `terrain/isaac_hf.py` serves it to
Isaac Lab through the exact machinery the desert already uses, and nothing
downstream learns a second file format.

WHAT THIS IS, FOURTH ITERATION, AND WHY THE FIRST THREE DIED
------------------------------------------------------------
The brief (operator, 2026-08-05/06): the desert's look — hills, ripples,
gravel — for a machine that is *learning* to walk, "like before, just not as
intense". Three original compositions preceded this file's current form, and
each died on a measurement worth keeping:

1. Custom layers, rubble band down to 0.8 m: slope P99 = 50.4 deg —
   short-wavelength height IS slope; accidental mountains at gravel scale.
2. Hills in a 15-40 m band at a fifth the desert's amplitude: the operator
   saw NO hills in the viewer, correctly. Training and viewing happen on 8 m
   tiles, each re-zeroed to its own baseline; of that layer's 0.283 m median
   per-tile contribution only 0.161 m survived removing the best-fit plane.
   LOW-amplitude long wavelengths are erased by tiling. (The desert's are
   not, because 5 m of relief makes even one tile of mountainside steep —
   amplitude is what lets a long wavelength survive the crop.)
3. Hills moved into a 6-20 m, then 10-30 m band so a crest and base fit
   inside a tile: mechanically real (0.43-0.52 m of per-tile shape), and the
   operator's screenshots settled the verdict — at human viewing distance a
   half-metre hill is 1-2% of the frame, i.e. crumpled paper. The robot's
   scale and the operator's scale disagree, and the operator is the one
   deciding whether the world looks right.

So iteration 4 stops composing its own landscape: `generate.build_height_m`
— the desert itself, every layer, every band — with its `mountain_amp` knob
at MOUNTAIN_AMP below instead of the desert's 3.3, on this asset's own seed.
Dunes, domain-warped ridged forms, gravel: identical morphology, the
mountains now ~2 m instead of ~4.5 m. Against the operator's two reference
screenshots this is the midpoint: hills that visibly tower over a 0.35 m
spider (~6x its height; the desert towers ~10x over the 0.36 m Hound)
without the desert's canyon walls.

WHAT "LESS INTENSE" MEANS, MEASURED AT BODY SCALE
-------------------------------------------------
Cell-scale slope percentiles mislead here: the recipe's 0.5-3 m gravel makes
steep CELLS that the simulator's slope threshold turns into centimetre
pebble-steps, not walls. The metric that decides walkability is the slope of
the 0.5 m-smoothed surface (what the body must climb) and the per-cell step
height (what a foot must clear):

                          body-slope P50/P90/P99      foot-step P99   span
    desert (committed)      18.9 / 47.7 / 61.4 deg       28.3 cm     5.05 m
    THIS ASSET              12.5 / 23.4 / 34.3 deg       14.0 cm     2.24 m

Half the span, half the body-slope at P90, half the step height. The hard
end is genuinely hard — 2 m hill flanks near 30 deg — and the ranked-patch
curriculum in `isaac_hf.py` orders tiles flattest-first, so a learner starts
on dune flats and earns the flanks. That is the same arrangement under which
the full desert trains ANYmal-C.

THE SPAN IS DECLARED, NOT EMERGENT
----------------------------------
`Z_SPAN_M` is exact by construction: the composed field is rescaled so its
min-to-max span equals the constant (the factor is ~1.004 — the knobbed
recipe's natural span is 2.24). The `.bin` stores samples normalised to
[0, 1] and cannot carry metres, so every reader must be told the span;
`isaac_hf.py` takes it as a config field and the env cfg passes this
constant. Keep them one constant.
"""

from __future__ import annotations

import argparse

import numpy as np

from bestiary import paths
from bestiary.terrain.generate import (
    CELL,
    GRID,
    HALF_EXTENT,
    build_height_m as _desert_recipe,
    save_hfield_bin,
    save_texture,
)

#: Total elevation span, metres, EXACT by rescale (see module docstring).
#: 2.25 m (7.4 ft) against the desert's 5.05 m (16.6 ft): mountains become
#: hills a learning spider can be dwarfed by and still climb.
Z_SPAN_M = 2.25

#: Generation seed. Arbitrary, fixed, and deliberately NOT the desert's 7 —
#: sharing a seed would correlate the two worlds' large-scale forms, and the
#: point of a second asset is a second world.
SEED = 11

#: The one knob: the desert's mountain layer runs at 3.3; this asset runs it
#: at 1.0. Chosen by measurement (the body-scale table in the module
#: docstring) from a sweep over {1.0, 1.3, 1.6}: 1.3 already puts body-slope
#: P99 at 39 deg — past the 36.9 deg wall — on the flanks. Dunes, warp,
#: gravel and the spawn pad are the desert's own, untouched.
MOUNTAIN_AMP = 1.0


def build_height_m(seed: int) -> np.ndarray:
    """The desert recipe at MOUNTAIN_AMP, rescaled to the exact declared span.

    Same conventions as `generate.build_height_m`: (GRID, GRID), row axis =
    y, col axis = x, spawn surface at 0 before normalisation.
    """
    h = _desert_recipe(seed, mountain_amp=MOUNTAIN_AMP)
    raw_span = float(np.ptp(h))
    if raw_span <= 0.0:
        raise ValueError(f"composed field is flat (span {raw_span}); the recipe is broken")
    h *= Z_SPAN_M / raw_span
    build_height_m.last_rescale = Z_SPAN_M / raw_span  # type: ignore[attr-defined]
    return h


def _stats(h: np.ndarray) -> dict[str, float]:
    """The numbers that decide walkable-but-dramatic. Computed, not asserted.

    Body-slope is measured on the 0.5 m-smoothed surface (FFT Gaussian, pure
    numpy) because cell-scale slope on this recipe is dominated by gravel the
    simulator renders as pebble-steps — the module docstring's argument.
    """
    k = np.fft.fftfreq(GRID, d=CELL)
    kx, ky = np.meshgrid(k, k)
    smooth = np.fft.ifft2(
        np.fft.fft2(h) * np.exp(-2.0 * np.pi**2 * 0.5**2 * (kx**2 + ky**2))
    ).real
    gy, gx = np.gradient(smooth, CELL)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    step = np.maximum(
        np.abs(np.diff(h, axis=0))[:, :-1], np.abs(np.diff(h, axis=1))[:-1, :]
    )
    tile = int(round(8.0 / CELL))
    reliefs = [
        float(np.ptp(h[i : i + tile, j : j + tile]))
        for i in range(0, GRID - tile + 1, tile)
        for j in range(0, GRID - tile + 1, tile)
    ]
    return {
        "span_m": float(np.ptp(h)),
        "body_slope_p50_deg": float(np.percentile(slope_deg, 50)),
        "body_slope_p90_deg": float(np.percentile(slope_deg, 90)),
        "body_slope_p99_deg": float(np.percentile(slope_deg, 99)),
        "foot_step_p99_cm": float(np.percentile(step, 99) * 100.0),
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
    print(f"desert recipe, mountain_amp {MOUNTAIN_AMP} (desert: 3.3); declared span "
          f"{Z_SPAN_M} m, rescale factor {rescale:.3f}")
    for key, value in _stats(h).items():
        print(f"  {key:<22} {value:8.3f}")

    if args.stats:
        return

    paths.TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    h_min, h_max = save_hfield_bin(h, paths.GENTLE_HFIELD)
    save_texture(h, paths.GENTLE_TEXTURE, args.seed)
    print(f"wrote {paths.GENTLE_HFIELD}")
    print(f"wrote {paths.GENTLE_TEXTURE}")
    print(f"XML check: <hfield ... size=\"{HALF_EXTENT:.0f} {HALF_EXTENT:.0f} "
          f"{h_max - h_min:.2f} 1.0\"/> and hfield geom pos = \"0 0 {h_min:.2f}\"")


if __name__ == "__main__":
    main()
