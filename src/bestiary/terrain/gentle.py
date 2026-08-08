"""The gentle terrain: the desert's own recipe with the mountains turned down.

    venv/bin/python -m bestiary.terrain.gentle            # writes the V5 asset
    venv/bin/python -m bestiary.terrain.gentle --stats    # numbers only, writes nothing

Writes `assets/terrain/gentle_v5_hfield.bin` (+ texture + preview PNG) in the
same MuJoCo custom-heightfield format as the desert, over the same 80 x 80 m
footprint at the same 7.8 cm sampling — so `terrain/isaac_hf.py` serves it to
Isaac Lab through the exact machinery the desert already uses, and nothing
downstream learns a second file format.

FIFTH ITERATION (2026-08-07): v4's ridged crests measured up to 47 deg on the
0.5 m-smoothed surface — past any angle of repose, knife-edged by the
`1-|f|` crease in both ridged layers. The operator reviewed a measured
4-candidate sweep at true robot scale and picked C1 "smooth-ridge": the same
world (seed, layers, layout) with every crease rounded (`ridge_eps` = 0.30 in
`generate.build_height_m`), the fine ridged layer made additive instead of
multiplied onto crests, and a 36 deg repose cap as a backstop guarantee.
**The committed v4 files (`gentle_hfield.bin` etc.) are untouched and stay** —
every Spyder policy so far trained on those exact bytes; this module now
generates v5 alongside, and the env cfgs keep pointing at v4 until the next
training arm deliberately switches (terrain is a one-way door). The v4 asset
no longer reproduces from this module; it needs `ridge_eps=0,
fine_additive=False` against the layer history in git.

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
#: 7.4 ft (2.25 m) against the desert's 16.6 ft (5.05 m): mountains become
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

#: v5 crest rounding, passed to `generate.build_height_m`. 0.30 was picked by
#: the measured 4-candidate sweep of 2026-08-07 (crest radius 6.2 -> 7.2 ft at
#: unchanged relief) and confirmed by the operator at true robot scale.
RIDGE_EPS = 0.30

#: Backstop, degrees: after composition NO 0.5 m-scale slope may exceed this.
#: Dry granular repose is 30-34 deg; 36 leaves margin for embedded rock while
#: killing the 47 deg crests v4 shipped with. Enforced by _repose_limit, and
#: loudly re-checked after the final span rescale.
REPOSE_CAP_DEG = 36.0


def _repose_limit(h: np.ndarray, cap_deg: float) -> np.ndarray:
    """Clamp every cell to min over 8 neighbours of (neighbour + repose rise).

    Morphological cone erosion — the talus operator. Iterated to convergence;
    afterwards no directional cell-to-cell slope exceeds `cap_deg`. Periodic
    edges, matching the FFT-synthesised field.
    """
    rise = np.tan(np.radians(cap_deg)) * CELL
    rise_d = rise * np.sqrt(2.0)
    m = h.copy()
    for _ in range(600):
        prev = m
        cand = np.full_like(m, np.inf)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            np.minimum(cand, np.roll(m, shift, axis=axis) + rise, out=cand)
        for sx in (1, -1):
            for sy in (1, -1):
                np.minimum(cand, np.roll(np.roll(m, sx, 0), sy, 1) + rise_d, out=cand)
        m = np.minimum(m, cand)
        if float(np.max(prev - m)) < 1e-9:
            return m
    raise ValueError(f"repose limit did not converge in 600 sweeps at {cap_deg} deg")


def build_height_m(seed: int) -> np.ndarray:
    """The v5 field: desert recipe at MOUNTAIN_AMP with rounded crests, capped
    at REPOSE_CAP_DEG, rescaled to the exact declared span.

    Same conventions as `generate.build_height_m`: (GRID, GRID), row axis =
    y, col axis = x, spawn surface at 0 before normalisation. The cap runs
    AFTER the span rescale (rescaling amplifies slopes, so capping first would
    ship steeper ground than the cap names), then the tiny relief the cap
    removes is restored by one more rescale+cap round and verified loudly.
    """
    h = _desert_recipe(seed, mountain_amp=MOUNTAIN_AMP,
                       ridge_eps=RIDGE_EPS, fine_additive=True)
    raw_span = float(np.ptp(h))
    if raw_span <= 0.0:
        raise ValueError(f"composed field is flat (span {raw_span}); the recipe is broken")
    build_height_m.last_rescale = Z_SPAN_M / raw_span  # type: ignore[attr-defined]
    for _ in range(2):
        h *= Z_SPAN_M / np.ptp(h)
        h = _repose_limit(h, REPOSE_CAP_DEG)
    h *= Z_SPAN_M / np.ptp(h)

    worst = float(np.max(np.abs(np.diff(h, axis=0)))) / CELL
    worst = max(worst, float(np.max(np.abs(np.diff(h, axis=1)))) / CELL)
    worst_deg = float(np.degrees(np.arctan(worst)))
    if worst_deg > REPOSE_CAP_DEG + 1.0:
        raise ValueError(
            f"final rescale re-steepened the field to {worst_deg:.2f} deg, past the "
            f"{REPOSE_CAP_DEG} deg cap + 1 deg tolerance — raise the cap rounds"
        )
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


#: 1 ft = 0.3048 m exactly (NIST Handbook 44). Display conversion only — every
#: computation in this module stays in metres.
_FT_PER_M = 1.0 / 0.3048


def write_preview(h: np.ndarray, path=None) -> None:
    """Bake the operator-facing preview PNG next to the asset.

    Labels read US-first with SI in parentheses — the workspace units
    convention for anything explained rather than computed. The heightfield
    itself stays metres, and this function writes ONLY the PNG: the pinned
    ``.bin`` (terrain-spec hash) cannot move from here. `path` defaults to
    the v5 preview; pass `paths.GENTLE_PREVIEW` to re-bake v4's from its bin.
    """
    if path is None:
        path = paths.GENTLE_V5_PREVIEW
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    half_ft = HALF_EXTENT * _FT_PER_M
    ls = LightSource(azdeg=315, altdeg=40)
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[2.4, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    rgb = ls.shade(h, cmap=plt.cm.gist_earth, vert_exag=2.0, blend_mode="soft",
                   dx=CELL, dy=CELL)
    ax.imshow(rgb, extent=[-half_ft, half_ft, -half_ft, half_ft], origin="lower")
    ax.set_title(
        f"gentle — desert recipe, mountains at {MOUNTAIN_AMP}/3.3, "
        f"span {np.ptp(h) * _FT_PER_M:.1f} ft ({np.ptp(h):.2f} m)"
    )
    ax.set_xlabel("x (ft)")
    ax.set_ylabel("y (ft)")

    ax2 = fig.add_subplot(gs[0, 1])
    half_zoom = int(round(12.5 / CELL))  # 82x82 ft (25x25 m) window
    mid = GRID // 2
    zoom = h[mid - half_zoom : mid + half_zoom, mid - half_zoom : mid + half_zoom]
    zoom_ft = 12.5 * _FT_PER_M
    rgb2 = ls.shade(zoom, cmap=plt.cm.gist_earth, vert_exag=2.0, blend_mode="soft",
                    dx=CELL, dy=CELL)
    ax2.imshow(rgb2, extent=[-zoom_ft, zoom_ft, -zoom_ft, zoom_ft], origin="lower")
    ax2.set_title("82x82 ft (25x25 m) around the spawn pad")
    ax2.set_xlabel("ft")

    ax3 = fig.add_subplot(gs[1, 1])
    y_cut_m = 14.7  # same cut the first hand-made preview showed
    row = h[int(round((y_cut_m + HALF_EXTENT) / CELL))] * _FT_PER_M
    x_ft = np.linspace(-half_ft, half_ft, GRID)
    ax3.fill_between(x_ft, row, row.min() - 0.3, color="tan")
    ax3.set_xlabel("x (ft)")
    ax3.set_ylabel("z (ft)")

    fig.tight_layout()

    # Title the cut with its measured vertical exaggeration, so nobody reads
    # display steepness as ground steepness again (it once read ~17x too
    # sharp). Computed from the final axes geometry, after tight_layout.
    box = ax3.get_position()
    fig_w, fig_h = fig.get_size_inches()
    x_ft_per_inch = (2 * half_ft) / (box.width * fig_w)
    z_ft_per_inch = float(np.ptp(row) + 0.3) / (box.height * fig_h)
    ax3.set_title(
        f"cross-section at y = +{y_cut_m * _FT_PER_M:.0f} ft (+{y_cut_m} m) — "
        f"z stretched ~{x_ft_per_inch / z_ft_per_inch:.0f}x"
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--stats", action="store_true", help="print numbers, write nothing")
    p.add_argument("--preview", action="store_true",
                   help="rewrite the preview PNG only; never touches the .bin")
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

    if args.preview:
        write_preview(h)
        print(f"wrote {paths.GENTLE_V5_PREVIEW} (preview only; .bin untouched)")
        return

    h_min, h_max = save_hfield_bin(h, paths.GENTLE_V5_HFIELD)
    save_texture(h, paths.GENTLE_V5_TEXTURE, args.seed)
    write_preview(h)
    print(f"wrote {paths.GENTLE_V5_HFIELD}")
    print(f"wrote {paths.GENTLE_V5_TEXTURE}")
    print(f"wrote {paths.GENTLE_V5_PREVIEW}")
    print(f"XML check: <hfield ... size=\"{HALF_EXTENT:.0f} {HALF_EXTENT:.0f} "
          f"{h_max - h_min:.2f} 1.0\"/> and hfield geom pos = \"0 0 {h_min:.2f}\"")


if __name__ == "__main__":
    main()
