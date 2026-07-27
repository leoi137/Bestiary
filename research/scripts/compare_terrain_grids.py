"""Is GRID=2048 a finer sampling of the same desert, or a different desert?

The question matters because `robots/hound/build.py` proposes regenerating the
heightfield at GRID=2048 to fix a real defect: terrain cells are 7.82 cm across
and the wheel radius is 8.5 cm, so every wheel permanently straddles a cell
boundary, resolves against two prisms whose contact normals disagree, and
collects a ~5 cm/s net backward push. Four cells under each wheel would
decouple the scales.

The unstated assumption in "regenerate at a finer grid" is that the result is
the SAME terrain seen more sharply -- that a policy trained on one could be
compared against a policy trained on the other, with the finer mesh being
strictly better information about identical ground. If that assumption holds,
the regen is a cheap upgrade. If it does not, the regen is an instrument
change that silently invalidates every baseline in the record.

It does not hold, and the reason is three lines up in `generate.py`:

    phase = rng.uniform(0.0, 2.0 * np.pi, (n, n))

The spectral synthesis draws an (n, n) array of random phases. Change `n` and
every phase in every layer is redrawn from a different position in the stream.
The seed is held fixed and the terrain is still completely different -- not
resampled, not refined, REROLLED. Same statistics, different world.

This script measures how different, so the claim rests on a number rather than
on reading the source and being convinced.

    venv/bin/python research/scripts/compare_terrain_grids.py

Runs in well under a minute on CPU and writes nothing. It deliberately calls
`build_height_m` rather than `main()`, so no asset on disk is touched.
"""
from __future__ import annotations

import numpy as np

from bestiary.terrain import generate

# The two grids in question: what is committed today, and what build.py:123
# proposes. 2048 is the interesting one because 80 m / 2048 = 3.91 cm puts
# roughly four cells under an 8.5 cm wheel radius instead of one.
GRID_NOW = 1024
GRID_PROPOSED = 2048

# generate.py's own default. Held FIXED across both builds -- that is the whole
# point of the test. If the terrain differed only because the seed moved, this
# would measure nothing.
SEED = 7


def build_at(grid: int, seed: int) -> np.ndarray:
    """Build the height field at an arbitrary grid, in meters.

    `generate.GRID` and `generate.CELL` are module-level constants read at call
    time by `build_height_m` and `_spectral_field`. Rebinding them is therefore
    the supported way to build at another resolution, and restoring them in a
    `finally` is not optional -- leaving GRID=2048 bound would make every later
    import in the same process silently generate a different world.
    """
    saved_grid, saved_cell = generate.GRID, generate.CELL
    try:
        generate.GRID = grid
        generate.CELL = 2 * generate.HALF_EXTENT / grid
        return generate.build_height_m(seed)
    finally:
        generate.GRID, generate.CELL = saved_grid, saved_cell


def block_mean(field: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by averaging factor x factor blocks.

    Block mean rather than subsampling: subsampling would throw away exactly
    the sub-cell detail the finer grid exists to capture, which would bias the
    comparison toward agreement and make a different terrain look like the
    same one. Averaging is the honest reduction -- it asks "does the fine
    terrain, viewed at the coarse scale, look like the coarse terrain?"
    """
    n = field.shape[0]
    if n % factor:
        raise ValueError(f"{n} is not divisible by factor {factor}")
    m = n // factor
    return field.reshape(m, factor, m, factor).mean(axis=(1, 3))


def main() -> int:
    if GRID_PROPOSED % GRID_NOW:
        raise ValueError(
            f"GRID_PROPOSED={GRID_PROPOSED} is not an integer multiple of "
            f"GRID_NOW={GRID_NOW}; block_mean cannot align them"
        )

    coarse = build_at(GRID_NOW, SEED)
    fine = build_at(GRID_PROPOSED, SEED)

    if coarse.shape != (GRID_NOW, GRID_NOW):
        raise AssertionError(f"coarse build returned {coarse.shape}, expected square {GRID_NOW}")
    if fine.shape != (GRID_PROPOSED, GRID_PROPOSED):
        raise AssertionError(f"fine build returned {fine.shape}, expected square {GRID_PROPOSED}")

    fine_at_coarse = block_mean(fine, GRID_PROPOSED // GRID_NOW)

    corr = float(np.corrcoef(coarse.ravel(), fine_at_coarse.ravel())[0, 1])
    diff = fine_at_coarse - coarse
    max_abs = float(np.abs(diff).max())
    rms = float(np.sqrt((diff ** 2).mean()))

    cell_now = 2 * generate.HALF_EXTENT / GRID_NOW * 100
    cell_proposed = 2 * generate.HALF_EXTENT / GRID_PROPOSED * 100

    print(f"seed {SEED}, held fixed across both builds")
    print()
    for name, grid, field, cell in (
        ("committed", GRID_NOW, coarse, cell_now),
        ("proposed ", GRID_PROPOSED, fine, cell_proposed),
    ):
        span = float(field.max() - field.min())
        print(f"  {name} GRID={grid:<5d} {cell:.3f} cm/cell   "
              f"span {span:.4f} m   min {field.min():+.4f}   max {field.max():+.4f}")

    print()
    print(f"  agreement of the two at the COARSE scale ({GRID_NOW} x {GRID_NOW}):")
    print(f"    correlation      {corr:+.4f}     (1.0 = same terrain, 0.0 = unrelated)")
    print(f"    max |difference| {max_abs:.4f} m")
    print(f"    rms difference   {rms:.4f} m")
    print()

    # A refinement of the same ground would correlate near 1. Anything near 0
    # means the two share only their statistics. 0.5 is a generous line: well
    # below any honest reading of "the same terrain, sampled more finely".
    verdict = "SAME TERRAIN, REFINED" if corr > 0.5 else "A DIFFERENT TERRAIN"
    print(f"  verdict: {verdict}")
    if corr <= 0.5:
        print("    Changing GRID rerolls generate.py's (n, n) phase array, so every")
        print("    spectral layer is redrawn. The seed is fixed and the world still")
        print("    changes. Any policy comparison across this regen is an instrument")
        print("    change, and the zero-action baseline must be re-measured with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
