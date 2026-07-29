"""Assertions on the desert-to-Isaac-Lab bridge. The oracle for `isaac_hf`.

WHY THIS EXISTS AND WHAT IT PROTECTS

`terrain/isaac_hf.py` is the third reader of `assets/terrain/desert_hfield.bin`,
and a terrain disagreement is the quietest failure this project has: every
checkpoint still loads, every guard stays green, and every ledger row silently
becomes incomparable. The cheapest thing that makes that class of bug
impossible — rather than merely fixed once — is a check that fails loudly when
the bridge and the asset stop describing the same ground.

The load-bearing assertion is `check_constants_match_xml`. `isaac_hf` hardcodes
the desert's extent, elevation span and grid because it must parse the `.bin`
without MuJoCo. Those three numbers are copied from `assets/hound16pd_desert.xml`
and `terrain/generate.py`, and a copy is a thing that goes stale. So this check
re-reads the `<hfield>` element and fails if they have drifted — which is
exactly what would happen the day the desert is regenerated at GRID=2048, the
change `research/scripts/compare_terrain_grids.py` measured at +0.0610
correlation with the committed field.

Run under the ISAAC LAB interpreter (it imports isaaclab for the decorator):

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.check_desert_terrain

Exit status is 0 on all-pass, 1 on any failure, so it can gate a launch.
"""

from __future__ import annotations

import re
import sys
import traceback
from typing import Callable

import numpy as np

from bestiary import paths
from bestiary.terrain.isaac_hf import (
    DESERT_GRID,
    DESERT_HALF_EXTENT_M,
    DESERT_NATIVE_CELL_M,
    DESERT_Z_SPAN_M,
    HfBestiaryDesertTerrainCfg,
    bestiary_desert_terrain,
    load_desert_m,
)

#: Tile size the checks build. 8 m matches Isaac Lab's shipped rough config.
TILE_M = 8.0


def _cfg(**overrides) -> HfBestiaryDesertTerrainCfg:
    """A tile config at the desert's own resolution, so no resampling hides bugs."""
    kwargs = dict(
        size=(TILE_M, TILE_M),
        horizontal_scale=DESERT_NATIVE_CELL_M,
        vertical_scale=0.005,
        slope_threshold=0.75,
        border_width=0.25,
        hfield_path=str(paths.DESERT_HFIELD),
    )
    kwargs.update(overrides)
    return HfBestiaryDesertTerrainCfg(**kwargs)


def check_constants_match_xml() -> None:
    """`isaac_hf`'s geometry constants still match the committed `<hfield>`.

    This is the check that matters. Everything else here tests behaviour; this
    one tests that the behaviour is about the right ground.
    """
    xml = paths.HOUND_PD_DESERT_XML.read_text()
    m = re.search(r'<hfield\b[^>]*\bsize="([^"]+)"', xml)
    if m is None:
        raise AssertionError(
            f"no <hfield ... size=\"...\"> element found in {paths.HOUND_PD_DESERT_XML}; "
            "isaac_hf's constants cannot be verified against the asset"
        )
    parts = [float(v) for v in m.group(1).split()]
    if len(parts) != 4:
        raise AssertionError(
            f'<hfield size="{m.group(1)}"> has {len(parts)} values, expected 4 '
            "(x half-extent, y half-extent, z span, base)"
        )
    half_x, half_y, z_span, _base = parts
    if half_x != half_y:
        raise AssertionError(
            f"desert is not square: half-extents {half_x} x {half_y}. isaac_hf's "
            "DESERT_HALF_EXTENT_M assumes one value for both axes."
        )
    if half_x != DESERT_HALF_EXTENT_M:
        raise AssertionError(
            f"XML half-extent is {half_x} m but isaac_hf.DESERT_HALF_EXTENT_M is "
            f"{DESERT_HALF_EXTENT_M} m — the desert was resized and the bridge was not updated"
        )
    if z_span != DESERT_Z_SPAN_M:
        raise AssertionError(
            f"XML elevation span is {z_span} m but isaac_hf.DESERT_Z_SPAN_M is "
            f"{DESERT_Z_SPAN_M} m — every height this bridge produces is scaled wrong "
            f"by a factor of {z_span / DESERT_Z_SPAN_M:.6f}"
        )
    # GRID is not in the XML; it is implied by the .bin header, checked below.


def check_asset_header() -> None:
    """The `.bin` parses, and its grid is the one the constants describe."""
    z = load_desert_m(paths.DESERT_HFIELD)
    if z.shape != (DESERT_GRID, DESERT_GRID):
        raise AssertionError(f"desert is {z.shape}, expected ({DESERT_GRID}, {DESERT_GRID})")
    if z.min() != 0.0:
        raise AssertionError(f"load_desert_m must shift the minimum to 0.0, got {z.min()!r}")
    relief = float(np.ptp(z))
    # The asset is normalised to [0, 1] and scaled by the span, so a desert that
    # uses its full range has relief == span. Anything far below means the
    # normalisation or the span is wrong.
    if not (0.9 * DESERT_Z_SPAN_M <= relief <= DESERT_Z_SPAN_M + 1e-6):
        raise AssertionError(
            f"desert relief is {relief:.6f} m, which is not consistent with a field "
            f"normalised to [0,1] and scaled by DESERT_Z_SPAN_M={DESERT_Z_SPAN_M} m"
        )


def check_native_cell() -> None:
    """The derived cell size is the ~7.8 cm `generate.py` documents."""
    expected = 2.0 * DESERT_HALF_EXTENT_M / DESERT_GRID
    if DESERT_NATIVE_CELL_M != expected:
        raise AssertionError(
            f"DESERT_NATIVE_CELL_M is {DESERT_NATIVE_CELL_M} but "
            f"2*{DESERT_HALF_EXTENT_M}/{DESERT_GRID} = {expected}"
        )


def check_mesh_is_produced() -> None:
    """A tile is a non-degenerate mesh with real relief."""
    meshes, origin = bestiary_desert_terrain(0.5, _cfg())
    if len(meshes) != 1:
        raise AssertionError(f"expected exactly 1 mesh per tile, got {len(meshes)}")
    mesh = meshes[0]
    if len(mesh.vertices) < 1000:
        raise AssertionError(
            f"tile has only {len(mesh.vertices)} vertices; an {TILE_M} m tile at "
            f"{DESERT_NATIVE_CELL_M:.6f} m/cell should have ~10^4"
        )
    relief = float(np.ptp(mesh.vertices[:, 2]))
    if relief < 0.05:
        raise AssertionError(
            f"tile relief is {relief:.4f} m — essentially flat. The desert has metres "
            "of relief, so this means the patch, the vertical scale, or the "
            "normalisation is wrong."
        )
    if not np.isfinite(origin).all():
        raise AssertionError(f"terrain origin is not finite: {origin!r}")


def check_difficulty_is_monotone() -> None:
    """Harder difficulty means rougher real ground, which is the curriculum's premise."""
    stds: list[float] = []
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        meshes, _ = bestiary_desert_terrain(d, _cfg())
        stds.append(float(meshes[0].vertices[:, 2].std()))
    # Patches come from measured terrain, so demand the trend rather than a
    # strictly increasing sequence: ranking is by patch std, but the mesh std
    # after bordering and interpolation can wobble slightly between neighbours.
    if stds[-1] <= stds[0]:
        raise AssertionError(
            f"roughness did not increase with difficulty: {stds[0]:.4f} m at 0.0 vs "
            f"{stds[-1]:.4f} m at 1.0 (full series {[round(s, 4) for s in stds]})"
        )
    if stds[-1] < 2.0 * stds[0]:
        raise AssertionError(
            f"difficulty barely changes the ground: std {stds[0]:.4f} -> {stds[-1]:.4f} m. "
            "The curriculum is supposed to span gentle to severe real terrain."
        )


def check_deterministic() -> None:
    """The same difficulty must always select the same patch.

    Two runs labelled 'difficulty 0.5' that stood on different ground are not
    comparable, and nothing downstream would notice.
    """
    a, _ = bestiary_desert_terrain(0.5, _cfg())
    b, _ = bestiary_desert_terrain(0.5, _cfg())
    za, zb = a[0].vertices[:, 2], b[0].vertices[:, 2]
    if za.shape != zb.shape or not np.array_equal(np.sort(za), np.sort(zb)):
        raise AssertionError(
            "two calls at difficulty 0.5 produced different terrain; patch selection "
            "is not deterministic and every comparison across runs is invalid"
        )


def check_rejects_missing_asset() -> None:
    """A wrong path fails loudly instead of silently producing flat ground."""
    try:
        load_desert_m(paths.TERRAIN_DIR / "does_not_exist.bin")
    except FileNotFoundError:
        return
    raise AssertionError(
        "load_desert_m accepted a nonexistent heightfield without raising "
        "FileNotFoundError; a typo in a path would train on the wrong ground"
    )


def check_tile_larger_than_desert_is_rejected() -> None:
    """A tile the asset cannot cover must raise, not silently tile or wrap."""
    too_big = 2.0 * DESERT_HALF_EXTENT_M * 2.0  # twice the desert's width
    try:
        bestiary_desert_terrain(0.5, _cfg(size=(too_big, too_big)))
    except ValueError:
        return
    raise AssertionError(
        f"a {too_big:.0f} m tile was accepted from an "
        f"{2 * DESERT_HALF_EXTENT_M:.0f} m desert without raising ValueError"
    )


CHECKS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("constants match <hfield> in the XML", check_constants_match_xml),
    ("asset header parses, grid as declared", check_asset_header),
    ("native cell size is derived, not guessed", check_native_cell),
    ("a tile is a real mesh with real relief", check_mesh_is_produced),
    ("difficulty raises roughness", check_difficulty_is_monotone),
    ("patch selection is deterministic", check_deterministic),
    ("missing asset raises", check_rejects_missing_asset),
    ("oversized tile raises", check_tile_larger_than_desert_is_rejected),
)


def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- a check reports, it does not crash the suite
            failures += 1
            print(f"FAIL  {name}", flush=True)
            print(f"      {type(exc).__name__}: {exc}", flush=True)
            if not isinstance(exc, AssertionError):
                traceback.print_exc()
        else:
            print(f"ok    {name}", flush=True)
    total = len(CHECKS)
    print(f"\n{total - failures}/{total} checks pass", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
