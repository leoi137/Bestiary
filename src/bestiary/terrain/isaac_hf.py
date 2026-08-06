"""The Bestiary desert, served to Isaac Lab as a sub-terrain.

WHY THIS FILE EXISTS

`terrain/generate.py` composes a desert and writes it as a MuJoCo custom binary
heightfield. `terrain/field.py` reads it back out of a *compiled MuJoCo model*
to answer "how high is the ground at (x, y)". Isaac Lab needs the same samples
but asks for them a different way: a callable that returns a discretised height
grid, which its `height_field_to_mesh` decorator triangulates into a trimesh.

So this module is a third reader of the same bytes, and that is the whole risk
it has to manage. Two readers of one file are two chances to disagree about
which ground a run stood on, and a terrain disagreement is the quietest failure
in this project -- every checkpoint still loads, every guard stays green, and
every ledger row silently becomes incomparable (see the terrain invariant in
CLAUDE.md). The defence here is that this reader parses the *committed asset
bytes* directly rather than re-deriving elevation from `generate.py`'s
synthesis, and that it refuses to guess: a header that disagrees with the file
length, or normalised samples outside [0, 1], raise with the actual numbers.

DELIBERATELY NOT A MUJOCO IMPORT

`field.py` needs `mujoco` because it reads a compiled model. This module reads
the `.bin` with `struct` and `numpy` only, so it can be imported by the Isaac
Lab interpreter without installing MuJoCo, torch, or the rest of the Bestiary
dependency set into that environment. That independence is the point; do not
"simplify" this by importing `bestiary.terrain.field`.

WHAT A SUB-TERRAIN IS, AND WHY THE DESERT IS CROPPED

Isaac Lab builds a grid of small tiles (`TerrainGeneratorCfg.size`, 8x8 m in
the shipped rough config) and drops robots on them. The desert is 80x80 m. It
is therefore **cropped, not rescaled**: squeezing 80 m of dunes into an 8 m
tile would multiply every slope by ten and produce terrain no machine could
walk, while a crop at native resolution keeps the real gradients, the real
dune wavelength, and the real 7.8 cm sampling.

The consequence to be honest about: a tile is a *patch* of the desert, not the
desert. A policy trained here has seen the same ground statistics, not the same
80x80 m world a MuJoCo run saw. Those are different claims and the ledger
should not conflate them.

HOW DIFFICULTY IS HONOURED

Isaac Lab passes `difficulty` in [0, 1] and expects harder terrain as it rises.
Rather than invent a difficulty knob, this module ranks candidate patches of
the real desert by elevation standard deviation and indexes that ranking by
difficulty -- so difficulty 0 is the flattest real ground in the asset and
difficulty 1 is the roughest. The curriculum is therefore made of measured
terrain rather than synthesised terrain, which is the only version of it that
transfers a claim back to the MuJoCo runs.
"""

from __future__ import annotations

import struct
from dataclasses import MISSING
from pathlib import Path

import numpy as np
from scipy import interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass

# ---------------------------------------------------------------------------
# The geometry of the committed asset. Every number here is READ FROM a file in
# this repository, not chosen in this module -- if the desert is regenerated at
# a different extent or grid, these are wrong and `load_desert_m` will say so.
#
#   assets/hound16pd_desert.xml:85
#       <hfield name="desert" file="terrain/desert_hfield.bin"
#               size="40 40 5.05 1.0"/>
#   -> x half-extent 40 m, y half-extent 40 m, elevation span 5.05 m
#
#   src/bestiary/terrain/generate.py:67
#       GRID = 1024
#   -> 2*40/1024 = 0.078125 m per cell, the "~7.8 cm" that file's comment cites
# ---------------------------------------------------------------------------
DESERT_HALF_EXTENT_M = 40.0
DESERT_Z_SPAN_M = 5.05
DESERT_GRID = 1024
DESERT_NATIVE_CELL_M = 2.0 * DESERT_HALF_EXTENT_M / DESERT_GRID  # 0.078125

#: Bytes of MuJoCo's custom heightfield header: two int32, nrow then ncol.
#: Written by `generate.py:169`  f.write(struct.pack("ii", *data.shape)).
_HEADER_BYTES = 8

# Parsed heightfields, keyed by (resolved path, z-span). Isaac Lab calls the
# terrain function once per tile (200 tiles in the shipped 10x20 config), and
# re-reading and re-ranking a 4 MiB asset 200 times is pure waste. The span is
# part of the key because the SAME bytes at a different declared span are a
# different world, and a cache that ignored that would hand one caller the
# other's ground.
_DESERT_CACHE: dict[tuple[Path, float], np.ndarray] = {}
_RANK_CACHE: dict[tuple[Path, float, int, int, int], list[tuple[int, int]]] = {}


def load_desert_m(path: str | Path, z_span_m: float = DESERT_Z_SPAN_M) -> np.ndarray:
    """Elevation of a committed heightfield asset, in metres, indexed ``[x, y]``.

    MuJoCo stores heightfield samples normalised to [0, 1] and multiplies by
    the ``<hfield>`` z-size at load, so metres are ``value * z_span_m`` — the
    ``.bin`` format cannot carry its own span, so the caller must know it (the
    desert's 5.05 comes from a committed ``<hfield>`` attribute; the gentle
    terrain's 1.0 from ``terrain/gentle.py:Z_SPAN_M``, exact by construction).
    The default keeps every desert caller unchanged.
    The returned array is shifted so its minimum is exactly 0.0 -- Isaac Lab
    places tiles on its own ground plane and has no use for MuJoCo's
    ``geom pos="0 0 -0.41"`` offset.

    The transpose matters and is easy to get wrong. `field.py` documents the
    on-disk layout as ``data[row, col]`` with *columns spanning x* and *rows
    spanning y*; Isaac Lab documents its height grids as ``(width, length)``,
    i.e. x first. So the array is transposed once, here, and every caller
    downstream can index ``[x, y]`` without thinking about it.

    Raises:
        FileNotFoundError: the asset is missing.
        ValueError: the header disagrees with the file length, the grid is not
            the square ``DESERT_GRID`` this module's constants describe, or the
            samples are not normalised to [0, 1].
    """
    if not z_span_m > 0.0:
        raise ValueError(f"z_span_m must be positive, got {z_span_m}")
    resolved = Path(path).expanduser().resolve()
    cached = _DESERT_CACHE.get((resolved, z_span_m))
    if cached is not None:
        return cached

    if not resolved.is_file():
        raise FileNotFoundError(f"desert heightfield not found at {resolved}")

    raw = resolved.read_bytes()
    if len(raw) < _HEADER_BYTES:
        raise ValueError(
            f"{resolved} is {len(raw)} bytes, too short for the {_HEADER_BYTES}-byte "
            "MuJoCo heightfield header (int32 nrow, int32 ncol)"
        )
    nrow, ncol = struct.unpack_from("ii", raw, 0)
    expected = _HEADER_BYTES + nrow * ncol * 4  # float32 samples
    if len(raw) != expected:
        raise ValueError(
            f"{resolved}: header declares {nrow}x{ncol} float32 samples, so the file "
            f"should be {expected} bytes, but it is {len(raw)}"
        )
    if (nrow, ncol) != (DESERT_GRID, DESERT_GRID):
        raise ValueError(
            f"{resolved} is {nrow}x{ncol}, but this module's geometry constants "
            f"describe a {DESERT_GRID}x{DESERT_GRID} grid over "
            f"{2 * DESERT_HALF_EXTENT_M:.0f}x{2 * DESERT_HALF_EXTENT_M:.0f} m. "
            "If the desert was regenerated at a new GRID, update "
            "DESERT_GRID/DESERT_HALF_EXTENT_M/DESERT_Z_SPAN_M from the <hfield> "
            "element and generate.py rather than relaxing this check."
        )

    normalised = np.frombuffer(
        raw, dtype=np.float32, count=nrow * ncol, offset=_HEADER_BYTES
    ).reshape(nrow, ncol)
    lo, hi = float(normalised.min()), float(normalised.max())
    if lo < -1e-6 or hi > 1.0 + 1e-6:
        raise ValueError(
            f"{resolved} samples span [{lo:.6f}, {hi:.6f}], but MuJoCo custom "
            "heightfields are normalised to [0, 1]. Either the file is not this "
            "format or generate.py's normalisation changed."
        )

    # [row=y, col=x] on disk -> [x, y] for Isaac Lab. See the docstring.
    metres = (normalised.astype(np.float64) * z_span_m).T.copy()
    metres -= metres.min()
    _DESERT_CACHE[(resolved, z_span_m)] = metres
    return metres


def _rank_patches_by_roughness(
    desert_m: np.ndarray, patch_x: int, patch_y: int, stride: int, path: Path,
    z_span_m: float = DESERT_Z_SPAN_M,
) -> list[tuple[int, int]]:
    """Top-left corners of candidate patches, flattest first.

    Roughness is elevation standard deviation inside the patch. Ties are broken
    by position so the ranking is deterministic: the same asset, patch size and
    stride must always yield the same curriculum, or two runs at "difficulty
    0.5" stood on different ground and are not comparable.
    """
    key = (path, z_span_m, patch_x, patch_y, stride)
    cached = _RANK_CACHE.get(key)
    if cached is not None:
        return cached

    scored: list[tuple[float, int, int]] = []
    for x0 in range(0, desert_m.shape[0] - patch_x + 1, stride):
        for y0 in range(0, desert_m.shape[1] - patch_y + 1, stride):
            patch = desert_m[x0 : x0 + patch_x, y0 : y0 + patch_y]
            scored.append((float(patch.std()), x0, y0))
    if not scored:
        raise ValueError(
            f"no {patch_x}x{patch_y}-cell patch fits in a "
            f"{desert_m.shape[0]}x{desert_m.shape[1]}-cell desert at stride {stride}"
        )
    scored.sort()
    ranked = [(x0, y0) for _, x0, y0 in scored]
    _RANK_CACHE[key] = ranked
    return ranked


@height_field_to_mesh
def bestiary_desert_terrain(difficulty: float, cfg: "HfBestiaryDesertTerrainCfg") -> np.ndarray:
    """A patch of the committed Bestiary desert, as an Isaac Lab height grid.

    Args:
        difficulty: 0 selects the flattest real patch, 1 the roughest.
        cfg: see :class:`HfBestiaryDesertTerrainCfg`.

    Returns:
        ``(width_px, length_px)`` int16 grid in units of ``cfg.vertical_scale``,
        which is the contract `height_field_to_mesh` and
        `convert_height_field_to_mesh` expect.
    """
    desert_m = load_desert_m(cfg.hfield_path, cfg.z_span_m)
    resolved = Path(cfg.hfield_path).expanduser().resolve()

    # Native cells needed to cover the tile footprint. `cfg.size` has already
    # been shrunk by the decorator to exclude the border, so this is the region
    # actually being filled.
    patch_x = int(round(cfg.size[0] / DESERT_NATIVE_CELL_M))
    patch_y = int(round(cfg.size[1] / DESERT_NATIVE_CELL_M))
    if patch_x < 2 or patch_y < 2:
        raise ValueError(
            f"tile {cfg.size[0]:.3f}x{cfg.size[1]:.3f} m covers only "
            f"{patch_x}x{patch_y} native desert cells at {DESERT_NATIVE_CELL_M:.6f} m/cell; "
            "at least 2x2 are needed to interpolate"
        )
    if patch_x > desert_m.shape[0] or patch_y > desert_m.shape[1]:
        raise ValueError(
            f"tile {cfg.size[0]:.3f}x{cfg.size[1]:.3f} m needs {patch_x}x{patch_y} "
            f"native cells but the desert is only {desert_m.shape[0]}x{desert_m.shape[1]}. "
            "Reduce TerrainGeneratorCfg.size, or regenerate the desert larger."
        )

    ranked = _rank_patches_by_roughness(
        desert_m, patch_x, patch_y, cfg.patch_stride_cells, resolved, cfg.z_span_m
    )
    # difficulty in [0, 1] -> index into flattest..roughest.
    index = int(round(float(np.clip(difficulty, 0.0, 1.0)) * (len(ranked) - 1)))
    x0, y0 = ranked[index]
    patch = desert_m[x0 : x0 + patch_x, y0 : y0 + patch_y]
    patch = patch - patch.min()

    # Optional elevation gain, so a tile can be made gentler than the real
    # desert without changing its shape. (1.0, 1.0) means "exactly the asset".
    gain_lo, gain_hi = cfg.z_gain_range
    patch = patch * (gain_lo + (gain_hi - gain_lo) * float(np.clip(difficulty, 0.0, 1.0)))

    # Resample native cells onto the pixel grid Isaac Lab asked for. When
    # horizontal_scale == DESERT_NATIVE_CELL_M this is very nearly identity;
    # at the shipped 0.1 m it is a mild downsample that preserves gradients.
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    src_x = np.arange(patch_x, dtype=np.float64)
    src_y = np.arange(patch_y, dtype=np.float64)
    spline = interpolate.RectBivariateSpline(src_x, src_y, patch, kx=1, ky=1)
    dst_x = np.linspace(0.0, patch_x - 1.0, width_px)
    dst_y = np.linspace(0.0, patch_y - 1.0, length_px)
    z_m = spline(dst_x, dst_y)

    return np.rint(z_m / cfg.vertical_scale).astype(np.int16)


@configclass
class HfBestiaryDesertTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a tile cut from the committed Bestiary desert."""

    function = bestiary_desert_terrain

    hfield_path: str = MISSING
    """Absolute path to a committed heightfield ``.bin`` (the desert, the
    gentle terrain, or any future sibling in the same format).

    Deliberately not defaulted. This module is imported by a different
    interpreter, from a different working directory, than the one that owns
    `bestiary.paths`; a wrong default here would silently train on the wrong
    ground, which is exactly the failure the terrain invariant exists to stop.
    Callers resolve the path and pass it.
    """

    z_span_m: float = DESERT_Z_SPAN_M
    """Metres of elevation the asset's normalised [0, 1] samples span.

    The ``.bin`` format cannot carry this number, so the config does — the
    same bytes at a different span are a different world, and the cache keys
    on both. Defaults to the desert's 5.05 (read from the committed
    ``<hfield>`` element) so existing desert callers are unchanged; the gentle
    terrain passes ``terrain/gentle.py:Z_SPAN_M``, which its generator makes
    exact by construction."""

    z_gain_range: tuple[float, float] = (1.0, 1.0)
    """Elevation multiplier at difficulty 0 and 1. ``(1.0, 1.0)`` is the asset
    unmodified, which is the honest default; ``(0.3, 1.0)`` ramps a tile from
    30% of real relief up to the real thing as the curriculum advances."""

    patch_stride_cells: int = 32
    """Spacing, in native cells, of candidate patch corners. 32 cells is 2.5 m
    at ``DESERT_NATIVE_CELL_M``, giving ~900 candidates over the 1024^2 grid --
    enough for a smooth difficulty ramp, cheap enough to rank once."""


if __name__ == "__main__":
    # Self-check: prove the reader agrees with the asset, and print the
    # curriculum it produces. Numbers in the record come from code, not prose.
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hfield", help="path to desert_hfield.bin")
    ap.add_argument("--tile-m", type=float, default=8.0, help="square tile size in m")
    args = ap.parse_args()

    z = load_desert_m(args.hfield)
    cells = int(round(args.tile_m / DESERT_NATIVE_CELL_M))
    print(f"desert       : {z.shape[0]}x{z.shape[1]} cells, "
          f"{2 * DESERT_HALF_EXTENT_M:.0f}x{2 * DESERT_HALF_EXTENT_M:.0f} m, "
          f"{DESERT_NATIVE_CELL_M * 100:.4f} cm/cell")
    # np.ptp(a), not a.ptp() -- the ndarray method was removed in NumPy 2.0.
    print(f"elevation    : {z.min():.3f} .. {z.max():.3f} m  (span {np.ptp(z):.3f} m)")
    print(f"tile         : {args.tile_m:.2f} m -> {cells}x{cells} native cells")
    ranked = _rank_patches_by_roughness(z, cells, cells, 32, Path(args.hfield).resolve())
    print(f"candidates   : {len(ranked)}")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        x0, y0 = ranked[int(round(d * (len(ranked) - 1)))]
        p = z[x0 : x0 + cells, y0 : y0 + cells]
        print(f"  difficulty {d:4.2f} -> corner ({x0:4d},{y0:4d})  "
              f"relief {np.ptp(p):5.3f} m  std {p.std():5.3f} m")
