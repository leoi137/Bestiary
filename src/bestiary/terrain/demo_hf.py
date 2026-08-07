"""The DEMO terrain: one continuous strip, flat at -x, hardest at +x.

    ~/isaaclab-env/bin/python -m bestiary.terrain.demo_hf --stats \\
        assets/terrain/gentle_hfield.bin

WHAT THIS IS FOR, AND WHY IT IS NOT THE TRAINING TERRAIN
---------------------------------------------------------
`isaac_hf.py` serves the committed assets as Isaac Lab SUB-TERRAINS: a grid of
8 m tiles, each cut from a different patch of the asset and ranked by
roughness, because that grid IS the difficulty curriculum — one tile per rung,
promoted and demoted per episode. It is the right shape for training and the
wrong shape for a camera. Adjacent tiles are cut from unrelated patches, so
every tile boundary is a vertical step, and the generator wraps the whole grid
in a flat border. On film that reads as a chessboard the machine periodically
falls off, and the steps are out-of-distribution cliffs no curriculum taught.

This module answers the other question: ONE surface, no tiles, no borders, no
seams, with difficulty as a smooth function of position instead of a property
of which square you are standing on. Nothing here trains anything. It exists so
a trained policy can be *watched* crossing ground that gets harder as it goes.

HOW IT IS BUILT, AND WHY IT CANNOT HAVE A SEAM
-----------------------------------------------
The surface is the committed asset's own morphology under an envelope:

    demo(x, y) = gain(x) * (asset(x, y) - mean(asset))

`asset` is read from committed bytes by `isaac_hf.load_desert_m` — the same
reader, the same validation, the same cache — so this is the real terrain's
dunes and gravel, not a second synthesised landscape (`gentle.py` records what
happens when this project composes its own: three dead iterations). `gain` is
scalar in x alone. A product of two continuous functions is continuous, so
there is no construction here that CAN produce a step, and difficulty is
monotone in x by the same argument.

`gain` is exactly 0 over the first `pad_frac` of the strip, which makes the
left end a mathematically flat plane rather than merely gentle ground — a
spawn pad, and the "flat" the demo starts on. It then rises by smoothstep to
`max_gain` at +x. Smoothstep, not linear: a linear ramp has a slope
discontinuity where it leaves the pad, and a machine crossing that boundary
gets a first derivative kick in its height scan for no reason anyone wanted.

SET `slope_threshold = None` ON THE GENERATOR THAT USES THIS. That knob makes
`convert_height_field_to_mesh` insert vertical correction faces at steep cells,
which is a deliberate cliff-maker — the opposite of this file's entire purpose.

WHAT "HARDEST" MEANS HERE
--------------------------
`max_gain` multiplies the asset's relief, and slope scales with it: at 1.0 the
right end is the committed asset unmodified. Above 1.0 the terrain becomes
harder than anything the policy trained on, which is a legitimate thing to film
and a dishonest thing to film silently. `--stats` prints per-zone body-slope
and step-height so the claim "it gets harder left to right" is measured rather
than asserted, using `gentle.py`'s metrics so the numbers are comparable to the
training asset's published table.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
from scipy import interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass

from bestiary.terrain.isaac_hf import DESERT_NATIVE_CELL_M, load_desert_m


def ramp_gain(t: np.ndarray, pad_frac: float, max_gain: float) -> np.ndarray:
    """Elevation multiplier along the strip. ``t`` is position in [0, 1].

    Exactly 0 on the pad, then smoothstep (3u^2 - 2u^3) to ``max_gain``. The
    smoothstep is C1 at both ends, so the pad-to-terrain transition has no
    slope discontinuity — see the module docstring.
    """
    if not 0.0 <= pad_frac < 1.0:
        raise ValueError(f"pad_frac must be in [0, 1), got {pad_frac}")
    u = np.clip((t - pad_frac) / (1.0 - pad_frac), 0.0, 1.0)
    return max_gain * (3.0 * u**2 - 2.0 * u**3)


@height_field_to_mesh
def demo_ramp_terrain(difficulty: float, cfg: "HfDemoRampTerrainCfg") -> np.ndarray:
    """One continuous strip: flat at -x, ``cfg.max_gain`` x the asset at +x.

    Args:
        difficulty: IGNORED, deliberately. Difficulty here is a function of
            position within the single tile, not of which tile you are on;
            accepting the argument keeps the Isaac Lab sub-terrain contract
            while making it explicit that no curriculum indexes this.
        cfg: see :class:`HfDemoRampTerrainCfg`.

    Returns:
        ``(width_px, length_px)`` int16 grid in units of ``cfg.vertical_scale``.
    """
    del difficulty  # see the docstring — position is the difficulty axis here

    asset_m = load_desert_m(cfg.hfield_path, cfg.z_span_m)

    # Native cells the strip needs. `cfg.size` is already border-shrunk by the
    # decorator, so this is the region actually being filled.
    need_x = int(round(cfg.size[0] / DESERT_NATIVE_CELL_M))
    need_y = int(round(cfg.size[1] / DESERT_NATIVE_CELL_M))
    if need_x > asset_m.shape[0] or need_y > asset_m.shape[1]:
        raise ValueError(
            f"demo strip {cfg.size[0]:.1f}x{cfg.size[1]:.1f} m needs "
            f"{need_x}x{need_y} native cells but the asset is only "
            f"{asset_m.shape[0]}x{asset_m.shape[1]} "
            f"({asset_m.shape[0] * DESERT_NATIVE_CELL_M:.1f}x"
            f"{asset_m.shape[1] * DESERT_NATIVE_CELL_M:.1f} m). Shorten the strip "
            "or generate a larger asset — do NOT tile it, which is the seam this "
            "module exists to remove."
        )

    # Centre the crop, so the strip is the middle of the asset rather than a
    # corner. Corners of a generated field are where its edge treatment lives.
    x0 = (asset_m.shape[0] - need_x) // 2
    y0 = (asset_m.shape[1] - need_y) // 2
    strip = asset_m[x0 : x0 + need_x, y0 : y0 + need_y]

    # About the mean, so gain 0 is a plane at z = 0 and gain 1 is the asset's
    # own shape. Subtracting the min instead would make the pad sit in a pit.
    strip = strip - float(strip.mean())

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    src_x = np.arange(need_x, dtype=np.float64)
    src_y = np.arange(need_y, dtype=np.float64)
    spline = interpolate.RectBivariateSpline(src_x, src_y, strip, kx=1, ky=1)
    z_m = spline(
        np.linspace(0.0, need_x - 1.0, width_px),
        np.linspace(0.0, need_y - 1.0, length_px),
    )

    gain = ramp_gain(
        np.linspace(0.0, 1.0, width_px), cfg.pad_frac, cfg.max_gain
    )[:, None]
    z_m = z_m * gain

    # Float the whole surface so its minimum is 0. The mesh is placed on Isaac
    # Lab's own origin and a negative minimum would bury the pad.
    z_m = z_m - float(z_m.min())
    return np.rint(z_m / cfg.vertical_scale).astype(np.int16)


@configclass
class HfDemoRampTerrainCfg(HfTerrainBaseCfg):
    """A single continuous demo strip cut from a committed heightfield."""

    function = demo_ramp_terrain

    hfield_path: str = MISSING
    """Absolute path to a committed heightfield ``.bin``. Deliberately not
    defaulted, for the reason `isaac_hf.HfBestiaryDesertTerrainCfg` gives: this
    module is imported by a different interpreter from a different working
    directory than the one that owns `bestiary.paths`."""

    z_span_m: float = MISSING
    """Metres the asset's normalised [0, 1] samples span. Not defaulted here —
    unlike the desert-shaped config this has no majority caller, and the same
    bytes at the wrong span are silently the wrong world."""

    pad_frac: float = 0.12
    """Fraction of the strip that is exactly flat. 0.12 of a 78 m strip is
    9.4 m — long enough that a machine bounding at 5 m/s gets ~2 s of level
    ground to establish a gait before the terrain starts."""

    max_gain: float = 1.0
    """Elevation multiplier at the far (+x) end. 1.0 is the committed asset
    unmodified, which is the honest default: the right end is exactly the
    ground the policy trained on, at its hardest. Above 1.0 is terrain harder
    than anything it has seen — legitimate to film, dishonest to film without
    saying so."""


if __name__ == "__main__":
    # The ramp is monotone BY MEASUREMENT, not by assertion. Same metrics as
    # `terrain/gentle.py` so the numbers compare to the training asset's table.
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hfield", help="path to a committed heightfield .bin")
    ap.add_argument("--z-span-m", type=float, required=True)
    ap.add_argument("--length-m", type=float, default=78.0)
    ap.add_argument("--width-m", type=float, default=50.0)
    ap.add_argument("--pad-frac", type=float, default=0.12)
    ap.add_argument("--max-gain", type=float, default=1.0)
    ap.add_argument("--zones", type=int, default=6)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    asset = load_desert_m(a.hfield, a.z_span_m)
    nx = int(round(a.length_m / DESERT_NATIVE_CELL_M))
    ny = int(round(a.width_m / DESERT_NATIVE_CELL_M))
    sx = (asset.shape[0] - nx) // 2
    sy = (asset.shape[1] - ny) // 2
    h = asset[sx : sx + nx, sy : sy + ny]
    h = h - float(h.mean())
    h = h * ramp_gain(np.linspace(0.0, 1.0, nx), a.pad_frac, a.max_gain)[:, None]
    h = h - float(h.min())

    cell = DESERT_NATIVE_CELL_M
    # Body-slope on the 0.5 m-smoothed surface, per gentle.py's argument that
    # cell-scale slope on this recipe is gravel the simulator renders as
    # pebble-steps, not wall.
    kx = np.fft.fftfreq(h.shape[0], d=cell)
    ky = np.fft.fftfreq(h.shape[1], d=cell)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    smooth = np.fft.ifft2(
        np.fft.fft2(h) * np.exp(-2.0 * np.pi**2 * 0.5**2 * (KX**2 + KY**2))
    ).real
    gx, gy = np.gradient(smooth, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    step = np.maximum(
        np.abs(np.diff(h, axis=0))[:, :-1], np.abs(np.diff(h, axis=1))[:-1, :]
    )

    print(f"strip {a.length_m:.0f} x {a.width_m:.0f} m from {a.hfield} "
          f"(z_span {a.z_span_m} m), pad {a.pad_frac:.2f}, max_gain {a.max_gain}")
    print(f"span {np.ptp(h):.3f} m over one continuous surface, {nx}x{ny} native cells")
    print(f"{'zone (x, m)':>18} {'gain':>6} {'slope P50':>10} {'P99':>7} {'step P99 cm':>12}")
    edges = np.linspace(0, nx, a.zones + 1).astype(int)
    for i in range(a.zones):
        lo, hi = edges[i], edges[i + 1]
        t_mid = (lo + hi) / 2.0 / nx
        g = float(ramp_gain(np.array([t_mid]), a.pad_frac, a.max_gain)[0])
        s = slope[lo:hi]
        st = step[lo : max(hi - 1, lo + 1)]
        print(f"{lo * cell:8.1f}-{hi * cell:<8.1f} {g:6.3f} "
              f"{np.percentile(s, 50):10.2f} {np.percentile(s, 99):7.2f} "
              f"{np.percentile(st, 99) * 100:12.2f}")
