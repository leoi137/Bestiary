"""Reading the ground height out of a compiled MuJoCo heightfield.

Both terrain envs need the same thing: given (x, y), how high is the ground?
World-absolute z is meaningless as a health check on a hill — a robot standing
on a 2 m dune is not "too high", it is standing — so the healthy-z band has to
be measured against the surface underneath, and resets have to spawn on it.

This was written inline inside envs/spyder.py first. It lives here now
because HoundEnv needs the identical calculation, and two copies of a
bilinear interpolation are two chances to fix a bug once.
"""
from __future__ import annotations

import mujoco
import numpy as np


class HeightField:
    """Bilinear lookup over the first heightfield in a compiled model.

    Construct with `HeightField.from_model(model)`, which returns None when
    the model has no heightfield (the flat worlds) — callers then treat the
    ground as z = 0 and every terrain-aware rule collapses back to the
    flat-world rule it generalizes.
    """

    def __init__(self, data: np.ndarray, size: np.ndarray, pos: np.ndarray):
        self.data = data          # (nrow, ncol), MuJoCo-normalized to [0, 1]
        self.size = size          # (x half-extent, y half-extent, z span, base)
        self.pos = pos            # world position of the hfield geom

    @classmethod
    def from_model(cls, model) -> "HeightField | None":
        if model.nhfield == 0:
            return None
        geom_ids = np.nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD)[0]
        if len(geom_ids) == 0:
            return None
        hid = int(model.geom_dataid[geom_ids[0]])
        nrow = int(model.hfield_nrow[hid])
        ncol = int(model.hfield_ncol[hid])
        return cls(
            data=model.hfield_data.reshape(nrow, ncol),
            size=model.hfield_size[hid].copy(),
            pos=model.geom_pos[geom_ids[0]].copy(),
        )

    def height_at(self, x: float, y: float) -> float:
        """World-z of the terrain surface under (x, y).

        Bilinear interpolation over the grid — the same surface MuJoCo
        collides against, whose hfield collider triangulates these cells.
        Coordinates beyond the field clamp to the edge cell. MuJoCo stores
        hfield data normalized to [0, 1], so world elevation is
        geom_z + value * z_span.
        """
        data = self.data
        rx, ry, zscale = self.size[:3]
        nrow, ncol = data.shape
        # Grid coords: col 0..ncol-1 spans x in [-rx, rx], rows span y.
        cx = (x - self.pos[0] + rx) / (2 * rx) * (ncol - 1)
        cy = (y - self.pos[1] + ry) / (2 * ry) * (nrow - 1)
        cx = min(max(cx, 0.0), ncol - 1.0)
        cy = min(max(cy, 0.0), nrow - 1.0)
        c0, r0 = int(min(cx, ncol - 2)), int(min(cy, nrow - 2))
        fx, fy = cx - c0, cy - r0
        v = (data[r0, c0] * (1 - fx) + data[r0, c0 + 1] * fx) * (1 - fy) + (
            data[r0 + 1, c0] * (1 - fx) + data[r0 + 1, c0 + 1] * fx
        ) * fy
        return float(self.pos[2] + v * zscale)


def ground_height_at(hfield: "HeightField | None", x: float, y: float) -> float:
    """height_at, or 0.0 on a model with no heightfield."""
    return 0.0 if hfield is None else hfield.height_at(x, y)
