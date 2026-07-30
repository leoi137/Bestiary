"""Read a binary STL and measure it. No dependencies beyond numpy.

    python -m bestiary.robots.whelp.stl assets/whelp/stl/thigh.stl

WHY NOT trimesh / numpy-stl
---------------------------
Because this file has exactly one job -- volume, bounding box, and "is the mesh
closed" -- and adding a dependency to `requirements.txt` so that a *mass
estimate* can run is a bad trade. The mesh library would also have to be
installed on any machine that regenerates the robot, including one that only
wants the URDF. Sixty lines of numpy is cheaper than that, forever.

WHAT IT IS FOR
--------------
export.py computes each part's mass analytically, from a hand-written sum of
primitive volumes in massmodel.py. That model is a *second implementation* of
the geometry in scad/parts/, and two implementations of the same thing drift.
So when OpenSCAD is installed, export.py measures the real STL with this module
and asserts the analytic model agrees within a tolerance. The analytic model is
what makes the URDF buildable without a CAD binary; this measurement is what
stops it from being quietly wrong.

If the two disagree the failure is loud, names both numbers, and names the part.
That is the entire point: a mass model that silently drifts produces a URDF whose
inertias are wrong, which produces a policy tuned for a robot that does not
exist, which produces the thing this project is trying to avoid.

THE WATERTIGHT CHECK
--------------------
Volume of a non-closed mesh is meaningless -- the divergence-theorem sum still
returns a number, it just is not a volume. OpenSCAD emits non-manifold output
when a CSG operation goes wrong (coincident faces, zero-thickness walls from a
difference that exactly touches). Those are common authoring mistakes and they
are invisible in the preview. Counting half-edges catches them: in a closed
orientable surface every edge is shared by exactly two triangles.
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Vertices are quantised to this grid (mm) before edge-pairing, because
#: OpenSCAD emits float32 and two triangles meeting at "the same" vertex can
#: differ in the last bit. 1e-4 mm is far below any printable feature and far
#: above float32 noise at the ~100 mm scale these parts live at.
_WELD_MM = 1e-4


@dataclass(frozen=True)
class Mesh:
    """A triangle soup, measured."""

    path: Path
    triangles: int
    #: mm^3. Signed: a negative volume means the mesh is inside-out, which is a
    #: real and reportable defect rather than something to take abs() of.
    volume_mm3: float
    #: mm^2, total surface area. Used to sanity-check shell-vs-infill mass.
    area_mm2: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    #: Edges belonging to exactly one triangle. Zero for a closed solid.
    open_edges: int
    #: Edges shared by three or more triangles. Zero for a manifold solid.
    nonmanifold_edges: int
    degenerate_triangles: int

    @property
    def size_mm(self) -> tuple[float, float, float]:
        return tuple(hi - lo for hi, lo in zip(self.bbox_max, self.bbox_min))

    @property
    def watertight(self) -> bool:
        return self.open_edges == 0 and self.nonmanifold_edges == 0

    @property
    def volume_cm3(self) -> float:
        return self.volume_mm3 / 1000.0

    def fits(self, bed_x: float, bed_y: float, bed_z: float) -> bool:
        """Does this part fit the build volume in *some* axis-aligned rotation?

        Only the six axis-aligned orientations are considered. A part that only
        fits diagonally is reported as not fitting on purpose: printing across
        the bed diagonal is real but fragile advice (gantry clearance, purge
        line, skirt), and a part that needs it should be split instead.
        """
        w, d, h = sorted(self.size_mm)
        bx, by, bz = sorted((bed_x, bed_y, bed_z))
        return w <= bx and d <= by and h <= bz

    def describe(self) -> str:
        sx, sy, sz = self.size_mm
        flag = "" if self.watertight else "  ** NOT WATERTIGHT **"
        return (
            f"{self.path.name:<28} {self.volume_cm3:8.2f} cm^3  "
            f"{sx:6.1f} x {sy:6.1f} x {sz:6.1f} mm  {self.triangles:>7} tri{flag}"
        )


def read(path: str | Path) -> Mesh:
    """Parse a binary or ASCII STL and measure it.

    Raises rather than guessing on a truncated file: a partially-read mesh
    produces a plausible-looking volume that is simply wrong, and that is the
    worst possible failure for something feeding a mass budget.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path}: {len(raw)} bytes is too short to be an STL")

    if raw[:5] == b"solid" and b"facet normal" in raw[:2048]:
        tris = _parse_ascii(raw, path)
    else:
        tris = _parse_binary(raw, path)

    if tris.shape[0] == 0:
        raise ValueError(f"{path}: contains no triangles")

    return _measure(path, tris)


def _parse_binary(raw: bytes, path: Path) -> np.ndarray:
    (count,) = struct.unpack("<I", raw[80:84])
    expected = 84 + count * 50
    if len(raw) != expected:
        raise ValueError(
            f"{path}: header declares {count} triangles, which needs {expected} bytes, "
            f"but the file is {len(raw)}. Truncated or not a binary STL."
        )
    # Each 50-byte record is: normal[3], v0[3], v1[3], v2[3] as float32, then a
    # uint16 attribute. Read as bytes and slice, rather than 50-byte struct
    # unpacking in a loop, which is ~100x slower on a 200k-triangle part.
    body = np.frombuffer(raw, dtype=np.uint8, count=count * 50, offset=84).reshape(count, 50)
    verts = body[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return verts.astype(np.float64)


def _parse_ascii(raw: bytes, path: Path) -> np.ndarray:
    coords: list[float] = []
    for line in raw.decode("ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            coords.extend(float(p) for p in parts[1:])
    if len(coords) % 9:
        raise ValueError(f"{path}: {len(coords) // 3} vertices is not a whole number of triangles")
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3, 3)


def _measure(path: Path, tris: np.ndarray) -> Mesh:
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

    # Divergence theorem: the volume enclosed by a closed oriented surface is
    # the sum of the signed volumes of tetrahedra from the origin to each face.
    # Valid for any origin, including one outside the mesh -- the outside
    # contributions cancel exactly.
    cross = np.cross(v1 - v0, v2 - v0)
    volume = float(np.einsum("ij,ij->i", v0, cross).sum() / 6.0)
    area = float(np.linalg.norm(cross, axis=1).sum() / 2.0)
    degenerate = int((np.linalg.norm(cross, axis=1) < 1e-12).sum())

    flat = tris.reshape(-1, 3)
    keys = np.round(flat / _WELD_MM).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)

    # An undirected edge as a sorted (lo, hi) pair. In a closed orientable
    # surface each appears exactly twice -- once from each adjoining triangle.
    e = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)

    return Mesh(
        path=path,
        triangles=int(tris.shape[0]),
        volume_mm3=volume,
        area_mm2=area,
        bbox_min=tuple(float(x) for x in flat.min(axis=0)),
        bbox_max=tuple(float(x) for x in flat.max(axis=0)),
        open_edges=int((counts == 1).sum()),
        nonmanifold_edges=int((counts > 2).sum()),
        degenerate_triangles=degenerate,
    )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[1].strip())
        return 2
    bad = 0
    for arg in argv:
        for p in sorted(Path().glob(arg)) or [Path(arg)]:
            try:
                m = read(p)
            except (OSError, ValueError) as exc:
                print(f"{p}: {exc}")
                bad += 1
                continue
            print(m.describe())
            if not m.watertight:
                print(f"    open edges {m.open_edges}, non-manifold {m.nonmanifold_edges}, "
                      f"degenerate triangles {m.degenerate_triangles}")
                bad += 1
            if m.volume_mm3 < 0:
                print("    volume is NEGATIVE: the mesh is inside-out")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
