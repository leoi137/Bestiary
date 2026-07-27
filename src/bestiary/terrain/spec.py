"""One identity for the ground a run trained on, and a hash of it.

`envs/obs_spec.py` and `envs/reward_spec.py` are the pattern. This is the same
treatment applied to the third input to a run's dynamics, and the last one that
had none. `research/anomalies.jsonl` (2026-07-27) states the gap exactly:

    The terrain is the third input to a run's dynamics and the only one with no
    hash, no config.json field, and no guard. obs_spec is
    declared/hashed/pinned/asserted-on-resume; reward_spec is too, as of
    fbbd9af. The heightfield is neither.

WHY THIS IS NOT A THEORETICAL GAP

`research/scripts/compare_terrain_grids.py` measured what the window plan's
proposed regen at GRID=2048 would actually do. `generate.py` draws its spectral
phases as an `(n, n)` array indexed by FFT bin, so changing `n` hands every
drawn phase to a different wavelength — the RNG stream is bit-identical and the
world still changes. The regenerated terrain correlates with the committed one
at **+0.0610**: same statistics, different desert.

Under the code as it stood, that swap would have kept every checkpoint loading
(the observation width is unchanged), kept every guard green (none of them read
the heightfield), and silently made every ledger row incomparable with every
later one. It is the exact failure obs_spec and reward_spec each exist to
close, on the leg that was still open.

WHY THIS MODULE LIVES IN `terrain/` AND NOT `envs/`

`obs_spec` and `reward_spec` sit in `envs/` because they are *declarations*: a
human writes the term list, the env holds it as `_obs_spec` / `_reward_spec`,
and the spec's job is to make the env live up to what was declared.

A terrain spec has nothing to declare. The ground is an asset — written by
`terrain/generate.py`, compiled into the model by MuJoCo, read back by
`terrain/field.py` — and its identity is *measured*, not authored. So it
belongs beside the code that writes and reads that asset, where the writer, the
reader, and the fingerprint are one package and cannot drift apart.

The measurement has a second benefit the declaration pattern cannot have: it
needs no cooperation from the env class. `Ant-v5`, `Walker2d-v5` and
`Humanoid-v5` are Gymnasium's, not ours, and they are covered by this anyway —
because the question is asked of the compiled `mjModel`, not of the env author.

WHAT IS HASHED, AND WHY THAT AND NOT THE FILE

The obvious digest is `sha256(assets/terrain/desert_hfield.bin)`. This hashes
the **compiled** heightfield instead, because the file is only one of the two
things that decide what the robot walks on:

    assets/terrain/desert_hfield.bin   the height samples, normalized to [0,1]
    <hfield ... size="40 40 5.05 1.0"/> and the floor geom's pos/quat, which
                                       turn those unitless samples into metres

Editing `size="40 40 5.05 1.0"` to `... 10.1 ...` doubles every slope in the
world without touching a byte of the .bin. A file digest is blind to it; this
is not. Hashing what MuJoCo actually collides against is also total over any
future terrain that does not come from a file at all.

The hashed fields are therefore the full state of `field.py`'s ground lookup:

    nrow, ncol            the grid — GRID=1024 today, and the number the regen
                          would have moved to 2048
    x/y_half_extent_m     the 80 x 80 m footprint
    z_span_m              elevation span; the metres one unit of sample means
    z_base_m              hfield_size[3], the solid skirt below the surface
    pos_m                 where the field sits; pos_m[2] IS the elevation min,
                          because a sample of 0.0 maps to exactly that height
    quat                  the floor geom's orientation
    data_sha256           every height sample, in order

Deliberately NOT hashed, and recorded anyway:

    source                the asset path the model carries. Provenance for a
                          human. Moving a file must not read as moving the
                          ground, for the same reason `ObsTerm.note` is not
                          hashed — a digest that moves on cosmetics is a digest
                          people learn to ignore.
    cell_cm               derived from ncol and the extent; recorded because a
                          reader comparing terrain to wheel radius should not
                          have to do the division, hashed via its inputs.

The file's byte count is not recorded: it is `8 + 4 * nrow * ncol` by
construction (`generate.save_hfield_bin` writes two int32 then float32 data),
so it carries no information the grid does not already carry, and a recorded
number that cannot disagree with its neighbours is decoration.

WHY TWO HASHES

`reward_spec` has `hash` and `shape_hash` because a retune and a re-shape are
different failures with different remedies. The same split is real here:

* **`field_hash`** — the samples and the grid. "*Which desert is this?*" It
  moves when the heightfield is regenerated at another seed or another GRID,
  and it is the number that answers whether two runs walked the same ground.
* **`hash`** — the samples plus their metric placement. "*Which ground is
  this?*" It also moves when the XML rescales or repositions the same desert.

Both break comparability. They are separated because the remedies point at
different files: `field_hash` moving means `assets/terrain/desert_hfield.bin`
changed, and only a regen or a `git checkout` restores it; `hash` moving alone
means someone edited a number in a `*_desert.xml`, which is a one-line revert.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from bestiary.terrain.field import HeightField

# Bump only when the hash INPUT changes (what goes into the digest), not when
# the terrain changes. Same reasoning as SPEC_HASH_VERSION in obs_spec.py and
# REWARD_HASH_VERSION in reward_spec.py: without it, a future change to the
# digest recipe silently invalidates every recorded hash and reads as universal
# drift — every run at once, for no physical reason.
TERRAIN_HASH_VERSION = 1

# Lengths are rounded before hashing so float formatting cannot invent a
# terrain change. 6 decimal places is 1 micrometre on an 80 m world: far finer
# than any number a human writes into an MJCF `size=` attribute (the finest
# today is z_span = 5.05 m), and coarse enough that a value written to
# config.json and read back round-trips to the same digest.
_LENGTH_PRECISION = 6

# Heights are hashed as little-endian float32 explicitly rather than as
# whatever `ndarray.tobytes()` gives on this machine. MuJoCo stores hfield data
# as float32 and x86 is little-endian, so this is the identity here — it costs
# nothing and stops a digest recorded on one machine from disagreeing with the
# same terrain read on another.
_SAMPLE_DTYPE = "<f4"


def _round(value: float) -> float:
    return round(float(value), _LENGTH_PRECISION)


@dataclass(frozen=True, slots=True)
class TerrainSpec:
    """The identity of one compiled heightfield.

    Construct with `TerrainSpec.from_model(model)`, which returns **None** for
    a world with no heightfield — the flat plane every non-desert env stands
    on. None is a real answer here, not a failure: see `from_model`.

    `source` is provenance for a reader and is deliberately excluded from both
    digests.
    """

    nrow: int
    ncol: int
    x_half_extent_m: float
    y_half_extent_m: float
    z_span_m: float
    z_base_m: float
    pos_m: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    data_sha256: str
    source: str = ""

    def __post_init__(self) -> None:
        # Loud and early, with the actual value. Every one of these is a state
        # MuJoCo cannot produce from a valid model, so reaching one means the
        # spec was rebuilt from a hand-edited record — which is exactly when a
        # hash comparison would otherwise "pass" against fiction.
        if self.nrow < 2 or self.ncol < 2:
            raise ValueError(
                f"terrain spec has a degenerate grid {self.nrow}x{self.ncol}; "
                f"a heightfield needs at least 2 samples per side to span a cell"
            )
        if self.z_span_m <= 0.0:
            raise ValueError(
                f"terrain spec has z_span_m={self.z_span_m}; the elevation span "
                f"scales the [0,1] samples into metres and cannot be <= 0"
            )
        if self.x_half_extent_m <= 0.0 or self.y_half_extent_m <= 0.0:
            raise ValueError(
                f"terrain spec has a non-positive footprint "
                f"{self.x_half_extent_m} x {self.y_half_extent_m} m"
            )
        if len(self.data_sha256) != 64 or not all(
            c in "0123456789abcdef" for c in self.data_sha256
        ):
            raise ValueError(
                f"terrain spec data_sha256 is not a 64-char lowercase sha256 "
                f"digest: {self.data_sha256!r}"
            )
        if len(self.pos_m) != 3:
            raise ValueError(f"terrain spec pos_m must be 3 values, got {self.pos_m!r}")
        if len(self.quat) != 4:
            raise ValueError(f"terrain spec quat must be 4 values, got {self.quat!r}")

    # --- derived, for a reader ---------------------------------------------

    @property
    def cell_cm(self) -> float:
        """Horizontal spacing between adjacent samples, in centimetres.

        `2 * half_extent / (ncol - 1)`, matching `field.py`'s `height_at`,
        which maps x in [-rx, rx] onto columns 0..ncol-1. Note this is NOT the
        `CELL = 2 * HALF_EXTENT / GRID` that `generate.py` synthesizes against
        — 1024 samples span 1023 gaps, so the true spacing is 7.8201 cm where
        generate.py assumes 7.8125 cm. A 0.1% horizontal scale offset in the
        synthesized wavelengths; recorded here as the number that describes the
        compiled world, which is the one the robot walks on.
        """
        return 2.0 * self.x_half_extent_m / (self.ncol - 1) * 100.0

    @property
    def z_min_m(self) -> float:
        """World elevation of a sample of 0.0 — the lowest point of the field."""
        return self.pos_m[2]

    @property
    def z_max_m(self) -> float:
        """World elevation of a sample of 1.0 — the highest point of the field."""
        return self.pos_m[2] + self.z_span_m

    # --- identity -----------------------------------------------------------

    @property
    def field_hash(self) -> str:
        """Digest over the height samples and the grid. "Which desert?"

        Invariant to the metric placement, so it stays put when a `*_desert.xml`
        rescales the same field and moves the moment the field itself is
        regenerated. Truncated to 16 hex chars for the same reason as
        `ObsSpec.hash`: this identifies a terrain among a handful, it is not a
        security primitive, and a short digest is one a human can compare
        between a config file and a guard line at a glance. The full
        `data_sha256` is recorded beside it for anyone who wants it.
        """
        payload = (
            f"v{TERRAIN_HASH_VERSION}|{self.nrow}x{self.ncol}|{self.data_sha256}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def hash(self) -> str:
        """Digest over the samples AND their metric placement. "Which ground?"

        The full identity, and the one a resume is asserted against.
        """
        payload = "|".join(
            [
                f"v{TERRAIN_HASH_VERSION}",
                f"{self.nrow}x{self.ncol}",
                self.data_sha256,
                f"ext:{_round(self.x_half_extent_m)!r},{_round(self.y_half_extent_m)!r}",
                f"z:{_round(self.z_span_m)!r},{_round(self.z_base_m)!r}",
                "pos:" + ",".join(repr(_round(v)) for v in self.pos_m),
                "quat:" + ",".join(repr(_round(v)) for v in self.quat),
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # --- serialization ------------------------------------------------------

    def to_record(self) -> dict:
        """What `config.json` stores, so a run says what ground it trained on."""
        return {
            "hash": self.hash,
            "field_hash": self.field_hash,
            "hash_version": TERRAIN_HASH_VERSION,
            "source": self.source,
            "nrow": self.nrow,
            "ncol": self.ncol,
            "cell_cm": round(self.cell_cm, 4),
            "x_half_extent_m": _round(self.x_half_extent_m),
            "y_half_extent_m": _round(self.y_half_extent_m),
            "z_span_m": _round(self.z_span_m),
            "z_base_m": _round(self.z_base_m),
            "pos_m": [_round(v) for v in self.pos_m],
            "quat": [_round(v) for v in self.quat],
            "data_sha256": self.data_sha256,
        }

    @classmethod
    def from_record(cls, record: dict) -> "TerrainSpec":
        """Rebuild a spec from what `config.json` stored.

        This exists so a guard can re-derive the digests from the fields
        recorded beside them and catch a hand-edited config — the same thing
        `guards/reward_spec.py` does by rebuilding a `RewardSpec` from its
        recorded terms. Without it, a recorded hash is only ever compared to
        itself and proves nothing about the record it sits in.

        Missing keys raise `KeyError` rather than defaulting: a record that has
        lost a field is not a terrain with a zero in it, and quietly filling one
        in would manufacture a digest that matches nothing real.
        """
        try:
            return cls(
                nrow=int(record["nrow"]),
                ncol=int(record["ncol"]),
                x_half_extent_m=float(record["x_half_extent_m"]),
                y_half_extent_m=float(record["y_half_extent_m"]),
                z_span_m=float(record["z_span_m"]),
                z_base_m=float(record["z_base_m"]),
                pos_m=tuple(float(v) for v in record["pos_m"]),
                quat=tuple(float(v) for v in record["quat"]),
                data_sha256=str(record["data_sha256"]),
                source=str(record.get("source", "")),
            )
        except KeyError as exc:
            raise KeyError(
                f"terrain record is missing {exc} — the recorded fields are "
                f"{sorted(record)}. A partial record cannot be re-hashed, so it "
                f"cannot be checked against the ground a run is standing on."
            ) from exc

    # --- measurement --------------------------------------------------------

    @classmethod
    def from_model(cls, model) -> "TerrainSpec | None":
        """Measure the ground of a compiled MuJoCo model, or None if it is flat.

        **None is how a flat world is handled, and it is derived, not listed.**
        The test is `HeightField.from_model(model) is None`, i.e. the model
        declares no heightfield or no geom uses one — the identical test
        `envs/hound.py` and `envs/spyder.py` already use to decide whether the
        ground is at z = 0. Reusing it means there is exactly one definition of
        "does this world have terrain?" in the repository, and `Spyder-v0`,
        `Hound-v0`, `HoundPD-v0`, `Ant-v5`, `Walker2d-v5` and `Humanoid-v5` all
        answer it the same way without appearing in any list here. A hardcoded
        list of terrain envs would be wrong the first time someone adds one.

        `hfield_data` is MuJoCo's normalization of the file to [0, 1].
        `generate.save_hfield_bin` already writes an exact [0, 1] span for
        precisely this reason, so that normalization is the identity and the
        digest is the file's own samples rather than an arithmetic function of
        them.
        """
        hfield = HeightField.from_model(model)
        if hfield is None:
            return None

        samples = np.ascontiguousarray(hfield.data, dtype=np.float32)
        digest = hashlib.sha256(samples.astype(_SAMPLE_DTYPE).tobytes()).hexdigest()

        nrow, ncol = samples.shape
        rx, ry, z_span, z_base = (float(v) for v in hfield.size[:4])

        return cls(
            nrow=int(nrow),
            ncol=int(ncol),
            x_half_extent_m=rx,
            y_half_extent_m=ry,
            z_span_m=z_span,
            z_base_m=z_base,
            pos_m=tuple(float(v) for v in hfield.pos[:3]),
            quat=tuple(float(v) for v in model.geom_quat[hfield.geom_id][:4]),
            data_sha256=digest,
            source=_hfield_source(model, hfield.hid),
        )

    def describe(self) -> str:
        """A few lines for a launch log that is worth reading a month later."""
        return (
            f"    {self.nrow}x{self.ncol} samples over "
            f"{2 * self.x_half_extent_m:.0f} x {2 * self.y_half_extent_m:.0f} m "
            f"({self.cell_cm:.2f} cm/cell)\n"
            f"    elevation {self.z_min_m:+.2f} .. {self.z_max_m:+.2f} m "
            f"(span {self.z_span_m:.2f} m)\n"
            f"    source {self.source or '(not recorded by the model)'}"
        )


def _hfield_source(model, hid: int) -> str:
    """The asset path the compiled model carries for heightfield `hid`.

    Provenance only — never hashed. `mjModel.paths` is a single null-separated
    byte blob and `hfield_pathadr` indexes into it; a procedurally-built
    heightfield has no entry and yields "". Both are normal, so neither is an
    error, and the empty string is recorded as itself rather than guessed at.
    """
    pathadr = getattr(model, "hfield_pathadr", None)
    blob = getattr(model, "paths", None)
    if pathadr is None or blob is None:
        # Older MuJoCo did not expose the resolved asset paths. Say so in the
        # record rather than leaving a blank that reads as "built in memory".
        return "(mujoco build exposes no asset paths)"
    start = int(pathadr[hid])
    if start < 0:
        return ""
    end = blob.find(b"\x00", start)
    raw = blob[start:] if end < 0 else blob[start:end]
    return raw.decode("utf-8", errors="replace")
