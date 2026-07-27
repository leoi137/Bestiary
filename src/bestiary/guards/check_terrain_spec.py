"""Assert the terrain-spec mechanism's verdicts, and that the hash really moves.

    python -m bestiary.guards.check_terrain_spec
    python -m bestiary.guards.check_terrain_spec -v    # print every verdict

Same reasoning as `guards/check_checkpoint_width.py`: a guard is the mechanism
that makes a lesson unforgettable, so a wrong guard is worse than no guard — it
reports coverage it does not have and everything downstream believes it.

The claims being defended, in the order they are checked:

1. **The digest is sensitive to the ground and to nothing else.** One float32
   sample out of a million moves both hashes; the asset's *path* moves neither.
2. **The two hashes split the way they are documented to split.** Rescaling the
   world moves `hash` and leaves `field_hash` alone; regenerating the field
   moves both.
3. **The guard fails on a terrain that moved and on a config edited by hand,
   and passes on a run that predates the record.** The third is the one that
   needs an oracle most: "legacy runs pass" is a hole by construction, and the
   only thing stopping it from becoming a mute button is that it triggers on an
   ABSENT key rather than on a null or a mismatch.
4. **The guard is registered in the tier that gates a launch.** A guard nobody
   runs before training is a guard that never fires.
5. **The whole thing works on the real asset.** The last section is not
   hermetic: it reproduces `assets/terrain/desert_hfield.bin` from
   `generate.build_height_m(seed=7)` and shows the digest matches, then rebuilds
   at GRID=2048 and shows it does not. That second number is the near-miss this
   mechanism was written for — `research/scripts/compare_terrain_grids.py`
   measured the two terrains at correlation +0.0610.

Sections 1-4 are hermetic: `paths.RUNS` points at a temporary tree inside the
repo and the guard's env introspection is replaced, so nothing builds MuJoCo,
touches the real `runs/`, or needs a GPU. Section 5 loads two models and
synthesizes terrain at two grids on CPU, which is why the whole file takes
around ten seconds rather than a tenth of one. **Nothing here writes to
`assets/`.**
"""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np

from bestiary import paths
from bestiary.guards import terrain_spec as mod
from bestiary.terrain.spec import TerrainSpec

# Inside the repo on purpose, for the same reason check_checkpoint_width says:
# a system temp path is out of scope for work in this workspace.
TMP = paths.REPO_ROOT / ".tmp_check_terrain_spec"

# generate.py's committed defaults, and the regen that was on the table. Both
# are read from the module rather than retyped, so a change to either is a
# change to what this file checks.
PROPOSED_GRID = 2048
GENERATE_SEED = 7


class FakeModel:
    """The handful of `mjModel` fields the terrain spec reads.

    Duck-typed rather than a real compiled model so that a case like "one
    sample differs" can be constructed at all — MuJoCo will not hand you a
    heightfield that differs from another by one float without writing a file,
    and writing terrain files is the one thing this must never do.

    Constructed to match a real model exactly: `hfield_data` is flat (MuJoCo
    stores it that way and `HeightField.from_model` reshapes it), `paths` is a
    null-separated blob, and `geom_type` carries the hfield enum so the geom
    search finds it.
    """

    def __init__(self, samples: np.ndarray, *,
                 size: tuple[float, float, float, float] = (40.0, 40.0, 5.05, 1.0),
                 pos: tuple[float, float, float] = (0.0, 0.0, -0.41),
                 quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
                 source: bytes = b"terrain/desert_hfield.bin") -> None:
        import mujoco

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        nrow, ncol = samples.shape
        self.nhfield = 1
        self.hfield_data = samples.reshape(-1)
        self.hfield_nrow = np.array([nrow], dtype=np.int32)
        self.hfield_ncol = np.array([ncol], dtype=np.int32)
        self.hfield_size = np.array([list(size)], dtype=np.float64)
        self.geom_type = np.array([int(mujoco.mjtGeom.mjGEOM_HFIELD)], dtype=np.int32)
        self.geom_dataid = np.array([0], dtype=np.int32)
        self.geom_pos = np.array([list(pos)], dtype=np.float64)
        self.geom_quat = np.array([list(quat)], dtype=np.float64)
        self.hfield_pathadr = np.array([0], dtype=np.int32)
        self.paths = source + b"\x00"


class FlatModel:
    """A world with no heightfield — every non-desert env in the repo."""

    nhfield = 0


def sample_field(n: int = 8, seed: int = 0) -> np.ndarray:
    """A tiny [0,1] field. Small on purpose: these cases test the digest's
    sensitivity, not MuJoCo's, and a 1024^2 array would make the file slow for
    no extra evidence."""
    rng = np.random.default_rng(seed)
    field = rng.random((n, n)).astype(np.float32)
    field[0, 0], field[-1, -1] = 0.0, 1.0   # a real hfield spans [0, 1] exactly
    return field


# --- the fake world the guard introspects ------------------------------------
BASE = TerrainSpec.from_model(FakeModel(sample_field()))
OTHER = TerrainSpec.from_model(FakeModel(sample_field(seed=1)))
ENV_GROUND: dict[str, TerrainSpec | None] = {
    "Desert-v0": BASE,
    "Flat-v0": None,
}


def fake_live(env_id: str):
    if env_id not in ENV_GROUND:
        raise ValueError(f"Environment `{env_id}` doesn't exist")
    return ENV_GROUND[env_id]


_ABSENT = object()


def write_run(name: str, env_id: str, terrain=_ABSENT) -> None:
    """One run directory. `terrain` absent = a run that predates the record."""
    run_dir = TMP / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict = {"env_id": env_id, "algo": "SAC"}
    if terrain is not _ABSENT:
        config["terrain_spec"] = terrain
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def clear_runs() -> None:
    shutil.rmtree(TMP / "runs", ignore_errors=True)
    (TMP / "runs").mkdir(parents=True)


def verdict_for(findings, needle: str) -> bool | None:
    hits = [f for f in findings if needle in f.label]
    return hits[0].ok if len(hits) == 1 else None


def detail_for(findings, needle: str) -> str:
    hits = [f for f in findings if needle in f.label]
    return hits[0].detail if len(hits) == 1 else ""


MATCH = "every recorded terrain"
LEGACY = "runs predating the terrain record"


def main() -> int:
    verbose = "-v" in sys.argv
    saved = (paths.RUNS, mod._live_terrain_spec)

    shutil.rmtree(TMP, ignore_errors=True)
    (TMP / "runs").mkdir(parents=True)
    paths.RUNS = TMP / "runs"
    mod._live_terrain_spec = fake_live

    passed = failed = 0

    def expect(label: str, got, want) -> None:
        nonlocal passed, failed
        if got == want:
            passed += 1
            if verbose:
                print(f"  [ok]   {label}   ({got!r})")
        else:
            failed += 1
            print(f"  [FAIL] {label}   expected {want!r}, got {got!r}")

    try:
        # --- 1. what the digest is, and is not, sensitive to ------------------
        print("the digest:")
        base = TerrainSpec.from_model(FakeModel(sample_field()))
        expect("1  the same ground hashes the same twice",
               (base.hash, base.field_hash), (BASE.hash, BASE.field_hash))

        nudged = sample_field()
        nudged[3, 4] = np.float32(nudged[3, 4] + 1e-4)
        one_off = TerrainSpec.from_model(FakeModel(nudged))
        expect("2  one changed sample moves field_hash", one_off.field_hash != base.field_hash, True)
        expect("2b ...and hash with it", one_off.hash != base.hash, True)

        renamed = TerrainSpec.from_model(FakeModel(sample_field(), source=b"terrain/moved.bin"))
        expect("3  the asset PATH is provenance, not identity",
               (renamed.hash, renamed.field_hash), (base.hash, base.field_hash))
        expect("3b ...and is still recorded", renamed.source, "terrain/moved.bin")

        # --- 2. the split between the two hashes ------------------------------
        print("\nthe two hashes:")
        taller = TerrainSpec.from_model(FakeModel(sample_field(), size=(40.0, 40.0, 10.1, 1.0)))
        expect("4  doubling z_span leaves field_hash alone",
               taller.field_hash, base.field_hash)
        expect("4b ...and moves hash, because every slope in the world doubled",
               taller.hash != base.hash, True)

        moved = TerrainSpec.from_model(FakeModel(sample_field(), pos=(0.0, 0.0, -1.0)))
        expect("5  repositioning the floor moves hash only",
               (moved.field_hash == base.field_hash, moved.hash != base.hash), (True, True))

        turned = TerrainSpec.from_model(FakeModel(sample_field(), quat=(0.0, 0.0, 0.0, 1.0)))
        expect("6  rotating the floor moves hash only",
               (turned.field_hash == base.field_hash, turned.hash != base.hash), (True, True))

        coarser = TerrainSpec.from_model(FakeModel(sample_field(n=4)))
        expect("7  a different grid moves field_hash", coarser.field_hash != base.field_hash, True)

        expect("8  from_record round-trips both hashes",
               (TerrainSpec.from_record(base.to_record()).hash,
                TerrainSpec.from_record(base.to_record()).field_hash),
               (base.hash, base.field_hash))

        record = base.to_record()
        del record["z_span_m"]
        try:
            TerrainSpec.from_record(record)
            expect("9  a record missing a field raises", "no exception", "KeyError")
        except KeyError as exc:
            expect("9  a record missing a field raises KeyError naming it",
                   "z_span_m" in str(exc), True)

        try:
            TerrainSpec.from_record({**base.to_record(), "z_span_m": 0.0})
            expect("10 a zero elevation span raises", "no exception", "ValueError")
        except ValueError as exc:
            expect("10 a zero elevation span raises ValueError with the value",
                   "0.0" in str(exc), True)

        # --- 3. the guard's verdicts ------------------------------------------
        print("\nthe guard, on runs:")
        clear_runs()
        write_run("on_it", "Desert-v0", BASE.to_record())
        expect("11 a run standing on the ground it recorded passes",
               verdict_for(mod._check_runs(), MATCH), True)

        clear_runs()
        write_run("swapped", "Desert-v0", OTHER.to_record())
        f = mod._check_runs()
        expect("12 a run whose heightfield was swapped FAILS", verdict_for(f, MATCH), False)
        d = detail_for(f, MATCH)
        expect("12b ...naming the recorded hash", OTHER.hash in d, True)
        expect("12c ...and the live one", BASE.hash in d, True)
        expect("12d ...and saying it is a different heightfield",
               "DIFFERENT heightfield" in d, True)

        clear_runs()
        rescaled = TerrainSpec.from_model(
            FakeModel(sample_field(), size=(40.0, 40.0, 10.1, 1.0)))
        write_run("rescaled", "Desert-v0", rescaled.to_record())
        f = mod._check_runs()
        expect("13 the same samples at another scale FAILS", verdict_for(f, MATCH), False)
        expect("13b ...and points at the XML, not the asset",
               "rescaled or repositioned" in detail_for(f, MATCH), True)

        clear_runs()
        forged = BASE.to_record()
        forged["z_span_m"] = 99.0          # fields say one thing, hash says another
        write_run("hand_edited", "Desert-v0", forged)
        f = mod._check_runs()
        expect("14 a config edited by hand FAILS", verdict_for(f, MATCH), False)
        expect("14b ...and says so",
               "edited by hand" in detail_for(f, MATCH), True)

        clear_runs()
        write_run("flat", "Flat-v0", None)
        expect("15 a run recorded as flat, on a flat env, passes",
               verdict_for(mod._check_runs(), MATCH), True)

        clear_runs()
        write_run("grew_ground", "Desert-v0", None)
        f = mod._check_runs()
        expect("16 a flat-recorded run whose env grew terrain FAILS",
               verdict_for(f, MATCH), False)
        expect("16b ...citing learnings/001", "learnings/001" in detail_for(f, MATCH), True)

        clear_runs()
        write_run("lost_ground", "Flat-v0", BASE.to_record())
        f = mod._check_runs()
        expect("17 a terrain run whose env went flat FAILS", verdict_for(f, MATCH), False)
        expect("17b ...and says the model lost its floor",
               "lost its floor" in detail_for(f, MATCH), True)

        # --- THE ONE THAT MATTERS ---------------------------------------------
        # Every run on this machine today predates the record, including the one
        # training right now. If this ever FAILS, `guards --fast` blocks every
        # launch and the pressure is to delete the guard. If the *absent* key
        # ever starts passing for the wrong reason -- say by being treated the
        # same as a null -- the guard silently stops covering flat runs.
        print("\nruns that predate the record:")
        clear_runs()
        write_run("legacy", "Desert-v0")   # no terrain_spec key at all
        f = mod._check_runs()
        expect("18 a run predating the record does NOT fail", verdict_for(f, LEGACY), True)
        expect("18b ...and is named, not silently skipped",
               "legacy" in detail_for(f, LEGACY), True)
        expect("18c ...and is not counted as verified",
               "legacy" in detail_for(f, MATCH), False)

        # A legacy config carrying the other two specs is the exact shape of
        # every config written between fbbd9af and this change.
        clear_runs()
        run_dir = TMP / "runs" / "obs_and_reward_only"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "env_id": "Desert-v0", "obs_spec": {"hash": "x"}, "reward_spec": {"hash": "y"},
        }), encoding="utf-8")
        expect("19 a config with obs+reward but no terrain is legacy, not a failure",
               verdict_for(mod._check_runs(), MATCH), True)

        clear_runs()
        write_run("dead_env", "Deleted-v0", BASE.to_record())
        f = mod._check_runs()
        expect("20 a run whose env will not build is not double-reported",
               verdict_for(f, MATCH), True)
        expect("20b ...but is named as uncompared",
               "dead_env" in detail_for(f, "could not be made"), True)

        clear_runs()
        (TMP / "runs" / "broken").mkdir(parents=True)
        (TMP / "runs" / "broken" / "config.json").write_text("{not json", encoding="utf-8")
        expect("21 an unparseable config FAILS rather than reading as legacy",
               verdict_for(mod._check_runs(), MATCH), False)

        # --- 4. registration ---------------------------------------------------
        print("\nregistration:")
        from bestiary.guards import registry
        fast = {g.name: g for g in registry(fast_only=True)}
        expect("22 the guard is registered", "terrain-spec" in {g.name for g in registry()}, True)
        expect("22b ...in the fast tier, which is what gates a launch",
               "terrain-spec" in fast, True)

    finally:
        paths.RUNS, mod._live_terrain_spec = saved
        shutil.rmtree(TMP, ignore_errors=True)

    # --- 5. the real asset ----------------------------------------------------
    # Not hermetic, and deliberately last: everything above could be right about
    # a fake model and wrong about MuJoCo.
    print("\nthe committed terrain (real models, no files written):")
    import mujoco

    from bestiary.terrain import generate

    live = TerrainSpec.from_model(
        mujoco.MjModel.from_xml_path(str(paths.HOUND_PD_DESERT_XML)))
    also = TerrainSpec.from_model(
        mujoco.MjModel.from_xml_path(str(paths.SPYDER_DESERT_XML)))
    expect("23 both desert models compile the same ground", also.hash, live.hash)
    expect("23b the flat twin has none",
           TerrainSpec.from_model(
               mujoco.MjModel.from_xml_path(str(paths.HOUND_PD_XML))), None)

    def digest_of(height_m: np.ndarray) -> str:
        """`save_hfield_bin`'s normalization, without writing anything."""
        lo, hi = height_m.min(), height_m.max()
        return TerrainSpec.from_model(
            FakeModel(((height_m - lo) / (hi - lo)).astype(np.float32))).data_sha256

    def build_at(grid: int) -> np.ndarray:
        """Synthesize at an arbitrary grid, in memory.

        `generate.GRID` and `generate.CELL` are module-level constants read at
        call time, so rebinding them is the supported way to build at another
        resolution -- the method `research/scripts/compare_terrain_grids.py`
        uses and documents. Restoring them in a `finally` is not optional:
        leaving GRID=2048 bound would make every later call in this process
        silently synthesize a different world.
        """
        saved_grid, saved_cell = generate.GRID, generate.CELL
        try:
            generate.GRID = grid
            generate.CELL = 2 * generate.HALF_EXTENT / grid
            return generate.build_height_m(GENERATE_SEED)
        finally:
            generate.GRID, generate.CELL = saved_grid, saved_cell

    # The committed asset IS generate.py at its defaults, bit for bit. This is
    # what licenses hashing MuJoCo's normalized array instead of the file: the
    # normalization is provably the identity here, because save_hfield_bin
    # already writes an exact [0, 1] span.
    expect("24 the committed asset reproduces from build_height_m(seed=7) at "
           f"GRID={generate.GRID}", digest_of(build_at(generate.GRID)), live.data_sha256)

    # ...and the regen that was proposed does not. This is the whole point.
    proposed = digest_of(build_at(PROPOSED_GRID))
    expect(f"25 the proposed GRID={PROPOSED_GRID} regen is a DIFFERENT terrain",
           proposed != live.data_sha256, True)
    print(f"       committed  {live.data_sha256[:32]}...  field_hash {live.field_hash}")
    print(f"       GRID={PROPOSED_GRID}  {proposed[:32]}...  "
          f"and correlates with it at +0.0610 "
          f"(research/scripts/compare_terrain_grids.py)")

    # The live envs, judged by the guard itself rather than by this file.
    env_findings = mod._check_envs()
    expect("26 every registered env's ground is identified and deterministic",
           all(x.ok for x in env_findings), True)
    if verbose:
        for x in env_findings:
            print(f"       {x}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
