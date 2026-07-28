"""Every run records the ground it trained on, and that ground has not moved.

Enforces `research/anomalies.jsonl` (2026-07-27) — *the terrain is the third
input to a run's dynamics and the only one with no hash, no config.json field,
and no guard* — and, through it, `learnings/001`, whose finding is that the
ground is not scenery: a Spyder policy scoring 1352 on the flat world scored
208 on the desert without a single line of the reward changing.

`checkpoint-width` asserts a run records the OBSERVATION it trained against.
`reward-spec` asserts it records the REWARD. This asserts the third leg, and
the three together are what makes a `config.json` provenance rather than a
launch command.

## Why this is the leg that had to fail silently

An observation change makes `SAC.load()` raise. A reward change at least leaves
an edited coefficient somebody can find with `git log -p`. A terrain change
leaves nothing at all: the widths match, the checkpoints load, the reward is
untouched, and every guard stays green while the robot walks on a world its
replay buffer has never seen.

It was nearly triggered. `research/scripts/compare_terrain_grids.py` measured
the proposed GRID=2048 regen against the committed terrain and found
correlation **+0.0610** — `generate.py` indexes its `(n, n)` phase array by FFT
bin, so changing `n` hands every phase to a different wavelength even at a
fixed seed. That regen would have made every ledger row incomparable with every
later one, with nothing on disk recording that the ground moved.

## Three states, and why `null` is one of them

`train.py` writes the `terrain_spec` key on every run, so a config says one of
three things:

    key absent   this run predates the record — reported, never failed
    key null     determined: a flat world with no heightfield
    key object   determined: this heightfield, by hash

The middle state is the reason a flat-plane run (`Spyder-v0`, `Hound-v0`,
`Ant-v5`, `Walker2d-v5`, `Humanoid-v5`) is not merely exempt from carrying a
terrain hash — it is *asserted flat*, and gaining a floor mid-run is a failure
here just as losing one is.

## Why legacy runs pass

Runs started before `terrain/spec.py` existed carry no key, and this guard does
not fail them. Back-filling a hash from whatever asset happens to be checked
out would manufacture exactly the false provenance the mechanism exists to
prevent — it would state, with a digest, that `hound_pd_desert_v0` trained on
today's `desert_hfield.bin`, which nobody can support. They are listed instead,
so the count is visible and shrinking. Same treatment `checkpoint-width` and
`reward-spec` give the runs that predate them, and for the same reason:
`guards --fast` gates every launch, so a guard that fails on history is a guard
that gets switched off.

## Cost

Fast tier. Building the two desert envs is one MuJoCo model load each (cached
per env id here), and the digest is a sha256 over a 4 MB float32 buffer —
single-digit milliseconds. It belongs in the tier that gates a launch, because
the launch is the moment the answer matters.
"""
from __future__ import annotations

import json
from functools import cache

from bestiary import paths
from bestiary.guards import Finding

# The naming convention `envs/__init__.py` establishes and documents ("Two
# robots, each in a flat world and a desert one"). Applied to the LIVE registry
# rather than to a list of env ids written down here, so a future
# `FooDesert-v0` is covered on the day it is registered and a deleted env
# cannot make the check pass vacuously.
_TERRAIN_ID_MARKER = "Desert"


@cache
def _our_env_ids() -> tuple[str, ...]:
    """The env ids `import bestiary.envs` registers, discovered not listed.

    Discovery is by entry point (`bestiary.envs.*`), which is a property of the
    code rather than of a constant somebody has to remember to update. The
    counter-argument in `guards/reward_spec.py` — that a registry walk passes
    vacuously if an env is deleted — is answered by `run()` asserting the set
    is non-empty and by `checkpoint-width` failing on any run whose env id no
    longer builds.
    """
    import bestiary.envs  # noqa: F401  -- registers the ids as an import side effect
    from gymnasium.envs.registration import registry

    return tuple(sorted(
        env_id for env_id, spec in registry.items()
        if str(spec.entry_point).startswith("bestiary.envs.")
    ))


def _measure(env_id: str):
    """Build the env from scratch and measure its ground. Never cached."""
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  -- registers the ids
    from bestiary.terrain import TerrainSpec

    env = gym.make(env_id)
    try:
        model = getattr(env.unwrapped, "model", None)
        if model is None:
            raise TypeError(
                f"{env_id} exposes no MuJoCo model, so its ground cannot be "
                f"measured; TerrainSpec.from_model needs a compiled mjModel"
            )
        return TerrainSpec.from_model(model)
    finally:
        env.close()


@cache
def _live_terrain_spec(env_id: str):
    """`_measure`, memoized per env id, or None for a flat world.

    Cached because building a desert env loads a 4 MB heightfield and several
    runs share an env id — the same reason `checkpoint_width._env_shapes` is
    cached. The determinism check below deliberately calls `_measure` instead,
    so it compares two real builds rather than a value against itself.
    """
    return _measure(env_id)


def _check_envs() -> list[Finding]:
    """Each env's ground is deterministic, and matches what its id claims."""
    env_ids = _our_env_ids()

    findings = [Finding(
        "importing bestiary.envs registers environments",
        bool(env_ids),
        f"{len(env_ids)} env id(s): {list(env_ids)}" if env_ids else
        "the gym registry holds no env with a bestiary.envs entry point — "
        "either registration broke or every env was deleted, and in both cases "
        "every per-env check below would have passed by having nothing to check",
        n=len(env_ids),  # the registry entries this walked
    )]

    # Each per-env finding below carries n=None: it asserts a property of ONE
    # named env, so there is no input set behind it and n=1 would invent one.
    for env_id in env_ids:
        try:
            spec = _live_terrain_spec(env_id)
        except Exception as exc:  # a broken env is a finding, not a crash
            findings.append(Finding(
                f"{env_id}: its ground is identified", False,
                f"could not measure the ground: {type(exc).__name__}: {exc}",
                n=None,
            ))
            continue

        # Rebuild and re-measure. A digest that depends on anything but the
        # compiled heightfield -- a float that formats differently, a geom
        # search that returns a different floor -- would make every recorded
        # hash unfalsifiable. Through `_measure`, not the cache, so this is two
        # genuine MuJoCo builds and not one value compared against itself.
        again = _measure(env_id)
        if (spec is None) != (again is None) or (
            spec is not None and again is not None and (
                spec.hash != again.hash or spec.field_hash != again.field_hash)
        ):
            findings.append(Finding(
                f"{env_id}: its ground is identified", False,
                f"the terrain hash is not deterministic across two builds: "
                f"{spec.hash if spec else None} then "
                f"{again.hash if again else None}. A recorded hash that cannot "
                f"be reproduced proves nothing.",
                n=None,
            ))
            continue

        expect_terrain = _TERRAIN_ID_MARKER in env_id
        has_terrain = spec is not None
        findings.append(Finding(
            f"{env_id}: its ground is identified",
            has_terrain == expect_terrain,
            (f"terrain {spec.hash} (field {spec.field_hash}) "
             f"{spec.nrow}x{spec.ncol} at {spec.cell_cm:.2f} cm/cell, "
             f"elevation {spec.z_min_m:+.2f}..{spec.z_max_m:+.2f} m, "
             f"from {spec.source!r}" if has_terrain else "flat world, no heightfield")
            + ("" if has_terrain == expect_terrain else
               f"  <- the id says {'terrain' if expect_terrain else 'flat'} and "
               f"the compiled model says {'terrain' if has_terrain else 'flat'}; "
               f"a run against it would record ground that contradicts its name"),
            n=None,
        ))
    return findings


def _check_runs() -> list[Finding]:
    """Recorded terrain re-hashes to itself, and still matches the live world."""
    from bestiary.terrain import TerrainSpec

    if not paths.RUNS.exists():
        # n=0, not None: on a fresh clone this stands in for every run check
        # below, and it quantified over zero runs.
        return [Finding("runs/ exists", True, "no runs yet — nothing to check", n=0)]

    bad: list[str] = []
    legacy: list[str] = []
    flat: list[str] = []
    verified: list[str] = []
    unverifiable: list[str] = []

    for config_path in sorted(paths.RUNS.glob("*/config.json")):
        run = config_path.parent.name
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            bad.append(f"{run}: config.json does not parse ({exc})")
            continue

        if "terrain_spec" not in config:
            legacy.append(run)
            continue

        record = config["terrain_spec"]
        env_id = config.get("env_id")

        try:
            live = _live_terrain_spec(env_id)
        except Exception as exc:
            # `checkpoint-width` owns "this run's env does not build" and
            # already fails it unless the run is declared retired. Reporting it
            # a second time here would double-count one problem and would fire
            # spuriously on a retired run, so this records that the comparison
            # did not happen rather than inventing a verdict for it.
            unverifiable.append(f"{run} ({env_id}: {type(exc).__name__})")
            continue

        if record is None:
            if live is None:
                flat.append(run)
            else:
                bad.append(
                    f"{run}: recorded as a FLAT world but {env_id} now compiles "
                    f"terrain {live.hash} ({live.nrow}x{live.ncol}). Every "
                    f"transition in its buffer was collected on level ground "
                    f"(learnings/001)."
                )
            continue

        # Re-derive the digests from the fields recorded beside them. This is
        # the only check that can catch a hand-edited config: comparing a
        # recorded hash to itself proves nothing about the record it sits in.
        try:
            rebuilt = TerrainSpec.from_record(record)
        except (KeyError, TypeError, ValueError) as exc:
            bad.append(f"{run}: terrain_spec cannot be rebuilt — {exc}")
            continue

        if rebuilt.hash != record.get("hash"):
            bad.append(
                f"{run}: recorded hash {record.get('hash')} but its own fields "
                f"hash to {rebuilt.hash} — config.json was edited by hand"
            )
        elif rebuilt.field_hash != record.get("field_hash"):
            bad.append(
                f"{run}: recorded field_hash {record.get('field_hash')} but its "
                f"own fields hash to {rebuilt.field_hash}"
            )
        elif live is None:
            bad.append(
                f"{run}: recorded terrain {record.get('hash')} but {env_id} now "
                f"compiles a FLAT world — the model lost its floor"
            )
        elif live.hash != record.get("hash"):
            same_field = live.field_hash == record.get("field_hash")
            bad.append(
                f"{run}: recorded {record.get('hash')} (field "
                f"{record.get('field_hash')}) but {env_id} is now {live.hash} "
                f"(field {live.field_hash}) — "
                + ("the same samples, rescaled or repositioned by a *_desert.xml "
                   "edit; one-line revert"
                   if same_field else
                   "a DIFFERENT heightfield. assets/terrain/desert_hfield.bin is "
                   "not the file this run trained on; another GRID changes the "
                   "world as surely as another seed does (correlation +0.0610 "
                   "for 1024 -> 2048, research/scripts/compare_terrain_grids.py)")
            )
        else:
            verified.append(run)

    findings = [Finding(
        "every recorded terrain still matches the ground its env compiles",
        not bad,
        "; ".join(bad) if bad else
        f"{len(verified)} run(s) on verified terrain, "
        f"{len(flat)} verified flat: {sorted(verified + flat)}",
        # The checkable set: runs carrying a terrain_spec key whose env built.
        # Legacy and unverifiable runs are named below, never verified here.
        n=len(verified) + len(flat) + len(bad),
    )]

    if unverifiable:
        # PASS on purpose: the env not building is checkpoint-width's failure,
        # not a second one. Reported so the coverage claim above stays honest.
        findings.append(Finding(
            "terrain comparisons that could not be made are named", True,
            f"{len(unverifiable)} run(s) whose env would not build, so their "
            f"terrain was not compared (checkpoint-width owns that failure): "
            f"{sorted(unverifiable)}",
            n=len(unverifiable),  # these runs ARE this assertion's subject
        ))

    # Not a failure: a fact, kept visible so the number is seen to shrink.
    findings.append(Finding(
        "runs predating the terrain record are declared, not back-filled", True,
        f"{len(legacy)} run(s) predate this record and cannot be checked: "
        f"{sorted(legacy)}" if legacy else "all runs pinned",
        n=len(legacy),  # the legacy runs ARE this assertion's subject
    ))
    return findings


def run() -> list[Finding]:
    return _check_envs() + _check_runs()
