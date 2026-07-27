"""Assert the checkpoint-width guard's verdicts, including the retirement gate.

    python -m bestiary.guards.check_checkpoint_width
    python -m bestiary.guards.check_checkpoint_width -v    # print every verdict

A guard is the mechanism that makes a lesson unforgettable, which makes a wrong
guard worse than no guard: it reports coverage it does not have, and everyone
downstream believes it. So the guard needs its own oracle, in the same spirit
as `robots/hound/check.py` — claims checked against behaviour rather than left
as prose that used to be true.

The case this file exists for is `retired_runs.jsonl`. That file turns a red
guard green, which makes it exactly the kind of mechanism that quietly grows
into a way to silence real failures. The claim being defended is:

    a retired run must be GENUINELY dead, and a declaration that stops being
    true is itself a FAIL

and reasoning about an inverted assertion is not evidence that it inverts
correctly. Case 8 below is the load-bearing one.

Hermetic and fast: `paths.RUNS` and `paths.RETIRED` are pointed at a temporary
tree inside the repo, and the two env-introspection helpers are replaced, so
nothing here builds MuJoCo, touches the real `runs/`, or needs a GPU. A
checkpoint is a real zip carrying the same plain-JSON `data` member the guard
reads out of a Stable-Baselines3 save, because reading that member IS the
behaviour under test.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile

from bestiary import paths
from bestiary.guards import checkpoint_width as mod

# Inside the repo on purpose: `Scriptorium/` must never appear in a tracked
# public file (the privacy guard fails on it, and it has caught exactly that
# before), and a system temp path is out of scope for work in this workspace.
TMP = paths.REPO_ROOT / ".tmp_check_checkpoint_width"


class FakeSpec:
    """Stands in for an `ObsSpec`; the guard only reads `.hash` and `.width`."""

    def __init__(self, hash_: str, width: int) -> None:
        self.hash = hash_
        self.width = width


def write_run(
    name: str,
    env_id: str,
    ckpt_obs: int,
    ckpt_act: int,
    *,
    recorded_hash: str | None = None,
    recorded_width: int | None = None,
    checkpoints: tuple[str, ...] = ("ant_sac.zip", "ant_sac_best.zip"),
) -> None:
    run_dir = TMP / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    config: dict = {"env_id": env_id, "algo": "SAC"}
    if recorded_hash is not None:
        config["obs_spec"] = {"hash": recorded_hash, "width": recorded_width}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    payload = json.dumps(
        {
            "observation_space": {"_shape": [ckpt_obs]},
            "action_space": {"_shape": [ckpt_act]},
        }
    )
    for ckpt in checkpoints:
        with zipfile.ZipFile(run_dir / ckpt, "w") as z:
            z.writestr("data", payload)


def write_retired(rows: list[dict]) -> None:
    (TMP / "research").mkdir(parents=True, exist_ok=True)
    paths.RETIRED.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )


def retirement(run: str, reason: str = "obs-width-changed") -> dict:
    return {
        "run": run,
        "reason": reason,
        "retired_at": "2026-07-27",
        "retired_by": "check",
        "note": "fixture",
    }


# --- the fake world the guard introspects ------------------------------------
# env id -> (obs, act). A missing id raises, standing in for an unregistered env.
ENV_SHAPES = {
    "Live-v0": (169, 16),
    "Reordered-v0": (169, 16),
}
ENV_SPECS = {
    "Live-v0": FakeSpec("aaaa1111", 169),
    "Reordered-v0": FakeSpec("bbbb2222", 169),  # same width, different hash
}


def fake_env_shapes(env_id: str) -> tuple[int, int]:
    if env_id not in ENV_SHAPES:
        raise ValueError(f"Environment `{env_id}` doesn't exist")
    return ENV_SHAPES[env_id]


def fake_env_obs_spec(env_id: str):
    if env_id not in ENV_SPECS:
        raise ValueError(f"Environment `{env_id}` doesn't exist")
    return ENV_SPECS[env_id]


def verdict_for(findings, needle: str) -> bool | None:
    """The ok-flag of the single finding whose label contains `needle`."""
    hits = [f for f in findings if needle in f.label]
    if len(hits) != 1:
        return None
    return hits[0].ok


def main() -> int:
    verbose = "-v" in sys.argv
    saved = (paths.RUNS, paths.RETIRED, mod._env_shapes, mod._env_obs_spec)

    if TMP.exists():
        shutil.rmtree(TMP)
    (TMP / "runs").mkdir(parents=True)
    (TMP / "research").mkdir(parents=True)

    paths.RUNS = TMP / "runs"
    paths.RETIRED = TMP / "research" / "retired_runs.jsonl"
    mod._env_shapes = fake_env_shapes
    mod._env_obs_spec = fake_env_obs_spec

    passed = failed = 0

    def expect(label: str, got: bool | None, want: bool | None) -> None:
        nonlocal passed, failed
        ok = got == want
        if ok:
            passed += 1
            if verbose:
                print(f"  [ok]   {label}   (verdict {got})")
        else:
            failed += 1
            print(f"  [FAIL] {label}   expected {want}, got {got}")

    try:
        # --- live runs: the guard's original job, unchanged ------------------
        write_run("healthy", "Live-v0", 169, 16, recorded_hash="aaaa1111", recorded_width=169)
        write_run("width_moved", "Live-v0", 141, 16)
        write_run("spec_moved", "Reordered-v0", 169, 16,
                  recorded_hash="aaaa1111", recorded_width=169)
        write_run("env_gone", "Deleted-v0", 141, 16)
        write_retired([])

        f = mod.run()
        print("live runs (no retirements declared):")
        expect("1  matching checkpoint passes", verdict_for(f, "healthy/ant_sac.zip"), True)
        expect("2  width mismatch FAILS", verdict_for(f, "width_moved/ant_sac.zip"), False)
        expect("3  spec hash moved at same width FAILS",
               verdict_for(f, "spec_moved: obs spec"), False)
        expect("4  unbuildable env FAILS", verdict_for(f, "env_gone: env"), False)

        # --- retired runs: the inverted assertion ----------------------------
        write_retired([
            retirement("width_moved", "obs-width-changed"),
            retirement("env_gone", "env-unregistered"),
            retirement("spec_moved", "obs-spec-changed"),
        ])
        f = mod.run()
        print("\ngenuinely dead runs, declared:")
        expect("5  dead by width passes as declared",
               verdict_for(f, "width_moved: orphaned as declared"), True)
        expect("6  dead by unregistered env passes as declared",
               verdict_for(f, "env_gone: env"), True)
        expect("7  dead by moved spec hash passes as declared",
               verdict_for(f, "spec_moved: orphaned as declared"), True)
        expect("7b healthy run is still judged normally alongside them",
               verdict_for(f, "healthy/ant_sac.zip"), True)

        # --- THE ONE THAT MATTERS -------------------------------------------
        # Declaring a healthy run must not silence anything. If this ever
        # passes, retired_runs.jsonl has become a mute button and every
        # checkpoint-width failure downstream is unenforceable.
        write_retired([retirement("healthy", "obs-width-changed")])
        f = mod.run()
        print("\nstale declaration (a healthy run declared dead):")
        expect("8  retiring a run that LOADS is a FAIL",
               verdict_for(f, "healthy: orphaned as declared"), False)

        # A retired run with no checkpoints cannot be shown to be dead, so it
        # is not accepted on trust. record.retire refuses to write such a row;
        # this is the guard refusing to honour a hand-edited one.
        write_run("empty", "Live-v0", 169, 16, checkpoints=())
        write_retired([retirement("empty", "obs-width-changed")])
        f = mod.run()
        print("\nunverifiable declaration:")
        expect("9  retired run with no checkpoints is a FAIL",
               verdict_for(f, "empty: orphaned as declared"), False)

        # ...and it must say so TRUTHFULLY. The first version of this branch
        # reused the staleness sentence, telling the operator the run LOADS —
        # it had not been read at all — and instructing them to delete the one
        # surviving record that learning 003 bit us. A FAIL whose remedy makes
        # things worse is more dangerous than no FAIL.
        detail = next(x.detail for x in f if "empty: orphaned as declared" in x.label)
        expect("9b its message does not claim a load that never happened",
               "it LOADS" not in detail, True)
        expect("9c its message says the checkpoints are missing",
               "NO readable checkpoint" in detail, True)
        # Not "does not say delete its line" — the word itself must be absent.
        # An instruction is read by its verbs, and "do NOT delete" scans as
        # "delete" to someone triaging a wall of output at 3am.
        expect("9d its message never uses the word 'delete'",
               "delete" not in detail.lower(), True)

        # A run with no recorded obs spec has nothing to contradict a load, and
        # the message must not assert a spec comparison that never happened.
        write_run("stale_unpinned", "Live-v0", 169, 16)   # no recorded_hash
        write_retired([retirement("stale_unpinned", "obs-width-changed")])
        f = mod.run()
        detail = next(x.detail for x in f if "stale_unpinned: orphaned" in x.label)
        expect("9e stale + unpinned FAILS", verdict_for(f, "stale_unpinned: orphaned"), False)
        expect("9f ...without claiming its spec matches",
               "its spec matches" not in detail, True)

        # --- the file itself -------------------------------------------------
        print("\nthe declaration file itself:")
        paths.RETIRED.write_text('{"run": "healthy"}\nnot json\n', encoding="utf-8")
        expect("10 a malformed row FAILS loudly rather than reading as empty",
               verdict_for(mod.run(), "retired_runs.jsonl parses"), False)

        # A row that parses as JSON but is not a usable record must fail the
        # same way. These escaped the first parser as TypeError and crashed the
        # guard with a traceback naming neither the file nor the line.
        for n, bad in enumerate(['["a"]', '{"run": ["a"]}', '{"run": null}', '42'], 1):
            paths.RETIRED.write_text(bad + "\n", encoding="utf-8")
            expect(f"10{'abcd'[n - 1]} non-record row {bad!r} FAILS as a parse error",
                   verdict_for(mod.run(), "retired_runs.jsonl parses"), False)

        # Append-only means append-only: a second row for the same run must not
        # silently rewrite what a reader sees about the first.
        write_retired([retirement("healthy"), retirement("healthy")])
        expect("10e a duplicate declaration FAILS rather than last-wins",
               verdict_for(mod.run(), "retired_runs.jsonl parses"), False)

        # A row missing 'retired_at' must not crash the guard. The env-build
        # branch used to index it directly while the other branch used .get().
        write_run("env_gone2", "Deleted-v0", 141, 16)
        paths.RETIRED.write_text(
            json.dumps({"run": "env_gone2", "reason": "env-unregistered"}) + "\n",
            encoding="utf-8",
        )
        expect("10f a row missing 'retired_at' does not crash the guard",
               verdict_for(mod.run(), "env_gone2: env"), True)

        # runs/ is gitignored, so a clone has the declarations but not the runs.
        # That must be reported, never failed.
        write_retired([retirement("never_existed", "obs-width-changed")])
        f = mod.run()
        expect("11 a declaration with no run dir is reported, not failed",
               verdict_for(f, "retired runs are accounted for"), True)

    finally:
        paths.RUNS, paths.RETIRED, mod._env_shapes, mod._env_obs_spec = saved
        shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
