"""Assert that `record.retire` refuses everything it claims to refuse.

    python -m bestiary.record.check_retire
    python -m bestiary.record.check_retire -v

`CLAUDE.md` states as an invariant that *retiring a healthy run is not a policy
violation caught in review, it is impossible*. That claim rests entirely on the
refusal gate in `build_row`, so the gate is the thing most worth an oracle: if
it silently stops refusing, `retired_runs.jsonl` becomes a mute button for the
guard that gates every training launch, and nothing else in the repo notices.

`guards/check_checkpoint_width.py` covers the other half — the guard's inverted
assertion, and its behaviour on a hand-edited file.

Hermetic: `paths.RUNS`/`paths.RETIRED` point at a temporary tree inside the
repo, and the two functions that touch gymnasium are replaced, so nothing here
builds MuJoCo or needs a GPU.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile

import gymnasium.error as gerr

from bestiary import paths
from bestiary.record import retire as mod

TMP = paths.REPO_ROOT / ".tmp_check_retire"

# What the fake world reports, rebound per case.
STATE: dict = {}


def fake_env_shapes(env_id: str):
    return STATE["live"], STATE["build_error"]


def fake_spec_hashes(config, env_id, buildable):
    return STATE["recorded_hash"], STATE["live_hash"]


def write_run(name: str, env_id: str, obs: int, act: int, *, with_ckpt: bool = True) -> None:
    d = TMP / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"env_id": env_id}), encoding="utf-8")
    if with_ckpt:
        payload = json.dumps(
            {"observation_space": {"_shape": [obs]}, "action_space": {"_shape": [act]}}
        )
        for ckpt in ("ant_sac.zip", "ant_sac_best.zip"):
            with zipfile.ZipFile(d / ckpt, "w") as z:
                z.writestr("data", payload)


def main() -> int:
    verbose = "-v" in sys.argv
    saved = (paths.RUNS, paths.RETIRED, mod.env_shapes, mod.spec_hashes)

    if TMP.exists():
        shutil.rmtree(TMP)
    (TMP / "runs").mkdir(parents=True)
    (TMP / "research").mkdir(parents=True)
    paths.RUNS = TMP / "runs"
    paths.RETIRED = TMP / "research" / "retired_runs.jsonl"
    mod.env_shapes = fake_env_shapes
    mod.spec_hashes = fake_spec_hashes

    passed = failed = 0

    def set_world(live, build_error=None, recorded_hash=None, live_hash=None) -> None:
        STATE.update(live=live, build_error=build_error,
                     recorded_hash=recorded_hash, live_hash=live_hash)

    def expect_refusal(label: str, run: str, reason: str, note: str = "why") -> None:
        nonlocal passed, failed
        try:
            mod.build_row(run, reason, note, "check")
        except SystemExit as exc:
            passed += 1
            if verbose:
                print(f"  [ok]   {label}\n           -> {str(exc).splitlines()[0][:100]}")
            return
        except Exception as exc:  # a traceback is not a refusal
            failed += 1
            print(f"  [FAIL] {label}   raised {type(exc).__name__} instead of refusing: {exc}")
            return
        failed += 1
        print(f"  [FAIL] {label}   IT WAS ACCEPTED")

    def expect_accepted(label: str, run: str, reason: str) -> None:
        nonlocal passed, failed
        try:
            row = mod.build_row(run, reason, "why", "check")
        except BaseException as exc:
            failed += 1
            print(f"  [FAIL] {label}   refused/raised: {str(exc).splitlines()[0][:120]}")
            return
        passed += 1
        if verbose:
            print(f"  [ok]   {label}   -> {row['reason']}")

    def expect(label: str, got, want) -> None:
        nonlocal passed, failed
        if got == want:
            passed += 1
            if verbose:
                print(f"  [ok]   {label}")
        else:
            failed += 1
            print(f"  [FAIL] {label}   expected {want}, got {got}")

    try:
        print("the refusal gate — a healthy run cannot be retired:")
        write_run("healthy", "Live-v0", 169, 16)
        set_world(live=(169, 16))
        expect_refusal("1  a run that still LOADS is refused", "healthy", "obs-width-changed")
        expect_refusal("2  ...whatever reason is claimed", "healthy", "obs-spec-changed")
        expect_refusal("3  ...and by the action reason too", "healthy", "action-width-changed")

        print("\nrows that cannot be verified are refused:")
        write_run("empty", "Live-v0", 169, 16, with_ckpt=False)
        expect_refusal("4  a run with no checkpoint is empty, not retired",
                       "empty", "obs-width-changed")
        expect_refusal("5  a row with no note is noise", "healthy", "obs-width-changed", "")
        expect_refusal("6  an unknown reason is refused", "healthy", "made-up-reason")
        (TMP / "runs" / "noenv").mkdir(parents=True, exist_ok=True)
        (TMP / "runs" / "noenv" / "config.json").write_text("{}", encoding="utf-8")
        expect_refusal("7  a config with no env_id refuses, not tracebacks",
                       "noenv", "obs-width-changed")
        (TMP / "runs" / "badjson").mkdir(parents=True, exist_ok=True)
        (TMP / "runs" / "badjson" / "config.json").write_text("{oops", encoding="utf-8")
        expect_refusal("8  an unreadable config refuses, not tracebacks",
                       "badjson", "obs-width-changed")
        expect_refusal("9  a run that does not exist is refused",
                       "no_such_run", "obs-width-changed")

        print("\nevery reason must be backed by a measurement:")
        write_run("width_moved", "Live-v0", 141, 16)
        set_world(live=(169, 16))
        expect_accepted("10 obs width really moved -> accepted", "width_moved",
                        "obs-width-changed")
        expect_refusal("11 ...but 'env-unregistered' is refused while the env builds",
                       "width_moved", "env-unregistered")

        write_run("act_moved", "Live-v0", 169, 12)
        expect_accepted("12 action width really moved -> accepted", "act_moved",
                        "action-width-changed")
        write_run("act_same", "Live-v0", 141, 16)
        expect_refusal("13 'action-width-changed' with an unchanged action width is refused",
                       "act_same", "action-width-changed")

        write_run("reordered", "Live-v0", 169, 16)
        set_world(live=(169, 16), recorded_hash="aaaa", live_hash="bbbb")
        expect_accepted("14 spec hash moved at identical width -> accepted",
                        "reordered", "obs-spec-changed")
        set_world(live=(169, 16), recorded_hash="aaaa", live_hash="aaaa")
        expect_refusal("15 'obs-spec-changed' with an unmoved hash is refused",
                       "reordered", "obs-spec-changed")

        print("\nan env that will not build is not the same as an env that is gone:")
        write_run("gone", "Deleted-v0", 141, 16)
        set_world(live=None, build_error=gerr.NameNotFound("Environment `Deleted` doesn't exist"))
        expect_accepted("16 a genuinely unregistered id -> accepted", "gone", "env-unregistered")
        set_world(live=None, build_error=gerr.VersionNotFound("version v99 does not exist"))
        expect_accepted("17 a version that no longer exists -> accepted",
                        "gone", "env-unregistered")

        # THE ONE THAT MATTERS in this file. If a merely BROKEN env counted as
        # a removed one, the load gate above could never run for it, and any
        # run whose env is temporarily unbuildable could be retired at will.
        set_world(live=None, build_error=FileNotFoundError("assets/terrain/desert.png"))
        expect_refusal("18 a BROKEN env (missing asset) is refused, not retired",
                       "gone", "env-unregistered")
        set_world(live=None, build_error=ImportError("cannot import name 'HoundEnv'"))
        expect_refusal("19 a BROKEN env (import error) is refused, not retired",
                       "gone", "env-unregistered")

        print("\nstructural guarantees:")
        expect("20 every reason has a measurement behind it",
               set(mod.CROSS_CHECKS) == set(mod.REASONS), True)

        print("\nreading the file back:")
        paths.RETIRED.write_text(
            json.dumps({"run": "a"}) + "\n" + json.dumps({"run": "a"}) + "\n", encoding="utf-8"
        )
        try:
            mod.read_rows()
            expect("21 a duplicate row raises", False, True)
        except ValueError as exc:
            expect("21 a duplicate row raises, naming the line", ":2" in str(exc), True)
        paths.RETIRED.write_text('{"run": ["a"]}\n', encoding="utf-8")
        try:
            mod.read_rows()
            expect("22 a non-string run name raises", False, True)
        except ValueError:
            expect("22 a non-string run name raises", True, True)

    finally:
        paths.RUNS, paths.RETIRED, mod.env_shapes, mod.spec_hashes = saved
        shutil.rmtree(TMP, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
