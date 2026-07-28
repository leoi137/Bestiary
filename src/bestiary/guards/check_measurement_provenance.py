"""Oracle for `measurement-provenance` and `record/freeze.py`.

A guard that has only ever been observed passing is not a guard, it is a
decoration. On the repo as it stands today `measurement-provenance` passes with
**zero** hashes actually verified — every measurement predates the standard — so
the green tick proves nothing at all until this file exists.

So this runs the real functions against a temporary directory built to fail,
one failure mode at a time, and asserts each one is caught. It imports the
guard rather than reimplementing its logic; a reimplemented oracle tests the
copy, and the copy is not what runs in preflight.

Hermetic: everything happens under a `tempfile.TemporaryDirectory`. Writes
nothing to `research/`, `runs/` or `assets/`, and needs no GPU, no mujoco and
no trained policy. ~0.1 s.

    venv/bin/python -m bestiary.guards.check_measurement_provenance
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from bestiary.record.freeze import freeze_checkpoint

_PASS = "PASS"
_FAIL = "FAIL"


def _emit(ok: bool, label: str, detail: str) -> bool:
    print(f"  [{_PASS if ok else _FAIL}] {label}\n        {detail}")
    return ok


def _freeze_cases(tmp: Path) -> list[bool]:
    """freeze.py: does the copy actually make the artifact immutable?"""
    print("\nfreeze_checkpoint -- the copy, not a check around a mutable read")
    out: list[bool] = []

    run_dir = tmp / "run_a"
    run_dir.mkdir()
    ckpt = run_dir / "ant_sac_best.zip"
    ckpt.write_bytes(b"weights-at-950k")

    first = freeze_checkpoint(ckpt)
    out.append(_emit(
        first.frozen.exists() and first.frozen.name == f"{first.sha256}.zip",
        "the frozen path IS the hash",
        f"{first.frozen.name} for sha {first.sha256[:16]}... — a wrong file cannot "
        "hide behind a right name, which a fixed filename like 'ant_sac_best.zip' allows",
    ))

    # THE ACTUAL FAILURE, reproduced: overwrite the source in place, exactly as
    # the training callback does whenever an eval beats the previous best.
    ckpt.write_bytes(b"weights-at-1300k-DIFFERENT-POLICY")
    second = freeze_checkpoint(ckpt)
    out.append(_emit(
        second.sha256 != first.sha256
        and first.frozen.exists()
        and first.frozen.read_bytes() == b"weights-at-950k",
        "overwriting the source does NOT touch an already-frozen measurement",
        f"source moved {first.sha256[:12]}... -> {second.sha256[:12]}..., and the "
        f"earlier frozen copy still holds its original bytes. This is anomalies row 20: "
        "the JSON was committed at 10:58:13 and the checkpoint it named was gone by 11:07:01",
    ))

    out.append(_emit(
        freeze_checkpoint(ckpt).frozen == second.frozen,
        "freezing unchanged bytes twice is a no-op",
        f"content-addressed, so a re-measurement reuses {second.frozen.name} and costs "
        "one hash instead of another 5 MB",
    ))

    # A frozen file edited out of band must not be trusted just because its
    # NAME is right -- that is what makes it content-addressed rather than a
    # normal directory.
    tampered = second.frozen
    tampered.write_bytes(b"tampered")
    try:
        freeze_checkpoint(ckpt)
        caught = False
        detail = "freeze_checkpoint RETURNED on a frozen file whose contents do not match its name"
    except RuntimeError as exc:
        caught = "hashes to" in str(exc)
        detail = f"RuntimeError naming both hashes: {str(exc)[:110]}..."
    out.append(_emit(caught, "a frozen file whose contents contradict its name is refused", detail))
    tampered.write_bytes(b"weights-at-1300k-DIFFERENT-POLICY")  # restore

    try:
        freeze_checkpoint(run_dir / "does_not_exist.zip")
        caught, detail = False, "a missing checkpoint was silently accepted"
    except FileNotFoundError as exc:
        caught, detail = "does_not_exist.zip" in str(exc), f"FileNotFoundError naming the path: {exc}"
    out.append(_emit(caught, "a missing checkpoint raises and names the path", detail))

    fields = second.as_json_fields()
    out.append(_emit(
        not Path(fields["checkpoint_frozen"]).is_absolute()
        and len(fields["checkpoint_sha256"]) == 64,
        "the JSON fields are portable and the hash is full-length",
        f"checkpoint_frozen={fields['checkpoint_frozen']!r} is relative (an absolute path "
        f"would publish a home directory to a public repo and break on any clone); "
        f"sha256 is {len(fields['checkpoint_sha256'])} hex chars, not truncated",
    ))
    return out


def _guard_cases(tmp: Path) -> list[bool]:
    """measurement_provenance.run(): does it FAIL when it should?"""
    print("\nmeasurement-provenance -- verified in the failing direction")
    from bestiary.guards import measurement_provenance as mp

    out: list[bool] = []
    meas = tmp / "measurements"
    meas.mkdir()
    run_dir = tmp / "run_b"
    run_dir.mkdir()
    ckpt = run_dir / "ant_sac.zip"
    ckpt.write_bytes(b"honest-weights")
    frozen = freeze_checkpoint(ckpt)

    from bestiary import paths
    real_dir, real_gf, real_root = mp._MEASUREMENTS, mp._GRANDFATHERED, paths.REPO_ROOT

    def _run_against(files: dict[str, dict], grandfathered=frozenset()) -> dict[str, tuple[bool, str]]:
        for f in meas.glob("*.json"):
            f.unlink()
        for name, payload in files.items():
            (meas / name).write_text(json.dumps(payload))
        mp._MEASUREMENTS = meas
        mp._GRANDFATHERED = grandfathered
        paths.REPO_ROOT = tmp
        try:
            return {f.label: (f.ok, f.detail) for f in mp.run()}
        finally:
            mp._MEASUREMENTS, mp._GRANDFATHERED = real_dir, real_gf
            paths.REPO_ROOT = real_root

    rel = frozen.frozen.relative_to(tmp).as_posix()
    honest = {
        "good.json": {"checkpoint": "ant_sac.zip", "checkpoint_sha256": frozen.sha256,
                      "checkpoint_frozen": rel},
    }

    res = _run_against(honest)
    hash_key = next(k for k in res if "still hashes" in k)
    out.append(_emit(all(ok for ok, _ in res.values()),
                     "an honest measurement passes, with the hash ACTUALLY verified",
                     f"{res[hash_key][1]} — note '1 verified', not the vacuous 0 the live repo reports"))

    # The failure this whole cycle is about: the artifact changed after the
    # number was published.
    frozen.frozen.write_bytes(b"someone-overwrote-the-frozen-copy")
    res = _run_against(honest)
    out.append(_emit(not res[hash_key][0],
                     "a frozen file that no longer matches its recorded hash FAILS",
                     res[hash_key][1][:190]))
    frozen.frozen.write_bytes(b"honest-weights")

    # Absence must NOT fail: frozen copies are gitignored, so a clone has none.
    res = _run_against({"gone.json": {"checkpoint": "ant_sac.zip",
                                      "checkpoint_sha256": frozen.sha256,
                                      "checkpoint_frozen": "runs/nope/measured/x.zip"}})
    out.append(_emit(res[hash_key][0],
                     "a frozen copy absent from THIS machine passes, and says so",
                     res[hash_key][1][:190] + "  (a fresh clone has no frozen copies at all; "
                     "failing on absence would make the guard fail for every cloner)"))

    # A hash with no path is unverifiable and must not pass as if it were.
    res = _run_against({"nopath.json": {"checkpoint": "ant_sac.zip",
                                        "checkpoint_sha256": frozen.sha256}})
    out.append(_emit(not res[hash_key][0],
                     "a recorded hash with no frozen path FAILS rather than passing unverified",
                     res[hash_key][1][:190]))

    # A new measurement that records no identity at all.
    delinquent_key = next(k for k in _run_against(honest) if "records its sha256" in k)
    res = _run_against({"new.json": {"checkpoint": "ant_sac_best.zip", "trained": {"mean": 1.0}}})
    out.append(_emit(not res[delinquent_key][0],
                     "a NEW measurement naming a checkpoint but no sha256 FAILS",
                     res[delinquent_key][1][:190]))

    # ...unless it is explicitly grandfathered, and only then.
    res = _run_against({"old.json": {"checkpoint": "ant_sac_best.zip"}},
                       grandfathered=frozenset({"old.json"}))
    out.append(_emit(res[delinquent_key][0],
                     "the same file passes ONLY when explicitly grandfathered by name",
                     res[delinquent_key][1][:190] + "  (an explicit list is visible debt that "
                     "shrinks; a date cutoff is invisible debt that never does)"))

    # The grandfather list must not silently accumulate names of dead files.
    shrink_key = next(k for k in res if "only files that exist" in k)
    res = _run_against(honest, grandfathered=frozenset({"long_deleted.json"}))
    out.append(_emit(not res[shrink_key][0],
                     "a grandfather entry naming a file that no longer exists FAILS",
                     res[shrink_key][1][:190]))

    # Truncated vs full hash disagreeing = two artifacts named in one file.
    trunc_key = next(k for k in res if "truncated" in k)
    res = _run_against({"conflict.json": {"checkpoint": "ant_sac.zip",
                                          "checkpoint_sha256": frozen.sha256,
                                          "checkpoint_frozen": rel,
                                          "checkpoint_sha256_16": "0000000000000000"}})
    out.append(_emit(not res[trunc_key][0],
                     "a truncated hash contradicting the full hash in the same file FAILS",
                     res[trunc_key][1][:190]))

    res = _run_against({"agree.json": {"checkpoint": "ant_sac.zip",
                                       "checkpoint_sha256": frozen.sha256,
                                       "checkpoint_frozen": rel,
                                       "checkpoint_sha256_16": frozen.sha256[:16]}})
    out.append(_emit(res[trunc_key][0],
                     "a truncated hash that agrees passes",
                     res[trunc_key][1][:190]))

    # REGRESSION, from the refutation of cycle 011. greedy_eval --json writes
    # {"best": {...}, "latest": {...}} by default, so a top-level-only check
    # skips the file entirely and stays green -- blind to the default output
    # shape of one of the two tools this guard exists to police.
    res = _run_against({"nested.json": {
        "best": {"checkpoint": "ant_sac_best.zip", "trained": {"mean": 1.0}},
        "latest": {"checkpoint": "ant_sac.zip", "trained": {"mean": 2.0}},
        "selection_delta_mean": 1.0,
    }})
    out.append(_emit(not res[delinquent_key][0],
                     "greedy_eval's nested {best, latest} shape with no hashes FAILS",
                     res[delinquent_key][1][:190] + "  (a top-level-only walk would report "
                     "'0 JSON(s) name a checkpoint' and pass)"))

    res = _run_against({"nested_ok.json": {
        "best": {"checkpoint": "ant_sac.zip", "checkpoint_sha256": frozen.sha256,
                 "checkpoint_frozen": rel},
        "latest": {"checkpoint": "ant_sac.zip", "checkpoint_sha256": frozen.sha256,
                   "checkpoint_frozen": rel},
    }})
    out.append(_emit(res[hash_key][0] and "2 verified" in res[hash_key][1],
                     "both blocks of a nested measurement are verified, not just one",
                     res[hash_key][1][:190]))

    # A corrupt JSON must be reported, not skipped into a smaller sample.
    parse_key = next(k for k in res if "parses" in k)
    (meas / "broken.json").write_text("{not json")
    mp._MEASUREMENTS = meas
    paths.REPO_ROOT = tmp
    try:
        res2 = {f.label: (f.ok, f.detail) for f in mp.run()}
    finally:
        mp._MEASUREMENTS, paths.REPO_ROOT = real_dir, real_root
    out.append(_emit(not res2[parse_key][0],
                     "an unparseable measurement FAILS instead of shrinking the sample silently",
                     res2[parse_key][1][:190]))
    return out


def main() -> int:
    print(__doc__.strip().split("\n")[0])
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        results = _freeze_cases(tmp) + _guard_cases(tmp)

    passed = sum(results)
    print(f"\n{'=' * 66}\n{passed}/{len(results)} assertions passed")
    if passed != len(results):
        print("ORACLE FAILED — measurement-provenance does not catch what it claims to.")
        return 1
    print("measurement-provenance verified in BOTH directions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
