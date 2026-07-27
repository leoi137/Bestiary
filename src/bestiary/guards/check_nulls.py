"""Oracle for the `nulls` guard: prove it fails when it should.

    venv/bin/python -m bestiary.guards.check_nulls

A guard that has only ever been observed passing is not evidence of anything --
it might be asserting nothing at all. `learnings/006` is this project's own
version of that lesson (an oracle that covered the robot and not the entry
point), and the `tracking-frame` guard shipped with a real bug that four hours
of green runs did not surface.

So this drives the guard's logic over synthetic inputs where the right answer is
known by construction, and asserts both directions: it fires on a breach, and it
stays quiet on a release. Nothing here touches `runs/` or `research/`.
"""
from __future__ import annotations

import sys

from bestiary.guards import nulls

# The real row-2 shape, so the oracle tests the condition actually in the file.
ROW2_GUARD = {
    "scope": {"kind": "env_id_regex", "pattern": "^Hound.*Desert-v0$"},
    "release_all_of": [
        {"kind": "reward_weight_abs_at_most", "term": "ctrl_cost", "value": 0.02},
        {"kind": "reward_term_absent", "term": "forward_velocity"},
        {"kind": "reward_term_present", "term": "track_cmd"},
    ],
}

RELEASED = {  # what hound_track_desert_s0 records
    "env_id": "HoundPDTrackDesert-v0",
    "reward_spec": {"terms": [
        {"name": "track_cmd", "weight": 1.0},
        {"name": "ctrl_cost", "weight": -0.01},
    ]},
}
BREACH = {  # what hound_pd_desert_s1 records
    "env_id": "HoundPDDesert-v0",
    "reward_spec": {"terms": [
        {"name": "forward_velocity", "weight": 1.0},
        {"name": "ctrl_cost", "weight": -0.01},
    ]},
}
OUT_OF_SCOPE = {  # a different robot entirely
    "env_id": "SpyderDesert-v0",
    "reward_spec": {"terms": [{"name": "forward_velocity", "weight": 1.0}]},
}
NO_SPEC = {"env_id": "HoundDesert-v0"}  # predates the reward spec
COSTLY = {  # released on terms, but the control cost is above the limit
    "env_id": "HoundPDTrackDesert-v0",
    "reward_spec": {"terms": [
        {"name": "track_cmd", "weight": 1.0},
        {"name": "ctrl_cost", "weight": -0.1},
    ]},
}


def _drive(rows, configs, harvested) -> dict[str, bool]:
    """Run the guard against injected data; return {label: ok}."""
    real = (nulls._rows, nulls._run_configs, nulls._harvested_runs)
    nulls._rows = lambda: rows
    nulls._run_configs = lambda: configs
    nulls._harvested_runs = lambda: frozenset(harvested)
    try:
        return {f.label: f.ok for f in nulls.run()}
    finally:
        nulls._rows, nulls._run_configs, nulls._harvested_runs = real


def _check(name: str, got: bool, want: bool, failures: list[str]) -> None:
    mark = "PASS" if got == want else "FAIL"
    print(f"  [{mark}] {name}   guard_ok={got}, expected {want}")
    if got != want:
        failures.append(name)


def main() -> int:
    failures: list[str] = []
    row = {"tried": "synthetic", "do_not_repeat_unless": "synthetic", "guard": ROW2_GUARD}
    breach_label = "nulls row 1: no unharvested run re-enters this dead end"

    # 1. An UNHARVESTED breach must fail. This is the assertion that would have
    #    caught the 10.7 GPU-hours while they were still being spent.
    r = _drive([row], [("live_breach", BREACH)], harvested=[])
    _check("an unharvested run re-entering a dead end FAILS", r[breach_label], False, failures)

    # 2. The same run, harvested, must NOT fail — spent hours are history.
    r = _drive([row], [("live_breach", BREACH)], harvested=["live_breach"])
    _check("the same run, once harvested, does not fail", r[breach_label], True, failures)

    # 3. A released run must not fail.
    r = _drive([row], [("released", RELEASED)], harvested=[])
    _check("a run satisfying every release clause passes", r[breach_label], True, failures)

    # 4. Scope must actually scope: another robot is not this dead end.
    r = _drive([row], [("spyder", OUT_OF_SCOPE)], harvested=[])
    _check("a run outside the scope regex is untouched", r[breach_label], True, failures)

    # 5. A run predating the reward spec is unchecked, not failed.
    r = _drive([row], [("old", NO_SPEC)], harvested=[])
    _check("a run with no reward_spec is named, not failed", r[breach_label], True, failures)

    # 6. Each clause must be load-bearing on its own: satisfying two of three
    #    is exactly the half-condition that was re-entered twice.
    r = _drive([row], [("costly", COSTLY)], harvested=[])
    _check("a single unmet clause (ctrl_cost 0.1 > 0.02) FAILS", r[breach_label], False, failures)

    # 7. Schema: a row with no `guard` block at all must fail, so the next dead
    #    end recorded cannot be unreadable by default.
    r = _drive([{"tried": "no guard block"}], [], harvested=[])
    _check(
        "a nulls row with no guard block FAILS",
        r["nulls row 1 declares how a machine would recognise it"], False, failures,
    )

    # 8. Schema: `checkable: false` without a reason must fail.
    r = _drive([{"tried": "x", "guard": {"checkable": False}}], [], harvested=[])
    _check(
        "checkable:false with no `why` FAILS",
        r["nulls row 1 declared uncheckable, with a reason"], False, failures,
    )

    # 9. An unimplemented clause kind must FAIL, never silently pass. A guard
    #    that skips what it does not understand reports coverage it lacks.
    bad = {"scope": {"kind": "env_id_regex", "pattern": "^Hound"},
           "release_all_of": [{"kind": "vibes", "term": "ctrl_cost"}]}
    r = _drive([{"tried": "x", "guard": bad}], [], harvested=[])
    _check(
        "an unimplemented clause kind FAILS",
        r["nulls row 1 release condition is evaluable"], False, failures,
    )

    # 10. Same for an unimplemented scope kind.
    bad_scope = {"scope": {"kind": "phase_of_moon"},
                 "release_all_of": [{"kind": "reward_term_present", "term": "track_cmd"}]}
    r = _drive([{"tried": "x", "guard": bad_scope}], [], harvested=[])
    _check(
        "an unimplemented scope kind FAILS",
        r["nulls row 1 release condition is evaluable"], False, failures,
    )

    print()
    if failures:
        print(f"ORACLE FAILED: {len(failures)} assertion(s): {failures}")
        return 1
    print("Oracle passed: the nulls guard fires on a live breach and stays quiet otherwise (10 checks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
