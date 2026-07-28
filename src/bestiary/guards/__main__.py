"""Run the guards.

    python -m bestiary.guards            # every guard
    python -m bestiary.guards --fast     # skip the ones that step physics
    python -m bestiary.guards --json     # machine-readable
    python -m bestiary.guards --only privacy   # one guard (used by the pre-push hook)

Exit status is 0 only if every guard passed, so this gates a training launch:

    python -m bestiary.guards --fast && python -m bestiary.train.train ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from bestiary.guards import Finding, registry


def _run_all(fast_only: bool, only: str | None = None) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_ok = True

    selected = registry(fast_only=fast_only)
    if only:
        selected = tuple(g for g in selected if g.name == only)
        if not selected:
            raise SystemExit(
                f"no guard named {only!r}; available: "
                + ", ".join(g.name for g in registry())
            )

    for guard in selected:
        started = time.perf_counter()
        try:
            findings = guard.run()
        except Exception as exc:  # a broken guard is a failure, never a skip
            findings = [Finding(f"{guard.name} executes", False, f"{type(exc).__name__}: {exc}")]
        elapsed = time.perf_counter() - started

        ok = all(f.ok for f in findings)
        all_ok &= ok
        results.append(
            {
                "guard": guard.name,
                "enforces": guard.enforces,
                "cost": guard.cost,
                "ok": ok,
                # How much this guard actually examined, so a reader can tell a
                # green guard that checked everything from a green guard that
                # checked nothing without reading every detail string.
                "verified": sum(f.n for f in findings if f.n is not None),
                # Counted separately from `verified`, because a guard of purely
                # scalar thresholds (memory) has no input set at all and must
                # not read as one that examined an empty one. None and 0 are
                # different facts here exactly as they are on the Finding.
                "quantified": sum(f.n is not None for f in findings),
                "vacuous": sum(f.vacuous for f in findings),
                "seconds": round(elapsed, 2),
                "findings": [
                    {"label": f.label, "ok": f.ok, "detail": f.detail, "n": f.n} for f in findings
                ],
            }
        )
    return results, all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip guards that step physics")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--only", metavar="NAME", help="run a single guard by name")
    args = parser.parse_args()

    results, all_ok = _run_all(fast_only=args.fast, only=args.only)

    if args.json:
        print(json.dumps({"ok": all_ok, "guards": results}, indent=2))
        return 0 if all_ok else 1

    failed = 0
    vacuous = 0
    for result in results:
        mark = "ok" if result["ok"] else "FAILED"
        # A guard whose every assertion examined nothing is green and worthless;
        # say so on its header line rather than only per-assertion.
        blind = (
            " [VERIFIED NOTHING]"
            if result["ok"] and result["quantified"] and result["verified"] == 0
            else ""
        )
        print(f"\n{result['guard']}  [{mark}]{blind}  enforces {result['enforces']}  "
              f"({result['seconds']}s)")
        for finding in result["findings"]:
            if not finding["ok"]:
                prefix = "FAIL"
            elif finding["n"] == 0:
                prefix = "VACUOUS"
            else:
                prefix = "PASS"
            detail = f"   {finding['detail']}" if finding["detail"] else ""
            print(f"  [{prefix}] {finding['label']}{detail}")
            failed += not finding["ok"]
            vacuous += prefix == "VACUOUS"

    print("\n" + "=" * 66)
    if all_ok:
        print(f"All guards passed ({len(results)} guards).")
    else:
        print(f"{failed} assertion(s) failed across {len(results)} guards.")
        print("Each failure names the lesson it enforces — read that lesson before")
        print("deciding the guard is wrong.")

    # Printed on green runs too, and deliberately: the failure in learnings/014
    # happened on a run where everything passed.
    if vacuous:
        print(f"{vacuous} assertion(s) VACUOUS — passed while examining an empty input set.")
        print("Vacuous is not failure (a fresh clone has nothing to check) and it is")
        print("not verification either. research/learnings/014.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
