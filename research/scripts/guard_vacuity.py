"""How much of the guard suite actually checks something?

This is the audit `research/learnings/014` named as its own falsifier:

    A coverage audit of the other twelve guards finds every one has a non-empty
    input set, and always did on its commit day. Then `measurement-provenance`
    was careless rather than symptomatic, and the general claim is overreach.

Run it:

    venv/bin/python research/scripts/guard_vacuity.py
    venv/bin/python research/scripts/guard_vacuity.py --json

Two numbers, and confusing them is the whole trap this script exists to avoid:

**VACUOUS TODAY** — assertions that passed while quantifying over an empty set
*on this repository, right now*. This is what the falsifier asks about.

**VACUITY-CAPABLE** — assertions that *could* go vacuous if their input set
emptied, i.e. every set-quantified assertion (`n is not None`). A guard reading
`runs/` is vacuity-capable because `runs/` is gitignored and a fresh clone has
none; a guard that constructs its own inputs is not, because its set cannot
empty without someone editing the guard.

A hand read of the guard source conflates these and overestimates the first.
That is not hypothetical either: cycle 012's recall sweep read the source by
hand, called ten of sixteen modules "vacuity-capable", and concluded the
falsifier "appears already refuted". The measured answer disagreed. Which is
why the number rule exists — this file is the difference.

**The ruling, declared in the cycle note before the count was taken**, because
otherwise the number is unreproducible: where a guard passes while naming an
uncheckable remainder ("9 run(s) predate the spec record"), `n` counts the
CHECKABLE set only. A pass over 2 verified and 11 grandfathered is n=2 — a real
check with poor coverage. A pass over 0 verified and 11 grandfathered is n=0,
vacuous, however honestly the remainder is named.

Counting `Finding(` literals in the source would be the wrong instrument: most
guards emit one finding per input item, so the assertion count is itself a
function of the input set. There are 95 static literals and ~117 live
assertions. Everything here comes from `registry()`.
"""
from __future__ import annotations

import argparse
import json

from bestiary.guards import Finding, registry


def audit() -> list[dict]:
    """Run every registered guard and describe what each assertion examined."""
    rows: list[dict] = []
    for guard in registry():
        try:
            findings = guard.run()
        except Exception as exc:  # a guard that cannot run verified nothing
            findings = [Finding(f"{guard.name} executes", False, f"{type(exc).__name__}: {exc}")]

        quantified = [f for f in findings if f.n is not None]
        rows.append(
            {
                "guard": guard.name,
                "cost": guard.cost,
                "assertions": len(findings),
                # Set-quantified assertions: the ones for which "how much did
                # you look at" is a meaningful question at all.
                "quantified": len(quantified),
                # Scalar thresholds and single named properties. Not a gap —
                # declaring a set size here would be a lie.
                "unquantified": len(findings) - len(quantified),
                "vacuous": sum(f.vacuous for f in findings),
                "failed": sum(not f.ok for f in findings),
                "verified": sum(f.n for f in quantified),
                "vacuous_labels": [f.label for f in findings if f.vacuous],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args()

    rows = audit()
    totals = {
        k: sum(r[k] for r in rows)
        for k in ("assertions", "quantified", "unquantified", "vacuous", "failed", "verified")
    }
    totals["guards"] = len(rows)
    totals["guards_with_vacuous"] = sum(r["vacuous"] > 0 for r in rows)
    # The falsifier's own question, in one boolean: is measurement-provenance
    # alone, or is vacuity a property of the suite?
    totals["vacuous_outside_measurement_provenance"] = sum(
        r["vacuous"] for r in rows if r["guard"] != "measurement-provenance"
    )

    if args.json:
        print(json.dumps({"totals": totals, "guards": rows}, indent=2))
        return 0

    print(f"{'guard':<26}{'cost':<6}{'asrt':>5}{'quant':>7}{'vac':>5}{'fail':>6}{'verified':>10}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['guard']:<26}{r['cost']:<6}{r['assertions']:>5}{r['quantified']:>7}"
            f"{r['vacuous']:>5}{r['failed']:>6}{r['verified']:>10}"
        )
    print("-" * 65)
    print(
        f"{'TOTAL':<26}{'':<6}{totals['assertions']:>5}{totals['quantified']:>7}"
        f"{totals['vacuous']:>5}{totals['failed']:>6}{totals['verified']:>10}"
    )

    print(f"\n{totals['guards']} guards, {totals['assertions']} assertions.")
    print(
        f"{totals['quantified']} are set-quantified (vacuity-CAPABLE); "
        f"{totals['unquantified']} are scalar thresholds or single named properties."
    )
    print(f"{totals['vacuous']} are VACUOUS TODAY, across {totals['guards_with_vacuous']} guard(s).")
    for r in rows:
        for label in r["vacuous_labels"]:
            print(f"    {r['guard']}: {label}")

    outside = totals["vacuous_outside_measurement_provenance"]
    print(f"\nlearnings/014's falsifier — vacuous assertions OUTSIDE "
          f"measurement-provenance: {outside}")
    if outside == 0:
        print("  → CONFIRMED on the 'today' clause: every other guard has a non-empty")
        print("    input set. measurement-provenance was careless, not symptomatic,")
        print("    and 014's generalisation is overreach as written.")
    else:
        print("  → REFUTED: vacuity is not confined to the guard 014 was written about.")
    print("  The falsifier's second clause — 'and always did on its commit day' —")
    print("  is NOT tested here. It needs git archaeology, not today's run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
