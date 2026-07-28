"""Arithmetic for lesson 010 — vacuous truth.

Two things, both of which the lesson quotes:

1. The logical fact, demonstrated rather than asserted: "every element of S has
   property P" evaluates TRUE for an empty S, for every P, including a P that
   nothing could ever satisfy.
2. The measured state of this repository's own guard suite, read from the
   committed audit in `research/scripts/guard_vacuity.py`.

    venv/bin/python docs/lessons/scripts/010_vacuous_truth.py
"""
from __future__ import annotations

import importlib.util
import sys

from bestiary.paths import RESEARCH

# The audit lives in research/scripts/, which is not an importable package, so
# it is loaded by path rather than copied. Copying it would let the lesson's
# numbers and the audit's numbers drift, which is the failure this repo's
# number rule exists to prevent.
_spec = importlib.util.spec_from_file_location(
    "guard_vacuity", RESEARCH / "scripts" / "guard_vacuity.py"
)
assert _spec and _spec.loader, f"cannot load the audit from {RESEARCH / 'scripts'}"
_guard_vacuity = importlib.util.module_from_spec(_spec)
sys.modules["guard_vacuity"] = _guard_vacuity
_spec.loader.exec_module(_guard_vacuity)
audit = _guard_vacuity.audit


def demonstrate_empty_set_truth() -> None:
    """Python's `all()` is the quantifier, and it agrees with the logic."""
    impossible = lambda x: x != x  # noqa: E731 — true for nothing at all

    populated = [1, 2, 3]
    empty: list[int] = []

    print("  all(impossible(x) for x in [1, 2, 3]) =", all(impossible(x) for x in populated))
    print("  all(impossible(x) for x in [])        =", all(impossible(x) for x in empty))
    print()
    print("  The second is TRUE. Not a Python quirk — it is what the quantifier")
    print("  means. 'Every x satisfies P' is shorthand for 'no x violates P',")
    print("  and an empty set cannot furnish a violator.")


def coverage(verified: int, claimed: int) -> float:
    """Fraction of the things an assertion SPEAKS ABOUT that it actually read.

    Deliberately separate from the assertion's own verdict: the whole lesson is
    that a guard can score 1.0 on the verdict and 0.0 here at the same time,
    with no contradiction.
    """
    return verified / claimed if claimed else float("nan")


def main() -> None:
    print("1. The logic\n")
    demonstrate_empty_set_truth()

    print("\n2. This repository's guard suite, measured\n")
    rows = audit()
    assertions = sum(r["assertions"] for r in rows)
    quantified = sum(r["quantified"] for r in rows)
    vacuous = sum(r["vacuous"] for r in rows)

    print(f"  guards                     {len(rows):>5}")
    print(f"  assertions                 {assertions:>5}")
    print(f"  set-quantified             {quantified:>5}   (can be vacuous at all)")
    print(f"  VACUOUS today              {vacuous:>5}   (passed over an empty set)")
    print(f"  vacuous as a share         {vacuous / quantified:>5.1%}")

    for r in rows:
        for label in r["vacuous_labels"]:
            print(f"    {r['guard']}: {label}")

    # The concrete case the lesson works through. measurement-provenance
    # asserted that every frozen checkpoint still hashes to what its
    # measurement recorded. Nine measurement JSONs named a checkpoint; zero of
    # them recorded a hash, so zero could be verified.
    print("\n3. The case worked\n")
    claimed, verified = 9, 0
    print(f"  measurements naming a checkpoint   {claimed}")
    print(f"  hashes actually verified           {verified}")
    print(f"  coverage c = {verified}/{claimed}              = {coverage(verified, claimed):.2f}")
    print("  guard verdict                      PASS")
    print()
    print("  Both are correct simultaneously, and that is the whole problem.")


if __name__ == "__main__":
    main()
