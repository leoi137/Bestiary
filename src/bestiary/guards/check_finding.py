"""Oracle for `Finding` itself — the door every assertion passes through.

Run it:

    venv/bin/python -m bestiary.guards.check_finding

Every guard in the suite reports through `Finding`, so a defect here is a
defect in all sixteen at once and in every guard not yet written. That makes it
worth its own oracle even though it holds no domain knowledge about robots.

The bug this exists for, 2026-07-28: a guard that compares two arrays gets a
`numpy.bool_` back rather than a `bool`. It is truthy, it prints as `True`, and
`if finding.ok` behaves identically — so the whole suite stayed green while
`--json` raised `TypeError: Object of type bool is not JSON serializable`. The
machine-readable output was dead for two days and no check noticed, because the
only thing that distinguishes the two types is serialisation.

The fix coerces in `Finding.__post_init__` rather than at the call site. This
oracle is what makes that fix permanent: it asserts the coercion happens for a
type the suite does not currently produce anywhere, so a future guard that
introduces one cannot quietly break `--json` again.

Hermetic — constructs its own objects, touches no run, no checkpoint, and no
file. Milliseconds.
"""

from __future__ import annotations

import json

import numpy as np

from bestiary.guards import Finding


def _check(label: str, ok: bool, detail: str) -> tuple[str, bool, str]:
    return (label, ok, detail)


def run() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    # 1. numpy.bool_ is coerced to builtin bool.
    f = Finding("probe", np.bool_(True), "from a numpy comparison", n=3)
    out.append(
        _check(
            "numpy.bool_ ok is coerced to builtin bool",
            type(f.ok) is bool,
            f"type(ok) = {type(f.ok).__name__}, expected bool",
        )
    )

    # 2. The coercion preserves the value, in both directions. A coercion that
    #    silently flipped a verdict would be far worse than the crash it fixes.
    t = Finding("probe-true", np.bool_(True), "")
    fa = Finding("probe-false", np.bool_(False), "")
    out.append(
        _check(
            "coercion preserves the verdict",
            t.ok is True and fa.ok is False,
            f"True -> {t.ok!r}, False -> {fa.ok!r}",
        )
    )

    # 3. numpy integer n is coerced. np.int64 serialises no better than np.bool_.
    g = Finding("probe-n", True, "", n=np.int64(7))
    out.append(
        _check(
            "numpy integer n is coerced to builtin int",
            type(g.n) is int and g.n == 7,
            f"type(n) = {type(g.n).__name__}, value {g.n!r}",
        )
    )

    # 4. n=None survives. None and 0 are deliberately different facts -- not
    #    set-quantified versus quantified over nothing -- so a coercion that
    #    turned None into 0 would silently invent vacuity.
    h = Finding("probe-none", True, "")
    out.append(
        _check(
            "n=None is left as None, not coerced to 0",
            h.n is None,
            f"n = {h.n!r}",
        )
    )

    # 5. THE ACTUAL FAILURE. A Finding's fields must survive json.dumps. This is
    #    the assertion that would have caught the original bug; everything above
    #    explains why, this one is the regression itself.
    payload = {"ok": f.ok, "n": f.n, "label": f.label, "detail": f.detail}
    try:
        json.dumps(payload)
        serialisable, why = True, "json.dumps succeeded"
    except TypeError as exc:
        serialisable, why = False, f"json.dumps raised: {exc}"
    out.append(_check("a Finding built from numpy scalars is JSON-serialisable", serialisable, why))

    # 6. The vacuity rule still reads correctly through a coerced numpy bool --
    #    the property is the thing cycle 012 built and it depends on `ok`.
    v = Finding("probe-vacuous", np.bool_(True), "", n=0)
    nf = Finding("probe-failed-empty", np.bool_(False), "", n=0)
    out.append(
        _check(
            "vacuity survives coercion: PASS over an empty set is vacuous, FAIL is not",
            v.vacuous is True and nf.vacuous is False,
            f"pass/empty -> {v.vacuous!r}, fail/empty -> {nf.vacuous!r}",
        )
    )

    # 7. The negative-n contract is unchanged by the coercion.
    try:
        Finding("probe-negative", True, "", n=-1)
        raised = False
    except ValueError:
        raised = True
    out.append(
        _check(
            "a negative n still raises ValueError",
            raised,
            "ValueError raised" if raised else "NO ValueError -- the contract was lost",
        )
    )

    return out


def main() -> int:
    results = run()
    width = max(len(label) for label, _, _ in results)
    failed = 0
    for label, ok, detail in results:
        mark = "ok   " if ok else "FAIL "
        if not ok:
            failed += 1
        print(f"  {mark} {label:<{width}}  {detail}")

    print()
    if failed:
        print(f"{failed} of {len(results)} assertions FAILED.")
        return 1
    print(f"All {len(results)} assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
