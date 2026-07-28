"""Guard: the ledger is well-formed, append-only, and honest about seeds.

Enforces `research/learnings/007` (a peak score hides an unreliable policy) and
the seed rule in `research/README.md`.

The ledger is the one artifact every later argument is built on. A row missing
`mean_eval_after_converge` lets a policy that scored 1218 once outrank one that
scored 1170 reliably; a row missing `seeds` lets a single noisy run be quoted
as an effect. Both mistakes read as normal rows, which is why they need a
machine to catch them.

Pure stdlib and no I/O beyond reading the ledger, so this runs in milliseconds
and can gate anything.
"""
from __future__ import annotations

import json
from datetime import date

from bestiary import paths
from bestiary.guards import Finding

BASE_FIELDS = (
    "run",
    "date",
    "robot",
    "env_id",
    "algo",
    "seed",
    "steps",
    "wall_clock_s",
    "fps",
    "best_eval_return",
    "final_ep_rew_mean",
    "final_ep_len_mean",
    "verdict",
    "notes",
)

# Added after learning 007 and the seed rule. Rows 1-2 predate them; every row
# from the third onward must carry all four.
FIELDS_FROM_ROW_3 = (
    "mean_eval_after_converge",
    "eval_crash_rate",
    "seeds",
    "provisional",
)

VERDICTS = frozenset({"plateau", "improved", "regressed", "crashed", "inconclusive"})

GRANDFATHERED_ROWS = 2


def run() -> list[Finding]:
    if not paths.LEDGER.exists():
        return [Finding("ledger exists", False, f"missing: {paths.LEDGER}")]

    lines = [ln for ln in paths.LEDGER.read_text().splitlines() if ln.strip()]
    findings: list[Finding] = [
        Finding("ledger is non-empty", bool(lines), f"{len(lines)} rows", n=len(lines))
    ]
    if not lines:
        return findings

    rows: list[dict[str, object]] = []
    for i, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(Finding(f"row {i} parses", False, str(exc), n=1))
            continue
        if not isinstance(row, dict):
            findings.append(
                Finding(f"row {i} is an object", False, type(row).__name__, n=1)
            )
            continue
        rows.append(row)

    findings.append(
        Finding("every row parses as an object", len(rows) == len(lines),
                f"{len(rows)}/{len(lines)}", n=len(lines))
    )

    missing_base = {
        str(r.get("run", f"row{i}")): sorted(set(BASE_FIELDS) - set(r))
        for i, r in enumerate(rows, start=1)
        if set(BASE_FIELDS) - set(r)
    }
    findings.append(
        # n is the rows examined, never the length of the violation list: a
        # clean check over every row and a check over no rows both print {}.
        Finding("every row has the base fields", not missing_base,
                "; ".join(f"{k}: {v}" for k, v in missing_base.items()),
                n=len(rows))
    )

    late = rows[GRANDFATHERED_ROWS:]
    missing_late = {
        str(r.get("run", "?")): sorted(set(FIELDS_FROM_ROW_3) - set(r))
        for r in late
        if set(FIELDS_FROM_ROW_3) - set(r)
    }
    findings.append(
        Finding(
            f"rows {GRANDFATHERED_ROWS + 1}+ carry the stability and seed fields",
            not missing_late,
            "; ".join(f"{k} missing {v}" for k, v in missing_late.items())
            or f"{len(late)} rows checked",
            # The grandfathered first rows are exempt, so they are not part of
            # what this verified.
            n=len(late),
        )
    )

    bad_verdict = {
        str(r.get("run", "?")): r.get("verdict")
        for r in rows
        if r.get("verdict") not in VERDICTS
    }
    findings.append(
        Finding("verdicts are from the allowed set", not bad_verdict,
                str(bad_verdict), n=len(rows))
    )

    # A single-seed row that does not admit it is the exact failure the seed
    # rule exists to prevent: one noisy run quoted later as an effect.
    dishonest = [
        str(r.get("run", "?"))
        for r in late
        if r.get("seeds") == 1 and r.get("provisional") is not True
    ]
    findings.append(
        Finding("single-seed rows are marked provisional", not dishonest,
                str(dishonest), n=len(late))
    )

    names = [str(r.get("run", "?")) for r in rows]
    dupes = sorted({n for n in names if names.count(n) > 1})
    findings.append(Finding("run names are unique", not dupes, str(dupes), n=len(names)))

    dates: list[date] = []
    unparseable: list[str] = []
    for r in rows:
        try:
            dates.append(date.fromisoformat(str(r.get("date"))))
        except ValueError:
            unparseable.append(str(r.get("run", "?")))
    findings.append(
        Finding("dates are ISO-8601", not unparseable, str(unparseable), n=len(rows))
    )
    findings.append(
        Finding(
            "rows are in non-decreasing date order",
            dates == sorted(dates),
            "a row inserted out of order means the file was rewritten, not appended",
            # Only the dates that parsed can be ordered; an unparseable one was
            # not placed in the sequence at all.
            n=len(dates),
        )
    )

    return findings
