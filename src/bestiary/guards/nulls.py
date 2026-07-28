"""Guard: a recorded dead end must not be re-entered without its release condition.

Enforces `research/nulls.jsonl` itself, which until now enforced nothing.

`nulls.jsonl` exists so a dead end is paid for once. Each row carries a
`do_not_repeat_unless` — the condition under which the thing is worth trying
again. Row 2's reads:

    ctrl_cost_weight is lowered to 0.02 AND the forward term is replaced by
    command tracking

Both hound runs did the first and neither did the second. The recorded dead end
was re-entered at half its condition, twice, for ~10.7 GPU-hours (29,097s +
9,291s), and no check could have fired because nothing in `src/` referenced the
file: no path constant, no schema, no reader. That is the memory system failing
at the exact moment it exists for.

**Why this guard could not have been written before 2026-07-27.** The condition
above is a statement about reward parameters, and `config.json` did not record
reward parameters — the anomaly noting that ends "Requires reward parameters in
config.json, which they are not." Cycle 006 put them there as `reward_spec`,
hashed. The blocker named in the record was removed and nothing re-examined the
record; cycle 007's recall found it while looking for something else.

Three assertions, and the third is the one that outlives the current rows:

1. **Every null row declares a machine-readable release condition, or declares
   in writing why it cannot have one.** A row with neither fails. This is what
   makes the guard grow with the file instead of decaying: the next person to
   record a dead end has to say how a machine would recognise it.
2. **A row that claims to be checkable is actually evaluable** — its scope and
   clauses use kinds this module implements. An unknown kind is a FAIL, never a
   silent pass, because a guard that skips what it does not understand reports
   coverage it does not have.
3. **No run in `runs/` matches a dead end's scope while failing its release
   condition.** This is the assertion that would have caught the 10.7 hours.

Rows that genuinely cannot be checked are *named*, not skipped — the same
treatment `checkpoint-width` gives runs predating the observation spec. Row 1
(warm-starting across a reward change) is the standing example: nothing on disk
records that a run was warm-started, which `learnings/002` already says in its
`guard:` field, so no amount of schema fixes it.
"""
from __future__ import annotations

import json
import re

from bestiary import paths
from bestiary.guards import Finding

# The closed set of clause kinds. Adding one is deliberate: a kind this module
# does not implement is a FAIL (assertion 2), so the file cannot quietly grow
# conditions nothing evaluates.
_SCOPE_KINDS = frozenset({"env_id_regex"})
_CLAUSE_KINDS = frozenset(
    {"reward_weight_abs_at_most", "reward_term_absent", "reward_term_present"}
)


def _rows() -> list[dict]:
    if not paths.NULLS.exists():
        return []
    return [
        json.loads(line)
        for line in paths.NULLS.read_text().splitlines()
        if line.strip()
    ]


def _run_configs() -> list[tuple[str, dict]]:
    """(run name, config.json) for every run that has one.

    A run without a `config.json` predates the record and is invisible here.
    That is reported by assertion 3's detail rather than hidden.
    """
    if not paths.RUNS.exists():
        return []
    out: list[tuple[str, dict]] = []
    for d in sorted(p for p in paths.RUNS.iterdir() if p.is_dir()):
        cfg = d / "config.json"
        if cfg.exists():
            try:
                out.append((d.name, json.loads(cfg.read_text())))
            except json.JSONDecodeError:
                continue
    return out


def _harvested_runs() -> frozenset[str]:
    """Run names that already have a ledger row — i.e. spent, recorded history."""
    if not paths.LEDGER.exists():
        return frozenset()
    names = set()
    for line in paths.LEDGER.read_text().splitlines():
        if line.strip():
            names.add(str(json.loads(line).get("run", "")))
    return frozenset(names)


def _terms(config: dict) -> dict[str, dict]:
    """Reward terms by name, or {} for a run predating the reward spec."""
    spec = config.get("reward_spec") or {}
    return {str(t.get("name")): t for t in spec.get("terms", [])}


def _in_scope(scope: dict, config: dict) -> bool:
    """Unknown scope kinds are rejected upstream, so this only sees known ones."""
    kind = scope.get("kind")
    if kind == "env_id_regex":
        return bool(re.search(str(scope["pattern"]), str(config.get("env_id", ""))))
    raise AssertionError(f"unreachable: scope kind {kind!r} passed validation")


def _clause_holds(clause: dict, config: dict) -> tuple[bool, str]:
    """Does one release clause hold for this run? Returns (holds, evidence)."""
    kind = clause.get("kind")
    terms = _terms(config)

    if kind == "reward_weight_abs_at_most":
        name, limit = str(clause["term"]), float(clause["value"])
        term = terms.get(name)
        if term is None:
            return False, f"no reward term {name!r} recorded"
        w = abs(float(term.get("weight", 0.0)))
        return w <= limit, f"|weight({name})| = {w:g}, limit {limit:g}"

    if kind == "reward_term_absent":
        name = str(clause["term"])
        return name not in terms, f"term {name!r} {'absent' if name not in terms else 'PRESENT'}"

    if kind == "reward_term_present":
        name = str(clause["term"])
        return name in terms, f"term {name!r} {'present' if name in terms else 'MISSING'}"

    raise AssertionError(f"unreachable: clause kind {kind!r} passed validation")


def _validate(guard: dict) -> str | None:
    """Return an error string if this row's guard block is not evaluable."""
    scope = guard.get("scope")
    if not isinstance(scope, dict) or scope.get("kind") not in _SCOPE_KINDS:
        return (
            f"scope kind {(scope or {}).get('kind')!r} is not implemented; "
            f"known: {sorted(_SCOPE_KINDS)}"
        )
    clauses = guard.get("release_all_of")
    if not isinstance(clauses, list) or not clauses:
        return "release_all_of must be a non-empty list"
    for c in clauses:
        if not isinstance(c, dict) or c.get("kind") not in _CLAUSE_KINDS:
            return (
                f"clause kind {(c or {}).get('kind')!r} is not implemented; "
                f"known: {sorted(_CLAUSE_KINDS)}"
            )
    return None


def run() -> list[Finding]:
    rows = _rows()
    if not rows:
        return [
            Finding("nulls.jsonl has rows to check", True, "no recorded dead ends", n=0)
        ]

    findings: list[Finding] = []
    configs = _run_configs()
    declared_uncheckable: list[str] = []
    checkable: list[tuple[int, dict, dict]] = []

    # Assertion 1 + 2 -- the schema, per row.
    for i, row in enumerate(rows, start=1):
        tried = str(row.get("tried", ""))[:60]
        guard = row.get("guard")

        if guard is None:
            findings.append(
                Finding(
                    f"nulls row {i} declares how a machine would recognise it",
                    False,
                    f"no `guard` block on {tried!r}  <- add a machine-readable "
                    f"release condition, or declare {{'checkable': false, 'why': ...}}",
                    n=1,   # this row
                )
            )
            continue

        if guard.get("checkable") is False:
            why = str(guard.get("why", "")).strip()
            findings.append(
                Finding(
                    f"nulls row {i} declared uncheckable, with a reason",
                    bool(why),
                    why or "checkable:false with no `why`  <- say what on disk is missing",
                    # This assertion is about the ROW's declaration, which exists,
                    # not about the runs the row cannot reach.
                    n=1,
                )
            )
            if why:
                declared_uncheckable.append(f"row {i} ({tried})")
            continue

        err = _validate(guard)
        findings.append(
            Finding(
                f"nulls row {i} release condition is evaluable",
                err is None,
                err or (
                    f"scope {guard['scope']['kind']}, "
                    f"{len(guard['release_all_of'])} release clause(s)"
                ),
                n=1,   # this row
            )
        )
        if err is None:
            checkable.append((i, row, guard))

    # Assertion 3 -- the one that would have caught the 10.7 GPU-hours.
    #
    # WHY A HARVESTED RUN IS REPORTED AND NOT FAILED. hound_pd_desert_s1 really
    # did re-enter row 2's dead end, and saying so is the point of this guard.
    # But it is finished, its hours are spent, and its ledger row is written --
    # so failing on it would make this guard permanently red for a fact that can
    # never be fixed. This repo already knows where that leads: a guard that
    # demands the impossible trains people to bypass the guard. The ledger is
    # the boundary. A run with a ledger row is history: named, counted, not
    # failed. A run WITHOUT one is live, crashed, or about to be resumed -- and
    # that is exactly the run still capable of burning GPU-hours on a dead end,
    # so that one fails.
    harvested = _harvested_runs()

    for i, row, guard in checkable:
        live_breaches: list[str] = []
        past_breaches: list[str] = []
        unchecked: list[str] = []
        in_scope = 0

        for name, config in configs:
            if not _in_scope(guard["scope"], config):
                continue
            in_scope += 1
            if not _terms(config):
                # Predates the reward spec. Naming it is the honest move; a
                # silent skip would report coverage this guard does not have.
                unchecked.append(name)
                continue
            unmet: list[str] = []
            for clause in guard["release_all_of"]:
                holds, evidence = _clause_holds(clause, config)
                if not holds:
                    unmet.append(evidence)
            if unmet:
                (past_breaches if name in harvested else live_breaches).append(
                    f"{name}: " + "; ".join(unmet)
                )

        detail = f"{in_scope} run(s) in scope"
        if live_breaches:
            detail += (
                f"; {len(live_breaches)} UNHARVESTED run(s) re-entered it: "
                + " | ".join(live_breaches)
                + f"  <- do_not_repeat_unless: {row.get('do_not_repeat_unless', '')!r}"
            )
        if past_breaches:
            detail += (
                f"; {len(past_breaches)} already-harvested run(s) re-entered it "
                f"(history, not a fail): " + " | ".join(past_breaches)
            )
        if unchecked:
            detail += f"; {len(unchecked)} predate the reward spec: {sorted(unchecked)}"

        findings.append(
            Finding(
                f"nulls row {i}: no unharvested run re-enters this dead end",
                not live_breaches,
                detail,
                # The CHECKABLE set only: runs in scope minus the ones that
                # predate the reward spec and so were named, not evaluated. A
                # pass over zero evaluable runs plus any number of unreachable
                # ones verified nothing, however honestly `detail` says so.
                n=in_scope - len(unchecked),
            )
        )

    # Named, never silent. A guard that hides what it cannot check reports
    # coverage it does not have.
    findings.append(
        Finding(
            "dead ends that cannot be machine-checked are named",
            True,
            f"{len(declared_uncheckable)} of {len(rows)} row(s) declared uncheckable: "
            f"{declared_uncheckable}" if declared_uncheckable
            else "every recorded dead end is machine-checkable",
            n=len(rows),   # the set being partitioned into named/checkable
        )
    )
    findings.append(
        Finding(
            "runs carrying a config.json are visible to this guard",
            True,
            f"{len(configs)} run(s) with config.json; runs predating it cannot be "
            f"checked against any dead end",
            n=len(configs),
        )
    )
    return findings
