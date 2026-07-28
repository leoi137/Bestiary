"""Guard: a published reward decomposition must account for the whole return.

Enforces `research/anomalies.jsonl` row 39 (2026-07-28).

`record/track_eval.py` decomposed a tracking run's return into a HARDCODED
four-term tuple. `HoundPDTrackRelDesert-v0` pays five -- it gained
`reward_shaping` when the PBRS term landed -- so every `drive_grid_reward_*`
field it published was a partial account of the return, presented as a complete
one, with nothing in the output saying so. Ledger row 4's entire published
conclusion was a decomposition argument ("ctrl_cost is 102.9% of the gap to
zero action"), which is why this is a claim-shaped defect and not a cosmetic
one.

The fix replaced the constant with discovery from the env's own step info plus
a per-step assertion that the terms SUM to the reward. This guard is the thing
that catches the fix being reverted, which the fix itself is not: an
independent refutation of cycle 014 pointed out that `discover_terms` and
`assert_decomposition_complete` were exercised only by a research script
nothing runs automatically, so the suite would have stayed green with the bug
reinstated. That is the repo's engineering standard unmet -- *a change is not
done until the thing that would catch its regression exists* -- and this file
is the debt being paid.

WHAT IT ASSERTS, IN TWO HALVES

**The mechanism**, on synthetic inputs whose right answers are known by
construction, calling the real functions rather than reimplementations of them
(`tracking-frame`'s lesson about constants, which holds for formulas too):

    1. the completeness check FIRES on a term list missing a term
    2. it PASSES on the complete one -- an always-red check bounds nothing
    3. discovery finds every `reward_*` key, in declaration order
    4. discovery REFUSES an env that reports no reward terms at all
    5. aggregation REFUSES episodes that disagree about their term list

**The committed record**, which is the half that would have caught the
original defect where it actually lived:

    6. every measurement JSON's per-term means sum to its own drive-grid
       aggregate

Assertion 6 works from committed JSON alone -- no env, no torch, no
checkpoint -- because both sides are means over the same drive cells, and the
mean is linear. It closes to 0.0 exactly on the four-term env's files.

THE GRANDFATHER LIST IS THE POINT, NOT AN ESCAPE HATCH

Three committed measurements were produced BY the bug and cannot be corrected
without re-running a full grid eval against `hound_track_rel_s1`'s
checkpoints. Failing on them would leave `guards --fast` red forever, and
`guards --fast` gates every training launch -- a permanently red launch gate
gets bypassed, which is worse than the defect.

So they are named here with their measured residual, and the guard PRINTS that
residual on every run. The omission becomes disclosed-in-code rather than
silent, which is the property the original failure lacked. Regenerating them is
what removes a name from this list; nothing else does, and a new measurement
cannot be added to it.

Fast tier: pure functions and committed JSON. No env is built, no physics is
stepped, no checkpoint is opened.
"""
from __future__ import annotations

import json

from bestiary import paths
from bestiary.guards import Finding
from bestiary.record.track_eval import (
    assert_decomposition_complete,
    discover_terms,
)
from bestiary.record import track_eval

# A synthetic step whose reward is the sum of its five terms by construction.
# The numbers are arbitrary; what matters is that no two subsets sum alike.
_INFO = {
    "reward_track": 0.90,
    "reward_shaping": -0.02,
    "reward_ctrl": -0.10,
    "reward_contact": -0.03,
    "reward_termination": 0.00,
    # Deliberate non-reward neighbours: discovery must not pick these up.
    "track_phi_v": 0.5,
    "potential": 0.25,
    "achieved_vx": 0.27,
}
_REWARD = 0.75          # 0.90 - 0.02 - 0.10 - 0.03 + 0.00

# The tuple that was hardcoded in `record/track_eval.py` until 2026-07-28.
OLD_HARDCODED_TERMS = (
    "reward_track", "reward_ctrl", "reward_contact", "reward_termination",
)

# Measurements produced BY the defect. Each was written before 2026-07-28's fix
# against HoundPDTrackRelDesert-v0, whose fifth term the instrument could not
# see. Removing a name requires REGENERATING the file, not editing this list.
GRANDFATHERED = {
    "track_rel_s1_best.json",
    "track_rel_s1_latest.json",
    "track_rel_zero_action.json",
}

# Absolute slack on assertion 6. Both sides are means over the same cells, so
# they agree exactly up to JSON round-trip; the four-term files close to 0.0.
JSON_TOL = 1e-6

TERM_PREFIX = "drive_grid_reward_"
# The aggregate the per-term means must sum to. Two spellings exist: track_eval
# writes `drive_grid_mean`, track_return_decomposition writes
# `drive_grid_return`. Both are the mean return over the same drive cells.
AGG_KEYS = ("drive_grid_mean", "drive_grid_return")


def _blocks(obj, path=""):
    """Every nested dict carrying a decomposition, with its path for the report."""
    if isinstance(obj, dict):
        if any(k.startswith(TERM_PREFIX) for k in obj):
            yield path or "<root>", obj
        for k, v in obj.items():
            yield from _blocks(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _blocks(v, f"{path}[{i}]")


def run() -> list[Finding]:
    findings: list[Finding] = []

    # --- half one: the mechanism ------------------------------------------
    full = discover_terms(_INFO)

    # 1 + 2. The check must fire on every proper subset that changes the sum,
    #        and pass on the complete list. Both, or it bounds nothing.
    truncations = [tuple(t for t in full if t != drop)
                   for drop in full if _INFO[drop] != 0.0]
    fired = 0
    for trunc in truncations:
        try:
            assert_decomposition_complete(trunc, _INFO, _REWARD, 0)
        except ValueError:
            fired += 1
    passes_complete = True
    try:
        assert_decomposition_complete(full, _INFO, _REWARD, 0)
    except ValueError:
        passes_complete = False
    findings.append(Finding(
        "the completeness check fires on an incomplete term list and passes "
        "on the complete one",
        fired == len(truncations) and passes_complete,
        f"{fired} of {len(truncations)} term-droppings raised (every dropped "
        f"term here is non-zero, so every one changes the sum); the complete "
        f"{len(full)}-term list passes: {passes_complete}. Dropping "
        f"reward_shaping specifically is the exact pre-2026-07-28 behaviour, "
        f"and it must raise",
        n=len(truncations) + 1,
    ))

    # 3. Discovery finds the reward terms and only the reward terms, in the
    #    order the env declares them -- the order the decomposition prints in.
    expected = tuple(k for k in _INFO if k.startswith("reward_"))
    findings.append(Finding(
        "discovery returns every reward term, in declaration order, and "
        "nothing else",
        full == expected,
        f"discovered {list(full)}; expected {list(expected)}. The non-reward "
        f"neighbours in the same info dict "
        f"{[k for k in _INFO if not k.startswith('reward_')]} must NOT appear, "
        f"and the order must match: it is what makes the four-term env's sums "
        f"bit-identical to the old hardcoded tuple's",
        n=len(_INFO),
    ))

    # 4. An env reporting no reward terms is refused, not decomposed into
    #    nothing and reported as complete.
    try:
        discover_terms({"track_phi_v": 0.5})
        refused = False
    except ValueError:
        refused = True
    findings.append(Finding(
        "an env that reports no reward terms is refused rather than decomposed "
        "into an empty account",
        refused,
        f"discover_terms({{'track_phi_v': 0.5}}) raised: {refused}. An empty "
        f"term list sums to 0.0, which would equal the reward only by accident "
        f"and would report a decomposition of nothing as complete",
        n=1,
    ))

    # 5. Aggregation refuses to average unlike decompositions under one name.
    def _ep(terms):
        return {"return": 0.0, "steps": 1000, "horizon": 1000,
                "terminated": False, "mean_phi_v": 0.3, "mean_phi_w": 1.0,
                "mean_track": 0.3, "track_income": 300.0, "achieved_vx": 0.0,
                "commanded_vx": 0.0, "terms": list(terms),
                **dict.fromkeys(terms, 0.0)}
    try:
        track_eval.aggregate_cell(
            (0.5, 0.0, 0.0), [_ep(full), _ep(OLD_HARDCODED_TERMS)])
        mixed_refused = False
    except ValueError:
        mixed_refused = True
    findings.append(Finding(
        "aggregation refuses a cell whose episodes disagree about their reward "
        "terms",
        mixed_refused,
        f"aggregate_cell over one 5-term and one 4-term episode raised: "
        f"{mixed_refused}. Averaging them would report a per-term mean over "
        f"episodes that do not share the term, under a single field name",
        n=2,
    ))

    # --- half two: the committed record ------------------------------------
    checked, delinquent, grandfathered_seen = 0, [], []
    measurements = sorted((paths.RESEARCH / "measurements").glob("*.json"))
    for path in measurements:
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue                      # measurement-provenance owns parseability
        for where, block in _blocks(doc):
            terms = [k for k in block if k.startswith(TERM_PREFIX)]
            agg = next((block[k] for k in AGG_KEYS if k in block), None)
            if not terms or agg is None:
                continue
            residual = sum(float(block[t]) for t in terms) - float(agg)
            if path.name in GRANDFATHERED:
                grandfathered_seen.append(f"{path.name}:{where} {residual:+.6f}")
                continue
            checked += 1
            if abs(residual) > JSON_TOL:
                delinquent.append(
                    f"{path.name}:{where} sums to {sum(float(block[t]) for t in terms):.6f} "
                    f"against {agg:.6f} (residual {residual:+.6f}, "
                    f"{len(terms)} terms)")

    findings.append(Finding(
        "every non-grandfathered measurement's per-term means sum to its own "
        "drive-grid aggregate",
        not delinquent,
        f"{checked} decomposition block(s) checked across {len(measurements)} "
        f"measurement file(s), {len(delinquent)} delinquent"
        + (f": {delinquent}" if delinquent else "")
        + f". {len(grandfathered_seen)} block(s) grandfathered (written by the "
        f"defect, correctable only by re-running the grid): "
        f"{grandfathered_seen or 'none'}. The grandfathered residuals are the "
        f"size of the reward_shaping term the old instrument could not see, "
        f"and they are printed rather than hidden",
        n=checked,
    ))

    return findings
