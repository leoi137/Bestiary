"""Guard: a score must say how many episodes it came from.

Enforces `research/learnings/008`.

`VideoEvalCallback` rolls exactly ONE episode per evaluation, logs its return as
`eval/mean_reward` — a name asserting a sample size the code does not take — and
saves `ant_sac_best.zip` whenever that single draw beats the stored record. So
`best_eval_return` is a maximum over single-episode draws, and `*_best.zip` is
the snapshot that got the luckiest one.

That is harmless when evaluation is low-variance and fatal when it is not. These
policies are bimodal: they complete the episode or fail early, at measured rates
of 26.7% and 10.0%. At 14 evaluations the probability that argmax selects a
good-mode snapshot is 1 - 0.267**14, i.e. one. The saved checkpoint reports the
policy's best mode and never its mixture, however often it actually fails.

Two assertions, and neither can be satisfied by editing prose:

1. **A ledger row carrying `best_eval_return` says what n produced it.** Without
   that, the number cannot be compared against anything — it grows with the
   number of evaluations, so a longer run scores higher for free.
2. **That n is greater than 1.** A one-episode maximum is not a measurement.

Runs predating this are named rather than skipped, the same way
`checkpoint-width` names the runs predating the observation spec. A silent skip
is how a guard reports coverage it does not have.
"""
from __future__ import annotations

import json

from bestiary import paths
from bestiary.guards import Finding

# Below this, a crash rate is not a rate. At a 26.7% failure rate five episodes
# come back clean 21% of the time (research/scripts/learning_008_math.py), which
# is how a policy that fails one run in four gets written up as flawless.
MIN_EVAL_EPISODES = 20

# The field a row should carry alongside best_eval_return.
N_FIELD = "best_eval_episodes"


def _rows() -> list[dict]:
    if not paths.LEDGER.exists():
        return []
    return [
        json.loads(line)
        for line in paths.LEDGER.read_text().splitlines()
        if line.strip()
    ]


def run() -> list[Finding]:
    rows = _rows()
    if not rows:
        return [Finding("ledger has rows to check", True, "empty ledger — nothing to do")]

    findings: list[Finding] = []
    unpinned: list[str] = []

    for row in rows:
        name = str(row.get("run", "<unnamed>"))
        if "best_eval_return" not in row:
            continue

        n = row.get(N_FIELD)
        if n is None:
            unpinned.append(name)
            continue

        findings.append(
            Finding(
                f"{name}: best_eval_return says how many episodes produced it",
                isinstance(n, int) and n >= MIN_EVAL_EPISODES,
                f"{N_FIELD}={n!r}, need an int >= {MIN_EVAL_EPISODES}"
                + ("" if isinstance(n, int) and n >= MIN_EVAL_EPISODES else
                   "  <- a maximum over few draws is not a policy score (learnings/008)"),
            )
        )

    # Reported, never silent, and never a FAIL: these runs are finished, and
    # inventing an episode count for them would be the false provenance the
    # number rule exists to prevent. The count should fall to zero as new rows
    # land under this guard rather than quietly sit here.
    findings.append(
        Finding(
            "every ledger row with a best score records its sample size",
            True,
            f"{len(unpinned)} row(s) predate this guard and carry a "
            f"best_eval_return of unknown n: {sorted(unpinned)}"
            if unpinned else "all rows pinned",
        )
    )
    return findings
