"""How much does `measurement-provenance` actually verify on this repo?

The arithmetic behind `research/learnings/014`. The guard is green; this asks
what being green is worth today, which is a different question and turned out
to have a very different answer.

    venv/bin/python research/scripts/measurement_provenance_coverage.py

Read-only. Touches no run, no GPU, no checkpoint.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bestiary import paths  # noqa: E402
from bestiary.guards import measurement_provenance as mp  # noqa: E402

def _load_call_sites(root: Path) -> list[tuple[Path, int]]:
    """Real `SAC.load(...)` CALLS, found by parsing rather than grepping.

    A grep for `SAC.load(` returns 13 hits here and 6 of them are prose: this
    repo's docstrings discuss `SAC.load()` constantly, because "an observation
    change makes SAC.load() raise" is one of its load-bearing invariants. A
    text match cannot tell an invariant being explained from one being used,
    and publishing 2/13 when the truth is 2/7 would be exactly the kind of
    unchecked arithmetic the number rule exists to stop.
    """
    out: list[tuple[Path, int]] = []
    for py in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "SAC"):
                out.append((py, node.lineno))
    return out


# A load that can turn weights into a number someone quotes in the record.
# train.py loads to RESUME and watch.py loads to RENDER; neither publishes a
# figure, so neither is in scope for provenance.
_PUBLISHES = ("record/", "research/scripts/")


def main() -> int:
    meas = sorted((paths.RESEARCH / "measurements").glob("*.json"))
    docs = []
    for p in meas:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            docs.append((p, d))

    naming = [(p, d) for p, d in docs if mp._blocks(d)]
    with_hash = [
        (p, d) for p, d in naming
        if any(b.get("checkpoint_sha256") for b in mp._blocks(d))
    ]

    findings = mp.run()
    verified = 0
    for f in findings:
        m = re.match(r"(\d+) verified", f.detail)
        if m:
            verified = int(m.group(1))

    # Assertions whose input set is empty are green for free.
    empty = [f.label for f in findings if re.search(r"\b0 (verified|file\(s\))", f.detail)]

    calls = _load_call_sites(paths.REPO_ROOT / "src") + _load_call_sites(paths.RESEARCH / "scripts")
    load_sites = [f"{p.relative_to(paths.REPO_ROOT).as_posix()}:{n}" for p, n in calls]
    publishing = [s for s in load_sites if any(k in s for k in _PUBLISHES)]
    frozen_sites = [
        s for s in publishing
        if "record/track_eval.py" in s or "record/greedy_eval.py" in s
    ]

    ledger = paths.RESEARCH / "ledger.jsonl"
    ledger_rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    ledger_hashed = [r for r in ledger_rows if r.get("eval_crash_rate_checkpoint_sha256")]

    git = subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    print(f"measurement-provenance coverage at {git}\n")
    print(f"  guard verdict                         {'ALL PASS' if all(f.ok for f in findings) else 'FAIL'}")
    print(f"  assertions                            {sum(f.ok for f in findings)}/{len(findings)} pass")
    print(f"  assertions green on an EMPTY input    {len(empty)}/{len(findings)}")
    for lab in empty:
        print(f"      - {lab[:72]}")
    print()
    print(f"  measurement JSONs (parseable)         {len(docs)}")
    print(f"    ...naming a checkpoint              {len(naming)}")
    print(f"    ...recording a sha256               {len(with_hash)}")
    print(f"    ...naming one but recording none    {len(naming) - len(with_hash)}")
    print(f"  checkpoint hashes ACTUALLY verified   {verified}")
    print()
    print(f"  real SAC.load CALLS (ast, not grep)   {len(load_sites)}")
    print(f"    ...that can publish a number        {len(publishing)}")
    print(f"    ...of those, freezing first         {len(frozen_sites)}")
    for s in load_sites:
        if s in frozen_sites:
            tag = "[frozen] "
        elif s in publishing:
            tag = "[MUTABLE]"
        else:
            tag = "[resume/render, out of scope]"
        print(f"      {tag} {s}")
    print()
    print(f"  ledger rows                           {len(ledger_rows)}")
    print(f"    ...carrying a checkpoint sha256     {len(ledger_hashed)}")
    print()
    pct = 100.0 * len(frozen_sites) / len(publishing) if publishing else 0.0
    print(f"  => the guard is green having verified {verified} artifact(s), and the freeze "
          f"covers {len(frozen_sites)}/{len(publishing)} = {pct:.0f}% of the load sites that can publish a number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
