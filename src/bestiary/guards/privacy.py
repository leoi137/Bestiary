"""Guard: nothing private has reached the public repository.

This is the only guard defending an **irreversible** failure. Every other
mistake here is recoverable — a bad experiment wastes GPU-hours, a wrong
learning gets superseded, a crashed run resumes. A private fact pushed to a
public repository is cloned and indexed within minutes and cannot be recalled.

The boundary, from the workspace rules: physics, code, robots, results,
lessons and teaching notes are public. Anything naming an agency, a customer,
or a dollar figure is private. So is strategy, and so is how the work is
driven.

Scans tracked text under this repository. Findings are `FAIL` rather than
warnings on purpose: a privacy check that reports advisories becomes a check
nobody reads, and this is the one that must never become wallpaper.

**Allowlisting is deliberate and costly.** Every entry names the file, the
pattern, and why that specific occurrence is acceptable. An allowlist entry
without a reason is not permitted — that is how a real leak gets waved
through by a future reader who assumes someone checked.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from bestiary import paths
from bestiary.guards import Finding

# Only text is scanned; binaries and generated assets carry no prose.
TEXT_SUFFIXES = frozenset({".md", ".py", ".txt", ".json", ".jsonl", ".toml", ".xml", ".yml", ".yaml"})

SKIP_DIRS = frozenset({".git", "venv", "runs", "assets", ".ruff_cache", "__pycache__"})

# (label, pattern). Deliberately broad — a false positive costs one allowlist
# line with a reason; a false negative is permanent.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Bare "USD" is deliberately NOT matched: in this domain it means Universal
    # Scene Description far more often than it means dollars, and a pattern that
    # fires on every Isaac Sim discussion trains the reader to skim past this
    # guard. Money is matched by its shape instead.
    ("dollar figure", re.compile(r"\$\s?\d|\b\d[\d,.]*\s*(?:USD|dollars|euros)\b", re.I)),
    ("agency or program", re.compile(
        r"\b(NASA|xTech|SBIR|STTR|DARPA|AFWERX|AFRL|DoD|ONR|ARPA-?E|ESA|JAXA)\b")),
    ("customer or award", re.compile(
        r"\b(our client|the client|our customer|the customer|contract award|"
        r"grant award|award amount|purchase order|invoice|proposal deadline)\b", re.I)),
    ("private repo content", re.compile(
        r"Scriptorium/(?:strategy|loop|funding|proposals|hardware|scratch)\b")),
    # "cost us" is deliberately absent — "caught before it cost us anything" is
    # an idiom, not a disclosure, and the dollar-figure pattern already catches
    # any version of it that names an actual amount.
    ("cost or pricing claim", re.compile(
        r"\b(we paid|budget of|priced at|quoted at|invoiced|per seat)\b", re.I)),
)

# path -> {pattern label: reason it is acceptable there}
ALLOW: dict[str, dict[str, str]] = {
    "src/bestiary/guards/privacy.py": {
        "agency or program": "this file defines the patterns it searches for",
        "customer or award": "this file defines the patterns it searches for",
        "dollar figure": "this file defines the patterns it searches for",
        "private repo content": "this file defines the patterns it searches for",
        "cost or pricing claim": "this file defines the patterns it searches for",
    },
}


def _tracked_files() -> list[Path]:
    """Files git actually tracks — untracked scratch is not published."""
    out = subprocess.run(
        ["git", "-C", str(paths.REPO_ROOT), "ls-files"],
        capture_output=True, text=True, check=True,
    )
    files = []
    for line in out.stdout.splitlines():
        rel = Path(line)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        if rel.suffix.lower() in TEXT_SUFFIXES:
            files.append(rel)
    return files


def scan() -> list[tuple[str, str, int, str]]:
    """(relative path, pattern label, line number, the offending line)."""
    hits: list[tuple[str, str, int, str]] = []
    for rel in _tracked_files():
        allowed = ALLOW.get(str(rel), {})
        try:
            text = (paths.REPO_ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if label in allowed:
                    continue
                if pattern.search(line):
                    hits.append((str(rel), label, lineno, line.strip()[:120]))
    return hits


def run() -> list[Finding]:
    try:
        hits = scan()
    except subprocess.CalledProcessError as exc:
        return [Finding("privacy scan runs", False, f"git ls-files failed: {exc}")]

    if not hits:
        return [
            Finding(
                "no private content in the public repository",
                True,
                f"{len(_tracked_files())} tracked text files scanned, "
                f"{len(PATTERNS)} patterns",
            )
        ]

    return [
        Finding(
            f"{path}:{lineno} — {label}",
            False,
            f"{line}\n         allowlist it with a reason, or move it to Scriptorium",
        )
        for path, label, lineno, line in hits
    ]
