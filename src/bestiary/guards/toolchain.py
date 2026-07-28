"""Guard: the interpreter running this is the repo's own venv.

The failure this fences is recorded in `research/anomalies.jsonl`. This venv was
created as `GymMuJoCo/venv` and moved here, so `source venv/bin/activate` still
exports `VIRTUAL_ENV` pointing at a directory **outside both repositories** and
prepends its `bin` to `PATH`. After sourcing it there is no `python` on `PATH`
at all — so the documented launch gate

    python -m bestiary.guards --fast || exit 1

exits **127**, which is a nonzero status, which reads exactly like a guard
failure. The gate reports a problem it never checked. Worse, if that stale
directory is ever recreated, the same line silently trains against a different
interpreter with different pinned package versions, and nothing anywhere says
so — the run just produces numbers that are not comparable to any other row in
the ledger.

Every other guard in this suite assumes it is running inside the environment
the training runs use. That assumption was never checked, which made it the one
lesson the suite could not enforce about itself.

These are scalar facts about one process, not quantifiers over a set, so they
leave `n` unset rather than claim an input-set size.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import bestiary
from bestiary import paths
from bestiary.guards import Finding

VENV = paths.REPO_ROOT / "venv"
EXPECTED_PYTHON = VENV / "bin" / "python"


def _resolve(p: str | os.PathLike) -> Path:
    return Path(p).resolve()


def run() -> list[Finding]:
    findings: list[Finding] = []

    # 1. The interpreter itself. `venv/bin/python` is a symlink to a system
    # python, so both sides are resolved and compared by the venv directory
    # they sit under rather than by filename.
    actual = _resolve(sys.executable)
    expected_prefix = _resolve(VENV)
    inside = actual == _resolve(EXPECTED_PYTHON) or expected_prefix in actual.parents
    findings.append(
        Finding(
            "the running interpreter is the repo's venv",
            inside or _resolve(sys.prefix) == expected_prefix,
            f"sys.executable={sys.executable}\n"
            f"         sys.prefix={sys.prefix}\n"
            f"         expected a python under {VENV}\n"
            "         call venv/bin/python directly; never `source venv/bin/activate`",
        )
    )

    # 2. The stale VIRTUAL_ENV. Unset is correct and expected — calling the
    # interpreter directly never sets it. Set-and-wrong is the activate bug.
    # An empty string is what `VIRTUAL_ENV=` sets, and it means "no venv" just
    # as much as absence does. Treating it as a path resolves it to the cwd and
    # reports a false failure that varies with where you stood.
    venv_env = os.environ.get("VIRTUAL_ENV") or None
    if venv_env is None:
        ok, detail = True, "VIRTUAL_ENV unset (interpreter called directly — correct)"
    else:
        ok = _resolve(venv_env) == expected_prefix
        detail = (
            f"VIRTUAL_ENV={venv_env}\n"
            f"         expected {VENV} or unset"
        )
        if not ok:
            detail += (
                "\n         this is the moved-venv bug: activate exports the path the "
                "venv was CREATED at,\n         which is outside both repositories, and "
                "leaves no `python` on PATH"
            )
    findings.append(Finding("VIRTUAL_ENV does not point outside the repo", ok, detail))

    # 3. The package being imported is this working copy, not a stray copy in
    # some site-packages. Installed with `pip install -e .`, so it must resolve
    # into src/. A non-editable copy would make every guard read code that is
    # not the code under test.
    pkg = _resolve(Path(bestiary.__file__).parent)
    want = _resolve(paths.REPO_ROOT / "src" / "bestiary")
    findings.append(
        Finding(
            "bestiary imports from this working copy",
            pkg == want,
            f"imported from {pkg}\n         expected {want}\n"
            "         a non-editable install means the guards check code that is "
            "not the code you edited",
        )
    )

    return findings
