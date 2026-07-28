"""Guard: enough system RAM is free to start a run, and this loop is capped.

Budget condition 8. The other seven conditions budget VRAM, disk and
wall-clock; before 2026-07-27 none of them budgeted system memory, and the
consequence was not theoretical. `systemd-oomd` killed a 21.9 GB process on
this 31 GiB machine at 03:38 and took the operator's editor with it, while the
loop was fully compliant with every rule it had.

Two properties make this the check that matters most unattended:

*The kill lands somewhere else.* `oomd` selects the most memory-pressured
cgroup and the kernel then kills the largest task in it — not the greediest
process on the machine. So the process that dies is usually not the one at
fault, and the cause is invisible from inside the thing that caused it. Nothing
in a training log would ever show why.

*This is the only reading that senses the other tenant.* The machine hosts a
second autonomous loop, which is out of scope to inspect and whose unit carries
no memory limit at all. Its pressure is nonetheless visible in `MemAvailable`.
Checking that number is how "the machine is shared" is honoured for RAM, the
same way the 6500 MiB VRAM pre-flight honours it for the card.

Written as a guard rather than as a line in a procedure because the procedure
already said it and the procedure is not what runs. `guards --fast` gates every
training launch (`guards --fast || exit 1`), so a check placed here fires on
every launch path that exists, including the ones nobody remembered to update.
That is the whole difference between a rule and a guard.

`MemAvailable` is the kernel's own estimate of what a new workload can claim
without swapping — reclaimable page cache included. `MemFree` is the wrong
number and would fail constantly on a healthy machine with a warm cache.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from bestiary.guards import Finding

# Scriptorium/SYSTEM.md. Headroom for one cycle (8 GiB cap) to start and still
# leave the desktop room; below this the machine is already carrying something
# large and a new run is what tips it.
FLOOR_MIB = 8000

MEMINFO = Path("/proc/meminfo")

# The transient unit each run is launched into, and the cycle's own unit.
# A cap of `infinity` means a runaway has nothing stopping it, which is the
# state this machine was in when it died.
CYCLE_UNIT = "robotics-cycle.service"
CYCLE_MAX_CEILING_GIB = 12.0  # a cycle capped looser than this is not capped


def _mem_available_mib() -> int | None:
    """Available MiB per the kernel, or None if /proc/meminfo is unreadable."""
    try:
        text = MEMINFO.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", text, re.MULTILINE)
    return int(m.group(1)) // 1024 if m else None


def _cycle_memory_max() -> str | None:
    """MemoryMax of the cycle unit as systemd reports it, or None if absent.

    Absent is not a failure: the loop is often driven by hand from a session
    that is not the systemd unit at all, and on a machine where the unit was
    never installed this guard must not block training.
    """
    if shutil.which("systemctl") is None:
        return None
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", CYCLE_UNIT, "-p", "MemoryMax", "--value"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def run() -> list[Finding]:
    # Every finding here reads one scalar — MemAvailable, or one systemd
    # property — so none of them quantifies over a set and all leave n unset.
    # A size of 0 or 1 here would be an invented collection.
    findings: list[Finding] = []

    avail = _mem_available_mib()
    if avail is None:
        findings.append(
            Finding(
                "system RAM is measurable",
                False,
                f"cannot read {MEMINFO}; budget condition 8 cannot be checked, so "
                "do not launch — an unverifiable ceiling is not a satisfied one",
            )
        )
        return findings

    findings.append(
        Finding(
            f"≥{FLOOR_MIB} MiB system RAM available",
            avail >= FLOOR_MIB,
            f"MemAvailable = {avail} MiB (floor {FLOOR_MIB} MiB)"
            + (
                ""
                if avail >= FLOOR_MIB
                else "  — the machine is already carrying something large. This is "
                "usually the second autonomous loop, NOT a leak of ours. Do not "
                "launch; block naming budget condition 8 and do non-GPU work."
            ),
        )
    )

    cap = _cycle_memory_max()
    if cap is None:
        findings.append(
            Finding(
                "cycle unit memory cap",
                True,
                "no systemd cycle unit on this machine — nothing to cap "
                "(driven by hand, not by the timer)",
            )
        )
        return findings

    if cap in ("infinity", "max"):
        capped, detail = False, (
            f"{CYCLE_UNIT} has MemoryMax={cap} — a runaway cycle can take the "
            "whole machine, which is exactly what happened on 2026-07-27. "
            "Reinstall the unit and `systemctl --user daemon-reload`."
        )
    else:
        try:
            gib = int(cap) / 1024**3
        except ValueError:
            capped, detail = False, f"{CYCLE_UNIT} MemoryMax unparseable: {cap!r}"
        else:
            capped = gib <= CYCLE_MAX_CEILING_GIB
            detail = (
                f"{CYCLE_UNIT} MemoryMax = {gib:.1f} GiB "
                f"(ceiling {CYCLE_MAX_CEILING_GIB:.0f} GiB)"
            )
            if not capped:
                detail += (
                    "  — a cap set near the size of the process that killed this "
                    "machine is not a cap. The run has its own transient unit; "
                    "the cycle does not need this much."
                )

    findings.append(Finding("the cycle's cgroup is capped", capped, detail))
    return findings
