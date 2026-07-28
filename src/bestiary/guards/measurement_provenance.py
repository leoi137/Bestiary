"""Guard: a committed measurement must still name the bytes it was computed from.

Enforces `research/learnings/013` and `research/anomalies.jsonl` rows 19, 20,
23 and 27.

WHAT WENT WRONG, TWICE

`ant_sac_best.zip` is rewritten in place every time the training callback sees a
better eval. Measurements were taken off that filename.

- 2026-07-27, row 19: a refutation re-measured a *different* checkpoint under
  the same filename, and only noticed because someone happened to have the sha.
- 2026-07-28, rows 20/23/27: a published x1.47 ratio turned out to be a property
  of one vanished checkpoint's crash rate. The JSON was committed at 10:58:13;
  the checkpoint it named was overwritten at 11:07:01, nine minutes later. Three
  cycles reasoned from that number before it was caught.

Cycle 007 recorded the defect and recorded why the obvious fix does not work:
hashing before and after "proves the artifact was stable DURING the measurement,
not that it is retrievable afterwards". So `record/freeze.py` copies the
checkpoint to a content-addressed path first and measures the copy, and this
guard is what stops that from quietly rotting.

WHAT THIS GUARD ASSERTS, AND WHY IT IS SHAPED THIS WAY

Three assertions, and the shape of each is deliberate:

1. **Recorded hashes still hold.** For every measurement JSON carrying a full
   64-hex `checkpoint_sha256`, if the frozen file it names is present on this
   machine, that file must still hash to the recorded value. If it is *absent*,
   that is reported and passes: the frozen copies are gitignored, so a fresh
   clone legitimately has none. Absence cannot be a failure without making the
   guard fail for everyone who clones the repo -- but a present-and-different
   file is the exact corruption this exists to catch, and that fails loudly.

2. **New measurements record identity.** Every measurement JSON with a
   `checkpoint` field must also carry `checkpoint_sha256`, EXCEPT the nine
   grandfathered files listed below. The grandfather list is explicit and
   committed rather than a date cutoff, for two reasons: file mtimes do not
   survive a clone, so a cutoff would silently stop grandfathering; and an
   explicit list is a visible debt that shrinks, where a cutoff is invisible
   debt that never does. Adding a name to this list is how the guard gets
   defeated, so the list must only ever get shorter.

3. **Truncated hashes are consistent.** One pre-existing JSON records a
   16-char `checkpoint_sha256_16` (an ad-hoc habit that predates the standard
   field). Where both are present they must agree on the prefix -- otherwise
   two fields would name two different artifacts in one file.

WHAT THIS GUARD DOES NOT ASSERT

That a measurement is *correct*. It pins the weights, not the metric
(`anomalies` row 28: instruments that recompute a reward term drift from what
the env paid), not the selection (`learnings/008`, `010`: `*_best.zip` is an
argmax over noisy evals), and not the protocol (`anomalies` row 18: a published
1078.2 nobody can reproduce because the seeds were never written down).
Reproducible and right are different properties, and this one only buys the
first.
"""

from __future__ import annotations

import json
from pathlib import Path

from bestiary import paths
from bestiary.guards import Finding
from bestiary.record.freeze import sha256_file

# Measurement JSONs written before freeze.py existed (2026-07-28). Their
# checkpoints are, in several cases, permanently gone -- which is the whole
# point. This list may shrink and must never grow.
_GRANDFATHERED = frozenset({
    "heading_ceiling_s0.json",
    "hound_track_desert_s0_deterministic_vs_stochastic.json",
    "hound_track_desert_s0_final_best.json",
    "hound_track_desert_s0_final_decomposition.json",
    "hound_track_desert_s0_final_sac.json",
    "hound_track_desert_s0_midrun_950k.json",
    "track_length_bias_s0.json",
    "track_length_bias_s0_best.json",
    "track_length_bias_s0_seed5000.json",
})

_MEASUREMENTS = paths.RESEARCH / "measurements"


def _measurement_jsons() -> list[Path]:
    if not _MEASUREMENTS.is_dir():
        return []
    return sorted(_MEASUREMENTS.glob("*.json"))


def run() -> list[Finding]:
    out: list[Finding] = []
    files = _measurement_jsons()

    parsed: list[tuple[Path, dict]] = []
    bad: list[str] = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            bad.append(f"{p.name}: {type(exc).__name__} {exc}")
            continue
        if isinstance(d, dict):
            parsed.append((p, d))

    detail = f"{len(parsed)} of {len(files)} parsed as JSON objects"
    if bad:
        detail += f"; UNPARSEABLE: {'; '.join(bad)}"
    out.append(Finding("every measurement JSON parses", not bad, detail))

    # ---- 1. recorded hashes still hold -------------------------------------
    verified, missing, mismatched = [], [], []
    for p, d in parsed:
        digest = d.get("checkpoint_sha256")
        if not digest:
            continue
        frozen = d.get("checkpoint_frozen")
        if not frozen:
            mismatched.append(f"{p.name}: records checkpoint_sha256 but no checkpoint_frozen path")
            continue
        fp = Path(frozen)
        if not fp.is_absolute():
            fp = paths.REPO_ROOT / fp
        if not fp.exists():
            missing.append(f"{p.name} -> {frozen}")
            continue
        actual = sha256_file(fp)
        if actual == digest:
            verified.append(p.name)
        else:
            mismatched.append(
                f"{p.name}: {frozen} now hashes to {actual}, JSON recorded {digest}"
            )

    out.append(Finding(
        "every frozen checkpoint present on this machine still hashes to what its measurement recorded",
        not mismatched,
        f"{len(verified)} verified, {len(missing)} not present locally (gitignored copies; "
        f"absence is not a failure), {len(mismatched)} MISMATCHED"
        + (f" -- {'; '.join(mismatched)}" if mismatched else "")
        + (f". Not present: {'; '.join(missing)}" if missing else ""),
    ))

    # ---- 2. new measurements record identity -------------------------------
    delinquent = [
        p.name for p, d in parsed
        if "checkpoint" in d
        and not d.get("checkpoint_sha256")
        and p.name not in _GRANDFATHERED
    ]
    out.append(Finding(
        "every non-grandfathered measurement naming a checkpoint records its sha256",
        not delinquent,
        f"{len([1 for _, d in parsed if 'checkpoint' in d])} JSON(s) name a checkpoint; "
        f"{len(_GRANDFATHERED)} grandfathered (pre-2026-07-28, several of their artifacts "
        f"are permanently gone); {len(delinquent)} delinquent"
        + (f": {', '.join(delinquent)}" if delinquent else ""),
    ))

    # ---- 2b. the grandfather list is debt, and must only shrink -------------
    stale = sorted(_GRANDFATHERED - {p.name for p, _ in parsed})
    out.append(Finding(
        "the grandfather list names only files that exist",
        not stale,
        f"{len(_GRANDFATHERED)} grandfathered, all present"
        if not stale
        else f"{len(stale)} listed but absent -- remove them so the list keeps shrinking: {', '.join(stale)}",
    ))

    # ---- 3. truncated hashes agree with full ones --------------------------
    conflicts = [
        f"{p.name}: sha256_16={d['checkpoint_sha256_16']} is not a prefix of {d['checkpoint_sha256']}"
        for p, d in parsed
        if d.get("checkpoint_sha256_16") and d.get("checkpoint_sha256")
        and not d["checkpoint_sha256"].startswith(d["checkpoint_sha256_16"])
    ]
    both = [p.name for p, d in parsed
            if d.get("checkpoint_sha256_16") and d.get("checkpoint_sha256")]
    out.append(Finding(
        "a truncated checkpoint hash agrees with the full one in the same file",
        not conflicts,
        f"{len(both)} file(s) carry both fields"
        + (f"; CONFLICT: {'; '.join(conflicts)}" if conflicts else "; no conflicts"),
    ))

    return out
