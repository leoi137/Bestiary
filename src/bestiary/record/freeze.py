"""Freeze a checkpoint to a content-addressed path before measuring it.

Why this exists
---------------
`ant_sac_best.zip` is overwritten in place, mid-run, every time the training
callback sees a better eval. A number measured off that filename names a file
that may no longer exist by the time anyone tries to reproduce it.

This has cost the record twice:

- 2026-07-27 (`anomalies` row 19): a refutation re-measured a *different*
  checkpoint under the same filename and only noticed because the sha had moved.
- 2026-07-28 (`anomalies` rows 20, 23, 27): a published x1.47 ratio turned out
  to be a property of one vanished checkpoint's crash rate. The measurement JSON
  was committed at 10:58:13; the checkpoint it named was overwritten at
  11:07:01, nine minutes later. Three cycles reasoned from the number.

Why hashing is not enough, and the copy is
------------------------------------------
Cycle 007 already tried the obvious fix -- hash before the measurement, hash
after, assert unchanged -- and recorded why it cannot work:

    "it proves the artifact was stable DURING the measurement, not that it is
    retrievable afterwards."

007's own hashes agreed (`42b8ef1d...` both times) and the file was still gone
40 minutes later. So the immutability has to come from **the copy**, not from a
check wrapped around a mutable read. `learnings/013` states the same conclusion:
a number is only as durable as the artifact it was computed from.

The shape
---------
`runs/<name>/measured/<sha256>.zip`, content-addressed, so the path *is* the
hash and a wrong file cannot hide behind a right name. `*.zip` is gitignored, so
these never enter git; what gets committed is the hash inside the measurement
JSON, and `guards/measurement_provenance.py` asserts the two still agree.

Content addressing makes this cheap to repeat: freezing the same bytes twice is
a no-op, so a re-measurement costs nothing and a genuinely new checkpoint costs
5 MB.

What this does NOT fix
----------------------
- **Selection bias.** `learnings/008` and `010`: `*_best.zip` is an argmax over
  noisy single-episode evals. Freezing it makes it *retrievable*, not
  *trustworthy*. Keep quoting it beside `ant_sac.zip`.
- **Instrument drift.** `anomalies` row 28: an instrument that recomputes a
  reward term drifts from what the env actually paid. Pinning the weights does
  not pin the metric.
- **Protocol identity.** `anomalies` row 18: a published 1078.2 nobody can
  reproduce, because the *seeds* were never written down. A hash is not
  reproducibility; the JSON must pin the protocol too. Both call sites already
  record their seed block and `deterministic` flag -- keep it that way.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB; checkpoints are ~5 MB, so this is 5 reads.


def sha256_file(path: Path) -> str:
    """Full 64-hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenCheckpoint:
    """An immutable copy of a checkpoint, and the identity of its bytes."""

    source: Path
    frozen: Path
    sha256: str
    size_bytes: int

    def as_json_fields(self) -> dict:
        """The fields a measurement JSON must carry to stay reproducible.

        `checkpoint` stays a bare filename for backward comparability with the
        nine measurement JSONs written before 2026-07-28; the other three are
        what make the number recomputable.

        `checkpoint_frozen` is written **relative to the repo root**, never
        absolute. These JSONs are committed to a public repository: an absolute
        path would publish the operator's home directory and would not resolve
        on any other machine, including a fresh clone of this one.
        """
        from bestiary import paths

        try:
            frozen = self.frozen.relative_to(paths.REPO_ROOT).as_posix()
        except ValueError:
            # Outside the repo. Record the name only rather than publishing an
            # absolute path; the sha is what identifies the bytes anyway.
            frozen = self.frozen.name

        return {
            "checkpoint": self.source.name,
            "checkpoint_sha256": self.sha256,
            "checkpoint_frozen": frozen,
            "checkpoint_bytes": self.size_bytes,
        }


def freeze_checkpoint(source: Path, *, run_dir: Path | None = None) -> FrozenCheckpoint:
    """Copy `source` to `<run_dir>/measured/<sha256>.zip` and return its identity.

    Load the policy from the returned `.frozen`, never from `source` -- reading
    the mutable path after freezing it reintroduces exactly the race this
    function exists to remove.

    Idempotent: the destination is content-addressed, so freezing unchanged
    bytes twice reuses the existing file and costs one hash.

    Raises rather than guessing, in every failure mode:
    - `source` missing -> FileNotFoundError naming the path
    - source rewritten mid-copy -> RuntimeError naming both hashes
    - an existing frozen file whose contents do not match its own name ->
      RuntimeError (corruption or a manual edit; silently trusting the filename
      is how a content-addressed store stops being one)
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"cannot freeze a checkpoint that does not exist: {source}")

    run_dir = Path(run_dir) if run_dir is not None else source.parent
    digest = sha256_file(source)
    dest_dir = run_dir / "measured"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest}{source.suffix}"

    if dest.exists():
        # Verify rather than trust. A content-addressed store whose names are
        # taken on faith degrades into a normal directory the first time
        # something writes to it by hand.
        existing = sha256_file(dest)
        if existing != digest:
            raise RuntimeError(
                f"frozen checkpoint {dest} hashes to {existing}, but its name claims "
                f"{digest}. The content-addressed store has been modified out of band; "
                "delete the file and re-freeze rather than measuring against it."
            )
        return FrozenCheckpoint(source, dest, digest, dest.stat().st_size)

    # copy2 preserves mtime, which keeps the frozen copy's timestamp meaningful
    # as "when the training run wrote these weights", not "when we copied them".
    shutil.copy2(source, dest)

    # Re-hash the COPY, not the source. This is the one thing hashing is good
    # for here: catching a training callback that overwrote `source` while
    # shutil was reading it, which would otherwise land a torn file under a
    # name asserting it is intact.
    copied = sha256_file(dest)
    if copied != digest:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source} changed while it was being frozen: read {digest} before the "
            f"copy and {copied} after. A training run overwrote the checkpoint "
            "mid-copy. Re-run the measurement; the partial copy has been removed."
        )

    return FrozenCheckpoint(source, dest, digest, dest.stat().st_size)
