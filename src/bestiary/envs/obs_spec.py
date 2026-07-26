"""One declaration of what an observation contains, and a hash of it.

`learnings/003` calls the observation list the one truly one-way door here: the
actor's first layer is `Linear(obs, 256)`, so changing the list does not degrade
a checkpoint, it makes `SAC.load()` raise. The lesson was written. The door was
walked through anyway, and nobody noticed for a day.

The reason it went unnoticed is structural, not careless. Each env stated its
observation **twice** — a width formula in `__init__` and a `np.concatenate` in
`_get_obs` — with nothing tying them together, and MuJoCo's base class only
*warns* when the two disagree. And no run recorded what it trained against:
`config.json` carries `env_id`, `algo`, `wrapper`, hyperparameters and `seed`,
but no width, no term list, no hash. So the only surviving evidence of a run's
observation was the pickled space width inside its checkpoint, and the only way
to detect a mismatch was to attempt a load and catch the exception.

That is an autopsy, not an instrument. Worse, it is blind to the changes most
likely to happen quietly: reordering two terms, or redefining what a term
*means*, keeps the width identical and is therefore undetectable by every check
in this repository — while orphaning every checkpoint just as completely,
silently, with the model loading fine and the policy behaving like nonsense.

So an observation is declared once, as an ordered list of named terms with
sizes, and that declaration:

* **sizes the Box** — the width is `sum(term.size)`, never a second formula;
* **validates every vector** `_get_obs` builds — and *raises*, with the actual
  numbers, rather than warning;
* **hashes** to a short digest that changes if any name, size, or *order*
  changes, so a width-preserving edit is as visible as a width change.

The hash is the part that did not exist before. Written into `config.json` at
launch and asserted at load, it turns "the observation changed" from something
reconstructed from timestamps into something a guard states.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Bump only when the hash INPUT changes (what goes into the digest), not when a
# robot's terms change. Without it, a future change to the digest recipe would
# silently invalidate every recorded hash and look like universal drift.
SPEC_HASH_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObsTerm:
    """One contiguous block of the observation vector.

    `note` is for the reader and is deliberately NOT hashed: rewording a
    comment must not read as a spec change, or the hash becomes noise and
    people learn to ignore it.
    """

    name: str
    size: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError(f"obs term {self.name!r} has negative size {self.size}")


@dataclass(frozen=True, slots=True)
class ObsSpec:
    """The ordered term list for one environment."""

    env: str
    terms: tuple[ObsTerm, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.env}: duplicate obs term names {dupes}")
        if not self.terms:
            raise ValueError(f"{self.env}: observation spec has no terms")

    @property
    def width(self) -> int:
        return sum(t.size for t in self.terms)

    @property
    def hash(self) -> str:
        """Stable digest over (name, size) in order. Independent of notes.

        Truncated to 16 hex chars: this identifies a spec among a handful per
        repo, it is not a security primitive, and a short digest is one a human
        can compare across a config file and a guard line at a glance.
        """
        payload = f"v{SPEC_HASH_VERSION}|" + "|".join(
            f"{t.name}:{t.size}" for t in self.terms
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate(self, vector: np.ndarray) -> np.ndarray:
        """Raise if a built observation does not match the declaration.

        Loud and immediate, with both numbers and the per-term breakdown --
        the base class only warns on a space mismatch, and a warning in a
        training launch scrolls past in the first second.
        """
        if vector.size != self.width:
            breakdown = ", ".join(f"{t.name}={t.size}" for t in self.terms)
            raise ValueError(
                f"{self.env}: _get_obs built {vector.size} values but the "
                f"declared spec is {self.width} ({breakdown}). The observation "
                f"list and the vector have diverged -- fix the spec or the "
                f"concatenation, and do NOT widen the space to match, because "
                f"that orphans every existing checkpoint (learnings/003)."
            )
        return vector

    def to_record(self) -> dict:
        """What `config.json` stores, so a run says what it trained against."""
        return {
            "hash": self.hash,
            "hash_version": SPEC_HASH_VERSION,
            "width": self.width,
            "terms": [{"name": t.name, "size": t.size} for t in self.terms],
        }
