"""One declaration of what a reward pays for, and a hash of it.

`envs/obs_spec.py` is the pattern; this is the same treatment applied to the
other half of the run's contract. The observation is now pinned into
`config.json` and asserted on resume. The reward is not, and
`research/anomalies.jsonl` records what that costs:

    hound_desert_test150k ran at ctrl_cost_weight 0.1 and the two ledger runs
    at 0.01, and nothing in any run directory records that.

So a ledger row's numbers cannot be attributed to a reward. Three hound runs
sit in the record, two of them compared against each other, and the only
evidence of which reward each trained under is the default value in whatever
`envs/hound.py` happened to say that day.

WHY A SEPARATE SHAPE HASH

`learnings/004` is the reason this file has two hashes instead of one. Its
finding is that locking a reward's *numbers* is not locking a reward:

    "That is a change to the reward's TERMS, not its coefficients. A term that
    is going to be added later is just as breaking as a coefficient that is
    going to change."

The two failures are not equally bad and must not be reported as one thing:

* **A weight moved.** `ctrl_cost_weight` 0.1 -> 0.01. The replay buffer is
  contaminated -- every stored transition is labelled with a reward nothing
  will pay again -- but the two runs are still *comparable in kind*. You can
  say "same reward, retuned" and mean something.
* **A term appeared, vanished, or was reordered.** The forward-velocity term
  replaced by command tracking. Nothing about the two runs is comparable, and
  no amount of relabelling fixes it, because the new reward reads an input the
  old one never had.

`shape_hash` covers names and order only, so it is stable across retuning and
moves the moment the reward starts paying for a different thing. That is the
number a ledger row should carry when it claims two runs can be compared.

WHAT THIS DELIBERATELY DOES NOT DO

It does not decide whether a reward shape is *final* -- learnings/004 is right
that this is a judgement and not a check. What it makes checkable is the much
narrower thing the record actually needed: whether the reward a run trained
under is **recorded**, and whether it **moved** underneath a resume.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Bump only when the hash INPUT changes (what goes into the digest), not when a
# robot's reward changes. Same reasoning as SPEC_HASH_VERSION in obs_spec.py:
# without it, a future change to the digest recipe silently invalidates every
# recorded hash and reads as universal drift.
REWARD_HASH_VERSION = 1

# Weights are rounded before hashing so that float formatting cannot invent a
# spec change. 9 decimal places is far finer than any weight this project uses
# (the smallest is contact_cost_weight = 5e-4) and coarse enough that a value
# reconstructed through JSON round-trips to the same digest.
_WEIGHT_PRECISION = 9


@dataclass(frozen=True, slots=True)
class RewardTerm:
    """One additive term of the reward.

    `weight` is the coefficient as the env applies it, sign included: a cost
    carries a negative weight here even though the env's constructor takes a
    positive `ctrl_cost_weight` and subtracts it. The record should read the
    way the arithmetic reads, or nobody can check the sum by eye.

    `note` is for the reader and is deliberately NOT hashed, for the same
    reason as ObsTerm.note: rewording a comment must not read as a spec change,
    or the hash becomes noise and people learn to ignore it.
    """

    name: str
    weight: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """The ordered term list for one environment's reward."""

    env: str
    terms: tuple[RewardTerm, ...]

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.env}: duplicate reward term names {dupes}")
        if not self.terms:
            raise ValueError(f"{self.env}: reward spec has no terms")

    @property
    def shape_hash(self) -> str:
        """Digest over term NAMES in order. Invariant to retuning.

        This is the one that matters for "are these two runs comparable?".
        """
        payload = f"v{REWARD_HASH_VERSION}|" + "|".join(t.name for t in self.terms)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def hash(self) -> str:
        """Digest over names AND weights in order. The full identity."""
        payload = f"v{REWARD_HASH_VERSION}|" + "|".join(
            f"{t.name}:{round(float(t.weight), _WEIGHT_PRECISION)!r}" for t in self.terms
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_record(self) -> dict:
        """What `config.json` stores, so a run says what reward it trained under."""
        return {
            "hash": self.hash,
            "shape_hash": self.shape_hash,
            "hash_version": REWARD_HASH_VERSION,
            "terms": [
                {"name": t.name, "weight": round(float(t.weight), _WEIGHT_PRECISION)}
                for t in self.terms
            ],
        }

    def describe(self) -> str:
        """One line per term, for a launch log that is worth reading later."""
        return "\n".join(f"    {t.weight:+.5g}  {t.name}" for t in self.terms)
