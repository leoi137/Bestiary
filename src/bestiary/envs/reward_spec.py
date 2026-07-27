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

    `params` IS hashed, and it exists because a weight is not always the whole
    coefficient. A term like

        track_cmd = 1.0 * healthy * Phi(|v - v_cmd| / sigma_v) * Phi(...)

    has weight 1.0 by normalization, and everything that decides what it pays
    lives in sigma_v, sigma_w, the kernel, and the frame. Halving sigma_v is a
    reward change as total as retuning ctrl_cost, and a (name, weight) digest
    is blind to it -- the run would record an identical hash for a different
    objective. `docs/theory/command-tracking-reward.md` Section 6 names this as
    one of two real holes in this machinery; this closes it.

    Values are stringified before hashing so an int, a float and a name all
    digest stably. Floats are rounded to _WEIGHT_PRECISION exactly as weights
    are, for the same reason: formatting must not invent a spec change.
    """

    name: str
    weight: float
    note: str = ""
    params: tuple[tuple[str, object], ...] = ()

    def _param_payload(self) -> str:
        """Canonical, order-independent rendering of `params` for the digest.

        Sorted by key so that reordering the declaration -- which changes
        nothing about the reward -- cannot read as a change. Term ORDER still
        matters and is still hashed; parameter order within a term does not.
        """
        if not self.params:
            return ""
        parts = []
        for key, value in sorted(self.params, key=lambda kv: kv[0]):
            if isinstance(value, float):
                rendered = repr(round(float(value), _WEIGHT_PRECISION))
            else:
                rendered = str(value)
            parts.append(f"{key}={rendered}")
        return "{" + ",".join(parts) + "}"


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """The ordered term list for one environment's reward.

    `cmd_dist` is the second hole Section 6 of the tracking-reward note names,
    and it is subtler than the parameter one. The command distribution is part
    of the objective but lives OUTSIDE the reward function: the sampler, not
    the reward, decides what `|v - v_cmd|` means in expectation. Two runs with
    byte-identical term lists and different command mixtures are not comparable
    -- one may be commanded to drive 80% of the time and the other 20% -- and
    without this field the exact class of silent change the hash exists to
    catch walks straight past it.

    It is a version STRING, not a structure, deliberately. The mixture is
    prose-heavy (three modes, a jittered resample interval, a sign bias), and
    a half-hashed structure is worse than an honest label: it would claim
    coverage it does not have. The string names a distribution defined in code;
    change the code, change the string.

    BACKWARD COMPATIBILITY IS LOAD-BEARING HERE. Both `params` and `cmd_dist`
    contribute NOTHING to the payload when empty, so every spec that predates
    them digests to exactly the byte string it digested before, and every hash
    already written into a `config.json` still verifies. That is why this is
    not a REWARD_HASH_VERSION bump: bumping would invalidate every recorded
    hash at once and read as universal drift, which is precisely what the
    version counter's docstring says it exists to prevent. The rule for a
    future editor: an addition that is inert when unused extends v1; anything
    that changes an existing spec's digest is a version bump.
    """

    env: str
    terms: tuple[RewardTerm, ...]
    cmd_dist: str = ""

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.env}: duplicate reward term names {dupes}")
        if not self.terms:
            raise ValueError(f"{self.env}: reward spec has no terms")

    @property
    def shape_hash(self) -> str:
        """Digest over term NAMES in order, plus the command distribution.

        This is the one that matters for "are these two runs comparable?" --
        which is exactly why `cmd_dist` belongs here and not only in `hash`.
        Two runs whose reward terms match name-for-name but whose commands are
        drawn from different mixtures are optimizing different objectives, and
        this is the field a ledger row carries to claim they can be compared.
        """
        payload = f"v{REWARD_HASH_VERSION}|" + "|".join(t.name for t in self.terms)
        if self.cmd_dist:
            payload += f"|cmd_dist={self.cmd_dist}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def hash(self) -> str:
        """Digest over names, weights, term parameters, and the command mixture."""
        payload = f"v{REWARD_HASH_VERSION}|" + "|".join(
            f"{t.name}:{round(float(t.weight), _WEIGHT_PRECISION)!r}{t._param_payload()}"
            for t in self.terms
        )
        if self.cmd_dist:
            payload += f"|cmd_dist={self.cmd_dist}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_record(self) -> dict:
        """What `config.json` stores, so a run says what reward it trained under."""
        record = {
            "hash": self.hash,
            "shape_hash": self.shape_hash,
            "hash_version": REWARD_HASH_VERSION,
            "terms": [
                {"name": t.name, "weight": round(float(t.weight), _WEIGHT_PRECISION)}
                | ({"params": {k: v for k, v in t.params}} if t.params else {})
                for t in self.terms
            ],
        }
        if self.cmd_dist:
            record["cmd_dist"] = self.cmd_dist
        return record

    def describe(self) -> str:
        """One line per term, for a launch log that is worth reading later."""
        return "\n".join(f"    {t.weight:+.5g}  {t.name}" for t in self.terms)
