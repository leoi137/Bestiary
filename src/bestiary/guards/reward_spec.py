"""Every run records the reward it trained under, and that record is honest.

Enforces `learnings/004` — *lock the reward shape, not just the weights* —
whose front matter has read `guard: none — 'is this reward shape final?' is a
judgement, not a check` since it was written. That is true of the question the
learning asks and false of the question the record actually needed answered.

"Is this shape final?" is a judgement. These are not:

* Does every environment declare what its reward pays for?
* Does every run's `config.json` record the reward it trained under?
* Is that record internally consistent, or has someone edited it by hand?

`research/anomalies.jsonl` is the reason this exists: three hound runs at two
different `ctrl_cost_weight` values, two of them compared against each other in
the public record, with nothing on disk saying which was which. That comparison
cannot be checked by any reader today and never will be.

WHY LEGACY RUNS PASS

Runs started before `reward_spec` existed carry no record, and this guard does
not fail them. Back-filling a spec from today's code would manufacture exactly
the false provenance the file exists to prevent -- it would state, with a hash,
that `spyder_desert_v0` trained under the reward `envs/spyder.py` has today,
which is a claim nobody can support. They are listed instead, so the count is
visible and shrinking rather than silently forgiven.

The boundary is `config.json`: a run either recorded a reward or it did not.
That makes the check total over everything from now on and honest about
everything before, which is the same shape as `checkpoint-width`'s treatment
of runs that predate the observation spec.
"""
from __future__ import annotations

import json

from bestiary import paths
from bestiary.guards import Finding

# The four env ids this repo registers. Listed rather than discovered from the
# gym registry because the assertion is "every env WE ship declares a reward":
# a registry walk would also pass vacuously on the day someone deletes an env.
ENV_IDS = ("Spyder-v0", "SpyderDesert-v0", "HoundDesert-v0", "HoundPDDesert-v0")


def _check_envs() -> list[Finding]:
    """Every registered env declares a reward spec, and it is deterministic."""
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  -- registers the env ids

    from bestiary.envs.reward_spec import RewardSpec

    findings: list[Finding] = []
    for env_id in ENV_IDS:
        try:
            env = gym.make(env_id).unwrapped
        except Exception as exc:  # a broken env is a finding, not a crash
            findings.append(Finding(
                f"{env_id}: declares a reward spec", False,
                f"env failed to build: {type(exc).__name__}: {exc}",
            ))
            continue

        spec = getattr(env, "_reward_spec", None)
        if not isinstance(spec, RewardSpec):
            findings.append(Finding(
                f"{env_id}: declares a reward spec", False,
                "no `_reward_spec` on the env. A run against it would record "
                "an observation and no reward, which is how the record ended "
                "up with three hound runs at two ctrl_cost_weights and no way "
                "to tell them apart (learnings/004).",
            ))
            continue

        # Rebuild and re-hash. A spec whose digest depends on anything but its
        # own declared contents -- a float that formats differently, a set
        # iterated in hash order -- would make every recorded hash unfalsifiable.
        again = gym.make(env_id).unwrapped._reward_spec
        if again.hash != spec.hash or again.shape_hash != spec.shape_hash:
            findings.append(Finding(
                f"{env_id}: declares a reward spec", False,
                f"the hash is not deterministic across two constructions: "
                f"{spec.hash}/{spec.shape_hash} then {again.hash}/{again.shape_hash}. "
                f"A recorded hash that cannot be reproduced proves nothing.",
            ))
            continue

        findings.append(Finding(
            f"{env_id}: declares a reward spec", True,
            f"{len(spec.terms)} terms, hash {spec.hash}, shape {spec.shape_hash}",
        ))
    return findings


def _check_runs() -> list[Finding]:
    """A recorded reward spec re-hashes to the hashes recorded beside it."""
    from bestiary.envs.reward_spec import RewardSpec, RewardTerm

    recorded, legacy, bad = [], [], []
    for config_path in sorted(paths.RUNS.glob("*/config.json")):
        run = config_path.parent.name
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            bad.append(f"{run}: config.json does not parse ({exc})")
            continue

        spec_record = config.get("reward_spec")
        if spec_record is None:
            legacy.append(run)
            continue

        terms = spec_record.get("terms")
        if not terms:
            bad.append(f"{run}: reward_spec present but carries no terms")
            continue

        rebuilt = RewardSpec(
            env=run,
            terms=tuple(RewardTerm(t["name"], t["weight"]) for t in terms),
        )
        if rebuilt.hash != spec_record.get("hash"):
            bad.append(
                f"{run}: recorded hash {spec_record.get('hash')} but its own "
                f"terms hash to {rebuilt.hash} — config.json was edited by hand"
            )
        elif rebuilt.shape_hash != spec_record.get("shape_hash"):
            bad.append(
                f"{run}: recorded shape_hash {spec_record.get('shape_hash')} but "
                f"its own terms hash to {rebuilt.shape_hash}"
            )
        else:
            recorded.append(run)

    findings = [Finding(
        "every recorded reward spec re-hashes to what is recorded beside it",
        not bad,
        "; ".join(bad) if bad else
        f"{len(recorded)} run(s) carry a verified reward spec",
    )]

    # Not a failure: a fact, kept visible so the number is seen to shrink.
    findings.append(Finding(
        "runs predating the reward spec are declared, not back-filled", True,
        f"{len(legacy)} run(s) carry no reward record and cannot be attributed "
        f"to a reward: {sorted(legacy)}" if legacy else "none",
    ))
    return findings


def run() -> list[Finding]:
    return _check_envs() + _check_runs()
