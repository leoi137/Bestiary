"""Executable lessons.

A learning written in prose is only as good as someone's willingness to read
it at the right moment. A learning written as an assertion is enforced whether
anyone remembers it or not — which is the difference between a record that
teaches and a record that merely testifies.

So: **every lesson that can become a check becomes a check.** Each guard here
names the `research/learnings/` entry it enforces, and that entry links back.
When a run is about to repeat a mistake this project has already paid for, a
guard fails and says which lesson it is repeating.

    python -m bestiary.guards            # everything
    python -m bestiary.guards --fast     # skip the ones that step physics
    python -m bestiary.guards --json     # machine-readable, for the record

Exit status is 0 only when every guard passes, so this is usable as a gate in
front of a training launch.

Adding one: write a module with a `run() -> list[Finding]`, then register it in
`REGISTRY` below. Registration is an explicit list rather than package scanning
— an import-time side effect that silently drops a guard is exactly the failure
mode guards exist to prevent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Cost = Literal["fast", "slow"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One assertion's outcome.

    `detail` is read by a human at 3am deciding whether to kill a run, so it
    carries the numbers that justify the verdict, not a restatement of `label`.
    """

    label: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.label}" + (f"   {self.detail}" if self.detail else "")


@dataclass(frozen=True, slots=True)
class Guard:
    """A named check, tied to the lesson it enforces."""

    name: str
    enforces: str  # e.g. "learnings/003" — the lesson this makes unforgettable
    cost: Cost
    run: Callable[[], list[Finding]]


def _registry() -> tuple[Guard, ...]:
    """Import guard modules lazily so `--fast` never pays for mujoco.

    Importing this package must stay free of heavy imports: the loop reads
    `REGISTRY` to decide what to run before it has decided to run anything.
    """
    from bestiary.guards import (
        checkpoint_width,
        disk,
        eval_sampling,
        ledger_schema,
        memory,
        metric_liveness,
        privacy,
        reward_spec,
        spawn_pad,
        standing,
        terrain_spec,
        tracking_frame,
    )

    return (
        # First, and first for a reason: the only irreversible failure here.
        Guard(
            name="privacy",
            enforces="the public/private boundary",
            cost="fast",
            run=privacy.run,
        ),
        Guard(
            name="disk",
            enforces="SYSTEM.md retention policy",
            cost="fast",
            run=disk.run,
        ),
        # Fast on purpose: it reads /proc/meminfo and one systemctl property,
        # and it must gate every launch. A machine that is already full is the
        # one state where starting a run costs more than skipping it.
        Guard(
            name="memory",
            enforces="budget condition 8",
            cost="fast",
            run=memory.run,
        ),
        # Slow tier: it synthesizes two full terrains, which is ~2 s. The
        # finding it protects (learnings/009) changes only when someone edits
        # terrain/generate.py, so it does not need to gate every launch.
        Guard(
            name="spawn-pad",
            enforces="learnings/009",
            cost="slow",
            run=spawn_pad.run,
        ),
        Guard(
            name="ledger-schema",
            enforces="learnings/007",
            cost="fast",
            run=ledger_schema.run,
        ),
        Guard(
            name="checkpoint-width",
            enforces="learnings/003",
            cost="fast",
            run=checkpoint_width.run,
        ),
        # Paired with checkpoint-width on purpose: that one asserts a run
        # records the OBSERVATION it trained against, this one the REWARD.
        # Both halves of the contract, or neither is provenance.
        Guard(
            name="reward-spec",
            enforces="learnings/004",
            cost="fast",
            run=reward_spec.run,
        ),
        # The third leg of the same contract. checkpoint-width asserts a run
        # records its OBSERVATION, reward-spec its REWARD, this one its GROUND
        # — the input that changes without raising, without editing a weight,
        # and without leaving anything in the source to find later.
        Guard(
            name="terrain-spec",
            enforces="anomalies.jsonl 2026-07-27, learnings/001",
            cost="fast",
            run=terrain_spec.run,
        ),
        # Fast, and it has to be: it gates the launch of the very runs it
        # protects. A world-frame regression is only visible in training logs
        # to someone already suspecting it, so catching it after the fact costs
        # a whole run.
        Guard(
            name="tracking-frame",
            enforces="docs/theory/command-tracking-reward.md §1, §2, failure mode 6",
            cost="fast",
            run=tracking_frame.run,
        ),
        Guard(
            name="eval-sampling",
            enforces="learnings/008",
            cost="fast",
            run=eval_sampling.run,
        ),
        Guard(
            name="metric-liveness",
            enforces="episodes/003",
            cost="slow",
            run=metric_liveness.run,
        ),
        Guard(
            name="standing-control",
            enforces="learnings/001, learnings/005",
            cost="slow",
            run=standing.run,
        ),
    )


def registry(fast_only: bool = False) -> tuple[Guard, ...]:
    guards = _registry()
    return tuple(g for g in guards if g.cost == "fast") if fast_only else guards
