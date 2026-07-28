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

    `n` is the size of the **input set this assertion actually quantified
    over** — the number of things it looked at, not the number it could have.
    An assertion over an empty set is *vacuously true*: "every X has property
    P" holds when there is no X. The verdict is then `PASS` and the assertion
    verified nothing, and those two facts are indistinguishable in a boolean.

    That is not hypothetical. `measurement-provenance` shipped green on
    2026-07-28 printing `0 verified, 0 MISMATCHED`, which was read as *no
    problems found* when it says *nothing was examined* — see
    `research/learnings/014`. Cycle 011's response was to write the count into
    `detail`, and 014's own falsifier notes that is necessary and not
    sufficient, because a human read the count and moved on anyway.

    So the size is a field rather than a sentence, and the runner renders
    `n == 0` as **VACUOUS**, never as `PASS`. Prose can be skimmed; a distinct
    verdict cannot.

    `n = None` means *not set-quantified* — a scalar threshold like "≥8000 MiB
    RAM available" has no input set and declaring one would be a lie. `None`
    and `0` are deliberately different facts, exactly as `terrain_spec: null`
    and an absent key are.

    Vacuity is **not** a failure. `runs/` and `*.zip` are gitignored, so a
    fresh clone legitimately has nothing to check, and failing there would make
    the suite useless to anyone who cloned this repo. It is a third status:
    visible, counted in the summary, and it does not gate a launch.
    """

    label: str
    ok: bool
    detail: str = ""
    n: int | None = None

    def __post_init__(self) -> None:
        if self.n is not None and self.n < 0:
            raise ValueError(
                f"Finding({self.label!r}) declares a negative input-set size n={self.n}; "
                "n counts things examined, so it is None (not set-quantified) or >= 0"
            )

    @property
    def vacuous(self) -> bool:
        """True when this assertion passed while examining nothing.

        A FAIL over an empty set is not vacuous — it is a guard that raised
        before it could count, which is a real failure and must read as one.
        """
        return self.ok and self.n == 0

    @property
    def status(self) -> str:
        return "FAIL" if not self.ok else ("VACUOUS" if self.vacuous else "PASS")

    def __str__(self) -> str:
        return f"  [{self.status}] {self.label}" + (f"   {self.detail}" if self.detail else "")


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
        measurement_provenance,
        memory,
        metric_liveness,
        nulls,
        parked_detector,
        privacy,
        reward_spec,
        spawn_pad,
        standing,
        terrain_spec,
        track_length_bias,
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
        # The same contract one level down. The three *-spec guards assert that
        # a RUN records the world it trained in; this one asserts that a
        # MEASUREMENT records the weights it was computed from. Both are
        # provenance, and this is the half that has actually failed in the wild
        # — twice, once putting a wrong number into a published episode that
        # three cycles then reasoned from.
        Guard(
            name="measurement-provenance",
            enforces="learnings/013, anomalies.jsonl rows 19/20/23/27",
            cost="fast",
            run=measurement_provenance.run,
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
        # Fast, and it gates launches for the same reason reward-spec does:
        # the failure it prevents is spending GPU-hours re-entering a dead end
        # this project already paid for. It reads two JSONL files and the
        # config.json of each run — no physics, no torch.
        # Fast: reads one committed JSON. It asserts a property of our own
        # DETECTOR, not of a policy -- learnings/011 was written because a
        # failure mode was named from a threshold nobody had checked against
        # the null policy it exists to catch.
        Guard(
            name="parked-detector",
            enforces="learnings/011",
            cost="fast",
            run=parked_detector.run,
        ),
        Guard(
            name="nulls",
            enforces="research/nulls.jsonl, anomalies.jsonl 2026-07-26",
            cost="fast",
            run=nulls.run,
        ),
        # Fast: pure arithmetic on synthetic episodes, no env and no disk. It
        # asserts the aggregation in record/track_eval.py directly rather than
        # a copy of it, which is the only version of this check that bounds
        # anything.
        Guard(
            name="track-length-bias",
            enforces="anomalies.jsonl row 20 (2026-07-27)",
            cost="fast",
            run=track_length_bias.run,
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
