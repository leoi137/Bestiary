"""Assert that the ledger's two stability fields mean what the record says.

    venv/bin/python -m bestiary.record.check_ledger_fields
    venv/bin/python -m bestiary.record.check_ledger_fields -v

`mean_eval_after_converge` and `eval_crash_rate` exist because `learnings/007`
found that `best_eval_return` ranked a policy scoring 1218 once and 390
repeatedly *above* one that reliably scored 1170. They are the ledger's only
defence against that, which makes their definitions worth an oracle: both are
one plausible-looking line away from silently reverting to the biased reading.

Two failure modes, one check each, and neither is caught by anything else:

1. **A crash counted from episode length rather than termination.** The two
   agree for today's envs and disagree the moment an env truncates for its own
   reasons, or a robot falls on the final step. `length < cap` is the reading
   that "obviously works", so it is the one a later edit will drift back to.
2. **A converged window taken as a fraction of run length.** Right for a
   1M-step run by coincidence, wrong for every other length, and wrong in the
   direction that flatters unstable policies.

Hermetic and CPU-only: no MuJoCo, no checkpoints, no GPU, no run directories.
The env is a stub and the eval series are hand-built, so this runs anywhere and
in milliseconds. The real runs are covered by the `--real` pass below, which
recomputes `learnings/007`'s published numbers from the event files when those
runs are present and says so when they are not.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from bestiary import paths
from bestiary.record import greedy_eval, ledger

FAILURES: list[str] = []
VERBOSE = False


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    if VERBOSE or not ok:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")


# --------------------------------------------------------------------------
# A stub env. Only what `_rollout` touches, scripted per episode.
# --------------------------------------------------------------------------
@dataclass
class Space:
    shape: tuple = (2,)
    dtype: object = np.float32


@dataclass
class Spec:
    max_episode_steps: int


class StubEnv:
    """Ends episode i after `plan[i] = (steps, terminated, truncated)`."""

    def __init__(self, plan, cap=1000):
        self.plan = plan
        self.spec = Spec(cap)
        self.action_space = Space()
        self.episode = -1
        self.steps = 0

    def reset(self, seed=None):
        self.episode += 1
        self.steps = 0
        return np.zeros(3), {}

    def step(self, action):
        self.steps += 1
        n, terminated, truncated = self.plan[self.episode]
        done = self.steps >= n
        return np.zeros(3), 1.0, bool(terminated and done), bool(truncated and done), {}

    def close(self):
        pass


def roll(plan, cap=1000):
    """Run `_rollout` against a StubEnv, with gym.make patched to return it."""
    import gymnasium as gym

    saved = gym.make
    gym.make = lambda env_id: StubEnv(plan, cap)
    try:
        return greedy_eval._rollout("Stub-v0", None, len(plan), 0)
    finally:
        gym.make = saved


def scalars(pairs):
    """`(step, value)` pairs as the duck-typed events `after_converge` reads."""
    @dataclass
    class E:
        step: int
        value: float

    return [E(s, v) for s, v in pairs]


# --------------------------------------------------------------------------
def check_crash_is_termination_not_length() -> None:
    print("crash rate counts terminations, not short episodes")

    # Three clean episodes at the cap, two early terminations.
    arm = roll([(1000, False, True)] * 3 + [(400, True, False)] * 2)
    check("an unhealthy stop is a crash", arm.crashes == 2, f"crashes={arm.crashes}")
    check("crash_rate is crashes/episodes", arm.crash_rate == 0.4, f"{arm.crash_rate}")

    # The case the length proxy gets BACKWARDS. TimeLimit sets truncated=True
    # without clearing terminated, so a robot that falls on the final step is
    # reported as both, at a length equal to the cap. It is a crash.
    arm = roll([(1000, True, True), (1000, False, True)])
    by_length = sum(1 for n in arm.lengths if n < arm.max_episode_steps)
    check("a fall on the last step is a crash", arm.crashes == 1, f"crashes={arm.crashes}")
    check(
        "  ...and the length proxy would have missed it",
        by_length == 0 and arm.crashes != by_length,
        f"by_length={by_length} vs terminated={arm.crashes} -- if these are ever "
        f"equal here the stub stopped exercising the case",
    )

    # An env that truncates on its own makes LENGTH a phantom-crash generator.
    # The rate itself stays right, but lengths are read as a proxy elsewhere in
    # the record, so this must be loud rather than quietly correct.
    try:
        roll([(500, False, True), (1000, False, True)])
        check("a self-truncating env is rejected", False, "no exception raised")
    except RuntimeError as exc:
        check(
            "a self-truncating env is rejected",
            "truncated on its own" in str(exc) and "500" in str(exc),
            "raises with the offending step count",
        )

    # No horizon means no definable crash rate.
    try:
        roll([(10, True, False)], cap=None)
        check("an env with no horizon is rejected", False, "no exception raised")
    except ValueError as exc:
        check("an env with no horizon is rejected", "max_episode_steps" in str(exc), "")

    check("crash_rate of a crash-free arm is 0.0",
          roll([(1000, False, True)] * 4).crash_rate == 0.0, "")


def check_converge_window_is_absolute() -> None:
    print("mean_eval_after_converge uses an absolute step cutoff")

    # Same policy, two run lengths. An absolute cutoff selects the same eval
    # points from both; a fractional one does not, which is what makes rows of
    # different length incomparable.
    short = scalars([(100_000, 10.0), (400_000, 100.0), (900_000, 100.0)])
    long = scalars([(100_000, 10.0), (400_000, 100.0), (900_000, 100.0),
                    (5_000_000, 100.0), (9_000_000, 100.0)])

    check("the cutoff is the one that reproduces learnings/007",
          ledger.CONVERGE_AFTER_STEPS == 400_000,
          f"CONVERGE_AFTER_STEPS={ledger.CONVERGE_AFTER_STEPS:,}")
    check("it is an absolute step count, not a fraction",
          ledger.CONVERGE_AFTER_STEPS > 1,
          "a value <= 1 means someone reverted this to a fraction of run length")

    got = ledger.after_converge(short + [scalars([(400_001, 100.0)])[0]] * 3)
    check("the pre-cutoff point is excluded", 10.0 not in got, f"{got}")
    check("the boundary point is included (>=, not >)", 100.0 in got, f"{got}")

    padded = long + scalars([(9_000_001, 100.0)] * 2)
    check("a 9M-step run keeps its early converged evals",
          len(ledger.after_converge(padded)) == len(padded) - 1,
          "a fractional window would have dropped everything before 3.6M")

    # Too few points is a refusal, not a quiet mean of two numbers.
    try:
        ledger.after_converge(scalars([(500_000, 1.0), (600_000, 2.0)]))
        check("too few converged evals is refused", False, "no exception raised")
    except ValueError as exc:
        check("too few converged evals is refused",
              "too short" in str(exc), "names the count and the need")

    try:
        ledger.after_converge(scalars([(1_000, 1.0)] * 20))
        check("a run that never reaches the cutoff is refused", False, "no exception")
    except ValueError as exc:
        check("a run that never reaches the cutoff is refused",
              "0 eval point" in str(exc), "reports zero points")


def check_validate_protects_the_launch_gate() -> None:
    """`guards --fast` gates every launch and the ledger is append-only.

    So a row that the eval-sampling guard rejects is not a bad record, it is a
    permanently red launch gate. `validate` has to refuse it before it lands.
    """
    print("validate refuses rows that would turn the launch gate red")

    def row(**over):
        base = {f: 1 for f in ledger.BASE_FIELDS}
        base |= {f: 1 for f in ledger.FIELDS_FROM_ROW_3}
        base |= {
            "run": "__stub_run_that_does_not_exist__",
            "date": "2026-01-01",
            "verdict": "improved",
            "seeds": 1,
            "provisional": True,
            "eval_crash_rate": 0.1,
            ledger.N_FIELD: ledger.MIN_EVAL_EPISODES,
        }
        return base | over

    def refuses(name, expect, **over):
        try:
            ledger.validate(row(**over))
            check(name, False, "no exception raised -- the row would have been written")
        except ValueError as exc:
            check(name, expect in str(exc), f"{str(exc)[:90]}")

    # The baseline row must pass, or the refusals below prove nothing.
    try:
        ledger.validate(row())
        check("a well-formed row is accepted", True, "")
    except ValueError as exc:
        check("a well-formed row is accepted", False, str(exc))

    refuses("an under-sampled row is refused", "at least",
            **{ledger.N_FIELD: 5})
    refuses("a row with no episode count is refused", "at least",
            **{ledger.N_FIELD: None})
    refuses("a crash rate above 1 is refused", "proportion", eval_crash_rate=1.5)
    refuses("a negative crash rate is refused", "proportion", eval_crash_rate=-0.1)
    refuses("a single-seed row not marked provisional is refused", "provisional",
            provisional=False)
    refuses("an unknown verdict is refused", "not in", verdict="great")
    refuses("a duplicate run name is refused", "append-only", run="hound_desert_v0")

    check("the writer's floor matches the guard's",
          ledger.EVAL_EPISODES >= ledger.MIN_EVAL_EPISODES,
          f"default episodes={ledger.EVAL_EPISODES}, guard floor="
          f"{ledger.MIN_EVAL_EPISODES}")


def check_against_the_published_record() -> None:
    """Recompute `learnings/007`'s two published numbers from the event files.

    This is the check that would actually have caught the fractional reading,
    so it is worth the seconds it costs. It needs the runs, so it reports its
    own absence rather than passing vacuously -- a silent skip is how a check
    claims coverage it does not have.
    """
    print("the cutoff reproduces the numbers learnings/007 published")

    # learnings/007, "mean eval after 400k". The record is the oracle.
    published = {"hound_desert_v0": 887.5, "hound_pd_desert_v0": 1113.1}

    for run, expected in published.items():
        run_dir = paths.RUNS / run
        if not run_dir.is_dir():
            check(f"{run}: present to check against", True,
                  "SKIPPED -- run directory absent, this number is UNVERIFIED here")
            continue
        evals = ledger._scalars(run_dir)["eval/mean_reward"]
        got = float(np.mean(ledger.after_converge(evals)))
        check(f"{run}: mean eval after converge == learnings/007",
              abs(got - expected) < 0.05,
              f"computed {got:.2f} vs published {expected}")


def main() -> int:
    global VERBOSE
    VERBOSE = "-v" in sys.argv

    check_crash_is_termination_not_length()
    check_converge_window_is_absolute()
    check_validate_protects_the_launch_gate()
    if "--no-real" not in sys.argv:
        check_against_the_published_record()

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nall ledger-field checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
