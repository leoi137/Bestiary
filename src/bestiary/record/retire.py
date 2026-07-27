"""Declare a run's checkpoints orphaned, so the guard asserts something true.

    python -m bestiary.record.retire --run ant12                       # dry run
    python -m bestiary.record.retire --run ant12 --reason ... --append

`learnings/003` says the observation list is a one-way door: change it and every
existing checkpoint stops loading. The `checkpoint-width` guard enforces that by
loading every checkpoint in `runs/` into its env.

Which leaves a problem the guard cannot solve alone. A run whose door has
*already* been walked through is permanently broken, correctly, forever — and
the guard re-reports it as a fresh failure on every invocation. Since
`guards --fast` gates every training launch, one historical orphan blocks all
future work, and the only ways out are both bad: delete the run (forbidden —
the checkpoints, `ant_tb/` and `config.json` *are* the run, and one of ours is
the evidence behind `research/CORE_PLAN.md`) or stop running the guard.

So an orphan is **declared** instead. The declaration is the record that
learning 003 bit us, and the guard then asserts the stronger pair of statements:
every undeclared run loads, **and every declared orphan really is dead**.

## Why declaring cannot be used to silence a real failure

A retirement mechanism is a way to turn a red guard green, which makes it
exactly the kind of thing that gets abused by a tired operator at 3am or by a
loop optimizing for a clean preflight. Two things prevent that:

1. **This writer refuses to retire a run that still loads.** Every number in the
   row is measured here — the checkpoint's width from its own zip, the env's
   width from a live `gym.make()` — never typed in.
2. **The guard fails on a stale declaration.** If a declared orphan becomes
   loadable again, that is a FAIL telling you to remove the line.

So the only rows that can exist are for genuinely dead checkpoints, and they
stop being valid the moment that stops being true. Retiring a *healthy* run is
not a policy violation to be caught in review; it is impossible.

## Two details that took a review to find

**A build failure is not automatically an unregistered id.** `gym.make` raises
for a missing terrain asset, a bad XML, an import error in an env module — and
if any of those were accepted as "the env is gone", the load gate below could
never run and the whole safety argument would evaporate for exactly the runs
whose envs are temporarily broken. Only `gymnasium.error.UnregisteredEnv` (the
parent of `NameNotFound` and `VersionNotFound`) means the id itself is gone.

**The reason cross-checks are a table, not an if-chain.** Every reason must be
backed by a measurement that contradicts the alternative, and the table is
asserted equal to `REASONS` at import — so a future reason cannot be added
without a check, which is how the `action-width-changed` gap got in.

Append-only, like the ledger: a process that appends cannot lose what is
already there, and a rewriting one can lose all of it on a crash.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import date
from pathlib import Path

from bestiary import paths

CHECKPOINTS = ("ant_sac.zip", "ant_sac_best.zip")

REASONS = {
    "obs-width-changed": "the env's observation list moved; the checkpoint cannot load",
    "env-unregistered": "the env id this run trained against no longer exists",
    "action-width-changed": "the env's action list moved; the checkpoint cannot load",
    # The quiet one. A reorder or a redefinition at identical width loads
    # perfectly and feeds the policy a permuted world, which orphans it just as
    # completely — and invisibly. `obs_spec`'s hash is the only thing that sees
    # it, which is why this reason exists rather than forcing a width story.
    "obs-spec-changed": "the observation was reordered or redefined at the same width",
}


# --- reading the file --------------------------------------------------------
# ONE parser, imported by the guard. Two parsers for one record file is the bug
# class where the writer and the reader quietly disagree about what a row means
# and nothing surfaces it.


def read_rows(path: Path | None = None) -> dict[str, dict]:
    """run name -> its row. Raises ValueError, naming the line, on anything odd.

    Strict on purpose. A malformed file read as "nothing is retired" would turn
    one typo into a wall of unrelated guard failures whose cause is nowhere in
    the output, and a duplicate read as last-wins would let a hand-written row
    silently rewrite what a reader sees about a genuine retirement.
    """
    path = paths.RETIRED if path is None else path
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected a JSON object, got {type(row).__name__}")
            if not isinstance(row.get("run"), str):
                raise TypeError(f"'run' must be a string, got {row.get('run')!r}")
            if row["run"] in rows:
                raise ValueError(
                    f"{row['run']!r} is declared a second time; this file is "
                    "append-only, not last-wins"
                )
            rows[row["run"]] = row
        except (ValueError, TypeError, KeyError) as exc:
            # ValueError covers json.JSONDecodeError, which subclasses it.
            raise ValueError(f"{path.name}:{n} is not a valid retirement row: {exc}") from exc
    return rows


# --- measuring the world -----------------------------------------------------


def checkpoint_shapes(zip_path: Path) -> tuple[int, int]:
    """(obs, act) recorded inside a checkpoint, read from its plain-JSON member."""
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())
    return (
        int(data["observation_space"]["_shape"][0]),
        int(data["action_space"]["_shape"][0]),
    )


def env_shapes(env_id: str) -> tuple[tuple[int, int] | None, Exception | None]:
    """((obs, act), None), or (None, why it would not build).

    The exception is returned rather than swallowed because *which* failure it
    was decides whether retirement is legitimate at all.
    """
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids as an import side effect

    try:
        env = gym.make(env_id)
    except Exception as exc:
        return None, exc
    try:
        return (int(env.observation_space.shape[0]), int(env.action_space.shape[0])), None
    finally:
        env.close()


def spec_hashes(config: dict, env_id: str, buildable: bool) -> tuple[str | None, str | None]:
    """(hash this run recorded, hash the env has now). Either may be None."""
    spec = config.get("obs_spec")
    recorded = None if spec is None else spec.get("hash")
    if not buildable:
        return recorded, None

    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids

    try:
        env = gym.make(env_id)
    except Exception:
        return recorded, None
    try:
        return recorded, env.unwrapped._obs_spec.hash
    except AttributeError:
        return recorded, None
    finally:
        env.close()


# --- the reason table --------------------------------------------------------
# Each entry answers: given what was measured, is this reason TRUE? Returning a
# string means no, and the string is the refusal. Keyed by reason and asserted
# complete at import, so a reason can never be added without its measurement.


def _check_env_unregistered(m: dict) -> str | None:
    if m["live"] is not None:
        return (f"reason 'env-unregistered' but {m['env_id']} builds fine "
                f"({m['live'][0]}obs/{m['live'][1]}act). Use 'obs-width-changed'.")
    import gymnasium.error as gerr

    if not isinstance(m["build_error"], gerr.UnregisteredEnv):
        return (
            f"reason 'env-unregistered' but {m['env_id']} failed to build for a "
            f"different reason: {type(m['build_error']).__name__}: {m['build_error']}\n"
            "  That is a BROKEN env, not a removed one, and while it is broken this\n"
            "  tool cannot check whether the checkpoint still loads -- which is the\n"
            "  whole safety gate. Fix the env, then retire the run if it is really\n"
            "  orphaned."
        )
    return None


def _needs_live(m: dict, reason: str) -> str | None:
    if m["live"] is None:
        return (f"reason {reason!r} but {m['env_id']} cannot be built at all. "
                "Use 'env-unregistered' if the id is gone, or fix the env.")
    return None


def _check_obs_width(m: dict) -> str | None:
    if (err := _needs_live(m, "obs-width-changed")) is not None:
        return err
    if not any(shape[0] != m["live"][0] for shape in m["ckpts"].values()):
        return ("reason 'obs-width-changed' but every checkpoint's observation "
                f"width already equals the env's ({m['live'][0]}).")
    return None


def _check_action_width(m: dict) -> str | None:
    if (err := _needs_live(m, "action-width-changed")) is not None:
        return err
    if not any(shape[1] != m["live"][1] for shape in m["ckpts"].values()):
        return ("reason 'action-width-changed' but every checkpoint's action "
                f"width already equals the env's ({m['live'][1]}).")
    return None


def _check_spec_changed(m: dict) -> str | None:
    if (err := _needs_live(m, "obs-spec-changed")) is not None:
        return err
    if not m["spec_moved"]:
        return ("reason 'obs-spec-changed' but the spec hash did not move "
                f"(recorded {m['recorded_hash']}, live {m['live_hash']}). A run with "
                "no recorded obs_spec cannot use this reason -- there is nothing "
                "to compare.")
    return None


CROSS_CHECKS = {
    "env-unregistered": _check_env_unregistered,
    "obs-width-changed": _check_obs_width,
    "action-width-changed": _check_action_width,
    "obs-spec-changed": _check_spec_changed,
}

# The whole point of the table. A reason with no measurement behind it is a
# reason anyone can assert, which is what the refusal gate exists to prevent.
assert set(CROSS_CHECKS) == set(REASONS), (
    "every reason needs a measurement that backs it: "
    f"missing {sorted(set(REASONS) - set(CROSS_CHECKS))}, "
    f"orphaned {sorted(set(CROSS_CHECKS) - set(REASONS))}"
)


def build_row(run: str, reason: str, note: str, operator: str) -> dict:
    """Measure the run, refuse if it is not actually dead, and return its row."""
    if reason not in REASONS:
        raise SystemExit(f"refusing: unknown reason {reason!r}; one of {sorted(REASONS)}")
    if not note.strip():
        raise SystemExit("refusing: --note is required; a row with no reason is noise")

    run_dir = paths.RUNS / run
    if not run_dir.is_dir():
        raise SystemExit(f"refusing: {run_dir} is not a directory")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"refusing: {config_path} does not exist -- not a run")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"refusing: {config_path} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("env_id"), str):
        raise SystemExit(
            f"refusing: {config_path} has no string 'env_id' -- this is not a run "
            "directory this tool understands"
        )
    env_id = config["env_id"]

    live, build_error = env_shapes(env_id)
    recorded_hash, live_hash = spec_hashes(config, env_id, buildable=live is not None)
    spec_moved = (
        recorded_hash is not None and live_hash is not None and recorded_hash != live_hash
    )

    ckpts = {}
    for name in CHECKPOINTS:
        p = run_dir / name
        if p.exists():
            ckpts[name] = checkpoint_shapes(p)
    if not ckpts:
        raise SystemExit(
            f"refusing: {run} has no checkpoints, so nothing is orphaned. "
            "A run with no .zip is not retired, it is empty."
        )

    # The gate. A row may only exist for a checkpoint that is genuinely dead,
    # in one of the ways deadness is visible: the shapes no longer match, or
    # the spec hash moved under it. Otherwise this file becomes a mute button.
    if live is not None and not spec_moved:
        loadable = [n for n, shape in ckpts.items() if shape == live]
        if loadable:
            raise SystemExit(
                f"refusing: {run} still LOADS into {env_id} "
                f"(env {live[0]}obs/{live[1]}act; {', '.join(sorted(loadable))} match)"
                + (", and its obs spec hash is unchanged" if recorded_hash else "")
                + ".\n"
                "  Retirement is for checkpoints that are already dead. A run that\n"
                "  loads is not retired -- if the guard is failing on it, the guard\n"
                "  is reporting a real regression and this is not the fix."
            )

    measured = {
        "env_id": env_id,
        "live": live,
        "build_error": build_error,
        "ckpts": ckpts,
        "spec_moved": spec_moved,
        "recorded_hash": recorded_hash,
        "live_hash": live_hash,
    }
    if (refusal := CROSS_CHECKS[reason](measured)) is not None:
        raise SystemExit(f"refusing: {refusal}")

    return {
        "run": run,
        "env_id": env_id,
        "reason": reason,
        "retired_at": date.today().isoformat(),
        "retired_by": operator,
        "checkpoints": {n: {"obs": o, "act": a} for n, (o, a) in sorted(ckpts.items())},
        "env_now": None if live is None else {"obs": live[0], "act": live[1]},
        "env_build_error": None if build_error is None else
            f"{type(build_error).__name__}: {build_error}",
        "obs_spec_hash": {"recorded": recorded_hash, "live": live_hash},
        "note": note.strip(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory name under runs/")
    ap.add_argument("--reason", default="obs-width-changed", choices=sorted(REASONS))
    ap.add_argument("--note", default="", help="why this run is kept rather than deleted")
    ap.add_argument("--operator", required=True, help="who is declaring it")
    ap.add_argument("--append", action="store_true", help="write it; otherwise dry run")
    args = ap.parse_args()

    if args.run in read_rows():
        print(f"{args.run} is already retired -- {paths.RETIRED.name} is append-only")
        return 1

    row = build_row(args.run, args.reason, args.note, args.operator)
    line = json.dumps(row, sort_keys=True)

    if not args.append:
        print(line)
        print("\n(dry run -- pass --append to write it)")
        return 0

    paths.RETIRED.parent.mkdir(parents=True, exist_ok=True)
    with paths.RETIRED.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(f"appended to {paths.RETIRED.relative_to(paths.REPO_ROOT)}:")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
