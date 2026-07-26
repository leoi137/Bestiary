"""Guard: every checkpoint still loads into the env that produced it.

Enforces `research/learnings/003` (changing the observation list throws away
every checkpoint).

The actor's first layer is `Linear(obs, 256)`, so a checkpoint trained at 113
observations does not degrade when the env moves to 141 — `SAC.load()` raises
and the run is gone. The failure surfaces hours later, at resume time, as a
shape error with no hint that an env edit three days ago caused it.

This reads the shapes out of the checkpoint's `data` member, which is plain
JSON inside the zip, so no torch import and no GPU. Comparing against a live
`gym.make()` costs one MuJoCo model load per distinct env id.
"""
from __future__ import annotations

import json
import zipfile
from functools import cache

from bestiary import paths
from bestiary.guards import Finding

CHECKPOINTS = ("ant_sac.zip", "ant_sac_best.zip")


@cache
def _env_shapes(env_id: str) -> tuple[int, int]:
    """(observation width, action width) of a freshly built env.

    Cached because building the desert envs loads a heightfield, and several
    runs share an env id.
    """
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids as an import side effect

    env = gym.make(env_id)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.shape[0])
    finally:
        env.close()


@cache
def _env_obs_spec(env_id: str):
    """The live `ObsSpec` for an env id. Cached for the same reason as above."""
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  — registers the ids

    env = gym.make(env_id)
    try:
        return env.unwrapped._obs_spec
    finally:
        env.close()


def _checkpoint_shapes(zip_path) -> tuple[int, int]:
    with zipfile.ZipFile(zip_path) as z:
        data = json.loads(z.read("data").decode())
    return (
        int(data["observation_space"]["_shape"][0]),
        int(data["action_space"]["_shape"][0]),
    )


def run() -> list[Finding]:
    if not paths.RUNS.exists():
        return [Finding("runs/ exists", True, "no runs yet — nothing to check")]

    findings: list[Finding] = []
    checked = 0
    unpinned: list[str] = []

    for run_dir in sorted(p for p in paths.RUNS.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
            env_id = config["env_id"]
        except (json.JSONDecodeError, KeyError) as exc:
            findings.append(Finding(f"{run_dir.name}: config is readable", False, str(exc)))
            continue

        try:
            want_obs, want_act = _env_shapes(env_id)
        except Exception as exc:  # a missing asset or an unregistered id
            findings.append(
                Finding(f"{run_dir.name}: env {env_id} builds", False,
                        f"{type(exc).__name__}: {exc}")
            )
            continue

        # The width comparison below only catches a change that RESIZES the
        # observation. A reorder or a redefinition keeps the width and loads
        # cleanly, feeding the policy a permuted world — so compare the
        # recorded spec hash too, when the run has one.
        recorded_spec = config.get("obs_spec")
        if recorded_spec is None:
            # Not a failure: these runs predate the spec record. Counted and
            # named rather than skipped, because a silent skip is how a guard
            # reports full coverage it does not have.
            unpinned.append(run_dir.name)
        else:
            try:
                live = _env_obs_spec(env_id)
            except Exception as exc:
                findings.append(
                    Finding(f"{run_dir.name}: {env_id} exposes an obs spec", False,
                            f"{type(exc).__name__}: {exc}")
                )
                live = None
            if live is not None:
                same = recorded_spec.get("hash") == live.hash
                findings.append(
                    Finding(
                        f"{run_dir.name}: obs spec still matches what it recorded",
                        same,
                        f"recorded {recorded_spec.get('hash')} "
                        f"(width {recorded_spec.get('width')}) vs live {live.hash} "
                        f"(width {live.width})"
                        + ("" if same else
                           "  <- the observation list changed under this run"),
                    )
                )

        for name in CHECKPOINTS:
            zip_path = run_dir / name
            if not zip_path.exists():
                continue
            checked += 1
            try:
                got_obs, got_act = _checkpoint_shapes(zip_path)
            except (KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                findings.append(
                    Finding(f"{run_dir.name}/{name}: readable", False,
                            f"{type(exc).__name__}: {exc}")
                )
                continue

            ok = (got_obs, got_act) == (want_obs, want_act)
            findings.append(
                Finding(
                    f"{run_dir.name}/{name} loads into {env_id}",
                    ok,
                    f"checkpoint {got_obs}obs/{got_act}act vs env {want_obs}/{want_act}"
                    + ("" if ok else "  <- this checkpoint is orphaned"),
                )
            )

    if not checked:
        findings.append(Finding("checkpoints found", True, "none on disk"))

    # Reported, never silent. Every unpinned run is one whose observation could
    # change without any check noticing, and the number should visibly fall to
    # zero as runs are started under the spec record rather than quietly sit
    # there. It is not a FAIL: these runs are finished, and inventing a spec
    # for them from today's code would be the false provenance this prevents.
    findings.append(
        Finding(
            "every run records the observation spec it trained against",
            True,
            f"{len(unpinned)} run(s) predate the spec record and cannot be "
            f"checked for a width-preserving change: {sorted(unpinned)}"
            if unpinned else "all runs pinned",
        )
    )
    return findings
