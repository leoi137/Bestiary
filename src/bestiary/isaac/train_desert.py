"""Train on the Bestiary desert, reusing Isaac Lab's own rsl_rl trainer.

WHY A RUNNER AND NOT JUST `isaaclab.sh train`

`isaaclab.sh train` execs Isaac Lab's script in a fresh process, and that
process imports `isaaclab_tasks` and nothing of ours -- so a Bestiary task id is
simply unknown to it. Isaac Lab has no external-task discovery hook; its own
answer, in `tools/template`, is that an external project ships its own entry
point.

This is that entry point, and it is deliberately thin. It registers our tasks
(cheap: `tasks.py` touches only gymnasium) and then calls Isaac Lab's
`train_rsl_rl.run(argv)` unchanged. Every hyperparameter, every logging
convention, every checkpoint layout is theirs. Reimplementing a PPO loop here
would be a second thing to keep correct for no benefit, and would quietly make
our numbers incomparable to theirs.

USAGE

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.train_desert \\
        --task Bestiary-Desert-Anymal-C-v0 --num_envs 1024 --max_iterations 100

Every flag is forwarded verbatim to Isaac Lab's trainer, so `--viz none`,
`--video`, `--seed`, `--resume` and the rest behave exactly as documented there.
Logs land in Isaac Lab's `logs/rsl_rl/<experiment>/<timestamp>/` as usual.

RESUMING, AND THE ONE FLAG THIS FILE ADDS

Isaac Lab's own resume is `--resume --load_run <dir> --checkpoint <file>`, and
it searches ONE place: `<cwd>/logs/rsl_rl/<experiment_name>/`, where
`experiment_name` comes from the task's agent config.
`isaaclab_tasks.utils.get_checkpoint_path` matches `--load_run` as a regex
against the directory NAMES it finds by scanning that root, so the search
cannot leave the experiment it belongs to.

That is exactly right for continuing a run and useless for a FINE-TUNE, which
by definition reads one experiment's checkpoint and writes another's log tree —
`Bestiary-Fast-Spyder-v0` resumes `spyder_overnight`'s `model_14999.pt` and must
file its own checkpoints under `spyder_fast`. Pointing `--experiment_name` at
the source would fix the read and break the write, putting the fine-tune's
checkpoints inside the run it inherited from.

So this file adds ONE flag, `--from-checkpoint <path>`, and nothing else:

    --from-checkpoint /workspace/Bestiary/logs/rsl_rl/spyder_overnight/<run>/model_14999.pt

It is consumed here, never forwarded. It requires the file to exist (a missing
checkpoint raises before the app boots, naming the path it looked for), injects
`--resume` so the trainer takes its load branch, and replaces the checkpoint
RESOLVER with one that returns the given path. Loading stays upstream's:
`OnPolicyRunner.load` is `strict=True`, so an observation or action width that
moved fails the load instead of silently training a permuted policy, and it
restores the optimiser state and `current_learning_iteration` -- which means
`--max_iterations N` on a resumed run is N ADDITIONAL iterations, not "train to
iteration N".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: `--load_run` value injected alongside `--from-checkpoint`. It is a regex, and
#: it is deliberately one that cannot match any directory a human would create.
#:
#: This is the whole guard against the quietest way the pass-through could fail.
#: If the resolver patch below ever stops taking -- upstream moves
#: `get_checkpoint_path`, or binds it at module import instead of at call time --
#: the trainer falls back to its own search of
#: `logs/rsl_rl/<experiment_name>/`. Without this sentinel that search would
#: succeed on the second fine-tune of a task and load THE WRONG CHECKPOINT with
#: no error at all. With it, the search matches nothing and upstream raises
#: `ValueError: No runs present in the directory ... match
#: '__bestiary_from_checkpoint__'`, which greps straight back to this constant.
_LOAD_RUN_SENTINEL = "__bestiary_from_checkpoint__"

#: Set by the resolver shim the first time the trainer asks it for a path. Read
#: after `run()` returns: a `--from-checkpoint` launch whose shim was never
#: called did not fine-tune anything, and must not exit 0 pretending it did.
_resolver_was_used = False


def _isaaclab_root() -> Path:
    """Locate the Isaac Lab checkout, so its trainer can be imported.

    `$ISAACLAB_PATH` wins if set (it is exported by `isaaclab.sh`). Otherwise the
    root is derived from the installed `isaaclab` package, which is an editable
    install pointing into the checkout:

        <root>/source/isaaclab/isaaclab/__init__.py  ->  parents[3] == <root>

    Raises with both candidates rather than falling back to a guess: a wrong
    root would import a *different* Isaac Lab than the one supplying our
    physics, and nothing downstream would notice.
    """
    env_root = os.environ.get("ISAACLAB_PATH")
    if env_root:
        root = Path(env_root)
        if (root / "scripts" / "reinforcement_learning" / "rsl_rl").is_dir():
            return root
        raise RuntimeError(
            f"$ISAACLAB_PATH={root} does not contain "
            "scripts/reinforcement_learning/rsl_rl -- it is not an Isaac Lab checkout"
        )

    import isaaclab

    root = Path(isaaclab.__file__).resolve().parents[3]
    if (root / "scripts" / "reinforcement_learning" / "rsl_rl").is_dir():
        return root
    raise RuntimeError(
        f"could not locate the Isaac Lab checkout. $ISAACLAB_PATH is unset and the "
        f"path derived from the isaaclab package ({root}) has no "
        "scripts/reinforcement_learning/rsl_rl directory. Set ISAACLAB_PATH."
    )


def _split_from_checkpoint(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Pull `--from-checkpoint` out of argv; everything else passes through.

    `add_help=False` so `-h` still reaches Isaac Lab's parser and prints ITS
    options, and `allow_abbrev=False` so a prefix of our flag cannot swallow one
    of theirs. Returns the resolved path (or None) and the argv to forward.

    Raises rather than resolving a conflict, in three cases, because each of
    them is a launch that would train something other than what was asked for:
    a checkpoint that does not exist, a path that is a directory, and
    `--load_run` / `--checkpoint` given alongside — those two are upstream's own
    search parameters and this flag overrides the search entirely, so accepting
    them would silently ignore what the launch line says.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--from-checkpoint", dest="from_checkpoint", type=str, default=None)
    args, rest = parser.parse_known_args(argv)

    if args.from_checkpoint is None:
        return None, rest

    clashes = sorted(
        flag
        for flag in ("--load_run", "--checkpoint")
        if any(a == flag or a.startswith(f"{flag}=") for a in rest)
    )
    if clashes:
        raise SystemExit(
            f"--from-checkpoint cannot be combined with {clashes}. It replaces "
            "Isaac Lab's checkpoint SEARCH with an explicit path, so those flags "
            "would be accepted and then ignored — which is how a fine-tune ends "
            "up resuming a checkpoint nobody named. Drop them, or drop "
            "--from-checkpoint and use upstream's --resume --load_run --checkpoint "
            "within one experiment_name."
        )

    path = Path(args.from_checkpoint).expanduser().resolve()
    if not path.exists():
        raise SystemExit(
            f"--from-checkpoint {path} does not exist. Nothing is resumed from a "
            "missing file: this run would have trained from scratch under a task "
            "whose whole purpose is to continue an existing policy, and the only "
            "sign would have been the reward curve starting near zero."
        )
    if not path.is_file():
        raise SystemExit(
            f"--from-checkpoint {path} is not a file. Give the checkpoint itself "
            "(a model_*.pt), not the run directory that holds it."
        )

    # `--resume` is what makes the trainer take its load branch at all
    # (`if agent_cfg.resume: ... runner.load(resume_path)`); the sentinel is the
    # guard documented on `_LOAD_RUN_SENTINEL`.
    forwarded = list(rest)
    if "--resume" not in forwarded:
        forwarded.append("--resume")
    forwarded += ["--load_run", _LOAD_RUN_SENTINEL]
    return path, forwarded


def _install_checkpoint_resolver(path: Path) -> None:
    """Point Isaac Lab's checkpoint resolver at one explicit file.

    `train_rsl_rl.run` does `from isaaclab_tasks.utils import get_checkpoint_path`
    INSIDE the function body, so the name is looked up on the module at call
    time and replacing the module attribute is enough. `isaaclab_tasks.utils`
    and `isaaclab_tasks.utils.parse_cfg` currently expose the same function
    object; both are replaced, so a future `from ...parse_cfg import` in the
    trainer stays covered.

    Raises if the attribute is missing on either module — an upstream rename
    must fail here, loudly, rather than leave a patch that silently does
    nothing and a run that silently starts from random weights.

    Importing `isaaclab_tasks.utils` pre-app is safe and was measured:
    `isaaclab_tasks` is already imported by `train_rsl_rl` at module scope, it
    already carries `.utils`, and neither import puts `pxr` in `sys.modules`
    (checked 2026-08-07). That matters because a pip `pxr` in the process before
    Kit boots is the measured `free(): invalid pointer` crash `commands_impl.py`
    documents.
    """
    import isaaclab_tasks.utils as tasks_utils
    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    def resolve(*_args, **_kwargs) -> str:
        global _resolver_was_used
        if not path.is_file():
            raise RuntimeError(
                f"--from-checkpoint {path} vanished between launch and load. "
                "Nothing downstream would notice a fine-tune that quietly "
                "became a fresh run."
            )
        _resolver_was_used = True
        return str(path)

    for module in (tasks_utils, parse_cfg):
        if not hasattr(module, "get_checkpoint_path"):
            raise RuntimeError(
                f"{module.__name__} has no `get_checkpoint_path` to replace. "
                "Isaac Lab moved or renamed the checkpoint resolver, so "
                "--from-checkpoint cannot take effect and this launch would "
                "train from scratch. Re-read the trainer's resume branch."
            )
        module.get_checkpoint_path = resolve


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    from_checkpoint, argv = _split_from_checkpoint(argv)

    if from_checkpoint is not None:
        print(
            "[bestiary] fine-tune: resuming from "
            f"{from_checkpoint}\n"
            "[bestiary] the trainer must print "
            f"'[INFO]: Loading model checkpoint from: {from_checkpoint}' before "
            "iteration 0",
            flush=True,
        )
    if any(a in ("-h", "--help") for a in argv):
        print(
            "bestiary addition to the options below:\n"
            "  --from-checkpoint PATH\n"
            "                        resume from THIS checkpoint file, whatever\n"
            "                        experiment it belongs to. Implies --resume\n"
            "                        and replaces --load_run/--checkpoint. Logs\n"
            "                        still land under this task's own\n"
            "                        experiment_name.\n",
            flush=True,
        )

    # Register first. tasks.py imports only gymnasium, and Isaac Lab's config
    # entry points are strings resolved lazily, so this must NOT pull in
    # isaaclab -- the app is not running yet.
    from bestiary.isaac import tasks

    tasks.register()

    # Isaac Lab's trainer lives in scripts/, which is not an installed package, so
    # its directories go on sys.path by hand. BOTH are required and it is not
    # obvious why: `train_rsl_rl` imports `cli_args` from its own directory, and
    # also `from common import ...`, which lives one level up in
    # scripts/reinforcement_learning/. Omitting the parent fails at import with a
    # bare "No module named 'common'".
    rl_dir = _isaaclab_root() / "scripts" / "reinforcement_learning"
    for path in (rl_dir / "rsl_rl", rl_dir):
        if not path.is_dir():
            raise RuntimeError(f"expected Isaac Lab trainer directory at {path}")
        sys.path.insert(0, str(path))

    import train_rsl_rl

    if from_checkpoint is not None:
        _install_checkpoint_resolver(from_checkpoint)

    train_rsl_rl.run(argv)

    if from_checkpoint is not None and not _resolver_was_used:
        raise RuntimeError(
            f"--from-checkpoint {from_checkpoint} was given but Isaac Lab never "
            "asked for a checkpoint path, so nothing was loaded and this run "
            "trained from random weights. It is NOT a fine-tune and must not be "
            "recorded as one. Most likely `--resume` stopped reaching the agent "
            "config; re-read the trainer's resume branch."
        )


if __name__ == "__main__":
    main()
