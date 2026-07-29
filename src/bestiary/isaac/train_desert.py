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
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

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

    train_rsl_rl.run(argv)


if __name__ == "__main__":
    main()
