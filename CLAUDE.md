# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## Hard rules

These are repeated from `../CLAUDE.md` on purpose, so they survive a clone.

1. **Scope.** Only `Bestiary/` and `Scriptorium/` may be read or written.
   Nothing else on the machine is in scope. If a task seems to need a file
   outside those two trees, stop and ask.
2. **Public vs private.** Physics, code, robots, results, lessons, and
   teaching notes live here. Anything naming an agency, a customer, or a
   dollar figure goes in Scriptorium. When unsure: Scriptorium.
3. **No unattended training.** Never start a training run unless the current
   instruction explicitly asks for one. Runs take hours and hold the only GPU
   (a single RTX 2080).

## Commit discipline

Use the `commit-push` skill every time. The standard, repeated from
`../CLAUDE.md`:

- **One commit = one independent, revertable intent.** Test: reverting only
  this commit must leave the tree correct.
- **Commit when a unit finishes, not when the session does.** Never
  accumulate 30 changed files and reconstruct history from the pile.
- **Mechanical and semantic changes never share a commit.** `git mv`-only
  commits carry no content edits.
- **In the research loop: one commit per learning, per decision, per episode,
  per ledger append.** Those are the units the record is read in.
- **When torn, split.** Over-splitting squashes away; under-splitting cannot
  be undone.

---

## What this is

A bestiary of legged robots — built, trained, and documented in simulation.
Two custom machines so far, both authored as MuJoCo MJCF by generator scripts
rather than hand-written XML:

- **Spyder** — 12-DoF spider.
- **Hound** — 16-DoF wheel-legged dog on Unitree Go2 kinematics and masses,
  with a driven hub wheel replacing each foot.

Training is SAC via Stable-Baselines3. The long-term direction is in
`ROADMAP.md`; the current bottleneck analysis is in
`research/episodes/001-hound-throughput.md`.

## Environment

```bash
source venv/bin/activate          # python is not on PATH without it
pip install -e . --no-deps        # already done; --no-deps protects pinned versions
```

GPU training assumes CUDA 12.1 (torch 2.5.1+cu121); `DEVICE = "cuda"` is
hardcoded in the trainer. Watching a policy runs on CPU.

## Layout

```
src/bestiary/          the importable library
  paths.py             EVERY path in the project resolves from here
  envs/                gym envs; importing registers the four env ids
  rewards/             reward-shaping wrappers, selected by --wrapper
  terrain/             generate.py writes heightfields; field.py reads them back
  train/               train.py, watch.py
  robots/<name>/       build.py (MJCF generator), check.py (assertions), CARD.md
concepts/anvil/        Blender concept art (ANVIL siege walker) — not RL
research/              learnings/, decisions/, episodes/, ledger.jsonl
docs/theory/           the teaching track: math written when it is load-bearing
assets/                GENERATED output — model XMLs, meshes, terrain, figures
runs/                  per-run artifacts; gitignored, tens of GB
```

## Commands

```bash
# Fresh run — --env is given ONCE at creation, then pinned in config.json
python -m bestiary.train.train --run-name hound_v1 --env HoundDesert-v0 --seed 0 --steps 1_000_000

# Resume — same --run-name; env/wrapper/seed come from config.json
python -m bestiary.train.train --run-name hound_v1 --steps 2_000_000

# Watch a trained policy
python -m bestiary.train.watch --run hound_v1            # best-eval checkpoint
python -m bestiary.train.watch --run hound_v1 --latest

# Validate a robot (the regression oracle — run this after ANY physics change)
python -m bestiary.robots.hound.check      # 38 assertions
python -m bestiary.robots.spyder.check     # shell is decorative, physics unchanged

# Lint — REQUIRED after any refactor. Catches the bug class the robot checks
# structurally cannot see (see research/learnings/006).
python -m ruff check --select F src/ concepts/

# Regenerate a robot model or the terrain
python -m bestiary.robots.hound.build
python -m bestiary.terrain.generate
```

`--steps` is **per-invocation** (additional steps), not a cumulative target.

## Invariants — do not break these

**`paths.py` is the only place that builds paths.** No module may use
`Path(__file__).parent.parent` chains, and nothing may be relative to the
current working directory. Both patterns broke during the refactor, and both
fail silently under an unattended loop launched from an arbitrary directory.

**Model XMLs must stay in `assets/`.** MuJoCo resolves
`<mesh file="meshes/...">` and `<hfield file="terrain/...">` relative to the
XML's own directory. Moving a model into its robot folder breaks asset
resolution with a bare file-not-found. Robot folders hold *source*; `assets/`
holds *generated output*.

**`config.json` wins on resume.** The first invocation writes `env_id`,
`wrapper`, `wrapper_kwargs`, `seed`, and hyperparameters into
`runs/<name>/config.json`. Every resume reads it back and **ignores** any
conflicting `--env`/`--wrapper`/`--seed` on the CLI. Changing env or reward
semantics mid-run would contaminate a replay buffer filled under different
dynamics. Resume vs. fresh is decided purely by whether `config.json` exists.

**Two checkpoints per run.** `<prefix>_sac.zip` (latest, used to resume) and
`<prefix>_sac_best.zip` (only overwritten when an eval beats the prior best,
used for watching). Saving happens in a `finally` block so an interrupted run
stays resumable.

**Artifact filenames are hardcoded with an `ant_` prefix** (`ant_sac.zip`,
`ant_buffer.pkl`) even for non-Ant runs. Cosmetic, kept so existing runs do
not break.

**The robot checks are a ROBOT oracle, not a repository oracle.** 38/38 green
means the machine is unchanged. It says nothing about `train.py` or
`watch.py` — nothing in the suite imports them. `--help` is not coverage
either: argparse exits before most of `main()` runs. To check an entry point,
actually run it (`--steps 2000` takes ~16 s). See learning 006.

**`FootContactRewardWrapper` is Ant-only.** It resolves four `*_ankle_geom`
bodies at init and raises if it does not find exactly four. Do not apply it to
2-legged envs.

**The Blender scripts are not importable here.** `concepts/anvil/*.py` and
`robots/spyder/build_mesh.py` run under Blender's own Python, which has no
`bestiary` package. They keep cwd-relative asset paths and must be invoked
from the repo root.

## Research conventions

- `research/learnings/` — one file per lesson, written when something
  surprises us. Format and standard are in that folder's `README.md`. These
  must be readable by a human who is learning, not just by a model.
- `research/decisions/` — a decision plus the **trigger that would reverse
  it**, so the loop does not re-litigate settled questions every few weeks.
- `research/episodes/` — one file per loop cycle: what was tried, what was
  predicted beforehand, what happened.
- `research/ledger.jsonl` — **append-only**, one row per finished run. Never
  rewrite this file; an unattended process that rewrites can clobber, one
  that appends cannot.

Write predictions *before* results are known. The habit is what makes the
record evidence rather than narration.

## The loop's skills

Version-controlled in `.claude/skills/`, so they travel with a clone and
change under review rather than drifting.

| Skill | Use |
|---|---|
| `run-episode` | one bounded cycle: read state → one experiment → predict → record → next episode. Also the body of a `/loop`. |
| `write-learning` | write a `research/learnings/` entry to standard: plain English, real math with real numbers, an explicit way to be wrong. |
| `robotics-research` | investigate a robotics/RL/simulation question against primary sources; land it as a decision with a trigger, or a theory note. |

`run-episode` will not start a training run without explicit authorization —
hard rule 3 above. It prepares the run and hands back instead.

> **Project skills are registered when a session starts.** A skill created
> mid-session is not invocable until the next one — creating the three above
> and then trying to call `run-episode` in the same session fails with
> "Unknown skill". Follow the contract in the file by hand until the session
> restarts. Found the first time the loop was exercised, 2026-07-25.

**Delegate to a subagent** when work would flood the main context with
material not worth keeping: long run logs, literature sweeps, multi-file
surveys. Ask for the conclusion and the numbers, never the file contents, and
keep the decision in the main thread.
