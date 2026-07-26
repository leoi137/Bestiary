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
3. **Training is authorized, within a budget.** A run may be started without
   asking first, subject to *all* of:
   - **GPU pre-flight.** `nvidia-smi --query-gpu=memory.free
     --format=csv,noheader,nounits` reports **≥ 6500 MiB free**. Below that,
     something else is using the card: do not launch, do not queue, do not
     poll — do non-GPU work instead.
   - **One run at a time, machine-wide.** Never two, not even two small ones.
   - **A declared wall-clock ceiling**, written down before launch. A run that
     exceeds it is stopped, not extended — checkpoints save in a `finally`
     block, so stopping is cheap and resuming is one command.
   - **Resource ceilings:** ≤6000 MiB VRAM, ≤6 env workers, ≤6 torch threads
     (`OMP_NUM_THREADS=6`), ≤12 GiB RAM, ≤60 GB total in `runs/`.
   - **No long run through an open one-way door.** If the run costs more than
     ~1 GPU-hour and the observation width is not locked, lock it first — see
     the invariant below.
   - **Disk checked first.** A run needs ~3 GB for its replay buffer.
4. **The machine is shared.** Another long-running workload uses this GPU and
   these cores. Never assume you have the machine — the ceilings above are
   roughly half of what exists, deliberately.
5. **Git touches only these two repositories.** Every `git` invocation must
   target `Bestiary/` or `Scriptorium/`. Never run git against any other
   repository on this machine, never change global git config, and never
   commit from a directory you have not confirmed is one of the two. There is
   no task here that requires otherwise.

## Commit discipline

Use the `commit-push` skill every time. The standard, repeated from
`../CLAUDE.md`:

- **One commit = one independent, revertable intent.** Test: reverting only
  this commit must leave the tree correct.
- **Commit when a unit finishes, not when the session does.** Never
  accumulate 30 changed files and reconstruct history from the pile.
- **Mechanical and semantic changes never share a commit.** `git mv`-only
  commits carry no content edits.
- **One commit per ledger append, per learning, per lesson, per episode.**
  Those are the units the record is read in.
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
`ROADMAP.md`; the reward and observation spec that gates the next serious run
is `research/CORE_PLAN.md`, and it is **not yet applied**.

## Environment

```bash
source venv/bin/activate          # python is not on PATH without it
pip install -e . --no-deps        # already done; --no-deps protects pinned versions
```

GPU training assumes CUDA 12.1 (torch 2.5.1+cu121); `DEVICE = "cuda"` is
hardcoded in the trainer. Watching a policy runs on CPU.

Hardware: one RTX 2080 (8 GB VRAM), i9-9900K (8c/16t), 31 GiB RAM.

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
research/              learnings/, decisions/, episodes/, ledger.jsonl, CORE_PLAN.md
docs/lessons/          the curriculum: one idea per page, from scratch
docs/theory/           the deep notes: math written when it is load-bearing
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

# Guards — the lessons this project already paid for, as assertions.
# --fast runs in well under a second, so it gates every training launch.
python -m bestiary.guards --fast && python -m bestiary.train.train ...
python -m bestiary.guards                  # everything, ~11 s
python -m bestiary.guards --json           # machine-readable

# How well calibrated our predictions have been
python -m bestiary.record.calibration

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
fail silently when a session is launched from an arbitrary directory.

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

**The observation width is a one-way door.** The actor's first layer is
`Linear(obs, 256)`, so changing the observation list makes every existing
checkpoint fail to load — not degrade, *fail*. Spyder is at 113 today and
`research/CORE_PLAN.md` locks it at 141 with reserved command and height
slots. Never start a multi-hour run while that width is still unsettled.

**Two checkpoints per run.** `<prefix>_sac.zip` (latest, used to resume) and
`<prefix>_sac_best.zip` (only overwritten when an eval beats the prior best,
used for watching). Saving happens in a `finally` block so an interrupted run
stays resumable.

**The replay buffer is resume-only.** `ant_buffer.pkl` is ~2.6 GB per run and
is 94% of everything under `runs/`. Once a run has finished and its ledger row
is written, its buffer can be deleted; the two `.zip` checkpoints, `ant_tb/`,
and `config.json` are the run. Never delete those.

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
  must be readable by a human who is learning, not just by a model. A learning
  that is later falsified is **superseded by a new numbered learning**, never
  edited and never deleted.
- `research/decisions/` — a decision plus the **trigger that would reverse
  it**, so a settled question is not re-litigated every few weeks.
- `research/episodes/` — one file per research cycle: what was tried, what was
  predicted beforehand, what happened.
- `research/ledger.jsonl` — **append-only**, one row per finished run. Never
  rewrite this file; an appending process cannot lose what is already there,
  a rewriting one can lose all of it on a crash.
- `research/calibration.jsonl` — one row per prediction, with its stated
  probability and its outcome. Append-only. Scored by
  `python -m bestiary.record.calibration`.
- `research/nulls.jsonl` — one line per dead end already paid for, with the
  condition that would make it worth retrying. Append-only.
- `research/anomalies.jsonl` — one line per surprising thing noticed and not
  explained. Append-only, `status` may change.
- `src/bestiary/guards/` — the lessons that became assertions. **Any learning
  that can be expressed as a check must also become a guard**, and the
  learning's front matter names it. Prose depends on someone reading it at the
  right moment; a guard depends on nothing.
- `docs/lessons/` — the curriculum. One idea per page, explained from scratch,
  with the equation worked on a real number from this repo.

`research/MEMORY.md` explains how these fit together as one system. Read it
before adding a new kind of artifact.

Three rules make this a record rather than a diary:

**Write predictions before results are known.** The habit is what makes the
record evidence rather than narration.

**The seed rule — no effect is claimed from one run.** A comparison needs ≥3
seeds per arm, reported as mean and spread, with exactly one variable changed
between arms. Between-seed variance in SAC is comparable to most effects worth
claiming. A single-seed result is a **probe**, written up as provisional, never
as a finding. The PD-versus-torque comparison in episode 003 is provisional on
both counts — one seed per arm, and the step budgets differed too.

**The number rule — no number enters the record unless code computed it.**
Every figure in a ledger row, a learning, an episode, or a plan comes from a
run log, a check script, a measurement command, or a small script committed
alongside it. Arithmetic done in prose is fluent, checkable, and unchecked; if
a calculation is worth recording it is worth four lines of Python.

## Delegating

Use a subagent when work would flood the main context with material not worth
keeping: long run logs, literature sweeps, multi-file surveys. Ask for the
conclusion and the numbers, never the file contents, and keep the decision in
the main thread.

**Always Opus 5** — pass `model: "opus"` explicitly, including for work that
looks mechanical. The product here is judgement, and a cheaper model returns
confident wrong numbers into an append-only record that later work trusts
without re-deriving.
