# Theory

The deep notes. One note per idea, written **when the idea becomes
load-bearing** — not upfront, and not as a survey.

**Start at [`../lessons/`](../lessons/README.md) instead if you are learning
the field.** A lesson is one page that answers *what is this thing*; a theory
note is the room behind that doorway, and assumes you now want the full
derivation. Lessons link into these notes; these notes do not link back.

The rule exists because theory learned in the abstract does not stick and
does not get used. Theory learned the week it decides something does both.
So: when the project is about to switch from torque control to PD position
targets, that is when `pd-control.md` gets written, with the actual stiffness
and damping values we are about to use.

## Standard

Same as `../../research/learnings/README.md`, because the audience is the
same — someone building the knowledge, not someone who already has it:

- Plain English. Define jargon the first time it appears.
- Write the equation. Define every symbol, with units.
- Work it with real numbers from this project, not a toy example.
- One sentence in plain English after the algebra saying what it *means*
  physically.

Where a note explains something the codebase does, link the file and line so
the theory and the implementation stay tied together.

## Planned, in the order they will be needed

- ~~`pd-control.md`~~ — **written 2026-07-25** with the PD-targets work.
  [The control law](pd-control.md), the gains checked against this robot's
  real inertia, and the Nyquist argument for why the PD loop runs at physics
  rate rather than policy rate.
- `torque-to-weight.md` — what the ratio measures, why the hound's 2.17 N·m/N
  makes locomotion hard, and how it compares to the spider and to Ant-v5.
- `sac.md` — Soft Actor-Critic: the entropy term, what `ent_coef` collapsing
  early actually indicates, and why the critic cannot be warm-started across
  a reward change (see learning 002).
- `heightfields.md` — how MuJoCo stores terrain, why cell size relative to
  wheel radius matters, and where the hound's passive backward creep comes
  from.

The remaining three are unwritten by design. Each lands when the project
needs it to decide something, not before.
