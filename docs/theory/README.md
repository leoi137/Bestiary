# Theory

The teaching track. One note per idea, written **when the idea becomes
load-bearing** — not upfront, and not as a survey.

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

- `pd-control.md` — PD position targets: the control law, why stiffness and
  damping are chosen as a ratio, and why emitting positions instead of
  torques removes pose-holding from what the policy must learn.
- `torque-to-weight.md` — what the ratio measures, why the hound's 2.17 N·m/N
  makes locomotion hard, and how it compares to the spider and to Ant-v5.
- `sac.md` — Soft Actor-Critic: the entropy term, what `ent_coef` collapsing
  early actually indicates, and why the critic cannot be warm-started across
  a reward change (see learning 002).
- `heightfields.md` — how MuJoCo stores terrain, why cell size relative to
  wheel radius matters, and where the hound's passive backward creep comes
  from.

Nothing here yet. The first note lands with the PD-targets work.
