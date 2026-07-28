# 0002 — The backward eval cell is off-distribution; report it separately, do not delete it

**Date:** 2026-07-28 · **Status:** accepted · **Robot:** hound · **Envs:**
`HoundPDTrackDesert-v0`, `HoundPDTrackRelDesert-v0`

**Decided before the result it affects was known.** `hound_track_rel_s1` was
still training when this was written and committed. That ordering is the whole
point: a rule about which cells count, chosen after seeing whether the cell
helped or hurt, is not a rule but a preference.

## The decision

`EVAL_GRID`'s `(-0.3, 0.0, 0.0)` cell **stays in the protocol and comes out of
the headline**. It is reported as its own number — an extrapolation probe —
exactly as the stop cell `(0, 0, 0)` already is, and for the same reason: it
answers a different question from the other five.

Until `track_eval` is changed to do this itself, **every cycle quoting a
`drive_grid_mean` quotes it twice**: over all six drive cells, and over the
five in-distribution ones. Neither number is allowed to appear alone.

## Why we asked

`envs/hound_track.py:100` sets `VX_MIN_BACKWARD = 0.4` — the env never samples
a backward command smaller than 0.4 m/s, because below that the standing
machine's own creep cancels the tracking error and the command is free money.
`record/track_eval.py:104` scores the policy at **−0.3**.

So one of six drive cells asks the policy for a command that has never once
appeared in training. This has been true since commit `a4e7ef5` raised the
floor; the grid was not raised with it.

It is not a rounding error. In `track_rel_zero_action.json` the do-nothing
control scores **+31.36** in that single cell against a six-cell
`drive_grid_mean` of **3.91**. Removing it takes the baseline to about
**−1.58**. The bar a policy must clear is therefore set almost entirely by a
command the policy was never taught, and a machine that simply stands still
collects most of it — the creep-cancels-error effect the 0.4 floor exists to
prevent, reintroduced through the measuring instrument.

## Why not the two obvious alternatives

**Delete the cell / move it to −0.4.** Cleanest-looking and wrong twice over.
It discards the only measurement in the protocol of what happens off the
training distribution, which for a machine meant to work on ground nobody has
surveyed is not a nuisance variable — it is closer to the actual question than
the in-distribution cells are. And it silently redefines `drive_grid_mean`, so
every published number under that name becomes incomparable with no marker in
the data saying so.

**Leave it and say nothing.** This is the status quo and it has already
distorted a conclusion: row 4's headline `−6.48 against zero action's 55.73`
carries this cell inside both arms, uninspected.

## What this costs

`drive_grid_mean` under the new reporting is not the same statistic as the one
in ledger row 4, and the frozen `track_rel_zero_action.json` was measured under
the old grouping. **Nothing needs re-measuring**: the JSON records per-cell
`mean` values, so the five-cell aggregate is recomputable from the artifact
already on disk, for both arms, exactly. That is why this decision is cheap
today and would not have been if the per-cell numbers had not been kept.

The code change to `track_eval` — emitting `drive_grid_mean_in_dist` and
`extrapolation_cell` alongside the existing key, rather than in place of it —
is deliberately **not** made in the same cycle as a harvest that was
pre-registered against the current output. Changing the instrument between the
prediction and the reading is the confound this project keeps writing learnings
about.

## The trigger that would reverse this

- **The env starts sampling backward commands at |vx| < 0.4** — then the cell
  is in-distribution, there is no extrapolation probe any more, and the split
  reporting is noise. Reverse it.
- **A second off-distribution cell is added deliberately** — then one-off
  special-casing stops scaling and the grid needs an explicit
  in-distribution/extrapolation partition as data, not a rule in prose.
- **`VX_MIN_BACKWARD` moves for a reason unrelated to creep** — the 0.4 figure
  is derived from the standing-drift cancellation argument in
  `envs/track_constants.py`. If that derivation changes, this decision is
  resting on a number that no longer means what it meant.

## How we would know this was the wrong call

If, across the next three harvests, the five-cell and six-cell headlines never
disagree in **sign or direction**, then the split cost real prose and bought
nothing, and the cell should simply be dropped. One disagreement in sign
justifies the decision permanently.
