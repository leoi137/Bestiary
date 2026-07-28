# 013 — What an observation is, and why its width is a one-way door

**One sentence:** The observation is the fixed-length array of floats the
policy is handed at every control step — the robot's entire world, since it
knows nothing else — and its *length* is welded into the first weight matrix of
every network trained on it, so changing the length does not make old policies
worse, it makes them impossible to load.

Assumes [008 — what a policy is](008-what-a-policy-is.md).

## The idea

Twenty times a second the environment builds one flat NumPy array and hands it
to the policy. That array is the observation. There is no other channel: no
image, no memory of the last step, no privileged simulator state. If a quantity
is not in the array, the policy cannot condition on it, however obviously it
exists in the simulator.

Which is why what's *missing* is as important as what's there. The hound's
array starts at `qpos_no_world_xy_no_wheels` — the trunk's height and
orientation and the twelve leg angles — with the trunk's **world x and y
deliberately dropped**. The robot does not know where it is. Nor do the four
wheel angles appear, because a wheel angle grows without bound and a policy
trained on a number that only ever increases learns something that stops being
true at minute ten. Dropping them is a modelling decision, and it is the reason
the same policy works anywhere on the map.

Every entry's provenance is declared in one place,
[`src/bestiary/envs/obs_spec.py`](../../src/bestiary/envs/obs_spec.py): an
ordered tuple of named terms with sizes. The width is `sum(term.size)`, and
`_get_obs` validates the vector it built against that same declaration and
*raises* on a mismatch. There is no second formula anywhere that could drift.

## The math

Read live off the envs and the checkpoints by
[`scripts/013_observation_width_math.py`](scripts/013_observation_width_math.py):

    HoundDesert-v0                      hash 11093686ef09fe13
        qpos_no_world_xy_no_wheels     17    trunk z + quaternion + 12 leg angles
        qvel                           22    trunk + joint velocities, wheels included
        cfrc_ext                      102    6 contact force/torque per non-world body
        command_reserved                3    [vx, vy, yaw] — zero-filled today
        height_reserved                25    5x5 terrain scan — zero-filled today
        sum                           169

    Spyder-v0    17 + 18 + 78                =  113     hash a97346acbaf6150f

All six Hound envs — flat, desert, torque, PD, and both tracking variants —
report width **169** and the identical hash. Now the first layer. The actor
begins with `Linear(obs, 256)`: a matrix `W1` of shape `(256, obs)` and a bias
of 256.

- `obs` — the observation width, a count of floats, one column of `W1` each
- 256 — the hidden layer width, one row each

Measured inside the committed checkpoints, not multiplied out in prose:

    run                       W1 shape   weights   bias    actor params
    spyder_walk_v3          (256, 113)     28928    256          101144
    hound_pd_desert_v0      (256, 169)     43264    256          117536

One extra observation value costs exactly 256 weights — one new column. The
113 → 169 difference is 56 columns, **14,336 weights that do not exist** in the
narrower checkpoint. `SAC.load()` does not pad them or guess them; the tensor
shapes disagree and it raises.

**Physically: the width of the observation is a physical dimension of the
brain, not a setting. Adding a sensor is not an upgrade to an existing robot,
it is a different robot with a different skull.**

The worse case is the one that *doesn't* raise. Take Spyder's declared terms
and swap the first two — same three terms, same 113 values, a checkpoint that
loads perfectly and a policy silently reading joint velocities where joint
angles used to be. That's why the spec hashes rather than merely counting:

    declared order            width 113   hash a97346acbaf6150f
    first two terms swapped   width 113   hash 7a50c72a606a6f06

`train.py::_record_or_verify_obs_spec` pins that hash into `config.json` on the
first launch and refuses any resume whose hash moved.

## Where it bites here

`runs/hound_desert_test150k/` — its actor's first layer is `(256, 141)` while
`HoundDesert-v0` is 169 today. Every other run on disk loads; that one is dead,
and no amount of retraining recovers it, because the checkpoint is the only
copy of what it learned. It was orphaned before the hash existed, which is why
it took git archaeology to work out what had happened rather than one line in
`config.json`. Only three runs carry a recorded spec at all; the rest predate
the instrument and say so, rather than being back-filled with a guess.

## If you want to go deeper

`src/bestiary/envs/obs_spec.py` — the module docstring is the argument for why
the declaration exists, written the day the door was walked through.
