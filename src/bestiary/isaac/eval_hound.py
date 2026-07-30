"""Judge a trained Isaac Lab Hound policy, cell by cell. The oracle for a run.

    PYTHONPATH=src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.eval_hound \\
        --checkpoint logs/rsl_rl/anymal_c_rough/<run>/model_1499.pt --label arm1_s1

Exit status is 0 when the battery ran and wrote its JSON, 1 on any failure, so it
can gate a ledger row. Writes `research/measurements/isaac_hound_<label>.json`
and prints the same numbers as a table.


WHY THIS EXISTS
---------------
`record/track_eval.py` does this for the MuJoCo track. Nothing did it for Isaac,
so an Isaac run's only public number was its aggregate return -- and this
repository's own record says three separate times that an aggregate return is not
evidence:

  * `research/learnings/011` -- the aggregate hid that driving cost 12.6x what it
    earned. Control cost was 105.5% of the entire policy-versus-control gap.
  * `research/learnings/015` -- a policy cleared the 5x success bar by 0.89% while
    running ONE FIXED TROT under every command: 0.271 m/s when 0.5 was commanded,
    0.309 m/s when 0.8 was. 98.8% of its win came from one grid cell, and it lost
    to doing nothing in three of six. The aggregate scored all of that as success.
  * `research/decisions/0005` -- the binding freeride on THIS command geometry is
    **point-and-park**: yaw to the commanded heading, then hold still. Under
    heading mode the machine's own yaw command then decays toward zero, so it
    collects the yaw kernel for standing there. The refutation that established
    this put the fraction of a competent driver's net that it captures close to
    all of it; that arithmetic has no committed script, so no figure for it is
    quoted here. What this battery measures is the BEHAVIOUR, per cell, so the
    freeride can be recognised without needing the estimate.

So the deliverable of this file is a **per-cell table**, and a single aggregate is
deliberately not offered as a headline. Every number below is computed from actual
rollouts of the checkpoint named in the JSON.


WHAT IT MEASURES, AND WHICH FAILURE EACH ONE CATCHES
----------------------------------------------------
1. **A fixed-command grid.** 13 cells, N deterministic episodes each (one per
   env), THE SAME SEED IN EVERY CELL so the only thing that differs between two
   rows is the command. Forward slow/mid/fast, backward, lateral both signs, pure
   yaw both signs, a mirrored forward+yaw pair, two heading-mode cells, and stop.
   Reported per cell: achieved v_x, v_y, yaw rate, planar speed, net displacement,
   episode length, terminations, reward per step, the reward's own tracking
   kernels, and leg joint acceleration rms.

2. **`vx_span_ratio`** -- ported from `src/bestiary/guards/command_independence.py`
   (which took it from `research/learnings/015`). It is the statistic that caught
   the one-trot policy after a regression slope failed to. Self-tested against
   two synthetic cell-sets whose answers are known by construction, so the
   formula is never reported without having first produced 1.0 and 0.0.

3. **The do-nothing control.** The same grid, the same seeds, zero action. Zero
   action is not "limp" on this body: `use_default_offset=True` on the leg action
   term makes a zero action the solved standing stance, so the control arm STANDS.
   `research/learnings/011` and `guards/parked_detector.py` are both about the
   same two-minute check catching a reward bug, and it has caught one twice, so it
   is not optional. A cell where the policy loses to this is named as such.

4. **A point-and-park discriminator.** See the block below; the statistic and its
   limits are stated there rather than implied.

5. **Leg joint acceleration rms.** `check_hound.check_reward_budget_against_011_and_015`
   prices `dof_acc_l2` at 12.5% of achievable income from an **assumed** 250
   rad/s^2 rms, labelled `[ASSUMED]` because no trained policy existed. A trained
   policy is the instrument that measures it. Reported here as the measured rms,
   the measured per-step charge read out of the reward manager itself, and the
   ratio to the assumption.


THE POINT-AND-PARK DISCRIMINATOR, AND WHAT IT CANNOT DO
-------------------------------------------------------
The statistic is `park_fraction`: on a cell whose commanded planar speed is at
least `MIN_COMMANDED_SPEED_MS`, the fraction of the episode's policy steps whose
achieved planar speed in the body frame is below `PARK_SPEED_MS`. It is reported
beside `mean_speed` and `displacement_m` (net world-frame planar displacement) and,
on the two heading cells, beside the mean |heading error| in the first and last
fifth of the episode.

The three together separate the two behaviours the record cares about:

    tracks the command   heading error falls, speed rises to the command,
                         displacement is metres, park_fraction near 0
    point-and-park       heading error falls, speed stays near 0,
                         displacement near 0, park_fraction near 1

**What it cannot distinguish.** It reads behaviour, not intent, so:

  * A policy that PARKS BECAUSE IT PAYS and a policy that NEVER LEARNED TO DRIVE
    are identical under it. Both stand. Deciding between them needs a training
    curve or a reward-decomposition argument, not this battery.
  * Parking from being STUCK -- a hull caught on terrain, wheels spinning without
    traction -- looks the same as parking by choice. Wheel joint velocity is not
    read here, so slip is invisible.
  * On the rate-mode cells there is no heading target at all, so "aimed once"
    cannot be separated from "never aimed"; only the two heading cells can see
    that half of the behaviour.
  * It says nothing about WHERE the machine went. A policy driving in a tight
    circle has high body-frame speed and near-zero displacement, which reads as
    tracking on `park_fraction` and as parking on `displacement_m`. That
    disagreement is informative and is printed rather than collapsed.
  * `achieved_speed` is the PLANAR NORM, so lateral drift counts towards it. A
    machine skidding sideways down a dune face scores speed it was never asked
    for, and on this body -- four wheels with fixed spin axes -- unwanted `v_y` is
    the expected failure rather than an exotic one. `speed_tracking_ratio` is
    therefore an upper bound on tracking and not a measure of it;
    `forward_tracking_ratio`, which uses `achieved_vx` against `commanded_vx` on
    the forward-only cells, is the one that cannot be inflated that way, and
    `achieved_vy` is printed on every row so the drift is visible.


THE ONE PLACE THIS DEPARTS FROM THE TRAINING CONFIG, DELIBERATELY
-----------------------------------------------------------------
Training ran `heading_command=True`: the yaw command is not sampled, it is
`clip(0.5 * heading_error, ang_vel_z_range)`, recomputed every step. Under that
mode a machine that has finished turning has a yaw command of zero, so **no fixed
nonzero yaw rate can be commanded at all** and the yaw axis cannot be tested.

Eleven of the thirteen cells therefore run `heading_command=False`, which makes
the whole 3-vector an exact constant for the whole episode. This is not
out-of-distribution for the policy: the observation is the same 3-vector either
way, and under heading mode a fresh episode's yaw command starts at
`clip(0.5 * U(-pi, pi), +-1)`, so constant yaw commands inside +-1 rad/s are
values the policy saw during training. The two `head_*` cells keep heading mode,
because that is the mode in which parking is REWARDED, and they are what makes
the freeride legible.

Every other departure from the training config is in the service of determinism
and is listed in `eval_env_cfg` with its reason.


ONE FRAME NOTE, FOR CROSS-TRACK COMPARABILITY
---------------------------------------------
`achieved_vx` / `achieved_vy` here are `root_lin_vel_b`, the FULL BODY FRAME, and
`achieved_wz` is `root_ang_vel_b[:, 2]`. That is not a choice this file made: it is
exactly what `isaaclab/envs/mdp/rewards.py:track_lin_vel_xy_exp` compares against
the command, so the instrument reads the same quantity the reward paid on.

The MuJoCo track uses the yaw-only HEADING frame instead, and
`src/bestiary/guards/tracking_frame.py` calls that choice load-bearing. The two
are not the same number: they differ by the trunk's roll and pitch, so on ground
with 5.05 m of relief a body-frame reading also picks up a slice of the vertical
velocity. This is NOT the failure that guard exists to prevent -- that failure is
a WORLD-frame implementation, which caps the kernel near 0.5 on every turning
segment, and Isaac Lab's is correctly body-relative. But an Isaac `achieved_vx`
and a MuJoCo `achieved_vx` are measured in different frames and should not be
differenced without saying so.


WHAT IT DOES NOT DO
-------------------
  * **It does not grade.** There is no pass/fail threshold on tracking anywhere in
    this file, on purpose. `guards/command_independence.py` owns the one bar this
    project has (`vx_span_ratio >= 0.25`) and owns it against committed
    measurements; a second, differently-shaped copy of that judgement here would
    be a threshold nobody reconciled.
  * **It measures one checkpoint, not a run.** `learnings/008` and `010`: a
    selected checkpoint is an argmax over noisy evals. Point it at the final model
    of every seed and read the three together, which is what the seed rule asks.
  * **It cannot see the training distribution.** The grid is flat; training was
    not. `learnings/015`'s last lesson is that a flat grid mean measures the grid.
    That is why the per-cell table is the output and the mean is not.


TWO LAUNCH CONDITIONS, BOTH OF WHICH FAIL QUIETLY
-------------------------------------------------
`OPENBLAS_NUM_THREADS=1` is mandatory on the remote GPU host: without it an
Omniverse Kit plugin's `fork()` deadlocks inside OpenBLAS's threadpool and the
process dies in Kit's breakpad handler rather than in anything this file wrote.
`research/learnings/016` has the backtrace and the reasoning.

`isaaclab.sh` silently falls back to the system interpreter when `VIRTUAL_ENV` is
unset, so a run can look normal while importing a different Isaac Lab than the one
supplying the physics. Check its first `[INFO] Using Python:` line before trusting
any number below it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# Constants. Every one names where it came from.
# ---------------------------------------------------------------------------

#: Body-frame planar speed below which a step counts as "parked", in m/s.
#:
#: 0.10 m/s (0.33 ft/s) is a third of the smallest forward command in the grid and
#: an eighth of the largest. It is chosen to sit below anything that could be
#: called driving and above the drift a standing machine shows on a 5.05 m-relief
#: desert -- and the do-nothing control arm is what checks the second half of that
#: claim, since the control's own `park_fraction` is printed on every cell. A
#: threshold the control does not reach is a threshold that measures nothing,
#: which is `guards/parked_detector.py`'s whole lesson.
PARK_SPEED_MS = 0.10

#: Commanded planar speed at or above which `park_fraction` is meaningful, m/s.
#: Below it, standing still is the CORRECT answer and a park fraction of 1.0 is a
#: pass rather than a failure -- exactly the confound `guards/parked_detector.py`
#: found when a pure-turn cell carried an "is it driving?" mean over the line.
MIN_COMMANDED_SPEED_MS = 0.20

#: Leg joint acceleration rms the reward budget ASSUMES, rad/s^2.
#:
#: From `check_hound.check_reward_budget_against_011_and_015`, where it is one of
#: the `[ASSUMED]` rows and the largest guess in the table: at -2.5e-7 on twelve
#: joints it prices `dof_acc_l2` at 12.5% of achievable income. Carried here only
#: to be divided into the measured number.
ASSUMED_LEG_ACC_RMS = 250.0

#: `guards/command_independence.py`'s bar, reported for reference and NOT asserted
#: here. That guard owns the judgement, against committed measurements; a second
#: copy of the same threshold in the instrument that produces them would be a
#: judgement made twice and reconciled never.
GUARD_MIN_SPAN_RATIO = 0.25

#: Command resampling interval imposed during eval, in seconds.
#:
#: Training resamples every 10 s inside a 20 s episode, so a training episode
#: carries TWO commands. A cell must carry one, or "achieved v_x under command c"
#: is an average over two different c. 1e6 s is longer than any episode.
NO_RESAMPLE_S = 1.0e6

#: Relative tolerance when comparing this eval's live config against the config
#: the checkpoint was trained under. Both sides are float64 round-trips of the
#: same literals through YAML, so anything above the float64 epsilon is a real
#: difference and not a formatting artifact.
CFG_RTOL = 1e-9

#: Absolute tolerance on the reward decomposition identity, per step.
#:
#: `RewardManager` accumulates in float32, so the sum of thirteen per-term rewards
#: differs from the total it returns by float32 rounding and nothing else. 1e-4 on
#: an episode-summed quantity of order 10 is roughly 1e-5 relative -- loose enough
#: for 1000 float32 additions, tight enough that a genuinely missing term (the
#: defect `research/anomalies.jsonl` row 36 records, a residual of 20% of the
#: control arm's baseline) cannot hide under it.
DECOMPOSITION_TOL = 1e-4


@dataclass(frozen=True, slots=True)
class Cell:
    """One command held constant for a whole episode.

    `heading is None` means rate mode: the 3-vector `(vx, vy, wz)` is exactly what
    the policy is shown, every step. `heading` set means the env's own heading
    controller supplies `wz` from the error to that absolute world heading, which
    is the mode the policy trained under and the mode in which parking is paid.
    """

    name: str
    vx: float
    vy: float
    wz: float
    heading: float | None = None
    note: str = ""

    @property
    def command(self) -> list[float]:
        return [self.vx, self.vy, self.wz]

    @property
    def commanded_speed(self) -> float:
        return math.hypot(self.vx, self.vy)


#: The grid. Ordered so the table reads top to bottom as "does bigger mean faster".
#:
#: `fwd_slow`/`fwd_mid`/`fwd_fast` are the three cells `vx_span_ratio` is computed
#: over: same zero lateral and zero yaw command, three different forward speeds
#: inside the trained +-1.0 m/s range. Two would be enough for the statistic to be
#: defined; three make a straight line distinguishable from a step.
#:
#: `lat_pos`/`lat_neg` are OUT OF DISTRIBUTION for arm 1, whose training command
#: had `lin_vel_y` collapsed to (0, 0) -- it was never asked to move sideways. They
#: are measured anyway because `research/decisions/0005` trigger 1 asks for exactly
#: this number ("a commanded pure-v_y rollout ... reports the largest sustained
#: body-frame |v_y| the machine can hold") and it is the cheapest gate in that
#: decision. The JSON labels them so nobody reads them as a tracking failure.
#:
#: `fwd_yaw_pos`/`fwd_yaw_neg` are a mirror pair. `learnings/015` found a 42.32
#: point gap across exactly this sign flip and read it as a built-in turning
#: handedness; a steering policy is roughly symmetric under it.
GRID: tuple[Cell, ...] = (
    Cell("stop", 0.0, 0.0, 0.0, note="all-zero command; standing is the right answer"),
    Cell("fwd_slow", 0.3, 0.0, 0.0, note="span cell"),
    Cell("fwd_mid", 0.6, 0.0, 0.0, note="span cell"),
    Cell("fwd_fast", 1.0, 0.0, 0.0, note="span cell; top of the trained range"),
    Cell("back", -0.6, 0.0, 0.0, note="sign of the command"),
    Cell("lat_pos", 0.0, 0.3, 0.0, note="OUT OF DISTRIBUTION for arm 1 (trained lin_vel_y = 0)"),
    Cell("lat_neg", 0.0, -0.3, 0.0, note="OUT OF DISTRIBUTION for arm 1 (trained lin_vel_y = 0)"),
    Cell("yaw_pos", 0.0, 0.0, 0.8, note="pure turn, rate mode"),
    Cell("yaw_neg", 0.0, 0.0, -0.8, note="pure turn, rate mode"),
    Cell("fwd_yaw_pos", 0.6, 0.0, 0.5, note="mirror pair with fwd_yaw_neg"),
    Cell("fwd_yaw_neg", 0.6, 0.0, -0.5, note="mirror pair with fwd_yaw_pos"),
    Cell("head_drive", 0.6, 0.0, 0.0, heading=0.0, note="HEADING MODE: drive while holding heading 0"),
    Cell("head_stop", 0.0, 0.0, 0.0, heading=0.0, note="HEADING MODE: aim at heading 0 and hold"),
)


# ---------------------------------------------------------------------------
# vx_span_ratio. Ported, not reinvented.
# ---------------------------------------------------------------------------
def span_ratio(cells: dict) -> tuple[float | None, int, str]:
    """(vx_span_ratio, cells used, one-line detail) over forward-only cells.

    **PORTED VERBATIM IN DEFINITION from `src/bestiary/guards/command_independence.py`,
    function `span_ratio`**, which took it from `research/learnings/015` ("What to
    do next time"). Same predicate on which cells qualify, same arithmetic, same
    `None` for undefined:

        vx_span_ratio = (max achieved_vx - min achieved_vx)
                      / (max commanded_vx - min commanded_vx)

    over the cells that command a positive forward speed with zero lateral and
    zero yaw command. 1.0 is a perfect tracker, 0.0 is one fixed gait, and a
    half-gain tracker reads 0.5. `hound_track_rel_s1` read 0.127.

    It is deliberately not a regression slope. A slope fitted across the backward
    cell is a SIGN detector: `learnings/015`'s `command_gain` read 0.382 against a
    0.05 bar on a policy whose forward speed barely moved with the command, purely
    because the machine crept backward on the one backward-commanded cell.

    `None` rather than 0.0 when fewer than two distinct qualifying commands were
    measured: a grid that never asked the question must not be reported as having
    failed it.
    """
    pts = [
        (c["commanded_vx"], c["achieved_vx"])
        for c in cells.values()
        if c["commanded_vx"] > 0 and c["command"][1] == 0 and c["command"][2] == 0
    ]
    if len({round(cmd, 6) for cmd, _ in pts}) < 2:
        return None, len(pts), f"only {len(pts)} forward-only cell(s); undefined"
    cmd_span = max(c for c, _ in pts) - min(c for c, _ in pts)
    ach_span = max(a for _, a in pts) - min(a for _, a in pts)
    return (
        ach_span / cmd_span,
        len(pts),
        f"achieved spread {ach_span:.3f} m/s over commanded spread {cmd_span:.3f} m/s",
    )


def self_test_span_ratio() -> None:
    """Produce 1.0 and 0.0 from cell-sets whose answers are known by construction.

    Runs before any rollout. `guards/command_independence.py` carries the same
    self-test and the reason is the same: a formula nobody ever watched produce a
    known value should not be the headline of a measurement. Raises rather than
    warning -- a battery whose own statistic is broken must not write a JSON.
    """

    def cells(pairs):
        return {
            str(i): {"command": [c, 0.0, 0.0], "commanded_vx": c, "achieved_vx": a}
            for i, (c, a) in enumerate(pairs)
        }

    perfect, _, _ = span_ratio(cells([(0.5, 0.5), (0.8, 0.8)]))
    frozen, _, _ = span_ratio(cells([(0.5, 0.27), (0.8, 0.27)]))
    undefined, _, _ = span_ratio(cells([(0.5, 0.27)]))
    if perfect is None or abs(perfect - 1.0) > 1e-12:
        raise AssertionError(f"span_ratio on a perfect tracker returned {perfect}, expected 1.0")
    if frozen is None or abs(frozen) > 1e-12:
        raise AssertionError(f"span_ratio on one fixed 0.27 m/s trot returned {frozen}, expected 0.0")
    if undefined is not None:
        raise AssertionError(f"span_ratio on one forward cell returned {undefined}, expected None")
    print(
        "[selftest] vx_span_ratio: perfect tracker -> 1.0, one fixed trot -> 0.0, "
        "single cell -> undefined",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _t(x):
    """An Isaac Lab 3.0 `.data.*` field as a torch tensor, whatever it arrives as.

    3.0 wraps these in a `ProxyArray` carrying both a warp and a torch view; the
    reward functions in `isaaclab/envs/mdp/rewards.py` all reach for `.torch`, so
    that is the view this instrument reads too. Reading the warp view instead
    would be a second, silently different copy of the same number.
    """
    if hasattr(x, "torch"):
        return x.torch
    return x


def _f(x) -> float:
    """A 0-d tensor or a python number as a float."""
    return float(x.item()) if hasattr(x, "item") else float(x)


def _tagged_yaml(path: Path) -> dict:
    """Load an Isaac Lab `params/env.yaml` without importing the classes it names.

    The dump carries `!!python/tuple` and `!!python/object/apply:builtins.slice`
    tags, so `safe_load` refuses it and `unsafe_load` would construct Isaac Lab
    objects -- which needs the exact class layout the dump was written under, i.e.
    the thing this comparison exists to stop assuming. Tuples become lists and
    every other tagged node becomes None: the fields this file compares are all
    plain scalars, lists and maps, so nothing load-bearing is discarded.
    """
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_multi_constructor(
        "tag:yaml.org,2002:python/tuple", lambda ldr, suffix, node: ldr.construct_sequence(node, deep=True)
    )
    _Loader.add_multi_constructor("", lambda ldr, suffix, node: None)
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)  # noqa: S506 -- _Loader is SafeLoader plus two tag stubs


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=CFG_RTOL, abs_tol=1e-12)
    return a == b


# ---------------------------------------------------------------------------
# The config the checkpoint was trained under, versus the one being stepped.
# ---------------------------------------------------------------------------
def training_fingerprint(env_yaml: dict) -> dict:
    """The load-bearing fields of a run's `params/env.yaml`.

    Only fields whose disagreement would make this battery's numbers describe a
    different machine, a different reward, or a different clock. The command
    ranges are NOT here: this eval overrides them on purpose, and they are
    reported separately.
    """
    obs = env_yaml["observations"]["policy"]
    return {
        "sim.dt": env_yaml["sim"]["dt"],
        "decimation": env_yaml["decimation"],
        "episode_length_s": env_yaml["episode_length_s"],
        "obs.policy.terms": [k for k, v in obs.items() if isinstance(v, dict) and "func" in v],
        "action.scales": {
            k: v.get("scale") for k, v in env_yaml["actions"].items() if isinstance(v, dict)
        },
        "reward.weights": {
            k: v["weight"] for k, v in env_yaml["rewards"].items() if isinstance(v, dict)
        },
        "terrain.horizontal_scale": env_yaml["scene"]["terrain"]["terrain_generator"]["horizontal_scale"],
        "terrain.num_rows": env_yaml["scene"]["terrain"]["terrain_generator"]["num_rows"],
        "terrain.num_cols": env_yaml["scene"]["terrain"]["terrain_generator"]["num_cols"],
        "terrain.max_init_terrain_level": env_yaml["scene"]["terrain"]["max_init_terrain_level"],
    }


def live_fingerprint(env, cfg) -> dict:
    """The same fields, read off the env that is actually being stepped.

    Read from the MANAGERS where a manager exists, not from the config: the config
    is what was asked for and the manager is what was built, and the whole point of
    the comparison is to catch the gap.
    """
    gen = cfg.scene.terrain.terrain_generator
    return {
        "sim.dt": cfg.sim.dt,
        "decimation": cfg.decimation,
        "episode_length_s": cfg.episode_length_s,
        "obs.policy.terms": list(env.observation_manager.active_terms["policy"]),
        "action.scales": {
            name: getattr(getattr(cfg.actions, name), "scale", None)
            for name in env.action_manager.active_terms
        },
        "reward.weights": {
            name: env.reward_manager.get_term_cfg(name).weight
            for name in env.reward_manager.active_terms
        },
        "terrain.horizontal_scale": gen.horizontal_scale,
        "terrain.num_rows": gen.num_rows,
        "terrain.num_cols": gen.num_cols,
        "terrain.max_init_terrain_level": cfg.scene.terrain.max_init_terrain_level,
    }


def compare_fingerprints(trained: dict, live: dict) -> dict:
    """Raise if the eval env differs from the training env where it must not.

    Loud, early, with both values in the message -- the engineering standard's
    rule, and the only shape that is useful here: a mismatch means every number
    this file is about to print describes something other than the run being
    judged, and printing them anyway is how a measurement becomes a lie nobody
    can find later.
    """
    diffs = []
    for key in sorted(set(trained) | set(live)):
        a, b = trained.get(key, "<absent>"), live.get(key, "<absent>")
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if not _close(a.get(sub, "<absent>"), b.get(sub, "<absent>")):
                    diffs.append(f"{key}[{sub}]: trained={a.get(sub, '<absent>')!r} live={b.get(sub, '<absent>')!r}")
        elif not _close(a, b):
            diffs.append(f"{key}: trained={a!r} live={b!r}")
    if diffs:
        raise AssertionError(
            "the env this battery is stepping is not the env the checkpoint was "
            "trained in, on "
            + str(len(diffs))
            + " load-bearing field(s):\n        "
            + "\n        ".join(diffs)
            + "\n      Every number below would describe a different machine, a "
            "different reward or a different clock. Fix the config or evaluate a "
            "checkpoint trained under this one."
        )
    return {"fields_compared": sorted(trained), "mismatches": []}


# ---------------------------------------------------------------------------
# The env.
# ---------------------------------------------------------------------------
def eval_env_cfg(num_envs: int, seed: int, device: str | None):
    """`HoundDesertEnvCfg` with everything stochastic or drifting turned off.

    The base is the TRAINING config, not `HoundDesertEnvCfg_PLAY`. Play exists to
    be watched: it swaps the terrain for a 3x3 grid at native 7.8 cm sampling, so
    a policy would be judged on nine patches of ground it never trained on. The
    training config's 10x20 grid at 0.1 m is the ground the run actually saw.

    Each override below removes one source of variation between two cells that are
    supposed to differ only in their command:

      enable_corruption        observation noise. Training injected it; an eval
                               that keeps it measures the noise as well as the
                               policy, and the noise is not what is being asked
                               about. This is what `*_PLAY` does too.
      push_robot               random shoves. A cell's episode would end early or
                               late depending on a coin flip.
      base_external_force_torque  the same, as a persistent wrench.
      curriculum.terrain_levels   THE ONE THAT WOULD BE EASY TO MISS. `mdp.
                               terrain_levels_vel` promotes or demotes an env's
                               terrain row on reset according to how far it walked
                               under its command. Left on, the ground a cell is
                               measured on becomes a function of how the PREVIOUS
                               cell went, and the grid stops being a controlled
                               comparison. `HoundDesertEnvCfg_PLAY` does not clear
                               this term (`AnymalCRoughEnvCfg_PLAY` does, and the
                               Hound config descends from the generic locomotion
                               cfg rather than from ANYmal's) -- worth fixing
                               there; cleared here regardless so this file does not
                               depend on that being done.
      resampling_time_range    training resamples the command every 10 s inside a
                               20 s episode, so a training episode carries two
                               commands. A cell must carry one.
      rel_standing_envs        2% of envs are told to stand regardless of their
                               command. Two percent of a 128-env cell is 2-3
                               episodes whose command is not the cell's command.
      rel_heading_envs         1.0 so that when heading mode IS used, it is used by
                               every env rather than by a random subset.

    What is deliberately NOT turned off: `events.reset_base`, which randomises the
    initial position and yaw. A single spawn pose would make the whole cell one
    episode repeated N times, and the random initial yaw is what gives the heading
    cells a real heading error to close. It is seeded, so it is identical across
    cells and across arms.
    """
    from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg

    cfg = HoundDesertEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    if device is not None:
        cfg.sim.device = device

    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.curriculum.terrain_levels = None

    cmd = cfg.commands.base_velocity
    cmd.resampling_time_range = (NO_RESAMPLE_S, NO_RESAMPLE_S)
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 1.0
    cmd.debug_vis = False
    return cfg


def apply_cell(env, cell: Cell, yaw_clip: tuple[float, float]) -> None:
    """Point the command term at one cell, exactly.

    Collapsing each range to a single value rather than writing `vel_command_b`
    directly is deliberate: the term still performs its own draws, so the RNG is
    consumed identically in every cell and the seed really does reproduce the same
    initial states across the grid. A hand-written buffer would desynchronise the
    stream and quietly make the cells incomparable.

    `yaw_clip` is the TRAINING `ang_vel_z` range. Under heading mode
    `_update_command` clips the derived yaw command to `cfg.ranges.ang_vel_z`, so
    that range is a saturation limit there rather than a command -- collapsing it
    the way the other axes are collapsed would pin the yaw command to zero and
    silently turn a heading cell into a stop cell.
    """
    term = env.command_manager.get_term("base_velocity")
    term.cfg.ranges.lin_vel_x = (cell.vx, cell.vx)
    term.cfg.ranges.lin_vel_y = (cell.vy, cell.vy)
    if cell.heading is None:
        term.cfg.heading_command = False
        term.cfg.ranges.ang_vel_z = (cell.wz, cell.wz)
        term.cfg.ranges.heading = None
    else:
        term.cfg.heading_command = True
        term.cfg.ranges.ang_vel_z = yaw_clip
        term.cfg.ranges.heading = (cell.heading, cell.heading)


# ---------------------------------------------------------------------------
# One cell, one arm.
# ---------------------------------------------------------------------------
def run_cell(
    *,
    env,
    wrapper,
    policy,
    cell: Cell,
    seed: int,
    max_steps: int,
    leg_ids,
    wheel_ids,
    yaw_clip: tuple[float, float],
    zero_action: bool,
) -> dict:
    """N episodes of one command, one per env, and every statistic from them.

    **Exactly one episode per env: the first one.** Isaac Lab auto-resets an env
    the instant it terminates, so a fixed number of steps otherwise mixes a first
    episode with a fragment of a second, and the fragments are not comparable
    across arms because the arms terminate at different times. Every accumulator
    below is masked by "this env's first episode is still running", so `n` really
    is `num_envs` episodes and episode length really is an episode length.

    The kinematic accumulators carry a second mask, `& ~done`. After the step that
    ends an episode, `robot.data` already describes the RESPAWNED robot -- the
    reward for that step is still the pre-reset reward and is counted, but its
    velocity is a fresh spawn at the origin and is not. The cost is one frame of
    kinematics out of an episode, on the envs that terminated.
    """
    import torch

    robot = env.scene["robot"]
    term = env.command_manager.get_term("base_velocity")
    dt = env.step_dt
    n = env.num_envs
    dev = env.device
    z64 = lambda: torch.zeros(n, dtype=torch.float64, device=dev)  # noqa: E731

    reward_terms = list(env.reward_manager.active_terms)
    if not hasattr(env.reward_manager, "_step_reward"):
        raise RuntimeError(
            "isaaclab's RewardManager has no `_step_reward` buffer, so the per-term "
            "reward decomposition this battery reports cannot be read from the manager "
            "that paid it. Recomputing the terms here instead is the defect "
            "research/anomalies.jsonl row 28 records (an instrument that recomputes a "
            "reward term drifts from what the env actually paid); refusing instead."
        )

    apply_cell(env, cell, yaw_clip)
    # A full reset with the seed, AFTER the command ranges are in place, so the
    # draws that set this cell's command come from the same point in the stream in
    # every cell. Reset is what re-seeds; the order matters.
    env.reset(seed=seed)
    # `get_observations` returns a TensorDict, NOT a (obs, extras) tuple -- that is
    # `reset`'s shape. Unpacking it here silently binds the first observation GROUP
    # name to `obs` and hands the policy a string.
    obs = wrapper.get_observations()

    n_kin, n_rew = z64(), z64()
    sum_vx, sum_vy, sum_wz, sum_speed = z64(), z64(), z64(), z64()
    sum_cmd_wz = z64()
    sum_leg_acc_sq, sum_wheel_acc_sq = z64(), z64()
    sum_rew = z64()
    sum_terms = torch.zeros(n, len(reward_terms), dtype=torch.float64, device=dev)
    park_steps = z64()
    early_abs_err, early_n = z64(), z64()
    late_abs_err, late_n = z64(), z64()

    alive = torch.ones(n, dtype=torch.bool, device=dev)
    ep_len = torch.zeros(n, dtype=torch.long, device=dev)
    terminated = torch.zeros(n, dtype=torch.bool, device=dev)
    timed_out = torch.zeros(n, dtype=torch.bool, device=dev)

    start_xy = _t(robot.data.root_pos_w)[:, :2].clone().double()
    last_xy = start_xy.clone()

    # First and last fifth of the window, for the heading cells: "did it aim" and
    # "was it still aiming, or had it stopped" are different questions and the
    # difference between the two windows is what separates them.
    early_cut = max(1, max_steps // 5)
    late_cut = max_steps - max(1, max_steps // 5)

    zeros_action = torch.zeros(n, env.action_manager.total_action_dim, device=dev)

    for step in range(max_steps):
        with torch.inference_mode():
            actions = zeros_action if zero_action else policy(obs)
        # The command the policy is acting on and the reward will be paid against.
        # `command_manager.compute` runs at the END of `step`, so this read is the
        # value in the observation the action was just chosen from.
        cmd = env.command_manager.get_command("base_velocity").clone()

        obs, rew, dones, extras = wrapper.step(actions)
        if not zero_action and hasattr(policy, "reset"):
            # What `play_rsl_rl.py` does: clear any recurrent state on the envs that
            # just reset. This actor is an MLP and has none, so it is a no-op today
            # -- and it is exactly the line whose absence would make a future
            # recurrent policy carry the last episode's memory into the next one.
            with torch.inference_mode():
                policy.reset(dones)
        done = dones.to(torch.bool)
        # `RslRlVecEnvWrapper` forwards gymnasium's `truncated` as `time_outs`; a
        # done without it is a real termination (trunk contact), which is the
        # number that separates "survived the cell" from "fell over in it".
        time_outs = extras.get("time_outs")
        if time_outs is None:
            raise RuntimeError(
                "the rsl_rl env wrapper did not report `time_outs`, so a fall cannot "
                "be told apart from reaching the time limit and the termination count "
                "in this table would be meaningless"
            )
        time_outs = time_outs.to(torch.bool)

        m_rew = alive
        m_kin = alive & ~done

        f_rew = m_rew.double()
        n_rew += f_rew
        sum_rew += rew.double() * f_rew
        # `_step_reward` is the reward RATE (the manager divides by dt), so dt puts
        # it back on the same scale as `rew` -- which the identity below checks.
        sum_terms += env.reward_manager._step_reward.double() * dt * f_rew[:, None]
        sum_cmd_wz += cmd[:, 2].double() * f_rew

        f_kin = m_kin.double()
        n_kin += f_kin
        lin_b = _t(robot.data.root_lin_vel_b).double()
        ang_b = _t(robot.data.root_ang_vel_b).double()
        sum_vx += lin_b[:, 0] * f_kin
        sum_vy += lin_b[:, 1] * f_kin
        sum_wz += ang_b[:, 2] * f_kin
        speed = torch.linalg.norm(lin_b[:, :2], dim=1)
        sum_speed += speed * f_kin
        park_steps += (speed < PARK_SPEED_MS).double() * f_kin

        acc = _t(robot.data.joint_acc).double()
        sum_leg_acc_sq += torch.sum(acc[:, leg_ids] ** 2, dim=1) * f_kin
        sum_wheel_acc_sq += torch.sum(acc[:, wheel_ids] ** 2, dim=1) * f_kin

        if cell.heading is not None:
            from isaaclab.utils import math as math_utils

            err = torch.abs(
                math_utils.wrap_to_pi(term.heading_target - _t(robot.data.heading_w))
            ).double()
            if step < early_cut:
                early_abs_err += err * f_kin
                early_n += f_kin
            if step >= late_cut:
                late_abs_err += err * f_kin
                late_n += f_kin

        last_xy = torch.where(m_kin[:, None], _t(robot.data.root_pos_w)[:, :2].double(), last_xy)

        ended = alive & done
        if ended.any():
            ep_len = torch.where(ended, torch.full_like(ep_len, step + 1), ep_len)
            terminated |= ended & ~time_outs
            timed_out |= ended & time_outs
        alive = alive & ~done
        if not bool(alive.any()):
            break

    # Envs still running when the window closed: their episode is CENSORED, not
    # finished. Counted and reported rather than folded into `timed_out`, because
    # a censored episode is an artifact of --steps and a time-out is a property of
    # the policy.
    censored = alive.clone()
    ep_len = torch.where(censored, torch.full_like(ep_len, max_steps), ep_len)

    # ---- the decomposition identity ---------------------------------------
    # The four-of-five accounting `research/anomalies.jsonl` row 36 records --
    # `track_eval` hardcoded four reward terms against a reward with five and left
    # a residual of 20% of the control arm's baseline -- is impossible here only if
    # it is checked. The term list comes from the manager, so a term added to the
    # reward appears automatically; this asserts that the terms the manager listed
    # really do sum to the reward it returned.
    residual = float(torch.max(torch.abs(sum_terms.sum(dim=1) - sum_rew)).item())
    if residual > DECOMPOSITION_TOL:
        raise AssertionError(
            f"cell {cell.name}: the {len(reward_terms)} reward terms the manager "
            f"reports sum to a value differing from the reward it returned by "
            f"{residual:.3e} on the worst env (tolerance {DECOMPOSITION_TOL:.1e}). "
            "The decomposition printed below would not account for the return."
        )

    steps_kin = torch.clamp(n_kin, min=1.0)
    steps_rew = torch.clamp(n_rew, min=1.0)
    mean_over_envs = lambda x: float(torch.mean(x).item())  # noqa: E731
    std_over_envs = lambda x: float(torch.std(x).item())  # noqa: E731

    per_env_vx = sum_vx / steps_kin
    per_env_rew_step = sum_rew / steps_rew
    displacement = torch.linalg.norm(last_xy - start_xy, dim=1)

    n_legs, n_wheels = len(leg_ids), len(wheel_ids)
    leg_acc_rms = float(torch.sqrt(torch.sum(sum_leg_acc_sq) / (torch.sum(n_kin) * n_legs)).item())
    wheel_acc_rms = float(
        torch.sqrt(torch.sum(sum_wheel_acc_sq) / (torch.sum(n_kin) * n_wheels)).item()
    )

    # The reward's own kernels, recovered from what it paid rather than recomputed:
    # `_step_reward` is `weight * kernel`, so dividing the recorded per-step term by
    # `weight * dt` returns the kernel exactly as the env evaluated it.
    kernels = {}
    for name, phi in (("track_lin_vel_xy_exp", "phi_v"), ("track_ang_vel_z_exp", "phi_w")):
        if name in reward_terms:
            w = env.reward_manager.get_term_cfg(name).weight
            idx = reward_terms.index(name)
            kernels[phi] = mean_over_envs(sum_terms[:, idx] / steps_rew / (w * dt))

    out = {
        "command": cell.command,
        "commanded_vx": cell.vx,
        "commanded_speed": cell.commanded_speed,
        "heading_target_rad": cell.heading,
        "mode": "heading" if cell.heading is not None else "rate",
        "note": cell.note,
        "n_episodes": n,
        "achieved_vx": mean_over_envs(per_env_vx),
        "achieved_vx_sd_over_envs": std_over_envs(per_env_vx),
        "achieved_vy": mean_over_envs(sum_vy / steps_kin),
        "achieved_wz": mean_over_envs(sum_wz / steps_kin),
        "achieved_speed": mean_over_envs(sum_speed / steps_kin),
        "commanded_wz_mean": mean_over_envs(sum_cmd_wz / steps_rew),
        "displacement_m": mean_over_envs(displacement),
        "park_fraction": mean_over_envs(park_steps / steps_kin),
        "episode_steps_mean": mean_over_envs(ep_len.double()),
        "episode_steps_min": int(ep_len.min().item()),
        "terminated": int(terminated.sum().item()),
        "timed_out": int(timed_out.sum().item()),
        "censored": int(censored.sum().item()),
        "reward_per_step": mean_over_envs(per_env_rew_step),
        "reward_per_step_sd_over_envs": std_over_envs(per_env_rew_step),
        "episode_return": mean_over_envs(sum_rew),
        "reward_terms_per_step": {
            name: mean_over_envs(sum_terms[:, i] / steps_rew) for i, name in enumerate(reward_terms)
        },
        "leg_acc_rms": leg_acc_rms,
        "wheel_acc_rms": wheel_acc_rms,
        "leg_acc_sq_sum": float(torch.sum(sum_leg_acc_sq).item()),
        "kinematic_steps": float(torch.sum(n_kin).item()),
        "decomposition_residual_worst_env": residual,
    }
    out.update(kernels)
    if cell.heading is not None:
        out["heading_abs_err_first_fifth"] = mean_over_envs(early_abs_err / torch.clamp(early_n, min=1.0))
        out["heading_abs_err_last_fifth"] = mean_over_envs(late_abs_err / torch.clamp(late_n, min=1.0))
    return out


# ---------------------------------------------------------------------------
# Derived, cross-cell statistics.
# ---------------------------------------------------------------------------
def arm_summary(cells: dict, *, n_legs: int, dt: float) -> dict:
    """The statistics that only exist across cells, and the ones the record asks for."""
    ratio, n_used, why = span_ratio(cells)

    # Pooled leg acceleration rms over the whole grid: sum of squares over every
    # counted step and joint, divided by the count. NOT the mean of the per-cell
    # rms values -- that would weight a cell that terminated in 40 steps the same
    # as one that ran 1000.
    sq = sum(c["leg_acc_sq_sum"] for c in cells.values())
    steps = sum(c["kinematic_steps"] for c in cells.values())
    pooled_rms = math.sqrt(sq / (steps * n_legs)) if steps else float("nan")

    drive = {k: c for k, c in cells.items() if c["commanded_speed"] >= MIN_COMMANDED_SPEED_MS}
    park = {
        "cells": sorted(drive),
        "mean_park_fraction": (
            sum(c["park_fraction"] for c in drive.values()) / len(drive) if drive else None
        ),
        "mean_speed": (
            sum(c["achieved_speed"] for c in drive.values()) / len(drive) if drive else None
        ),
        "mean_displacement_m": (
            sum(c["displacement_m"] for c in drive.values()) / len(drive) if drive else None
        ),
        # An UPPER BOUND on tracking, not a measure of it: the numerator is the
        # planar norm, so lateral drift the machine was never asked for counts
        # towards it. It can exceed 1.0 while forward tracking is poor.
        "speed_tracking_ratio": (
            sum(c["achieved_speed"] / c["commanded_speed"] for c in drive.values()) / len(drive)
            if drive
            else None
        ),
    }

    # The ratio that cannot be inflated by drift: forward velocity against forward
    # command, on the cells that ask for forward motion and nothing else.
    fwd = {
        k: c
        for k, c in cells.items()
        if c["commanded_vx"] > 0 and c["command"][1] == 0 and c["command"][2] == 0
    }
    park["forward_tracking_ratio"] = {
        k: c["achieved_vx"] / c["commanded_vx"] for k, c in sorted(fwd.items())
    }
    park["forward_tracking_ratio_mean"] = (
        sum(park["forward_tracking_ratio"].values()) / len(fwd) if fwd else None
    )

    mirror = None
    if "fwd_yaw_pos" in cells and "fwd_yaw_neg" in cells:
        a, b = cells["fwd_yaw_pos"], cells["fwd_yaw_neg"]
        mirror = {
            "reward_per_step": [a["reward_per_step"], b["reward_per_step"]],
            "abs_delta_reward_per_step": abs(a["reward_per_step"] - b["reward_per_step"]),
            "achieved_wz": [a["achieved_wz"], b["achieved_wz"]],
            "yaw_rate_symmetry": (
                abs(a["achieved_wz"] + b["achieved_wz"]) / max(abs(a["achieved_wz"]), abs(b["achieved_wz"]))
                if max(abs(a["achieved_wz"]), abs(b["achieved_wz"])) > 0
                else None
            ),
        }

    return {
        "vx_span_ratio": ratio,
        "vx_span_ratio_cells": n_used,
        "vx_span_ratio_detail": why,
        "vx_span_ratio_guard_bar": GUARD_MIN_SPAN_RATIO,
        "leg_acc_rms_pooled": pooled_rms,
        "leg_acc_rms_assumed": ASSUMED_LEG_ACC_RMS,
        "leg_acc_rms_measured_over_assumed": pooled_rms / ASSUMED_LEG_ACC_RMS,
        "dof_acc_l2_per_step_measured": {
            k: c["reward_terms_per_step"].get("dof_acc_l2") for k, c in cells.items()
        },
        # Step-weighted, not the mean of the per-cell means: a cell that terminated
        # in 40 steps must not weigh the same as one that ran 1000.
        "dof_acc_l2_per_step_grid": (
            sum(
                c["reward_terms_per_step"].get("dof_acc_l2", 0.0) * c["kinematic_steps"]
                for c in cells.values()
            )
            / steps
            if steps
            else None
        ),
        "point_and_park": park,
        "mirror_pair": mirror,
        "dt_s": dt,
        "n_leg_joints": n_legs,
    }


def achievable_income_per_step(env_yaml: dict, dt: float) -> dict:
    """What the two tracking terms can pay per step, from the TRAINING command range.

    Re-derived rather than copied, the same way `check_hound` re-derives it and for
    the same reason: `research/decisions/0005` B2 states 0.4409 but its Part B has
    no committed script, so the number may not be cited. Four wheels with fixed
    spin axes cannot hold body-frame lateral velocity, so the best any competence
    reaches on the y channel is `e_y = c_y`, and with the kernel
    `exp(-(e_x^2 + e_y^2)/std^2)` at `e_x = 0` and `c_y ~ U(-h, +h)`,

        E[exp(-(c_y/std)^2)] = (std*sqrt(pi)/(2h)) * erf(h/std)

    with the `h -> 0` limit written out as 1: with the channel pinned to a value
    the machine can hold, the whole term is earnable.

    This is the DENOMINATOR the reward budget's percentages are taken against, so
    it has to come from the config that trained the checkpoint -- not from the eval
    override, which pins the command and would silently report a different ceiling.
    """
    rewards = env_yaml["rewards"]
    lin, ang = rewards["track_lin_vel_xy_exp"], rewards["track_ang_vel_z_exp"]
    std = float(lin["params"]["std"])
    lo, hi = env_yaml["commands"]["base_velocity"]["ranges"]["lin_vel_y"]
    half = 0.5 * (float(hi) - float(lo))
    ceiling = 1.0 if half == 0.0 else (std * math.sqrt(math.pi) / (2.0 * half)) * math.erf(half / std)
    income = (float(lin["weight"]) * ceiling + float(ang["weight"]) * 1.0) * dt
    return {
        "trained_lin_vel_y_range": [float(lo), float(hi)],
        "lin_term_ceiling": ceiling,
        "achievable_income_per_step": income,
        "kernel_std": std,
    }


def price_leg_acceleration(
    *,
    leg_acc_rms: float,
    charged_per_step: float,
    weight: float,
    n_legs: int,
    dt: float,
    income: float,
) -> dict:
    """The `dof_acc_l2` budget line, priced three ways, because two of them disagree.

    `check_hound.check_reward_budget_against_011_and_015` prices this term as
    `|weight| * n_legs * rms^2 * dt` from an ASSUMED 250 rad/s^2. This function
    reports that, and then two independent measurements of the same term:

      **the charge** -- `charged_per_step`, the mean per-step `dof_acc_l2` reward the
      `RewardManager` actually paid. This is the authoritative number for the
      budget, because it *is* the budget line. Inverting the same formula gives
      `charge_implied_rms`, the rms that reproduces it by construction.

      **the direct read** -- `leg_acc_rms`, pooled from `robot.data.joint_acc`
      sampled once per policy step immediately after `env.step()` returns.

    THESE TWO HAVE BEEN OBSERVED TO DISAGREE, by 1.7x to 2.9x in rms on the
    2026-07-30 arm-1 seeds (direct 43-51 rad/s^2 against charge-implied 91-106).
    Both are printed and `rms_disagreement_ratio` is recorded, because the record's
    rule for exactly this situation (`research/anomalies.jsonl` row 28: an
    instrument that recomputes a reward term drifts from what the env paid) is to
    name the gap rather than to choose a side quietly.

    The likely mechanism, UNVERIFIED: `ArticulationData.joint_acc` in Isaac Lab 3.0
    is a lazily refreshed finite difference over the time since it was last
    accessed, so two consumers per step do not necessarily divide by the same
    interval. Settling it needs one instrumented step, which is a cheap experiment
    and has not been run.

    What does NOT depend on resolving it: the budget conclusion. The charge is
    measured directly, so the term's real share of achievable income is known
    whichever rms is quoted.
    """
    cost = lambda rms: abs(weight) * n_legs * rms**2 * dt  # noqa: E731
    charged = abs(charged_per_step)
    implied = math.sqrt(charged / (abs(weight) * n_legs * dt)) if charged > 0 else 0.0
    return {
        "weight": weight,
        "n_legs": n_legs,
        "assumed_rms": ASSUMED_LEG_ACC_RMS,
        "assumed_cost_per_step": cost(ASSUMED_LEG_ACC_RMS),
        "assumed_share_of_income": cost(ASSUMED_LEG_ACC_RMS) / income,
        # Authoritative: what the reward manager paid, step-weighted over the grid.
        "charged_per_step": -charged,
        "charged_share_of_income": charged / income,
        "charge_implied_rms": implied,
        # The second, disagreeing path to the same physical quantity.
        "direct_read_rms": leg_acc_rms,
        "direct_read_cost_per_step": cost(leg_acc_rms),
        "direct_read_share_of_income": cost(leg_acc_rms) / income,
        "rms_disagreement_ratio": (implied / leg_acc_rms) if leg_acc_rms > 0 else None,
    }


# ---------------------------------------------------------------------------
# Printing.
# ---------------------------------------------------------------------------
def print_table(title: str, cells: dict) -> None:
    print(f"\n=== {title} " + "=" * max(0, 76 - len(title)), flush=True)
    head = (
        f"{'cell':<13}{'command (vx,vy,wz)':>21} {'ach vx':>7} {'ach vy':>7} {'ach wz':>7} "
        f"{'|v|':>6} {'disp':>6} {'steps':>6} {'term':>5} {'r/step':>8} "
        f"{'phi_v':>6} {'phi_w':>6} {'park':>6} {'legacc':>7}"
    )
    print(head, flush=True)
    print("-" * len(head), flush=True)
    for name, c in cells.items():
        cmd = c["command"]
        tag = "H" if c["mode"] == "heading" else " "
        print(
            f"{name:<13}{tag}({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})".ljust(34)
            + f" {c['achieved_vx']:+7.3f} {c['achieved_vy']:+7.3f} {c['achieved_wz']:+7.3f}"
            f" {c['achieved_speed']:6.3f} {c['displacement_m']:6.2f}"
            f" {c['episode_steps_mean']:6.1f} {c['terminated']:5d} {c['reward_per_step']:+8.4f}"
            f" {c.get('phi_v', float('nan')):6.3f} {c.get('phi_w', float('nan')):6.3f}"
            f" {c['park_fraction']:6.3f} {c['leg_acc_rms']:7.1f}",
            flush=True,
        )
    print(
        "  (H = heading mode: wz is the env's own controller output, not a fixed command."
        "  disp = net planar displacement, m."
        f"  park = fraction of steps with |v| < {PARK_SPEED_MS} m/s)",
        flush=True,
    )


def print_comparison(policy_cells: dict, control_cells: dict) -> None:
    print("\n=== policy versus the do-nothing control, per cell " + "=" * 30, flush=True)
    head = (
        f"{'cell':<13} {'pol r/step':>11} {'ctl r/step':>11} {'gap':>9} "
        f"{'pol |v|':>8} {'ctl |v|':>8} {'pol disp':>9} {'ctl disp':>9}"
    )
    print(head, flush=True)
    print("-" * len(head), flush=True)
    losses = []
    for name, p in policy_cells.items():
        c = control_cells.get(name)
        if c is None:
            continue
        gap = p["reward_per_step"] - c["reward_per_step"]
        if gap < 0:
            losses.append(name)
        print(
            f"{name:<13} {p['reward_per_step']:+11.4f} {c['reward_per_step']:+11.4f} "
            f"{gap:+9.4f} {p['achieved_speed']:8.3f} {c['achieved_speed']:8.3f} "
            f"{p['displacement_m']:9.2f} {c['displacement_m']:9.2f}",
            flush=True,
        )
    print(
        f"  loses to doing nothing in {len(losses)} of {len(policy_cells)} cells"
        + (f": {', '.join(losses)}" if losses else ""),
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="path to a model_*.pt. Repeat to evaluate several seeds in one Kit "
        "startup; the terrain and the control arm are then shared, which is the "
        "expensive half.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="citable name for each checkpoint, parallel to --checkpoint. Output "
        "goes to research/measurements/isaac_hound_<label>.json. Defaults to the "
        "checkpoint's run-directory name.",
    )
    parser.add_argument("--num_envs", type=int, default=128, help="episodes per cell (one per env)")
    parser.add_argument("--seed", type=int, default=1000, help="the seed used in EVERY cell")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="policy steps per episode. 0 (default) = the env's own episode length, "
        "so every episode ends in a termination or a time-out and none is censored.",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default="",
        help="comma-separated subset of grid cell names. Empty = the whole grid.",
    )
    parser.add_argument(
        "--no_control",
        action="store_true",
        help="skip the zero-action arm. research/learnings/011 says do not.",
    )
    parser.add_argument("--task", type=str, default="Bestiary-Desert-Hound-v0")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(visualizer=["none"])
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    # From the module, not the package: `load_cfg_from_registry` is not re-exported
    # by `isaaclab_tasks.utils.__init__`, so the shorter import raises ImportError.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    from bestiary import paths
    from bestiary.isaac import tasks
    from bestiary.isaac.hound_cfg import LEG_JOINT_EXPR, WHEEL_JOINT_EXPR
    from bestiary.record.freeze import freeze_checkpoint

    self_test_span_ratio()

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required; a battery with nothing to judge is not a run")
    ckpts = [Path(c).resolve() for c in args.checkpoint]
    for c in ckpts:
        if not c.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {c}")
    labels = args.label or [c.parent.name for c in ckpts]
    if len(labels) != len(ckpts):
        raise SystemExit(f"{len(ckpts)} --checkpoint but {len(labels)} --label; they must be parallel")

    grid = GRID
    if args.cells:
        want = [s.strip() for s in args.cells.split(",") if s.strip()]
        by_name = {c.name: c for c in GRID}
        unknown = [w for w in want if w not in by_name]
        if unknown:
            raise SystemExit(f"unknown cell(s) {unknown}; the grid is {sorted(by_name)}")
        grid = tuple(by_name[w] for w in want)

    # Every checkpoint must have been trained under the same env, or one env cannot
    # judge them all. Compared pairwise on the load-bearing fields before anything
    # is built, so the failure costs a second rather than a Kit startup.
    run_yamls = {}
    for c, label in zip(ckpts, labels):
        p = c.parent / "params" / "env.yaml"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} does not exist, so there is no record of the config {c.name} was "
                "trained under and no way to check this eval is stepping it"
            )
        run_yamls[label] = _tagged_yaml(p)
    base_label = labels[0]
    base_fp = training_fingerprint(run_yamls[base_label])
    for label in labels[1:]:
        compare_fingerprints(base_fp, training_fingerprint(run_yamls[label]))

    tasks.register()
    cfg = eval_env_cfg(args.num_envs, args.seed, args.device)
    env = gym.make(args.task, cfg=cfg).unwrapped

    config_check = compare_fingerprints(base_fp, live_fingerprint(env, cfg))
    print(
        f"[bestiary] config check: {len(config_check['fields_compared'])} load-bearing "
        f"field groups agree with {base_label}/params/env.yaml",
        flush=True,
    )

    trained_cmd = run_yamls[base_label]["commands"]["base_velocity"]
    yaw_clip = tuple(float(v) for v in trained_cmd["ranges"]["ang_vel_z"])
    dt = env.step_dt
    max_steps = args.steps if args.steps > 0 else int(env.max_episode_length)

    robot = env.scene["robot"]
    leg_ids, leg_names = robot.find_joints(list(LEG_JOINT_EXPR))
    wheel_ids, wheel_names = robot.find_joints(list(WHEEL_JOINT_EXPR))
    if len(leg_ids) != 12 or len(wheel_ids) != 4:
        raise AssertionError(
            f"expected 12 leg joints and 4 wheel joints, resolved {len(leg_ids)} "
            f"({leg_names}) and {len(wheel_ids)} ({wheel_names}). The leg "
            "acceleration rms and the reward budget it prices are both per-joint sums, "
            "so a wrong count silently rescales them."
        )

    wrapper = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        from importlib import metadata

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    except Exception:  # noqa: BLE001 -- shim, absent in some versions; the load below is the real check
        pass
    runner = OnPolicyRunner(wrapper, agent_cfg.to_dict(), log_dir=None, device=str(env.device))

    income = achievable_income_per_step(run_yamls[base_label], dt)
    protocol = {
        "task": args.task,
        "grid": [c.name for c in grid],
        "episodes_per_cell": args.num_envs,
        "seed_every_cell": args.seed,
        "steps_per_episode": max_steps,
        "policy_dt_s": dt,
        "deterministic": True,
        "action": "rsl_rl runner.get_inference_policy() -- the same inference path "
        "IsaacLab/scripts/reinforcement_learning/rsl_rl/play_rsl_rl.py uses, in eval mode",
        "velocity_frame": "root_lin_vel_b / root_ang_vel_b -- the same body-frame "
        "quantities isaaclab.envs.mdp.rewards reads, not a reimplementation",
        "kernels_source": "recovered from RewardManager._step_reward, not recomputed "
        "(research/anomalies.jsonl row 28)",
        "eval_overrides_vs_training": {
            "observation_noise": "off",
            "push_robot": "off",
            "base_external_force_torque": "off",
            "curriculum.terrain_levels": "off",
            "command_resampling_s": NO_RESAMPLE_S,
            "rel_standing_envs": 0.0,
            "heading_command": "False on rate cells, True on head_* cells",
        },
        "park_speed_ms": PARK_SPEED_MS,
        "min_commanded_speed_ms": MIN_COMMANDED_SPEED_MS,
    }

    def run_arm(policy, zero_action: bool, title: str) -> dict:
        cells = {}
        for cell in grid:
            cells[cell.name] = run_cell(
                env=env,
                wrapper=wrapper,
                policy=policy,
                cell=cell,
                seed=args.seed,
                max_steps=max_steps,
                leg_ids=leg_ids,
                wheel_ids=wheel_ids,
                yaw_clip=yaw_clip,
                zero_action=zero_action,
            )
            c = cells[cell.name]
            print(
                f"[{title}] {cell.name:<13} vx {c['achieved_vx']:+.3f}  |v| {c['achieved_speed']:.3f}"
                f"  wz {c['achieved_wz']:+.3f}  steps {c['episode_steps_mean']:.0f}"
                f"  term {c['terminated']}  r/step {c['reward_per_step']:+.4f}",
                flush=True,
            )
        return cells

    # The control arm first: it does not depend on any checkpoint, so one run of it
    # serves every seed, and having it in hand makes each policy arm readable the
    # moment it finishes.
    control = None
    if not args.no_control:
        control_cells = run_arm(None, True, "zero-action")
        control = {
            "arm": "zero_action",
            "action": "exactly zero. On this body a zero leg action IS the solved "
            "standing stance (use_default_offset=True), so this arm stands rather "
            "than collapsing.",
            "cells": control_cells,
            **arm_summary(control_cells, n_legs=len(leg_ids), dt=dt),
        }
        print_table("zero-action control", control_cells)

    written = []
    for ckpt, label in zip(ckpts, labels):
        frozen = freeze_checkpoint(ckpt, run_dir=paths.RUNS / f"isaac_hound_{label}")
        runner.load(str(frozen.frozen))
        policy = runner.get_inference_policy(device=env.device)
        cells = run_arm(policy, False, label)
        print_table(f"policy {label} ({ckpt.name})", cells)
        if control is not None:
            print_comparison(cells, control["cells"])

        summary = arm_summary(cells, n_legs=len(leg_ids), dt=dt)
        acc_price = price_leg_acceleration(
            leg_acc_rms=summary["leg_acc_rms_pooled"],
            charged_per_step=summary["dof_acc_l2_per_step_grid"] or 0.0,
            weight=env.reward_manager.get_term_cfg("dof_acc_l2").weight,
            n_legs=len(leg_ids),
            dt=dt,
            income=income["achievable_income_per_step"],
        )
        doc = {
            "measurement": f"isaac_hound_{label}",
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "robot": "hound",
            "simulator": "isaac_lab",
            "run": ckpt.parent.name,
            "protocol": protocol,
            "config_check": config_check,
            "achievable_income": income,
            "dof_acc_l2_budget": acc_price,
            **frozen.as_json_fields(),
            "policy": {"arm": label, "cells": cells, **summary},
        }
        if control is not None:
            doc["zero_action"] = control
            doc["cells_lost_to_zero_action"] = sorted(
                k
                for k, c in cells.items()
                if c["reward_per_step"] < control["cells"][k]["reward_per_step"]
            )

        out = paths.RESEARCH / "measurements" / f"isaac_hound_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
        written.append(out)

        print(f"\n--- {label}: the numbers the record asks for " + "-" * 30, flush=True)
        r = summary["vx_span_ratio"]
        print(
            f"  vx_span_ratio            {r if r is None else f'{r:.3f}'}   "
            f"({summary['vx_span_ratio_detail']}; guard bar {GUARD_MIN_SPAN_RATIO}, "
            "1.0 = perfect tracker, 0.0 = one fixed gait)",
            flush=True,
        )
        print(
            f"  dof_acc_l2 charge        {acc_price['charged_per_step']:+.6f}/step = "
            f"{acc_price['charged_share_of_income'] * 100:.2f}% of achievable income "
            f"({income['achievable_income_per_step']:.6f}/step). The budget's [ASSUMED] "
            f"{ASSUMED_LEG_ACC_RMS:.0f} rad/s^2 priced it at "
            f"{acc_price['assumed_share_of_income'] * 100:.2f}%",
            flush=True,
        )
        ratio = acc_price["rms_disagreement_ratio"]
        flag = "" if ratio is None or abs(ratio - 1.0) <= 0.2 else "   <<< THESE DISAGREE"
        print(
            f"  leg joint acc rms        {acc_price['charge_implied_rms']:.1f} rad/s^2 implied "
            f"by that charge, {acc_price['direct_read_rms']:.1f} read directly off "
            f"robot.data.joint_acc"
            + (f" (ratio {ratio:.2f})" if ratio is not None else "")
            + flag,
            flush=True,
        )
        if flag:
            print(
                "                           the charge is authoritative for the budget "
                "(it IS the budget line); see price_leg_acceleration's docstring for the "
                "unresolved instrument question. anomalies.jsonl row 28 shape.",
                flush=True,
            )
        park = summary["point_and_park"]
        if park["mean_park_fraction"] is not None:
            print(
                f"  point-and-park           over {len(park['cells'])} cells with a real "
                f"linear command: park_fraction {park['mean_park_fraction']:.3f}, "
                f"mean |v| {park['mean_speed']:.3f} m/s, "
                f"mean displacement {park['mean_displacement_m']:.2f} m",
                flush=True,
            )
            print(
                f"  forward tracking         "
                + ", ".join(
                    f"{k} {v:.3f}" for k, v in park["forward_tracking_ratio"].items()
                )
                + f"  (achieved_vx / commanded_vx; mean "
                f"{park['forward_tracking_ratio_mean']:.3f}. |v|/|c| is "
                f"{park['speed_tracking_ratio']:.3f} but the norm counts lateral drift, "
                "so it is an upper bound only)",
                flush=True,
            )
        for cell_name in ("head_drive", "head_stop"):
            c = cells.get(cell_name)
            if c and "heading_abs_err_first_fifth" in c:
                print(
                    f"  {cell_name:<24} |heading err| {c['heading_abs_err_first_fifth']:.3f} rad "
                    f"-> {c['heading_abs_err_last_fifth']:.3f} rad, "
                    f"mean |v| {c['achieved_speed']:.3f} m/s, "
                    f"disp {c['displacement_m']:.2f} m, phi_w {c.get('phi_w', float('nan')):.3f}",
                    flush=True,
                )
        mirror = summary["mirror_pair"]
        if mirror:
            print(
                f"  mirror pair              wz {mirror['achieved_wz'][0]:+.3f} vs "
                f"{mirror['achieved_wz'][1]:+.3f}, r/step "
                f"{mirror['reward_per_step'][0]:+.4f} vs {mirror['reward_per_step'][1]:+.4f} "
                f"(|delta| {mirror['abs_delta_reward_per_step']:.4f})",
                flush=True,
            )
        print(f"  written                  {out}", flush=True)

    print("\n[bestiary] wrote:", flush=True)
    for p in written:
        print(f"    {p}", flush=True)
    return 0


def _exit(status: int) -> None:
    """Leave the process with `status`, and actually mean it.

    Copied from `check_hound._exit`, and for the reason recorded there:
    `SimulationApp.close()` ENDS THE PROCESS ITSELF with status 0, so a `sys.exit(1)`
    placed after it makes a failing battery report success. Flush, then `os._exit`,
    and no `close()` -- which also explains why every print here passes `flush=True`.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


if __name__ == "__main__":
    _args = _parse_args()
    AppLauncher(_args)
    try:
        _exit(main(_args))
    except Exception:
        traceback.print_exc()
        _exit(1)
