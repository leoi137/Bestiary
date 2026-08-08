"""HOUND-16, reward = v_x ONLY, on the v5 ground. The wheel-legged diagnostic.

    Bestiary-ForwardV5-Hound-v0        the diagnostic config
    Bestiary-ForwardV5-Hound-Play-v0   few robots, no noise, no shoving

WHAT THIS TASK IS FOR
---------------------
`research/episodes/014` ran the same experiment on the Spyder: delete the whole
reward table, pay base-frame `v_x` at weight 1.0, and see whether locomotion
still emerges. It did — 44.72 m per 20 s episode by iteration 1499, from a
reward with no tracking kernel, no contact timing and no joint penalties. That
result is about a stack (Isaac Lab + PPO + the MJCF->USD port) and about a body
with twelve leg joints and four feet.

This task asks the same question of the OTHER body in the bestiary, whose feet
are driven hub wheels, on the terrain `research/decisions/0007` mandates for
every new arm. It is a PROBE by construction — one seed, one arm, no control —
and the seed rule applies to whatever it produces.

WHAT FORWARD-ONLY MEANS ON A MACHINE WITH WHEELS
-------------------------------------------------
`r = v_x` pays metres travelled and says NOTHING about how they were travelled.
On the Spyder there is only one way to earn it: step. On the Hound there are
two, and the reward cannot tell them apart:

  * ROLL. Spin the four hub wheels. `hound_cfg.wheel_velocity_gain()` is derived
    so the drive reaches a commanded speed in one control period, and at the
    3.0 N·m effort ceiling it saturates at 10.665 rad/s — 0.906 m/s of rim
    speed. Past that error the wheel is a pure torque source at the traction
    limit, so 0.906 m/s is where COMMANDED speed stops, not a hard wall.
  * GALLOP. Use the twelve leg joints and bound, with the wheels along for the
    ride. Episode 014 measured what an unshaped `v_x` selects when legs are the
    only option: 4.2–5.4 m/s, airborne, visibly violent — several times the
    Hound drive's saturation speed.

Every term that expressed a preference between the two is gone with the table
that carried it. `HoundRewardsCfg` charged `dof_torques_l2` and `dof_acc_l2` on
the legs ONLY and gave the wheels a 100x weaker acceleration penalty — a
deliberate thumb on the scale for rolling (`research/decisions/0004` Part B) —
and `lin_vel_z_l2` at -2.0 priced exactly the vertical bouncing a gallop is made
of. This task deletes all of it. Nothing left in the reward prefers a mode.

**That ambiguity is the experiment.** The question is not which mode is correct
— nothing here encodes correct — but which one an unshaped speed objective
SELECTS on a body that has both, and the two outcomes read differently:

  * It rolls. Then forward-only reward reproduces driving on this machine, and
    the desert table's careful wheel-vs-leg scoping was buying obedience and
    smoothness rather than locomotion.
  * It gallops. Then the unshaped objective has found that this machine's legs
    outrun its drive, and every wheel-specific coefficient in `HoundRewardsCfg`
    was pricing a mode the reward never had to prefer in the first place.

Either way the run cannot be read as "the Hound's reward is right/wrong". It
measures what speed alone buys, on ground nothing has trained on yet.

WHAT MOVES AGAINST `Bestiary-Desert-Hound-v0`, AND WHAT DOES NOT
-----------------------------------------------------------------
Two sections, and `check_hound.py`'s `forward-v5 task is v_x on v5 ground,
nothing else` pins them by dumping both configs with `to_dict()`:

    rewards   the whole table replaced by the one term
    scene     the terrain generator (which lives under `scene`, so the section
              is unavoidable — the assertion descends into it and requires the
              moved leaves to be the terrain generator's sub-terrains and
              nothing else in the scene: not the robot, not the sensors, not
              the env count or spacing)

INHERITED UNTOUCHED, on purpose:

  * TERMINATIONS. A trunk-contact reset is episode machinery, not a reward —
    nothing pays or charges for it. Deleting it would let a collapsed machine
    slide downhill accumulating `v_x` forever, which is a different experiment.
  * OBSERVATIONS. The desert task's eight policy terms, with the four wheel
    ANGLES already dropped from `joint_pos` (`hound_desert_env_cfg.py` explains
    why an unbounded integrator is not an observation). The width is a one-way
    door — the actor's first layer is sized to it — so a diagnostic does not
    move it.
  * ACTIONS. Twelve position targets then four wheel speeds, in that order.
  * COMMANDS. `lin_vel_x/y/z` ranges, the standing fraction and heading mode are
    the desert task's. `v_x` does not read the command, so the command channel
    in the observation is paid for nothing and the policy is free to ignore it —
    the same deliberate waste `spyder_forward_env_cfg.py` documents, kept for the
    same reason: removing the command would move the observation width.

    The caveat, stated so it is not discovered later: upstream's
    `terrain_levels_vel` (this task keeps it — the Hound never adopted the
    arc-corrected bar) demotes when displacement falls below
    `||cmd_xy|| * T / 2`, which reads the LINEAR command only. A command-deaf
    policy that drives straight and fast clears that bar regardless of what yaw
    was asked for, and a standing command's bar is 0, which no displacement can
    fall below. So the curriculum here is biased toward PROMOTION. Left as is,
    per the one-variable rule.

THE REWARD SURGERY IS THE SPYDER'S, DELIBERATELY
-------------------------------------------------
`use_forward_velocity_only` and `ForwardOnlyRewardsCfg` are imported from
`spyder_forward_env_cfg`, not restated here. They are robot-agnostic — one
config class with one field, plus a post-condition that the surviving term is
`bestiary.isaac.rewards.forward_velocity` at weight 1.0 — and two copies of a
"the entire reward is this one term" surgery is two things to keep identical
across every future edit. The import direction looks odd and has precedent in
both directions already: `spyder_gentle_env_cfg` imports `retarget` from
`hound_desert_env_cfg`. Shared machinery lives where it was written; only the
robots are separate.

WHAT THE TENSORBOARD NUMBER MEANS
----------------------------------
`RewardManager` scales every term by `weight * step_dt`, so at weight 1.0 the
per-step reward is `v_x * 0.02 s` and the EPISODE RETURN IS METRES OF FORWARD
TRAVEL. It is directly comparable to `Bestiary-ForwardV5-Spyder-v0`'s return in
units only: same reward, same terrain, different machine and different episode
dynamics. It is NOT comparable to any `anymal_c_rough`-filed Hound arm: those
returns came from the desert task's kernel-based tracking table, which is
bounded per step and pays for obedience rather than for distance.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary.isaac.hound_desert_env_cfg import HoundDesertEnvCfg, HoundDesertEnvCfg_PLAY
from bestiary.isaac.spyder_forward_env_cfg import use_forward_velocity_only
from bestiary.isaac.spyder_forward_v5_env_cfg import use_gentle_v5_mix


def apply_forward_v5(cfg) -> None:
    """The whole variant: one reward term, v5 ground. Called by both configs.

    Ordered reward-then-terrain for readability only; the two are independent.
    Shared by the training config and its Play twin so a future edit cannot
    reach one and miss the other — `check_hound.py` checks both configs anyway,
    because "cannot" here means "cannot without editing this function".
    """
    use_forward_velocity_only(cfg)
    use_gentle_v5_mix(cfg)


@configclass
class HoundForwardV5EnvCfg(HoundDesertEnvCfg):
    """The Hound with its entire reward replaced by `v_x`, on v5 ground."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_forward_v5(self)


@configclass
class HoundForwardV5EnvCfg_PLAY(HoundDesertEnvCfg_PLAY):
    """Viewer config for the diagnostic: few robots, nothing random.

    Descends from `HoundDesertEnvCfg_PLAY`, NOT from `HoundForwardV5EnvCfg`, so
    the Play overrides (16 envs, native terrain sampling, a 3x3 grid with the
    curriculum off, corruption and pushes off) are inherited rather than copied.
    The v5 mix is rebuilt at whatever sampling and grid shape that config
    declared, so the viewer shows the same ground the training config trains on,
    at the viewer's own resolution.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_forward_v5(self)
