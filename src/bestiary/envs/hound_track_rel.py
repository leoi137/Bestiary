"""HoundPDTrackRelDesert-v0: the tolerance scales with the command.

WHY THIS ENV EXISTS

`envs/hound_track.py` priced the task correctly and still lost to a machine
doing nothing. Measured over a six-cell drive grid at 1.5M steps, 20
episodes/cell: the trained policy earned 0.07175/step of tracking against a
zero-action policy's 0.06501/step, and paid 0.06889/step of control cost to get
it -- **12.6x more than it earned**. `research/learnings/011` has the four-term
decomposition (control cost 105.5% of the gap, crashing 0.9%);
`research/learnings/012` shows the heading collapse is a tax on locomotion and
not on the terrain, so easier ground and more steps do not reach it.

Two things were wrong, and the second one is the interesting one.

**1. THE KERNEL TAIL, NOT THE PRODUCT.** The record's diagnosis was that the
multiplicative Phi_v*Phi_w is the defect because a standing machine collects
Phi_w = 0.97 for free. That is a misreading: a product needs BOTH factors, and
Phi_w = 0.97 is *correct scoring* -- a standing machine genuinely satisfies
w_cmd = 0. The leak is that Phi_v fails to zero it out. At the flagship cell a
standing machine's speed error is 0.535 m/s, and

    Phi_cauchy(0.535/0.15) = 1/(1+12.72) = 0.0729,  x 0.9705 = 0.0707

which reproduces the measured 0.07069 to four figures. The Cauchy tail IS the
freeride. It was a recorded, deliberately accepted tradeoff -- the theory note
budgeted it at "~4% of a good policy's return" against an ASSUMED good policy
scoring 0.87/step. The achieved policy scores 0.0718/step, so the same absolute
leak became 90.6% of the return. **A fixed absolute floor was budgeted as a
fraction of a performance level nobody had reached yet.**

**2. NO FIXED TOLERANCE CAN WORK, WHATEVER THE KERNEL.** This is the part that
is not obvious and that a sharper kernel alone does not fix.

    the policy's achieved speed error at command 0.8 is  0.400 m/s
    a STANDING machine's speed error at command 0.3 is   0.335 m/s

Any kernel of error magnitude is monotone decreasing, so "pay the policy at
command 0.8" and "starve a stander at command 0.3" ask the same function to be
simultaneously large at 0.400 and small at 0.335. Unsatisfiable, by any family,
at any scale. Sharpening the kernel at a fixed sigma does not fix it -- it
helps the one cell where the policy already tracks well and hurts the four
where it does not, moving the grid-mean gap from +0.0067 to +0.0071, which is
nothing (`research/scripts/tracking_reward_separation.py`).

**The resolution is to scale the tolerance BY THE COMMAND.** A standing
machine's error *is* the command, so in relative error it sits at ~1.04 at
every command, while the policy sits at 0.49-0.53 at every forward command.
The overlap disappears because the quantity being measured changed, not
because the kernel got sharper.

    r = 1[healthy] * K_v(e_v; c) * K_w(e_w; c)            <- income
      + 0.99 * P(s') * 1[not terminated] - P(s)           <- shaping, lambda = 1
      - 0.01  * sum(a^2)                                  <- UNCHANGED
      - 5e-4  * sum(clip(cfrc, -1, 1)^2)                  <- UNCHANGED
      - 10.0  * 1[unhealthy termination]                  <- UNCHANGED

    K(e; alpha) = exp(-(e/alpha)^2)
    alpha_v(c)  = max(0.15, |vx_cmd| / 2)
    alpha_w(c)  = max(0.10, 0.75 * |w_cmd|)

WHY THE OLD REWARD IS STILL HERE, AS THE POTENTIAL

A light tail buys the economics and costs the gradient: a from-scratch SAC
policy spends its first million steps far outside any sharp kernel, where
exp(-(e/alpha)^2) is numerically flat. That is exactly why Cauchy was chosen
originally, and the reason has not stopped being true.

So the old reward is kept -- demoted from income to **potential**. Under
Ng-Harada-Russell, F = gamma*P(s') - P(s) leaves every policy's ordering of
returns unchanged, so the shaping cannot make standing profitable and cannot be
farmed; it telescopes. What it does is put the far-field slope into the
immediate reward, where the critic sees it after ONE Bellman backup instead of
inferring it from basin income many backups away.

The choice of P is not a guess. The failed 1.5M-step run is an existence proof
that precisely this gradient field carries a from-scratch policy from a speed
error of ~0.75 m/s down to ~0.25 m/s. That run did not fail to learn; it
learned and then discovered the task did not pay. **So reuse its gradient and
replace its economics.** Income by kernel, gradient by shaping, and the two
jobs are now done by two different terms instead of one term doing both badly.

THE BOUNDARY CONVENTIONS ARE THE EASIEST THING HERE TO GET WRONG

The theorem requires P(absorbing) = 0, which makes the shaping at a true
unhealthy termination exactly -P(s_t): an extra death penalty, correctly
signed, bounded by 1.

**Timeout is not termination.** At step 1000 the episode has not ended in the
MDP, so the shaping must use the real P(s_{t+1}). This env never sees the
timeout -- `max_episode_steps` is enforced by Gymnasium's TimeLimit wrapper
outside it -- so the correct behaviour is what falls out of only zeroing the
potential on `terminated`. Zeroing it on truncation instead would inject a
spurious -0.9 spike into a transition whose TD target also bootstraps the next
state, which is the large and silent version of this bug.

**The potential jumps at a command resample, and that is correct.** The command
is part of the state, so P legitimately moves when the command does. It is
exogenous and action-independent, so the jump is unbiased noise the policy
cannot exploit, bounded by 1, once per 200-300 steps. Do NOT "fix" it by
freezing the command inside P -- that makes P non-Markov and quietly breaks the
invariance theorem, trading a visible harmless spike for an invisible harmful
bias.

WHAT DELIBERATELY DID NOT CHANGE, AND WHY

The two costs, the termination penalty, the command distribution, the
observation, the terrain, and the heading frame. This env exists to measure ONE
structural change; a coefficient moved in the same commit would confound the
comparison it is here to make. The termination penalty in particular now has a
better justification than it had: K = c/(1-gamma) at the ACHIEVED drive-cell
income under this kernel (~0.11/step) gives ~11, so the inherited 10.0 is
approximately right for a measured reason rather than an assumed one -- see
`docs/lessons/006` for why the sign of that formula is the opposite of what it
looks like.

The backward command floor is retirable but retained. Under this kernel a
standing machine's take is nearly floor-independent (0.043 / 0.035 / 0.031 at
|vx| = 0.3 / 0.4 / 0.5) because its relative error is ~1 everywhere, so the
0.40 asymmetry the Cauchy era needed is no longer load-bearing. Keeping it is
one-variable discipline, not a claim that it is required.
"""
from __future__ import annotations

import numpy as np
from gymnasium import utils

from bestiary import paths
from bestiary.envs.hound_track import HoundTrackEnv, kernel as cauchy_kernel
from bestiary.envs.reward_spec import RewardSpec, RewardTerm

# --- The tolerance scales. Every one is the centre of a derived window. ------

# Velocity tolerance at a STOPPED command, m/s. Two-sided: the noise floor
# needs K_v(0.0361) >= 0.9, so alpha >= 0.111; a machine drifting at 0.3 m/s
# under a stop command must score <= 0.05, so alpha <= 0.173. Window
# [0.111, 0.173]. 0.15 is the incumbent value and sits inside it, so it is kept
# -- now with its inequality attached rather than as a bare constant.
ALPHA_V_MIN = 0.15

# Relative velocity tolerance: alpha_v = BETA_V * |vx_cmd|. Lower bound comes
# from the fastest cell having to pay 1.5x its control cost at the achieved
# relative error of ~0.5, giving BETA_V >= 0.44. Upper bound comes from a
# standing machine's take staying under 25% of the policy's net at the worst
# forward command, giving BETA_V <= 0.54. Window [0.44, 0.54]; 0.5 is its
# centre, and it makes alpha_v continuous at the command floor: 0.5*0.3 = 0.15
# is exactly ALPHA_V_MIN, so the two branches meet without a step.
BETA_V = 0.5

# Yaw tolerance at a STRAIGHT command, rad/s. Noise floor K_w(0.0182) >= 0.9
# gives alpha >= 0.056; the freeride bound -- an unsteered but DRIVING machine
# yawing 0.12695 must score <= 0.45 -- gives alpha <= 0.142. Window
# [0.056, 0.142]. The incumbent 0.10 is kept; it scores that freeride at 0.20,
# comfortably tighter than the 0.45 cap.
ALPHA_W_MIN = 0.10

# Relative yaw tolerance: alpha_w = BETA_W * |w_cmd|. This is the tightest
# window in the design and it is squeezed from both sides. Below 0.72 the
# (0.5, 0, -0.4) cell goes net-negative at the achieved yaw error. Above 0.764
# an unsteered machine under a 0.4 rad/s command clears the 0.45 freeride cap.
# Window [0.72, 0.76]. Both jaws open as tracking improves, so this constant is
# the one most likely to want revisiting after a run that actually tracks.
BETA_W = 0.75

# The shaping discount. MUST equal SAC's gamma: a mismatch breaks the
# invariance theorem by O(|delta gamma| * P) and would make the shaping a real
# term in the objective rather than a telescoping one.
SHAPING_GAMMA = 0.99

# Potential tolerances -- deliberately the OLD reward's, unchanged. The
# potential's whole job is to reproduce the gradient field the failed run
# demonstrated was sufficient to learn from.
POTENTIAL_SIGMA_V = 0.15
POTENTIAL_SIGMA_W = 0.10


def relative_kernel(err: float, alpha: float) -> float:
    """K(e; alpha) = exp(-(e/alpha)^2).

    Note this is exp(-(e/alpha)^2), NOT the exp(-(e/alpha)^2/2) of a standard
    Gaussian density -- the factor of two is absorbed into alpha, and the
    windows above are derived against this form. Writing it the other way with
    the same alphas would widen every tolerance by sqrt(2) and reopen the
    freeride.
    """
    return float(np.exp(-((err / alpha) ** 2)))


def velocity_tolerance(vx_cmd: float) -> float:
    """alpha_v(c) = max(ALPHA_V_MIN, BETA_V*|vx_cmd|)."""
    return max(ALPHA_V_MIN, BETA_V * abs(float(vx_cmd)))


def yaw_tolerance(w_cmd: float) -> float:
    """alpha_w(c) = max(ALPHA_W_MIN, BETA_W*|w_cmd|)."""
    return max(ALPHA_W_MIN, BETA_W * abs(float(w_cmd)))


class HoundTrackRelEnv(HoundTrackEnv):
    """The hound, with a tracking tolerance proportional to what was asked."""

    def __init__(self, xml_file: str = str(paths.HOUND_PD_DESERT_XML), **kwargs):
        # Set before super().__init__: the base constructor can reach _get_obs,
        # and a half-built env that AttributeErrors is the good failure mode.
        self._potential_prev = 0.0

        super().__init__(xml_file=xml_file, **kwargs)

        utils.EzPickle.__init__(self, xml_file, **kwargs)

        # `track_cmd` KEEPS ITS NAME ON PURPOSE. nulls.jsonl row 2 is a
        # machine-enforced launch gate asserting that this term is present and
        # that `forward_velocity` is absent on any ^Hound.*Desert-v0$ env;
        # renaming it would turn the gate red and block every future launch.
        # The shaping is a genuinely separate term and is declared separately,
        # which is also what stops `shape_hash` from claiming this reward is
        # the old one retuned -- it is not, and the record must not be able to
        # read it that way.
        self._reward_spec = RewardSpec(
            env=type(self).__name__,
            cmd_dist=self._reward_spec.cmd_dist,
            terms=(
                RewardTerm(
                    "track_cmd", 1.0,
                    "healthy * K(e_v; alpha_v(c)) * K(e_w; alpha_w(c)); "
                    "K(e;a) = exp(-(e/a)^2), tolerances scale WITH the command",
                    params=(
                        ("alpha_v_min", ALPHA_V_MIN),
                        ("beta_v", BETA_V),
                        ("alpha_w_min", ALPHA_W_MIN),
                        ("beta_w", BETA_W),
                        ("kernel", "gaussian_sq"),
                        ("frame", "heading"),
                    ),
                ),
                RewardTerm(
                    "pbrs_shaping", 1.0,
                    "gamma*P(s') - P(s), P = the OLD cauchy product at "
                    "sigma_v=0.15 sigma_w=0.10, P(terminal)=0; "
                    "policy-invariant by Ng-Harada-Russell",
                    params=(
                        ("gamma", SHAPING_GAMMA),
                        ("potential_sigma_v", POTENTIAL_SIGMA_V),
                        ("potential_sigma_w", POTENTIAL_SIGMA_W),
                        ("potential_kernel", "cauchy"),
                    ),
                ),
                RewardTerm("ctrl_cost", -self._ctrl_cost_weight, "-w * sum(action^2)"),
                RewardTerm("contact_cost", -self._contact_cost_weight,
                           "-w * sum(clipped external contact forces^2)"),
                RewardTerm("termination", -self._termination_penalty,
                           "one-time, on unhealthy termination only"),
            ),
        )

    # --- The potential -------------------------------------------------------

    def _potential(self) -> float:
        """P(s) = Phi_cauchy(e_v/0.15) * Phi_cauchy(e_w/0.10), the old reward.

        Reads the LIVE command, so after a resample this is the potential of
        the new command. That is correct and deliberate -- see the module note.
        """
        v_b = self.heading_velocity
        e_v = float(np.linalg.norm(v_b - self._cmd[:2]))
        e_w = abs(self.yaw_rate - float(self._cmd[2]))
        return float(cauchy_kernel(e_v / POTENTIAL_SIGMA_V)
                     * cauchy_kernel(e_w / POTENTIAL_SIGMA_W))

    # --- Gym API -------------------------------------------------------------

    def step(self, action: np.ndarray):
        self.do_simulation(self.action_to_ctrl(action), self.frame_skip)

        v_b = self.heading_velocity
        w = self.yaw_rate
        v_cmd = self._cmd[:2]
        w_cmd = float(self._cmd[2])

        err_v = float(np.linalg.norm(v_b - v_cmd))
        err_w = abs(w - w_cmd)
        alpha_v = velocity_tolerance(v_cmd[0])
        alpha_w = yaw_tolerance(w_cmd)
        k_v = relative_kernel(err_v, alpha_v)
        k_w = relative_kernel(err_w, alpha_w)

        healthy = self.is_healthy
        track_reward = k_v * k_w if healthy else 0.0
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        terminated = self._terminate_when_unhealthy and not healthy
        termination_cost = self._termination_penalty if terminated else 0.0

        scored_cmd = (float(v_cmd[0]), float(v_cmd[1]), w_cmd)

        # Resample AFTER scoring, so the obs returned by this step carries the
        # command the NEXT step is scored against -- the base class's invariant.
        self._steps_until_resample -= 1
        if self._steps_until_resample <= 0:
            self._resample_command()

        # P(s_{t+1}) is read AFTER the resample, because s_{t+1} carries
        # cmd_{t+1}. On a true termination the next state is absorbing and the
        # theorem requires P = 0 there; on a TIMEOUT this env is never told, so
        # the real potential is used, which is exactly right.
        potential_next = 0.0 if terminated else self._potential()
        shaping = SHAPING_GAMMA * potential_next - self._potential_prev
        self._potential_prev = potential_next

        reward = track_reward + shaping - ctrl_cost - contact_cost - termination_cost

        observation = self._get_obs()
        wheels = self.wheel_speeds
        info = {
            "reward_track": track_reward,
            "reward_shaping": shaping,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_termination": -termination_cost,
            "track_phi_v": k_v,
            "track_phi_w": k_w,
            "track_alpha_v": alpha_v,
            "track_alpha_w": alpha_w,
            # The relative errors are the quantity this env is built around, so
            # they are logged directly rather than left to be reconstructed.
            "track_rel_err_v": err_v / alpha_v,
            "track_rel_err_w": err_w / alpha_w,
            "track_err_v": err_v,
            "track_err_w": err_w,
            "potential": potential_next,
            "cmd_vx": scored_cmd[0],
            "cmd_vy": scored_cmd[1],
            "cmd_w": scored_cmd[2],
            "achieved_vx": float(v_b[0]),
            "achieved_vy": float(v_b[1]),
            "achieved_w": w,
            "x_position": float(self.data.body("trunk").xpos[0]),
            "y_position": float(self.data.body("trunk").xpos[1]),
            "trunk_upright": self.trunk_upright,
            "height_above_ground": self.height_above_ground,
            "wheel_speed_mean": float(np.mean(wheels)),
            "wheel_speed_std": float(np.std(wheels)),
        }
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self) -> np.ndarray:
        obs = super().reset_model()
        # Seed the telescoping sum with P(s_0). Without this the first step of
        # every episode carries a spurious -P(s_0) or a stale potential from
        # the previous episode, which is a real bias and an easy one to miss.
        self._potential_prev = self._potential()
        return obs
