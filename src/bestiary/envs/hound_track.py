"""HoundPDTrackDesert-v0: the hound paid for tracking a command, not for existing.

WHY THIS ENV EXISTS

`envs/hound.py`'s reward pays a forward-velocity term plus an alive bonus. A
machine that stands perfectly still collects nearly all of it. Measured at
n=60 deterministic episodes on the PD arm, the trained policy scores 1140.6
against a zero-action policy's 955.6 -- a ratio of 1.194. That is not a policy
that learned to walk badly; it is a reward whose payment is decoupled from the
task, so the optimizer correctly discovered that existence pays and locomotion
barely does.

This env removes the mechanism rather than repricing it. The ONLY positive
term is a product of two tolerance kernels, one on linear velocity and one on
yaw rate, both measured against an explicitly sampled command. There is no
term a non-tracking policy can collect, and no coefficient rebalance can
reopen one.

The full derivation -- both tolerances, the command distribution, the
termination penalty, the predicted separations, and nine ranked failure modes
-- is `docs/theory/command-tracking-reward.md`. That note is the contract; this
module is its implementation and deliberately re-derives nothing. Its
plain-language doorway is `docs/lessons/003-add-or-multiply.md`.

    r = 1.0 * 1[healthy] * Phi(u_v) * Phi(u_w)      <- the ONLY positive term
      - 0.01  * sum(a^2)                            <- carried over UNCHANGED
      - 5e-4  * sum(clip(cfrc, -1, 1)^2)            <- carried over UNCHANGED
      - 10.0  * 1[unhealthy termination]            <- new, one-time

    Phi(u) = 1 / (1 + u^2)          (Cauchy)
    u_v    = ||v_heading - (vx_cmd, vy_cmd)|| / 0.15      [m/s]
    u_w    = |yaw_rate - w_cmd|     / 0.10               [rad/s]

THE TWO COSTS ARE CARRIED OVER UNCHANGED ON PURPOSE. They were not implicated
in the exploit, and changing them in the same commit as the structural fix
would confound the before/after comparison -- the one thing this whole window
exists to measure. `docs/theory/command-tracking-reward.md` failure mode 8
records that the inherited contact cost is itself odd (it charges ~0.045/step
for supporting the robot's weight, a negative alive bonus wearing a cost's
clothes) and flags fixing it as a SEPARATE later commit, for exactly this
reason.

Note the contact cost here is the env's existing `sum(clip(F,-1,1)^2)`, while
the theory note's Section 1 writes it as `sum|F|`. The note's own prose is what
governs -- "carried over unchanged" -- so the implementation follows the
existing code, not the note's shorthand. Flagged rather than silently
reconciled.

THE FRAME IS LOAD-BEARING, AND IT IS THE EASIEST THING HERE TO GET WRONG

The velocity error is computed in the trunk's HEADING frame, never the world
frame. Under a nonzero yaw command, the world-frame velocity of a correctly
driving body rotates continuously, so tracking a fixed world-frame command
while turning is unsatisfiable -- it would cap Phi(u_v) near 0.5 on every
turning segment and no amount of training would close it. The old reward's
"x_velocity of the trunk" convention must not be inherited silently, and this
module does not inherit it: `HoundEnv.step` is not called at all.

`guards/tracking_frame.py` asserts this rather than trusting the comment. It
drives the trunk at a known velocity under a known yaw and checks that the
measured heading-frame velocity is the rotated one, which fails loudly if
anyone reverts to world frame. That is failure mode 6 in the theory note,
turned from a thing to watch for into a thing that cannot ship.

OBSERVATION -- UNCHANGED AT 169, AND THIS IS NOT A ONE-WAY DOOR

The three command values go into the `command_reserved` slots that
`envs/hound.py` cut for exactly this purpose and has been zero-filling since
its first commit. Width does not move, the obs spec hash does not move, and
every existing hound checkpoint still loads. This is the single case in the
project where a feature lands without touching the one-way door -- because
someone paid for the slots up front.

The commands are written RAW (m/s and rad/s), not normalized. They are already
O(0.1-1), the same scale as the qvel entries beside them.
"""
from __future__ import annotations

import numpy as np
from gymnasium import utils

from bestiary import paths
from bestiary.envs.hound import HoundEnv
from bestiary.envs.reward_spec import RewardSpec, RewardTerm

# --- The command distribution (theory note Section 3) ------------------------
#
# Every number here is defended in the note; the one-line reasons are repeated
# because a reader of this file should not have to open that one to review the
# sampler. `CMD_DIST_VERSION` below is BUILT FROM THESE CONSTANTS rather than
# hand-written, so it cannot drift from what the sampler actually does -- edit
# any number and the hashed version string changes with it, automatically.

P_STOP = 0.10        # large enough that stopping is learned, small enough that
                     # the always-stand policy's legitimate harvest is ~11% of
                     # a good policy's rate
P_TURN = 0.10        # turn in place
# remainder, 0.80, is DRIVE -- the dominant real command

VX_MIN, VX_MAX = 0.3, 0.8   # min = 2*sigma_v, keeps every drive command two
                            # kernel widths off the standing point; max is
                            # PROVISIONAL, see the module note below
P_FORWARD = 0.8             # sign bias within DRIVE

W_TURN_MIN, W_TURN_MAX = 0.3, 0.6   # 2.4x the unsteered drift 0.127, so no
                                    # passive yaw accidentally matches a command
W_DRIVE_MAX = 0.6                   # 6*sigma_w
P_DRIVE_STRAIGHT = 0.5              # point mass at w_cmd = 0 within DRIVE

RESAMPLE_MIN, RESAMPLE_MAX = 200, 300   # jittered; a fixed interval is
                                        # something a recurrent-ish policy can
                                        # phase-lock to (failure mode 7)

# VX_MAX = 0.8 IS PROVISIONAL AND THE THEORY NOTE SAYS SO. It extrapolates
# linearly from ONE open-loop measurement (wheel command 0.3 -> 0.3449 m/s),
# suggesting ~1.1 m/s at saturation, capped at ~80% of that. The extrapolation
# is unverified. Commanding the untrackable would recreate the
# punished-for-unremovable-error problem in reverse, so this is the first
# number to check if trained Phi_v saturates low on the fastest command cells.

SIGMA_V = 0.15       # m/s   -- theory note Section 2, window [0.12, 0.146], rounded top
SIGMA_W = 0.10       # rad/s -- theory note Section 2, window [0.055, 0.115]
TERMINATION_PENALTY = 10.0   # = c/(1-gamma) = 0.10/0.01 at gamma=0.99

# Hashed into the reward spec. Built from the constants above so it cannot lie.
CMD_DIST_VERSION = (
    f"stop{P_STOP:g}|turn{P_TURN:g}@{W_TURN_MIN:g}-{W_TURN_MAX:g}"
    f"|drive vx{VX_MIN:g}-{VX_MAX:g}p+{P_FORWARD:g}"
    f",w0p{P_DRIVE_STRAIGHT:g}else+-{W_DRIVE_MAX:g},vy0"
    f"|resample{RESAMPLE_MIN}-{RESAMPLE_MAX}"
)


def kernel(u: np.ndarray | float) -> np.ndarray | float:
    """Phi(u) = 1/(1+u^2), the Cauchy tolerance kernel.

    Cauchy rather than Gaussian, and the choice is deliberate and recorded:
    a Gaussian's gradient is numerically zero beyond ~3 sigma, and a
    from-scratch SAC policy spends its first million steps entirely out there.
    Cauchy's gradient 2u/(1+u^2)^2 decays only as 1/u^3, so the far field still
    points home. The price is a polynomial tail -- "near zero" means 0.03-0.16
    on the easiest drive slice, not 1e-4 -- which the theory note quantifies at
    ~4% of a good policy's return and accepts explicitly.

    This resolves an inconsistency the record carried: STATE.md and CORE_PLAN
    said Gaussian, the measurement script said Cauchy. The derivation chose
    Cauchy, for the reason above. This function is now the single authority.
    """
    return 1.0 / (1.0 + np.square(u))


class HoundTrackEnv(HoundEnv):
    """The hound, rewarded for matching a commanded velocity and yaw rate."""

    def __init__(
        self,
        xml_file: str = str(paths.HOUND_PD_DESERT_XML),
        sigma_v: float = SIGMA_V,
        sigma_w: float = SIGMA_W,
        termination_penalty: float = TERMINATION_PENALTY,
        **kwargs,
    ):
        # Set BEFORE super().__init__: MujocoEnv's constructor can reach
        # _get_obs, and _get_obs reads the command. A half-built env that
        # AttributeErrors here is the good failure; one that reads a stale
        # zero is the bad one.
        self._sigma_v = float(sigma_v)
        self._sigma_w = float(sigma_w)
        self._termination_penalty = float(termination_penalty)
        self._cmd = np.zeros(3)
        self._steps_until_resample = 0

        super().__init__(xml_file=xml_file, **kwargs)

        # Re-declare for EzPickle: HoundEnv recorded only ITS OWN arguments, so
        # an unpickled copy would silently fall back to the default tolerances.
        # SubprocVecEnv pickles envs; DummyVecEnv does not. Being correct under
        # both costs three lines.
        utils.EzPickle.__init__(
            self, xml_file, sigma_v, sigma_w, termination_penalty, **kwargs
        )

        # Replace the inherited reward spec wholesale. Section 6 of the theory
        # note is explicit that `track_cmd` must NOT be decomposed into
        # `track_lin` + `track_ang` for the hash: that list would collide with
        # a genuinely ADDITIVE two-term reward, which is a different objective
        # -- the exact objective this env exists to avoid. One name, with the
        # tolerances, the kernel and the frame recorded as its parameters.
        self._reward_spec = RewardSpec(
            env=type(self).__name__,
            cmd_dist=CMD_DIST_VERSION,
            terms=(
                RewardTerm(
                    "track_cmd", 1.0,
                    "healthy * Phi(|v_heading - v_cmd|/sigma_v) * Phi(|w - w_cmd|/sigma_w); "
                    "multiplicative BY DESIGN -- see theory note Section 1",
                    params=(
                        ("sigma_v", self._sigma_v),
                        ("sigma_w", self._sigma_w),
                        ("kernel", "cauchy"),
                        ("frame", "heading"),
                    ),
                ),
                RewardTerm("ctrl_cost", -self._ctrl_cost_weight, "-w * sum(action^2)"),
                RewardTerm("contact_cost", -self._contact_cost_weight,
                           "-w * sum(clipped external contact forces^2)"),
                RewardTerm("termination", -self._termination_penalty,
                           "one-time, on unhealthy termination only"),
            ),
        )

    # --- Command sampling ----------------------------------------------------

    def _resample_command(self) -> None:
        """Draw one command from the Section 3 mixture. Uses self.np_random."""
        rng = self.np_random
        roll = rng.uniform()
        if roll < P_STOP:
            self._cmd = np.zeros(3)
        elif roll < P_STOP + P_TURN:
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            self._cmd = np.array(
                [0.0, 0.0, sign * rng.uniform(W_TURN_MIN, W_TURN_MAX)]
            )
        else:
            sign = 1.0 if rng.uniform() < P_FORWARD else -1.0
            vx = sign * rng.uniform(VX_MIN, VX_MAX)
            # v_y is commanded ZERO, always. Nobody has measured whether this
            # wheel configuration can hold a lateral velocity at all, and
            # commanding a channel of unverified controllability injects
            # unremovable error into u_v and hands the gradient to noise. The
            # error term still INCLUDES v_y via the planar norm, so lateral
            # drift is correctly penalized against the commanded zero. Widen
            # only after a controllability measurement (theory note Section 3).
            w = 0.0 if rng.uniform() < P_DRIVE_STRAIGHT else rng.uniform(
                -W_DRIVE_MAX, W_DRIVE_MAX
            )
            self._cmd = np.array([vx, 0.0, w])
        self._steps_until_resample = int(rng.integers(RESAMPLE_MIN, RESAMPLE_MAX + 1))

    # --- Measured quantities -------------------------------------------------

    @property
    def heading_velocity(self) -> np.ndarray:
        """Planar trunk velocity in the trunk's own yaw frame, [v_forward, v_left].

        Read from qvel rather than finite-differenced xy. The two agree to
        integration error, and qvel is the quantity the physics actually
        holds -- a finite difference across a terrain step also picks up the
        vertical-to-planar coupling of a wheel climbing a cell wall.
        """
        v_world = np.asarray(self.data.qvel[:2], dtype=float)
        # xmat is body->world, row-major. Column 0 is the body's +x axis in
        # world coordinates, so its planar part IS the heading direction.
        m = np.asarray(self.data.body("trunk").xmat).reshape(3, 3)
        yaw = np.arctan2(m[1, 0], m[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        # R(yaw)^T @ v_world -- rotate the WORLD velocity INTO the heading frame.
        return np.array([c * v_world[0] + s * v_world[1],
                         -s * v_world[0] + c * v_world[1]])

    @property
    def yaw_rate(self) -> float:
        """Trunk yaw rate about the body z-axis, rad/s.

        For a MuJoCo free joint, qvel[3:6] is the angular velocity expressed in
        the BODY frame, so index 5 is the body-z component directly. This is
        the same convention legged_gym uses (`base_ang_vel[:, 2]`), and it is
        what the theory note's Section 1 specifies. For an upright trunk it is
        within a cosine of the world-frame yaw rate; the health check
        terminates past ~72 degrees, so the two can never diverge far here.
        """
        return float(self.data.qvel[5])

    # --- Gym API -------------------------------------------------------------

    def step(self, action: np.ndarray):
        # HoundEnv.step is deliberately NOT called: its reward is the thing
        # being replaced, and its x_velocity is world-frame.
        self.do_simulation(self.action_to_ctrl(action), self.frame_skip)

        v_b = self.heading_velocity
        w = self.yaw_rate
        v_cmd = self._cmd[:2]
        w_cmd = float(self._cmd[2])

        u_v = float(np.linalg.norm(v_b - v_cmd)) / self._sigma_v
        u_w = abs(w - w_cmd) / self._sigma_w
        phi_v = float(kernel(u_v))
        phi_w = float(kernel(u_w))

        healthy = self.is_healthy
        # Health gates ONLY the positive term. Costs stay outside the gate:
        # gating them by health would make death erase pending costs, which is
        # a small suicide subsidy (theory note Section 4).
        track_reward = phi_v * phi_w if healthy else 0.0
        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        terminated = self._terminate_when_unhealthy and not healthy
        termination_cost = self._termination_penalty if terminated else 0.0

        reward = track_reward - ctrl_cost - contact_cost - termination_cost

        # Snapshot the command THIS step was scored against, before resampling
        # can move it. Everything in `info` must describe one consistent
        # instant: `track_phi_v` and `achieved_vx` are measured under this
        # command, and the failure-mode-3 diagnostic regresses one on the
        # other. A first version of this method read `self._cmd` back out of
        # the env when building `info`, which is correct on 99.6% of steps and
        # wrong on exactly the resample boundaries -- so `cmd_w` (a local) and
        # `cmd_vx` (re-read) disagreed once per ~250 steps. The smoke test
        # caught it at step 217. Report the snapshot, never the live field.
        scored_cmd = (float(v_cmd[0]), float(v_cmd[1]), w_cmd)

        # Resample AFTER the reward, so the observation returned by this step
        # carries the command the NEXT step's reward will be measured against.
        # The invariant is: obs returned at step t holds cmd_{t+1}, and info
        # returned at step t holds cmd_t. Getting this backwards teaches the
        # policy to track a command it has not been shown yet.
        self._steps_until_resample -= 1
        if self._steps_until_resample <= 0:
            self._resample_command()

        observation = self._get_obs()
        wheels = self.wheel_speeds
        info = {
            "reward_track": track_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_termination": -termination_cost,
            # The per-channel factors are the primary diagnostic. Failure mode
            # 1 (parked in the standing basin) is phi_v low with total return
            # ~120; failure mode 2 (yaw freeride) is phi_w low while phi_v is
            # high. Neither is visible from the scalar reward alone, which is
            # why both are logged every step rather than derived later.
            "track_phi_v": phi_v,
            "track_phi_w": phi_w,
            "track_err_v": float(np.linalg.norm(v_b - v_cmd)),
            "track_err_w": float(abs(w - w_cmd)),
            "cmd_vx": scored_cmd[0],
            "cmd_vy": scored_cmd[1],
            "cmd_w": scored_cmd[2],
            # Command gain: regressing achieved on commanded velocity across an
            # eval detects failure mode 3, a policy that ignores the three obs
            # slots entirely. Slope ~1 is tracking; slope ~0 is a dead input.
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

    def _get_obs(self) -> np.ndarray:
        # Identical to HoundEnv's, except the three command slots carry the
        # live command instead of zeros. The height block stays reserved.
        self._reserved[: self._n_command] = self._cmd
        return super()._get_obs()

    def reset_model(self) -> np.ndarray:
        self._resample_command()
        return super().reset_model()
