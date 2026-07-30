"""The arithmetic behind `research/decisions/0006`: Hound's reward economics.

    venv/bin/python research/scripts/0006_reward_economics_refutation.py

`research/decisions/0005` recorded what survived one adversarial pass over the
Isaac Hound stack, and marked its own Part B arithmetic as uncitable because no
script held it. A second refutation then attacked 0005's successor reasoning and
the `lin_vel_y` edits made under it. This script is what makes that second pass'
figures citable: **every number `0006` quotes is printed here**, and any figure
this script does not print is barred from the record.

WHAT IT COMPUTES, AND WHY EACH PIECE IS NEEDED

1. **Command geometry.** The distribution the reward is scored against, and the
   one quantity the *terrain curriculum* reads off it -- `E||c_xy||`, which sets
   the demote bar in `terrain_levels_vel`. Narrowing `lin_vel_y` moves that bar
   without touching the curriculum, which is a confound, not a reward change.

2. **The tracking-income ceiling, and the budget guard's own denominator.**
   `check_hound.py::check_reward_budget_against_011_and_015` divides every
   penalty by "achievable income", and that denominator is a function of the
   `lin_vel_y` range. So a command-range edit silently rescales the guard that
   exists to catch `learnings/011`.

3. **The penalty basket** at that check's own [ASSUMED] operating point, priced
   against each candidate denominator. Same weights, three different verdicts.

4. **The freeride triple: stander, POINT-AND-PARK, competent driver.** The
   binding fake under heading mode is not a stander. `heading_command=True` with
   `heading_control_stiffness=0.5` makes the scored yaw command a feedback
   signal, so a machine that yaws to the commanded heading and then holds still
   drives its own yaw command to zero and scores the yaw kernel at 1.0. Section 4
   prices that behaviour, which no prior pass did.

4b. **The same triple as the chat record computed it, reproduced to 1e-6.**
   Section 4 prices the three behaviours against the COMMITTED table. Section 4b
   reproduces the chat record's own figures exactly and then names the four
   inputs that differ -- two of which are defects the same refutation lists
   elsewhere. Reproducing a number is worth more than disagreeing with it: it
   turns a headline into a citable figure with its assumptions attached.

5. **The axle algebra, by forward kinematics on the committed MJCF.** Whether any
   joint configuration steers the wheel -- and, separately, whether the rolling
   *direction* can leave the sagittal plane, which is a different question with a
   different answer.

6. **The no-lateral-slip constraint's nullspace**, numerically, at a common
   abduct roll.

7. **What the y channel charges when `c_y == 0`.** Not a skid price.

8. **Wheel-torque economics.** Whether the friction cone binds, and what a
   reward term would have to charge to price a resting limb. It charges nothing
   today.

9. **The count-form contact term**, its deterrence weight, and the
   self-termination arithmetic that makes one particular re-enabling dangerous.

WHAT IT DOES NOT DO

No GPU, no Isaac Lab, no simulation stepping. MuJoCo is used for forward
kinematics only (`mj_kinematics` on the committed model, zero dynamics steps).

EVERY CLAIMED FIGURE IS PRINTED NEXT TO THE COMPUTED ONE

Not as decoration. A recomputation that quietly replaces a number destroys the
ability to see that it moved, and three separate figures in the chat record turn
out to have been computed against a command range, a competence parameter or a
penalty table other than the committed one. Those are only visible as deltas.
One disagreement is left standing and not resolved: section 8's friction-cone
fraction at saturation, where `CARD.md` and `robots/hound/check.py` say 5% and
the static-load arithmetic says 22.65%. Neither has an instrument recorded.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bestiary.robots.hound.build import SPEC  # noqa: E402

# ---------------------------------------------------------------------------
# Inputs. Every one is read from a named source; nothing here is chosen.
# ---------------------------------------------------------------------------

#: Control period, s. `velocity_env_cfg.py` decimation 4 x sim.dt 0.005, and
#: `hound_cfg.CONTROL_DT_S`, which `HoundDesertEnvCfg.__post_init__` asserts
#: equal to the env's own `decimation * sim.dt`.
DT = 0.02

#: Tracking kernel width. `RewardsCfg` passes `std = math.sqrt(0.25)` to both
#: `track_lin_vel_xy_exp` and `track_ang_vel_z_exp`
#: (`velocity_env_cfg.py:288-293`), and the kernel is `exp(-error / std**2)`
#: (`isaaclab/envs/mdp/rewards.py:314-338`), so the divisor is 0.25.
STD = math.sqrt(0.25)

#: Reward weights actually live in `HoundRewardsCfg`, read 2026-07-29.
W_LIN, W_ANG = 1.0, 0.5
W_LIN_VEL_Z, W_ANG_VEL_XY = -2.0, -0.05
W_DOF_TORQUES, W_DOF_ACC, W_DOF_ACC_WHEEL = -1.0e-5, -2.5e-7, -2.5e-9
W_ACTION_RATE = -0.01

#: Command ranges. `lin_vel_x=(-1,1)`, `ang_vel_z=(-1,1)`,
#: `heading=(-pi,pi)`, `heading_command=True`, `rel_heading_envs=1.0`,
#: `heading_control_stiffness=0.5`, `rel_standing_envs=0.02`
#: (`velocity_env_cfg.py:140-151`). `lin_vel_y` is the one this repo overrides.
CX_HALF = 1.0
WZ_CLIP = 1.0
HEADING_STIFFNESS = 0.5
REL_STANDING_ENVS = 0.02

#: The three `lin_vel_y` half-widths in the history of this config, newest last.
#: `db88770` set (0,0); `b2e2634` set (-0.3,0.3); the inherited value is (-1,1).
CY_HALVES = {
    "inherited (-1.0, +1.0)": 1.0,
    "collapsed  (0.0,  0.0)": 0.0,
    "committed (-0.3, +0.3)": 0.3,
}

#: Joint counts. 12 leg joints + 4 wheels; 16 actions.
N_LEGS, N_WHEELS, N_ACTIONS = 12, 4, 16

#: `check_hound.py`'s own [ASSUMED] operating point, copied verbatim from
#: `check_reward_budget_against_011_and_015` (`vz_rms, wxy_rms, tau_rms,
#: acc_rms, dact_rms = 0.15, 0.5, 5.0, 250.0, 0.1`). NOT measured -- no Hound
#: policy exists. Reproduced rather than re-chosen so the shares this script
#: prints are the shares that check prints.
VZ_RMS, WXY_RMS, TAU_RMS, ACC_RMS, DACT_RMS = 0.15, 0.5, 5.0, 250.0, 0.1

#: The wheel drive's design acceleration, rad/s^2. DERIVED: the drive's time
#: constant is one control period, so a step to the largest commandable speed is
#: that speed over that period (`hound_cfg.wheel_velocity_gain`).
WHEEL_SPIN_INERTIA = 0.5 * SPEC.wheel_mass * SPEC.wheel_r**2 + SPEC.wheel_armature
WHEEL_VEL_GAIN = WHEEL_SPIN_INERTIA / DT
WHEEL_ACTION_SCALE = SPEC.gear_wheel / WHEEL_VEL_GAIN
WHEEL_ACC = WHEEL_ACTION_SCALE / DT

#: Figures quoted in the chat record this script exists to make citable, carried
#: so the deltas are visible rather than silently replaced.
CLAIMED = {
    "stander_net": 0.012003,
    "park_net": 0.018987,
    "driver_net": 0.020545,
    "park_share_of_driver": 0.924,
    "guard_gain": 1.59,
    "demote_drop": 0.347,
    "cone_at_saturation": 0.227,
    "cone_at_cruise": 0.125,
    "lateral_wedge": 0.332,
    "undesired_12_bodies": 8.31,
    "deterrence_weight": 0.7217,
    "income_ratio_committed_over_collapsed": 0.964,
    "cone_used_per_card": 0.05,
}


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# 1. Command geometry, and the one number the terrain curriculum reads
# ---------------------------------------------------------------------------
def mean_kernel_uniform(half: float, std: float = STD) -> float:
    """E[exp(-(c/std)^2)] for c uniform on [-half, +half].

    Closed form: (std*sqrt(pi)/(2h)) * erf(h/std). At h = 0 the closed form
    divides by zero and the limit is 1, because erf(x) -> 2x/sqrt(pi); written
    out because `check_hound.py` takes the same limit and the two must agree.

    This is the income a machine collects on ONE channel when it holds that
    channel at zero and the command does not -- i.e. the achievability ceiling
    of an axis the body cannot produce.
    """
    if half == 0.0:
        return 1.0
    return (std * math.sqrt(math.pi) / (2.0 * half)) * math.erf(half / std)


def mean_kernel_clipped_heading(std: float = STD, n: int = 2_000_001) -> float:
    """E[exp(-(c_w/std)^2)] for the yaw command a NON-TURNING machine sees.

    Under `heading_command=True` with `rel_heading_envs=1.0` the sampled
    `ang_vel_z` is discarded and replaced every step by

        c_w = clip(heading_control_stiffness * wrap_to_pi(psi_target - psi),
                   ang_vel_z[0], ang_vel_z[1])

    (`isaaclab/envs/mdp/commands/velocity_command.py:185-194`). A machine whose
    heading never changes keeps the heading error it drew, and `heading` is
    uniform on (-pi, pi], so c_w = clip(0.5 * U(-pi, pi), -1, +1).

    Quadrature rather than the closed form because of the clip: the clipped mass
    sits at |c_w| = 1 and contributes exp(-4) apiece.
    """
    dpsi = np.linspace(-math.pi, math.pi, n)
    c_w = np.clip(HEADING_STIFFNESS * dpsi, -WZ_CLIP, WZ_CLIP)
    return float(np.trapezoid(np.exp(-((c_w / std) ** 2)), dpsi) / (2.0 * math.pi))


def mean_command_norm(cx_half: float, cy_half: float, n: int = 2001) -> float:
    """E||(c_x, c_y)|| over the command rectangle.

    This is the ONLY command statistic the terrain curriculum reads. In
    `terrain_levels_vel` the demote bar is

        move_down = distance < ||command_xy|| * max_episode_length_s * 0.5

    (`.../velocity/mdp/curriculums.py:42-54`), so the bar is proportional to
    ||c_xy|| and its expectation moves whenever a command range moves. The
    promote bar, `size[0] / 2`, does not. Narrowing `lin_vel_y` therefore edits
    the curriculum, and nothing in the config says so.
    """
    if cy_half == 0.0:
        return cx_half / 2.0  # E|c_x| for c_x ~ U(-h, h)
    # Midpoint rule, not `linspace(...).mean()`: linspace includes both endpoints,
    # which over-weights the corners of the rectangle and gave 0.7656 against the
    # square's analytic 0.7652. Midpoints reproduce the closed form.
    x = -cx_half + (np.arange(n) + 0.5) * (2.0 * cx_half / n)
    y = -cy_half + (np.arange(n) + 0.5) * (2.0 * cy_half / n)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    return float(np.hypot(gx, gy).mean())


def section_1() -> dict[str, float]:
    rule("1. COMMAND GEOMETRY, AND THE DEMOTE BAR NOBODY MEANT TO MOVE")
    yaw_static = mean_kernel_clipped_heading()
    print(f"  yaw kernel, machine that does not turn      {yaw_static:.6f}")
    print("    (c_w = clip(0.5*U(-pi,pi), +/-1); 0005 B1 published 0.2876)")
    print()
    print(f"  {'lin_vel_y range':<24s} {'E|c_xy|':>10s} {'demote bar vs inherited':>26s}")
    norms = {}
    base = mean_command_norm(CX_HALF, 1.0)
    for label, half in CY_HALVES.items():
        norm = mean_command_norm(CX_HALF, half)
        norms[label] = norm
        print(f"  {label:<24s} {norm:10.6f} {norm / base - 1.0:+25.2%}")
    print()
    print(f"  Analytic check on the inherited square: (sqrt(2)+asinh(1))/3 = "
          f"{(math.sqrt(2) + math.asinh(1)) / 3:.6f}")
    collapsed_drop = 1.0 - norms["collapsed  (0.0,  0.0)"] / base
    print(f"  THE COLLAPSE TO (0,0) DROPPED THE DEMOTE BAR BY {collapsed_drop:.2%} "
          f"(claimed {CLAIMED['demote_drop']:.1%})")
    print("  The promote bar (terrain size[0]/2 = 4.0 m) did not move. Any")
    print("  comparison spanning that edit is confounded on terrain difficulty:")
    print("  the same policy demotes less often, so it sits on harder ground.")
    return {"yaw_static": yaw_static, "collapsed_drop": collapsed_drop, **norms}


# ---------------------------------------------------------------------------
# 2. The income ceiling, and the guard's own denominator
# ---------------------------------------------------------------------------
def income_per_step(cy_half: float) -> tuple[float, float]:
    """(lin ceiling, achievable income per step) exactly as `check_hound.py` does.

    Reproduces `check_reward_budget_against_011_and_015` line for line: the lin
    ceiling is `mean_kernel_uniform(half)` on the assumption that the machine
    holds the achievable axis perfectly and eats the unachievable one, the yaw
    ceiling is 1.0, and income is `(w_lin*ceiling + w_ang*1.0) * dt`.
    """
    ceiling = mean_kernel_uniform(cy_half)
    return ceiling, (W_LIN * ceiling + W_ANG * 1.0) * DT


def section_2() -> dict[str, float]:
    rule("2. THE GUARD'S DENOMINATOR IS A FUNCTION OF THE COMMAND RANGE")
    out = {}
    print(f"  {'lin_vel_y range':<24s} {'lin ceiling':>12s} {'income/step':>13s} "
          f"{'vs inherited':>13s}")
    _, base = income_per_step(1.0)
    for label, half in CY_HALVES.items():
        ceiling, income = income_per_step(half)
        out[label] = income
        print(f"  {label:<24s} {ceiling:12.6f} {income:13.6f} {income / base:12.4f}x")
    gain = out["collapsed  (0.0,  0.0)"] / out["inherited (-1.0, +1.0)"]
    print()
    print(f"  A ZERO-WIDTH lin_vel_y INFLATES THE DENOMINATOR BY {gain:.4f}x "
          f"(claimed {CLAIMED['guard_gain']:.2f}x)")
    print("  so every '% of income' the guard prints is that much SMALLER and the")
    print("  30% flag fires that much later -- with no reward weight moving.")
    print("  This is the guard whose stated job is to keep learnings/011 from")
    print("  repeating (check_hound.py PENALTY_BUDGET_FLAG_FRACTION = 0.30).")
    ratio = out["committed (-0.3, +0.3)"] / out["collapsed  (0.0,  0.0)"]
    print()
    print(f"  Income kept by +/-0.3 against the collapse: {ratio:.4f} "
          f"(claimed {CLAIMED['income_ratio_committed_over_collapsed']:.3f} "
          "-- DOES NOT REPRODUCE, see the note in 0006)")
    return out


# ---------------------------------------------------------------------------
# 3. The penalty basket, priced against all three denominators
# ---------------------------------------------------------------------------
def penalty_basket() -> tuple[dict[str, float], float]:
    """`check_hound.py`'s penalty rows, per step, at its own assumed operating point."""
    rows = {
        "dof_acc_l2 (12 legs)": abs(W_DOF_ACC) * N_LEGS * ACC_RMS**2 * DT,
        "dof_torques_l2 (12 legs)": abs(W_DOF_TORQUES) * N_LEGS * TAU_RMS**2 * DT,
        "lin_vel_z_l2": abs(W_LIN_VEL_Z) * VZ_RMS**2 * DT,
        "ang_vel_xy_l2": abs(W_ANG_VEL_XY) * 2 * WXY_RMS**2 * DT,
        "action_rate_l2 (16 actions)": abs(W_ACTION_RATE) * N_ACTIONS * DACT_RMS**2 * DT,
        f"dof_acc_wheel_l2 @ {WHEEL_ACC:.0f} rad/s^2": (
            abs(W_DOF_ACC_WHEEL) * N_WHEELS * WHEEL_ACC**2 * DT
        ),
    }
    return rows, sum(rows.values())


def section_3(incomes: dict[str, float]) -> float:
    rule("3. THE SAME WEIGHTS, THREE VERDICTS")
    rows, total = penalty_basket()
    for label, cost in rows.items():
        print(f"  {label:<34s} {-cost:+.6f}/step")
    print(f"  {'PENALTY SUM':<34s} {-total:+.6f}/step")
    print()
    print(f"  {'lin_vel_y range':<24s} {'basket as % of income':>22s} {'flag?':>7s}")
    for label, income in incomes.items():
        share = total / income
        print(f"  {label:<24s} {share:21.2%} {'FLAG' if share > 0.30 else '-':>7s}")
    print()
    print("  hound_desert_env_cfg.py's docstring records '134.42% to 28.15% of")
    print("  income' for deleting undesired_contacts. 28.15% is the INHERITED row")
    print("  above; the collapse then reported the same basket at 17.66% and the")
    print("  committed +/-0.3 reports it at 19.03%. No weight moved between them.")
    return total


# ---------------------------------------------------------------------------
# 4. Point-and-park: the freeride nobody priced
# ---------------------------------------------------------------------------
def freeride_table(cy_half: float, basket: float, standing_mixture: bool = True) -> dict[str, float]:
    """Per-step net for three behaviours, under one `lin_vel_y` half-width.

    THE MODEL, STATED, because every number below depends on it:

      * With probability `rel_standing_envs = 0.02` the whole command is zeroed
        (`velocity_command.py:196-199`), and all three behaviours then score both
        kernels at 1.0. That branch is priced separately.
      * STANDER: v == 0, omega == 0, heading never changes, so the yaw command
        stays at its drawn value and the yaw kernel is `mean_kernel_clipped_heading`.
        The lin kernel is the product of the x and y channel means (the exponent
        separates and the two commands are independent).
      * POINT-AND-PARK: yaws to the commanded heading, then holds still. In
        steady state the heading error is zero, so ITS OWN YAW COMMAND IS ZERO
        and, being still, it scores the yaw kernel at exactly 1.0. The lin
        kernel is the stander's -- it is not moving either.
      * COMPETENT DRIVER: tracks c_x and c_w exactly, and cannot make c_y (the
        conservative reading, because whether it can is the contested claim of
        section 5-6). Its lin kernel is therefore the y-channel ceiling.

    Penalties: the stander and the parker pay ~0 -- no body rates, no joint
    accelerations, static holding torque only. The driver is charged
    `check_hound.py`'s [ASSUMED] basket. Both driver figures are returned so the
    margin is bracketed rather than asserted.

    `standing_mixture=False` drops the 2% branch. That is not a better model --
    the branch is in the shipped config -- but 0005 B1 and the chat record both
    computed without it, and the flag is how this script reproduces their figures
    instead of merely disagreeing with them.
    """
    yaw_static = mean_kernel_clipped_heading()
    lin_still = mean_kernel_uniform(CX_HALF) * mean_kernel_uniform(cy_half)
    lin_driver = mean_kernel_uniform(cy_half)

    p = REL_STANDING_ENVS if standing_mixture else 0.0
    q = 1.0 - p
    standing_branch = p * (W_LIN * 1.0 + W_ANG * 1.0) * DT

    stander = q * (W_LIN * lin_still + W_ANG * yaw_static) * DT + standing_branch
    park = q * (W_LIN * lin_still + W_ANG * 1.0) * DT + standing_branch
    driver_gross = q * (W_LIN * lin_driver + W_ANG * 1.0) * DT + standing_branch
    nominal_max = (W_LIN + W_ANG) * DT
    return {
        "lin_still": lin_still,
        "lin_driver": lin_driver,
        "stander": stander,
        "park": park,
        "driver_gross": driver_gross,
        "driver_net": driver_gross - basket,
        "stander_frac": stander / nominal_max,
        "park_frac": park / nominal_max,
        "park_over_stander": park / stander,
        "park_over_driver_gross": park / driver_gross,
        "park_over_driver_net": park / (driver_gross - basket),
    }


def section_4(basket: float) -> dict[str, dict[str, float]]:
    rule("4. POINT-AND-PARK, AND WHY LEARNING 015 IS NOT CLOSED")
    print("  Heading mode makes the SCORED yaw command a feedback signal, so a")
    print("  machine that yaws to the commanded heading and then stops drives its")
    print("  own yaw command to zero and collects the yaw kernel at 1.0. A plain")
    print("  stander cannot: it keeps whatever heading error it drew.")
    print()
    out = {}
    for label, half in CY_HALVES.items():
        t = freeride_table(half, basket)
        out[label] = t
        print(f"  --- lin_vel_y {label}")
        print(f"      lin factor: still machine {t['lin_still']:.6f}, "
              f"driver {t['lin_driver']:.6f}; yaw: stander "
              f"{mean_kernel_clipped_heading():.6f}, parker 1.000000")
        print(f"      {'stander':<22s} {t['stander']:+.6f}/step   "
              f"{t['stander_frac']:6.2%} of nominal max")
        print(f"      {'POINT-AND-PARK':<22s} {t['park']:+.6f}/step   "
              f"{t['park_frac']:6.2%} of nominal max")
        print(f"      {'driver, gross':<22s} {t['driver_gross']:+.6f}/step")
        print(f"      {'driver, net of basket':<22s} {t['driver_net']:+.6f}/step")
        print(f"      park / stander           {t['park_over_stander']:.3f}x   "
              f"park / driver  {t['park_over_driver_gross']:.2%} gross, "
              f"{t['park_over_driver_net']:.2%} net")
        print()

    bare = freeride_table(1.0, basket, standing_mixture=False)
    print("  MACHINERY CHECK. 0005 B1 published the additive stander at 0.2255 of")
    print("  nominal max on the INHERITED command set, computed independently by a")
    print("  different session. With the 2% standing branch dropped -- which is how")
    print(f"  0005 computed it -- this script gets {bare['stander_frac']:.4f}. "
          "The machinery agrees.")
    print(f"  On that same command set POINT-AND-PARK gets {bare['park_frac']:.4f}, "
          f"{bare['park_frac'] / bare['stander_frac']:.2f}x the")
    print("  fake 0005 priced, and it BEATS the competent driver net of the assumed")
    print(f"  basket: {bare['park']:.6f} against {bare['driver_net']:.6f}/step "
          f"({bare['park_over_driver_net']:.1%}).")
    print("  THE FENCE WAS PRICED AGAINST THE WRONG FREERIDE.")
    print()
    print("  WHICH RANGE DID THE CHAT RECORD'S TRIPLE COME FROM? Its stander and")
    print("  parker are matched here against all three, with the 2% branch kept:")
    print(f"      {'range':<24s} {'stander delta':>15s} {'park delta':>13s}")
    for label, half in CY_HALVES.items():
        t = out[label]
        print(f"      {label:<24s} {t['stander'] - CLAIMED['stander_net']:+15.6f} "
              f"{t['park'] - CLAIMED['park_net']:+13.6f}")
    print("  It came from the COLLAPSED (0,0) RANGE -- both match to 5.8e-5, a")
    print("  constant offset, while the committed +/-0.3 range is out by 8.8e-4.")
    print("  So the headline 92.4% describes a command range the tree no longer")
    print("  has (`b2e2634` replaced it). Under the number rule the claimed triple")
    print("  is barred and the rows above are what 0006 cites.")
    print()
    committed = out["committed (-0.3, +0.3)"]
    print("  What reproduces on EVERY range, to five significant figures, is the")
    print(f"  difference park - stander: claimed 0.006984, computed "
          f"{committed['park'] - committed['stander']:.6f}.")
    print("  It is range-independent because the lin term cancels: the whole")
    print("  mechanism lives on the yaw channel, and the yaw half is exactly right.")
    q = 1.0 - REL_STANDING_ENVS
    implied_ceiling = (
        CLAIMED["driver_net"] + basket - REL_STANDING_ENVS * (W_LIN + W_ANG) * DT
    ) / (q * DT) - W_ANG
    print(f"  The claimed driver net {CLAIMED['driver_net']:.6f} reproduces on no range.")
    print(f"  Net of the same basket it implies a lin ceiling of "
          f"{implied_ceiling:.4f}, which is not")
    print("  any committed command range (those are 0.441041, 0.891923, 1.000000).")
    print("  So the claimed 92.4% is a ratio between a stander priced on one")
    print("  command range and a driver priced on another. Barred.")
    print()
    print("  EITHER WAY THE CONCLUSION HOLDS. On the committed range point-and-park")
    print(f"  earns {committed['park_over_stander']:.2f}x the stander and "
          f"{committed['park_over_driver_gross']:.0%}-{committed['park_over_driver_net']:.0%} "
          "of a competent driver's net;")
    print(f"  on the inherited range it earns {bare['park_over_driver_net']:.0%} of it, "
          "i.e. more than the driver.")
    print("  Nothing in the contact term or the command range moves that, because")
    print("  the parker touches the ground exactly as a driver does and its income")
    print("  is bought entirely on the yaw channel. The axis is the LIN/YAW WEIGHT")
    print("  RATIO (and the sign of what the yaw term scores), not the fence.")
    return out


# ---------------------------------------------------------------------------
# 4b. The claimed triple, reproduced exactly, with its assumptions named
# ---------------------------------------------------------------------------
def mean_kernel_gaussian(sigma: float, std: float = STD) -> float:
    """E[exp(-e^2/std^2)] for zero-mean Gaussian error of standard deviation sigma.

        E[K] = 1 / sqrt(1 + 2 sigma^2 / std^2)

    This is the "expectation arm" of the kernel, the distinction
    `docs/theory/reward-composition.md`'s appended refutation insists on: a mean
    kernel is not the kernel of a mean, and quoting one against the other is how
    that note's alpha_w window came out empty.
    """
    return 1.0 / math.sqrt(1.0 + 2.0 * sigma**2 / std**2)


def mean_kernel_relative(rho: float, half: float = CX_HALF, std: float = STD) -> float:
    """E[exp(-(rho*c/std)^2)] for c uniform on [-half, half]: a tracker at constant
    RELATIVE error rho. At rho = 1 the achieved speed is zero, so this collapses
    onto `mean_kernel_uniform` -- the identity that makes rho = 1 *be* the stander
    on the speed channel.
    """
    if rho < 1e-9:
        return 1.0
    return (math.sqrt(math.pi) / 2.0) * (std / (rho * half)) * math.erf(rho * half / std)


def section_4b(basket: float) -> None:
    rule("4b. THE CLAIMED TRIPLE, REPRODUCED EXACTLY -- AND WHAT IT ASSUMES")
    print("  The chat record's triple IS reproducible, from four inputs that section 4")
    print("  deliberately does not adopt. Reproducing it is worth more than")
    print("  disagreeing with it, because it turns the headline into a citable number")
    print("  WITH its assumptions attached instead of a number with none.")
    print()

    # (i) The still machine's own static-stance floor. DERIVED from Spec: holding
    #     the solved stance costs the three static joint torques, and a still
    #     machine's only other cost is the deterministic action-rate component.
    tau = SPEC.static_torques()
    pen_still = (
        abs(W_DOF_TORQUES) * 4 * sum(t**2 for t in tau.values())
        + abs(W_ACTION_RATE) * N_ACTIONS * DACT_RMS**2
    ) * DT
    print("  (i) the STILL machine is not free: holding the solved stance costs the")
    print("      three static joint torques, from SPEC.static_torques() --")
    print(f"      abduct {tau['abduct']:+.4f}, hip {tau['hip']:+.4f}, "
          f"knee {tau['knee']:+.4f} N*m --")
    print(f"      so a stander pays {pen_still:.6f}/step, not zero. Section 4 charges")
    print("      it nothing, which flatters the fake; this is the tighter reading.")
    print()

    # (ii) c_y == 0: a still machine has v_y = 0 exactly, so the y channel pays 1.0.
    lin_still = mean_kernel_uniform(CX_HALF) * 1.0
    # (iii) the driver's competence, taken as rho = 0.46.
    sigma = 0.10
    lin_driver = mean_kernel_relative(0.46) * mean_kernel_gaussian(sigma)
    yaw_driver = mean_kernel_gaussian(sigma)
    # (iv) a PROPOSED table's basket, not the committed one.
    proposed_basket = (
        1.0 * 0.05                      # a contact term at 5% duty; not in the committed table
        + 1.0e-7 * N_LEGS * ACC_RMS**2  # dof_acc_l2 at -1e-7, not the committed -2.5e-7
        + 2.0 * VZ_RMS**2
        + 0.05 * 2 * WXY_RMS**2
        + 1.0e-5 * N_LEGS * 8.0**2      # tau_rms 8.0 N*m, not check_hound.py's 5.0
        + 2.5e-9 * N_WHEELS * WHEEL_ACC**2
        + 0.01 * N_ACTIONS * DACT_RMS**2
    ) * DT

    p, q = REL_STANDING_ENVS, 1.0 - REL_STANDING_ENVS

    def income(lin: float, yaw: float) -> float:
        return (q * (W_LIN * lin + W_ANG * yaw) + p * (W_LIN + W_ANG)) * DT

    yaw_static = mean_kernel_clipped_heading()
    stander = income(lin_still, yaw_static) - pen_still
    park = income(lin_still, 1.0) - pen_still
    driver = income(lin_driver, yaw_driver) - proposed_basket

    print(f"      {'':<18s} {'computed':>11s} {'claimed':>11s} {'delta':>11s}")
    for label, got, claim in (
        ("stander", stander, CLAIMED["stander_net"]),
        ("POINT-AND-PARK", park, CLAIMED["park_net"]),
        ("driver", driver, CLAIMED["driver_net"]),
    ):
        print(f"      {label:<18s} {got:11.6f} {claim:11.6f} {got - claim:+11.2e}")
    print(f"      {'park / driver':<18s} {park / driver:11.4f} "
          f"{CLAIMED['park_share_of_driver']:11.4f} "
          f"{park / driver - CLAIMED['park_share_of_driver']:+11.2e}")
    print(f"      {'driver / park':<18s} {driver / park:11.4f}")
    print(f"      {'park / stander':<18s} {park / stander:11.4f}")
    print()
    print("  ALL THREE REPRODUCE TO 1e-6. So the figures are citable -- provided the")
    print("  four assumptions travel with them, and two of them are defects:")
    print()
    print("  1. c_y == 0. The y channel pays a still machine exactly 1.0, which is")
    print("     the COLLAPSED range `b2e2634` replaced. On the committed +/-0.3 a")
    print("     still machine's y factor is "
          f"{mean_kernel_uniform(0.3):.4f}, not 1.0.")
    print("  2. rho = 0.46 as the driver's competence. THE SAME REFUTATION NAMES")
    print("     THIS AS A DEFECT in its own provenance list: reward-composition.md's")
    print("     appended refutation established that rho = |0.271 - 0.50| / 0.50 is")
    print("     the relative error OF THE FIXED TROT. Note the identity that makes")
    print("     this fatal rather than sloppy: at rho = 1 the relative-error kernel")
    print(f"     is {mean_kernel_relative(1.0):.6f} and the stander's own lin factor is "
          f"{mean_kernel_uniform(CX_HALF):.6f}")
    print("     -- rho = 1 IS the stander on the speed channel. So the 'driver' is")
    print("     parameterised on the same axis as the fake it is being compared to.")
    print(f"  3. A PROPOSED table's penalty basket: {proposed_basket:.6f}/step against the")
    print(f"     committed table's {basket:.6f}/step. It charges dof_acc_l2 at -1e-7")
    print("     (committed: -2.5e-7), 8.0 N*m rms torque (check_hound.py: 5.0), and a")
    print("     contact term at 5% duty that the committed table does not have at all.")
    print("  4. The static-stance floor (i), which section 4 omits.")
    print()
    print("  WHAT THIS MEANS FOR THE MARGIN. The claimed 1.08x driver-over-parker is")
    print("  a margin between a parker priced on the collapsed range and a driver")
    print("  scored at the fixed trot's own relative error under a table that was")
    print("  never committed. The direction survives every one of those corrections")
    print("  -- section 4 gets 65%-103% on the committed table -- but the 1.08x")
    print("  itself must never be quoted without all four.")


# ---------------------------------------------------------------------------
# 5. The axle, by forward kinematics on the committed model
# ---------------------------------------------------------------------------
def axle_sweep(n: int = 13) -> dict[str, float]:
    """Largest |x-component| of any wheel axle, over the full joint range.

    Forward kinematics on `assets/hound16pd.xml` -- the file a MuJoCo run
    actually loads -- rather than on the algebra, so the claim is about the
    committed machine and not about an intended one. `mj_kinematics` only; no
    dynamics, no stepping.

    The wheel's spin axis is (0,1,0) in its own body frame, so the axle in the
    trunk frame is column 1 of the wheel body's rotation matrix, with the trunk
    at identity.
    """
    import mujoco

    from bestiary import paths

    model = mujoco.MjModel.from_xml_path(str(paths.HOUND_PD_XML))
    data = mujoco.MjData(model)

    qadr = {}
    for leg in ("FL", "FR", "RL", "RR"):
        for joint in ("abduct", "hip", "knee", "wheel"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}")
            qadr[(leg, joint)] = model.jnt_qposadr[jid]

    grid = {
        "abduct": np.linspace(*SPEC.abduct_range, n),
        "hip": np.linspace(*SPEC.hip_range, n),
        "knee": np.linspace(*SPEC.knee_range, n),
        "wheel": np.linspace(-math.pi, math.pi, n),
    }
    worst = 0.0
    tested = 0
    for a in grid["abduct"]:
        for h in grid["hip"]:
            for k in grid["knee"]:
                for w in grid["wheel"]:
                    for leg in ("FL", "FR", "RL", "RR"):
                        data.qpos[qadr[(leg, "abduct")]] = a
                        data.qpos[qadr[(leg, "hip")]] = h
                        data.qpos[qadr[(leg, "knee")]] = k
                        data.qpos[qadr[(leg, "wheel")]] = w
                    mujoco.mj_kinematics(model, data)
                    for leg in ("FL", "FR", "RL", "RR"):
                        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel")
                        axle = data.xmat[bid].reshape(3, 3)[:, 1]
                        worst = max(worst, abs(float(axle[0])))
                    tested += 4
    return {"max_axle_x": worst, "configs": tested}


def section_5() -> dict[str, float]:
    rule("5. NO JOINT STEERS -- THE ALGEBRA HOLDS")
    print("  Read from the committed MJCF: abduct turns about (1,0,0) and hip,")
    print("  knee and wheel all about (0,1,0), on all four legs. R_y leaves")
    print("  (0,1,0) invariant, so the axle in the trunk frame is")
    print("  R_x(phi)*(0,1,0) = (0, cos phi, sin phi) for ANY hip/knee/wheel.")
    out = axle_sweep()
    print(f"  FK over {out['configs']:,} wheel configurations spanning the full")
    print(f"  joint range: max |axle_x| = {out['max_axle_x']:.3e}")
    print("  That is float64 noise. The impossibility half of the proof is right.")
    return out


# ---------------------------------------------------------------------------
# 6. But the rolling DIRECTION is not sagittal
# ---------------------------------------------------------------------------
def lateral_fraction(phi: float, slope_rad: float) -> float:
    """|d_y| / |d| for the rolling direction d = axle x normal, body frame.

    With axle a = (0, cos phi, sin phi) and ground normal n = (n_x, n_y, n_z),

        d = a x n = (cos phi * n_z - sin phi * n_y,
                     sin phi * n_x,
                    -cos phi * n_x)

    so d_y = sin(phi) * n_x, which is nonzero whenever the trunk's pitch differs
    from the local ground plane. That is a POSTURE, not a terrain property: the
    twelve leg joints set trunk pitch independently of the ground, so n_x != 0 is
    reachable on flat ground too.

    Taking n = (-sin s, 0, cos s) for a pitch offset s:
        |d_y|/|d| = sin(phi) sin(s) / sqrt(cos^2 phi + sin^2 phi sin^2 s)
    """
    a = np.array([0.0, math.cos(phi), math.sin(phi)])
    n = np.array([-math.sin(slope_rad), 0.0, math.cos(slope_rad)])
    d = np.cross(a, n)
    return float(abs(d[1]) / np.linalg.norm(d))


def slip_constraint_nullspace(phi: float) -> dict[str, object]:
    """Rank and nullspace dimension of the four no-lateral-slip constraints.

    For wheel i with axle a_i and contact point r_i (trunk frame), rolling
    without lateral slip means the contact-point velocity has no component along
    the axle:

        a_i . (v + omega x r_i) = 0   ->   [a_i^T, (r_i x a_i)^T] [v; omega] = 0

    Four wheels give a 4x6 matrix. THE ABDUCT AXIS IS (1,0,0) ON ALL FOUR LEGS
    (read from the MJCF in section 5), so a COMMON abduct roll gives all four
    wheels the SAME axle, the four rows differ only in their second block, and
    the rank collapses. Whatever the nullspace dimension is, pure translation
    perpendicular to the common axle is in it, because a . v = 0 with omega = 0.
    """
    a = np.array([0.0, math.cos(phi), math.sin(phi)])
    # Contact points: axle position minus the wheel radius along the downward
    # direction perpendicular to the axle. Geometry from Spec; only the four
    # positions matter, and the rank is re-checked below under a random
    # perturbation of them, because a rank that depends on exact symmetry is not
    # a claim about the machine.
    down = np.array([0.0, 0.0, -1.0])
    down = down - np.dot(down, a) * a
    down /= np.linalg.norm(down)
    axles = [
        np.array([sx * SPEC.hip_x, sy * 0.11, -SPEC.axle_drop])
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]
    contacts = [r + SPEC.wheel_r * down for r in axles]
    m = np.array([np.concatenate([a, np.cross(r, a)]) for r in contacts])
    rank = int(np.linalg.matrix_rank(m, tol=1e-9))
    rng = np.random.default_rng(0)
    jittered = np.array(
        [
            np.concatenate([a, np.cross(r + 0.02 * rng.standard_normal(3), a)])
            for r in contacts
        ]
    )
    perturbed = int(np.linalg.matrix_rank(jittered, tol=1e-9))
    # A pure lateral-in-the-ground-plane twist: the rolling direction itself.
    n = np.array([0.0, 0.0, 1.0])
    d = np.cross(a, n)
    residual = float(np.abs(m @ np.concatenate([d, np.zeros(3)])).max())
    return {
        "rank": rank,
        "nullspace_dim": 6 - rank,
        "rank_perturbed": perturbed,
        "rolling_dir_residual": residual,
    }


def section_6() -> dict[str, float]:
    rule("6. THE ROLLING DIRECTION LEAVES THE SAGITTAL PLANE -- THE INFERENCE DOES NOT HOLD")
    print("  d = axle x normal, and d_y = sin(phi) * n_x. |d_y|/|d|, in percent:")
    phis = [0.2, 0.4, 0.6, 0.8]
    slopes_deg = [10.0, 18.4, 20.0, 38.8, 45.0]
    header = "  phi \\ pitch " + "".join(f"{s:>8.1f}d" for s in slopes_deg)
    print(header)
    for phi in phis:
        cells = "".join(
            f"{lateral_fraction(phi, math.radians(s)):>8.1%} " for s in slopes_deg
        )
        print(f"  {phi:>11.2f} {cells}")
    wedge = lateral_fraction(0.8, math.radians(20.0))
    print()
    print("  At the abduct limit phi = 0.8 rad on a 20 deg (0.35 rad) pitch")
    print(f"  offset: |v_y|/|v| = {wedge:.4f}  (claimed {CLAIMED['lateral_wedge']:.3f})")
    print("  18.4 deg and 38.8 deg are 0005 B6's MEASURED median tile slopes at")
    print("  terrain difficulty 0.25 and 0.50, so the middle columns are the")
    print("  ground this machine is actually being trained on.")
    print()
    ns = slip_constraint_nullspace(0.8)
    print(f"  No-lateral-slip constraint matrix (4x6): rank {ns['rank']}, "
          f"NULLSPACE DIMENSION {ns['nullspace_dim']}")
    print(f"  The rolling direction itself is in it: max |A x| = "
          f"{ns['rolling_dir_residual']:.3e}")
    print("  So sustained lateral motion is available by PURE ROLLING, with no")
    print("  stepping gait and no slip. The (0,0) collapse deleted a capability.")
    return {"wedge": wedge, "nullspace_dim": float(ns["nullspace_dim"])}


# ---------------------------------------------------------------------------
# 7. What the y channel charges when c_y == 0
# ---------------------------------------------------------------------------
def section_7() -> None:
    rule("7. WITH c_y == 0 THE y CHANNEL IS NOT A SKID PRICE")
    print("  With the command pinned to zero, e_y is whatever lateral velocity the")
    print("  machine has, and the fraction of the LARGEST reward term it destroys")
    print("  is 1 - exp(-(v_y/std)^2). But v_y is not slip -- section 6 shows it is")
    print("  what a pitched, abducted, purely-rolling machine produces. So the")
    print("  charge lands on stance-widening while pitched, which is the posture")
    print("  that stabilises a wheeled machine on a dune face.")
    print()
    print("  Cost as a fraction of track_lin_vel_xy_exp, at phi = 0.8 rad:")
    print(f"  {'speed |v| m/s (ft/s)':<24s}" + "".join(
        f"{s:>10.1f}d" for s in (18.4, 20.0, 38.8)))
    for v in (0.20, 0.30, 0.50, 0.80):
        cells = ""
        for s in (18.4, 20.0, 38.8):
            vy = lateral_fraction(0.8, math.radians(s)) * v
            cells += f"{1.0 - math.exp(-((vy / STD) ** 2)):>10.2%} "
        print(f"  {v:<6.2f} ({v * 3.2808:>4.2f})        {cells}")
    print()
    print("  0.20 m/s is the terrain curriculum's own promote threshold")
    print("  (4.0 m of displacement over a 20 s episode), so the top row is the")
    print("  slowest speed at which this machine is allowed to progress at all.")


# ---------------------------------------------------------------------------
# 8. Wheel-torque economics: nothing prices a resting limb
# ---------------------------------------------------------------------------
def section_8() -> dict[str, float]:
    rule("8. 'SELF-PRICING THROUGH PHYSICS' IS REFUTED")
    weight = SPEC.total_mass * 9.81
    normal = weight / 4.0
    cap = SPEC.wheel_friction[0] * normal * SPEC.wheel_r
    a_sat = 2.0          # CARD.md: measured saturation
    a_wheelie = weight * SPEC.hip_x / SPEC.stand_z / SPEC.total_mass  # CARD: 5.22
    demand_sat = (SPEC.total_mass * a_sat / 4.0) * SPEC.wheel_r
    demand_wheelie = (SPEC.total_mass * a_wheelie / 4.0) * SPEC.wheel_r
    cruise = SPEC.wheel_brake_torque()
    print(f"  static load per wheel                 {normal:8.3f} N")
    print(f"  friction cap, mu = {SPEC.wheel_friction[0]}                 "
          f"{cap:8.4f} N*m/wheel   (CARD.md: 3.19)")
    print(f"  gear_wheel                            {SPEC.gear_wheel:8.4f} N*m       "
          f"({SPEC.gear_wheel / cap:.1%} of the cap)")
    print()
    print(f"  demand at the MEASURED saturation {a_sat:.1f} m/s^2   "
          f"{demand_sat:8.4f} N*m   {demand_sat / cap:6.2%} of the cone "
          f"(claimed {CLAIMED['cone_at_saturation']:.1%})")
    print(f"  demand at the CARD wheelie limit {a_wheelie:.2f} m/s^2  "
          f"{demand_wheelie:8.4f} N*m   {demand_wheelie / cap:6.2%} of the cone")
    print(f"  demand at STEADY CRUISE (frictionloss only) {cruise:8.4f} N*m   "
          f"{cruise / cap:6.2%} of the cone "
          f"(claimed {CLAIMED['cone_at_cruise']:.1%})")
    print()
    print("  NOTE A DISAGREEMENT INSIDE THE RECORD. CARD.md and")
    print(f"  robots/hound/check.py:276 both say the cone is 'barely "
          f"{CLAIMED['cone_used_per_card']:.0%} used' at")
    print("  saturation; the static-load arithmetic above says "
          f"{demand_sat / cap:.1%}. Both agree the")
    print("  cone does not bind -- top speed is set by the drive's velocity limit")
    print("  and thrust by a WHEELIE (check.py:321, 'the limit is UNLOADING, not")
    print("  the friction cone') -- but the two figures are not the same number and")
    print("  neither has an instrument recorded. 0006 cites the computed one.")
    print()
    for resting_n in (20.0, 40.0, 60.0):
        held = weight - resting_n
        cap_unloaded = SPEC.wheel_friction[0] * (held / 4.0) * SPEC.wheel_r
        print(f"  a limb resting at {resting_n:5.1f} N ({resting_n / weight:5.1%} of weight): "
              f"cone {cap_unloaded:.4f} N*m/wheel, cruise uses "
              f"{cruise / cap_unloaded:.1%}, saturation {demand_sat / cap_unloaded:.1%}")
    print()
    print("  So unloading the wheels does NOT price a resting limb: the cone is")
    print("  still slack by a factor of several. And no reward term charges wheel")
    print("  torque at all -- dof_torques_l2 is scoped to LEG_JOINT_EXPR, and the")
    print("  only wheel-scoped term is dof_acc_wheel_l2, which prices ACCELERATION")
    print("  and is therefore exactly zero at constant speed:")
    print(f"    dof_acc_wheel_l2 at 0 rad/s^2 = "
          f"{abs(W_DOF_ACC_WHEEL) * N_WHEELS * 0.0 * DT:.6f}/step")
    print("  A policy cruising with a limb leaning on the ground earns full income")
    print("  at zero cost. The exploit is live and nothing in the table sees it.")
    return {"cap": cap, "cone_at_saturation": demand_sat / cap, "cone_at_cruise": cruise / cap}


# ---------------------------------------------------------------------------
# 9. The count-form contact term, and why one re-enabling is dangerous
# ---------------------------------------------------------------------------
def section_9(incomes: dict[str, float]) -> dict[str, float]:
    rule("9. THE CONTACT TERM: COUNT FORM, DETERRENCE WEIGHT, AND SELF-TERMINATION")
    income = incomes["committed (-0.3, +0.3)"]
    print("  FORM. `undesired_contacts` sums BOOLEANS (rewards.py:272-282), so its")
    print("  per-step charge is bounded by n_bodies * |w| * dt. `contact_forces`")
    print("  takes max-over-HISTORY, subtracts the threshold and clips only at")
    print("  min=0 (rewards.py:296-306) -- NO UPPER CLIP -- so a solver impact")
    print("  spike enters the return unbounded:")
    for peak in (200.0, 1_000.0, 5_000.0, 50_000.0):
        cost = 1.5e-4 * max(0.0, peak - 100.0) * DT
        print(f"    contact_forces @ -1.5e-4, one body, {peak:9,.0f} N peak: "
              f"{cost:.6f}/step = {cost / income:7.2%} of income")
    print("  The count form is the safer object. That part of the design survives.")
    print()
    w_min = 0.5 * income / DT
    print("  DETERRENCE. For one contacting body to cost at least half of")
    print(f"  achievable income, w * dt >= 0.5 * I with I = {income:.6f}/step:")
    print(f"    w >= {w_min:.4f}   (claimed {CLAIMED['deterrence_weight']:.4f})")
    print()
    print("  DANGEROUS. Re-enabling undesired_contacts at -1.0 over the leg bodies:")
    for n_bodies in (1, 12, 13):
        cost = 1.0 * n_bodies * DT
        print(f"    {n_bodies:2d} bodies in contact: {-cost:+.4f}/step = "
              f"{cost / income:8.2%} of income  "
              f"(12-body figure claimed at {CLAIMED['undesired_12_bodies']:.2f}x)"
              if n_bodies == 12 else
              f"    {n_bodies:2d} bodies in contact: {-cost:+.4f}/step = "
              f"{cost / income:8.2%} of income")
    fallen = 12 * 1.0 * DT
    implied = 12 * 1.0 * DT / CLAIMED["undesired_12_bodies"]
    print("  (12 = the leg links; 13 adds the trunk, as robot_lab's")
    print("   '^(?!.*_foot).*' would resolve on this body.)")
    print()
    print("  THE TWO CLAIMED FIGURES ABOVE SHARE ONE DEFECT AND IT IS WORTH NAMING.")
    print(f"  {CLAIMED['deterrence_weight']:.4f} implies I = "
          f"{CLAIMED['deterrence_weight'] * DT * 2:.6f}/step and "
          f"{CLAIMED['undesired_12_bodies']:.2f}x implies I = {implied:.6f}/step --")
    print("  the same income, and it is not an income this script can derive from")
    print("  any committed command range (the three are 0.018821, 0.027838,")
    print("  0.030000). Both are therefore barred; the computed rows stand.")
    print()
    print("  AND THE INHERITED TABLE HAS NO TERMINATION REWARD TERM AT ALL")
    print("  (velocity_env_cfg.RewardsCfg, read 2026-07-29: eleven terms, none of")
    print("  them is_terminated), so V(terminate) = 0 EXACTLY. A fallen machine")
    print(f"  paying {fallen:.2f}/step therefore has negative value, and flopping the")
    print("  trunk down -- which `terminations.base_contact` accepts -- is optimal.")
    ratio = fallen / income
    print(f"  Being down costs {ratio:.2f}x per step what being up EARNS, so a get-up")
    print("  only pays if it finishes inside R / (1 + ratio) of the remaining")
    print("  episode time R:")
    for remaining in (20.0, 15.0, 10.0, 5.0):
        print(f"    {remaining:5.1f} s left -> get-up must complete in "
              f"{remaining / (1.0 + ratio):.2f} s")
    print("  Nothing in this repository has demonstrated a Hound get-up at all.")
    return {"fallen_share": ratio, "deterrence_w": w_min}


# ---------------------------------------------------------------------------
# 10. Provenance defects, arithmetic only
# ---------------------------------------------------------------------------
def section_10() -> None:
    rule("10. PROVENANCE DEFECTS IN THE REFUTED PASS")
    print(f"  'one tap per second is 4% of steps' -- at dt = {DT} s a second is")
    print(f"  {1 / DT:.0f} steps, so one tap per second is {DT:.0%} of steps, not 4%.")
    print()
    print("  '2.0 m/s^2, wheelie-limited [VERIFIED]' -- CARD.md's traction budget")
    print("  gives the WHEELIE limit as 88.78 N -> 5.22 m/s^2 and reports 2.0 m/s^2")
    print("  separately as MEASURED SATURATION. They are two different numbers:")
    weight = SPEC.total_mass * 9.81
    print(f"    wheelie limit  = M g hip_x / stand_z = {weight:.2f} * "
          f"{SPEC.hip_x} / {SPEC.stand_z:.4f} = "
          f"{weight * SPEC.hip_x / SPEC.stand_z:.2f} N -> "
          f"{weight * SPEC.hip_x / SPEC.stand_z / SPEC.total_mass:.2f} m/s^2")
    print("    measured saturation ~2.0 m/s^2 (robots/hound/check.py, torque sweep)")
    print("  Labelling the second one 'wheelie-limited [VERIFIED]' attaches a")
    print("  measurement's authority to a mechanism the measurement contradicts.")
    print()
    print("  'init_std = 1.0 [ASSUMED]' -- it is read from source:")
    print("    anymal_c/agents/rsl_rl_ppo_cfg.py:23,")
    print("    GaussianDistributionCfg(init_std=1.0). Marking a source-read value")
    print("    ASSUMED is the mirror of the usual defect and just as costly: it")
    print("    invites a later cycle to spend a probe re-deriving a config literal.")
    print()
    print("  'rho = 0.46, the driver's relative control error' -- reward-")
    print("  composition.md's own appended refutation already established that")
    print("  rho = |0.271 - 0.50| / 0.50 is the relative error OF THE FIXED TROT,")
    print("  not of a tracker. A command-following policy's relative control error")
    print("  has still never been measured. Resurrecting it as competence compares")
    print("  the fake against itself.")


def main() -> None:
    print(__doc__.splitlines()[0])
    print(f"Hound: {SPEC.total_mass:.3f} kg, wheel r = {SPEC.wheel_r} m, "
          f"dt = {DT} s, std^2 = {STD ** 2}")
    section_1()
    incomes = section_2()
    basket = section_3(incomes)
    section_4(basket)
    section_4b(basket)
    section_5()
    section_6()
    section_7()
    section_8()
    section_9(incomes)
    section_10()
    print()
    print("Every figure `research/decisions/0006` cites appears above. Figures")
    print("printed as 'claimed' are NOT cited: they are recorded so the record")
    print("shows what moved and why.")


if __name__ == "__main__":
    main()
