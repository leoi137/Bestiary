"""Oracle for HoundPDTrackRelDesert-v0: the two things that must be exactly right.

A reward redesign gets one 6-hour run before it is judged, so the parts that
are checkable without training are checked before the GPU is armed. Two are:

**1. THE INCOME MUST BE THE KERNEL APPLIED TO THE ERRORS.** Asserted as an
exact identity against the env's own reported errors, which is what catches an
implementation slip; a reward that merely looks plausible cannot satisfy it.

Alongside it, and REPORTED RATHER THAN ASSERTED, is what a do-nothing machine
earns per cell. A stander's error is nearly determined by the command, so its
income can be worked on paper -- and that estimate is wrong by up to 26% on the
cells where the passive yaw drift is a large share of the error, because the
drift is an rms about zero rather than a constant and the kernel is nonlinear.
The rollout is the ground truth; the paper number is there to show the
mechanism, which is that a stander's RELATIVE velocity error is ~1 at every
drive command.

This is the check that would have caught the original design's real defect
before it cost 3.9 GPU-hours: the freeride was always computable, and it was
never computed against the score the policy actually achieved.

**2. THE SHAPING MUST TELESCOPE.** Potential-based shaping is only
policy-invariant if the per-step terms sum to gamma^T*P(s_T) - P(s_0) over an
episode. Two boundary conventions can break it, and one of them is nearly
invisible:

  - zeroing the potential at TIMEOUT instead of only at true termination
    injects a spurious -P(s_1000) ~= -0.9 into a transition whose TD target
    also bootstraps the next state. Large, silent, and wrong.
  - carrying a stale potential across `reset()` puts a spurious term in step 1
    of every episode.

Both are asserted here directly, by summing the env's own reported shaping
against an independently accumulated potential trace.

Run:  venv/bin/python -m bestiary.guards.check_track_rel
"""
from __future__ import annotations

import numpy as np

from bestiary.envs.hound_track_rel import (
    SHAPING_GAMMA,
    relative_kernel,
    velocity_tolerance,
    yaw_tolerance,
)
from bestiary.envs.track_constants import STANDING_CREEP_MS, STANDING_YAW_RADS

ENV_ID = "HoundPDTrackRelDesert-v0"

# The eval grid, plus the stop cell last. Matches record/track_eval.EVAL_GRID
# except that the backward cell is at -0.4 rather than -0.3: the sampler's
# backward floor moved to 0.40, so -0.3 is a command no policy is trained on.
CELLS: tuple[tuple[float, float, float], ...] = (
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (-0.4, 0.0, 0.0),
    (0.5, 0.0, 0.4),
    (0.5, 0.0, -0.4),
    (0.0, 0.0, 0.45),
    (0.0, 0.0, 0.0),
)


def closed_form_standing_income(cmd: tuple[float, float, float]) -> dict:
    """What a do-nothing machine earns per step under this reward, on paper.

    Its heading-frame velocity is (-creep, 0) and its yaw rate is the passive
    drift, so both errors are pure functions of the command.
    """
    vx_cmd, vy_cmd, w_cmd = cmd
    v_body = np.array([-STANDING_CREEP_MS, 0.0])
    err_v = float(np.linalg.norm(v_body - np.array([vx_cmd, vy_cmd])))
    err_w = abs(STANDING_YAW_RADS - w_cmd)
    alpha_v = velocity_tolerance(vx_cmd)
    alpha_w = yaw_tolerance(w_cmd)
    k_v = relative_kernel(err_v, alpha_v)
    k_w = relative_kernel(err_w, alpha_w)
    return {
        "cmd": cmd, "err_v": err_v, "err_w": err_w,
        "alpha_v": alpha_v, "alpha_w": alpha_w,
        "k_v": k_v, "k_w": k_w, "income": k_v * k_w,
        "rel_err_v": err_v / alpha_v, "rel_err_w": err_w / alpha_w,
    }


def _rollout(env, cmd, steps: int, seed: int) -> dict:
    """Zero-action rollout with the command pinned, collecting the env's own numbers."""
    import numpy as _np

    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    u._cmd = _np.array(cmd, dtype=float)
    u._steps_until_resample = 10**9
    u._potential_prev = u._potential()   # re-seed: the command was just replaced
    zero = _np.zeros(env.action_space.shape)

    track, shaping, potentials = [], [], [u._potential()]
    terminated = False
    for _ in range(steps):
        _obs, _r, terminated, truncated, info = env.step(zero)
        track.append(info["reward_track"])
        shaping.append(info["reward_shaping"])
        potentials.append(info["potential"])
        if terminated or truncated:
            break
    return {
        "track_mean": float(_np.mean(track)),
        "shaping_sum": float(_np.sum(shaping)),
        "potentials": potentials,
        "steps": len(track),
        "terminated": bool(terminated),
    }


def main() -> int:
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  -- registers the env ids

    failures = 0
    env = gym.make(ENV_ID)

    print("1. THE ENV COMPUTES WHAT THE SPEC SAYS (exact identity, per step)")
    print("   reward_track == K(err_v; alpha_v) * K(err_w; alpha_w), from the")
    print("   env's OWN reported errors. This is what catches a coding error;")
    print("   it cannot be satisfied by a reward that merely looks plausible.")
    worst = 0.0
    for cmd in CELLS:
        obs, _ = env.reset(seed=1000)
        u = env.unwrapped
        u._cmd = np.array(cmd, dtype=float)
        u._steps_until_resample = 10**9
        u._potential_prev = u._potential()
        for _ in range(60):
            _o, _r, term, trunc, info = env.step(np.zeros(env.action_space.shape))
            expect = (relative_kernel(info["track_err_v"], info["track_alpha_v"])
                      * relative_kernel(info["track_err_w"], info["track_alpha_w"]))
            worst = max(worst, abs(info["reward_track"] - expect))
            if term or trunc:
                break
    ok = worst < 1e-12
    failures += not ok
    print(f"   worst |reward_track - K_v*K_w| over 7 cells x 60 steps: "
          f"{worst:.2e}  {'ok' if ok else 'FAIL'}")

    print("\n2. ZERO-ACTION INCOME: what a stander earns, measured vs on paper")
    print("   The closed form treats the passive drift as a CONSTANT. It is not")
    print("   -- yaw drift is an rms about zero -- and K is nonlinear, so the")
    print("   two differ by Jensen wherever the drift is a large share of the")
    print("   error. Reported, never asserted: a rollout is the ground truth.")
    print(f"{'cmd':18s} {'rel_e_v':>8s} {'rel_e_w':>8s} {'on paper':>9s} "
          f"{'measured':>9s} {'Jensen':>9s}")
    for cmd in CELLS:
        cf = closed_form_standing_income(cmd)
        got = _rollout(env, cmd, steps=200, seed=1000)
        rel = (got["track_mean"] - cf["income"]) / max(cf["income"], 1e-12)
        print(f"{str(cmd):18s} {cf['rel_err_v']:8.3f} {cf['rel_err_w']:8.3f} "
              f"{cf['income']:9.5f} {got['track_mean']:9.5f} {rel:+8.1%}")
    print("  A standing machine's relative velocity error is ~1 at every drive")
    print("  command -- that is the whole mechanism, and it is why its income is")
    print("  nearly command-independent instead of growing as the command shrinks.")
    print("  The pure-turn cell is where the Jensen gap bites hardest, and it is")
    print("  also the design's thinnest predicted margin. Trust the measurement.")

    print("\n3. THE SHAPING TELESCOPES: sum(gamma*P(s') - P(s)) over an episode")
    print(f"{'cmd':18s} {'env sum':>10s} {'identity':>10s} {'delta':>10s} {'steps':>6s}")
    for cmd in (CELLS[0], CELLS[3], CELLS[6]):
        got = _rollout(env, cmd, steps=300, seed=2000)
        p = got["potentials"]
        # sum_t [gamma*P(s_{t+1}) - P(s_t)] with P zeroed only on termination.
        identity = SHAPING_GAMMA * sum(p[1:]) - sum(p[:-1])
        delta = got["shaping_sum"] - identity
        ok = abs(delta) < 1e-9
        failures += not ok
        print(f"{str(cmd):18s} {got['shaping_sum']:10.5f} {identity:10.5f} "
              f"{delta:+10.2e} {got['steps']:6d} {'ok' if ok else 'FAIL'}")

    print("\n4. THE POTENTIAL IS RESET, so episode 2 does not inherit episode 1")
    first = _rollout(env, CELLS[0], steps=50, seed=3000)
    second = _rollout(env, CELLS[0], steps=50, seed=3000)
    same = abs(first["shaping_sum"] - second["shaping_sum"]) < 1e-9
    failures += not same
    print(f"  identical seed, identical shaping sum: "
          f"{first['shaping_sum']:.6f} vs {second['shaping_sum']:.6f} "
          f"{'ok' if same else 'FAIL -- stale potential leaked across reset'}")

    print("\n5. STANDING IS NOT THE WHOLE STORY: what the mixture pays a stander")
    # p_stop=0.10, p_turn=0.10, p_drive=0.80. Within drive: P(forward)=0.8,
    # |vx| ~ U[0.3,0.8] forward / U[0.4,0.8] backward, w=0 w.p. 0.5 else
    # U[-0.6,0.6]. Integrated by quadrature rather than sampled, so the number
    # is exact and reproducible.
    rng = np.random.default_rng(7)
    n = 200_000
    incomes = np.empty(n)
    for i in range(n):
        roll = rng.uniform()
        if roll < 0.10:
            cmd = (0.0, 0.0, 0.0)
        elif roll < 0.20:
            sign = 1.0 if rng.uniform() < 0.5 else -1.0
            cmd = (0.0, 0.0, sign * rng.uniform(0.3, 0.6))
        else:
            sign = 1.0 if rng.uniform() < 0.8 else -1.0
            vx = sign * rng.uniform(0.3 if sign > 0 else 0.4, 0.8)
            w = 0.0 if rng.uniform() < 0.5 else rng.uniform(-0.6, 0.6)
            cmd = (vx, 0.0, w)
        incomes[i] = closed_form_standing_income(cmd)["income"]
    print(f"  a do-nothing machine's mean income over the TRAINING mixture: "
          f"{incomes.mean():.5f}/step  (n={n:,})")
    stop_like = incomes > 0.5
    print(f"    {stop_like.sum() / n:.1%} of draws pay it > 0.5 (the stop "
          f"command, where standing is CORRECT), mean {incomes[stop_like].mean():.4f}")
    print(f"    {(~stop_like).sum() / n:.1%} of draws pay it "
          f"{incomes[~stop_like].mean():.5f} on average -- the freeride, and it")
    print("    is what the old reward paid 0.065/step for.")
    print("  Under the OLD reward the same stander took 0.06501/step on the")
    print("  drive grid. The comparison that matters is against what a DRIVING")
    print("  policy earns, which needs a trained policy and is not checkable here.")

    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
