"""Falsifier 4 of `learnings/011`: is the control-cost measurement an artifact
of `deterministic=True`?

    venv/bin/python research/scripts/deterministic_vs_stochastic.py
    venv/bin/python research/scripts/deterministic_vs_stochastic.py --episodes 5

Every number in `learnings/011` -- the four-term decomposition, the per-cell
grid, the 12.6x -- comes from rollouts taken with `deterministic=True`, i.e. the
MEAN action of SAC's squashed-Gaussian actor. SAC did not optimise that policy.
It optimised the STOCHASTIC one, and the control cost is quadratic:

    E[sum_i a_i^2] = sum_i (E[a_i])^2 + sum_i Var(a_i)

so a sampled action pays strictly more than the mean action it is centred on,
by exactly the summed action variance. The deterministic rollout therefore
UNDER-reports what training was paying. This script measures by how much.

Three parts, in the order they should be read:

1. THE ACTION DISTRIBUTION. Over states the policy actually visits, the actor's
   pre-squash mean and log-std, the deterministic action tanh(mu), and the
   Monte-Carlo moments of the sampled action. Because the squash is nonlinear,
   the deterministic action is the MEDIAN of the action distribution, not its
   mean, so the excess splits into two named pieces rather than one:

       E[sum a^2] - sum a_det^2  =  (sum E[a]^2 - sum a_det^2)  +  sum Var(a)
                                     `-- mean shift --'          `- variance -'

2. BOTH ARMS OF THE ACTUAL COMPARISON. `record.track_eval` is the instrument and
   this script calls it rather than reimplementing it: the same 6-cell drive
   grid + stop cell, the same episodes/cell and the same seed block as
   `research/measurements/hound_track_desert_s0_final_sac.json`, under
   `deterministic=True` and `deterministic=False`. The stochastic arm is run
   TWICE on the same action seeds; a stochastic measurement nobody can reproduce
   is not a measurement.

3. THE HEADLINE RATIO, RECOMPUTED. `learnings/011` divides cost by gain:
   0.06889 / 0.00545 = 12.6. Reported here under both policies and under three
   length conventions, because `anomalies.jsonl` row 20 records that per-step
   means over unequal episodes flatter a crashing policy and the stochastic arm
   may well crash more.

Inference only, CPU only: it loads the checkpoint with `device="cpu"` and never
touches the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

import gymnasium as gym
import numpy as np
import torch

import bestiary.envs  # noqa: F401 -- registers the env ids
from bestiary import paths
from bestiary.record.track_eval import (
    EVAL_GRID,
    STOP_CELL,
    TRACK_ENV,
    _arm,
    discover_terms,
)

RUN = "hound_track_desert_s0"
CHECKPOINT = "ant_sac.zip"          # the FINAL, UNSELECTED checkpoint
REFERENCE = "hound_track_desert_s0_final_sac.json"
OUT = "hound_track_desert_s0_deterministic_vs_stochastic.json"

# learnings/011's published per-step figures, quoted here so the recomputation
# is checked against the claim it is testing rather than against memory.
L011 = {"track_rate": 0.00545, "ctrl_rate": 0.06889, "ratio": 12.6}


def ent_coef_trajectory(run: str) -> dict | None:
    """ent_coef against total_timesteps, read out of the run's own train.log.

    anomalies.jsonl row 16 is about this number collapsing, and a collapse is a
    claim about a trajectory, not about one sample of it. The minimum and the
    final value are different facts and are reported as such.
    """
    log = paths.RUNS / run / "train.log"
    if not log.exists():
        return None
    text = log.read_text()
    steps, ents, cur = [], [], None
    for line in text.splitlines():
        m = re.search(r"\|\s*total_timesteps\s*\|\s*([0-9]+)\s*\|", line)
        if m:
            cur = int(m.group(1))
        m = re.search(r"\|\s*ent_coef\s*\|\s*([0-9.eE+-]+)\s*\|", line)
        if m and cur is not None:
            steps.append(cur)
            ents.append(float(m.group(1)))
    if not ents:
        return None
    e = np.array(ents)
    i = int(e.argmin())
    return {
        "points": len(e),
        "first": float(e[0]), "first_step": steps[0],
        "min": float(e[i]), "min_step": steps[i],
        "final": float(e[-1]), "final_step": steps[-1],
        "recovery_from_min": float(e[-1] / e[i]),
    }


# --------------------------------------------------------------------------
# part 1 -- the action distribution
# --------------------------------------------------------------------------
def collect_states(env, policy, episodes: int, seed0: int, stride: int,
                   deterministic: bool, action_seed0: int | None) -> np.ndarray:
    """Observations the policy visits, subsampled, via the real eval protocol."""
    obs_seen: list[np.ndarray] = []

    def on_step(i, obs_before, action, info):
        if i % stride == 0:
            obs_seen.append(np.asarray(obs_before, dtype=np.float64))

    _arm(env, policy, episodes, seed0, deterministic=deterministic,
         action_seed0=action_seed0, on_step=on_step)
    return np.stack(obs_seen)


def action_moments(policy, obs: np.ndarray, n_samples: int, seed: int) -> dict:
    """Actor mean/log-std at each state, and Monte-Carlo moments of tanh(N)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mu_pre, log_std, _ = policy.actor.get_action_dist_params(obs_t)
        sigma_pre = log_std.exp()
        a_det = torch.tanh(mu_pre)                                  # (S, A)

        # (S, N, A) samples of the SAME distribution predict(deterministic=
        # False) draws from: tanh of a diagonal Gaussian.
        eps = torch.randn((obs.shape[0], n_samples, mu_pre.shape[1]),
                          generator=gen)
        a = torch.tanh(mu_pre[:, None, :] + sigma_pre[:, None, :] * eps)

        e_a = a.mean(dim=1)                                         # (S, A)
        # ddof=0 deliberately: with the population form the Monte-Carlo
        # identity E[sum a^2] = sum(E a)^2 + sum Var holds to floating point,
        # which is what the printed identity check is for. The 1/N bias is
        # 0.2% at N=512 and is not the quantity under test.
        var_a = a.var(dim=1, unbiased=False)                        # (S, A)
        e_sum_a2 = (a ** 2).sum(dim=2).mean(dim=1)                  # (S,)
        sum_det2 = (a_det ** 2).sum(dim=1)                          # (S,)

        # Entropy of the same squashed Gaussian, in nats, by the same samples.
        # SAC maximises r + alpha*H, so this is what the objective PAID for the
        # action noise that the control cost above charges for. The tanh
        # correction is SB3's own (sac/policies via distributions.py).
        g = mu_pre[:, None, :] + sigma_pre[:, None, :] * eps
        log_prob = (
            -0.5 * ((g - mu_pre[:, None, :]) / sigma_pre[:, None, :]) ** 2
            - log_std[:, None, :] - 0.5 * float(np.log(2 * np.pi))
        ).sum(dim=2) - torch.log(1 - a ** 2 + 1e-6).sum(dim=2)
        entropy = (-log_prob).mean(dim=1)                            # (S,)

    return {
        "entropy": entropy.numpy(),
        "n_states": int(obs.shape[0]),
        "n_samples": n_samples,
        "mean_action": e_a.numpy(),
        "sum_det_sq": sum_det2.numpy(),
        "e_sum_sq": e_sum_a2.numpy(),
        "sum_var": var_a.sum(dim=1).numpy(),
        "sum_mean_sq": (e_a ** 2).sum(dim=1).numpy(),
        "sigma_pre": sigma_pre.numpy(),
        "sd_action": var_a.sqrt().numpy(),
        "abs_det": a_det.abs().numpy(),
    }


def validate_sampler(policy, obs: np.ndarray, n_states: int, n_draws: int,
                     seed: int) -> dict:
    """Check the parameterisation above against the real `predict` path.

    If `tanh(mu + sigma*eps)` is not what `predict(deterministic=False)` draws,
    every number in part 1 is about a distribution the rollouts never sampled.
    So compare empirical moments from `predict` itself against the MC moments.
    """
    torch.manual_seed(seed)
    idx = np.linspace(0, obs.shape[0] - 1, n_states).astype(int)
    worst_mean = worst_sd = worst_sumsq = 0.0
    noise_mean = noise_sumsq = 0.0
    for j in idx:
        draws = np.stack([policy.predict(obs[j], deterministic=False)[0]
                          for _ in range(n_draws)])
        # 40k model samples so the model side contributes little of the
        # difference; what is left is the sampling error of `draws`.
        m = action_moments(policy, obs[j][None, :], 40_000, seed + int(j))
        worst_mean = max(worst_mean, float(
            np.abs(draws.mean(axis=0) - m["mean_action"][0]).max()))
        worst_sd = max(worst_sd, float(
            np.abs(draws.std(axis=0, ddof=1) - m["sd_action"][0]).max()))
        worst_sumsq = max(worst_sumsq, abs(
            float((draws ** 2).sum(axis=1).mean()) - float(m["e_sum_sq"][0])))
        # 2-sigma Monte-Carlo band the differences above must be judged against
        noise_mean = max(noise_mean, float(
            2 * draws.std(axis=0, ddof=1).max() / np.sqrt(n_draws)))
        noise_sumsq = max(noise_sumsq, float(
            2 * (draws ** 2).sum(axis=1).std(ddof=1) / np.sqrt(n_draws)))
    return {
        "states_checked": len(idx),
        "draws_per_state": n_draws,
        "max_abs_diff_mean_per_dim": worst_mean,
        "max_abs_diff_sd_per_dim": worst_sd,
        "max_abs_diff_E_sum_a2": worst_sumsq,
        "mc_2sigma_mean_per_dim": noise_mean,
        "mc_2sigma_E_sum_a2": noise_sumsq,
    }


# --------------------------------------------------------------------------
# part 2 -- the two arms
# --------------------------------------------------------------------------
def run_arm(env, policy, episodes: int, seed0: int, deterministic: bool,
            action_seed0: int | None) -> tuple[dict, list[np.ndarray]]:
    """One full-grid arm, keeping the per-step reward terms of every episode.

    `on_step` fires inside the very rollouts `_arm` aggregates, so the traces
    cannot drift from the table. Episode boundaries are step_index == 0.
    """
    traces: list[list[list[float]]] = []
    # The trace's COLUMN ORDER is the env's own term order, discovered on the
    # first step rather than taken from a constant (anomalies.jsonl row 39 --
    # the constant was a 4-tuple and this env family now has envs paying 5).
    # It is captured once and asserted stable, because a column order that
    # changed mid-arm would silently reindex every rate computed below.
    seen: list[tuple[str, ...]] = []

    def on_step(i, obs_before, action, info):
        if i == 0:
            traces.append([])
        terms = discover_terms(info)
        if not seen:
            seen.append(terms)
        elif terms != seen[0]:
            raise SystemExit(f"reward terms changed mid-arm: {seen[0]} -> {terms}")
        traces[-1].append([float(info[t]) for t in terms])

    arm = _arm(env, policy, episodes, seed0, deterministic=deterministic,
               action_seed0=action_seed0, on_step=on_step)
    arrays = [np.asarray(t, dtype=np.float64) for t in traces]
    term_order = seen[0]

    # The traces must be the same episodes the table came from, in order.
    expected = episodes * len(EVAL_GRID)
    if len(arrays) != expected:
        raise SystemExit(f"trace/episode mismatch: {len(arrays)} != {expected}")
    for ci, cell in enumerate(EVAL_GRID):
        got = np.mean([len(a) for a in arrays[ci * episodes:(ci + 1) * episodes]])
        want = arm["cells"][str(cell)]["mean_steps"]
        if abs(got - want) > 1e-9:
            raise SystemExit(f"trace misaligned on {cell}: {got} != {want}")
    # The trace columns and the table's own decomposition must be the same
    # list in the same order -- `per_step_rates` indexes the traces by looking
    # names up in `arm["terms"]`, so this is what makes that lookup sound.
    if tuple(arm["terms"]) != term_order:
        raise SystemExit(
            f"trace columns {term_order} do not match the table's terms "
            f"{tuple(arm['terms'])}"
        )
    return arm, arrays


def drive_traces(arrays: list[np.ndarray], episodes: int) -> list[np.ndarray]:
    """The drive-grid episodes only, in EVAL_GRID order (stop cell dropped)."""
    out = []
    for ci, cell in enumerate(EVAL_GRID):
        if tuple(cell) == STOP_CELL:
            continue
        out.extend(arrays[ci * episodes:(ci + 1) * episodes])
    return out


def per_step_rates(arm: dict, traces: list[np.ndarray], prefix: int) -> dict:
    """Per-step track and ctrl rates under three length conventions.

    (a) `learnings/011`'s: mean(episode total) / mean(episode steps). Identical
        to sum(totals)/sum(steps) here because every cell has the same episode
        count, i.e. it is the step-weighted rate.
    (b) episode-weighted:  mean(total/steps) -- a 300-step episode counts as
        much as a 1000-step one. This is the convention `track_eval`'s
        `mean_track` uses, and anomalies.jsonl row 20 is about the difference.
    (c) common prefix:     the first `prefix` steps of EVERY episode, so a
        crashing arm and a surviving one are compared over the same window.
        The only one of the three with no length bias at all.
    """
    terms = list(arm["terms"])
    i_track, i_ctrl = terms.index("reward_track"), terms.index("reward_ctrl")
    steps = np.array([len(t) for t in traces], dtype=float)
    tot_track = np.array([t[:, i_track].sum() for t in traces])
    tot_ctrl = np.array([-t[:, i_ctrl].sum() for t in traces])   # info is signed
    pre_track = np.array([t[:prefix, i_track].sum() for t in traces])
    pre_ctrl = np.array([-t[:prefix, i_ctrl].sum() for t in traces])
    return {
        "mean_steps": float(steps.mean()),
        "min_steps": float(steps.min()),
        "median_steps": float(np.median(steps)),
        "a_track": float(tot_track.mean() / steps.mean()),
        "a_ctrl": float(tot_ctrl.mean() / steps.mean()),
        "b_track": float((tot_track / steps).mean()),
        "b_ctrl": float((tot_ctrl / steps).mean()),
        "c_track": float(pre_track.mean() / prefix),
        "c_ctrl": float(pre_ctrl.mean() / prefix),
        # spread of the per-episode rate, which is what says whether the
        # stochastic arm's extra noise swamps the effect
        "sd_ctrl_per_ep": float((tot_ctrl / steps).std(ddof=1)),
        "sd_track_per_ep": float((tot_track / steps).std(ddof=1)),
    }


def arm_diff(a: dict, b: dict) -> float:
    """Largest absolute difference between two arms' reported numbers."""
    worst = 0.0
    for cell in a["cells"]:
        for k, v in a["cells"][cell].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                worst = max(worst, abs(float(v) - float(b["cells"][cell][k])))
    for k, v in a.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            worst = max(worst, abs(float(v) - float(b[k])))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20,
                    help="episodes per grid cell; 20 is what the record used")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="episode seed block; 1000 is what the record used")
    ap.add_argument("--action-seed", type=int, default=7000)
    ap.add_argument("--probe-episodes", type=int, default=3,
                    help="episodes per cell for the action-distribution probe")
    ap.add_argument("--probe-stride", type=int, default=20)
    ap.add_argument("--probe-samples", type=int, default=512)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="part 1 only (the action distribution); implies "
                         "--no-write, since the JSON is the two-arm result")
    args = ap.parse_args()
    if args.probe_only:
        args.no_write = True

    from stable_baselines3 import SAC

    ckpt = paths.RUNS / RUN / CHECKPOINT
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]

    env = gym.make(TRACK_ENV)
    w_ctrl = float(env.unwrapped._ctrl_cost_weight)
    policy = SAC.load(ckpt, device="cpu")

    # anomalies.jsonl row 16 reports this run's ent_coef collapsing to a floor
    # ~37-100x below the old-reward runs', and reads it as "so the policy is
    # nearly deterministic". ent_coef prices entropy in the OBJECTIVE; the
    # actor's own log-std head is what sets the action noise. Print both.
    ent_coef = None
    if getattr(policy, "log_ent_coef", None) is not None:
        ent_coef = float(policy.log_ent_coef.detach().exp().item())
    elif getattr(policy, "ent_coef_tensor", None) is not None:
        ent_coef = float(policy.ent_coef_tensor.item())

    print(f"{RUN} [{CHECKPOINT}]  sha256:{sha}  "
          f"num_timesteps={policy.num_timesteps}")
    print(f"  ent_coef at save {ent_coef:.3e}   "
          f"target_entropy {float(policy.target_entropy):.1f}")
    traj = ent_coef_trajectory(RUN)
    if traj is not None:
        print(f"  ent_coef in train.log ({traj['points']} points): "
              f"first {traj['first']:.3e} @{traj['first_step']}, "
              f"MIN {traj['min']:.3e} @{traj['min_step']}, "
              f"final {traj['final']:.3e} @{traj['final_step']} "
              f"({traj['recovery_from_min']:.1f}x above its minimum)")
    print(f"{TRACK_ENV}   w_ctrl={w_ctrl}   {len(EVAL_GRID) - 1} drive cells "
          f"+ stop cell,  {args.episodes} episodes/cell, seeds "
          f"{args.seed0}-{args.seed0 + args.episodes - 1}\n")

    # ---------------- part 1 ------------------------------------------------
    print("=" * 78)
    print("1. THE ACTION DISTRIBUTION  (ctrl cost is w_ctrl * sum_i a_i^2)")
    print("=" * 78)
    probes = {}
    for label, det, aseed in (("deterministic-arm states", True, None),
                              ("stochastic-arm states", False, args.action_seed)):
        obs = collect_states(env, policy, args.probe_episodes, args.seed0,
                             args.probe_stride, det, aseed)
        m = action_moments(policy, obs, args.probe_samples, seed=4242)
        probes[label] = m
        det_c = w_ctrl * m["sum_det_sq"].mean()
        sto_c = w_ctrl * m["e_sum_sq"].mean()
        print(f"\n  states from the {label}:  "
              f"{m['n_states']} states x {m['n_samples']} samples")
        print(f"    sum_i a_det,i^2      (deterministic action) "
              f"{m['sum_det_sq'].mean():9.4f}")
        print(f"    sum_i (E a_i)^2      (mean of the samples)  "
              f"{m['sum_mean_sq'].mean():9.4f}")
        print(f"    sum_i Var(a_i)       THE UNDER-REPORT       "
              f"{m['sum_var'].mean():9.4f}")
        print(f"    E[sum_i a_i^2]       (what SAC pays for)    "
              f"{m['e_sum_sq'].mean():9.4f}")
        print(f"      identity check  E[sum a^2] - sum(E a)^2 - sum Var = "
              f"{float(np.abs(m['e_sum_sq'] - m['sum_mean_sq'] - m['sum_var']).max()):.2e}")
        print(f"      mean-shift piece  sum(E a)^2 - sum a_det^2 = "
              f"{m['sum_mean_sq'].mean() - m['sum_det_sq'].mean():+9.4f}")
        print(f"    per-dim pre-squash sigma   mean {m['sigma_pre'].mean():.4f}"
              f"   min {m['sigma_pre'].min():.4f}   max {m['sigma_pre'].max():.4f}")
        print(f"    per-dim sd of the ACTION   mean {m['sd_action'].mean():.4f}"
              f"   min {m['sd_action'].min():.4f}   max {m['sd_action'].max():.4f}")
        print(f"    mean |a_det| per dim       {m['abs_det'].mean():.4f}"
              f"   (1.0 = saturated)")
        print(f"    => ctrl cost / step   deterministic {det_c:.5f}"
              f"   stochastic {sto_c:.5f}   ratio {sto_c / det_c:.3f}x")
        if ent_coef is not None:
            h = float(m["entropy"].mean())
            tgt = float(policy.target_entropy)
            # Differential entropy on a bounded support is negative and that is
            # not a sign of anything: uniform on (-1,1)^16 is 16*ln2 = +11.09
            # nats, so H = -10.9 is a concentrated distribution, not a
            # degenerate one. What SAC's temperature responds to is H vs
            # target_entropy, and the sign of that difference is the whole
            # story of the ent_coef collapse in anomalies row 16.
            print(f"    entropy H of that distribution {h:8.3f} nats/step"
                  f"   (target_entropy {tgt:.1f}, uniform on the action box "
                  f"{16 * float(np.log(2)):.2f})")
            print(f"      alpha*H = {ent_coef * h:+.5f}/step; H - target = "
                  f"{h - tgt:+.3f}, so the policy is "
                  f"{'MORE' if h > tgt else 'LESS'} stochastic than the target"
                  f" -> alpha is driven {'DOWN' if h > tgt else 'UP'}")

    val_obs = collect_states(env, policy, 1, args.seed0,
                             args.probe_stride * 5, True, None)
    val = validate_sampler(policy, val_obs, n_states=6, n_draws=4000, seed=99)
    print(f"\n  sampler validation against policy.predict(deterministic=False): "
          f"{val['states_checked']} states x {val['draws_per_state']} draws")
    print(f"    max |mean_empirical - mean_model| per dim "
          f"{val['max_abs_diff_mean_per_dim']:.5f}"
          f"   (MC 2-sigma {val['mc_2sigma_mean_per_dim']:.5f})")
    print(f"    max |sd_empirical - sd_model| per dim   "
          f"{val['max_abs_diff_sd_per_dim']:.5f}")
    print(f"    max |E[sum a^2]_emp - E[sum a^2]_model| "
          f"{val['max_abs_diff_E_sum_a2']:.5f}"
          f"   (MC 2-sigma {val['mc_2sigma_E_sum_a2']:.5f})")

    if args.probe_only:
        env.close()
        return

    # ---------------- part 2 ------------------------------------------------
    print("\n" + "=" * 78)
    print("2. BOTH ARMS, SAME GRID, SAME SEEDS")
    print("=" * 78)
    zero, zero_tr = run_arm(env, None, args.episodes, args.seed0, True, None)
    det, det_tr = run_arm(env, policy, args.episodes, args.seed0, True, None)
    sto, sto_tr = run_arm(env, policy, args.episodes, args.seed0, False,
                          args.action_seed)
    sto2, sto2_tr = run_arm(env, policy, args.episodes, args.seed0, False,
                            args.action_seed)

    repro = arm_diff(sto, sto2)
    print(f"\n  REPRODUCIBILITY: stochastic arm run twice on action seed "
          f"{args.action_seed}")
    print(f"    max |run1 - run2| over every reported number: {repro:.3e}"
          f"   ({'REPRODUCIBLE' if repro == 0.0 else 'NOT reproducible'})")
    print(f"    run1 drive-grid mean {sto['drive_grid_mean']:9.3f}"
          f"      run2 {sto2['drive_grid_mean']:9.3f}")

    # Regression check on the instrument: this script added parameters to
    # track_eval, so the deterministic arm must still reproduce the committed
    # measurement exactly.
    ref_path = paths.RESEARCH / "measurements" / REFERENCE
    ref = json.loads(ref_path.read_text())
    if ref["episodes_per_cell"] == args.episodes and ref["seed0"] == args.seed0:
        d_zero = max(abs(zero["cells"][k]["mean"] - ref["zero_action"]["cells"][k]["mean"])
                     for k in zero["cells"])
        d_det = max(abs(det["cells"][k]["mean"] - ref["trained"]["cells"][k]["mean"])
                    for k in det["cells"])
        print(f"\n  INSTRUMENT REGRESSION vs {REFERENCE}")
        print(f"    max |zero arm now - committed| {d_zero:.3e}")
        print(f"    max |det  arm now - committed| {d_det:.3e}"
              f"   ({'unchanged' if max(d_zero, d_det) < 1e-9 else 'CHANGED'})")

    hdr = (f"\n  {'command':>18} {'arm':>6} {'return':>9} {'sd':>7} "
           f"{'steps':>7} {'track':>7} {'ctrl/stp':>9} {'crash':>5}")
    print(hdr)
    for k in zero["cells"]:
        for name, arm in (("zero", zero), ("det", det), ("stoch", sto)):
            c = arm["cells"][k]
            ctrl_rate = -c["reward_ctrl"] / c["mean_steps"]
            label = k if name == "zero" else ""
            print(f"  {label:>18} {name:>6} {c['mean']:9.2f} {c['sd']:7.2f} "
                  f"{c['mean_steps']:7.1f} {c['mean_track']:7.4f} "
                  f"{ctrl_rate:9.5f} {c['crashes']:5d}"
                  + ("   <- stop cell, EXCLUDED" if name == "zero"
                     and tuple(c["command"]) == STOP_CELL else ""))

    n_drive = args.episodes * (len(EVAL_GRID) - 1)
    print(f"\n  {'':>10} {'drive_mean':>11} {'drive_track':>12} {'steps':>8} "
          f"{'crashes':>9}  (of {n_drive})")
    for name, arm in (("zero", zero), ("det", det), ("stoch", sto)):
        dc = sum(arm["cells"][str(c)]["crashes"] for c in EVAL_GRID
                 if tuple(c) != STOP_CELL)
        print(f"  {name:>10} {arm['drive_grid_mean']:11.3f} "
              f"{arm['drive_grid_track']:12.4f} {arm['drive_grid_steps']:8.1f} "
              f"{dc:9d}")

    # ---------------- part 3 ------------------------------------------------
    print("\n" + "=" * 78)
    print("3. THE HEADLINE RATIO, RECOMPUTED")
    print("=" * 78)
    d_zero_tr = drive_traces(zero_tr, args.episodes)
    d_det_tr = drive_traces(det_tr, args.episodes)
    d_sto_tr = drive_traces(sto_tr, args.episodes)
    prefix = int(min(min(len(t) for t in d_det_tr),
                     min(len(t) for t in d_sto_tr),
                     min(len(t) for t in d_zero_tr)))
    rates = {
        "zero": per_step_rates(zero, d_zero_tr, prefix),
        "det": per_step_rates(det, d_det_tr, prefix),
        "stoch": per_step_rates(sto, d_sto_tr, prefix),
    }
    print(f"\n  drive grid only. common prefix = {prefix} steps "
          f"(shortest drive episode in any arm)")
    print(f"  {'arm':>7} {'steps':>7} {'min':>6} {'med':>6} "
          f"{'track/stp':>10} {'ctrl/stp':>9} {'track_b':>9} {'ctrl_b':>9} "
          f"{'track_c':>9} {'ctrl_c':>9}")
    for name in ("zero", "det", "stoch"):
        r = rates[name]
        print(f"  {name:>7} {r['mean_steps']:7.1f} {r['min_steps']:6.0f} "
              f"{r['median_steps']:6.0f} {r['a_track']:10.5f} {r['a_ctrl']:9.5f} "
              f"{r['b_track']:9.5f} {r['b_ctrl']:9.5f} {r['c_track']:9.5f} "
              f"{r['c_ctrl']:9.5f}")
    print("    a = mean(total)/mean(steps)  [learnings/011's, = step-weighted]")
    print("    b = mean(total/steps)        [episode-weighted, = mean_track's]")
    print(f"    c = first {prefix} steps of every episode "
          f"[length-matched, anomalies row 20]")

    print(f"\n  per-episode spread of the per-step rates (sd across "
          f"{len(d_det_tr)} drive episodes)")
    for name in ("zero", "det", "stoch"):
        r = rates[name]
        print(f"    {name:>7}  track {r['sd_track_per_ep']:.5f}"
              f"   ctrl {r['sd_ctrl_per_ep']:.5f}")

    print("\n  cost of driving vs gain from driving, against zero action:")
    print(f"  {'arm':>7} {'conv':>5} {'gain/step':>11} {'cost/step':>11} "
          f"{'RATIO':>9}")
    ratios = {}
    for name in ("det", "stoch"):
        for conv in ("a", "b", "c"):
            gain = rates[name][f"{conv}_track"] - rates["zero"][f"{conv}_track"]
            cost = rates[name][f"{conv}_ctrl"] - rates["zero"][f"{conv}_ctrl"]
            ratio = cost / gain if gain != 0 else float("nan")
            ratios[f"{name}_{conv}"] = {"gain": gain, "cost": cost,
                                        "ratio": ratio}
            flag = "   (gain <= 0: driving buys NOTHING)" if gain <= 0 else ""
            print(f"  {name:>7} {conv:>5} {gain:11.5f} {cost:11.5f} "
                  f"{ratio:9.2f}{flag}")

    a_det, a_sto = ratios["det_a"], ratios["stoch_a"]
    print(f"\n  learnings/011 published, convention a: gain "
          f"{L011['track_rate']:.5f}, cost {L011['ctrl_rate']:.5f}, ratio "
          f"{L011['ratio']}")
    print(f"  this script,     convention a: gain {a_det['gain']:.5f}, cost "
          f"{a_det['cost']:.5f}, ratio {a_det['ratio']:.2f}   [deterministic]")
    print(f"  this script,     convention a: gain {a_sto['gain']:.5f}, cost "
          f"{a_sto['cost']:.5f}, ratio {a_sto['ratio']:.2f}   [stochastic]")
    d = probes["deterministic-arm states"]
    excess = w_ctrl * float(d["e_sum_sq"].mean() - d["sum_det_sq"].mean())
    print(f"\n  predicted extra ctrl cost/step from the action distribution "
          f"alone: {excess:+.5f}")
    print(f"  observed  extra ctrl cost/step, stochastic - deterministic:     "
          f"{rates['stoch']['a_ctrl'] - rates['det']['a_ctrl']:+.5f}")
    print("  (the gap between those two is the STATE distribution moving, "
          "not the action noise)")

    if not args.no_write:
        out = paths.RESEARCH / "measurements" / OUT
        payload = {
            "run": RUN, "checkpoint": CHECKPOINT, "checkpoint_sha256_16": sha,
            "env": TRACK_ENV, "w_ctrl": w_ctrl,
            "episodes_per_cell": args.episodes, "seed0": args.seed0,
            "action_seed": args.action_seed,
            "num_timesteps": int(policy.num_timesteps),
            "ent_coef_at_save": ent_coef,
            "target_entropy": float(policy.target_entropy),
            "ent_coef_trajectory": traj,
            "action_distribution": {
                label: {
                    "n_states": m["n_states"], "n_samples": m["n_samples"],
                    "mean_sum_det_sq": float(m["sum_det_sq"].mean()),
                    "mean_sum_mean_sq": float(m["sum_mean_sq"].mean()),
                    "mean_sum_var": float(m["sum_var"].mean()),
                    "mean_E_sum_sq": float(m["e_sum_sq"].mean()),
                    "ctrl_per_step_deterministic":
                        w_ctrl * float(m["sum_det_sq"].mean()),
                    "ctrl_per_step_stochastic":
                        w_ctrl * float(m["e_sum_sq"].mean()),
                    "sigma_pre_mean": float(m["sigma_pre"].mean()),
                    "sigma_pre_min": float(m["sigma_pre"].min()),
                    "sigma_pre_max": float(m["sigma_pre"].max()),
                    "sd_action_mean": float(m["sd_action"].mean()),
                    "mean_abs_det_action": float(m["abs_det"].mean()),
                    "entropy_nats": float(m["entropy"].mean()),
                    "alpha_times_entropy_per_step":
                        None if ent_coef is None
                        else ent_coef * float(m["entropy"].mean()),
                } for label, m in probes.items()
            },
            "sampler_validation": val,
            "reproducibility_max_abs_diff": repro,
            "arms": {"zero_action": zero, "deterministic": det,
                     "stochastic": sto, "stochastic_repeat": sto2},
            "common_prefix_steps": prefix,
            "per_step_rates": rates,
            "cost_gain_ratios": ratios,
            "learnings_011_published": L011,
        }
        out.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\n  wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
