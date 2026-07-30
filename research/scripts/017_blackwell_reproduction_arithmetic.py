"""The arithmetic behind `research/learnings/017`.

A second machine with a Blackwell-class GPU ran the Isaac Hound stack. This
script does three things with the numbers it reported, and the third is the one
that matters:

1. **Checks them against each other.** A throughput figure and an iteration time
   measured in the same run are not independent: `steps/s` must equal
   `num_envs x num_steps_per_env / iteration_time`, and the collection and
   learning splits must add up to the iteration time. If they do not, one of the
   two was mis-transcribed, and this is the cheapest place to find that out.

2. **Converts iterations to wall clock**, which is what a run budget is actually
   denominated in, and re-prices the two published sample budgets from decision
   0004 at a rate that was for once measured on the RIGHT ROBOT.

3. **Refuses to publish a speedup.** The obvious ratio -- remote steps/s over
   the local figure in the record -- is computed here and printed with its
   confounds spelled out, precisely so nobody has to recompute it later and
   guess whether it was controlled. It was not: different robot, different body
   count, different environment count, different GPU. Two uncontrolled ratios
   printed with their confounds are more honest than one printed without.

    venv/bin/python research/scripts/017_blackwell_reproduction_arithmetic.py

Pure arithmetic. No GPU, no Isaac, no imports beyond the standard library.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Inputs. Every one of these is a REPORTED measurement, not a local one, and
# each carries where it came from. Nothing below is derived in this block.
# ---------------------------------------------------------------------------

# --- The second machine, 2026-07-29, Bestiary Hound on Isaac Lab 3.0.0-beta2
#     (commit af1bab4dc173ba69b08fab779c14ead61d13fd33), one seed, 30 iterations.
REMOTE = {
    "num_envs": 2048,
    "num_steps_per_env": 24,          # AnymalCRoughPPORunnerCfg.num_steps_per_env
    "iter_s": 2.21,                   # reported mean iteration time
    "collection_s": 2.07,
    "learning_s": 0.146,
    "steps_per_s": 22_241,            # reported
    "vram_peak_mib": 5_718,           # instrument NOT recorded -- anomalies 55, 57
    "vram_total_mib": 32_607,
    "iterations_run": 30,
}

# --- This machine, from research/decisions/0003 (line 31) and 0004 (line 308).
#     ANYmal-C, RTX 2080. A DIFFERENT ROBOT: 12 DoF / 13 bodies / 4 point feet,
#     against Hound's 16 DoF / 17 bodies / 4 rolling cylinders (anomaly 55).
LOCAL = {
    "num_envs": 1024,
    "num_steps_per_env": 24,
    "steps_per_s": 7_630,
    "vram_peak_mib": 4_649,           # instrument NOT recorded -- anomaly 55
    "vram_total_mib": 8_192,
}

#: The project's own VRAM ceiling for THIS machine, from CLAUDE.md hard rule 3.
LOCAL_VRAM_CEILING_MIB = 6_000

#: Iteration count shipped in AnymalCRoughPPORunnerCfg (verified in the
#: installed Isaac Lab tree), and the seed count the repo's seed rule demands.
SHIPPED_MAX_ITERATIONS = 1_500
SEEDS_PER_ARM = 3

#: Sample budgets from research/decisions/0004:317-319, both quoted from
#: published recipes rather than measured here.
BUDGET_ANYMAL_SAMPLES = 1.47e8        # Rudin et al. operating point
BUDGET_WHEELED_SAMPLES = 1.97e9       # the published Go2-W recipe

#: Tolerance on the internal-consistency checks. A transcription slip is a
#: factor, not a percent; anything inside this is rounding in the reported value.
CONSISTENCY_TOL = 0.01


def _check(label: str, derived: float, reported: float, tol: float = CONSISTENCY_TOL) -> None:
    rel = abs(derived - reported) / reported
    verdict = "consistent" if rel <= tol else "MISMATCH"
    print(f"  {label:<46s} derived {derived:12,.1f}  reported {reported:12,.1f}"
          f"  {rel * 100:5.2f}%  {verdict}")
    if rel > tol:
        raise AssertionError(
            f"{label}: derived {derived} vs reported {reported}, {rel * 100:.2f}% apart, "
            f"over the {tol * 100:.0f}% tolerance. One of the two numbers is wrong and "
            "neither should enter the record until it is known which."
        )


def main() -> None:
    print("1. INTERNAL CONSISTENCY of the reported run")
    batch = REMOTE["num_envs"] * REMOTE["num_steps_per_env"]
    _check("steps/s from envs x T / iter_s", batch / REMOTE["iter_s"], REMOTE["steps_per_s"])
    _check("iter_s from collection + learning",
           REMOTE["collection_s"] + REMOTE["learning_s"], REMOTE["iter_s"])
    print(f"  batch per iteration = {REMOTE['num_envs']:,} envs x "
          f"{REMOTE['num_steps_per_env']} steps = {batch:,} samples")
    coll = REMOTE["collection_s"] / REMOTE["iter_s"]
    print(f"  collection is {coll * 100:.1f}% of the iteration, learning "
          f"{REMOTE['learning_s'] / REMOTE['iter_s'] * 100:.1f}%")
    print("  ^ the run is simulation-bound, not optimisation-bound, so a faster")
    print("    GPU buys throughput and a bigger network costs almost nothing.")
    print()

    print("2. VRAM, and the one number that is genuinely decision-relevant")
    r_frac = REMOTE["vram_peak_mib"] / REMOTE["vram_total_mib"]
    print(f"  remote peak  {REMOTE['vram_peak_mib']:,} MiB of {REMOTE['vram_total_mib']:,} "
          f"= {r_frac * 100:.1f}% of the card")
    against_ceiling = REMOTE["vram_peak_mib"] / LOCAL_VRAM_CEILING_MIB
    print(f"  the same footprint against THIS machine's {LOCAL_VRAM_CEILING_MIB:,} MiB "
          f"ceiling = {against_ceiling * 100:.1f}%")
    print("  ^ 2048 envs of Hound would fit under the local ceiling with about")
    print(f"    {LOCAL_VRAM_CEILING_MIB - REMOTE['vram_peak_mib']:,} MiB spare -- IF the two cards")
    print("    allocate alike, which is not established.")
    print()
    r_per_env = REMOTE["vram_peak_mib"] / REMOTE["num_envs"]
    l_per_env = LOCAL["vram_peak_mib"] / LOCAL["num_envs"]
    print(f"  per-env footprint   remote {r_per_env:.3f} MiB/env   "
          f"local {l_per_env:.3f} MiB/env   ratio {r_per_env / l_per_env:.3f}")
    print("  ^ THIS IS A FLAG, NOT A FINDING. The bigger robot (17 bodies vs 13)")
    print(f"    at twice the environment count reports {(1 - r_per_env / l_per_env) * 100:.0f}% LESS memory per")
    print("    environment. Either the two 'peak' figures were sampled by")
    print("    different instruments or one of them is not a peak. Neither")
    print("    instrument is recorded anywhere -- anomaly 55 for the local base,")
    print("    and it now applies to the remote figure too.")
    print()

    print("3. ITERATIONS -> WALL CLOCK")
    per_seed_h = SHIPPED_MAX_ITERATIONS * REMOTE["iter_s"] / 3600
    print(f"  {SHIPPED_MAX_ITERATIONS:,} iterations x {REMOTE['iter_s']} s = "
          f"{SHIPPED_MAX_ITERATIONS * REMOTE['iter_s']:,.0f} s = {per_seed_h:.3f} h per seed")
    print(f"  x {SEEDS_PER_ARM} seeds (the seed rule) = {per_seed_h * SEEDS_PER_ARM:.3f} h per arm")
    print(f"  samples in one such run = {SHIPPED_MAX_ITERATIONS * batch:,}")
    local_batch = LOCAL["num_envs"] * LOCAL["num_steps_per_env"]
    local_same_iters_h = SHIPPED_MAX_ITERATIONS * local_batch / LOCAL["steps_per_s"] / 3600
    print(f"  the same {SHIPPED_MAX_ITERATIONS:,} iterations on this machine (ANYmal, "
          f"{LOCAL['num_envs']:,} envs) = {local_same_iters_h:.3f} h")
    print(f"  ^ so: {SHIPPED_MAX_ITERATIONS:,} iterations, twice the parallelism, in "
          f"{per_seed_h / local_same_iters_h * 100:.0f}% of the wall clock.")
    print("    That comparison holds iteration COUNT fixed and lets sample count")
    print("    double, which is the honest way to state it -- the two runs do not")
    print("    see the same number of samples and are not the same experiment.")
    print()

    print("4. RE-PRICING decision 0004's two published sample budgets")
    for label, samples in (("Rudin operating point", BUDGET_ANYMAL_SAMPLES),
                           ("published Go2-W recipe", BUDGET_WHEELED_SAMPLES)):
        h_local = samples / LOCAL["steps_per_s"] / 3600
        h_remote = samples / REMOTE["steps_per_s"] / 3600
        iters_remote = samples / batch
        print(f"  {label:<24s} {samples:9.2e} samples   "
              f"local {h_local:6.1f} h   remote {h_remote:6.1f} h   "
              f"({iters_remote:,.0f} remote iterations)")
    print("  ^ the wheeled-legged budget is the interesting row: 0004 priced it at")
    print("    72 h using an ANYmal rate, because that was the only rate we had.")
    print("    The remote rate was measured on HOUND, so that row is the first")
    print("    projection of the Go2-W budget at a rate from the right robot.")
    print()

    print("5. THE SPEEDUP WE ARE NOT PUBLISHING")
    raw = REMOTE["steps_per_s"] / LOCAL["steps_per_s"]
    per_env_remote = REMOTE["steps_per_s"] / REMOTE["num_envs"]
    per_env_local = LOCAL["steps_per_s"] / LOCAL["num_envs"]
    print(f"  raw ratio of steps/s                 {raw:.3f}x")
    print(f"  per-environment steps/s              remote {per_env_remote:.2f}  "
          f"local {per_env_local:.2f}  ratio {per_env_remote / per_env_local:.3f}x")
    print("  Confounds between those two arms, all four moving at once:")
    print("    robot        Hound (16 DoF, 17 bodies, 4 rolling cylinders)")
    print("                 vs ANYmal-C (12 DoF, 13 bodies, 4 point feet + LSTM actuator net)")
    print(f"    envs         {REMOTE['num_envs']:,} vs {LOCAL['num_envs']:,}")
    print("    GPU          Blackwell (cc 12.0) vs Turing (RTX 2080)")
    print("    host cores   48 vs 16")
    print("  NEITHER RATIO IS A SPEEDUP. Under the seed rule a comparison needs")
    print("  >=3 seeds per arm and exactly ONE variable changed; this has one seed")
    print("  per arm and four variables. Both numbers are printed so that the next")
    print("  reader inherits the confounds along with the arithmetic.")


if __name__ == "__main__":
    main()
