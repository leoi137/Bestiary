"""Guard: the tracking score must say whether it is length-biased, and be.

Enforces `research/anomalies.jsonl` row 20 (2026-07-27).

`track_eval.py`'s docstring said `mean_track` "survives cost-coefficient
retuning **and episode-length changes**". The second half was false, and false
in the direction that flatters a crashing policy: `mean_track` is a mean over
episodes of each episode's own per-step mean, so an episode that ends at step
300 carries the same weight as one that runs the full 1000.

That matters more than a normal docstring error because `drive_grid_track` is
the metric the record explicitly tells readers to trust when `drive_grid_ratio`
goes unreadable (returns here can be negative, and a ratio of two negatives is
not a performance ordering). STATE.md, the cycle 006 note and the run-episode
handoff all say *read `drive_grid_track` first*. The designated safe number
carried an undocumented bias.

Three aggregates now exist and they answer three different questions:

    mean_track          while it is up, how well does it track?   length-biased
    mean_track_stepw    per step actually taken, how well?        not biased
    track_per_horizon   how much did it BANK per unit horizon?    not biased

`track_per_horizon` is the tracking analogue of the return, because the reward
is a per-step integral over a fixed horizon: a crash forfeits exactly the steps
it forfeits.

WHAT THIS GUARD ASSERTS, AND WHY IT IS SHAPED THIS WAY

It does not assert any measured value — those move with every run. It asserts
the *arithmetic*, on synthetic episodes whose right answers are known by
construction, by calling `track_eval.aggregate_cell` itself. Importing the real
function is the whole point: a guard that reimplements the formula it is
checking bounds nothing, which is the lesson `tracking-frame` records about
constants and which applies just as well to formulas.

Fast tier: no physics is stepped, no env is built, nothing is read from disk
except this repo's own source. Well under a second, so it gates launches.
"""
from __future__ import annotations

import inspect

from bestiary.guards import Finding
from bestiary.record import track_eval

# One synthetic episode is fully determined by a constant per-step tracking
# rate and a length. Everything else is filler that aggregate_cell needs and
# this guard does not read.
HORIZON = 1000


def _ep(rate: float, steps: int, horizon: int = HORIZON) -> dict:
    """An episode that tracks at exactly `rate` for exactly `steps` steps."""
    return {
        "return": 0.0,
        "steps": steps,
        "horizon": horizon,
        "terminated": steps < horizon,
        "mean_phi_v": rate,
        "mean_phi_w": 1.0,
        "mean_track": rate,
        "track_income": rate * steps,
        "achieved_vx": 0.0,
        "commanded_vx": 0.0,
        # A synthetic one-term decomposition. The names no longer come from a
        # module constant -- there isn't one -- and this guard is about the
        # length correction, not about which terms an env pays.
        "terms": ["reward_track"],
        "reward_track": 0.0,
    }


CELL = (0.5, 0.0, 0.0)


def run() -> list[Finding]:
    findings: list[Finding] = []

    # 1. The degenerate case. If every episode runs the full horizon there is no
    #    bias to have, and all three aggregates must agree exactly. A failure
    #    here means the new fields are not measuring tracking at all.
    full = track_eval.aggregate_cell(CELL, [_ep(0.30, HORIZON) for _ in range(5)])
    agree = (abs(full["mean_track"] - 0.30) < 1e-12
             and abs(full["mean_track_stepw"] - 0.30) < 1e-12
             and abs(full["track_per_horizon"] - 0.30) < 1e-12)
    findings.append(Finding(
        "with no early termination all three tracking aggregates agree",
        agree,
        f"5 episodes at rate 0.30 for the full {HORIZON} steps: "
        f"mean_track={full['mean_track']:.6f}, "
        f"mean_track_stepw={full['mean_track_stepw']:.6f}, "
        f"track_per_horizon={full['track_per_horizon']:.6f} — all must be 0.30, "
        f"because a length correction with nothing to correct must be identity",
        n=5,   # 5 constructed episodes, aggregated by the real aggregate_cell
    ))

    # 2. THE assertion. Same tracking RATE, half the survival. mean_track cannot
    #    tell these apart -- that is its definition, not a defect -- and
    #    track_per_horizon must halve, because half the horizon was banked.
    half = track_eval.aggregate_cell(CELL, [_ep(0.30, HORIZON // 2) for _ in range(5)])
    blind = abs(half["mean_track"] - full["mean_track"]) < 1e-12
    halved = abs(half["track_per_horizon"] - 0.15) < 1e-12
    findings.append(Finding(
        "a policy that crashes at half the horizon reads identically on "
        "mean_track and exactly half on track_per_horizon",
        blind and halved,
        f"at rate 0.30 for {HORIZON // 2} of {HORIZON} steps: "
        f"mean_track={half['mean_track']:.6f} (full-length arm reads "
        f"{full['mean_track']:.6f}, identical: {blind}); "
        f"track_per_horizon={half['track_per_horizon']:.6f}, must be 0.150000 "
        f"({halved}). This is anomalies row 20: the two arms are equally "
        f"COMPETENT and one banks half as much, and only one of these numbers "
        f"can see it",
        n=5,   # the 5 half-length episodes, read against the full-length arm
    ))

    # 3. The step-weighted aggregate must actually weight by steps. Mixing a
    #    long bad episode with a short good one is where a mean-of-means and a
    #    step-weighted mean visibly part company; equal-length episodes would
    #    pass both formulas and assert nothing.
    mixed = track_eval.aggregate_cell(CELL, [_ep(0.90, 100), _ep(0.10, 900)])
    # by hand: (0.90*100 + 0.10*900) / 1000 = 180/1000 = 0.18; mean of means = 0.50
    stepw_ok = abs(mixed["mean_track_stepw"] - 0.18) < 1e-12
    naive_differs = abs(mixed["mean_track"] - 0.50) < 1e-12
    findings.append(Finding(
        "mean_track_stepw weights by steps taken, not by episode count",
        stepw_ok and naive_differs,
        f"one episode at rate 0.90 for 100 steps and one at 0.10 for 900: "
        f"mean_track_stepw={mixed['mean_track_stepw']:.6f} (must be 0.180000, "
        f"the true per-step rate) vs mean_track={mixed['mean_track']:.6f} "
        f"(must be 0.500000, the mean of means). A 2.8x gap between two numbers "
        f"that were reported under one name until 2026-07-28",
        n=2,   # the two mixed-length episodes
    ))

    # 4. The horizon is asked of the env, never assumed. A hardcoded 1000 would
    #    pass every assertion above and silently mis-scale the day a TimeLimit
    #    moves -- exactly the class of failure that is invisible until it has
    #    already contaminated a published number.
    short = track_eval.aggregate_cell(
        CELL, [_ep(0.30, 250, horizon=500) for _ in range(3)])
    findings.append(Finding(
        "track_per_horizon divides by the episode's own recorded horizon",
        abs(short["track_per_horizon"] - 0.15) < 1e-12,
        f"3 episodes at rate 0.30 for 250 of a {500}-step horizon read "
        f"track_per_horizon={short['track_per_horizon']:.6f}; must be 0.150000. "
        f"Against a hardcoded {HORIZON} it would read 0.075000 — half, and "
        f"wrong in a way no other assertion here would catch",
        n=3,   # 3 constructed episodes on a 500-step horizon
    ))

    # 5. The false claim must not come back. This is the cheapest half of the
    #    guard and the reason the bias survived from cycle 006 to cycle 010: the
    #    number was fine to compute, the DOCSTRING is what misled three separate
    #    readers into quoting it as length-safe.
    doc = inspect.getdoc(track_eval) or ""
    claims_safe = "survives cost-coefficient retuning and episode-length changes" in doc
    names_bias = "mean_track_stepw" in doc and "track_per_horizon" in doc
    findings.append(Finding(
        "track_eval's docstring no longer claims mean_track is length-safe",
        (not claims_safe) and names_bias,
        f"the false claim is {'STILL PRESENT' if claims_safe else 'gone'}; the "
        f"two length-aware fields are {'documented' if names_bias else 'MISSING'} "
        f"from the module docstring. The record tells readers to trust "
        f"drive_grid_track when the ratio is unreadable, so what this docstring "
        f"says about it is load-bearing",
        # One named property of one docstring. There is no set behind it.
        n=None,
    ))

    # 6. The identity that lets a READER check track_per_horizon without
    #    trusting this code at all. It holds by algebra for any episode set:
    #        mean_track_stepw * mean_steps / track_per_horizon == horizon
    #    Every committed measurement JSON carries all four numbers, so a
    #    reader can recover the divisor and see it is 1000 rather than assume
    #    the right TimeLimit was found.
    recovered = (mixed["mean_track_stepw"] * mixed["mean_steps"]
                 / mixed["track_per_horizon"])
    findings.append(Finding(
        "the horizon is recoverable from the reported fields alone",
        abs(recovered - HORIZON) < 1e-9 and mixed["horizon"] == HORIZON,
        f"mean_track_stepw * mean_steps / track_per_horizon = {recovered:.6f}, "
        f"and the cell reports horizon={mixed['horizon']} — these must agree "
        f"and equal {HORIZON}. This is what makes the divisor auditable from a "
        f"committed JSON instead of taken on trust",
        n=2,   # the identity is recovered from the 2-episode mixed cell
    ))

    # 7. `rollout` must REFUSE an env with no TimeLimit rather than substitute
    #    the episode's own length. That fallback existed until 2026-07-28 and
    #    was the one hole an adversarial review found in this guard: it turns
    #    track_per_horizon silently into mean_track_stepw, leaves every
    #    assertion above green, and records nothing in the emitted JSON about
    #    which of the two a number is. A designed silent-degradation path with
    #    no assertion behind it is exactly what this repo's standard forbids.
    class _Spec:
        id, max_episode_steps = "NoLimit-v0", None

    class _NoLimit:
        spec = _Spec()

    class _Fine:
        spec = type("S", (), {"id": "Fine-v0", "max_episode_steps": 400})()

    def _refuses(env) -> bool:
        try:
            track_eval.resolve_horizon(env)
        except ValueError:
            return True
        return False

    # Both directions: it must refuse the limitless env AND accept a real one,
    # or "it raises" could be satisfied by a function that always raises.
    raised = _refuses(_NoLimit()) and not _refuses(_Fine())
    raised = raised and track_eval.resolve_horizon(_Fine()) == 400
    findings.append(Finding(
        "rollout refuses an env with no episode limit instead of guessing one",
        raised,
        "an env whose spec declares no max_episode_steps must raise ValueError — "
        "'per horizon' is undefined without a horizon, and the earlier fallback "
        "to the episode's own length degraded track_per_horizon into "
        "mean_track_stepw with no trace in the output"
        + ("" if raised else "  <- it did NOT raise, so the silent path is back"),
        n=2,   # both constructed env specs: the limitless one and the real one
    ))

    return findings
