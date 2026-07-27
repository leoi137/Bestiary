"""How much wall-clock is left, and how many steps fit in it.

A cycle decides its own step budget, and it has to decide it *before* launching
because a detached run outlives the session that started it. Getting that wrong
in either direction is expensive: too long and the run is killed mid-way by a
later cycle enforcing the ceiling, too short and the GPU idles until the next
firing.

Two facts make the sizing possible rather than guesswork:

* **The operator arms the loop for a bounded stretch** — "overnight", "7 days".
  That deadline is supplied by the caller through `$ROBOTICS_ARMED_UNTIL` (see
  below), and a run must not be launched that cannot finish before the window
  closes, or the window ends with a half-trained policy and no ledger row.
* **Throughput is measured, not assumed.** Every finished run records its own
  `fps` in the ledger, per env. `HoundDesert-v0` ran at 129 steps/s and
  `HoundPDDesert-v0` at 108, so the same step budget is a 16% longer run on the
  PD env. Using a single remembered number for both is how a ceiling gets blown.

    venv/bin/python -m bestiary.record.budget                       # what is left
    venv/bin/python -m bestiary.record.budget --env HoundDesert-v0  # and what fits

## Where the deadline comes from

`$ROBOTICS_ARMED_UNTIL` is either a Unix epoch or a path to a file containing
one. Unset means unbounded. It is an environment variable rather than a path
this module knows, deliberately: the scheduling machinery lives outside this
repository, and code here must not encode where. Passing it in keeps this
module testable, keeps the boundary intact, and means a different scheduler
needs no change here.

This module only reads the deadline. Arming and disarming belong to the
operator.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from bestiary import paths

# Supplied by the caller: an epoch, or a path to a file holding one. Unset means
# no window. Never a path hardcoded here -- see "Where the deadline comes from".
ARMED_UNTIL_ENV = "ROBOTICS_ARMED_UNTIL"

# Leave room for the cycle that harvests the run: rolling 20 deterministic
# episodes for the ledger row, the refutation pass, and the commits. Measured
# at ~4 minutes of CPU for the eval alone on a 1000-step env; 30 covers the
# rest without being generous enough to matter on a multi-hour run.
HARVEST_RESERVE_S = 30 * 60

# Never launch a run shorter than this. Below it the replay buffer barely fills
# and the result answers nothing, so the GPU time is better left unspent.
MIN_USEFUL_S = 45 * 60

# Fallback throughput when the ledger has no row for an env yet. The slowest
# rate ever recorded here, because guessing low overruns nothing.
FALLBACK_FPS = 108


def window_remaining_s(armed_until: str | None = None) -> float | None:
    """Seconds until the arming window closes, or None if unbounded."""
    raw = armed_until if armed_until is not None else os.environ.get(ARMED_UNTIL_ENV)
    if not raw:
        return None

    # A bare number is the deadline. Anything else is a path to one.
    try:
        return float(raw) - time.time()
    except ValueError:
        pass

    candidate = Path(raw)
    if not candidate.exists():
        # NOT an error: the caller passes this path unconditionally and the file
        # exists only while a window is set, so "absent" is how "no window" is
        # spelled. The absence is reported in the output rather than swallowed,
        # so a mistyped path reads as "unbounded (no file at X)" instead of
        # silently becoming an unbounded run.
        return None

    body = candidate.read_text().strip()
    if not body:
        raise ValueError(
            f"{candidate} exists but is empty -- a window with no deadline is "
            f"neither armed nor bounded. Re-arm rather than guessing."
        )
    try:
        return float(body) - time.time()
    except ValueError:
        raise ValueError(
            f"{candidate} holds {body!r}, which is not a Unix epoch"
        ) from None


def measured_fps(env_id: str) -> tuple[int, str]:
    """Throughput for this env from the ledger, with where it came from."""
    if paths.LEDGER.exists():
        rates = [
            int(row["fps"])
            for line in paths.LEDGER.read_text().splitlines() if line.strip()
            for row in [json.loads(line)]
            if row.get("env_id") == env_id and row.get("fps")
        ]
        if rates:
            # The slowest observed rate: sizing off the fastest is how a run
            # that fitted "on paper" runs past the window.
            return min(rates), f"slowest of {len(rates)} ledger row(s) for {env_id}"
    return FALLBACK_FPS, f"no ledger row for {env_id} — fallback"


def plan(env_id: str | None = None, ceiling_s: float | None = None,
         armed_until: str | None = None) -> dict:
    """What a cycle needs to decide a step budget."""
    remaining = window_remaining_s(armed_until)

    # The usable stretch is the window minus what harvesting will need.
    if remaining is None:
        usable = ceiling_s
        bound = "operator ceiling" if ceiling_s else None
    else:
        usable = remaining - HARVEST_RESERVE_S
        bound = "arming window"
        if ceiling_s is not None and ceiling_s < usable:
            usable, bound = ceiling_s, "operator ceiling"

    source = armed_until if armed_until is not None else os.environ.get(ARMED_UNTIL_ENV)
    result = {
        "window_source": source,
        "window_remaining_s": None if remaining is None else round(remaining),
        "window_expired": remaining is not None and remaining <= 0,
        "harvest_reserve_s": HARVEST_RESERVE_S,
        "usable_s": None if usable is None else round(usable),
        "bound_by": bound,
        "can_launch": True if usable is None else usable >= MIN_USEFUL_S,
    }

    if env_id:
        fps, source = measured_fps(env_id)
        result["env_id"] = env_id
        result["fps"] = fps
        result["fps_source"] = source
        # Only when a run is actually launchable. A negative or trivially small
        # step budget is not a smaller plan, it is the absence of one, and
        # emitting it invites a cycle to launch something pointless.
        if usable is not None and result["can_launch"]:
            result["max_steps"] = int(usable * fps)
            result["suggested_ceiling_s"] = round(usable)
    return result


def _format(p: dict) -> str:
    lines = []
    if p["window_remaining_s"] is None:
        src = p.get("window_source")
        where = f"no file at {src}" if src else f"no ${ARMED_UNTIL_ENV}"
        lines.append(f"arming window: none set — unbounded ({where})")
    elif p["window_expired"]:
        lines.append("arming window: EXPIRED — preflight must halt this cycle")
    else:
        h = p["window_remaining_s"] / 3600
        lines.append(f"arming window: {h:.1f}h left")
    if p["usable_s"] is not None and p["can_launch"]:
        lines.append(f"usable for training: {p['usable_s'] / 3600:.1f}h "
                     f"(bound by {p['bound_by']}, minus a "
                     f"{p['harvest_reserve_s'] // 60}min harvest reserve)")
    lines.append(f"can launch: {'yes' if p['can_launch'] else 'NO — too little time to be useful'}")
    if "max_steps" in p:
        lines.append(f"{p['env_id']}: {p['fps']} steps/s ({p['fps_source']})")
        lines.append(f"  -> at most {p['max_steps']:,} steps, "
                     f"ceiling {p['suggested_ceiling_s'] / 3600:.1f}h")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", help="size a run for this env id")
    parser.add_argument("--ceiling-hours", type=float,
                        help="an explicit ceiling, if tighter than the window")
    parser.add_argument("--armed-until",
                        help=f"epoch or file holding one; defaults to ${ARMED_UNTIL_ENV}")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    p = plan(args.env, args.ceiling_hours * 3600 if args.ceiling_hours else None,
             args.armed_until)
    print(json.dumps(p, indent=2) if args.json else _format(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
