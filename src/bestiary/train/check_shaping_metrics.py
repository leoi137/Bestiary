"""Oracle for `train.shaping_metrics` — the omit-vs-zero distinction.

The bug this fences: `info.get("shaping/idle_legs", 0.0)` logged a metric that
was never measured as the number 0.0. TensorBoard cannot distinguish that from
a real measurement of zero, so nine runs carry a flat, dead
`eval/mean_idle_legs` and `metric-liveness` reports each as a failure.

The regression is silent by construction — a run trains fine, the series
exists, and only a human comparing the wrapper against the chart would notice.
So it gets an oracle rather than a comment.

    venv/bin/python -m bestiary.train.check_shaping_metrics
"""
from __future__ import annotations

import sys

from bestiary.train.train import shaping_metrics

BASE = "eval/base_reward"
IDLE = "eval/mean_idle_legs"


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main() -> int:
    print("shaping_metrics — a metric never observed must be ABSENT, not 0.0")
    results = []

    # No wrapper: neither key ever appeared. This is the nine-run case.
    out = shaping_metrics(total_reward=1234.5, total_shaping=0.0, n_shaping=0,
                          total_idle=0.0, n_idle=0)
    results.append(_check(
        "no wrapper omits both", out == {},
        f"got {out!r}, want {{}} — a key here is the original bug"))
    results.append(_check(
        "no wrapper does not emit a zero", IDLE not in out and BASE not in out,
        f"{IDLE} and {BASE} both absent"))

    # Wrapper active, and the idle count genuinely IS zero. The metric must be
    # present and equal to 0.0 — the case that proves absence means absence
    # rather than 'the value happened to be zero'.
    out = shaping_metrics(total_reward=100.0, total_shaping=10.0, n_shaping=50,
                          total_idle=0.0, n_idle=50)
    results.append(_check(
        "a real measured zero is REPORTED", out.get(IDLE) == 0.0,
        f"{IDLE} = {out.get(IDLE)!r}, want 0.0 present"))
    results.append(_check(
        "base_reward subtracts shaping", out.get(BASE) == 90.0,
        f"{BASE} = {out.get(BASE)!r}, want 100.0 - 10.0 = 90.0"))

    # The mean divides by the steps that CARRIED the key, not by episode
    # length. Dividing by n_steps would silently scale the metric down
    # whenever the key is intermittent.
    out = shaping_metrics(total_reward=0.0, total_shaping=0.0, n_shaping=1,
                          total_idle=8.0, n_idle=4)
    results.append(_check(
        "mean divides by observed count", out.get(IDLE) == 2.0,
        f"{IDLE} = {out.get(IDLE)!r}, want 8.0/4 = 2.0"))

    # One key present and the other absent must not drag its partner along.
    out = shaping_metrics(total_reward=5.0, total_shaping=1.0, n_shaping=3,
                          total_idle=0.0, n_idle=0)
    results.append(_check(
        "keys are independent", set(out) == {BASE},
        f"got {sorted(out)}, want only [{BASE}]"))

    # A negative count is a caller bug and must be loud, not silently falsy.
    try:
        shaping_metrics(0.0, 0.0, -1, 0.0, 0)
        results.append(_check("negative count raises", False, "no exception"))
    except ValueError as exc:
        results.append(_check(
            "negative count raises", "n_shaping=-1" in str(exc),
            f"ValueError naming the value: {exc}"))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} assertions passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
