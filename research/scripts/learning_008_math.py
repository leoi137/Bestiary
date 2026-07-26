"""The arithmetic in learnings/008, computed rather than asserted.

The number rule: no number enters the record unless code in the repo computed
it. This is that code. Inputs are the measured failure counts from
record/greedy_eval.py at n=60 (three disjoint seed blocks) and the evaluation
counts recovered from each run's TensorBoard event files.

    venv/bin/python research/scripts/learning_008_math.py
"""
from __future__ import annotations

# Measured: failures out of 60 deterministic episodes, seeds 0-19/100-119/200-219.
TORQUE_FAIL, TORQUE_N, TORQUE_EVALS = 16, 60, 14
PD_FAIL, PD_N, PD_EVALS = 6, 60, 19
SPYDER_BELOW, SPYDER_N = 57, 60

p_fail_torque = TORQUE_FAIL / TORQUE_N
p_fail_pd = PD_FAIL / PD_N
p_good_torque = 1 - p_fail_torque

print(f"torque failure rate : {TORQUE_FAIL}/{TORQUE_N} = {p_fail_torque:.4f}")
print(f"PD failure rate     : {PD_FAIL}/{PD_N} = {p_fail_pd:.4f}")
print(f"spyder below standing: {SPYDER_BELOW}/{SPYDER_N} = {SPYDER_BELOW/SPYDER_N:.4f}")
print()

# Selection bias: argmax over N single-episode evals picks a good-mode snapshot
# unless EVERY eval landed in the bad mode.
p_all_bad = p_fail_torque ** TORQUE_EVALS
print(f"P(all {TORQUE_EVALS} torque evals land in the bad mode) = "
      f"{p_fail_torque:.4f}**{TORQUE_EVALS} = {p_all_bad:.3e}")
print(f"P(best.zip is a good-mode snapshot)                = "
      f"1 - {p_all_bad:.3e} = {1 - p_all_bad:.6f}")
print()

# Why n=5 misleads.
p_clean5 = p_good_torque ** 5
print(f"P(0 failures in 5 torque episodes) = {p_good_torque:.4f}**5 = {p_clean5:.4f}")

p_pd_shows = 1 - (1 - p_fail_pd) ** 5
print(f"P(>=1 failure in 5 PD episodes)    = 1 - {1 - p_fail_pd:.4f}**5 = {p_pd_shows:.4f}")
print(f"P(the exact misleading n=5 picture) = {p_clean5:.4f} * {p_pd_shows:.4f} "
      f"= {p_clean5 * p_pd_shows:.4f}  (about 1 in {1 / (p_clean5 * p_pd_shows):.0f})")
