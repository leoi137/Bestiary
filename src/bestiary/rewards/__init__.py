"""Reward-shaping wrappers, selected by name via `--wrapper`.

Each wrapper adds its per-step shaping term to the reward AND writes the same
number into `info` under a `shaping/*` key, so the eval callback can log
shaped and base reward separately. Without that split you cannot tell whether
a policy improved or the shaping term just got more generous.
"""
from bestiary.rewards.shaping import WRAPPERS

__all__ = ["WRAPPERS"]
