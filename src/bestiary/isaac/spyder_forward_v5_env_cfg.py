"""Spyder-12, the forward diagnostic, on the v5 ground. One variable: the terrain.

    Bestiary-ForwardV5-Spyder-v0        the diagnostic config, on v5
    Bestiary-ForwardV5-Spyder-Play-v0   few robots, no noise, no shoving

WHAT THIS VARIANT IS FOR
------------------------
`research/decisions/0007` regenerated the gentle training terrain as **v5**:
the same world as v4 (seed 11, same layers, same layout, same 2.25 m span) with
the crest mathematics changed so the ground can physically exist. v4's dune and
mountain layers were ridged fields — `1 - |noise|` puts a C0 knife-edge at every
crest — and the smoothed surface reached **47 degrees**, past any angle of
repose. v5 rounds those creases and enforces a 36-degree repose cap.

0007's rule: **all future arms train on v5, and v4 stays committed and
untouched**, because terrain is a one-way door and repointing an existing
lineage's env cfg would silently invalidate every run, video and grid eval that
already stood on those bytes. Nothing in this file touches
`Bestiary-Forward-Spyder-v0`; this is a NEW task beside it.

EXACTLY ONE VARIABLE MOVES AGAINST `Bestiary-Forward-Spyder-v0`
---------------------------------------------------------------
The reward is still base-frame `v_x` at weight 1.0 and nothing else — inherited
from `SpyderForwardEnvCfg`, not restated here, so the two diagnostics cannot
drift apart. The robot, the observations (235 wide — a one-way door), the
actions, the commands, the events, the terminations, the curriculum, the
terrain MIX (50% bestiary tile / 25% pyramid slope / 25% random rough), the tile
size, the sampling and the grid shape are all inherited untouched.

What moves is **which bytes the bestiary tile reads**: `gentle_hfield.bin` ->
`gentle_v5_hfield.bin`. That is one leaf of one sub-terrain config, and
`check_spyder.py`'s `forward-v5-changes-only-the-ground` asserts exactly that —
it dumps both configs with `to_dict()` and requires the difference to be the
single dotted path `scene.terrain.terrain_generator.sub_terrains.
bestiary_gentle.hfield_path`.

The terrain is repointed by MUTATING the inherited generator rather than by
building a fresh one. A fresh `gentle_terrain_cfg(...)` call would be a second
statement of the mix's proportions, border widths and sub-terrain parameters —
two statements that agree today and drift on the day someone edits one of them.
Mutating one field of the object the parent already built makes "everything
else about the ground is identical" true by construction instead of by
inspection.

WHAT THE NUMBER MEANS, AND WHAT IT MAY NOT BE COMPARED TO
----------------------------------------------------------
`RewardManager` scales every term by `weight * step_dt`, so at weight 1.0 the
episode return is still METRES OF FORWARD TRAVEL, exactly as in
`research/episodes/014`.

It is nevertheless **not** the same measurement. 014's 44.72 m/episode was
earned on v4 — measured body slope P99 34.4 degrees, max 47 — and this task
runs on ground whose P99 is 29.8 degrees with a 36-degree cap and a footstep
P99 of 2.2 in (5.7 cm) instead of 5.5 in (14 cm). Easier ground pays more
metres for the same competence. The two returns belong in different columns,
and a run of this task is a fresh probe under the seed rule, never a
continuation of 014's.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from bestiary import paths
from bestiary.isaac.spyder_forward_env_cfg import SpyderForwardEnvCfg, SpyderForwardEnvCfg_PLAY
from bestiary.isaac.spyder_gentle_env_cfg import gentle_terrain_cfg
from bestiary.terrain.gentle import Z_SPAN_M as GENTLE_Z_SPAN_M

#: The key the gentle mix files its bestiary tile under
#: (`spyder_gentle_env_cfg.gentle_terrain_cfg`). Named once so the swap below
#: and the oracle's check cannot disagree about which sub-terrain is ours.
GENTLE_SUBTERRAIN_KEY = "bestiary_gentle"

#: Metres of elevation the v5 asset's normalised [0, 1] samples span.
#:
#: 2.25 is `research/decisions/0007`'s number, typed here from the decision
#: rather than imported, deliberately — and then checked against
#: `terrain/gentle.py:Z_SPAN_M`, which the generator makes exact by
#: construction (it rescales the composed field until `ptp(h)` equals it). Two
#: independent statements of one physical fact that must agree: the decision
#: this task cites, and the generator that wrote the bytes. If a later terrain
#: version moves the span, `assert_v5_span_is_the_decisions` fails loudly here
#: instead of a run silently standing on ground 2x its documented relief.
V5_Z_SPAN_M = 2.25


def assert_v5_span_is_the_decisions() -> None:
    """The generator's span and decision 0007's number are the same number."""
    if abs(GENTLE_Z_SPAN_M - V5_Z_SPAN_M) > 1e-12:
        raise AssertionError(
            f"terrain/gentle.py:Z_SPAN_M is {GENTLE_Z_SPAN_M} m but "
            f"research/decisions/0007 declares the v5 span as {V5_Z_SPAN_M} m. "
            "The generator rescales the field to its own constant, so the "
            "committed bytes span the generator's number — one of the two is "
            "now wrong, and every slope on this terrain is scaled by their "
            "ratio."
        )


def use_gentle_v5_terrain(cfg) -> None:
    """Repoint the config's bestiary tile at the v5 heightfield. One field.

    Called after `super().__post_init__()`, on a config whose terrain generator
    is already the gentle mix, so everything about the ground except WHICH bytes
    it reads is inherited by construction: the 50/25/25 proportions, the 8x8 m
    tiles, the border widths, the two Isaac Lab sub-terrains at their shipped
    parameters, the horizontal sampling and the grid shape.

    Raises rather than no-opping when the generator carries no bestiary tile — a
    silent miss here is a run that trains on v4 while its task id, its log
    directory and its ledger row all say v5, which is the exact class of quiet
    terrain failure the terrain invariant in `CLAUDE.md` exists to stop.
    """
    assert_v5_span_is_the_decisions()

    gen = cfg.scene.terrain.terrain_generator
    if gen is None:
        raise AssertionError(
            "the config has no terrain generator to repoint at v5 — this task "
            "trains on a heightfield mix, not on a plane."
        )
    sub = gen.sub_terrains.get(GENTLE_SUBTERRAIN_KEY)
    if sub is None:
        raise AssertionError(
            f"no {GENTLE_SUBTERRAIN_KEY!r} sub-terrain to repoint; the generator "
            f"carries {sorted(gen.sub_terrains)}. The v5 swap changes which "
            "bytes the bestiary tile reads, so it needs that tile to exist."
        )
    sub.hfield_path = str(paths.GENTLE_V5_HFIELD)
    sub.z_span_m = V5_Z_SPAN_M


def use_gentle_v5_mix(cfg) -> None:
    """Replace a NON-gentle generator with the gentle mix on v5 ground.

    For a config whose parent stands on different terrain entirely — the Hound,
    whose desert tiles are the wrong ground for a new arm under decision 0007.
    The sampling, the grid shape and the curriculum flag are read off the
    generator being replaced and written back onto the new one, so this changes
    WHICH GROUND and nothing about how much of it there is or how finely it is
    sampled; a config that declared 3x3 tiles at native sampling with the
    curriculum off (every Play twin) keeps all three.

    `gentle_terrain_cfg` and `desert_terrain_cfg` already agree on tile size,
    border widths, vertical scale, slope threshold, caching, colour scheme and
    the two Isaac Lab sub-terrains' parameters — they were written from one
    another — so the surviving difference is the bestiary tile itself.
    """
    old = cfg.scene.terrain.terrain_generator
    if old is None:
        raise AssertionError(
            "the config has no terrain generator to replace — this task trains "
            "on a heightfield mix, not on a plane."
        )
    gen = gentle_terrain_cfg(old.horizontal_scale)
    gen.num_rows = old.num_rows
    gen.num_cols = old.num_cols
    gen.curriculum = old.curriculum
    cfg.scene.terrain.terrain_generator = gen
    use_gentle_v5_terrain(cfg)


@configclass
class SpyderForwardV5EnvCfg(SpyderForwardEnvCfg):
    """The forward diagnostic with its bestiary tile repointed at v5."""

    def __post_init__(self) -> None:
        super().__post_init__()
        use_gentle_v5_terrain(self)


@configclass
class SpyderForwardV5EnvCfg_PLAY(SpyderForwardEnvCfg_PLAY):
    """Viewer config for the v5 diagnostic: few robots, nothing random.

    Descends from `SpyderForwardEnvCfg_PLAY`, NOT from `SpyderForwardV5EnvCfg`,
    for the reason that file gives: the Play overrides (16 envs, native terrain
    sampling, corruption and pushes off) are inherited rather than copied, and a
    copy is what drifts. The cost is the same one — a change to the training
    class does not reach here — so the v5 swap is a second call to the shared
    mutator, and `check_spyder.py` checks BOTH configs precisely because an edit
    that reached one call and not the other would produce a viewer that plays a
    policy on ground it never trained on.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        use_gentle_v5_terrain(self)
