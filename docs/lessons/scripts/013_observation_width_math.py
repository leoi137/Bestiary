"""Arithmetic for lesson 013 — what an observation is, and why its width is a
one-way door.

Everything here is read off live objects, never transcribed:

  * each registered env is built, and its `_obs_spec` is asked for its terms,
    its width and its hash (`src/bestiary/envs/obs_spec.py`);
  * the observation space and one real `reset()` vector are cross-checked
    against that declared width, so the number printed is the number the
    policy actually receives;
  * a committed checkpoint's `policy.pth` is opened and the actor's first
    weight matrix is measured directly — shape and element count — rather
    than multiplied out in prose.

No GPU, no training, writes nothing.

    venv/bin/python docs/lessons/scripts/013_observation_width_math.py
"""
from __future__ import annotations

import io
import zipfile

import gymnasium as gym
import numpy as np
import torch

import bestiary.envs  # noqa: F401  -- importing registers the env ids
from bestiary.paths import RUNS

ENV_IDS = [
    "Spyder-v0",
    "SpyderDesert-v0",
    "Hound-v0",
    "HoundDesert-v0",
    "HoundPD-v0",
    "HoundPDDesert-v0",
    "HoundPDTrackDesert-v0",
    "HoundPDTrackRelDesert-v0",
]

# One finished run per distinct observation width, so the checkpoint section
# measures a real first layer rather than a representative one.
CHECKPOINTS = [
    ("spyder_walk_v3", "Spyder-v0"),
    ("hound_pd_desert_v0", "HoundPDDesert-v0"),
    ("hound_track_rel_s1", "HoundPDTrackRelDesert-v0"),
]

HIDDEN = 256  # asserted below against the real matrix, not assumed


def env_facts(env_id: str) -> dict:
    env = gym.make(env_id)
    u = env.unwrapped
    spec = u._obs_spec
    obs, _ = env.reset(seed=0)
    # Three independent statements of the same width. If they ever disagree,
    # the number in the lesson is meaningless, so this is checked not trusted.
    assert env.observation_space.shape == (spec.width,), (
        f"{env_id}: Box{env.observation_space.shape} vs spec width {spec.width}")
    assert obs.shape == (spec.width,), (
        f"{env_id}: reset() gave {obs.shape}, spec says {spec.width}")
    env.close()
    return {
        "id": env_id,
        "width": spec.width,
        "hash": spec.hash,
        "terms": [(t.name, t.size) for t in spec.terms],
        "act": int(np.prod(env.action_space.shape)),
    }


def first_layer(run: str) -> dict:
    """Shape and element count of the actor's first weight matrix, read out of
    the committed checkpoint's `policy.pth`."""
    path = RUNS / run / "ant_sac_best.zip"
    with zipfile.ZipFile(path) as z:
        blob = z.read("policy.pth")
    sd = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "actor.latent_pi.0.weight" not in sd:
        sd = dict(sd)
    w = sd["actor.latent_pi.0.weight"]
    b = sd["actor.latent_pi.0.bias"]
    actor = {k: v for k, v in sd.items() if k.startswith("actor.")}
    return {
        "run": run,
        "bytes": path.stat().st_size,
        "shape": tuple(w.shape),
        "weights": int(w.numel()),
        "bias": int(b.numel()),
        "actor_params": int(sum(v.numel() for v in actor.values())),
        "total_params": int(sum(v.numel() for v in sd.values()
                                if hasattr(v, "numel"))),
    }


def main() -> None:
    print("1. What each env's policy actually sees\n")
    print(f"  {'env':<28}{'obs':>6}{'act':>6}  {'spec hash':<18}")
    facts = {}
    for env_id in ENV_IDS:
        f = env_facts(env_id)
        facts[env_id] = f
        print(f"  {f['id']:<28}{f['width']:>6}{f['act']:>6}  {f['hash']:<18}")

    print("\n2. Where the numbers come from — the declared term list\n")
    for env_id in ("Spyder-v0", "HoundDesert-v0", "HoundPDTrackRelDesert-v0"):
        f = facts[env_id]
        print(f"  {f['id']}  (width {f['width']}, hash {f['hash']})")
        for name, size in f["terms"]:
            print(f"      {name:<34}{size:>5}")
        print(f"      {'sum':<34}{sum(s for _, s in f['terms']):>5}\n")

    print("3. The first layer, measured inside a committed checkpoint\n")
    print(f"  {'run':<24}{'W1 shape':>14}{'weights':>10}{'bias':>7}"
          f"{'actor':>10}{'file bytes':>12}")
    for run, env_id in CHECKPOINTS:
        c = first_layer(run)
        w = facts[env_id]["width"]
        assert c["shape"] == (HIDDEN, w), (
            f"{run}: W1 is {c['shape']}, env {env_id} declares width {w}")
        assert c["weights"] == HIDDEN * w
        print(f"  {run:<24}{str(c['shape']):>14}{c['weights']:>10}"
              f"{c['bias']:>7}{c['actor_params']:>10}{c['bytes']:>12}")

    print("\n4. What a width change costs: the row count of W1\n")
    a = facts["Spyder-v0"]
    b = facts["HoundPDTrackRelDesert-v0"]
    for f in (a, b):
        print(f"  {f['id']:<28}{HIDDEN} x {f['width']:<5} = "
              f"{HIDDEN * f['width']:>7} weights in the first layer")
    print(f"\n  one extra observation value adds {HIDDEN} weights "
          f"(one column of W1)")
    print(f"  {a['width']} -> {b['width']} is "
          f"{b['width'] - a['width']} columns = "
          f"{HIDDEN * (b['width'] - a['width'])} weights that do not exist "
          f"in the smaller checkpoint")

    print("\n5. The dangerous case: a reorder at identical width\n")
    from bestiary.envs.obs_spec import ObsSpec, ObsTerm
    terms = tuple(ObsTerm(n, s) for n, s in a["terms"])
    same = ObsSpec(env="Spyder", terms=terms)
    swapped = ObsSpec(env="Spyder", terms=(terms[1], terms[0]) + terms[2:])
    print(f"  declared order   width {same.width}   hash {same.hash}")
    print(f"  first two terms swapped   width {swapped.width}   "
          f"hash {swapped.hash}")
    print(f"  widths equal: {same.width == swapped.width}      "
          f"hashes equal: {same.hash == swapped.hash}")
    print("  -> SAC.load() succeeds; train.py's obs-hash check is the only "
          "thing that fires")

    print("\n6. What the runs on disk actually recorded\n")
    import json
    print(f"  {'run':<24}{'width':>7}  {'hash':<18}")
    for d in sorted(RUNS.iterdir()):
        cfg = d / "config.json"
        if not cfg.is_file():
            continue
        rec = json.loads(cfg.read_text()).get("obs_spec")
        if rec is None:
            print(f"  {d.name:<24}{'-':>7}  {'(predates the record)':<18}")
        else:
            print(f"  {d.name:<24}{rec['width']:>7}  {rec['hash']:<18}")


    print("\n7. The run that was actually orphaned\n")
    for run, env_id in (("hound_desert_test150k", "HoundDesert-v0"),
                        ("hound_desert_v0", "HoundDesert-v0")):
        c = first_layer(run)
        live = facts[env_id]["width"]
        verdict = "loads" if c["shape"][1] == live else "DEAD"
        print(f"  {run:<24}W1 {str(c['shape']):>12}   env {env_id} is "
              f"{live:>4}   -> {verdict}")


if __name__ == "__main__":
    main()
