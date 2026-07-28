"""The arithmetic for `docs/lessons/009-actor-and-critic.md`.

Opens a real checkpoint this repo produced and counts what is actually inside
it, so the lesson can say "the critic is 3.0x the actor" from the bytes rather
than from memory.

Reads the frozen, content-addressed copy (`record/freeze.py`) rather than
`ant_sac_best.zip`, because the mutable one is overwritten mid-run and a lesson
that quotes it would be quoting a file that may not exist -- which is
`learnings/013` and the whole subject of cycle 011.

    venv/bin/python docs/lessons/scripts/actor_critic_math.py

Read-only. Needs no GPU and does not construct an env.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch  # noqa: E402

from bestiary import paths  # noqa: E402

RUN = "hound_track_rel_s1"
SHA = "607447d93151f987038334078f848ca3b7610d683b1e8d2aac2316d2d333a81b"
GAMMA = 0.99  # train.py


def main() -> int:
    ckpt = paths.RUNS / RUN / "measured" / f"{SHA}.zip"
    if not ckpt.exists():
        print(f"frozen checkpoint absent: {ckpt}")
        return 1

    with zipfile.ZipFile(ckpt) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        blob = z.read("policy.pth")
    state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)

    groups: dict[str, int] = {}
    shapes: dict[str, tuple] = {}
    for key, tensor in state.items():
        if not hasattr(tensor, "numel"):
            continue
        head = key.split(".")[0]
        groups[head] = groups.get(head, 0) + tensor.numel()
        shapes[key] = tuple(tensor.shape)

    total = sum(groups.values())
    actor = groups.get("actor", 0)
    critic = groups.get("critic", 0)

    print(f"checkpoint  {ckpt.name[:16]}...  {ckpt.stat().st_size:,} bytes\n")
    print("  inside the zip")
    for name, size in sizes.items():
        print(f"    {name:26s} {size:>10,} bytes")

    print("\n  learnable parameters in policy.pth")
    for name in sorted(groups, key=lambda k: -groups[k]):
        print(f"    {name:26s} {groups[name]:>10,}  ({100 * groups[name] / total:5.1f}%)")
    print(f"    {'TOTAL':26s} {total:>10,}")

    print("\n  the first and last layer of each")
    for key in ("actor.latent_pi.0.weight", "actor.mu.weight",
                "critic.qf0.0.weight", "critic.qf0.4.weight"):
        if key in shapes:
            print(f"    {key:34s} {shapes[key]}")

    print("\n  the numbers the lesson quotes")
    obs = shapes.get("actor.latent_pi.0.weight", (0, 0))[1]
    act = shapes.get("actor.mu.weight", (0, 0))[0]
    print(f"    observation width                {obs}")
    print(f"    action width                     {act}")
    print(f"    critic / actor parameters        {critic:,} / {actor:,} = {critic / actor:.2f}x")
    print(f"    critic optimizer / actor         {sizes['critic.optimizer.pth']:,} / "
          f"{sizes['actor.optimizer.pth']:,} = "
          f"{sizes['critic.optimizer.pth'] / sizes['actor.optimizer.pth']:.2f}x")
    # The horizon in STEPS is dimensionless; converting it to seconds needs the
    # env's own control rate, and guessing it is how the first draft of the
    # lesson said 100 Hz / "one second" when this env runs at 20 Hz / 5 s.
    import gymnasium as gym

    import bestiary.envs  # noqa: F401  (registers the env ids)
    env = gym.make("HoundPDTrackRelDesert-v0")
    dt = env.unwrapped.dt
    env.close()

    horizon = 1 / (1 - GAMMA)
    print(f"    control dt                       {dt} s  ({1 / dt:.0f} Hz)")
    print(f"    horizon 1/(1-gamma)              {horizon:.0f} steps at gamma={GAMMA}"
          f"  = {horizon * dt:.1f} s")

    # Why there are two critics and a target copy: SAC keeps twin Q networks and
    # a slow-moving copy of each. Count them from the keys rather than assert it.
    qfs = sorted({k.split(".")[1] for k in shapes if k.startswith("critic.")})
    targets = sorted({k.split(".")[0] for k in shapes if "target" in k})
    print(f"    Q networks inside the critic     {len(qfs)}  {qfs}")
    print(f"    target copies                    {len(targets)}  {targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
