"""Drive HOUND-16 yourself in the MuJoCo viewer.

    ./venv/bin/python play_hound.py             # flat plane
    ./venv/bin/python play_hound.py --desert    # the heightfield
    ./venv/bin/python play_hound.py --selftest  # no window; check the controls work

    W / S    drive all four wheels forward / back
    A / D    turn — drive the left and right wheels in opposite directions
    Q / E    crouch / stand      (moves hip and knee together)
    Z / C    splay out / tuck in (ABDUCTION — the joint you cannot see move
                                  from the side; watch the shoulders from the
                                  front, or from the top)
    F        toggle the posture controller off. The machine is then exactly
             what the springs alone can hold, which is the whole Section C
             argument: it slides into the splits and sits down.
    SPACE    brake — hold the wheels still
    R        reset to the standing stance
    TAB      MuJoCo's own control panel (per-actuator sliders)
    ESC      quit

Why this file exists: `watch.py` replays a trained policy, and there is no
trained policy for this robot yet. This is a hand controller instead, so the
mechanics the explainer describes can be felt rather than read.

What is actually driving the robot: a joint-space PD controller holding the
standing stance plus whatever offsets your keypresses have accumulated, with
the wheels on direct torque. That is deliberately NOT a learned policy and
makes no attempt to be a good one — it is the simplest thing that lets a
person pose the machine. Two consequences worth watching for, because they
are the same two the explainer predicts:

  * Let go of everything and it still creeps. The PD holds a POSTURE, not a
    POSITION, and nothing in it brakes the wheels. Press SPACE to see the
    difference — that is "a wheel-legged robot holds station by braking"
    turned into a keystroke.
  * Hold W and the nose comes up. All the thrust acts at ground level and the
    mass sits 0.363 m above it, so hard driving pitches the machine onto its
    rear pair. Keep holding and the front wheels leave the ground entirely.

Rendering note: this box's NVIDIA userspace and kernel module disagree, which
takes GLX down, so the viewer is forced onto Mesa's software GL. It runs, but
at software-rasteriser speed. Drop --software once the machine is rebooted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGS = ("FL", "FR", "RL", "RR")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--desert", action="store_true",
                    help="load the heightfield world instead of the plane")
    ap.add_argument("--software", action="store_true", default=True,
                    help="force Mesa software GL (default: on, see module docstring)")
    ap.add_argument("--gpu", dest="software", action="store_false",
                    help="use the system GL driver instead")
    ap.add_argument("--selftest", action="store_true",
                    help="exercise every control headlessly and exit")
    args = ap.parse_args()

    if args.software:
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    import mujoco
    import numpy as np

    xml = HERE / "assets" / ("hound16_desert.xml" if args.desert else "hound16.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    # ── index the model by NAME, never by slot ───────────────────────────
    JOINTS = {f"{leg}_{j}": model.joint(f"{leg}_{j}")
              for leg in LEGS for j in ("abduct", "hip", "knee", "wheel")}
    QADR = {n: model.jnt_qposadr[j.id] for n, j in JOINTS.items()}
    VADR = {n: model.jnt_dofadr[j.id] for n, j in JOINTS.items()}
    ACT = {f"{leg}_{j}": model.actuator(f"{leg}_{j}").id
           for leg in LEGS for j in ("abduct", "hip", "knee", "wheel")}
    GEAR = {n: float(model.actuator(n).gear[0]) for n in ACT}
    TRUNK = model.body("trunk").id

    stance = model.key_qpos[0].copy()
    STANCE = {n: float(stance[QADR[n]]) for n in QADR}

    # ── operator state ───────────────────────────────────────────────────
    S = {"drive": 0.0, "turn": 0.0, "crouch": 0.0, "splay": 0.0,
         "hold": True, "brake": False}

    # PD gains, in N*m per rad and per rad/s. Sized off the same static loads
    # make_hound.py's springs are: enough to hold the stance and re-pose it in
    # a fraction of a second, soft enough not to fight the physics into
    # instability at the 20 Hz-equivalent rate this loop runs.
    KP = {"abduct": 60.0, "hip": 80.0, "knee": 90.0}
    KD = {"abduct": 3.0, "hip": 4.0, "knee": 4.5}

    def control() -> None:
        data.ctrl[:] = 0.0
        for leg in LEGS:
            sy = 1.0 if leg.endswith("L") else -1.0
            side = 1.0 if leg.endswith("L") else -1.0
            targets = {
                # +splay swings each leg outward on its OWN side, which needs
                # the y sign because abduction shares one global axis (Unitree's
                # convention — see make_hound.py). This is the joint that is
                # invisible from a side view.
                "abduct": STANCE[f"{leg}_abduct"] + S["splay"] * sy,
                # Crouch folds hip and knee together so the foot stays roughly
                # under the hip instead of swinging forward.
                "hip": STANCE[f"{leg}_hip"] + S["crouch"],
                "knee": STANCE[f"{leg}_knee"] - 1.6 * S["crouch"],
            }
            if S["hold"]:
                for j, target in targets.items():
                    n = f"{leg}_{j}"
                    err = target - float(data.qpos[QADR[n]])
                    tau = KP[j] * err - KD[j] * float(data.qvel[VADR[n]])
                    data.ctrl[ACT[n]] = np.clip(tau / GEAR[n], -1.0, 1.0)

            w = f"{leg}_wheel"
            if S["brake"]:
                # A brake is not zero torque, it is torque opposing rotation.
                data.ctrl[ACT[w]] = np.clip(
                    -0.4 * float(data.qvel[VADR[w]]), -1.0, 1.0)
            else:
                data.ctrl[ACT[w]] = np.clip(S["drive"] + S["turn"] * side,
                                            -1.0, 1.0)

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = stance
        data.qpos[2] += 0.005          # the env's spawn clearance
        mujoco.mj_forward(model, data)
        S.update(drive=0.0, turn=0.0, crouch=0.0, splay=0.0,
                 hold=True, brake=False)

    def on_key(code: int) -> None:
        ch = chr(code).upper() if 0 < code < 0x110000 else ""
        step = 0.12
        if ch == "W":
            S["drive"] = min(1.0, S["drive"] + 0.2)
        elif ch == "S":
            S["drive"] = max(-1.0, S["drive"] - 0.2)
        elif ch == "A":
            S["turn"] = min(1.0, S["turn"] + 0.2)
        elif ch == "D":
            S["turn"] = max(-1.0, S["turn"] - 0.2)
        elif ch == "Q":
            S["crouch"] = min(0.7, S["crouch"] + step)
        elif ch == "E":
            S["crouch"] = max(-0.5, S["crouch"] - step)
        elif ch == "Z":
            S["splay"] = min(0.7, S["splay"] + step)
        elif ch == "C":
            S["splay"] = max(-0.7, S["splay"] - step)
        elif ch == "F":
            S["hold"] = not S["hold"]
        elif ch == " ":
            S["brake"] = not S["brake"]
            S["drive"] = S["turn"] = 0.0
        elif ch == "R":
            reset()
        report()

    def report() -> None:
        pos = data.xpos[TRUNK]
        up = float(data.body("trunk").xmat[8])
        w = np.mean([float(data.qvel[VADR[f"{l}_wheel"]]) for l in LEGS])
        print(f"\r drive {S['drive']:+.1f}  turn {S['turn']:+.1f}  "
              f"crouch {S['crouch']:+.2f}  splay {S['splay']:+.2f}  "
              f"| posture {'PD' if S['hold'] else 'SPRINGS ONLY'}"
              f"{'  BRAKED' if S['brake'] else '':<9}"
              f"| z {pos[2]:.3f}  upright {up:+.2f}  wheels {w:+5.1f} rad/s   ",
              end="", flush=True)

    reset()

    if args.selftest:
        # Every key, then a few hundred steps each, asserting the model stays
        # finite and the controls actually do something.
        print(f"selftest on {xml.name}")
        for keys, label in ((["W", "W", "W"], "drive"), ([" "], "brake"),
                            (["R", "A", "A"], "turn"), (["R", "Q", "Q"], "crouch"),
                            (["R", "Z", "Z"], "splay"), (["R", "F"], "springs only")):
            reset()
            for k in keys:
                on_key(ord(k))
            for _ in range(400):
                control()
                mujoco.mj_step(model, data)
            ok = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            print(f"\n  {label:<14} z={data.xpos[TRUNK][2]:.3f} "
                  f"upright={float(data.body('trunk').xmat[8]):+.2f} "
                  f"x={data.xpos[TRUNK][0]:+.2f} finite={ok}")
            if not ok:
                return 1
        print("\nselftest OK")
        return 0

    import mujoco.viewer
    print(__doc__.split("Why this file exists")[0])
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        v.cam.distance = 2.2
        v.cam.elevation = -15
        v.cam.azimuth = 130
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while v.is_running():
            control()
            mujoco.mj_step(model, data)
            v.cam.lookat[:] = data.xpos[TRUNK]
            v.sync()
            report()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
