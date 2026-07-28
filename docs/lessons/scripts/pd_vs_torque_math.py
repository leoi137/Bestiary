"""Arithmetic behind lesson 011 — torque control versus PD position targets.

Every number lesson 011 quotes is printed here, and every constant is read
off the committed model XMLs, the env source, and research/ledger.jsonl
rather than typed in from memory.  Run from the repository root:

    venv/bin/python docs/lessons/scripts/pd_vs_torque_math.py

Nothing here touches the GPU: MuJoCo is loaded on the CPU only to ask the
model for the knee joint's inertia about its own axis, which is the one
number the second-order response needs and the one number a regex cannot
get out of the XML.
"""

from __future__ import annotations

import json
import re

import mujoco
import numpy as np

from bestiary.paths import ASSETS, RESEARCH

PD_XML = ASSETS / "hound16pd_desert.xml"
TORQUE_XML = ASSETS / "hound16_desert.xml"
LEDGER = RESEARCH / "ledger.jsonl"


def first_actuator(xml_text: str, tag: str, joint_suffix: str) -> dict[str, str]:
    """Attributes of the first <tag> actuator whose joint ends in joint_suffix."""
    for line in xml_text.splitlines():
        line = line.strip()
        if not line.startswith(f"<{tag} "):
            continue
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', line))
        if attrs.get("joint", "").endswith(joint_suffix):
            return attrs
    raise LookupError(f"no <{tag}> actuator on a *_{joint_suffix} joint")


def ledger_row(run: str) -> dict:
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("run") == run:
            return row
    raise LookupError(f"no ledger row for {run!r}")


def main() -> None:
    pd_text = PD_XML.read_text()
    torque_text = TORQUE_XML.read_text()

    # ── the control rate, read from the model and the env, never remembered ──
    timestep = float(re.search(r'timestep="([\d.]+)"', pd_text).group(1))
    hound_src = (
        __import__("bestiary.envs.hound", fromlist=["x"]).__file__
    )
    frame_skip = int(
        re.search(r"frame_skip: int = (\d+)", open(hound_src).read()).group(1)
    )
    dt = timestep * frame_skip

    print("== control rates ==")
    print(f"physics timestep        {timestep} s   ({1 / timestep:.0f} Hz)")
    print(f"frame_skip              {frame_skip}")
    print(f"one control step        {dt:.3f} s   ({1 / dt:.0f} Hz)")
    print(f"PD updates per action   {frame_skip}")

    # ── the two action spaces, same joint, same robot ────────────────────────
    knee_pd = first_actuator(pd_text, "position", "knee")
    knee_tq = first_actuator(torque_text, "motor", "knee")
    kp = float(knee_pd["kp"])
    kv = float(knee_pd["kv"])
    lo, hi = (float(v) for v in knee_pd["ctrlrange"].split())
    f_lo, f_hi = (float(v) for v in knee_pd["forcerange"].split())
    gear = float(knee_tq["gear"])

    print("\n== the knee, both ways ==")
    print(f"PD      kp={kp} N*m/rad  kv={kv} N*m/(rad/s)")
    print(f"        ctrl is an ANGLE in [{lo}, {hi}] rad")
    print(f"        forcerange +/-{f_hi} N*m")
    print(f"torque  gear={gear} N*m, ctrl is a FRACTION in [-1, 1]")
    print(f"        same ceiling: forcerange {f_hi} == gear {gear} -> {f_hi == gear}")

    # ── work tau = kp*(q_des - q) - kv*qdot on the real stance pose ──────────
    model = mujoco.MjModel.from_xml_path(str(PD_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "FL_knee")
    q_stance = float(data.qpos[model.jnt_qposadr[jid]])

    dof = model.jnt_dofadr[jid]
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, full_m, data.qM)
    inertia = float(full_m[dof, dof])

    q_des = -1.2  # a modest knee extension, well inside ctrlrange
    assert lo <= q_des <= hi, (q_des, lo, hi)
    qdot = 0.0
    err = q_des - q_stance
    tau = kp * err - kv * qdot
    frac = tau / gear

    print("\n== one step, worked ==")
    print(f"q      (stance knee)    {q_stance:.5f} rad")
    print(f"q_des  (commanded)      {q_des} rad")
    print(f"error  q_des - q        {err:.5f} rad")
    print(f"qdot                    {qdot} rad/s")
    print(f"tau = kp*err - kv*qdot  {tau:.3f} N*m")
    print(f"forcerange ceiling      {f_hi} N*m  -> saturated? {tau > f_hi}")
    print(f"same torque under gear  ctrl = tau/gear = {frac:.4f}")
    print(f"error that saturates    forcerange/kp = {f_hi / kp:.4f} rad "
          f"({np.degrees(f_hi / kp):.1f} deg)")
    print(f"full-travel error       {hi - q_stance:.5f} rad -> "
          f"kp*err = {kp * (hi - q_stance):.1f} N*m, clipped to {f_hi}")

    # a constant action: PD settles, torque accelerates
    tau_at_target = kp * 0.0 - kv * 0.0
    alpha = tau / inertia
    print("\n== what a CONSTANT action does ==")
    print(f"knee inertia about its axis  {inertia:.6f} kg*m^2")
    print(f"PD, once q reaches q_des     tau = {tau_at_target:.1f} N*m (holds)")
    print(f"torque, ctrl held at {frac:.4f}   tau = {tau:.3f} N*m forever")
    print(f"  -> angular accel           {alpha:.1f} rad/s^2")
    print(f"  -> after one step {dt:.2f} s     {0.5 * alpha * dt ** 2:.3f} rad")

    omega = (kp / inertia) ** 0.5
    zeta = kv / (2.0 * (kp * inertia) ** 0.5)
    print("\n== the inner loop as a second-order system ==")
    print(f"omega_n = sqrt(kp/I)    {omega:.2f} rad/s  ({omega / (2 * np.pi):.2f} Hz)")
    print(f"zeta = kv/(2*sqrt(kp*I)) {zeta:.3f}  ->  "
          f"{'overdamped, no overshoot' if zeta > 1 else 'underdamped'}")

    # ── what the ledger measured ─────────────────────────────────────────────
    tq = ledger_row("hound_desert_v0")
    pd = ledger_row("hound_pd_desert_v0")
    # Both step counts are stated in the PD row's own notes field; parse them
    # out rather than retyping, so this script cannot drift from the ledger.
    m = re.search(
        r"eval>=1100 at ([\d,]+) steps vs the torque run's ([\d,]+)", pd["notes"]
    )
    steps_pd = int(m.group(1).replace(",", ""))
    steps_tq = int(m.group(2).replace(",", ""))
    print("\n== what the ledger measured (1 seed per arm) ==")
    print(f"torque run  {tq['run']:<20} peak eval {tq['best_eval_return']:.1f}")
    print(f"PD run      {pd['run']:<20} peak eval {pd['best_eval_return']:.1f}")
    print(f"steps to eval>=1100   torque {steps_tq:,}   PD {steps_pd:,}")
    print(f"ratio                 {steps_tq / steps_pd:.2f}x fewer samples")
    print(f"peak eval ratio       PD / torque = "
          f"{pd['best_eval_return'] / tq['best_eval_return']:.3f}")
    print(f"peak eval difference  {pd['best_eval_return'] - tq['best_eval_return']:.1f} "
          "return points (PD is LOWER)")
    print(f"seeds per arm         {tq.get('seeds', 1)} and {pd.get('seeds', 1)} "
          "-> a probe, provisional")


if __name__ == "__main__":
    main()
