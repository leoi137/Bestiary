"""Render HOUND-16: previews, a labelled leg diagram, and the mechanics plots.

    ./venv/bin/python render_hound.py            # everything, into assets/hound/
    ./venv/bin/python render_hound.py --only preview

Outputs
    assets/hound/preview.png        three-quarter view on the plane
    assets/hound/desert.png         the same machine on the desert heightfield
    assets/hound/legdiagram.png     one leg, four joints, labelled and to scale
    assets/hound/mechanics.png      the three measurements that define it

Rendering note: this repo's box has an NVIDIA driver whose kernel module and
userspace disagree, which takes GLX, EGL and CUDA down together. MuJoCo's
offscreen renderer still works through Mesa's software GL, so this file forces
that before importing mujoco — a few seconds per frame instead of instant, and
completely deterministic, which for figures that get committed is a fair
trade. Delete the two os.environ lines once the box has been rebooted.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch  # noqa: E402

from make_hound import SPEC  # noqa: E402

OUT = Path(__file__).resolve().parent / "assets" / "hound"
LEGS = ("FL", "FR", "RL", "RR")

# The joint colour code, shared by every figure here and by the model's own
# materials, so a hub in a render and a line in a plot mean the same joint.
C_ABDUCT = "#C22B29"
C_HIP = "#26A63F"
C_KNEE = "#3366D9"
C_WHEEL = "#D18C1F"
C_INK = "#1B1D21"


# ── MuJoCo renders ───────────────────────────────────────────────────────────
def render(model_path: str, out: Path, width=1600, height=1000, *,
           azimuth=135.0, elevation=-18.0, distance=1.5, lookat=(0, 0, 0.20),
           settle_steps=200, ctrl=None, shadow=True):
    """One offscreen frame of the settled machine."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    d.qpos[2] += 0.005
    mujoco.mj_forward(m, d)
    for k in range(settle_steps):
        d.ctrl[:] = 0 if ctrl is None else ctrl(k)
        mujoco.mj_step(m, d)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    trunk = d.xpos[m.body("trunk").id]
    cam.lookat[:] = [trunk[0] + lookat[0], trunk[1] + lookat[1], lookat[2]]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = False
    scn = mujoco.MjvScene(m, maxgeom=10000)
    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow

    with mujoco.Renderer(m, height, width) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        r._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
        px = r.render()
    plt.imsave(out, px)
    print(f"  wrote {out}  ({width}x{height})")
    return px


def render_previews() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render("assets/hound16.xml", OUT / "preview.png",
           azimuth=132, elevation=-16, distance=1.45)
    render("assets/hound16_desert.xml", OUT / "desert.png",
           azimuth=118, elevation=-12, distance=1.9, lookat=(0, 0, 0.25))


def render_contact_sheet() -> None:
    """Four views in one image: the machine read as hardware."""
    OUT.mkdir(parents=True, exist_ok=True)
    views = [
        ("three-quarter", dict(azimuth=132, elevation=-16, distance=1.45)),
        ("side (+X is forward)", dict(azimuth=90, elevation=-4, distance=1.35)),
        ("front", dict(azimuth=180, elevation=-6, distance=1.25)),
        ("top", dict(azimuth=90, elevation=-80, distance=1.35)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor="white")
    for ax, (label, kw) in zip(axes.ravel(), views):
        px = render("assets/hound16.xml", OUT / "_tmp.png", 1100, 760, **kw)
        ax.imshow(px)
        ax.set_title(label, fontsize=11, color=C_INK, pad=6)
        ax.axis("off")
    (OUT / "_tmp.png").unlink(missing_ok=True)
    fig.suptitle("HOUND-16 — 4 legs x (abduction + hip + knee + wheel)",
                 fontsize=14, color=C_INK, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "views.png", dpi=110, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / 'views.png'}")


# ── The leg diagram ──────────────────────────────────────────────────────────
def leg_diagram() -> None:
    """One leg, to scale, with all four joints labelled.

    Drawn from Spec rather than traced from a render, so it cannot go stale:
    change a link length and this picture changes with it.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    t1, t2 = SPEC.stance_hip, SPEC.stance_hip + SPEC.stance_knee
    L1, L2, r = SPEC.thigh_len, SPEC.calf_len, SPEC.wheel_r
    hip = np.array([0.0, 0.0])
    knee = hip + np.array([-L1 * np.sin(t1), -L1 * np.cos(t1)])
    axle = knee + np.array([-L2 * np.sin(t2), -L2 * np.cos(t2)])
    ground = axle[1] - r

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.5, 7.4), facecolor="white",
        gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- left panel: the sagittal chain -------------------------------------
    axL.axhline(ground, color="#C9CCD1", lw=2, zorder=0)
    axL.fill_between([-0.56, 0.30], ground - 0.06, ground,
                     color="#EDEFF2", zorder=0)
    axL.text(0.28, ground - 0.028, "ground", ha="right", va="top",
             fontsize=9, color="#8A9099")

    # trunk stub
    axL.plot([-0.02, 0.16], [0, 0], color=C_INK, lw=9, solid_capstyle="round",
             zorder=2)
    axL.text(0.17, 0.012, "trunk", fontsize=10, color=C_INK, va="bottom")

    # links
    axL.plot(*zip(hip, knee), color="#3E434A", lw=7, solid_capstyle="round",
             zorder=3)
    axL.plot(*zip(knee, axle), color="#3E434A", lw=5.5, solid_capstyle="round",
             zorder=3)

    # wheel
    axL.add_patch(Circle(axle, r, facecolor="#17181A", edgecolor=C_WHEEL,
                         lw=2.5, zorder=4))
    axL.add_patch(Circle(axle, r * 0.30, facecolor=C_WHEEL, zorder=5))
    for a in (t2, t2 + np.pi / 2):
        axL.plot([axle[0] - r * 0.82 * np.cos(a), axle[0] + r * 0.82 * np.cos(a)],
                 [axle[1] - r * 0.82 * np.sin(a), axle[1] + r * 0.82 * np.sin(a)],
                 color="#8C9298", lw=1.6, zorder=5)

    # joints
    # Explicit label positions: the wheel is 0.085 m across, so a generic
    # offset puts its caption inside the tyre.
    for pt, col, name, sub, xy in (
        (hip, C_HIP, "hip", "pitch, +Y\n23.7 N·m", (-0.30, 0.10)),
        (knee, C_KNEE, "knee", "pitch, +Y\n40 N·m", (-0.40, -0.02)),
        (axle, C_WHEEL, "wheel", "spin, +Y · UNLIMITED\n3 N·m", (-0.40, -0.20)),
    ):
        axL.add_patch(Circle(pt, 0.019, facecolor=col, edgecolor="white",
                             lw=2, zorder=6))
        axL.annotate(f"{name}\n{sub}", pt, xytext=xy,
                     fontsize=9.5, color=col, weight="bold",
                     arrowprops=dict(arrowstyle="-", color=col, lw=1.2,
                                     shrinkA=0, shrinkB=8))

    # abduction lives out of plane — show it as a marker on the hip
    axL.add_patch(Circle(hip, 0.030, facecolor="none", edgecolor=C_ABDUCT,
                         lw=2, ls=(0, (3, 2)), zorder=5))
    axL.annotate("abduction\nroll, +X (out of page)\n23.7 N·m",
                 hip, xytext=(-0.20, 0.10), fontsize=9.5,
                 color=C_ABDUCT, weight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_ABDUCT, lw=1.2,
                                 shrinkA=0, shrinkB=4))

    # the two dimensions that matter
    axL.annotate("", (hip[0] + 0.10, hip[1]), (hip[0] + 0.10, ground),
                 arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.3))
    axL.text(hip[0] + 0.115, (hip[1] + ground) / 2,
             f"stand height\n{SPEC.stand_z:.3f} m", fontsize=9.5, color="#6B7280",
             va="center")
    axL.plot([hip[0], hip[0]], [hip[1] + 0.02, ground - 0.03], color=C_HIP,
             lw=1.1, ls=(0, (2, 3)), zorder=1)
    axL.text(hip[0] - 0.005, ground - 0.045,
             "the axle is SOLVED to sit here —\ndirectly under the hip, so the\n"
             "ground reaction makes no\nmoment about it",
             fontsize=8.6, color=C_HIP, ha="center", va="top")

    axL.set_xlim(-0.56, 0.30)
    axL.set_ylim(ground - 0.24, 0.20)
    axL.set_aspect("equal")
    axL.axis("off")
    axL.set_title("One leg, four joints  (to scale, standing stance)",
                  fontsize=12.5, color=C_INK, pad=10)

    # ---- right panel: why the wheel is the odd one out ----------------------
    axR.axis("off")
    axR.set_title("Why the fourth joint is not just a fifth link",
                  fontsize=12.5, color=C_INK, pad=10)
    rows = [
        ("", "abduction / hip / knee", "wheel"),
        ("range", "limited (±46°, 218°, 115°)", "NONE — turns forever"),
        ("rest pose", "sprung toward the stance", "none; a spring would\nundo every metre driven"),
        ("in the observation", "angle AND velocity", "velocity ONLY — the angle\nis an unbounded integrator"),
        ("sized by", "the load it must hold", "what the ground accepts"),
        ("peak torque", "23.7 / 23.7 / 40 N·m", "3.0 N·m"),
        ("passive stability", "stable (a foot grips)", "UNSTABLE fore-aft —\nthe contact rolls away"),
    ]
    y = 0.92
    for i, (a, b, c) in enumerate(rows):
        head = i == 0
        axR.text(0.00, y, a, fontsize=10, color="#6B7280", va="top",
                 weight="bold" if head else "normal")
        axR.text(0.31, y, b, fontsize=10, va="top", color=C_INK,
                 weight="bold" if head else "normal")
        axR.text(0.70, y, c, fontsize=10, va="top",
                 color=C_WHEEL if head else C_INK,
                 weight="bold" if head else "normal")
        y -= 0.075 + 0.045 * (max(b.count("\n"), c.count("\n")))
        if head:
            axR.plot([0.0, 1.0], [y + 0.045, y + 0.045], color="#D8DBE0", lw=1)
    axR.text(0.0, y - 0.02,
             "The last row is the one that changes the robot.\n"
             "A point foot grips, so a legged robot pivots about its foot and\n"
             "geometry fights back. A wheel rolls, so the contact patch slides\n"
             "out from under the leg: the hip becomes an inverted pendulum and\n"
             "needs 11.6 N·m/rad of stiffness before it will stand at all.\n\n"
             "A real wheel-legged robot holds station by BRAKING its wheels.\n"
             "So must a policy trained here.",
             fontsize=9.6, color="#4B5158", va="top", linespacing=1.55)
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUT / "legdiagram.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / 'legdiagram.png'}")


# ── The mechanics plots ──────────────────────────────────────────────────────
def _drive(model, tau_scale=1.0):
    m = mujoco.MjModel.from_xml_path(model)
    for leg in LEGS:
        m.actuator(f"{leg}_wheel").gear[0] = SPEC.gear_wheel * tau_scale
    return m


def mechanics_plots() -> None:
    """The three measurements that decide how this robot must be trained."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), facecolor="white")

    # (a) the wheel angle runs away; its velocity does not -------------------
    ax = axes[0]
    m = mujoco.MjModel.from_xml_path("assets/hound16.xml")
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    wa = m.jnt_qposadr[m.joint("FL_wheel").id]
    wv = m.jnt_dofadr[m.joint("FL_wheel").id]
    acts = [m.actuator(f"{l}_wheel").id for l in LEGS]
    ang, vel, ts = [], [], []
    for k in range(10_000):                        # 50 s = one episode
        c = np.zeros(m.nu)
        c[acts] = 1.0
        d.ctrl[:] = c
        mujoco.mj_step(m, d)
        if k % 20 == 0:
            ang.append(float(d.qpos[wa]))
            vel.append(float(d.qvel[wv]))
            ts.append(k * m.opt.timestep)
    ax.plot(ts, ang, color=C_WHEEL, lw=2, label="wheel ANGLE (rad)")
    ax.set_xlabel("time in one 50 s episode (s)")
    ax.set_ylabel("wheel angle (rad)", color=C_WHEEL)
    ax.tick_params(axis="y", colors=C_WHEEL)
    ax2 = ax.twinx()
    ax2.plot(ts, vel, color=C_KNEE, lw=1.6, alpha=0.85,
             label="wheel VELOCITY (rad/s)")
    ax2.set_ylabel("wheel velocity (rad/s)", color=C_KNEE)
    ax2.tick_params(axis="y", colors=C_KNEE)
    ax2.set_ylim(0, max(vel) * 2.2)
    ax.set_title("(a)  why the wheel angle is not an observation",
                 fontsize=11.5, color=C_INK)
    ax.text(0.03, 0.94,
            f"angle reaches {ang[-1]:,.0f} rad ({ang[-1] / (2 * np.pi):,.0f} turns)\n"
            "and never repeats a value — a de-facto\nepisode clock the critic would learn.\n\n"
            "velocity is bounded and stationary.\nHoundEnv keeps the blue, drops the amber.",
            transform=ax.transAxes, fontsize=8.8, va="top", color="#4B5158",
            linespacing=1.5)
    ax.grid(alpha=0.25)

    # (b) thrust saturates, and the reason is unloading -----------------------
    ax = axes[1]
    T = 0.30
    taus, accs, loads = [], [], []
    weight = SPEC.total_mass * 9.81
    for frac in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        mm = _drive("assets/hound16.xml", frac)
        dd = mujoco.MjData(mm)
        dd.qpos[:] = mm.key_qpos[0]
        mujoco.mj_forward(mm, dd)
        for _ in range(int(T / mm.opt.timestep)):
            c = np.zeros(mm.nu)
            c[[mm.actuator(f"{l}_wheel").id for l in LEGS]] = 1.0
            dd.ctrl[:] = c
            mujoco.mj_step(mm, dd)
        fn = 0.0
        for i in range(dd.ncon):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(mm, dd, i, f6)
            fn += f6[0]
        taus.append(SPEC.gear_wheel * frac)
        accs.append(float(dd.qvel[0]) / T)
        loads.append(100 * fn / weight)
    ideal = [4 * t / (SPEC.wheel_r * SPEC.total_mass) for t in taus]
    ax.plot(taus, ideal, color="#B6BBC2", lw=1.8, ls="--",
            label="ideal  a = 4τ/(rM)")
    ax.axhline(SPEC.wheel_friction[0] * 9.81, color=C_ABDUCT, lw=1.4, ls=":",
               label="friction cone  μg = 8.8")
    ax.plot(taus, accs, "o-", color=C_KNEE, lw=2.4, ms=6, label="MEASURED")
    ax.axvline(SPEC.gear_wheel, color=C_WHEEL, lw=1.6)
    ax.text(SPEC.gear_wheel * 1.10, 30.0, "shipped\ngear 3.0", fontsize=8.6,
            color=C_WHEEL, weight="bold")
    ax.set_xlabel("wheel torque, each of four (N·m)")
    ax.set_ylabel("acceleration from rest (m/s²)")
    ax.set_title("(b)  more motor buys nothing", fontsize=11.5, color=C_INK)
    ax.legend(fontsize=8.4, loc="upper left", bbox_to_anchor=(0.30, 0.99))
    ax.grid(alpha=0.25)
    ax.text(0.30, 0.13,
            "MEASURED saturates at ~2 m/s² with the friction cone only 5% used:\n"
            "the machine pitches back and UNLOADS its wheels. Geometry binds\n"
            "long before grip does.",
            transform=ax.transAxes, fontsize=8.8, va="bottom", color="#4B5158",
            linespacing=1.6)

    # (c) the hip stiffness the stance actually needs -------------------------
    ax = axes[2]
    crit = SPEC.critical_stiffness()["hip"]
    ks = [4, 6.2, 9, 11.6, 14, 18.6, 24, 32]
    tau_hip = SPEC.static_torques()["hip"]
    zs = []
    for k in ks:
        mm = mujoco.MjModel.from_xml_path("assets/hound16.xml")
        for leg in LEGS:
            j = mm.joint(f"{leg}_hip")
            mm.jnt_stiffness[j.id] = k
            mm.qpos_spring[mm.jnt_qposadr[j.id]] = SPEC.stance_hip - tau_hip / k
        dd = mujoco.MjData(mm)
        dd.qpos[:] = mm.key_qpos[0]
        dd.qpos[2] += 0.005
        mujoco.mj_forward(mm, dd)
        for _ in range(2500):
            dd.ctrl[:] = 0
            mujoco.mj_step(mm, dd)
        zs.append(float(dd.xpos[mm.body("trunk").id][2]))
    ax.axhline(SPEC.stand_z, color="#B6BBC2", lw=1.6, ls="--",
               label=f"drawn stance {SPEC.stand_z:.3f} m")
    ax.axvspan(0, crit, color=C_ABDUCT, alpha=0.10)
    ax.axvline(crit, color=C_ABDUCT, lw=1.8)
    ax.text(crit * 0.96, 0.20, "unstable\n← spring weaker than\n   the inverted\n   pendulum",
            fontsize=8.6, color=C_ABDUCT, ha="right", va="bottom")
    ax.axvline(SPEC.stiffness()["hip"], color=C_HIP, lw=1.8)
    ax.text(SPEC.stiffness()["hip"] * 1.05, 0.20,
            f"shipped\n{SPEC.stiffness()['hip']:.1f} = 1.6×", fontsize=8.6,
            color=C_HIP, va="bottom")
    ax.plot(ks, zs, "o-", color=C_INK, lw=2.4, ms=6,
            label="settled trunk height")
    ax.set_xlabel("hip spring stiffness (N·m/rad)")
    ax.set_ylabel("trunk height after 12 s, no torque (m)")
    ax.set_title("(c)  a wheeled leg needs a stiffness floor",
                 fontsize=11.5, color=C_INK)
    ax.set_ylim(0.15, 0.40)
    ax.legend(fontsize=8.4, loc="lower right")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "mechanics.png", dpi=125, facecolor="white")
    plt.close(fig)
    print(f"  wrote {OUT / 'mechanics.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["preview", "views", "leg", "mechanics"],
                    help="render just one output")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "preview": render_previews,
        "views": render_contact_sheet,
        "leg": leg_diagram,
        "mechanics": mechanics_plots,
    }
    for name, fn in jobs.items():
        if args.only in (None, name):
            print(f"{name}:")
            fn()


if __name__ == "__main__":
    main()
