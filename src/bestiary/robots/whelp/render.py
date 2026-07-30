"""Draw WHELP-16 from its own kinematics, so the geometry can be looked at.

    python -m bestiary.robots.whelp.render          # assets/whelp/whelp16.png
    python -m bestiary.robots.whelp.render --show   # open a window instead

WHAT THIS IS AND IS NOT
-----------------------
It is a SKELETON drawing: joint origins from geometry.link_frames(), links as
lines between them, wheels as circles at their loaded radius, the trunk as its
collision box. Every number comes from the same functions the URDF is generated
from, so if this picture is wrong the simulation is wrong in the same way.

It is NOT the printed shape. The parts have channels, fillets, servo pockets and
bosses that only OpenSCAD can show:

    sudo apt install openscad
    openscad src/bestiary/robots/whelp/scad/build.scad

Use this to check that the robot STANDS where it should -- axle under hip, all
four contacts on the ground, the legs clear of the trunk -- and use OpenSCAD to
check that the parts fit together.

WHY A SKELETON IS WORTH DRAWING AT ALL
--------------------------------------
Because the assertions in check.py are true statements about numbers, and a
person cannot see a number. "Every wheel axle sits directly under its own hip
pivot, worst offset 1.4e-14 mm" is exactly the sort of thing that is easy to
assert and easy to be wrong about at the level of what it MEANS. One picture
settles whether the machine that satisfies those assertions is the machine
anybody intended.
"""
from __future__ import annotations

import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from bestiary import paths
from bestiary.robots.whelp import geometry as geo
from bestiary.robots.whelp import torque
from bestiary.robots.whelp.massmodel import link_bodies
from bestiary.robots.whelp.spec import SPEC

OUT = paths.ASSETS / "whelp" / "whelp16.png"

#: Front legs solid, rear legs faded, so an overlapping side view is readable.
_ALPHA = {"FL": 1.0, "FR": 0.35, "RL": 1.0, "RR": 0.35}
_COLOR = {"FL": "#1f4e79", "FR": "#1f4e79", "RL": "#c05621", "RR": "#c05621"}


def _mm(v):
    return [x * 1000.0 for x in v]


def _leg_points(pos, leg, spec):
    """Joint origins down one leg, plus the contact patch, in millimetres."""
    chain = [pos[f"{leg}_abduct"], pos[f"{leg}_hip"], pos[f"{leg}_knee"],
             pos[f"{leg}_wheel"]]
    return [_mm(p) for p in chain]


def _draw(ax, pos, spec, ax_i, ax_j, title, xlabel, ylabel):
    """One orthographic projection. ax_i/ax_j pick which world axes to plot."""
    stand = geo.stand_height_mm(spec)
    r_free = spec.wheel_radius_mm
    r_load = r_free - spec.tire_static_sag_mm

    # Ground. The wheels rest on it by construction: stand_height is the axle
    # drop plus the LOADED radius, which is why the contacts land at zero and not
    # a millimetre or two under.
    ax.axhline(0, color="#444", lw=1.4, zorder=1)
    ax.axhspan(-14, 0, color="#eee", zorder=0)

    # Trunk, as the collision box the URDF actually emits.
    half = {0: spec.trunk_len_mm / 2, 1: spec.trunk_width_mm / 2,
            2: spec.trunk_height_mm / 2}
    ax.add_patch(Rectangle(
        (-half[ax_i], stand - half[ax_j] if ax_j == 2 else -half[ax_j]),
        2 * half[ax_i], 2 * half[ax_j],
        fill=True, facecolor="#dbe7f3", edgecolor="#1f4e79", lw=1.6, zorder=2))

    for leg in geo.LEGS:
        pts = _leg_points(pos, leg, spec)
        xs = [p[ax_i] for p in pts]
        ys = [(p[ax_j] + stand) if ax_j == 2 else p[ax_j] for p in pts]
        a, c = _ALPHA[leg], _COLOR[leg]
        ax.plot(xs, ys, "-", color=c, lw=3.2, alpha=a, solid_capstyle="round",
                zorder=3)
        ax.plot(xs, ys, "o", color="white", mec=c, mew=1.8, ms=6, alpha=a, zorder=4)

        wx, wy = xs[-1], ys[-1]
        if ax_j == 2:
            # Wheel, in a view that shows its diameter. Free radius dashed and
            # loaded radius solid, because the difference is the tire's static
            # squash and it is the reason the robot spawns on the floor rather
            # than in it.
            ax.add_patch(Circle((wx, wy), r_free, fill=False, ec=c, lw=1.0,
                                ls=":", alpha=a, zorder=3))
            ax.add_patch(Circle((wx, wy), r_load, fill=True, fc="#2b2b2b",
                                ec=c, lw=1.4, alpha=a * 0.85, zorder=3))
        else:
            ax.add_patch(Rectangle(
                (wx - spec.wheel_width_mm / 2, wy - r_load),
                spec.wheel_width_mm, 2 * r_load, fill=True, fc="#2b2b2b",
                ec=c, lw=1.2, alpha=a * 0.85, zorder=3))

    ax.set_title(title, fontsize=11, loc="left", color="#222")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4, alpha=0.3)
    ax.tick_params(labelsize=7)


def render(show: bool = False):
    spec = SPEC
    chain = geo.build_chain(spec)
    joint_pos, frames = geo.link_frames(chain)
    pos = dict(joint_pos)
    for name, (_r, p) in frames.items():
        pos.setdefault(name, p)

    bodies = link_bodies(spec)
    a = torque.analyse(spec)
    stand = geo.stand_height_mm(spec)
    knee = geo.solve_stance_knee(spec)

    fig = plt.figure(figsize=(14.5, 5.6), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.0, 1.2, 1.35], wspace=0.32)

    _draw(fig.add_subplot(gs[0, 0]), pos, spec, 0, 2,
          "SIDE  (+X forward, +Z up)   front legs solid, rear faded",
          "x  mm", "z  mm")
    _draw(fig.add_subplot(gs[0, 1]), pos, spec, 1, 2,
          "FRONT  (+Y left)", "y  mm", "z  mm")

    ax = fig.add_subplot(gs[0, 2])
    for leg in geo.LEGS:
        pts = _leg_points(pos, leg, spec)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                color=_COLOR[leg], lw=2.4, ms=5, mfc="white", alpha=_ALPHA[leg])
        c = _mm(geo.contact_points(chain, spec)[leg])
        ax.add_patch(Circle((c[0], c[1]), spec.wheel_width_mm / 2, fc="#2b2b2b",
                            ec=_COLOR[leg], alpha=0.8))
    ax.add_patch(Rectangle((-spec.trunk_len_mm / 2, -spec.trunk_width_mm / 2),
                           spec.trunk_len_mm, spec.trunk_width_mm, fill=True,
                           fc="#dbe7f3", ec="#1f4e79", lw=1.6))
    ax.set_title("TOP", fontsize=11, loc="left")
    ax.set_xlabel("x  mm", fontsize=8)
    ax.set_ylabel("y  mm", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.4, alpha=0.3)
    ax.tick_params(labelsize=7)

    # The numbers, next to the picture, so the drawing is checkable rather than
    # merely suggestive.
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    lg = a["landing"]
    wh = a["wheel"]
    rows = [
        ("WHELP-16", ""),
        ("mass", f"{sum(b.mass_kg for b in bodies.values()):.3f} kg"),
        ("servo", spec.servo_variant),
        ("", ""),
        ("stand height", f"{stand:.0f} mm"),
        ("wheelbase x track", f"{2 * spec.abduct_x_mm:.0f} x "
                              f"{2 * spec.wheel_centre_y_mm:.0f} mm"),
        ("stance hip", f"{spec.stance_hip_rad:+.3f} rad"),
        ("stance knee", f"{knee:+.3f} rad  SOLVED"),
        ("knee lever", f"{geo.knee_lever_mm(spec):.1f} mm"),
        ("", ""),
        ("worst sustained", f"{max(c['worst']['knee']['nm'] for c in a['cases'] if c['sustained']):.3f} N.m"),
        ("   vs rated", f"{spec.leg_servo_rated_nm:.2f} N.m"),
        ("hip holds", f"{a['cases'][0]['worst']['hip']['nm']:.3f} N.m"),
        ("", ""),
        ("top speed", f"{wh['top_speed_m_s']:.2f} m/s"),
        ("max accel", f"{wh['accel_max_m_s2']:.2f} m/s^2"),
        ("effective mass", f"{wh['effective_mass_kg']:.0f} kg  (gearbox)"),
        ("drop envelope", f"{lg['max_drop_3leg_with_margin_mm']:.0f} mm at 2x"),
    ]
    y = 0.98
    for k, v in rows:
        if k == "WHELP-16":
            ax.text(0.0, y, k, fontsize=13, weight="bold", va="top")
        elif not k and not v:
            pass
        else:
            ax.text(0.0, y, k, fontsize=8.5, va="top", color="#555")
            ax.text(0.62, y, v, fontsize=8.5, va="top", weight="medium")
        y -= 0.055

    fig.suptitle(
        "WHELP-16 skeleton, drawn from geometry.link_frames() at the solved stance. "
        "Dotted circle = free tire radius, solid = loaded. "
        "This is not the printed shape -- run OpenSCAD on scad/build.scad for that.",
        fontsize=8.5, y=0.015, color="#555")

    if show:
        plt.show()
        return None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT


def main(argv: list[str]) -> int:
    out = render(show="--show" in argv)
    if out is not None:
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
