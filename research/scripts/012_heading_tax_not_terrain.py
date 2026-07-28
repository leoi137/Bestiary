"""Every derived number quoted in `research/learnings/012`, recomputed.

    venv/bin/python research/scripts/012_heading_tax_not_terrain.py

WHY THIS EXISTS

`research/scripts/heading_ceiling.py` ran the sweep and wrote
`research/measurements/heading_ceiling_s0.json`. That file holds per-cell means.
Every *further* figure the learning quotes -- bands, ratios, deficits,
monotonicity counts, correlations, crash tallies, the Cauchy inversion -- is
arithmetic on top of it, and the repo's number rule says arithmetic in prose is
fluent, checkable and unchecked. So it lives here instead.

Nothing is measured here. This script only reads and reduces. If a claim in the
learning is not printed below, the learning is wrong to make it.
"""
from __future__ import annotations

import json
import math

# paths.py is the only place that builds paths (see CLAUDE.md invariants).
from bestiary import paths

SRC = paths.RESEARCH / "measurements" / "heading_ceiling_s0.json"

# The three straight-drive commands share w_cmd = 0, so their heading factor is
# pure disturbance rejection. The turning cell asks the body to yaw ON PURPOSE
# and is therefore a different question; it is reported separately everywhere.
STRAIGHT = ("drive_slow", "drive_mid", "drive_fast")
SPEED_OF = {"drive_slow": 0.30, "drive_mid": 0.55, "drive_fast": 0.80}


def short(label: str) -> str:
    """'drive_mid     (vx=0.55, w=0)' -> 'drive_mid'."""
    return label.split()[0]


def yaw_err(phi: float, sigma: float) -> float:
    """Invert the Cauchy kernel: Phi = 1/(1+u^2), u = |err|/sigma."""
    return sigma * math.sqrt(1.0 / phi - 1.0)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy)


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    d = json.loads(SRC.read_text())
    arms = d["arms"]
    sigma_w = d["sigma_w"]
    n_per_cell = d["episodes_per_cell"]

    # Fail loudly if the measurement is not the shape this reduction assumes.
    if len(arms) != 16:
        raise SystemExit(f"expected 16 cells, got {len(arms)}")
    for a in arms:
        for k in ("policy", "zero_action"):
            if a[k]["n"] != n_per_cell:
                raise SystemExit(
                    f"cell {a['alpha']}/{a['command']} arm {k} has n={a[k]['n']}, "
                    f"expected {n_per_cell}"
                )

    alphas = sorted({a["alpha"] for a in arms})
    st = [a for a in arms if short(a["command"]) in STRAIGHT]
    turn = [a for a in arms if short(a["command"]) == "turn"]

    print(f"source     {SRC.relative_to(paths.REPO_ROOT)}")
    print(f"run        {d['run']} / {d['checkpoint']}")
    print(f"cells      {len(arms)}  ({len(alphas)} alpha x 4 commands), "
          f"n={n_per_cell} per cell per arm, seeds "
          f"{d['seed0']}..{d['seed0'] + n_per_cell - 1}")
    print(f"sigma_w    {sigma_w} rad/s = {math.degrees(sigma_w):.2f} deg/s")
    print(f"sigma_v    {d['sigma_v']} m/s")
    print(f"terrain    committed elevation span {d['original_elevation_m']:.3f} m; "
          f"alphas {alphas}")
    print(f"spawn      z={d['spawn_z_m']:.4f} m over ground "
          f"{d['ground_at_spawn_m']:.5f} m -> clearance "
          f"{d['clearance_at_spawn_m']:.4f} m, held constant across arms")

    # ---------------------------------------------------------------- 1
    rule("1. ZERO ACTION: what the GROUND alone does to the heading factor")
    # Within one alpha the three straight cells are the same rollouts (w_cmd=0,
    # no policy), so they are identical by construction; take one per alpha.
    z_by_alpha = {}
    for al in alphas:
        cells = [a for a in st if a["alpha"] == al]
        vals = {c["zero_action"]["phi_w"] for c in cells}
        if max(vals) - min(vals) > 1e-12:
            raise SystemExit(f"zero-action straight cells disagree at alpha={al}: {vals}")
        c = cells[0]
        z_by_alpha[al] = (c["zero_action"]["phi_w"], c["zero_action"]["phi_w_sd"],
                          c["elevation_m"])
    for al in alphas:
        phi, sd, elev = z_by_alpha[al]
        print(f"  alpha={al:<5} elev={elev:7.4f} m   Phi_w={phi:.4f} "
              f"(sd={sd:.4f}, sem={sd / math.sqrt(n_per_cell):.4f})   "
              f"yaw err={yaw_err(phi, sigma_w):.5f} rad/s "
              f"= {math.degrees(yaw_err(phi, sigma_w)):.2f} deg/s")
    zv = [z_by_alpha[al][0] for al in alphas]
    spread = max(zv) - min(zv)
    mean_sem = sum(z_by_alpha[al][1] for al in alphas) / len(alphas) / math.sqrt(n_per_cell)
    print(f"  band across alpha:  {min(zv):.4f} .. {max(zv):.4f}   "
          f"spread={spread:.5f} = {100 * spread / (sum(zv) / len(zv)):.2f}% of the mean")
    print(f"  mean within-cell SEM = {mean_sem:.5f}  ->  spread/SEM = "
          f"{spread / mean_sem:.2f}   (monotone in alpha? "
          f"{'yes' if zv == sorted(zv) or zv == sorted(zv, reverse=True) else 'NO'})")
    print("  full elevation span traversed: "
          f"{z_by_alpha[alphas[0]][2]:.3f} m -> {z_by_alpha[alphas[-1]][2]:.3f} m")

    # ---------------------------------------------------------------- 2
    rule("2. POLICY: the heading factor, and where its maximum is")
    best = max(arms, key=lambda a: a["policy"]["phi_w"])
    print(f"  global max over all {len(arms)} cells: Phi_w={best['policy']['phi_w']:.4f} "
          f"at alpha={best['alpha']}, {short(best['command'])} "
          f"(crashes {best['policy']['crashes']}/{best['policy']['n']}, "
          f"mean steps {best['policy']['steps_mean']:.1f})")
    for al in alphas:
        row = [a for a in st if a["alpha"] == al]
        row.sort(key=lambda a: SPEED_OF[short(a["command"])])
        cells = "  ".join(
            f"{short(a['command']).split('_')[1]:>4}={a['policy']['phi_w']:.4f}"
            f"({a['policy']['crashes']:>2}/{a['policy']['n']})" for a in row)
        print(f"  alpha={al:<5} {cells}   zero={z_by_alpha[al][0]:.4f}")

    # ---------------------------------------------------------------- 3
    rule("3. THE LOAD-BEARING CELLS: full 1000-step, zero crashes, no length bias")
    unc = [a for a in arms if a["policy"]["crashes"] == 0]
    unc_st = [a for a in unc if short(a["command"]) in STRAIGHT]
    print(f"  {len(unc)} of {len(arms)} cells have 0/{n_per_cell} policy crashes "
          f"({len(unc_st)} of them straight-drive)")
    for a in sorted(unc_st, key=lambda a: (a["alpha"], SPEED_OF[short(a["command"])])):
        p, z = a["policy"], a["zero_action"]
        if p["steps_mean"] != 1000.0 or p["steps_min"] != 1000:
            raise SystemExit(f"uncrashed cell not full-length: {a['alpha']} {a['command']}")
        print(f"  alpha={a['alpha']:<5} {short(a['command']):<11} "
              f"Phi_w={p['phi_w']:.4f} vs standing {z['phi_w']:.4f}   "
              f"ratio={p['phi_w'] / z['phi_w']:.4f}   "
              f"deficit={100 * (1 - p['phi_w'] / z['phi_w']):5.1f}%   "
              f"yaw err {yaw_err(p['phi_w'], sigma_w):.4f} vs "
              f"{yaw_err(z['phi_w'], sigma_w):.4f} rad/s "
              f"({math.degrees(yaw_err(p['phi_w'], sigma_w)):5.2f} vs "
              f"{math.degrees(yaw_err(z['phi_w'], sigma_w)):.2f} deg/s)")
    uv = [a["policy"]["phi_w"] for a in unc_st]
    ur = [a["policy"]["phi_w"] / a["zero_action"]["phi_w"] for a in unc_st]
    print(f"  BAND on uncrashed straight cells: Phi_w {min(uv):.4f} .. {max(uv):.4f}")
    print(f"  ratio to standing:                {min(ur):.4f} .. {max(ur):.4f}   "
          f"(deficit {100 * (1 - max(ur)):.1f}% .. {100 * (1 - min(ur)):.1f}%)")
    print(f"  standing's own variation over the SAME alpha range: "
          f"{100 * spread / (sum(zv) / len(zv)):.2f}%")
    z_pct = 100 * spread / (sum(zv) / len(zv))
    print(f"  -> driving costs {100 * (1 - max(ur)):.1f}-{100 * (1 - min(ur)):.1f}% "
          f"of the heading factor; the terrain costs standing {z_pct:.2f}%.")
    print(f"  ratio of the two effects: {100 * (1 - max(ur)) / z_pct:.0f}x to "
          f"{100 * (1 - min(ur)) / z_pct:.0f}x")

    # Does the POLICY's deficit at least track roughness, at a fixed command?
    # Asked separately because "the terrain still adds something" is the natural
    # fallback claim, and it needs its own evidence rather than an assumption.
    print("  and at a FIXED command, across the alphas where the cell is crash-free:")
    for cmdname in STRAIGHT:
        row = sorted((a for a in unc_st if short(a["command"]) == cmdname),
                     key=lambda a: a["alpha"])
        if len(row) < 2:
            print(f"    {cmdname:<11} only {len(row)} crash-free cell(s); no trend")
            continue
        defs = [100 * (1 - a["policy"]["phi_w"] / a["zero_action"]["phi_w"])
                for a in row]
        mono = defs == sorted(defs) or defs == sorted(defs, reverse=True)
        note = "" if len(row) > 2 else "  (2 points: monotone is vacuous)"
        print(f"    {cmdname:<11} "
              + "  ".join(f"a={a['alpha']}:{dv:.1f}%" for a, dv in zip(row, defs))
              + f"   rougher-minus-smoother = {defs[-1] - defs[0]:+.1f} pp"
              + f"   monotone? {'yes' if mono else 'NO'}{note}")
    print("    -> the sign of that last column is not consistent across commands,")
    print("       so even the POLICY's deficit does not track roughness cleanly.")

    # ---------------------------------------------------------------- 4
    rule("4. CRASHES: flat ground is where this policy dies")
    for al in alphas:
        cells = [a for a in arms if a["alpha"] == al]
        pc = sum(c["policy"]["crashes"] for c in cells)
        zc = sum(c["zero_action"]["crashes"] for c in cells)
        n = sum(c["policy"]["n"] for c in cells)
        smean = sum(c["policy"]["steps_mean"] for c in cells) / len(cells)
        print(f"  alpha={al:<5} elev={z_by_alpha[al][2]:7.4f} m   "
              f"policy {pc:>3}/{n} crashed   zero action {zc:>3}/{n}   "
              f"policy mean steps {smean:6.1f}")
    tot_p = sum(a["policy"]["crashes"] for a in arms)
    tot_n = sum(a["policy"]["n"] for a in arms)
    tot_z = sum(a["zero_action"]["crashes"] for a in arms)
    print(f"  TOTAL  policy {tot_p}/{tot_n} = {100 * tot_p / tot_n:.1f}%   "
          f"zero action {tot_z}/{tot_n}")
    flat = [a for a in arms if a["alpha"] == 0.0]
    print("  flat (alpha=0) policy mean steps by cell: "
          + ", ".join(f"{short(a['command'])}={a['policy']['steps_mean']:.1f}" for a in flat))
    print(f"  zero-action mean steps everywhere: "
          f"{min(a['zero_action']['steps_mean'] for a in arms):.1f} .. "
          f"{max(a['zero_action']['steps_mean'] for a in arms):.1f}")

    # ---------------------------------------------------------------- 5
    rule("5. MONOTONICITY IN COMMANDED SPEED (the brief's claim, tested)")
    pairs = [("slow>mid", "drive_slow", "drive_mid"),
             ("mid>fast", "drive_mid", "drive_fast"),
             ("slow>fast", "drive_slow", "drive_fast")]
    holds = {k: 0 for k, _, _ in pairs}
    for al in alphas:
        row = {short(a["command"]): a["policy"]["phi_w"] for a in st if a["alpha"] == al}
        marks = []
        for name, hi, lo in pairs:
            ok = row[hi] > row[lo]
            holds[name] += ok
            marks.append(f"{name}:{'Y' if ok else 'N'}")
        print(f"  alpha={al:<5} "
              f"0.30={row['drive_slow']:.4f} 0.55={row['drive_mid']:.4f} "
              f"0.80={row['drive_fast']:.4f}   " + "  ".join(marks))
    for name, _, _ in pairs:
        print(f"  {name:<10} holds at {holds[name]}/{len(alphas)} alphas")
    strict_at, clean_at = [], []
    for al in alphas:
        r = {short(a["command"]): a["policy"]["phi_w"] for a in st if a["alpha"] == al}
        c = {short(a["command"]): a["policy"]["crashes"] for a in st if a["alpha"] == al}
        if r["drive_slow"] > r["drive_mid"] > r["drive_fast"]:
            strict_at.append(al)
            if sum(c.values()) == 0:
                clean_at.append(al)
    print(f"  strictly monotone decreasing in commanded speed: "
          f"{len(strict_at)}/{len(alphas)} alphas, at alpha={strict_at}")
    print(f"  ... of which crash-free (so not length-biased): "
          f"{len(clean_at)}/{len(alphas)}, at alpha={clean_at}")
    print("  -> the brief's 'monotone at every roughness' is NOT supported; what")
    print("     survives is slow>mid and slow>fast at 4/4, i.e. Phi_w is lower at")
    print("     0.55 and 0.80 m/s than at 0.30 m/s on every ground tested.")

    # ---------------------------------------------------------------- 6
    rule("6. THE 011 CANCELLATION, cell by cell (moving vs not moving)")
    both = sum(1 for a in st
               if a["policy"]["phi_v"] > a["zero_action"]["phi_v"]
               and a["policy"]["phi_w"] < a["zero_action"]["phi_w"])
    print(f"  cells where policy Phi_v > standing AND policy Phi_w < standing: "
          f"{both}/{len(st)} straight-drive cells")
    for a in sorted(st, key=lambda a: (a["alpha"], SPEED_OF[short(a["command"])])):
        p, z = a["policy"], a["zero_action"]
        print(f"  alpha={a['alpha']:<5} {short(a['command']):<11} "
              f"Phi_v {z['phi_v']:.4f}->{p['phi_v']:.4f} (x{p['phi_v'] / z['phi_v']:6.2f})   "
              f"Phi_w {z['phi_w']:.4f}->{p['phi_w']:.4f} (x{p['phi_w'] / z['phi_w']:.2f})   "
              f"track {z['track']:.4f}->{p['track']:.4f} "
              f"(x{p['track'] / z['track']:.2f})"
              f"{'' if p['crashes'] == 0 else '  [length-biased]'}")

    # ---------------------------------------------------------------- 7
    rule("7. IS IT A TRADE? Phi_v against Phi_w within the policy, across cells")
    pv = [a["policy"]["phi_v"] for a in st]
    pw = [a["policy"]["phi_w"] for a in st]
    print(f"  all {len(st)} straight cells:      r(Phi_v, Phi_w) = {pearson(pv, pw):+.4f}")
    uv2 = [a["policy"]["phi_v"] for a in unc_st]
    uw2 = [a["policy"]["phi_w"] for a in unc_st]
    print(f"  {len(unc_st)} uncrashed straight cells: r(Phi_v, Phi_w) = "
          f"{pearson(uv2, uw2):+.4f}")
    print("  A trade-off across commands would give r < 0. Both factors fall")
    print("  TOGETHER as the command gets harder, so the trade is against")
    print("  STANDING STILL (section 6), not between the two factors.")

    # ---------------------------------------------------------------- 8
    rule("8. THE TURN CELL: the only command where doing nothing cannot win")
    for a in sorted(turn, key=lambda a: a["alpha"]):
        p, z = a["policy"], a["zero_action"]
        print(f"  alpha={a['alpha']:<5} policy Phi_w={p['phi_w']:.4f} vs "
              f"standing {z['phi_w']:.4f}   ratio={p['phi_w'] / z['phi_w']:.2f}   "
              f"policy crashes {p['crashes']}/{p['n']}, steps {p['steps_mean']:.1f}")
    tp = [a["policy"]["phi_w"] for a in turn]
    tz = [a["zero_action"]["phi_w"] for a in turn]
    print(f"  policy band {min(tp):.4f} .. {max(tp):.4f}   "
          f"standing band {min(tz):.4f} .. {max(tz):.4f}")
    print(f"  policy beats standing on Phi_w at "
          f"{sum(1 for a in turn if a['policy']['phi_w'] > a['zero_action']['phi_w'])}"
          f"/{len(turn)} alphas")

    # ---------------------------------------------------------------- 9
    rule("9. WHAT THE HEADING FACTOR COSTS, in tracking reward per step")
    print("  Caveat computed, not assumed: track is mean_t[Phi_v*Phi_w], which is")
    print("  NOT mean[Phi_v]*mean[Phi_w]. The gap below is the covariance term,")
    print("  so the counterfactual is an estimate to ~that accuracy, not exact.")
    for a in sorted(unc_st, key=lambda a: (a["alpha"], SPEED_OF[short(a["command"])])):
        p, z = a["policy"], a["zero_action"]
        prod = p["phi_v"] * p["phi_w"]
        cf = p["phi_v"] * z["phi_w"]
        print(f"  alpha={a['alpha']:<5} {short(a['command']):<11} "
              f"track={p['track']:.4f}  mean(Phi_v)*mean(Phi_w)={prod:.4f} "
              f"(cov gap {p['track'] - prod:+.4f})   "
              f"if heading were held at standing: {cf:.4f}  "
              f"-> heading removes {cf - prod:.4f}/step = "
              f"{100 * (1 - prod / cf):.1f}%")

    # The worked example the learning's math section walks through, printed in
    # full so no intermediate in that section is hand-arithmetic.
    w = next(a for a in unc_st
             if a["alpha"] == 1.0 and short(a["command"]) == "drive_slow")
    pw_, zw_ = w["policy"]["phi_w"], w["zero_action"]["phi_w"]
    ep, ez = yaw_err(pw_, sigma_w), yaw_err(zw_, sigma_w)
    print("  WORKED CELL alpha=1.00 drive_slow (committed terrain, 0/10 crashed):")
    print(f"    u_w standing = sqrt(1/{zw_:.4f} - 1) = {math.sqrt(1 / zw_ - 1):.4f}"
          f"  -> |w| = {ez:.5f} rad/s = {math.degrees(ez):.2f} deg/s")
    print(f"    u_w policy   = sqrt(1/{pw_:.4f} - 1) = {math.sqrt(1 / pw_ - 1):.4f}"
          f"  -> |w| = {ep:.5f} rad/s = {math.degrees(ep):.2f} deg/s")
    print(f"    the policy yaws {ep / ez:.2f}x as fast as a machine lying still")

    # --------------------------------------------------------------- 10
    rule("10. THE FALSIFIER LINE")
    thresh = 0.95
    over = [a for a in arms if a["policy"]["phi_w"] >= thresh]
    print(f"  cells where the policy reaches Phi_w >= {thresh}: {len(over)}/{len(arms)}")
    print(f"  the closest it gets, anywhere: {best['policy']['phi_w']:.4f} "
          f"(alpha={best['alpha']}, {short(best['command'])}, "
          f"{best['policy']['crashes']}/{best['policy']['n']} crashed)")
    print(f"  closest among full-length cells: "
          f"{max(a['policy']['phi_w'] for a in unc):.4f}")

    # --------------------------------------------------------------- 11
    rule("11. RECONCILING learnings/011's FALSIFIER 2 (the 0.513 threshold)")
    # 011 wrote: "If the same policy on flat ground holds Phi_w near zero
    # action's 0.513 while driving, then heading loss is a terrain-contact
    # problem." That 0.513 is NOT a straight-drive number -- it is the mean over
    # the six-cell drive grid, half of which command a yaw rate. Quoting it as a
    # threshold against this sweep's straight-drive cells would compare two
    # different command mixtures, so it is reconstructed here from its own file.
    base = json.loads((paths.RESEARCH / "measurements" /
                       "tracking_baseline_zero_action.json").read_text())
    grid = base["grid"]
    drive = {k: v for k, v in grid.items() if v["command"] != [0.0, 0.0, 0.0]}
    dmean = sum(v["mean_phi_w"] for v in drive.values()) / len(drive)
    straight = {k: v for k, v in drive.items() if v["command"][2] == 0.0}
    yawed = {k: v for k, v in drive.items() if v["command"][2] != 0.0}
    smean = sum(v["mean_phi_w"] for v in straight.values()) / len(straight)
    ymean = sum(v["mean_phi_w"] for v in yawed.values()) / len(yawed)
    print(f"  011's control figure, recomputed: mean Phi_w over the "
          f"{len(drive)} drive-grid cells = {dmean:.4f}")
    print(f"    of which {len(straight)} straight (w_cmd=0): {smean:.4f}")
    print(f"    of which {len(yawed)} yaw-commanded:         {ymean:.4f}")
    print(f"  This sweep's matched control (straight, committed terrain, "
          f"alpha=1.00): {z_by_alpha[1.0][0]:.4f}")
    print(f"  agreement with the independent baseline file: "
          f"{abs(smean - z_by_alpha[1.0][0]):.4f} absolute "
          f"({100 * abs(smean - z_by_alpha[1.0][0]) / smean:.2f}%)")
    print("  -> 0.513 is a MIXTURE average and is not the right threshold for a")
    print("     straight-drive cell. The matched comparison is 0.969, and the")
    print("     policy is below it at 12/12 straight cells (section 6).")

    # --------------------------------------------------------------- 12
    rule("12. THE PRE-POLICY REFERENCE: an OPEN-LOOP driver, measured in 2026-07")
    # `tracking_noise.json` predates the tracking env entirely. Its two arms are
    # the same robot under a constant wheel command: 0.0 (lying still) and 0.3
    # (rolling forward, no steering, no policy). That makes it an independent
    # measurement of the SAME effect this learning is about, taken before there
    # was a policy to blame.
    #
    # CAVEAT, stated because it is not a nitpick: these arms report an rms yaw
    # rate, so Phi(rms) below is NOT the E[Phi] the sweep reports. Phi is neither
    # convex nor concave over the relevant range, so the two can differ in either
    # direction. This is the repo's existing convention (guards/tracking_frame.py
    # assertion 5 uses it) and it is quoted here as an order-of-magnitude
    # reference point, never as a cell of the sweep's table.
    noise = json.loads((paths.RESEARCH / "measurements" /
                        "tracking_noise.json").read_text())
    for name, a in noise["arms"].items():
        rms = a["yaw"]["rms"]
        phi = 1.0 / (1.0 + (rms / sigma_w) ** 2)
        print(f"  {name:<10} env={a['env_id']:<20} wheel cmd={a['wheel_command']}"
              f"  achieved vx={a['linear']['mean_vx']:+.5f} m/s")
        print(f"             yaw rms={rms:.5f} rad/s "
              f"({math.degrees(rms):.2f} deg/s) -> Phi(rms)={phi:.4f}")
    still = noise["arms"]["wheel_0"]["yaw"]["rms"]
    roll = noise["arms"]["wheel_0.3"]["yaw"]["rms"]
    print(f"  open-loop rolling yaws {roll / still:.2f}x as fast as lying still, "
          f"with NO policy involved at all")
    trained = next(a for a in unc_st
                   if a["alpha"] == 1.0 and short(a["command"]) == "drive_slow")
    tphi = trained["policy"]["phi_w"]
    lo = 1.0 / (1.0 + (roll / sigma_w) ** 2)
    hi = z_by_alpha[1.0][0]
    print(f"  the trained policy at alpha=1.00, 0.30 m/s sits at Phi_w={tphi:.4f}, "
          f"between")
    print(f"  open-loop rolling ({lo:.4f}) and lying still ({hi:.4f}) -- it recovers "
          f"{100 * (tphi - lo) / (hi - lo):.0f}% of what driving costs, and pays the rest.")


if __name__ == "__main__":
    main()
