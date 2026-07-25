"""Assert the mechanics of HOUND-16, and demonstrate them.

    ./venv/bin/python check_hound.py
    ./venv/bin/python check_hound.py -v      # also print the demo tables

Two jobs, and they are the same job. Every claim make_hound.py's docstring
makes about this robot — the wheel is traction-limited, the stance is a real
equilibrium, the wheel angle is unbounded and therefore excluded from the
observation, being alive pays more than dying — is checked here against the
compiled model rather than left as prose that used to be true.

The wheeled morphology is what makes this worth its own file. The spider's
check (check_shell_physics.py) has one thing to prove: decoration does not
touch the dynamics. Here the fourth joint per leg changes the observation
design, the actuator sizing, the contact model and the reset, and each of
those is a place where an edit to Spec can quietly invalidate the env.

Sections:
    1  STRUCTURE     DoF, actuator order, mass budget, observation width
    2  STANCE        the stance is an equilibrium and the geometry is right
    3  WHEEL         unbounded angle, excluded from obs, traction limit
    4  REWARD        alive is net-positive against a saturated policy
    5  TERRAIN       resets survive on the heightfield
    6  REGRESSION    the shared terrain helper did not change Spyder

Re-run this after editing make_hound.py's Spec, envs/hound_env.py, or
envs/terrain.py.
"""
from __future__ import annotations

import hashlib
import sys

import mujoco
import numpy as np

from make_hound import SPEC

MODELS = ["assets/hound16.xml", "assets/hound16_desert.xml"]
LEGS = ("FL", "FR", "RL", "RR")
VERBOSE = "-v" in sys.argv

_failures: list[str] = []
_checks = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _checks
    _checks += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def stance_qpos(model) -> np.ndarray:
    """The authored stance, read from the model's keyframe."""
    return model.key_qpos[0].copy()


def settle(model, qpos=None, steps=2000, ctrl=None, clearance=0.0):
    """Run the passive (or driven) model and return the final MjData."""
    data = mujoco.MjData(model)
    data.qpos[:] = stance_qpos(model) if qpos is None else qpos
    data.qpos[2] += clearance
    mujoco.mj_forward(model, data)
    for k in range(steps):
        data.ctrl[:] = 0 if ctrl is None else ctrl(k)
        mujoco.mj_step(model, data)
    return data


def grounded_non_wheel(model, data) -> list[str]:
    names = set()
    for i in range(data.ncon):
        for gid in (data.contact[i].geom1, data.contact[i].geom2):
            n = model.geom(gid).name
            if n and "wheel_geom" not in n and n != "floor":
                names.add(n)
    return sorted(names)


# ── 1. Structure ─────────────────────────────────────────────────────────────
def section_structure() -> None:
    print("\n1  STRUCTURE")
    for path in MODELS:
        m = mujoco.MjModel.from_xml_path(path)
        tag = path.split("/")[-1]
        check(f"{tag}: 16 actuators", m.nu == 16, f"nu={m.nu}")
        check(f"{tag}: nq=23 nv=22 (free joint + 16 hinges)",
              m.nq == 23 and m.nv == 22, f"nq={m.nq} nv={m.nv}")
        check(f"{tag}: 17 non-world bodies", m.nbody - 1 == 17, f"{m.nbody - 1}")
        check(f"{tag}: total mass {SPEC.total_mass:.3f} kg",
              close(float(m.body_subtreemass[1]), SPEC.total_mass, 1e-3),
              f"{float(m.body_subtreemass[1]):.4f} kg")

    m = mujoco.MjModel.from_xml_path(MODELS[0])

    # The action-space contract quoted in envs/hound_env.py's docstring.
    expected = [f"{leg}_{j}" for leg in LEGS
                for j in ("abduct", "hip", "knee", "wheel")]
    actual = [m.actuator(i).name for i in range(m.nu)]
    check("actuator order is per-leg [abduct, hip, knee, wheel], FL FR RL RR",
          actual == expected, "" if actual == expected else f"got {actual}")

    # Gears are what the docstring says they are.
    gears = {n: float(m.actuator(n).gear[0]) for n in actual}
    check("wheel gear is the small one (traction-limited)",
          all(close(gears[f"{l}_wheel"], SPEC.gear_wheel, 1e-6) for l in LEGS)
          and gears["FL_wheel"] < gears["FL_hip"],
          f"wheel {gears['FL_wheel']} vs hip {gears['FL_hip']} vs knee {gears['FL_knee']}")

    # Decoration must be weightless and non-colliding — the same invariant
    # check_shell_physics.py enforces for the spider's shell. Geom mass is
    # not kept in mjModel (the compiler folds it into the body), so the test
    # is that every BODY weighs exactly its one structural link: any mass
    # leaking out of a density=0 decorative geom would show up here.
    n_deco = int(np.sum((m.geom_contype == 0) & (m.geom_conaffinity == 0)))
    expected_mass = {"trunk": SPEC.trunk_mass, "abduct": SPEC.hip_mass,
                     "thigh": SPEC.thigh_mass, "calf": SPEC.calf_mass,
                     "wheel": SPEC.wheel_mass}
    worst, worst_name = 0.0, ""
    for i in range(1, m.nbody):
        name = m.body(i).name
        want = expected_mass[name if name == "trunk" else name.split("_")[1]]
        err = abs(float(m.body(i).mass[0]) - want)
        if err > worst:
            worst, worst_name = err, name
    check(f"{n_deco} decorative geoms add no mass to any body",
          worst < 1e-9, f"largest error {worst:.2e} kg on {worst_name}")

    # Only the tyres are supposed to reach the ground in a healthy stance;
    # everything else that collides is structure, and section 2 checks that
    # none of it actually touches.
    colliders = sorted({m.geom(i).name for i in range(m.ngeom)
                        if m.geom_conaffinity[i] != 0 and m.geom(i).name
                        and m.geom(i).name != "floor"})
    tyres = [c for c in colliders if "wheel_geom" in c]
    check("exactly 4 tyres collide, alongside the structural capsules",
          len(tyres) == 4, f"{len(tyres)} tyres of {len(colliders)} collidable")


# ── 2. Stance ────────────────────────────────────────────────────────────────
def section_stance() -> None:
    print("\n2  STANCE")
    m = mujoco.MjModel.from_xml_path(MODELS[0])
    d = mujoco.MjData(m)
    d.qpos[:] = stance_qpos(m)
    mujoco.mj_forward(m, d)

    # stance_knee is solved so each axle sits under its own hip pivot.
    offsets = []
    bottoms = []
    for leg in LEGS:
        hip = d.xanchor[m.joint(f"{leg}_hip").id]
        axle = d.xpos[m.body(f"{leg}_wheel").id]
        offsets.append(abs(float(axle[0] - hip[0])))
        bottoms.append(float(axle[2]) - SPEC.wheel_r)
    check("every wheel axle sits directly under its hip pivot",
          max(offsets) < 1e-6, f"max |dx| = {max(offsets):.2e} m")
    check("every wheel touches z=0 in the stance",
          max(abs(b) for b in bottoms) < 1e-6,
          f"max |bottom z| = {max(abs(b) for b in bottoms):.2e} m")
    check(f"trunk stands at {SPEC.stand_z:.4f} m",
          close(float(d.xpos[m.body('trunk').id][2]), SPEC.stand_z, 1e-6))

    # The springs are PRELOADED, so the stance must be a true equilibrium:
    # released with zero control and zero velocity, the machine holds.
    for path in MODELS:
        mm = mujoco.MjModel.from_xml_path(path)
        dd = settle(mm, steps=2000, clearance=0.005)
        z = float(dd.xpos[mm.body("trunk").id][2])
        up = float(dd.body("trunk").xmat[8])
        bad = grounded_non_wheel(mm, dd)
        tag = path.split("/")[-1]
        # The plane must hold the drawn height almost exactly. The desert is
        # allowed to sag and lean, because 10 s of the documented creep
        # (make_hound.py, KNOWN LIMITATION) walks it ~0.5 m backwards and it
        # leans into the drift as it goes. What is NOT negotiable on either
        # world is that it stays on its wheels and does not collapse — those
        # are the assertions that would catch a real regression.
        if "desert" in path:
            ok = z > 0.30 and up > 0.90 and not bad
            note = "(creep tolerated, collapse not)"
        else:
            ok = close(z, SPEC.stand_z, 0.01) and up > 0.98 and not bad
            note = ""
        check(f"{tag}: holds the stance for 10 s with no torque {note}",
              ok,
              f"z={z:.4f} (drawn {SPEC.stand_z:.4f}) upright={up:.3f} "
              f"grounded={bad or 'none'}")

    # Recovers from a drop, which is what the springs are actually for.
    mm = mujoco.MjModel.from_xml_path(MODELS[0])
    q = stance_qpos(mm); q[2] += 0.25
    dd = settle(mm, qpos=q, steps=1500)
    check("recovers from a 25 cm drop with only the tyres touching",
          float(dd.xpos[mm.body("trunk").id][2]) > 0.33
          and not grounded_non_wheel(mm, dd),
          f"z={float(dd.xpos[mm.body('trunk').id][2]):.4f}")


# ── 3. The wheel ─────────────────────────────────────────────────────────────
def section_wheel() -> None:
    print("\n3  WHEEL")
    import gymnasium as gym
    import envs  # noqa: F401  (registers Hound-v0)

    m = mujoco.MjModel.from_xml_path(MODELS[0])

    # Unlimited, unsprung — the three properties the docstring claims.
    for leg in LEGS:
        j = m.joint(f"{leg}_wheel")
        if not check(f"{leg}_wheel is unlimited and unsprung",
                     not bool(m.jnt_limited[j.id])
                     and float(m.jnt_stiffness[j.id]) == 0.0,
                     f"limited={bool(m.jnt_limited[j.id])} "
                     f"stiffness={float(m.jnt_stiffness[j.id])}"):
            break

    # The angle really is unbounded: drive for one episode and read it.
    wheel_adr = m.jnt_qposadr[m.joint("FL_wheel").id]
    wheel_act = [m.actuator(f"{l}_wheel").id for l in LEGS]

    def drive(_k):
        c = np.zeros(m.nu)
        c[wheel_act] = 1.0
        return c

    d = settle(m, steps=10_000, ctrl=drive)   # 50 s = one full episode
    angle = float(d.qpos[wheel_adr])
    check("wheel angle grows without bound over one episode",
          abs(angle) > 500,
          f"FL_wheel reached {angle:.0f} rad ({abs(angle) / (2 * np.pi):.0f} turns) "
          f"in 50 s — never revisits a value, so it cannot be an observation")

    # ...and the env therefore excludes it. Prove it by construction: change
    # ONLY the wheel angles and confirm the observation is untouched.
    env = gym.make("Hound-v0").unwrapped
    env.reset(seed=0)
    obs_a = env._get_obs().copy()
    for leg in LEGS:
        env.data.qpos[env.model.jnt_qposadr[env.model.joint(f"{leg}_wheel").id]] += 137.0
    mujoco.mj_forward(env.model, env.data)
    obs_b = env._get_obs().copy()
    check("observation is invariant to wheel angle (137 rad added to each)",
          np.array_equal(obs_a, obs_b),
          f"max |delta| = {np.abs(obs_a - obs_b).max():.3e}")
    check("observation is 169-dim as documented (141 live + 28 reserved)",
          obs_a.shape == (169,), f"{obs_a.shape}")
    check("the 28 reserved slots are all zero in v0",
          np.array_equal(obs_a[-28:], np.zeros(28)), f"{obs_a[-28:].sum():g}")

    # ...but NOT to wheel velocity, which is the part that carries information.
    for leg in LEGS:
        env.data.qvel[env.model.jnt_dofadr[env.model.joint(f"{leg}_wheel").id]] = 12.0
    mujoco.mj_forward(env.model, env.data)
    check("observation DOES respond to wheel velocity",
          not np.array_equal(obs_a, env._get_obs()))
    env.close()

    # WHAT ACTUALLY LIMITS THRUST — and it is not what make_hound.py's
    # sizing argument assumes, which is exactly why this is measured.
    #
    # The Spec sizes gear_wheel against the friction cone: a wheel carrying
    # a quarter of 17 kg at mu=0.9 breaks traction at 3.19 N*m, so a bigger
    # motor should only buy wheelspin. Sound as an upper bound, and it is
    # still the right reason not to fit a 20 N*m hub drive. But sweep the
    # torque and the machine saturates at ~2 m/s^2, a QUARTER of the mu*g
    # = 8.8 m/s^2 that a pure friction limit would allow, and it saturates
    # with the friction cone barely 5% used.
    #
    # The binding constraint is a WHEELIE. All the thrust is applied at
    # ground level while the mass sits 0.363 m up, so driving hard pitches
    # the machine back onto its rear wheels; the front pair unloads, and
    # past about 6 N*m the total normal force reaches zero — every wheel
    # off the ground, spinning freely. A legged robot with point feet can
    # push through its whole leg and does not do this. This one is a short
    # wheelbase carrying a tall load on four free-spinning contacts, and it
    # is the first quadruped in this repo that can pop a wheelie.
    #
    # So the assertion is the qualitative one that survives the correction:
    # thrust grows with torque, then stops, and the mechanism is unloading.
    T = 0.30                                    # seconds of acceleration
    rows = []
    for frac in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        tau = SPEC.gear_wheel * frac
        mm = mujoco.MjModel.from_xml_path(MODELS[0])
        for leg in LEGS:                        # raise the gear past its limit
            mm.actuator(f"{leg}_wheel").gear[0] = tau
        dd = settle(mm, steps=int(T / mm.opt.timestep), ctrl=drive)
        v = float(dd.qvel[0])
        w = float(np.mean([dd.qvel[mm.jnt_dofadr[mm.joint(f"{l}_wheel").id]]
                           for l in LEGS]))
        fn = 0.0
        for i in range(dd.ncon):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(mm, dd, i, f6)
            fn += f6[0]
        surface = w * SPEC.wheel_r
        rows.append((tau, v / T, fn, 1.0 - v / surface if abs(surface) > 1e-6 else 0.0))
    weight = SPEC.total_mass * 9.81
    if VERBOSE:
        print(f"      {'torque':>8}{'a m/s^2':>10}{'normal N':>10}{'% weight':>10}"
              f"{'slip':>8}")
        for tau, a, fn, slip in rows:
            print(f"      {tau:8.2f}{a:10.2f}{fn:10.1f}{100 * fn / weight:9.0f}%"
                  f"{slip:8.2f}"
                  + ("   <- wheels airborne" if fn < 0.2 * weight else ""))
    a_peak = max(a for _, a, _, _ in rows)
    a_top = rows[-1][1]
    check("thrust grows with wheel torque, then stops",
          rows[0][1] < rows[2][1] and a_top <= a_peak * 1.02,
          f"peak a = {a_peak:.2f} m/s^2; 5x the shipped gear gives "
          f"{a_top:.2f} m/s^2 — more motor buys nothing")
    check("the limit is UNLOADING, not the friction cone",
          rows[-1][2] < 0.5 * weight and rows[-1][3] > 0.5,
          f"at {rows[-1][0]:.1f} N*m the wheels carry "
          f"{100 * rows[-1][2] / weight:.0f}% of the machine's weight and slip "
          f"{100 * rows[-1][3]:.0f}% — it has pitched back onto its rear pair. "
          f"mu*g would have allowed {SPEC.wheel_friction[0] * 9.81:.1f} m/s^2.")


# ── 4. Reward budget ─────────────────────────────────────────────────────────
def section_reward() -> None:
    print("\n4  REWARD")
    import gymnasium as gym
    import envs  # noqa: F401

    env = gym.make("Hound-v0").unwrapped
    w = env._ctrl_cost_weight
    nu = env.model.nu
    worst = w * nu               # every motor saturated
    check("alive out-earns a fully saturated torque bill (no suicide hack)",
          worst < env._healthy_reward,
          f"worst-case ctrl cost {worst:.2f} < alive bonus "
          f"{env._healthy_reward:.2f}  (w={w} x nu={nu})")

    # And empirically: standing still must beat terminating.
    env.reset(seed=0)
    total = 0.0
    for _ in range(100):
        _, r, term, _, _ = env.step(np.zeros(nu))
        total += r
        if term:
            break
    check("100 steps of doing nothing earns a positive return",
          total > 0 and not term, f"return {total:.1f}")
    env.close()


# ── 5. Terrain ───────────────────────────────────────────────────────────────
def section_terrain() -> None:
    print("\n5  TERRAIN")
    import gymnasium as gym
    import envs  # noqa: F401

    for eid in ("Hound-v0", "HoundDesert-v0"):
        env = gym.make(eid).unwrapped
        heights, uprights, fell = [], [], 0
        for seed in range(20):
            env.reset(seed=seed)
            for _ in range(40):                    # 2 s of settling, no torque
                _, _, term, _, info = env.step(np.zeros(16))
                if term:
                    fell += 1
                    break
            heights.append(info["height_above_ground"])
            uprights.append(info["trunk_upright"])
        check(f"{eid}: 20 noisy resets all settle upright",
              fell == 0 and min(uprights) > 0.9 and min(heights) > 0.28,
              f"height min {min(heights):.3f} mean {np.mean(heights):.3f}, "
              f"upright min {min(uprights):.3f}, terminated {fell}/20")
        env.close()

    # The heightfield lookup agrees with what MuJoCo actually collides.
    from envs.terrain import HeightField
    m = mujoco.MjModel.from_xml_path(MODELS[1])
    hf = HeightField.from_model(m)
    check("heightfield helper resolves on the desert model", hf is not None)
    check("spawn pad is flat under the robot",
          close(hf.height_at(0.2, 0.15), hf.height_at(-0.2, -0.15), 1e-9),
          f"h(0.2,0.15)={hf.height_at(0.2, 0.15) * 1000:+.4f} mm")
    check("flat model has no heightfield (lookup degrades to z=0)",
          HeightField.from_model(mujoco.MjModel.from_xml_path(MODELS[0])) is None)


# ── 6. Regression ────────────────────────────────────────────────────────────
def section_regression() -> None:
    print("\n6  REGRESSION")
    import gymnasium as gym
    import envs  # noqa: F401

    # envs/terrain.py was extracted OUT of envs/spyder_env.py. These hashes
    # were taken from the pre-extraction code and must not move: the spider
    # has a 3.75M-step checkpoint trained against those exact dynamics.
    EXPECTED = {
        "Spyder-v0": "ebd1224ce86bcf40efed845e503788e1",
        "SpyderDesert-v0": "cd8f2b2dac0f703f25d54f93f96a5bd5",
    }
    for eid, want in EXPECTED.items():
        env = gym.make(eid)
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        acc = [obs.copy()]
        for k in range(500):
            obs, _, term, _, _ = env.step(rng.uniform(-1, 1, 12))
            acc.append(obs.copy())
            if term:
                obs, _ = env.reset(seed=k)
        got = hashlib.sha256(np.concatenate(acc).tobytes()).hexdigest()[:32]
        check(f"{eid} rollout unchanged by the terrain-helper extraction",
              got == want, f"sha256 {got}")
        env.close()


def main() -> int:
    print(__doc__.splitlines()[0])
    section_structure()
    section_stance()
    section_wheel()
    section_reward()
    section_terrain()
    section_regression()
    print(f"\n{'=' * 66}")
    if _failures:
        print(f"FAILED {len(_failures)}/{_checks}:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"All {_checks} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
