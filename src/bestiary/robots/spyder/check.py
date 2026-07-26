"""Prove the Spyder visual shell is invisible to the physics.

    python -m bestiary.robots.spyder.check

assets/meshes/*.obj (built by robots/spyder/build_mesh.py) exist only to make the
robot look like hardware. They are declared contype=0 conaffinity=0 density=0,
which SHOULD mean no contacts and no mass — but "should" is doing a lot of work
in a model where inertiafromgeom="true" derives every body's inertia from its
geoms, and where a 3.75M-step policy is already trained against the capsules.
If a mesh ever leaked into the dynamics, the symptom would be a policy that
quietly walks worse, which is the most expensive kind of bug to notice late.

So this asserts it instead. For each model it builds a BASELINE by deleting
every class="visual" geom from the very XML being shipped, then checks:

    compile-time   nq/nv/nbody, per-body mass and inertia, and the set of
                   geoms that can actually collide
    run-time       2000 steps of an identical pseudo-random action sequence,
                   comparing qpos/qvel BIT-FOR-BIT (not np.allclose)

Bit-for-bit is the right bar. These are the same float ops in the same order on
the same inputs; anything other than an exact match means a mesh perturbed the
model, and a tolerance would just hide it. Random actions are used rather than
a settle-in-place rollout because they drive the robot into the ground and
exercise contacts — the path a stray collidable mesh would show up on.

Regenerating the meshes or editing the shell? Re-run this.
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bestiary import paths

MODELS = [str(paths.SPYDER_XML), str(paths.SPYDER_DESERT_XML)]
STEPS = 2000
SEED = 0


def strip_shell(xml_path):
    """Compile `xml_path` with every class="visual" geom removed.

    The stripped copy is written next to the original so its relative asset
    paths (meshes/, terrain/) still resolve. Unused mesh/material assets are
    left declared — MuJoCo compiles them but an unreferenced asset cannot
    reach the dynamics, which is itself part of what we are asserting.
    """
    tree = ET.parse(xml_path)
    removed = 0
    for parent in tree.iter():
        for geom in [g for g in parent if g.tag == "geom" and g.get("class") == "visual"]:
            parent.remove(geom)
            removed += 1

    fd, tmp = tempfile.mkstemp(
        suffix=".xml", prefix=".baseline_", dir=os.path.dirname(xml_path)
    )
    os.close(fd)
    try:
        tree.write(tmp)
        return mujoco.MjModel.from_xml_path(tmp), removed
    finally:
        os.unlink(tmp)


def rollout(model):
    """Record every quantity SpyderEnv's 113-dim observation is built from.

    cfrc_ext is included deliberately: it is the per-body external-contact
    block of the observation, and it is the channel a stray collidable mesh
    would corrupt most directly. qpos/qvel agreement alone would arguably
    imply it, but the observation is what the policy actually consumes, so it
    gets checked rather than inferred.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(SEED)
    actions = rng.uniform(-1.0, 1.0, size=(STEPS, model.nu))
    qpos = np.empty((STEPS, model.nq))
    qvel = np.empty((STEPS, model.nv))
    cfrc = np.empty((STEPS, model.nbody, 6))
    for i, action in enumerate(actions):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        qpos[i] = data.qpos
        qvel[i] = data.qvel
        cfrc[i] = data.cfrc_ext
    return qpos, qvel, cfrc


def collidable(model):
    """Names of geoms that can actually produce a contact."""
    return sorted(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"<geom {i}>"
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    )


def check(xml_path):
    shipped = mujoco.MjModel.from_xml_path(xml_path)
    baseline, removed = strip_shell(xml_path)

    print(f"\n{xml_path}")
    print(f"  removed {removed} visual geoms -> baseline "
          f"({shipped.ngeom} geoms shipped, {baseline.ngeom} baseline)")

    failures = []

    def expect(label, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{detail}")
        if not ok:
            failures.append(f"{xml_path}: {label}")

    for field in ("nq", "nv", "nbody", "nu"):
        expect(field, getattr(shipped, field) == getattr(baseline, field))

    expect("body_mass", np.array_equal(shipped.body_mass, baseline.body_mass),
           f"  (total {shipped.body_subtreemass[1]:.9f} kg)")
    expect("body_inertia", np.array_equal(shipped.body_inertia, baseline.body_inertia))
    expect("body_ipos", np.array_equal(shipped.body_ipos, baseline.body_ipos))

    a, b = collidable(shipped), collidable(baseline)
    expect("collidable geoms", a == b, f"  ({len(a)}: no mesh among them)"
           if a == b else f"  {set(a) ^ set(b)}")

    a_roll, b_roll = rollout(shipped), rollout(baseline)
    for label, x, y in zip(("qpos", "qvel", "cfrc_ext"), a_roll, b_roll):
        expect(f"{label} over {STEPS} steps", np.array_equal(x, y),
               f"  (max |d| = {np.abs(x - y).max():g})")

    return failures


def main():
    failures = []
    for xml_path in MODELS:
        failures += check(xml_path)

    print()
    if failures:
        print("FAILED — the visual shell is affecting physics:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — the shell is decorative; physics is bit-for-bit unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
