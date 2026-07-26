"""Build the Spyder-v0 visual shell in Blender and export it for MuJoCo.

Run with (Blender is a separate app, NOT a pip package):

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python robots/spyder/build_mesh.py -- assets/meshes

Writes nine OBJ files consumed by assets/spyder12.xml. They are VISUAL ONLY:
every mesh geom is contype=0 conaffinity=0 density=0 group=2, so the robot's
mass, inertia and every contact still come from the capsules authored in the
MJCF. Nothing here changes a single number the physics sees -- which is the
whole point, because Spyder-v0 already has a 3.75M-step policy trained against
those capsules and a prettier robot is not worth reburning it.

Why a shell instead of remodelling the robot: MuJoCo collides a mesh geom as
its CONVEX HULL, never as its actual triangles. A concave carapace or a
clevis-forked leg would collide as the solid blob wrapped around it, so meshes
that replaced the capsules would silently make the robot fatter than it looks.
Keeping meshes decorative and capsules structural is the standard MJCF split
(Unitree, Franka and the DeepMind menagerie models all do exactly this), and it
is the only version where "what you see" and "what collides" stay honest.

    CAVEAT worth knowing: the shell hugs the capsules but is not identical to
    them. The tibia tapers from 0.086 m at the knee to 0.05 m at the ankle
    while its capsule stays 0.08 m the whole way, so mid-shin contacts land
    slightly outside the visible shin. The foot is the case that matters and
    it is exact: the toe pad is a true r=0.08 sphere centred on the capsule's
    end cap, so the surface the foot actually pushes off is the surface drawn.

GEOMETRY CONTRACT -- these numbers are read off assets/spyder12.xml and the two
files must be changed together:

    torso     sphere r=0.25 at body origin; four stub capsules r=0.08 running
              origin -> (+-0.2, +-0.2, 0)
    coxa      body at (+-0.2, +-0.2, 0); capsule r=0.08 origin -> (+-0.12, +-0.12, 0)
    femur     body at (+-0.12, +-0.12, 0); capsule r=0.08 origin -> (+-0.22, +-0.22, 0.25)
    tibia     body at (+-0.22, +-0.22, 0.25); capsule r=0.08 origin -> (+-0.22, +-0.22, -0.52)

All four legs are the SAME shape rotated about z by 45/135/225/315 degrees, so
the leg parts are authored once in a canonical frame with the leg pointing
along +x and instanced four times with a quat in the MJCF. That is why the
canonical lengths below carry a sqrt(2): a leg segment that spans (0.12, 0.12)
in torso coordinates spans 0.12*sqrt(2) along its own axis.

Each part splits into a dark structural `shell` and a coloured `hub`, because
MuJoCo's OBJ loader ignores .mtl -- colour is a per-geom material in the XML,
so one colour needs one mesh. The hub colours are not decoration: they are the
red/green/blue hip/lift/knee markers the old model drew as bare spheres,
promoted into actual servo rotors. The joint layout stays as readable in a
render as it was before, and now it looks like hardware.

Normals are computed here rather than exported from Blender. MuJoCo averages
face normals per vertex when an OBJ has no `vn` lines, which would round off
every machined edge; Blender's own smoothing API meanwhile moved twice between
3.x and 4.1. Writing angle-thresholded corner normals ourselves (SMOOTH_ANGLE
below) sidesteps both: curved surfaces stay smooth, bevels and facets stay
crisp, and the script does not depend on which Blender generation runs it.
"""

import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

# ── Geometry read from assets/spyder12.xml ──────────────────────────────────
SQRT2 = math.sqrt(2.0)
CAP_R = 0.08  # every collision capsule's radius

HIP_R = 0.2 * SQRT2  # torso centre -> hip joint, along the leg's own axis
COXA_LEN = 0.12 * SQRT2  # hip -> lift joint
KNEE = Vector((0.22 * SQRT2, 0.0, 0.25))  # lift -> knee, canonical frame
TOE = Vector((0.22 * SQRT2, 0.0, -0.52))  # knee -> toe, canonical frame

# Faces meeting at less than this angle are smoothed together; sharper ones
# stay hard. 35 deg smooths 24-gon cylinders and bevel fillets while keeping
# the 45 deg facets of the octagonal leg beams and every cap/side edge crisp.
SMOOTH_ANGLE = math.radians(35.0)


# ── Blender primitive helpers ───────────────────────────────────────────────
def _bevel(obj, width, segments=2):
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(30.0)
    return obj


def _aim(obj, direction):
    """Point the object's local +Z (every primitive's axis) along `direction`."""
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
        Vector(direction).normalized()
    )
    return obj


def beam(p0, p1, r0, r1, sides=8, bevel=0.006):
    """Tapered prism from p0 to p1 -- the machined-link primitive."""
    p0, p1 = Vector(p0), Vector(p1)
    span = p1 - p0
    bpy.ops.mesh.primitive_cone_add(
        vertices=sides, radius1=r0, radius2=r1, depth=span.length
    )
    obj = bpy.context.active_object
    _aim(obj, span)
    obj.location = p0 + span * 0.5
    return _bevel(obj, bevel)


def barrel(center, axis, radius, length, sides=24, bevel=0.010):
    """Cylinder aligned to `axis` -- servo housings and pivot bosses."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=radius, depth=length)
    obj = bpy.context.active_object
    _aim(obj, axis)
    obj.location = Vector(center)
    return _bevel(obj, bevel)


def blob(center, radius, scale=(1.0, 1.0, 1.0), segments=24, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=rings, radius=radius
    )
    obj = bpy.context.active_object
    obj.location = Vector(center)
    obj.scale = Vector(scale)
    return obj


def box(center, size, rotation_z=0.0, bevel=0.008):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    obj.scale = Vector(size)
    obj.rotation_euler = (0.0, 0.0, rotation_z)
    obj.location = Vector(center)
    return _bevel(obj, bevel)


# ── Accumulating parts ──────────────────────────────────────────────────────
class Part:
    """Triangle soup for one exported OBJ.

    Primitives are baked in as independent vertex islands rather than boolean-
    unioned. Booleans are the flaky step in any headless Blender pipeline and
    buy nothing here: the mesh is never collided with, so interpenetrating
    closed shells render exactly like a fused one. A robot assembled from
    visibly separate housings is also just what a robot looks like.
    """

    def __init__(self, name):
        self.name = name
        self.verts = []
        self.tris = []
        self.objects = []  # only populated in --blend mode

    def add(self, obj):
        deps = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(deps))
        mesh.transform(obj.matrix_world)
        mesh.calc_loop_triangles()

        base = len(self.verts)
        self.verts.extend(tuple(v.co) for v in mesh.vertices)
        self.tris.extend(
            tuple(i + base for i in tri.vertices) for tri in mesh.loop_triangles
        )

        bpy.data.meshes.remove(mesh)
        if KEEP_OBJECTS:
            # Keep the live object (modifiers unapplied) so the saved .blend
            # stays editable — the bevels are still parameters, not baked.
            self.objects.append(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
        return self

    def add_all(self, objs):
        for obj in objs:
            self.add(obj)
        return self


def _face_normals(verts, tris):
    normals = []
    for a, b, c in tris:
        va, vb, vc = Vector(verts[a]), Vector(verts[b]), Vector(verts[c])
        n = (vb - va).cross(vc - va)
        normals.append(n.normalized() if n.length > 1e-12 else Vector((0.0, 0.0, 1.0)))
    return normals


def _corner_normals(verts, tris, faces):
    """Per-corner normals: average only over neighbours within SMOOTH_ANGLE.

    This is crease-angle shading done by hand. A vertex on a cylinder wall
    averages with its ring neighbours (15 deg apart) and comes out smooth; the
    same vertex on the cap rim refuses to average across the 90 deg edge and
    stays sharp.
    """
    incident = [[] for _ in verts]
    for fi, tri in enumerate(tris):
        for vi in tri:
            incident[vi].append(fi)

    cos_limit = math.cos(SMOOTH_ANGLE)
    out = []
    for fi, tri in enumerate(tris):
        own = faces[fi]
        corners = []
        for vi in tri:
            acc = Vector((0.0, 0.0, 0.0))
            for other in incident[vi]:
                if faces[other].dot(own) >= cos_limit:
                    acc += faces[other]
            corners.append(acc.normalized() if acc.length > 1e-9 else own)
        out.append(corners)
    return out


def write_obj(part, path):
    faces = _face_normals(part.verts, part.tris)
    corners = _corner_normals(part.verts, part.tris, faces)

    normals, index = [], {}
    for tri_corners in corners:
        for n in tri_corners:
            key = (round(n.x, 5), round(n.y, 5), round(n.z, 5))
            if key not in index:
                index[key] = len(normals)
                normals.append(key)

    lines = [
        f"# {part.name} -- Spyder-v0 visual shell, generated by robots/spyder/build_mesh.py",
        "# visual only: MuJoCo collides the capsules in spyder12.xml, not this mesh",
    ]
    lines.extend("v %.6f %.6f %.6f" % v for v in part.verts)
    lines.extend("vn %.5f %.5f %.5f" % n for n in normals)
    for tri, tri_corners in zip(part.tris, corners):
        slots = []
        for vi, n in zip(tri, tri_corners):
            ni = index[(round(n.x, 5), round(n.y, 5), round(n.z, 5))]
            slots.append(f"{vi + 1}//{ni + 1}")
        lines.append("f " + " ".join(slots))

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(part.verts), len(part.tris)


# ── The robot ───────────────────────────────────────────────────────────────
def build_carapace():
    """Torso, authored in the torso body frame (no canonical rotation).

    The chassis is an octagon turned 22.5 deg so a FLAT face, not a corner,
    points down each of the four leg axes -- the shoulder arms grow out of
    flats, the way a milled part would.
    """
    shell = Part("carapace_shell")
    lens = Part("carapace_lens")

    # Chassis plate, dome and belly. Together they enclose the r=0.25
    # collision sphere: 0.278*cos(22.5) = 0.257 at the flats, dome to z=0.183.
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.278, depth=0.20)
    chassis = bpy.context.active_object
    chassis.rotation_euler = (0.0, 0.0, math.radians(22.5))
    shell.add(_bevel(chassis, 0.030, segments=3))
    shell.add(blob((0.0, 0.0, 0.055), 0.205, scale=(1.0, 1.0, 0.56), segments=32))
    shell.add(blob((0.0, 0.0, -0.045), 0.215, scale=(1.0, 1.0, 0.55), segments=32))
    # Access hatch on the crown. Purely to break up the dome — from directly
    # overhead (the angle most eval renders use) a bare ellipsoid reads as an
    # undifferentiated blob and the robot loses all sense of scale.
    hatch = barrel((0.0, 0.0, 0.148), (0.0, 0.0, 1.0), 0.098, 0.030, sides=8, bevel=0.006)
    hatch.rotation_mode = "XYZ"
    hatch.rotation_euler = (0.0, 0.0, math.radians(22.5))
    shell.add(hatch)

    for deg in (45.0, 135.0, 225.0, 315.0):
        a = math.radians(deg)
        u = Vector((math.cos(a), math.sin(a), 0.0))
        hip = u * HIP_R  # = (+-0.2, +-0.2, 0), the coxa body origin
        # Shoulder arm: covers the stub capsule (r=0.08) from the chassis out.
        shell.add(beam(u * 0.13, hip, 0.100, 0.088, sides=6))
        # Hip stator. The coxa's red rotor is narrower but taller, so it caps
        # this housing top and bottom and the joint reads as a servo. Keep the
        # housing deep enough that only a ~15 mm band of red shows on each
        # face: any more and four saturated discs dominate the whole robot.
        shell.add(barrel(hip, (0.0, 0.0, 1.0), 0.108, 0.145))

    # Sensor pod on +x. Spyder has no intrinsic front, but +x IS the reward
    # direction, so marking it costs nothing and makes heading legible in a
    # render -- you can see at a glance whether the robot is facing its goal.
    shell.add(box((0.268, 0.0, 0.058), (0.075, 0.125, 0.062)))
    lens.add(barrel((0.309, 0.0, 0.058), (1.0, 0.0, 0.0), 0.042, 0.026, bevel=0.004))
    return shell, lens


def build_coxa():
    """Hip segment, canonical frame: origin at the hip, leg along +x."""
    shell = Part("coxa_shell")
    hub = Part("coxa_hub")
    hub.add(barrel((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.096, 0.175))
    shell.add(beam((0.0, 0.0, 0.0), (COXA_LEN, 0.0, 0.0), 0.088, 0.080))
    # Yoke at the lift joint; the femur's green boss nests inside it.
    shell.add(barrel((COXA_LEN, 0.0, 0.0), (0.0, 1.0, 0.0), 0.092, 0.155))
    return shell, hub


def build_femur():
    """Thigh, canonical frame: origin at the lift joint, knee at KNEE."""
    shell = Part("femur_shell")
    hub = Part("femur_hub")
    hub.add(barrel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.075, 0.170))
    shell.add(beam((0.0, 0.0, 0.0), KNEE, 0.088, 0.070))
    shell.add(barrel(KNEE, (0.0, 1.0, 0.0), 0.086, 0.145))
    return shell, hub


def build_tibia():
    """Shin, canonical frame: origin at the knee, toe at TOE."""
    shell = Part("tibia_shell")
    hub = Part("tibia_hub")
    pad = Part("tibia_pad")
    hub.add(barrel((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.070, 0.162))
    ankle = TOE * 0.80
    shell.add(beam((0.0, 0.0, 0.0), ankle, 0.086, 0.045))
    shell.add(beam(ankle, TOE, 0.045, 0.050))
    # Exactly the capsule's end cap: r=0.08 centred on TOE. The one place the
    # shell must not lie, because it is the surface that pushes off the ground.
    pad.add(blob(TOE, CAP_R, segments=28, rings=14))
    return shell, hub, pad


# ── Optional .blend assembly ────────────────────────────────────────────────
# The OBJ pipeline never needs an assembled robot: MuJoCo does the assembly
# from the MJCF body tree, which is why the parts are authored one-per-body in
# canonical frames and would all pile up at the origin if opened directly.
# --blend replays the MJCF's kinematics at the zero pose so the GUI shows a
# spider instead of a heap. It is for LOOKING and EDITING; the OBJs remain the
# artifacts MuJoCo loads.
KEEP_OBJECTS = False
TORSO_Z = 0.35  # torso body pos in spyder12.xml

# name -> (material, rgb, roughness, metallic), matching the MJCF materials
BLEND_LOOK = {
    "carapace_shell": ("Shell", (0.19, 0.20, 0.23), 0.35, 0.70),
    "carapace_lens": ("Lens", (0.95, 0.62, 0.15), 0.15, 0.00),
    "coxa_shell": ("Shell", (0.19, 0.20, 0.23), 0.35, 0.70),
    "coxa_hub": ("Hip", (0.76, 0.17, 0.16), 0.40, 0.30),
    "femur_shell": ("Shell", (0.19, 0.20, 0.23), 0.35, 0.70),
    "femur_hub": ("Lift", (0.15, 0.65, 0.25), 0.40, 0.30),
    "tibia_shell": ("Shell", (0.19, 0.20, 0.23), 0.35, 0.70),
    "tibia_hub": ("Knee", (0.20, 0.40, 0.85), 0.40, 0.30),
    "tibia_pad": ("Pad", (0.08, 0.08, 0.09), 0.85, 0.00),
}


def _material(name, rgb, roughness, metallic):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        # Blender <=4.x starts a material with no node tree; 5.x always has one
        # and deprecates use_nodes (removed in 6.0), so only set it when needed.
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:  # input names drift between Blender generations; these three don't
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    mat.diffuse_color = (*rgb, 1.0)  # solid-shading viewport colour
    return mat


def save_blend(parts, path):
    by_name = {p.name: p for p in parts}
    for part in parts:
        mat = _material(*BLEND_LOOK[part.name])
        for obj in part.objects:
            obj.data.materials.append(mat)

    def place(names, frame, collection):
        for name in names:
            for obj in by_name[name].objects:
                dup = obj.copy()  # links to the same mesh data — cheap
                dup.matrix_world = frame @ obj.matrix_world
                collection.objects.link(dup)

    root = bpy.context.scene.collection
    torso = bpy.data.collections.new("torso")
    root.children.link(torso)
    torso_frame = Matrix.Translation((0.0, 0.0, TORSO_Z))
    place(("carapace_shell", "carapace_lens"), torso_frame, torso)

    # Walk the same chain the MJCF declares: each body sits at its parent's
    # frame plus the parent segment's length, all rotated into the leg's axis.
    for leg, deg in enumerate((45.0, 135.0, 225.0, 315.0), start=1):
        rot = Matrix.Rotation(math.radians(deg), 4, "Z")
        basis = rot.to_3x3()
        coxa = Matrix.Translation(
            torso_frame.to_translation() + basis @ Vector((HIP_R, 0.0, 0.0))
        ) @ rot
        femur = Matrix.Translation(
            coxa.to_translation() + basis @ Vector((COXA_LEN, 0.0, 0.0))
        ) @ rot
        tibia = Matrix.Translation(coxa.to_translation()
                                   + basis @ Vector((COXA_LEN, 0.0, 0.0))
                                   + basis @ KNEE) @ rot

        collection = bpy.data.collections.new(f"leg_{leg}")
        root.children.link(collection)
        place(("coxa_shell", "coxa_hub"), coxa, collection)
        place(("femur_shell", "femur_hub"), femur, collection)
        place(("tibia_shell", "tibia_hub", "tibia_pad"), tibia, collection)

    # Drop the canonical originals; the placed copies share their mesh data.
    for part in parts:
        for obj in part.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    return path


def main():
    global KEEP_OBJECTS
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    blend_path = None
    if "--blend" in argv:
        i = argv.index("--blend")
        rest = argv[i + 1 :]
        blend_path = rest[0] if rest else "assets/spyder12.blend"
        argv = argv[:i]
    KEEP_OBJECTS = blend_path is not None

    out_dir = argv[0] if argv else "assets/meshes"
    os.makedirs(out_dir, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    parts = [*build_carapace(), *build_coxa(), *build_femur(), *build_tibia()]

    print(f"\n{'mesh':<18}{'verts':>8}{'tris':>8}")
    total = 0
    for part in parts:
        nv, nt = write_obj(part, os.path.join(out_dir, f"{part.name}.obj"))
        total += nt
        print(f"{part.name:<18}{nv:>8}{nt:>8}")
    print(f"{'TOTAL':<18}{'':>8}{total:>8}")
    print(f"\nwrote {len(parts)} meshes to {out_dir}/")

    if blend_path:
        print(f"assembled robot saved to {save_blend(parts, blend_path)}")


if __name__ == "__main__":
    main()
