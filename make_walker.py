"""ANVIL — a four-legged siege walker, built in Blender against a 12-DoF rig.

Run with (Blender is a separate app, NOT a pip package):

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python make_walker.py -- --stage build --render

    skeleton   the Empty hierarchy alone — 14 joints, nothing else. Cheap to
               print and verify, and the only thing that must be right before
               any geometry exists.
    build      the machine: every link chamfered, panel-cut and greebled, with
               the four shaders attached. Writes assets/<callsign>/walker.blend,
               which make_walker_scene.py imports to light and render.

The machine's name is a PARAMETER, not a string sprinkled through the file:
CALLSIGN below drives the hull stencils, the material names and every output
path, so renaming the asset is --callsign and a rebuild, never a search and
replace across two files and a folder of renders.

The proportion review that preceded this lives in assets/<callsign>/blockout_*: two
Specs rendered as flat silhouettes and compared before a single bevel existed.
It settled three things that are now baked in below: the belly slings LOW
(2.4 m, not 4.0) so the femur can lift and the machine reads heavy instead of
table-like; each leg root carries its own armor slab rather than one strip per
flank; and the hero camera keeps its 50 mm but stands back far enough to hold
the whole machine in a 2.39:1 frame.

THE RIG IS THE SPEC. The machine is a beauty asset, but its skeleton is a
contract with envs/spyder_env.py's morphology: 4 legs x 3 joints, yaw at the
hip, pitch at the two joints below it, driven by 12 actuators. The turret adds
2 non-locomotion joints (yaw, then barrel pitch) that hang off the chassis and
never enter the action space.

    root
    └─ chassis                        (free body)
       ├─ turret_yaw       (Z)
       │  └─ turret_pod
       │     └─ barrel_pitch (Y)
       │        └─ barrel
       └─ {FL,FR,RL,RR}_coxa_yaw      (Z)
          └─ {..}_coxa
             └─ {..}_femur_pitch      (Y)
                └─ {..}_femur
                   └─ {..}_tibia_pitch (Y)
                      └─ {..}_tibia

Two conventions make that hierarchy usable rather than merely correct:

  * Every link's object origin sits ON its parent joint's pivot, and every
    link is modelled in its own local frame with the limb running along local
    +X. A link therefore never needs a corrective transform: parenting it to
    its Empty with an identity local matrix places it exactly. This is also
    what makes the four legs one leg — build_leg() runs once per leg over the
    same canonical numbers, and a leg differs only by the fixed z-yaw baked
    into its coxa_yaw Empty.
  * Rest pose is a STRAIGHT leg (all joint angles zero), and the standing
    stance is a set of joint angles applied to the Empties, exactly like a
    qpos vector. Nothing about the stance is baked into geometry, so re-posing
    is a number change and never a remodel. The one exception is the foot,
    which must meet flat ground: it is built world-level and counter-rotated
    by the shin's own angle, so it tracks the pose instead of ignoring it.

Positive pitch is nose-down: rotating about local +Y sends local +X toward
-Z, so a NEGATIVE femur pitch lifts the knee into the arachnid arch.

Modelling rules the whole file obeys:

  * Nothing is booleaned. Primitives are separate closed islands inside one
    link mesh — booleans are the flaky step in a headless pipeline, a machine
    is visibly an assembly of parts anyway, and a recessed panel is cheaper
    and cleaner as a backing plate ringed by four frame bars than as a cut.
  * No n-gons: quads everywhere, cap_tris on every cylinder, and the one
    curve-derived mesh (stencil text) is triangulated on the way in.
  * Every link carries an angle-limited Bevel modifier, left UNAPPLIED so the
    chamfer stays a parameter in the .blend. That single modifier is what
    enforces the brief's "no large plane without a bevel" across the machine.
  * Detail is clustered, not smeared: roughly 60% of each surface is left as
    calm armor, and greeble is packed into the 40% where a real machine would
    put it — hatches, seams, actuator mounts, the shoulder of a panel.

Axis convention: +X forward, +Y left, +Z up, meters, Z-up — Blender's native
frame and MuJoCo's, so nothing needs converting at the boundary.
"""

import argparse
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

TAU = math.tau
SEG = 24  # default cylinder resolution

# The machine's name. Drives the pod stencil, the hull ID, the material
# datablock names and the default output paths — change it here (or with
# --callsign) and nothing else needs touching. Keep it short: it is painted
# on the pod flank at 0.27 m tall and has to fit above the countermeasure
# rack, which is about a metre wide.
CALLSIGN = "ANVIL"
HULL_NUMBER = "04"


class Spec:
    """Every number the machine is built from (Part 3 of the brief).

    Derived quantities are properties, never stored, so a change to one
    dimension cannot leave a stale copy of itself somewhere downstream.
    """

    hull = (6.4, 3.0, 2.2)          # core hull L x W x H
    clearance = 2.40                # ground -> chassis underside, standing
    slab = (2.6, 0.5, 2.4)          # armor slab, one per leg root
    pod = (5.8, 2.7, 2.0)           # turret pod L x W x H
    pod_overhang = 1.2              # how far the pod's nose passes the hull's
    coxa_len, coxa_dia = 0.9, 1.1
    femur_len = 3.0
    tibia_len = 3.4
    foot_span = 1.2
    antenna_len, antenna_rake = 2.5, math.radians(30.0)

    # Stance: femur_deg is the input (a lifted knee), the shin absorbs
    # whatever drop is left so the foot lands on the ground.
    femur_deg = -22.0
    ankle_z = 0.55

    hip_inset = 0.35                # hip pivot above the hull's underside
    hip_x, hip_y = 2.20, 1.25       # hip pivots, hull frame
    collar_h = 0.30                 # yoke collar: deck -> pod underside
    yaw_x = 0.40                    # turret axis, just forward of hull centre

    legs = {"FL": 45.0, "FR": -45.0, "RL": 135.0, "RR": -135.0}

    @property
    def hull_z(self):
        return self.clearance + self.hull[2] / 2.0

    @property
    def deck_z(self):
        return self.clearance + self.hull[2]

    @property
    def hip_z(self):
        return self.clearance + self.hip_inset

    @property
    def stance(self):
        """(femur pitch, knee pitch, shin angle from horizontal), radians.

        With a 6.4 m leg the drop from hip to ankle is the whole budget: pick
        the femur and the shin takes the remainder. Slinging the belly to
        2.4 m is what buys the femur enough slack to lift at all — at 4.0 m
        the knee would need to sit 4.35 m above the foot on a 3.4 m shin.
        """
        drop = self.hip_z - self.ankle_z
        femur = math.radians(self.femur_deg)
        tibia_drop = drop - self.femur_len * math.sin(femur)
        if abs(tibia_drop) > self.tibia_len:
            raise ValueError(
                f"unreachable stance: lifting the femur {-self.femur_deg:.0f} deg "
                f"leaves {tibia_drop:.2f} m for a {self.tibia_len:.2f} m shin"
            )
        shin = math.asin(tibia_drop / self.tibia_len)
        return femur, shin - femur, shin


SPEC = Spec()


def asset_dir():
    """Where this machine's outputs live — derived from its name."""
    return f"assets/{CALLSIGN.lower()}"

# Paint. Two-tone break straight from the reference: graphite pod and slabs,
# sand hull between them. The value split is what keeps the silhouette legible
# at night, so it is a hard requirement, not a palette preference.
GRAPHITE = (0.0110, 0.0122, 0.0132)   # #1C1E20, linearised
SAND = (0.2500, 0.2100, 0.1300)       # #8A8065, linearised
STEEL = (0.045, 0.043, 0.040)
BRASS = (0.240, 0.145, 0.045)
AMBER = (1.000, 0.3300, 0.0250)       # #FFA124, linearised
DUST = (0.300, 0.225, 0.140)


# ── Blender plumbing ────────────────────────────────────────────────────────
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    return scene


def collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def empty(name, coll, parent=None, location=(0, 0, 0), rotation=(0, 0, 0), size=0.5):
    """A joint. Its local frame IS the hinge frame — no offsets, no surprises."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = size
    obj.location = location
    obj.rotation_euler = rotation
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def link_mesh(name, mesh, coll, parent, bevel=0.022):
    """Attach a link to its joint.

    Assigning .parent directly (rather than via the parent operator) leaves
    matrix_parent_inverse at identity, so the object's local transform is
    measured straight from the joint frame. Location stays (0,0,0): the mesh
    was authored about the pivot, so it is already where it belongs.
    """
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.parent = parent
    if bevel:
        mod = obj.modifiers.new("chamfer", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(30.0)
        mod.miter_outer = "MITER_ARC"
        mod.harden_normals = False
    # Angle-based smoothing: cylinder walls stay round, every machined facet
    # stays crisp. Blender 4.1 moved auto-smooth off the mesh and onto a
    # modifier, so the operator is the call that survives that move; the flat
    # fallback is only there so a future rename degrades instead of crashing.
    previous = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(35.0))
    except (AttributeError, RuntimeError):
        obj.data.shade_flat()
    obj.select_set(False)
    bpy.context.view_layer.objects.active = previous
    return obj


def _finish(name, bm):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate()
    return mesh


# ── Primitives ──────────────────────────────────────────────────────────────
def box(size, center=(0, 0, 0), rot=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(
        bm,
        matrix=Matrix.Translation(center) @ (rot or Matrix.Identity(4))
        @ Matrix.Diagonal(Vector((*size, 1.0))),
        verts=bm.verts,
    )
    return _finish("box", bm)


def cyl(radius, length, center=(0, 0, 0), axis="Z", r2=None, segments=SEG, rot=None):
    """Capped cylinder/cone. cap_tris because an n-gon cap is a modelling bug
    waiting to happen the moment anything bevels or subdivides it."""
    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=True, segments=segments,
        radius1=radius, radius2=radius if r2 is None else r2, depth=length,
    )
    spin = {"Z": Matrix.Identity(4),
            "X": Matrix.Rotation(math.radians(90.0), 4, "Y"),
            "Y": Matrix.Rotation(math.radians(-90.0), 4, "X")}[axis]
    bmesh.ops.transform(
        bm, matrix=Matrix.Translation(center) @ (rot or Matrix.Identity(4)) @ spin,
        verts=bm.verts,
    )
    return _finish("cyl", bm)


def ball(radius, center=(0, 0, 0), subdivisions=2):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center), verts=bm.verts)
    return _finish("ball", bm)


def hexa(verts):
    """Eight corners -> six quads. The wedge primitive: a box whose faces are
    allowed to be non-parallel, which is what most panels on this machine are.

    Corner order: bottom rear-left, rear-right, front-right, front-left, then
    the same four on top.

    Winding matters and is easy to get backwards. Every face below is wound so
    its normal points OUT of the solid — a flipped wedge would not just shade
    inside-out, it would invert the surface normal the dust mask reads, and
    dust would settle on the undersides. Build.mesh() recalculates normals as
    a backstop, but the winding is authored correctly here regardless.
    """
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new("hexa")
    mesh.from_pydata([Vector(v) for v in verts], [], faces)
    mesh.validate()
    return mesh


# Flank frames for stencil text: the matrix that makes flat text read
# correctly to a viewer standing outside that side of the machine.
FLANK_L = Matrix.Rotation(math.pi, 4, "Z") @ Matrix.Rotation(math.pi / 2, 4, "X")
FLANK_R = Matrix.Rotation(math.pi / 2, 4, "X")


def stencil(body, size, at, frame=FLANK_L, depth=0.012, spacing=0.06):
    """Extruded block text, triangulated (curve conversion is the one place
    n-gons could sneak in). Worn paint is the material's job, not the mesh's."""
    curve = bpy.data.curves.new("stencil", type="FONT")
    curve.body = body
    curve.size = size
    curve.extrude = depth
    curve.space_character = 1.0 + spacing
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    obj = bpy.data.objects.new("stencil", curve)
    bpy.context.scene.collection.objects.link(obj)
    deps = bpy.context.evaluated_depsgraph_get()
    tmp = bpy.data.meshes.new_from_object(obj.evaluated_get(deps))
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.curves.remove(curve)

    bm = bmesh.new()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.transform(bm, matrix=Matrix.Translation(at) @ frame, verts=bm.verts)
    return _finish("stencil", bm)


class Build:
    """Accumulates tagged primitives into one link mesh with material slots.

    A link is one object (the rig contract) but never one shape, so every
    primitive is added with the material it wears and the slot table is built
    as it goes.
    """

    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.slots = []

    def add(self, mesh, mat="armor"):
        if mat not in self.slots:
            self.slots.append(mat)
        index = self.slots.index(mat)
        start = len(self.bm.faces)
        self.bm.from_mesh(mesh)
        bpy.data.meshes.remove(mesh)
        self.bm.faces.ensure_lookup_table()
        for face in self.bm.faces[start:]:
            face.material_index = index
        return self

    def all(self, meshes, mat="armor"):
        for mesh in meshes:
            self.add(mesh, mat)
        return self

    def mesh(self):
        # Every island is a closed solid, so an outward recalculation is
        # unambiguous — and it is cheap insurance against one hand-wound wedge
        # shading inside-out somewhere in a thousand primitives.
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces[:])
        out = bpy.data.meshes.new(self.name)
        self.bm.to_mesh(out)
        self.bm.free()
        out.validate()
        for name in self.slots:
            out.materials.append(MATERIALS[name])
        return out


# ── Greeble kit ─────────────────────────────────────────────────────────────
# The scale cues from Part 1: hand-sized fasteners, handles, rungs, latches.
# These are the only reason the eye reads the machine as enormous, so they are
# generated in bulk rather than placed one at a time.
def bolts(count, start, step, radius=0.035, height=0.03, axis="Z"):
    """A row of fastener heads from `start`, stepping by `step`."""
    start, step = Vector(start), Vector(step)
    return [cyl(radius, height, center=start + step * i, axis=axis, segments=8)
            for i in range(count)]


def bolt_ring(count, center, radius, axis="Z", head=0.04, height=0.03):
    center = Vector(center)
    out = []
    for i in range(count):
        a = TAU * i / count
        if axis == "Z":
            offset = Vector((math.cos(a) * radius, math.sin(a) * radius, 0))
        elif axis == "Y":
            offset = Vector((math.cos(a) * radius, 0, math.sin(a) * radius))
        else:
            offset = Vector((0, math.cos(a) * radius, math.sin(a) * radius))
        out.append(cyl(head, height, center=center + offset, axis=axis, segments=8))
    return out


def panel(size, center, normal="Y", inset=0.16, rim=0.055, depth=0.05):
    """A recessed rectangular panel — backing plate ringed by four frame bars.

    No boolean: the frame stands proud of the backing plate, so the middle
    reads as recessed from any angle a camera will ever see it from.
    """
    w, h = size
    cx, cy, cz = center
    out = []
    if normal == "Y":
        out.append(box((w, depth * 0.5, h), (cx, cy, cz)))
        for dx, dz, bw, bh in ((0, h / 2 - rim / 2, w, rim), (0, -h / 2 + rim / 2, w, rim),
                               (w / 2 - rim / 2, 0, rim, h), (-w / 2 + rim / 2, 0, rim, h)):
            out.append(box((bw, depth, bh), (cx + dx, cy, cz + dz)))
    else:  # normal == "Z", a deck panel
        out.append(box((w, h, depth * 0.5), (cx, cy, cz)))
        for dx, dy, bw, bh in ((0, h / 2 - rim / 2, w, rim), (0, -h / 2 + rim / 2, w, rim),
                               (w / 2 - rim / 2, 0, rim, h), (-w / 2 + rim / 2, 0, rim, h)):
            out.append(box((bw, bh, depth), (cx + dx, cy + dy, cz)))
    return out


def louvers(count, size, center, pitch, normal="Y", tilt=math.radians(28.0)):
    """A bank of angled slats inside a shallow frame — the vent language."""
    w, h = size
    cx, cy, cz = center
    out = [box((w + 0.10, 0.04, count * pitch + 0.10), (cx, cy - 0.02, cz))]
    for i in range(count):
        z = cz - (count - 1) * pitch / 2 + i * pitch
        out.append(box((w, 0.05, h), (cx, cy, z), rot=Matrix.Rotation(tilt, 4, "X")))
    return out


def handle(span, center, normal="Z", thickness=0.045, lift=0.10):
    """A recessed grab handle: two stand-offs and a bar. Hand-sized on
    purpose — it is the scale cue doing the most work in a wide shot."""
    cx, cy, cz = center
    out = []
    if normal == "Z":
        for dy in (-span / 2, span / 2):
            out.append(box((thickness, thickness, lift), (cx, cy + dy, cz + lift / 2)))
        out.append(box((thickness, span + thickness, thickness),
                       (cx, cy, cz + lift)))
    else:
        for dx in (-span / 2, span / 2):
            out.append(box((thickness, lift, thickness), (cx + dx, cy + lift / 2, cz)))
        out.append(box((span + thickness, thickness, thickness), (cx, cy + lift, cz)))
    return out


def hatch(size, center, normal="Z"):
    w, h = size
    cx, cy, cz = center
    out = [box((w, h, 0.06), (cx, cy, cz))]
    out += bolts(4, (cx - w / 2 + 0.10, cy - h / 2 + 0.10, cz + 0.03),
                 ((w - 0.20) / 3, 0, 0))
    out += bolts(4, (cx - w / 2 + 0.10, cy + h / 2 - 0.10, cz + 0.03),
                 ((w - 0.20) / 3, 0, 0))
    out += handle(min(w, h) * 0.45, (cx, cy, cz + 0.03), normal=normal, lift=0.07)
    return out


def ladder(rungs, start, step, width=0.42):
    """Stand-off rungs on a flank. Reads as climbable, which is the point."""
    start, step = Vector(start), Vector(step)
    out = []
    for i in range(rungs):
        base = start + step * i
        out.append(box((0.06, 0.10, 0.05), base + Vector((0, 0.05, 0))))
        out.append(box((width, 0.05, 0.05), base + Vector((0, 0.10, 0))))
    return out


# ── Materials ───────────────────────────────────────────────────────────────
# Four shaders (Part 5). "Raw mechanical" is instanced twice — steel and brass
# are the same graph with a different base colour, which is what the brief's
# brass-toned countermeasure rims need without inventing a fifth look.
MATERIALS = {}


def _nodes(mat):
    # Blender 5 gives every material a node tree and deprecates use_nodes, so
    # the flag is only touched on the older generation that still needs it.
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    return nt, out


# ShaderNodeMix carries one socket per data type, so RGBA lives at fixed
# indices rather than under a unique name. Named lookup would grab the float
# sockets and silently mix nothing.
MIX_FAC, MIX_A, MIX_B, MIX_RESULT = 0, 6, 7, 2


def _mix(nt, fac, color_a, color_b, loc):
    """Colour mix; either colour may be a literal or an upstream socket."""
    node = nt.nodes.new("ShaderNodeMix")
    node.data_type = "RGBA"
    node.location = loc
    for index, value in ((MIX_A, color_a), (MIX_B, color_b)):
        if hasattr(value, "is_linked"):
            nt.links.new(value, node.inputs[index])
        else:
            node.inputs[index].default_value = (*value, 1.0)
    if hasattr(fac, "is_linked"):
        nt.links.new(fac, node.inputs[MIX_FAC])
    else:
        node.inputs[MIX_FAC].default_value = fac
    return node.outputs[MIX_RESULT]


def _weathering(nt, base_color, base_rough, metallic, wear_color=(0.09, 0.085, 0.08)):
    """Dust by height, grime under overhangs, bare metal on every edge.

    Three masks, all procedural, all driven by geometry rather than UVs:

      dust   world Z (heavy at the feet, gone by the turret deck) times how
             upward-facing the surface is — dust falls, it does not stick to
             walls. Broken up by noise so the line is never a band.
      grime  downward-facing surfaces darkened where ambient occlusion says
             they are tucked under something. This is the gradient under
             every horizontal overhang the brief asks for.
      wear   mesh pointiness: convex edges rub through to bare metal. The
             chamfer modifier gives every panel a real bevel to catch it, so
             wear lands exactly where the light already breaks.

    Returns the Principled BSDF, wired and ready to attach to an output.
    """
    def math(op, a, b, loc):
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = op
        node.location = loc
        for index, value in ((0, a), (1, b)):
            if hasattr(value, "is_linked"):
                nt.links.new(value, node.inputs[index])
            else:
                node.inputs[index].default_value = value
        return node.outputs[0]

    def ramp(x, lo, hi, loc):
        """Remap a signal into 0..1 over [lo, hi], clamped. lo > hi inverts."""
        node = nt.nodes.new("ShaderNodeMapRange")
        node.location = loc
        node.inputs[1].default_value = lo
        node.inputs[2].default_value = hi
        node.clamp = True
        nt.links.new(x, node.inputs[0])
        return node.outputs[0]

    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1400, 0)
    normal = nt.nodes.new("ShaderNodeSeparateXYZ")
    normal.location = (-1200, -150)
    nt.links.new(geo.outputs["Normal"], normal.inputs[0])
    position = nt.nodes.new("ShaderNodeSeparateXYZ")
    position.location = (-1200, -400)
    nt.links.new(geo.outputs["Position"], position.inputs[0])

    # ── dust: falls from the sky, so it needs an up-face and low ground ─────
    up = ramp(normal.outputs["Z"], 0.10, 0.75, (-1000, -150))
    low = ramp(position.outputs["Z"], 4.40, 0.60, (-1000, -400))
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-1200, -650)
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 6.0
    grain = ramp(noise.outputs["Fac"], 0.30, 0.70, (-1000, -650))
    dust = math("MULTIPLY", math("MULTIPLY", up, low, (-800, -250)), grain, (-620, -250))
    dust = math("MULTIPLY", dust, 0.9, (-450, -250))

    # ── grime: downward-facing AND tucked under something ───────────────────
    down = ramp(normal.outputs["Z"], -0.10, -0.80, (-1000, -880))
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.location = (-1000, -1080)
    ao.inputs["Distance"].default_value = 0.9
    ao.only_local = True
    cavity = math("SUBTRACT", 1.0, ao.outputs["Color"], (-800, -1080))
    grime = math("MULTIPLY", math("MULTIPLY", down, cavity, (-620, -950)), 0.75,
                 (-450, -950))

    # ── wear: convex edges rub through to bare metal ────────────────────────
    # Pointiness is a mesh signal, and the chamfer modifier is what gives it
    # something to find: every panel edge becomes a narrow convex band, so the
    # wear lands exactly where the light already breaks.
    edge = nt.nodes.new("ShaderNodeValToRGB")
    edge.location = (-1000, 300)
    edge.color_ramp.elements[0].position = 0.50
    edge.color_ramp.elements[1].position = 0.62
    nt.links.new(geo.outputs["Pointiness"], edge.inputs["Fac"])
    edge_noise = nt.nodes.new("ShaderNodeTexNoise")
    edge_noise.location = (-1200, 520)
    edge_noise.inputs["Scale"].default_value = 26.0
    edge_noise.inputs["Detail"].default_value = 8.0
    wear = math("MULTIPLY", edge.outputs["Color"], edge_noise.outputs["Fac"], (-700, 380))
    wear = math("MULTIPLY", wear, 1.35, (-520, 380))

    # ── colour chain: paint -> dust -> grime -> bare metal ──────────────────
    dusted = _mix(nt, dust, base_color, DUST, (-250, -100))
    grimed = _mix(nt, grime, dusted, (0.020, 0.017, 0.014), (-60, -100))
    worn = _mix(nt, wear, grimed, wear_color, (140, 0))

    rough = _mix(nt, dust, (base_rough,) * 3, (0.92, 0.92, 0.92), (140, -260))
    metal = math("MAXIMUM", metallic, wear, (140, -460))

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (500, 0)
    nt.links.new(worn, bsdf.inputs["Base Color"])
    nt.links.new(rough, bsdf.inputs["Roughness"])
    nt.links.new(metal, bsdf.inputs["Metallic"])
    # Anisotropic scuffing along the panels, subtle enough to only show in a
    # grazing highlight — which at night is most of what is visible.
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = 0.35
    return bsdf


def build_materials():
    """Four shaders: graphite armor, sand hull, raw mechanical, amber emissive."""
    MATERIALS.clear()

    for name, color, rough, metallic in (
        ("armor", GRAPHITE, 0.55, 0.15),
        ("hull", SAND, 0.65, 0.05),
        ("steel", STEEL, 0.30, 1.00),
        ("brass", BRASS, 0.35, 1.00),
    ):
        mat = bpy.data.materials.new(f"{CALLSIGN}_{name}")
        nt, out = _nodes(mat)
        bsdf = _weathering(nt, color, rough, metallic)
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        MATERIALS[name] = mat

    # Amber emissive. One shader for every fixture: the strength differences
    # the brief describes come from real lamps placed at the headlights in
    # make_walker_scene.py, which is also the only way the light spills onto
    # ground and the machine's own lower armor.
    mat = bpy.data.materials.new(f"{CALLSIGN}_amber")
    nt, out = _nodes(mat)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.location = (500, 0)
    emit.inputs["Color"].default_value = (*AMBER, 1.0)
    emit.inputs["Strength"].default_value = 18.0
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    MATERIALS["amber"] = mat
    return MATERIALS


# ── Chassis ─────────────────────────────────────────────────────────────────
def build_chassis(spec, colls, joints):
    """Sand hull: stacked armor blocks, slot window, headlights, canisters.

    Modelled as three stacked blocks rather than one box — a belly that tucks
    in, a full-width middle, a narrower top deck. That horizontal break is
    what stops a 6.4 m box from reading as a crate, and it gives the rivet
    rows and hatches a seam to live on.
    """
    b = Build("chassis")
    L, W, H = spec.hull
    hx, hy, hz = L / 2, W / 2, H / 2

    # Belly, middle, deck.
    b.add(hexa([(-hx + 0.25, hy - 0.30, -hz), (-hx + 0.25, -hy + 0.30, -hz),
                (hx - 0.15, -hy + 0.30, -hz), (hx - 0.15, hy - 0.30, -hz),
                (-hx + 0.10, hy - 0.05, -hz + 0.75), (-hx + 0.10, -hy + 0.05, -hz + 0.75),
                (hx, -hy + 0.05, -hz + 0.75), (hx, hy - 0.05, -hz + 0.75)]), "hull")
    b.add(box((L, W, 0.95), (0, 0, -hz + 1.22)), "hull")
    # Deck: narrower, with the front sloped down into a glacis.
    b.add(hexa([(-hx + 0.15, hy - 0.22, -hz + 1.70), (-hx + 0.15, -hy + 0.22, -hz + 1.70),
                (hx - 0.10, -hy + 0.22, -hz + 1.70), (hx - 0.10, hy - 0.22, -hz + 1.70),
                (-hx + 0.15, hy - 0.35, hz), (-hx + 0.15, -hy + 0.35, hz),
                (hx - 0.95, -hy + 0.35, hz), (hx - 0.95, hy - 0.35, hz)]), "hull")
    # Front glacis wedge — the nose the headlights and window sit on.
    b.add(hexa([(hx - 0.95, hy - 0.35, -hz + 1.70), (hx - 0.95, -hy + 0.35, -hz + 1.70),
                (hx, -hy + 0.30, -hz + 1.70), (hx, hy - 0.30, -hz + 1.70),
                (hx - 0.95, hy - 0.35, hz), (hx - 0.95, -hy + 0.35, hz),
                (hx - 0.30, -hy + 0.35, hz - 0.16), (hx - 0.30, hy - 0.35, hz - 0.16)]),
          "hull")

    # ── Front face: slot window with warm spill, and two headlights ─────────
    win_z = -hz + 1.62
    b.add(box((0.10, 2.05, 0.42), (hx + 0.02, 0, win_z)), "hull")      # surround
    b.add(box((0.06, 1.80, 0.20), (hx + 0.06, 0, win_z)), "amber")     # the glow
    for i in range(7):                                                  # mullions
        b.add(box((0.05, 0.05, 0.22), (hx + 0.09, -0.78 + i * 0.26, win_z)), "steel")
    b.all(bolts(9, (hx + 0.02, -1.00, win_z + 0.235), (0, 0.25, 0), axis="X"), "steel")

    for sign in (1.0, -1.0):
        y = sign * 0.92
        b.add(cyl(0.26, 0.30, (hx - 0.02, y, -hz + 1.06), axis="X"), "hull")
        b.add(cyl(0.30, 0.10, (hx + 0.10, y, -hz + 1.06), axis="X"), "steel")
        b.add(cyl(0.21, 0.06, (hx + 0.16, y, -hz + 1.06), axis="X"), "amber")
        b.all(bolt_ring(8, (hx + 0.13, y, -hz + 1.06), 0.26, axis="X"), "steel")
        # Marker cluster low on the flank, forward — the reference's little
        # amber dots that give the hull its only warm accent below the window.
        for i in range(3):
            b.add(box((0.09, 0.05, 0.13), (hx - 1.15 + i * 0.30, sign * (hy + 0.03),
                                           -hz + 0.52)), "amber")
            b.add(box((0.13, 0.04, 0.17), (hx - 1.15 + i * 0.30, sign * (hy + 0.01),
                                           -hz + 0.52)), "hull")

    # ── Flanks: panel insets, rivet seams, hatches, ladder, stencils ────────
    for sign in (1.0, -1.0):
        y = sign * (hy + 0.02)
        b.all(panel((1.5, 0.72), (-0.55, y, -hz + 1.25)), "hull")
        b.all(panel((1.1, 0.72), (-2.25, y, -hz + 1.25)), "hull")
        b.all(bolts(14, (-2.95, y, -hz + 0.83), (0.32, 0, 0), axis="Y"), "steel")
        b.all(bolts(14, (-2.95, y, -hz + 1.68), (0.32, 0, 0), axis="Y"), "steel")
        b.all(louvers(5, (0.62, 0.10), (1.35, y, -hz + 1.28), 0.15), "hull")
        b.all(ladder(3, (-1.30, sign * hy, -hz + 0.35), (0, 0, 0.42)), "steel")
    hull_id = f"{CALLSIGN}-{HULL_NUMBER}"
    b.add(stencil(hull_id, 0.30, (-0.40, hy + 0.07, -hz + 0.55), FLANK_L), "armor")
    b.add(stencil(hull_id, 0.30, (-0.40, -hy - 0.07, -hz + 0.55), FLANK_R), "armor")

    # ── Deck: hatches and grab handles ──────────────────────────────────────
    for x, y in ((-2.35, 0.62), (-2.35, -0.62), (-1.30, 0.0)):
        b.all(hatch((0.78, 0.66), (x, y, hz + 0.02)), "hull")
    b.all(bolts(10, (-3.05, 0.0, hz + 0.02), (0.62, 0, 0)), "steel")

    chassis = link_mesh("chassis", b.mesh(), colls["chassis"],
                        joints["chassis_frame"])

    # ── Two large canisters slung under the nose ────────────────────────────
    c = Build("chassis_canisters")
    for sign in (1.0, -1.0):
        y = sign * 0.66
        c.add(cyl(0.46, 2.30, (hx - 1.60, y, -hz - 0.40), axis="X"), "hull")
        for x in (hx - 2.75, hx - 0.45):
            c.add(cyl(0.50, 0.12, (x, y, -hz - 0.40), axis="X"), "steel")
        for x in (hx - 2.30, hx - 0.90):      # retaining straps
            c.add(cyl(0.49, 0.09, (x, y, -hz - 0.40), axis="X"), "steel")
        c.add(cyl(0.09, 0.55, (hx - 0.30, y, -hz - 0.40), axis="X"), "steel")
    link_mesh("chassis_canisters", c.mesh(), colls["chassis"],
              joints["chassis_frame"])
    return chassis


# ── Armor slab ──────────────────────────────────────────────────────────────
def build_slab(spec, leg, colls, joints):
    """The pauldron over one leg root — the darkest, calmest shape on the
    machine, and the reason four limbs read as one mass.

    Carried BY the leg (parented to coxa_yaw), so the armor swings with the
    hip the way the reference's stacked plates clearly do. Its face is left
    deliberately empty apart from two recessed insets and a perimeter bolt
    row: this is the 60% calm side of the 60/40 ratio.
    """
    b = Build(f"{leg}_slab")
    t, w, h = spec.slab[1], spec.slab[0], spec.slab[2]
    x = spec.coxa_len + 0.28

    # The outboard face cuts back at the bottom, so the plate reads as a
    # hanging shield rather than a crate: a rectangle seen at 45 degrees looks
    # like a box, and one clipped corner is all it takes to stop that.
    b.add(hexa([(x - t / 2, w / 2, -h / 2), (x - t / 2, -w / 2, -h / 2),
                (x + t / 2, -w / 2 + 0.10, -h / 2 + 0.52),
                (x + t / 2, w / 2 - 0.10, -h / 2 + 0.52),
                (x - t / 2, w / 2, h / 2), (x - t / 2, -w / 2, h / 2),
                (x + t / 2, -w / 2 + 0.10, h / 2 - 0.18),
                (x + t / 2, w / 2 - 0.10, h / 2 - 0.18)]))
    for dy, dz in ((0.62, 0.42), (-0.62, 0.42), (0.0, -0.48)):
        b.all(panel((0.86, 0.72), (x + t / 2 + 0.01, dy, dz), normal="Y"), "armor")
    for dz in (h / 2 - 0.16, -h / 2 + 0.72):
        b.all(bolts(7, (x + t / 2 + 0.02, -w / 2 + 0.24, dz), (0, 0.36, 0), axis="X"),
              "steel")
    # Slim top bracket and two arms back to the coxa: the plate must visibly
    # HANG off something, or the armor looks glued on.
    b.add(box((t + 0.10, w * 0.55, 0.15), (x, 0, h / 2 - 0.04)))
    for dy in (0.62, -0.62):
        b.add(box((0.62, 0.20, 0.26), (x - t / 2 - 0.26, dy, h / 2 - 0.30)), "steel")
    b.add(box((0.50, 0.30, 0.30), (x - t / 2 - 0.22, 0.0, 0.30)), "steel")

    # Rotate the plate into its leg's frame is unnecessary — it is authored in
    # that frame already, facing outboard along the limb.
    link_mesh(f"{leg}_slab", b.mesh(), colls[f"leg_{leg}"],
              joints[f"{leg}_coxa_yaw"])


# ── Turret ──────────────────────────────────────────────────────────────────
def build_turret(spec, colls, joints):
    """Wedge pod on a visible yoke: stencil, countermeasures, antennas, vents.

    Local origin = the yaw axis at the top of the collar. The pod is pushed
    forward so its nose clears the hull by pod_overhang — that forward lean is
    the most striking thing in the reference, so it is derived from the hull
    rather than eyeballed.
    """
    b = Build("turret_pod")
    nose = spec.hull[0] / 2 - spec.yaw_x + spec.pod_overhang     # +4.00
    tail = nose - spec.pod[0]                                    # -1.80
    hw, ph = spec.pod[1] / 2, spec.pod[2]

    # Main wedge: rear taller, top sloping down and forward, blunt nose.
    b.add(hexa([(tail, hw, 0.0), (tail, -hw, 0.0),
                (nose, -hw + 0.22, 0.38), (nose, hw - 0.22, 0.38),
                (tail, hw, ph), (tail, -hw, ph),
                (nose, -hw + 0.22, ph * 0.62), (nose, hw - 0.22, ph * 0.62)]))
    # Raised rear deck — where the antennas and the big hatches live.
    b.add(hexa([(tail + 0.10, hw - 0.12, ph - 0.05), (tail + 0.10, -hw + 0.12, ph - 0.05),
                (tail + 2.30, -hw + 0.12, ph - 0.05), (tail + 2.30, hw - 0.12, ph - 0.05),
                (tail + 0.18, hw - 0.20, ph + 0.30), (tail + 0.18, -hw + 0.20, ph + 0.30),
                (tail + 2.15, -hw + 0.20, ph + 0.22), (tail + 2.15, hw - 0.20, ph + 0.22)]))
    # Cheek chamfer: a lower facet that tucks the flank under before it meets
    # the belly. Kept to the bottom 0.35 m on purpose — everything above it is
    # the calm forward flank the stencil and countermeasures need, and a facet
    # that climbed higher would leave nowhere flat to put them.
    for sign in (1.0, -1.0):
        b.add(hexa([(tail + 0.30, sign * (hw - 0.26), 0.0),
                    (tail + 0.30, sign * (hw - 0.30), -0.03),
                    (nose - 0.20, sign * (hw - 0.48), 0.35),
                    (nose - 0.20, sign * (hw - 0.44), 0.32),
                    (tail + 0.30, sign * hw, 0.36), (tail + 0.30, sign * (hw - 0.04), 0.33),
                    (nose - 0.20, sign * (hw - 0.22), 0.68),
                    (nose - 0.20, sign * (hw - 0.18), 0.71)]))

    # ── Belly: a recessed underside with ribs, not a void ───────────────────
    # From a low camera the pod is read from beneath as much as from the side,
    # and an untouched flat bottom is the fastest way to make a hero asset
    # look unfinished.
    b.add(box((4.60, 1.90, 0.16), (nose - 2.55, 0, -0.05)))
    for sign in (1.0, -1.0):
        b.add(box((4.90, 0.22, 0.30), (nose - 2.50, sign * 1.10, -0.02)))
        b.all(bolts(9, (nose - 4.55, sign * 1.10, -0.18), (0.52, 0, 0)), "steel")
    for x in (nose - 1.10, nose - 2.40, nose - 3.70):
        b.add(box((0.26, 2.00, 0.22), (x, 0, -0.10)), "steel")

    # ── Nose: blunt cap plate, a facet break, the optic housing shoulder ────
    b.add(box((0.14, 1.55, 0.55), (nose - 0.03, 0, 0.72)))
    b.all(bolt_ring(10, (nose - 0.01, 0, 0.72), 0.62, axis="X"), "steel")

    # ── Amber strips: one vertical bar forward, one long horizontal bar ─────
    # running rearward along the lower edge. These are the brightest things on
    # the machine and they define its lower silhouette at night.
    for sign in (1.0, -1.0):
        y = sign * (hw - 0.18)
        b.add(box((0.16, 0.05, 0.62), (nose - 0.62, y, 0.68)), "amber")
        b.add(box((0.24, 0.04, 0.72), (nose - 0.62, y - sign * 0.03, 0.68)))
        b.add(box((2.60, 0.05, 0.15), (nose - 2.30, y, 0.16)), "amber")
        b.add(box((2.75, 0.04, 0.26), (nose - 2.30, y - sign * 0.03, 0.14)))

    # ── Forward-left cheek: 2x3 countermeasure tubes, brass inner rims ──────
    # Stacked UNDER the stencil, both on the forward flank, exactly as the
    # reference arranges them. The pod's top slopes down toward the nose, so
    # the usable flank here is only ~1.4 m tall and the two clusters have to
    # be sized to share it rather than fight for it.
    plate_x = nose - 1.35
    b.add(box((1.02, 0.10, 0.84), (plate_x, hw - 0.06, 0.69)))
    for row in range(2):
        for col in range(3):
            x = plate_x - 0.32 + col * 0.32
            z = 0.52 + row * 0.34
            b.add(cyl(0.130, 0.10, (x, hw + 0.01, z), axis="Y", segments=16), "steel")
            b.add(cyl(0.100, 0.13, (x, hw + 0.03, z), axis="Y", segments=16), "brass")
            b.add(cyl(0.078, 0.17, (x, hw + 0.06, z), axis="Y", segments=16), "armor")

    # ── Callsign, worn block letters, high on the forward left flank ───────
    b.add(stencil(CALLSIGN, 0.27, (plate_x + 0.02, hw + 0.02, 1.24), FLANK_L),
          "hull")

    # ── Flank vents and deck furniture ──────────────────────────────────────
    for sign in (1.0, -1.0):
        b.all(louvers(6, (0.70, 0.11), (tail + 1.35, sign * (hw + 0.01), 0.95), 0.16))
        b.all(bolts(11, (tail + 0.35, sign * (hw + 0.02), 0.10), (0.34, 0, 0), axis="Y"),
              "steel")
    for x, y in ((tail + 0.75, 0.62), (tail + 0.75, -0.62), (tail + 1.75, 0.0)):
        b.all(hatch((0.70, 0.62), (x, y, ph + 0.28)))
    b.all(handle(0.40, (tail + 2.55, 0.95, ph - 0.02)), "steel")
    b.all(handle(0.40, (tail + 2.55, -0.95, ph - 0.02)), "steel")

    # ── Yoke: heavy pivot collar + two thick trunnion arms ──────────────────
    # They belong to the pod, not the hull: they turn with it, and a pod that
    # floats over a gap is the classic tell of a rig nobody thought about.
    b.add(cyl(0.86, spec.collar_h + 0.10, (0, 0, -spec.collar_h / 2), segments=32),
          "steel")
    b.add(cyl(0.70, 0.22, (0, 0, 0.06), segments=32), "steel")
    b.all(bolt_ring(16, (0, 0, 0.14), 0.74, head=0.05, height=0.05), "steel")
    for sign in (1.0, -1.0):
        b.add(hexa([(-0.55, sign * 0.55, -0.10), (-0.55, sign * 1.05, -0.10),
                    (1.05, sign * 1.05, -0.10), (1.05, sign * 0.55, -0.10),
                    (-0.45, sign * 0.60, 0.62), (-0.45, sign * 0.98, 0.62),
                    (0.95, sign * 0.98, 0.62), (0.95, sign * 0.60, 0.62)]), "steel")
        b.add(cyl(0.30, 0.52, (0.72, sign * 0.80, 0.30), axis="Y", segments=16), "steel")

    pod = link_mesh("turret_pod", b.mesh(), colls["turret"], joints["turret_yaw"])

    # ── Whip antennas: 2.5 m, slight bend ───────────────────────────────────
    # Built as a chain of short tapering segments each tilted a little further
    # than the last, so the whip curves instead of kinking.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        a = Build(f"turret_antenna_{side}")
        base = Vector((tail + 0.55, sign * 0.92, spec.pod[2] + 0.28))
        a.add(cyl(0.075, 0.22, base + Vector((0, 0, 0.11)), segments=10), "steel")
        segments, cursor, tilt = 7, base + Vector((0, 0, 0.22)), spec.antenna_rake
        length = spec.antenna_len / segments
        for i in range(segments):
            angle = tilt * (0.45 + 0.75 * i / segments)     # gathers as it rises
            direction = Vector((math.sin(angle), 0, math.cos(angle)))
            r0 = 0.045 * (1 - i / segments) + 0.014
            r1 = 0.045 * (1 - (i + 1) / segments) + 0.012
            a.add(cyl(r0, length, cursor + direction * (length / 2), axis="Z",
                      r2=r1, segments=8,
                      rot=Matrix.Rotation(angle, 4, "Y")), "steel")
            cursor = cursor + direction * length
        link_mesh(f"turret_antenna_{side}", a.mesh(), colls["turret"], pod, bevel=0)

    # ── barrel_pitch hangs off the POD MESH, per the brief's hierarchy ──────
    barrel_pitch = empty("barrel_pitch", colls["turret"], pod,
                         location=(nose - 0.72, -0.42, 0.62), size=0.5)
    joints["barrel_pitch"] = barrel_pitch

    # A short forward sensor/optic barrel low on the nose — not ordnance.
    o = Build("barrel")
    o.add(cyl(0.30, 0.46, (0.16, 0, 0), axis="X", segments=24), "armor")
    o.all(bolt_ring(8, (0.36, 0, 0), 0.24, axis="X"), "steel")
    o.add(cyl(0.20, 0.72, (0.72, 0, 0), axis="X", segments=24), "steel")
    o.add(cyl(0.245, 0.14, (1.06, 0, 0), axis="X", segments=24), "armor")
    o.add(cyl(0.165, 0.05, (1.14, 0, 0), axis="X", segments=24), "amber")
    o.add(box((0.34, 0.30, 0.16), (0.45, 0, 0.28)), "armor")
    link_mesh("barrel", o.mesh(), colls["turret"], barrel_pitch)
    return pod


# ── Leg ─────────────────────────────────────────────────────────────────────
def build_leg(spec, leg, colls, joints):
    """One leg, authored once in the canonical frame and run four times.

    Canonical frame: origin on the joint, limb along +X. Local +Z on the shin
    points outboard once the stance is applied, which is why the shin guard
    lives on that face and the cable run hides on the opposite one.
    """
    coll = colls[f"leg_{leg}"]
    _, _, shin_angle = spec.stance

    # ── Coxa: rotator housing + armored cowl ────────────────────────────────
    c = Build(f"{leg}_coxa")
    c.add(cyl(spec.coxa_dia / 2, 1.25, segments=32), "steel")          # rotator
    c.add(cyl(spec.coxa_dia / 2 + 0.09, 0.34, (0, 0, 0.50), segments=32))
    c.add(cyl(spec.coxa_dia / 2 + 0.09, 0.34, (0, 0, -0.50), segments=32))
    c.all(bolt_ring(12, (0, 0, 0.68), 0.52), "steel")
    for i in range(8):                                                  # rotator ribs
        a = TAU * i / 8
        c.add(box((0.10, 0.10, 0.70),
                  (math.cos(a) * 0.58, math.sin(a) * 0.58, 0)), "steel")
    c.add(hexa([(0.10, 0.52, -0.48), (0.10, -0.52, -0.48),
                (spec.coxa_len + 0.08, -0.44, -0.40), (spec.coxa_len + 0.08, 0.44, -0.40),
                (0.10, 0.52, 0.48), (0.10, -0.52, 0.48),
                (spec.coxa_len + 0.08, -0.44, 0.42), (spec.coxa_len + 0.08, 0.44, 0.42)]))
    c.all(panel((0.52, 0.46), (0.55, 0.47, 0.02), normal="Y"))
    c.add(cyl(0.34, 1.05, (spec.coxa_len, 0, 0), axis="Y", segments=24), "steel")
    link_mesh(f"{leg}_coxa", c.mesh(), coll, joints[f"{leg}_coxa_yaw"])

    # ── Femur: structural beam + exposed hydraulic ram + cable bundle ───────
    f = Build(f"{leg}_femur")
    fl = spec.femur_len
    f.add(hexa([(0.05, 0.42, -0.46), (0.05, -0.42, -0.46),
                (fl - 0.05, -0.32, -0.34), (fl - 0.05, 0.32, -0.34),
                (0.05, 0.42, 0.46), (0.05, -0.42, 0.46),
                (fl - 0.05, -0.32, 0.36), (fl - 0.05, 0.32, 0.36)]))
    f.all(panel((1.05, 0.52), (fl * 0.42, 0.40, 0.0), normal="Y"))
    f.all(bolts(9, (0.35, 0.0, 0.44), (0.30, 0, 0)), "steel")
    f.add(cyl(0.40, 1.02, (0, 0, 0), axis="Y", segments=24), "steel")     # lift boss
    f.add(cyl(0.52, 1.02, (fl, 0, 0), axis="Y", segments=24))            # knee boss
    f.add(cyl(0.30, 1.12, (fl, 0, 0), axis="Y", segments=24), "steel")
    f.all(bolt_ring(10, (fl, 0.56, 0), 0.36, axis="Y"), "steel")
    # Hydraulic ram running alongside, above the beam: clevis, barrel, rod.
    f.add(box((0.22, 0.30, 0.22), (0.30, 0, 0.60)), "steel")
    f.add(cyl(0.17, 1.35, (1.15, 0, 0.62), axis="X", segments=20), "steel")
    f.add(cyl(0.20, 0.10, (0.50, 0, 0.62), axis="X", segments=20), "steel")
    f.add(cyl(0.20, 0.10, (1.80, 0, 0.62), axis="X", segments=20), "steel")
    f.add(cyl(0.085, 0.95, (2.30, 0, 0.62), axis="X", segments=16), "steel")
    f.add(box((0.20, 0.26, 0.26), (fl - 0.18, 0, 0.60)), "steel")
    # Cable bundle clipped to the inboard face.
    for i, dz in enumerate((-0.30, -0.38, -0.46)):
        f.add(cyl(0.045, fl * 0.78, (fl * 0.45, -0.34 - i * 0.015, dz), axis="X",
                  segments=8), "steel")
    for x in (0.75, 1.85, 2.65):
        f.add(box((0.10, 0.14, 0.30), (x, -0.36, -0.38)), "steel")
    link_mesh(f"{leg}_femur", f.mesh(), coll, joints[f"{leg}_femur_pitch"])

    # ── Tibia: tapered shin + guard plate + actuator + clawed foot ──────────
    t = Build(f"{leg}_tibia")
    tl = spec.tibia_len
    t.add(cyl(0.34, 1.02, (0, 0, 0), axis="Y", segments=24), "steel")     # knee boss
    t.add(hexa([(0.06, 0.34, -0.38), (0.06, -0.34, -0.38),
                (tl - 0.35, -0.19, -0.20), (tl - 0.35, 0.19, -0.20),
                (0.06, 0.34, 0.38), (0.06, -0.34, 0.38),
                (tl - 0.35, -0.19, 0.22), (tl - 0.35, 0.19, 0.22)]))
    t.add(cyl(0.19, 0.52, (tl - 0.12, 0, 0), axis="X", r2=0.16, segments=16), "steel")
    # Shin guard on the outboard face (+Z once posed), standing off the beam.
    t.add(hexa([(0.55, 0.30, 0.34), (0.55, -0.30, 0.34),
                (tl - 0.55, -0.22, 0.20), (tl - 0.55, 0.22, 0.20),
                (0.62, 0.26, 0.50), (0.62, -0.26, 0.50),
                (tl - 0.60, -0.19, 0.34), (tl - 0.60, 0.19, 0.34)]))
    t.all(bolts(6, (0.80, 0.0, 0.46), (0.34, 0, 0)), "steel")
    # Second, smaller actuator tucked behind the shin.
    t.add(box((0.18, 0.24, 0.18), (0.42, 0, -0.42)), "steel")
    t.add(cyl(0.115, 1.05, (1.05, 0, -0.44), axis="X", segments=16), "steel")
    t.add(cyl(0.062, 0.85, (1.95, 0, -0.44), axis="X", segments=12), "steel")
    t.add(box((0.16, 0.20, 0.20), (2.42, 0, -0.42)), "steel")

    # ── Foot: three splayed spiked toes plus a heel spur, treaded ───────────
    # Built world-level and counter-rotated by the shin's own pitch, so it
    # meets flat ground for whatever stance the Spec is solved to.
    level = Matrix.Rotation(-shin_angle, 4, "Y")
    origin = Vector((tl, 0, 0)) + level @ Vector((0, 0, -spec.ankle_z))

    def planted(mesh_fn, offset, spin=0.0):
        """Place a world-aligned piece at `offset` above the contact patch."""
        rot = level @ Matrix.Rotation(spin, 4, "Z")
        return mesh_fn(rot, origin + level @ Vector(offset))

    t.add(ball(0.30, (tl, 0, 0), subdivisions=2), "steel")               # ankle ball
    t.add(planted(lambda r, p: box((0.62, 0.86, 0.34), p, rot=r), (0, 0, 0.42)))
    t.add(planted(lambda r, p: box((1.02, 0.94, 0.16), p, rot=r), (0.05, 0, 0.20)))
    for spin_deg in (-38.0, 0.0, 38.0):                                  # toes
        t.add(planted(_toe, (0.34, 0, 0.15), math.radians(spin_deg)))
    t.add(planted(_heel_spur, (-0.42, 0, 0.16)))
    for i in range(4):                                                    # tread ribs
        t.add(planted(lambda r, p: box((0.13, 0.80, 0.09), p, rot=r),
                      (-0.24 + i * 0.20, 0, 0.045)), "steel")
    link_mesh(f"{leg}_tibia", t.mesh(), coll, joints[f"{leg}_tibia_pitch"])


def _toe(rot, at):
    """A splayed, spiked toe: wide at the ankle, tapering to a chisel point
    that digs in. Authored flat, placed by the caller's matrix."""
    verts = [(-0.16, 0.19, -0.15), (-0.16, -0.19, -0.15),
             (0.46, -0.075, -0.155), (0.46, 0.075, -0.155),
             (-0.16, 0.22, 0.17), (-0.16, -0.22, 0.17),
             (0.40, -0.085, -0.02), (0.40, 0.085, -0.02)]
    mesh = hexa(verts)
    mesh.transform(Matrix.Translation(at) @ rot)
    return mesh


def _heel_spur(rot, at):
    """A short rearward claw that stops the foot reading as a plate."""
    mesh = hexa([(0.10, 0.24, -0.16), (0.10, -0.24, -0.16),
                 (-0.46, -0.10, -0.14), (-0.46, 0.10, -0.14),
                 (0.10, 0.26, 0.22), (0.10, -0.26, 0.22),
                 (-0.40, -0.11, 0.04), (-0.40, 0.11, 0.04)])
    mesh.transform(Matrix.Translation(at) @ rot)
    return mesh


# ── Skeleton ────────────────────────────────────────────────────────────────
def build_skeleton(spec):
    """Every joint, nothing else. Returns the frames geometry hangs off."""
    colls = {key: collection(key) for key in
             ("chassis", "turret", "leg_FL", "leg_FR", "leg_RL", "leg_RR")}
    femur_a, knee_a, _ = spec.stance

    root = empty("root", colls["chassis"], size=1.0)
    joints = {"root": root}
    chassis_frame = empty("chassis_frame", colls["chassis"], root,
                          location=(0, 0, spec.hull_z), size=0.9)
    joints["chassis_frame"] = chassis_frame
    joints["turret_yaw"] = empty(
        "turret_yaw", colls["turret"], chassis_frame,
        location=(spec.yaw_x, 0, spec.deck_z + spec.collar_h - spec.hull_z), size=1.2)

    for leg, deg in spec.legs.items():
        sx = 1.0 if leg[0] == "F" else -1.0
        sy = 1.0 if leg[1] == "L" else -1.0
        coxa_yaw = empty(
            f"{leg}_coxa_yaw", colls[f"leg_{leg}"], chassis_frame,
            location=(sx * spec.hip_x, sy * spec.hip_y, spec.hip_z - spec.hull_z),
            rotation=(0, 0, math.radians(deg)), size=0.8)
        femur_pitch = empty(f"{leg}_femur_pitch", colls[f"leg_{leg}"], coxa_yaw,
                            location=(spec.coxa_len, 0, 0),
                            rotation=(0, femur_a, 0), size=0.7)
        tibia_pitch = empty(f"{leg}_tibia_pitch", colls[f"leg_{leg}"], femur_pitch,
                            location=(spec.femur_len, 0, 0),
                            rotation=(0, knee_a, 0), size=0.6)
        joints.update({f"{leg}_coxa_yaw": coxa_yaw,
                       f"{leg}_femur_pitch": femur_pitch,
                       f"{leg}_tibia_pitch": tibia_pitch})
    return colls, joints


def studio_world(scene):
    """A neutral grey dome, for LOOKING at the asset in the GUI.

    Not part of the design — make_walker_scene.py swaps in the night sky.
    It exists because the shaders below use Pointiness and Ambient Occlusion,
    which are Cycles features, and a Cycles material preview with no world is
    a black screen. Opening the asset should show the asset.
    """
    scene.render.engine = "CYCLES"
    world = bpy.data.worlds.new("studio")
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    background = nt.nodes.new("ShaderNodeBackground")
    background.location = (-200, 0)
    background.inputs["Color"].default_value = (0.22, 0.24, 0.28, 1.0)
    background.inputs["Strength"].default_value = 1.0
    nt.links.new(background.outputs["Background"], out.inputs["Surface"])
    scene.world = world

    key = bpy.data.lights.new("studio_key", type="AREA")
    key.energy = 4000.0
    key.size = 12.0
    node = bpy.data.objects.new("studio_key", key)
    node.location = (10.0, 12.0, 14.0)
    node.rotation_euler = (Vector((-10.0, -12.0, -14.0))
                           .to_track_quat("-Z", "Y").to_euler())
    scene.collection.objects.link(node)
    return world


def build_machine(spec):
    """The whole asset: materials, skeleton, then every link."""
    build_materials()
    colls, joints = build_skeleton(spec)
    build_chassis(spec, colls, joints)
    build_turret(spec, colls, joints)
    for leg in spec.legs:
        build_leg(spec, leg, colls, joints)
        build_slab(spec, leg, colls, joints)
    bpy.context.view_layer.update()
    return colls, joints


# ── Verification ────────────────────────────────────────────────────────────
def print_hierarchy(root, depth=0):
    for obj in sorted(root.children, key=lambda o: o.name):
        kind = "joint" if obj.type == "EMPTY" else " link"
        pos = obj.matrix_world.translation
        axis = ""
        if obj.type == "EMPTY":
            e = obj.rotation_euler
            axis = f"  rest[{math.degrees(e.y):+6.1f}Y {math.degrees(e.z):+6.1f}Z]"
        print(f"    {'  ' * depth}{kind} {obj.name:<22}"
              f"world({pos.x:+6.2f} {pos.y:+6.2f} {pos.z:+6.2f}){axis}")
        print_hierarchy(obj, depth + 1)


def report(spec, scene):
    """The numbers a proportion review actually turns on."""
    deps = bpy.context.evaluated_depsgraph_get()
    corners, tris = [], 0
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        corners += [obj.matrix_world @ v.co for v in mesh.vertices]
        mesh.calc_loop_triangles()
        tris += len(mesh.loop_triangles)
        evaluated.to_mesh_clear()
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                 min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                 max(c.z for c in corners)))
    femur_a, knee_a, shin_a = spec.stance
    tibia = bpy.data.objects.get("FL_tibia")

    print("\n  measured")
    print(f"    overall            {hi.x - lo.x:5.2f} L x {hi.y - lo.y:5.2f} W "
          f"x {hi.z - lo.z:5.2f} H   (brief: total height ~9.5)")
    print(f"    hull underside     {spec.clearance:5.2f}      deck {spec.deck_z:5.2f}"
          f"      pod top {spec.deck_z + spec.collar_h + spec.pod[2]:5.2f}")
    print(f"    ground contact     z = {lo.z:+.3f}  (feet planted at 0)")
    if tibia:
        knee = tibia.matrix_world.translation
        ankle = tibia.matrix_world @ Vector((spec.tibia_len, 0, 0))
        print(f"    FL knee            ({knee.x:+.2f}, {knee.y:+.2f}, {knee.z:+.2f})")
        print(f"    FL ankle           ({ankle.x:+.2f}, {ankle.y:+.2f}, {ankle.z:+.2f})"
              f"   stance {2 * abs(ankle.x):.1f} x {2 * abs(ankle.y):.1f}")
    print(f"    stance angles      femur {math.degrees(femur_a):+.1f} deg, "
          f"knee {math.degrees(knee_a):+.1f} deg  "
          f"(shin {math.degrees(shin_a):.0f} deg from horizontal)")
    print(f"    geometry           "
          f"{len([o for o in scene.objects if o.type == 'MESH'])} link objects, "
          f"{tris:,} triangles after bevel")
    print(f"    materials          {', '.join(sorted(MATERIALS))}")


# ── Clay check ──────────────────────────────────────────────────────────────
# Workbench, not Cycles: this pass exists to read FORM. Lighting comes later
# and in another file, and a noise-free matte in a second beats a denoised one
# in a minute when the only question is whether the greeble clusters land.
SHOTS = [
    ("clay_hero",  (30.5, 21.3, 2.20), (0, 0, 3.4), 50, (2000, 837)),
    ("clay_pod",   (9.6, 7.2, 6.60),   (1.2, 0, 5.6), 70, (1600, 1000)),
    ("clay_leg",   (12.4, 10.6, 3.60), (3.9, 3.2, 2.0), 85, (1600, 1000)),
    ("clay_front", (16.0, 0.0, 3.20),  (0, 0, 3.4), 50, (1400, 1200)),
]


def make_camera(name, eye, target, lens):
    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = eye
    cam.rotation_euler = (Vector(target) - Vector(eye)).to_track_quat("-Z", "Y").to_euler()
    return cam


def render_clay(scene, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.image_settings.file_format = "PNG"
    scene.display.render_aa = "16"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "SINGLE"
    shading.single_color = (0.44, 0.44, 0.46)
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.08, 0.09, 0.11)
    shading.show_shadows = True
    shading.shadow_intensity = 0.5
    shading.show_cavity = True
    shading.cavity_type = "BOTH"
    shading.curvature_ridge_factor = 1.6
    shading.curvature_valley_factor = 1.4
    shading.show_object_outline = False

    written = []
    for name, eye, target, lens, res in SHOTS:
        scene.camera = make_camera(f"cam_{name}", eye, target, lens)
        scene.render.resolution_x, scene.render.resolution_y = res
        scene.render.resolution_percentage = 100
        scene.render.filepath = os.path.abspath(os.path.join(out_dir, f"{name}.png"))
        bpy.ops.render.render(write_still=True)
        written.append(scene.render.filepath)
    return written


# ── Entry point ─────────────────────────────────────────────────────────────
def main():
    global CALLSIGN
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("skeleton", "build"), default="build")
    ap.add_argument("--render", action="store_true", help="write the clay pass")
    ap.add_argument("--callsign", default=CALLSIGN,
                    help="machine name: stencils, materials, output paths")
    ap.add_argument("--out", default=None)
    ap.add_argument("--renders", default=None)
    args = ap.parse_args(argv)

    CALLSIGN = args.callsign
    out = args.out or f"{asset_dir()}/walker.blend"
    renders = args.renders or f"{asset_dir()}/clay"

    scene = reset_scene()
    if args.stage == "skeleton":
        _, joints = build_skeleton(SPEC)
    else:
        _, joints = build_machine(SPEC)

    print(f"\n  {args.stage} hierarchy")
    print_hierarchy(joints["root"])
    if args.stage == "build":
        report(SPEC, scene)

    if args.render and args.stage == "build":
        print()
        for path in render_clay(scene, renders):
            print(f"    rendered {path}")

    if args.stage == "build":
        studio_world(scene)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
