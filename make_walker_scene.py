"""Light ANVIL in a night desert and render the hero frame (Part 6 of the brief).

    /Applications/Blender.app/Contents/MacOS/Blender --background \
        --python make_walker_scene.py -- --samples 512

    --preview      half resolution, few samples, for checking composition
    --samples N    Cycles sample cap (adaptive sampling usually stops earlier)
    --shot NAME    hero (the brief's 2.39:1 frame), detail, or prowl

make_walker.py is imported rather than duplicated, so the machine in this
frame is the same asset that file writes to assets/<callsign>/walker.blend —
one definition, two consumers. The machine's name comes from there too, so
--callsign on that file renames the renders here without a second edit.

The lighting brief in one line: warm practicals against cold ambient. Every
fixture on the machine is amber and they are the brightest things in frame;
everything else — sky, moon, the sand between the dunes — is blue-grey. That
temperature split is doing all the mood work, so nothing here is allowed to
put a warm light in the sky or a cool light on the machine.

Four things are built procedurally rather than loaded, because an HDRI or a
scanned rock would have to be committed as a binary blob and this repo keeps
its assets generatable:

    sky        a night gradient with a real star field and a Milky Way band
               arcing across the upper right. The band is a great circle: the
               mask is the dot product of the view direction with the pole of
               that circle, so it curves correctly across the whole dome
               instead of being a stripe painted on a backdrop.
    ground     120 x 120 m of displaced sand — dune-scale turbulence, then
               wind ripples at a ~1.25 m wavelength, then grain — over a
               coarse apron out to 350 m so the surface never visibly ends.
               Eased flat under the machine so the feet meet level ground.
               The ripples are ALSO a bump in the shader: at a 1.25 m eyeline
               the sand is grazed at about three degrees, where real relief
               projects to nothing and only shading survives.
    ridges     three layers of distant mountain silhouette as camera-facing
               cards, each a 1-D fractal skyline. They are EMISSION, not lit
               surfaces: at night a backdrop that depends on the moon just
               goes black, and atmospheric perspective is a value decision,
               not a lighting accident.
    fog        a shallow volume box at ankle height so the headlight beams
               have something to cut through and the dust reads.
"""

import argparse
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_walker as walker  # noqa: E402

# ── Framing (Part 6, with the pull-back agreed at the silhouette review) ────
# 50 mm and a 1.2 m eyeline as briefed. 22 m would frame 6.6 m of height
# against a 9.7 m machine, so the distance — and only the distance — moved.
LENS = 50.0
EYE_HEIGHT = 1.25
DISTANCE = 34.0
AZIMUTH = math.radians(33.0)          # three-quarter FRONT-LEFT
LOOK_AT = (0.0, 0.0, 4.60)
ASPECT = (2000, 837)                  # 2.39:1

GROUND_SIZE, GROUND_RES = 120.0, 520
SEED = 11


def camera_basis():
    """(view, right, up) for the hero camera, in world space.

    Anything whose placement is described in FRAME terms — "the band arcs
    across the upper right", "the moon rakes in from the upper left" — is
    derived from this rather than hand-typed as a rotation, so moving the
    camera cannot silently push the Milky Way out of shot.
    """
    view = -Vector((math.cos(AZIMUTH), math.sin(AZIMUTH), 0.0))
    up = Vector((0.0, 0.0, 1.0))
    right = view.cross(up).normalized()
    return view, right, up


# ── Sky ─────────────────────────────────────────────────────────────────────
def build_sky():
    """Night dome: cool gradient, star field, Milky Way band upper right."""
    world = bpy.data.worlds.new("night")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (900, 0)

    # The view direction. In a world shader this is the ray leaving the
    # camera, which is exactly the dome coordinate everything below wants.
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-1400, 0)
    direction = coord.outputs["Generated"]

    def vmath(op, a, b, loc, scale=None):
        node = nt.nodes.new("ShaderNodeVectorMath")
        node.operation = op
        node.location = loc
        for index, value in ((0, a), (1, b)):
            if value is None:
                continue
            if hasattr(value, "is_linked"):
                nt.links.new(value, node.inputs[index])
            else:
                node.inputs[index].default_value = value
        if scale is not None:
            node.inputs[3].default_value = scale
        return node

    def math(op, a, b, loc):
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = op
        node.location = loc
        for index, value in ((0, a), (1, b)):
            if value is None:              # ABSOLUTE and friends take one input
                continue
            if hasattr(value, "is_linked"):
                nt.links.new(value, node.inputs[index])
            else:
                node.inputs[index].default_value = value
        return node.outputs[0]

    # ── vertical gradient: deep zenith, faint warm-grey haze at the horizon ─
    up = nt.nodes.new("ShaderNodeSeparateXYZ")
    up.location = (-1200, -300)
    nt.links.new(direction, up.inputs[0])
    horizon = nt.nodes.new("ShaderNodeValToRGB")
    horizon.location = (-1000, -300)
    horizon.color_ramp.interpolation = "EASE"
    horizon.color_ramp.elements[0].position = 0.40
    horizon.color_ramp.elements[0].color = (0.020, 0.028, 0.048, 1.0)
    horizon.color_ramp.elements[1].position = 0.72
    horizon.color_ramp.elements[1].color = (0.003, 0.005, 0.013, 1.0)
    nt.links.new(up.outputs["Z"], horizon.inputs["Fac"])

    # ── Milky Way: a great circle, masked by distance from its plane ────────
    # dot(direction, pole) is 0 exactly on the band and +-1 at its poles, so
    # a narrow window around 0 is the band — and because it is defined on the
    # sphere it arcs naturally instead of running straight across frame.
    # The band has to CROSS the frame, and the pole is what decides that: a
    # pole equal to the view axis puts the band a full 90 degrees off-centre,
    # i.e. nowhere near a 17-degree frame. Aiming it perpendicular to a
    # diagonal through the upper right is what lands it in shot.
    view, right, up_axis = camera_basis()
    through = (view + right * 0.15 + up_axis * 0.12).normalized()
    along = (right * 0.7 + up_axis * 0.7).normalized()
    pole = vmath("DOT_PRODUCT", direction, tuple(through.cross(along).normalized()),
                 (-1000, 250))
    off_band = math("ABSOLUTE", pole.outputs["Value"], None, (-820, 250))
    band = nt.nodes.new("ShaderNodeValToRGB")
    band.location = (-640, 250)
    band.color_ramp.interpolation = "EASE"
    band.color_ramp.elements[0].position = 0.02
    band.color_ramp.elements[0].color = (1, 1, 1, 1)
    band.color_ramp.elements[1].position = 0.23
    band.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(off_band, band.inputs["Fac"])

    # Cloud structure inside the band, plus dust lanes cutting across it.
    clouds = nt.nodes.new("ShaderNodeTexNoise")
    clouds.location = (-1000, 520)
    clouds.inputs["Scale"].default_value = 3.2
    clouds.inputs["Detail"].default_value = 8.0
    clouds.inputs["Roughness"].default_value = 0.72
    nt.links.new(direction, clouds.inputs["Vector"])
    lanes = nt.nodes.new("ShaderNodeTexNoise")
    lanes.location = (-1000, 760)
    lanes.inputs["Scale"].default_value = 6.5
    lanes.inputs["Detail"].default_value = 6.0
    nt.links.new(direction, lanes.inputs["Vector"])
    lane_mask = math("ADD", math("MULTIPLY", lanes.outputs["Fac"], 0.85, (-820, 760)),
                     0.30, (-660, 760))
    milky = math("MULTIPLY", band.outputs["Color"], clouds.outputs["Fac"], (-450, 400))
    milky = math("MULTIPLY", milky, lane_mask, (-300, 400))
    milky = math("MULTIPLY", milky, 1.9, (-150, 400))

    # ── stars: Voronoi cell centres, thresholded hard, density varied ──────
    star_cells = nt.nodes.new("ShaderNodeTexVoronoi")
    star_cells.location = (-1000, 40)
    star_cells.feature = "F1"
    star_cells.inputs["Scale"].default_value = 260.0
    nt.links.new(direction, star_cells.inputs["Vector"])
    star_ramp = nt.nodes.new("ShaderNodeValToRGB")
    star_ramp.location = (-820, 40)
    star_ramp.color_ramp.elements[0].position = 0.0
    star_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)
    star_ramp.color_ramp.elements[1].position = 0.055
    star_ramp.color_ramp.elements[1].color = (0, 0, 0, 1)
    nt.links.new(star_cells.outputs["Distance"], star_ramp.inputs["Fac"])
    # Thin the field out so it is not a uniform sprinkle, and let the band
    # carry more of them — which is what actually makes a Milky Way read.
    thin = nt.nodes.new("ShaderNodeTexNoise")
    thin.location = (-1000, -80)
    thin.inputs["Scale"].default_value = 14.0
    nt.links.new(direction, thin.inputs["Vector"])
    density = math("ADD", math("MULTIPLY", thin.outputs["Fac"], 1.4, (-820, -120)),
                   0.25, (-660, -120))
    stars = math("MULTIPLY", star_ramp.outputs["Color"], density, (-450, 40))
    stars = math("MULTIPLY", stars,
                 math("ADD", math("MULTIPLY", band.outputs["Color"], 2.2, (-450, 150)),
                      0.55, (-300, 150)), (-150, 40))
    stars = math("MULTIPLY", stars, 12.0, (0, 40))

    # ── combine: gradient + band glow + stars ───────────────────────────────
    # This layer is ADDED to the gradient, so its "no band here" colour must be
    # BLACK. Anything else is a constant lift applied to the entire dome, which
    # both washes the night out and hides the band inside its own glow.
    band_color = walker._mix(nt, milky, (0.0, 0.0, 0.0), (0.255, 0.238, 0.280),
                          (150, 300))
    sky = nt.nodes.new("ShaderNodeMix")
    sky.data_type = "RGBA"
    sky.location = (350, 100)
    sky.inputs[walker.MIX_FAC].default_value = 1.0
    nt.links.new(horizon.outputs["Color"], sky.inputs[walker.MIX_A])
    nt.links.new(band_color, sky.inputs[walker.MIX_B])
    sky.blend_type = "ADD"

    total = nt.nodes.new("ShaderNodeMix")
    total.data_type = "RGBA"
    total.location = (550, 0)
    total.blend_type = "ADD"
    total.inputs[walker.MIX_FAC].default_value = 1.0
    nt.links.new(sky.outputs[walker.MIX_RESULT], total.inputs[walker.MIX_A])
    star_rgb = nt.nodes.new("ShaderNodeCombineColor")
    star_rgb.location = (350, -180)
    for channel, gain in zip(("Red", "Green", "Blue"), (0.92, 0.95, 1.0)):
        nt.links.new(math("MULTIPLY", stars, gain, (200, -180)),
                     star_rgb.inputs[channel])
    nt.links.new(star_rgb.outputs["Color"], total.inputs[walker.MIX_B])

    background = nt.nodes.new("ShaderNodeBackground")
    background.location = (720, 0)
    background.inputs["Strength"].default_value = 1.0
    nt.links.new(total.outputs[walker.MIX_RESULT], background.inputs["Color"])
    nt.links.new(background.outputs["Background"], out.inputs["Surface"])
    return world


# ── Ground ──────────────────────────────────────────────────────────────────
def sand_height(x, y):
    """Desert surface in meters: dunes, then wind ripples, then grain.

    Ripples are the layer that sells the scale — a 1.2 m wavelength read at a
    1.25 m eyeline is what tells the eye how big the machine standing in it
    is. Their phase is warped by low-frequency noise so the crests wander
    instead of marching in parallel.
    """
    p = Vector((x, y, 0.0))
    dunes = 0.55 * noise.turbulence(p * 0.035, 4, False)
    warp = 1.9 * noise.noise(p * 0.055)
    ripple_axis = (x * 0.94 + y * 0.34) * (math.tau / 1.25)
    ripples = 0.105 * math.sin(ripple_axis + warp)
    grain = 0.022 * noise.turbulence(p * 1.4, 3, True)
    height = dunes + ripples + grain

    # Settle the ground the machine stands on: the feet are authored at z=0,
    # so the sand under them is eased flat and then dropped a few centimetres
    # so the claws sit IN the surface rather than on it.
    radius = math.hypot(x, y)
    calm = min(max((radius - 6.5) / 12.0, 0.0), 1.0)
    calm = calm * calm * (3.0 - 2.0 * calm)
    return height * (0.42 + 0.58 * calm) - 0.05 * (1.0 - calm)


def build_ground():
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=GROUND_RES, y_segments=GROUND_RES,
                          size=GROUND_SIZE / 2.0)
    for vert in bm.verts:
        vert.co.z = sand_height(vert.co.x, vert.co.y)
    mesh = bpy.data.meshes.new("ground")
    bm.to_mesh(mesh)
    bm.free()

    mat = bpy.data.materials.new("sand")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-200, 0)
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.location = (-800, 0)
    grain.inputs["Scale"].default_value = 42.0
    grain.inputs["Detail"].default_value = 8.0
    color = walker._mix(nt, grain.outputs["Fac"], (0.090, 0.070, 0.046),
                     (0.150, 0.120, 0.082), (-520, 0))
    nt.links.new(color, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95

    # Wind ripples as BUMP, not just as geometry.
    #
    # The mesh really is rippled, but at a 1.25 m eyeline the sand is seen at
    # a three-degree grazing angle, where a 10 cm crest projects to almost no
    # screen height and disappears. A bump shades from the surface normal
    # instead of from screen-space relief, so it survives the grazing view —
    # and it is what actually tells the eye how big the machine standing in
    # this sand is. Anisotropic noise (stretched across the wind axis) gives
    # the wandering, quasi-parallel crests without a single sine wave.
    ripple_coords = nt.nodes.new("ShaderNodeTexCoord")
    ripple_coords.location = (-1500, -420)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-1300, -420)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(20.0))
    mapping.inputs["Scale"].default_value = (0.85, 7.5, 1.0)
    nt.links.new(ripple_coords.outputs["Object"], mapping.inputs["Vector"])
    ripples = nt.nodes.new("ShaderNodeTexNoise")
    ripples.location = (-1080, -420)
    ripples.inputs["Scale"].default_value = 1.25
    ripples.inputs["Detail"].default_value = 3.0
    ripples.inputs["Roughness"].default_value = 0.45
    nt.links.new(mapping.outputs["Vector"], ripples.inputs["Vector"])
    ripple_bump = nt.nodes.new("ShaderNodeBump")
    ripple_bump.location = (-700, -420)
    ripple_bump.inputs["Strength"].default_value = 0.85
    ripple_bump.inputs["Distance"].default_value = 0.09
    nt.links.new(ripples.outputs["Fac"], ripple_bump.inputs["Height"])

    coarse = nt.nodes.new("ShaderNodeTexNoise")
    coarse.location = (-1080, -760)
    coarse.inputs["Scale"].default_value = 26.0
    coarse.inputs["Detail"].default_value = 6.0
    grit = nt.nodes.new("ShaderNodeBump")
    grit.location = (-450, -600)
    grit.inputs["Strength"].default_value = 0.35
    grit.inputs["Distance"].default_value = 0.012
    nt.links.new(coarse.outputs["Fac"], grit.inputs["Height"])
    nt.links.new(ripple_bump.outputs["Normal"], grit.inputs["Normal"])
    nt.links.new(grit.outputs["Normal"], bsdf.inputs["Normal"])

    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mesh.materials.append(mat)

    obj = bpy.data.objects.new("ground", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.shade_smooth()
    return obj, mat


def build_apron(material):
    """Coarse sand out to 350 m, under the detailed ground.

    At a 1.25 m eyeline the true horizon never arrives: a 120 m plane simply
    ENDS about a degree below eye level, right inside frame, and the sky shows
    beneath it. The apron carries the surface out past the ridgelines with
    only the dune term, and is dropped 15 cm so the detailed ground overlaps
    it rather than fighting it for the same pixels along the seam.
    """
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=150, y_segments=150, size=350.0)
    for vert in bm.verts:
        p = Vector((vert.co.x, vert.co.y, 0.0))
        vert.co.z = 0.55 * noise.turbulence(p * 0.035, 3, False) - 0.15
    mesh = bpy.data.meshes.new("apron")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new("apron", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.shade_smooth()
    return obj


def stone_material():
    mat = bpy.data.materials.new("stone")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-200, 0)
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.location = (-800, 0)
    grain.inputs["Scale"].default_value = 18.0
    grain.inputs["Detail"].default_value = 6.0
    nt.links.new(walker._mix(nt, grain.outputs["Fac"], (0.030, 0.026, 0.021),
                          (0.062, 0.053, 0.043), (-520, 0)),
                 bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.92
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def scatter_rocks(material):
    """Rocks and a few foreground boulders — the other half of the scale cue.

    Placed with a seeded RNG so the frame is reproducible, kept out of the
    machine's own footprint, and deliberately clustered near the camera where
    a boulder can sit against a foot and say how big that foot is.
    """
    rng = random.Random(SEED)
    bm_all = bmesh.new()
    placed = 0

    def rock(size, at, squash, seed):
        """One weathered stone.

        Displacing each vertex by an INDEPENDENT random offset — the obvious
        first idea — produces a spiky crystal, because neighbouring vertices
        get uncorrelated pushes. Displacing along the normal by a smooth noise
        field keeps neighbours agreeing. But a smooth field at high subdivision
        goes too far the other way and yields river pebbles, so this stays at
        subdivision 2 with hard turbulence and FLAT shading: desert stone is
        fractured and angular, and the facets are the whole point.
        """
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=size)
        offset = Vector((seed * 3.7, seed * 1.9, seed * 6.1))
        for vert in bm.verts:
            lumps = noise.turbulence(vert.co * (2.1 / size) + offset, 3, True)
            vert.co += vert.normal * (size * 0.34 * lumps)
        bmesh.ops.transform(
            bm,
            matrix=(Matrix.Translation(at)
                    @ Matrix.Rotation(rng.uniform(0, math.tau), 4, "Z")
                    @ Matrix.Diagonal(Vector((1.0, rng.uniform(0.75, 1.3),
                                              squash, 1.0)))),
            verts=bm.verts,
        )
        temp = bpy.data.meshes.new("rock")
        bm.to_mesh(temp)
        bm.free()
        bm_all.from_mesh(temp)
        bpy.data.meshes.remove(temp)

    for i in range(430):
        angle = rng.uniform(0, math.tau)
        radius = rng.uniform(6.0, 48.0)
        x, y = math.cos(angle) * radius, math.sin(angle) * radius
        # Small and mostly buried: a field of head-sized stones reads as scale,
        # a field of car-sized ones just reads as rubble.
        size = (0.05 + 0.30 * rng.random() ** 3) * (
            1.0 + 0.55 * max(0.0, 1.0 - radius / 24.0))
        rock(size, (x, y, sand_height(x, y) + size * 0.18), 0.52, i)
        placed += 1

    # Foreground boulders, between camera and machine, for depth and scale.
    eye = Vector((math.cos(AZIMUTH), math.sin(AZIMUTH), 0)) * DISTANCE
    for i in range(10):
        t = rng.uniform(0.24, 0.62)
        base = eye * t + Vector((rng.uniform(-7, 7), rng.uniform(-7, 7), 0))
        size = rng.uniform(0.30, 0.70)
        rock(size, (base.x, base.y, sand_height(base.x, base.y) + size * 0.10),
             0.60, 100 + i)
        placed += 1

    mesh = bpy.data.meshes.new("rocks")
    bm_all.to_mesh(mesh)
    bm_all.free()
    mesh.materials.append(material)
    obj = bpy.data.objects.new("rocks", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return placed


# ── Distant ridgelines ──────────────────────────────────────────────────────
def build_ridges():
    """Three layered skyline cards with atmospheric perspective.

    Emission, not lit geometry: the value of a distant ridge at night is a
    compositional choice — it separates the machine from the sky — and making
    it depend on a dim moon would just render it black.
    """
    rng = random.Random(SEED + 3)
    view = Vector((math.cos(AZIMUTH), math.sin(AZIMUTH), 0.0))
    right = Vector((-view.y, view.x, 0.0))

    for layer, (distance, height, value) in enumerate((
        (150.0, 26.0, (0.0045, 0.0060, 0.0100)),
        (240.0, 42.0, (0.0080, 0.0105, 0.0165)),
        (360.0, 62.0, (0.0130, 0.0165, 0.0240)),
    )):
        span, steps = distance * 2.6, 220
        points = []
        base_phase = rng.uniform(0, 100)
        for i in range(steps + 1):
            t = i / steps
            u = (t - 0.5) * span
            profile = (noise.turbulence(Vector((u * 0.004 + base_phase, layer * 7.3, 0)),
                                        5, False) * 0.5 + 0.5)
            ridge = (noise.turbulence(Vector((u * 0.013 + base_phase, layer * 3.1, 0)),
                                      3, True) * 0.5 + 0.5)
            points.append((u, height * (0.30 + 0.70 * profile) * (0.65 + 0.45 * ridge)))

        bm = bmesh.new()
        bottom, top = [], []
        for u, h in points:
            base = -view * distance + right * u
            bottom.append(bm.verts.new((base.x, base.y, -20.0)))
            top.append(bm.verts.new((base.x, base.y, h)))
        for i in range(len(points) - 1):
            bm.faces.new((bottom[i], bottom[i + 1], top[i + 1], top[i]))
        mesh = bpy.data.meshes.new(f"ridge_{layer}")
        bm.to_mesh(mesh)
        bm.free()

        mat = bpy.data.materials.new(f"ridge_{layer}")
        if mat.node_tree is None:
            mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.location = (-200, 0)
        emit.inputs["Color"].default_value = (*value, 1.0)
        emit.inputs["Strength"].default_value = 1.0
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        mesh.materials.append(mat)
        obj = bpy.data.objects.new(f"ridge_{layer}", mesh)
        bpy.context.scene.collection.objects.link(obj)


# ── Atmosphere ──────────────────────────────────────────────────────────────
def build_fog():
    """A shallow volume at ankle height: it makes the headlight beams visible
    and puts the dust the brief asks for around the feet. Kept as a thin box
    rather than world volume so the render cost stays where the camera is."""
    mesh = bpy.data.meshes.new("fog")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    # Ankle height means ANKLE height: the top of this box sits below the
    # machine's belly, so the haze pools around the feet and the legs stay
    # readable above it. A taller box just fogs the whole subject.
    bmesh.ops.transform(bm, matrix=Matrix.Translation((0, 0, 1.35))
                        @ Matrix.Diagonal(Vector((150.0, 150.0, 3.6, 1.0))),
                        verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()

    mat = bpy.data.materials.new("fog")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    volume = nt.nodes.new("ShaderNodeVolumePrincipled")
    volume.location = (-200, 0)
    volume.inputs["Color"].default_value = (0.62, 0.60, 0.58, 1.0)
    volume.inputs["Anisotropy"].default_value = 0.35

    # Density falls off with height instead of stopping at the box lid. A
    # constant-density box draws a dead-straight horizontal edge across the
    # frame wherever its top is — the one thing real ground haze never does.
    # Broken up by noise so the dust drifts rather than lies in a sheet.
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-1000, -200)
    axis = nt.nodes.new("ShaderNodeSeparateXYZ")
    axis.location = (-820, -200)
    nt.links.new(geo.outputs["Position"], axis.inputs[0])
    falloff = nt.nodes.new("ShaderNodeMapRange")
    falloff.location = (-640, -200)
    falloff.inputs[1].default_value = 2.30       # gone by here
    falloff.inputs[2].default_value = 0.05       # thickest at the sand
    falloff.clamp = True
    nt.links.new(axis.outputs["Z"], falloff.inputs[0])
    drift = nt.nodes.new("ShaderNodeTexNoise")
    drift.location = (-820, -480)
    drift.inputs["Scale"].default_value = 0.09
    drift.inputs["Detail"].default_value = 4.0
    varied = nt.nodes.new("ShaderNodeMath")
    varied.operation = "MULTIPLY_ADD"
    varied.location = (-440, -400)
    varied.inputs[1].default_value = 0.9
    varied.inputs[2].default_value = 0.45
    nt.links.new(drift.outputs["Fac"], varied.inputs[0])
    density = nt.nodes.new("ShaderNodeMath")
    density.operation = "MULTIPLY"
    density.location = (-260, -260)
    nt.links.new(falloff.outputs[0], density.inputs[0])
    nt.links.new(varied.outputs[0], density.inputs[1])
    scaled = nt.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.location = (-110, -260)
    scaled.inputs[1].default_value = 0.018
    nt.links.new(density.outputs[0], scaled.inputs[0])
    nt.links.new(scaled.outputs[0], volume.inputs["Density"])
    nt.links.new(volume.outputs["Volume"], out.inputs["Volume"])
    mesh.materials.append(mat)
    obj = bpy.data.objects.new("fog", mesh)
    obj.visible_shadow = False
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ── Light ───────────────────────────────────────────────────────────────────
def build_lights(spec):
    """A dim cold moon, and the machine's own warm practicals.

    The moon exists only to rim the top surfaces and lift the machine off the
    sky. Everything the eye reads as light in this frame comes from the
    machine itself, which is why the headlights are real spot lamps and not
    just emissive discs: an emissive disc lights nothing.
    """
    # A lamp shines down its local -Z, so aiming is a track-quat from a
    # direction — never a hand-typed Euler. The first pass at this pointed the
    # headlights out of the BACK of the machine, which is invisible in a wire
    # view and obvious the moment fog is added.
    def aim(obj, direction):
        obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
        return obj

    view, right, up_axis = camera_basis()

    moon = bpy.data.lights.new("moon", type="SUN")
    moon.energy = 3.1
    moon.color = (0.56, 0.68, 1.0)
    moon.angle = math.radians(1.2)
    obj = bpy.data.objects.new("moon", moon)
    bpy.context.scene.collection.objects.link(obj)
    # Upper LEFT of frame: the moon sits off screen-left and high, so its
    # light travels toward screen-right and down — a rake across the top
    # surfaces that separates the machine from the sky without filling it.
    aim(obj, right * 0.80 - up_axis * 0.40 + view * 0.14)

    warm = (1.0, 0.60, 0.24)
    hull_front = spec.hull[0] / 2
    for sign in (1.0, -1.0):
        light = bpy.data.lights.new(f"headlight_{sign:+.0f}", type="SPOT")
        light.energy = 2600.0
        light.color = warm
        light.spot_size = math.radians(56.0)
        light.spot_blend = 0.55
        light.shadow_soft_size = 0.16
        node = bpy.data.objects.new(f"headlight_{sign:+.0f}", light)
        bpy.context.scene.collection.objects.link(node)
        node.location = (hull_front + 0.25, sign * 0.92, spec.clearance + 1.06)
        aim(node, (0.94, sign * 0.10, -0.33))    # forward and down onto the sand

    # Local warm spill so the amber strips actually paint the armor under
    # them — the brief asks for the practicals to fall on the machine's own
    # lower plates, and emission alone will not carry that far.
    for name, position, energy in (
        ("spill_pod", (2.10, 0.0, 5.30), 820.0),
        ("spill_hull", (2.70, 0.0, 3.10), 330.0),
        ("spill_markers", (2.40, 1.62, 2.95), 320.0),
        ("spill_markers_r", (2.40, -1.62, 2.95), 320.0),
        ("spill_belly", (1.20, 0.0, 2.10), 420.0),
        ("spill_legs_l", (2.20, 2.90, 2.30), 260.0),
        ("spill_legs_r", (2.20, -2.90, 2.30), 260.0),
    ):
        light = bpy.data.lights.new(name, type="POINT")
        light.energy = energy
        light.color = warm
        light.shadow_soft_size = 0.45
        node = bpy.data.objects.new(name, light)
        node.location = position
        bpy.context.scene.collection.objects.link(node)


# ── Camera, render, output ──────────────────────────────────────────────────
# Alternate framings. The hero is the brief's shot; the others exist to show
# the asset holds up when the camera walks in, which a 2.39:1 wide never
# proves on its own.
SHOTS = {
    #        lens  distance  eye z  azimuth  look_at              resolution
    "hero":   (LENS, DISTANCE, EYE_HEIGHT, 33.0, LOOK_AT,          (2000, 837)),
    "detail": (85.0, 20.5, 2.55, 17.0, (2.35, 0.75, 4.15),         (1800, 1125)),
    "prowl":  (35.0, 17.0, 1.05, 74.0, (-0.4, 0.0, 3.60),          (1800, 900)),
}


def build_camera(shot="hero"):
    lens, distance, eye_z, azimuth, look_at, _ = SHOTS[shot]
    data = bpy.data.cameras.new(shot)
    data.lens = lens
    data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new(shot, data)
    bpy.context.scene.collection.objects.link(cam)
    angle = math.radians(azimuth)
    eye = Vector((math.cos(angle) * distance, math.sin(angle) * distance, eye_z))
    cam.location = eye
    cam.rotation_euler = (Vector(look_at) - eye).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def _blur1d(a, radius, axis):
    """Box blur along one axis via a running sum — O(n) regardless of radius."""
    import numpy as np

    width = 2 * radius + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    running = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    count = running.shape[axis]
    hi = np.take(running, np.arange(width, count), axis=axis)
    lo = np.take(running, np.arange(0, count - width), axis=axis)
    return (hi - lo) / width


def post_bloom(path, threshold=0.62, strength=0.42, radius=14):
    """Subtle bloom, applied to the saved frame.

    Blender 5 replaced the scene compositor with a node group, and wiring a
    Glare into it here produced a white frame and a render that finished in
    1.6 seconds — the compositor was answering instead of Cycles. Rather than
    ship a fragile graph, the glow is done explicitly: pull the highlights,
    blur them with three box passes (a close enough Gaussian), and screen them
    back over the frame. It runs on the display-referred image, which is what
    a glare node is doing to your eye anyway.
    """
    import numpy as np

    image = bpy.data.images.load(path)
    image.colorspace_settings.name = "Non-Color"   # operate on display values
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    frame = buffer.reshape(height, width, 4)

    rgb = frame[..., :3]
    peak = rgb.max(axis=-1)
    mask = np.clip((peak - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    glow = rgb * mask[..., None]
    for _ in range(3):
        glow = _blur1d(_blur1d(glow, radius, 1), radius, 0)

    frame[..., :3] = np.clip(rgb + strength * glow, 0.0, 1.0)
    image.pixels.foreach_set(frame.reshape(-1))
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def configure_render(scene, samples, preview, resolution):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.012
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.transmission_bounces = 4
    scene.cycles.volume_bounces = 2
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.sample_clamp_indirect = 12.0
    scene.cycles.film_exposure = 1.25

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        devices = prefs.get_devices_for_type("METAL")
        for device in devices:
            device.use = True
        scene.cycles.device = "GPU"
        print(f"    GPU: {', '.join(d.name for d in devices if d.use)}")
    except Exception as exc:
        scene.cycles.device = "CPU"
        print(f"    (CPU fallback: {type(exc).__name__}: {exc})")

    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 50 if preview else 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.view_settings.exposure = 0.0


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-render", action="store_true",
                    help="build and save the .blend without rendering")
    ap.add_argument("--shot", choices=tuple(SHOTS), default="hero")
    ap.add_argument("--out", default=None)
    ap.add_argument("--blend", default=None)
    args = ap.parse_args(argv)
    blend = args.blend or f"{walker.asset_dir()}/scene.blend"

    scene = walker.reset_scene()
    walker.build_machine(walker.SPEC)
    print("    machine built")

    build_sky()
    ground, sand = build_ground()
    build_apron(sand)
    rocks = scatter_rocks(stone_material())
    build_ridges()
    build_fog()
    build_lights(walker.SPEC)
    build_camera(args.shot)
    print(f"    scene: ground {GROUND_RES}^2, {rocks} rocks, 3 ridge layers")

    configure_render(scene, args.samples, args.preview, SHOTS[args.shot][5])
    bpy.context.view_layer.update()

    os.makedirs(os.path.dirname(os.path.abspath(blend)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blend))

    out_path = args.out or f"{walker.asset_dir()}/{args.shot}.png"
    scene.render.filepath = os.path.abspath(out_path)
    if args.no_render:
        print(f"    built {blend} on the {args.shot} camera (no render)")
        return
    print(f"    rendering {args.shot} at {args.samples} samples, "
          f"{scene.render.resolution_percentage}% -> {out_path}")
    bpy.ops.render.render(write_still=True)
    post_bloom(os.path.abspath(out_path))
    print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
