// wheel.scad — the WHELP-16 wheel: a PETG hub and a TPU 95A tire.
//
// Two printed parts, one interface between them, and that interface is the
// whole design problem. Everything else here is mass.
//
// FRAME
//   Origin is the WHEEL AXIS. The axis runs along Y, matching the robot frame
//   (+X forward, +Y left, +Z up). The wheel occupies y in [-WHEEL_W/2, +WHEEL_W/2].
//
// SIDE
//   Authored for the LEFT leg: the drive servo is INBOARD, on -Y, so the horn
//   pad is the -Y face. The assembly mirrors in Y for the right side.
//
//   Both parts are bodies of revolution about the wheel axis apart from the
//   spoke slots and the horn bolt circle, and BOTH of those patterns are mirror
//   symmetric about a plane containing the axis. That is not an accident and it
//   is worth keeping: it means the mirrored hub is the same solid as the hub
//   rotated 180 degrees about X, so ONE hub STL and ONE tire STL serve all four
//   wheels. Add a handed feature -- a spiral spoke, a three-bolt pattern, an
//   arrow in the tread -- and the print list doubles.
//
// PRINT ORIENTATION (both parts: WHEEL AXIS VERTICAL)
//
//   hub   axis vertical, HORN PAD FACE DOWN, i.e. exactly as wheel_hub_print()
//         is authored. The web is then the first layer -- a solid disc, no
//         bridging anywhere in the part -- and the rim's hoop stress and the
//         spokes' torque shear both run along extrusion lines. Turn it the
//         other way up and the web becomes a 15 mm annular bridge.
//
//   tire  axis vertical, i.e. the annulus lies FLAT on the bed, exactly as
//         tire_print() is authored. This one is not negotiable. The tire's
//         working stress is hoop tension plus the shear that drags each tread
//         block backwards, and printed flat both of those run in-plane.
//         Printed on its side instead, every tread block hangs off an
//         interlayer weld: UltiMaker measure TPU 95A at 23.7 MPa in XY and
//         6.4 MPa in Z, so the same tire loses roughly three quarters of its
//         strength exactly where the ground touches it.
//
// To preview a part directly in OpenSCAD, uncomment one of:
//   wheel_hub();
//   tire();

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

// util.scad's EPS is not visible here: `use` imports modules and functions, not
// variables. Same value, same reason -- larger than float noise, smaller than
// any printable feature, so overshooting a cut by it can never change a fit.
EPS = 0.02;

// Set so that opening this file renders at the same facet count the exporter
// uses. A wheel is the part where a low $fn is most visible and most
// misleading: an under-facetted "circle" measures undersize across the flats.
$fn = FN;

// ─────────────────────────────────────────────────────────────────────────────
// THE NUMBERS THIS PART ADDS ON TOP OF THE SPEC
//
// Everything dimensional comes from params_gen.scad. What follows are the few
// quantities that exist only inside this part, each with the reason it is what
// it is. None of them may be edited into the generated files.
// ─────────────────────────────────────────────────────────────────────────────

HUB_OUTER_R  = WHEEL_R - TIRE_T;        // the tire seat: 12 mm of TPU to the ground
RIM_WALL     = CALF_WALL;               // the wheel hangs off the calf and is sized
                                        // like it; there is no WHEEL_WALL in the
                                        // spec and inventing one here would put a
                                        // number outside spec.py's provenance
RIM_INNER_R  = HUB_OUTER_R - RIM_WALL;

WEB_T        = CALF_WALL;               // the spoked disc, same wall standard
BOSS_R       = 15;                      // clears the horn bolt circle by 5.3 mm:
                                        // SERVO_HORN_BOLT_CIRCLE_D/2 plus half a
                                        // bolt hole reaches 9.7 mm, and the boss
                                        // has to have meat left to bear against
BOSS_H       = HORN_BOSS_THICK;         // bolt BEARING area is diameter x depth, and
                                        // this constant exists for exactly that

LIP_H        = 2.5;                     // radial height of each retention lip. Deep
                                        // enough that the tire must stretch ~7% of
                                        // its bore to climb out, shallow enough that
                                        // it still leaves 9.5 mm of TPU over the lip
LIP_LAND     = 3.0;                     // axial length of each lip at full radius
LIP_OUTER_R  = HUB_OUTER_R + LIP_H;
EDGE_BREAK   = 1.0;                     // 45 degree break on every external edge that
                                        // touches the bed or that the tire slides over

// Spoke lightening. See hub_lightening_slots() for why these are aggressive.
SPOKE_COUNT   = 5;
LH_R          = 4.0;                    // end radius of each stadium-shaped slot
LH_CENTRE_R   = 22.5;                   // slot centreline radius
LH_HALF_ANG   = 16;                     // half the slot's angular length, degrees

// Tire.
TIRE_FIT      = 0.3;                    // RADIAL interference: the tire's bore is
                                        // authored 0.3 mm small everywhere, so it
                                        // goes on stretched (~1% pre-strain) and
                                        // grips the hub instead of rattling on it
SHOULDER_CH   = 4.0;                    // shoulder break -- see tire_print()
TREAD_DEPTH   = 0.8;
TREAD_R       = 1.25;                   // cutter radius; sets the groove's rounded
                                        // root and its 2.3 mm width at the surface
TREAD_COUNT   = 24;
TREAD_INSET   = 2.0;                    // how far short of the shoulder each groove
                                        // stops. Load-bearing: see tire_tread_cuts()

// The fillet at the web/boss corner cannot be FILLET_R, because the boss stands
// only BOSS_H - WEB_T proud of the web. A fillet taller than the wall it
// fillets is not a fillet, it is a floating ring of plastic above a face the
// horn bolt heads have to reach.
BOSS_FILLET_R = BOSS_H - WEB_T;

// ─────────────────────────────────────────────────────────────────────────────
// CHECKS
//
// These fire at render, before four hours of printing. Every one of them is a
// way a regenerated params_gen.scad could quietly make this part wrong: the
// spec is edited in spec.py and exported, and nothing downstream re-derives the
// local numbers above.
// ─────────────────────────────────────────────────────────────────────────────

module wheel_checks() {
    assert(TIRE_T < WHEEL_R,
           str("TIRE_T ", TIRE_T, " mm is not less than WHEEL_R ", WHEEL_R,
               " mm; the hub would have zero or negative radius"));
    assert(RIM_INNER_R > BOSS_R + FILLET_R,
           str("hub rim inner radius ", RIM_INNER_R,
               " mm leaves no web between the boss and the rim"));
    assert(BOSS_FILLET_R > 0,
           str("HORN_BOSS_THICK ", HORN_BOSS_THICK, " mm is not thicker than the web ",
               WEB_T, " mm; the horn boss does not stand proud and has nothing to fillet"));
    assert(SERVO_HORN_BOLT_CIRCLE_D / 2 + SERVO_HORN_BOLT_CLEAR_D / 2 < BOSS_R,
           str("horn bolt circle reaches ",
               SERVO_HORN_BOLT_CIRCLE_D / 2 + SERVO_HORN_BOLT_CLEAR_D / 2,
               " mm, outside the ", BOSS_R, " mm boss; the bolts would break out"));
    assert(WHEEL_W > 2 * (LIP_LAND + LIP_H),
           str("WHEEL_W ", WHEEL_W, " mm leaves no tire seat between the two lips"));
    assert(LH_CENTRE_R + LH_R < RIM_INNER_R - FILLET_R,
           str("lightening slots reach ", LH_CENTRE_R + LH_R,
               " mm and would cut into the rim fillet"));
    assert(LH_CENTRE_R - LH_R > BOSS_R + BOSS_FILLET_R,
           str("lightening slots reach in to ", LH_CENTRE_R - LH_R,
               " mm and would cut into the horn boss fillet"));
    assert(LIP_H < TIRE_T / 2,
           str("retention lip ", LIP_H, " mm is more than half the tire section ",
               TIRE_T, " mm; the tire would be thinner over the lip than beside it"));
    assert(2 * WHEEL_R <= min(BED_X, BED_Y),
           str("wheel diameter ", 2 * WHEEL_R, " mm does not fit the ", BED_X, "x", BED_Y,
               " mm bed in the declared axis-vertical orientation"));
    assert(WHEEL_W <= BED_Z,
           str("wheel width ", WHEEL_W, " mm exceeds the ", BED_Z, " mm Z envelope"));
    assert(TREAD_DEPTH < TIRE_T / 4,
           str("tread depth ", TREAD_DEPTH, " mm is a large fraction of the ", TIRE_T,
               " mm section; shallow tread is a transfer decision, not a styling one"));
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED 2D PIECES
// ─────────────────────────────────────────────────────────────────────────────

// The material to ADD at a 90-degree internal corner, as a 2D cross-section, so
// it can be swept by rotate_extrude() into a ring fillet.
//
// util.scad's fillet_in() is the linear_extrude version of the same shape; a
// wheel's internal corners are circles, not lines, so they need the swept form.
// The corner sits at the origin; the two faces run along +x and +y; the concave
// side -- the void being filled -- is toward (r, r).
module fillet2d_out(r) {
    difference() {
        square([r, r]);
        translate([r, r]) circle(r = r);
    }
}

// Mirror image of the above: faces along -x and +y, void toward (-r, r).
module fillet2d_in(r) {
    difference() {
        translate([-r, 0]) square([r, r]);
        translate([-r, r]) circle(r = r);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HUB
// ─────────────────────────────────────────────────────────────────────────────

// The hub's body of revolution: horn boss, web, rim, and both retention lips,
// as one profile so the wall thicknesses cannot drift apart from each other.
//
// Points are [radius, z] and the profile is traced counter-clockwise: out along
// the bed face, up the outside past both lips, in across the rim's top face,
// down the rim's bore, in across the web, up the boss and back to the axis.
//
// THE RETENTION CHANNEL is the four points between z = LIP_LAND and
// z = WHEEL_W - LIP_LAND. A press-on TPU ring with nothing but friction holding
// it comes off the first time the robot slides sideways into a table leg, and
// it comes off at the moment the wheel is already loaded laterally. So the tire
// sits in a channel between two proud lips and has to stretch LIP_H radially --
// about 7% hoop strain on a 33 mm bore -- to escape in either direction.
//
// Both walls of that channel are 45-degree ramps rather than square shoulders.
// A square shoulder would retain better, but the upper one would be a fully
// horizontal downward face on a part printed axis-vertical, which is the one
// overhang FDM cannot do at all. 45 degrees is the steepest wall that still
// prints without support, on this part AND on the tire's matching bore, and a
// retention feature that needs support scars on its sealing surface is not a
// retention feature.
//
// The two channel roots are the only unfilleted internal corners in this part.
// They are 135-degree vertices rather than 90-degree ones -- the ramps already
// halve the corner's severity -- and a FILLET_R = 3 mm radius simply does not
// fit inside a 3 mm rim wall. The omission is dimensional, not an oversight.
module hub_profile() {
    rotate_extrude()
        polygon(points = [
            [0,                          0],
            [LIP_OUTER_R - EDGE_BREAK,   0],                              // bed face
            [LIP_OUTER_R,                EDGE_BREAK],                     // elephant-foot break
            [LIP_OUTER_R,                LIP_LAND],                       // inboard lip
            [HUB_OUTER_R,                LIP_LAND + LIP_H],               // ramp into the channel
            [HUB_OUTER_R,                WHEEL_W - LIP_LAND - LIP_H],     // the tire seat
            [LIP_OUTER_R,                WHEEL_W - LIP_LAND],             // ramp out, 45 deg
            [LIP_OUTER_R,                WHEEL_W - EDGE_BREAK],           // outboard lip
            [LIP_OUTER_R - EDGE_BREAK,   WHEEL_W],
            [RIM_INNER_R,                WHEEL_W],                        // rim top face
            [RIM_INNER_R,                WEB_T],                          // rim bore
            [BOSS_R,                     WEB_T],                          // web top face
            [BOSS_R,                     BOSS_H],                         // horn boss wall
            [0,                          BOSS_H]
        ]);
}

// Ring fillets at the two internal corners the profile above leaves square.
//
// Both are load paths, not cosmetics. The rim-to-web corner carries every
// landing: the ground pushes the rim, the rim twists about its junction with
// the web, and that junction is where the part breaks if it is square. A 0.3 mm
// as-modelled corner runs a stress concentration factor of 2.99 in bending
// against 1.48 for a 3 mm fillet, so leaving it out roughly halves fatigue life
// for zero saved mass.
module hub_fillets() {
    // Rim bore meets the web's top face. Void is inboard and above.
    rotate_extrude()
        translate([RIM_INNER_R, WEB_T])
            fillet2d_in(FILLET_R);

    // Horn boss wall meets the web's top face. Void is outboard and above.
    rotate_extrude()
        translate([BOSS_R, WEB_T])
            fillet2d_out(BOSS_FILLET_R);
}

// The spoke lightening.
//
// This is UNSPRUNG MASS at the far end of the leg, and a gram here is not worth
// a gram anywhere else on the robot. It is accelerated by every bump instead of
// by the gait, so it sets how hard the ground hits back; it sits at the end of
// a 210 mm lever from the hip, so it costs the hip torque as the square of its
// distance; and none of that shows up as a number the policy can see -- it
// shows up as a leg that cannot follow the terrain it is standing on. The web
// is therefore cut to about a third of the annulus it started as. The same
// treatment applied to the trunk would save more grams and matter less.
//
// Stadium slots, not rectangles: the corner radius is the whole feature. A
// square lightening hole in a member that carries torque is a crack starter,
// and the mass saved is lost the first time the wheel is shock-loaded.
// SPOKE_COUNT and LH_HALF_ANG are set so the narrowest remaining spoke is
// ~7.4 mm across -- more than twice the wall standard -- and wheel_checks()
// asserts the slots stay clear of both fillets.
module hub_lightening_slots() {
    for (i = [0 : SPOKE_COUNT - 1])
        rotate([0, 0, i * 360 / SPOKE_COUNT])
            hull()
                for (s = [-1, 1])
                    rotate([0, 0, s * LH_HALF_ANG])
                        translate([LH_CENTRE_R, 0, -EPS])
                            cylinder(r = LH_R, h = WEB_T + 2 * EPS);
}

// Everything cut into the horn boss.
//
// The bolt pattern comes from sts3215.scad rather than being re-typed, because
// this is the interface most likely to be wrong in a way that only shows up
// after printing, and there is exactly one place it should be described.
//
// The central relief is taken all the way THROUGH rather than blind. The four
// horn bolts carry all of the drive torque and the centre carries none, so the
// 1 mm of plastic a blind pocket would leave over the horn's own retaining
// screw is not structure -- it is an unsupported roof over a hole, printed as
// the very first layers, in the exact spot a dropped wheel lands. Opening it
// also means the horn's retaining screw can be checked without pulling the
// wheel, and grit that gets past the horn falls out instead of collecting.
module hub_horn_cuts() {
    sts3215_horn_holes(BOSS_H);

    translate([0, 0, -EPS])
        cylinder(d = SERVO_HORN_HUB_CLEAR_D, h = BOSS_H + 2 * EPS);

    // Lead-in on the bore's bed-side edge. This edge, not the rim, is what
    // matters for seating: the face that beds against the horn is only the
    // annulus the horn actually touches, and its inner boundary is here. First
    // layers spread 0.1-0.3 mm into the bore, and a lip of squished plastic
    // there holds the hub off the horn by exactly the amount that becomes
    // wobble. util.scad's elephant_relief() is not used because its footprint
    // is rectangular; on a disc it would break four tangent points and nothing
    // else. The rim's own bed edge is broken in hub_profile() instead.
    translate([0, 0, -EPS])
        cylinder(d1 = SERVO_HORN_HUB_CLEAR_D + 2 * EDGE_BREAK,
                 d2 = SERVO_HORN_HUB_CLEAR_D,
                 h = EDGE_BREAK + EPS);
}

// The hub in its PRINT frame: axis along +Z, horn pad face on the bed at z = 0.
//
// z = 0 is the face that beds against the OUTER face of the servo horn, which
// is the frame sts3215_horn_holes() is authored in, so the horn interface needs
// no transform of its own here.
module wheel_hub_print() {
    wheel_checks();

    difference() {
        union() {
            hub_profile();
            hub_fillets();
        }
        hub_lightening_slots();
        hub_horn_cuts();
    }
}

// The hub in the ASSEMBLY frame: wheel axis along Y, horn pad on the -Y face,
// because on a left leg the drive servo is inboard.
//
// PETG. Print as wheel_hub_print() is authored -- axis vertical, pad face down.
module wheel_hub() {
    translate([0, -WHEEL_W / 2, 0])
        rotate([-90, 0, 0])
            wheel_hub_print();
}

// ─────────────────────────────────────────────────────────────────────────────
// TIRE
// ─────────────────────────────────────────────────────────────────────────────

// The tire's body of revolution.
//
// The bore is the hub's outer profile offset inward by TIRE_FIT everywhere, so
// the bead that fills the hub's channel and the recesses that clear its lips
// are one feature rather than two things that have to be kept in agreement.
//
// SHOULDERS. A square tire edge is where a sideways impact starts a tear, and a
// tear in TPU runs along the layer plane at roughly a quarter of the material's
// in-plane strength, so the edge is both the place it starts and the place it
// runs. The shoulder is broken at 45 degrees rather than given a true radius:
// on a part printed axis-vertical a radius approaches a horizontal tangent at
// the bed, which is the one overhang that cannot be printed at all. 45 degrees
// is the roundest a vertical-axis print allows, and it removes the square edge,
// which is the point. Both shoulders get the same break -- an asymmetric tire
// slides differently left and right, and a policy will find that.
//
// The tire is modelled SOLID. Its compliance is the slicer's infill fraction,
// which spec.py carries as a material property; keeping it out of the geometry
// means the softness can be retuned from a test print without re-exporting a
// mesh, and means the URDF's tire stroke and this file cannot disagree.
module tire_profile() {
    bore_lip  = LIP_OUTER_R - TIRE_FIT;
    bore_seat = HUB_OUTER_R - TIRE_FIT;

    rotate_extrude()
        polygon(points = [
            [bore_lip,             0],
            [WHEEL_R - SHOULDER_CH, 0],                                    // bed face
            [WHEEL_R,               SHOULDER_CH],                          // shoulder break
            [WHEEL_R,               WHEEL_W - SHOULDER_CH],                // the tread land
            [WHEEL_R - SHOULDER_CH, WHEEL_W],
            [bore_lip,              WHEEL_W],
            [bore_lip,              WHEEL_W - LIP_LAND],                   // clears outboard lip
            [bore_seat,             WHEEL_W - LIP_LAND - LIP_H],           // 45 deg, matches hub
            [bore_seat,             LIP_LAND + LIP_H],                     // the bead
            [bore_lip,              LIP_LAND]                              // clears inboard lip
        ]);
}

// The tread.
//
// SHALLOW AND NOT AGGRESSIVE, AND THAT IS A TRANSFER DECISION. Published
// wheel-legged work has repeatedly found that a policy trained against
// generous simulated grip learns to lean on it -- it plants and pushes in ways
// that only work at a friction coefficient the simulator invented -- and the
// behaviour does not survive contact with a floor. The contact model is a point
// or a small patch with a Coulomb coefficient; a real compliant hysteretic tire
// with tread blocks that squirm, roll under and recover is not that, and the
// mismatch is worst exactly where the tread is most aggressive. A shallow,
// sparse tread stays close to the thing the simulator can actually represent.
// If more grip is wanted later, the honest way to get it is to measure the
// tire's real coefficient and put THAT number in the model.
//
// Each groove is cut by a capsule -- a hull of two spheres -- so its root is a
// radius rather than a corner and its ends fade out instead of stopping in a
// notch. Straight and axial, not circumferential: on a part printed with the
// axis vertical an axial groove is a plain vertical slot with no overhang
// anywhere, while a circumferential one would have a horizontal roof.
//
// TREAD_INSET is the load-bearing number here. The grooves stop short of both
// shoulders, so no groove ever runs out to the tire's edge. A groove that
// reaches the edge is a pre-cut starting point for the tear the rounded
// shoulder exists to prevent, which would undo the shoulder entirely.
module tire_tread_cuts() {
    cutter_r = WHEEL_R - TREAD_DEPTH + TREAD_R;
    z0 = SHOULDER_CH + TREAD_INSET;
    z1 = WHEEL_W - SHOULDER_CH - TREAD_INSET;

    // Written as one string per str() argument on purpose: OpenSCAD has no
    // implicit concatenation of adjacent string literals, so splitting a message
    // across lines without a comma is a parse error rather than a long line.
    assert(z1 > z0,
           str("tread land is ", z1 - z0, " mm; SHOULDER_CH and TREAD_INSET have ",
               "eaten the whole width and there is nowhere to put a groove"));

    for (i = [0 : TREAD_COUNT - 1])
        rotate([0, 0, i * 360 / TREAD_COUNT])
            hull() {
                translate([cutter_r, 0, z0]) sphere(r = TREAD_R, $fn = 20);
                translate([cutter_r, 0, z1]) sphere(r = TREAD_R, $fn = 20);
            }
}

// The tire in its PRINT frame: axis along +Z, annulus flat on the bed.
//
// No elephant-foot relief: neither flat face of the tire seats against
// anything, so a squashed first layer costs appearance and nothing else. The
// faces that have to fit are the bore and the bead, and both are vertical.
module tire_print() {
    wheel_checks();

    difference() {
        tire_profile();
        tire_tread_cuts();
    }
}

// The tire in the ASSEMBLY frame: wheel axis along Y, seated on the hub.
//
// TPU 95A. 95A rather than a softer shore: 85A yields around 4 MPa and a 2.5 kg
// robot standing on it takes a permanent flat spot. Print as tire_print() is
// authored -- axis vertical, flat on the bed.
module tire() {
    translate([0, -WHEEL_W / 2, 0])
        rotate([-90, 0, 0])
            tire_print();
}
