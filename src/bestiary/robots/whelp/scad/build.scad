// build.scad — WHELP-16. One file, every part, and the posed assembly.
//
//   openscad build.scad                                  the whole robot, standing
//   openscad -D 'PART="thigh"' -o thigh.stl build.scad    one part, for printing
//   openscad -D 'MODE="plate"' build.scad                 everything laid out on the bed
//
// Normally you do not call OpenSCAD by hand:
//
//   python -m bestiary.robots.whelp.export --stl --check-mass
//
// renders every part, measures each mesh, and asserts the analytic mass model
// agrees with the real geometry. That last step is the only thing standing
// between "the CAD and the URDF describe the same robot" and "they used to".
//
// NEVER HAND-EDIT AN STL, AND NEVER HAND-EDIT params_gen.scad
// -----------------------------------------------------------
// Both are outputs. params_gen.scad and derived_gen.scad are written from
// spec.py and geometry.py, and check.py fails if regenerating them changes a
// byte -- so an edit here is not merely discouraged, it is caught. To change a
// dimension, edit spec.py, where the number also has to declare where it came
// from.
//
// THE ASSEMBLY IS POSED AT THE SOLVED STANCE
// -------------------------------------------
// STANCE_KNEE in derived_gen.scad is not typed. It is the output of
// geometry.solve_stance_knee(), which puts each wheel axle directly under its
// own hip pivot -- and the URDF is generated from that same call. So the render
// below and an Isaac Lab spawn are the same pose by construction rather than by
// two people typing the same number.
//
// If the render looks wrong, the simulation is wrong in the same way, and that
// is the point: an assembly preview that cannot disagree with the physics is
// worth more than a prettier one that can.

include <lib/params_gen.scad>
include <lib/derived_gen.scad>
use <lib/util.scad>
use <lib/sts3215.scad>
use <parts/trunk.scad>
use <parts/abduct_bracket.scad>
use <parts/thigh.scad>
use <parts/calf.scad>
use <parts/wheel.scad>
use <parts/fuse.scad>
use <parts/fitcheck.scad>

// ─────────────────────────────────────────────────────────────────────────────
// ENTRY POINTS
// ─────────────────────────────────────────────────────────────────────────────
// PART selects a single part for STL export. export.py passes it with -D.
// MODE selects what to draw when PART is empty.
PART = "";
MODE = "assembly";      // "assembly" | "plate" | "leg"
SHOW_SERVOS = true;     // draw the bought parts too, so interference is visible

$fn = FN;

if (PART != "") {
    part(PART);
} else if (MODE == "plate") {
    print_plate();
} else if (MODE == "leg") {
    leg_assembly(1, 1);
} else {
    assembly();
}

// Dispatch by name. The names here MUST match massmodel.PRINTED_PARTS, because
// export.py renders one STL per key of that dict and check.py asserts the two
// sets agree -- a part that exists in CAD but not in the mass model is a part
// whose weight is missing from the URDF.
module part(which) {
    if (which == "trunk_front")     trunk_front();
    else if (which == "trunk_rear") trunk_rear();
    else if (which == "trunk_lid")  trunk_lid();
    else if (which == "abduct_bracket") abduct_bracket();
    else if (which == "thigh")      thigh();
    else if (which == "calf")       calf();
    else if (which == "wheel_hub")  wheel_hub();
    else if (which == "tire")       tire();
    else if (which == "fuse")       fuse();
    else if (which == "fitcheck")   fitcheck();
    else assert(false, str("build.scad: unknown PART \"", which, "\". ",
                           "It must be a key of massmodel.PRINTED_PARTS."));
}

// ─────────────────────────────────────────────────────────────────────────────
// ASSEMBLY
// ─────────────────────────────────────────────────────────────────────────────

// One leg, posed at the standing stance.
//
// fx, fy are the leg's signs: +1 front / -1 rear, +1 left / -1 right, matching
// geometry.SIGNS. Parts are authored for the LEFT side and mirrored for the
// right, so exactly one printed geometry exists per part and a left and a right
// leg are the same file.
//
// The rotation order is the kinematic chain: abduction about X, then hip and
// knee about Y. Reading it top to bottom is reading the URDF's joint list.
module leg_assembly(fx, fy) {
    mirror([0, fy < 0 ? 1 : 0, 0])
        translate([fx * ABDUCT_X, ABDUCT_Y, 0])
            rotate([STANCE_ABDUCT_DEG, 0, 0]) {
                abduct_bracket();
                if (SHOW_SERVOS)
                    translate([0, ABDUCT_TO_HIP, -ABDUCT_TO_HIP_DROP])
                        rotate([90, 0, 0]) sts3215_body();

                translate([0, ABDUCT_TO_HIP, -ABDUCT_TO_HIP_DROP])
                    rotate([0, STANCE_HIP_DEG, 0]) {
                        thigh();
                        translate([0, 0, -THIGH_L])
                            rotate([0, STANCE_KNEE_DEG, 0]) {
                                calf();
                                if (FUSE_ENABLE)
                                    translate([0, 0, -CALF_L + 8]) fuse();
                                translate([0, CALF_TO_WHEEL_PLANE, -CALF_L]) {
                                    wheel_hub();
                                    color("DimGray") tire();
                                }
                            }
                    }
            }
}

// The whole machine, standing on z = 0.
//
// The trunk origin sits at STAND_HEIGHT, which is the axle drop plus the LOADED
// wheel radius -- not the free radius. A tire carrying load is a squashed tire,
// and an assembly built at the free radius is one that spawns a few millimetres
// into the floor in simulation, which PhysX resolves by launching the robot.
module assembly() {
    translate([0, 0, STAND_HEIGHT]) {
        trunk_assembly();
        for (i = [0 : len(LEG_SIGNS) - 1])
            leg_assembly(LEG_SIGNS[i][0], LEG_SIGNS[i][1]);
    }
    // A ground plane, so "is it actually standing on the floor" is something you
    // can see rather than something you assume.
    color("Gainsboro") translate([0, 0, -1]) cube([600, 400, 1], center = true);
}

// NO TRANSFORMS. parts/trunk.scad authors both halves and both lids in the
// ASSEMBLED frame -- the two halves are cut out of one shell by a single clip
// solid, so tongue and recess are complementary by construction rather than by
// two sets of numbers agreeing. An earlier version of this module translated
// each half by +/-TRUNK_L/4, which double-offset them, and called trunk_lid()
// with no argument so both lids rendered as the front one.
//
// The servo placement below also has to match trunk.scad's own convention
// exactly, because the pockets are cut there and the bodies drawn here. It is
// therefore delegated: abduct_servo_at() is the single definition of where an
// abduction servo sits and how it is rolled, and this file calls it rather than
// re-deriving a rotation that has to agree with it.
module trunk_assembly() {
    trunk_front();
    trunk_rear();
    trunk_lid("front");
    trunk_lid("rear");
    if (SHOW_SERVOS)
        for (i = [0 : len(LEG_SIGNS) - 1])
            abduct_servo_at(LEG_SIGNS[i][0], LEG_SIGNS[i][1]) sts3215_body();
}

// ─────────────────────────────────────────────────────────────────────────────
// PRINT PLATE
// ─────────────────────────────────────────────────────────────────────────────

// Every unique part laid out flat, with the bed drawn under them.
//
// This is a sanity view, not a slicer arrangement: it shows the parts at their
// authored orientation on a BED_X x BED_Y rectangle so that "does it fit" is
// visible. The real assertion is in check.py, which measures each rendered STL's
// bounding box -- eyeballing a layout is how a part ends up 2 mm over.
module print_plate() {
    color("Gainsboro") translate([0, 0, -1]) cube([BED_X, BED_Y, 1], center = true);
    translate([-BED_X / 2 + 75, BED_Y / 2 - 40, 0])   trunk_front();
    translate([-BED_X / 2 + 75, BED_Y / 2 - 110, 0])  trunk_rear();
    translate([BED_X / 2 - 45, BED_Y / 2 - 40, 0])    thigh();
    translate([BED_X / 2 - 45, BED_Y / 2 - 100, 0])   calf();
    translate([BED_X / 2 - 45, -BED_Y / 2 + 50, 0])   abduct_bracket();
    translate([0, -BED_Y / 2 + 50, 0])                wheel_hub();
    translate([60, -BED_Y / 2 + 50, 0])               tire();
    translate([-60, -BED_Y / 2 + 20, 0])              fuse();
}
