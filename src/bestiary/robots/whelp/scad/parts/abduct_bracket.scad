// abduct_bracket.scad — the hip carrier. One module: abduct_bracket().
//
// FRAME. Origin AT THE ABDUCTION AXIS, which runs along X (fore-aft). This part
// rotates about X and carries the HIP servo, whose output axis is along Y at
// (0, ABDUCT_TO_HIP, -ABDUCT_TO_HIP_DROP). Authored LEFT; mirrored for the right.
//
// This is the most three-dimensional part on the robot: it takes drive about X
// and delivers a mount for a servo whose axis is Y, so it turns a corner. It is
// also the part that carries the entire leg -- about 830 g at up to 190 mm of
// reach -- so it is where a missing fillet costs the most.
//
// ─────────────────────────────────────────────────────────────────────────────
// AN UNRESOLVED INTERFACE, STATED RATHER THAN PAPERED OVER
// ─────────────────────────────────────────────────────────────────────────────
// This file and parts/trunk.scad do not yet agree on how the bracket leaves the
// trunk, and neither has been rendered, so the disagreement is documented here
// instead of being hidden behind a guess.
//
// The kinematics is unambiguous. geometry.build_chain() puts the abduction pivot
// at (+/-ABDUCT_X, +/-ABDUCT_Y, 0) and the hip pivot one step outboard of it at
// (0, +/-ABDUCT_TO_HIP, 0) in the bracket's own frame -- i.e. SAME X, 36 mm
// further out in Y. So the hip axis lands at |y| = 72 mm, which is 20 mm OUTBOARD
// of the trunk's 52 mm half-width. The bracket must therefore cross the trunk's
// SIDE boundary, and it must do so at x = +/-ABDUCT_X.
//
// trunk.scad instead puts a JOURNAL_D plain journal through the trunk's END wall
// at |x| = 125, coaxial with the abduction axis, and has the bracket reach out
// through it. That is a good bearing arrangement and it is a different topology:
// it would place the hip pivot outboard in X, not in Y, which the URDF does not
// describe.
//
// This file is authored to the KINEMATICS, because the URDF is what a policy is
// trained against and the plastic is what has to match it -- not the other way
// round. It exits over the trunk's top rim and down its outside face, which is
// possible precisely because the shell is open-topped for printing. The cost is
// that the second support is a boss on the shell's outer face rather than a
// journal in the end wall, and that boss is NOT modelled here because it belongs
// to the trunk.
//
// WHAT TO DO ABOUT IT, in order of preference:
//   1. Install OpenSCAD, render trunk_front() and abduct_bracket() together, and
//      look. Neither has ever been executed; that is the only honest next step
//      and it is one apt-get away.
//   2. If the over-rim route fouls the lid, move the abduction pivot outboard
//      (raise abduct_y_mm toward the side wall) and shorten abduct_to_hip_mm to
//      keep the track and the abduction moment arm where torque.py wants them.
//      Both are single numbers in spec.py with provenance entries.
//   3. Only then consider changing the URDF, and if so, change spec.py so the
//      kinematics and the CAD are generated from one description again.
//
// Do NOT resolve it by editing params_gen.scad. It is generated, and check.py
// fails if regenerating it changes a byte.
// ─────────────────────────────────────────────────────────────────────────────
//
// PRINT ORIENTATION: the X-axis hub laid on the bed, so the horn pad's face is a
// vertical wall and the leg's bending moment about the abduction axis runs
// in-plane with the extrusions. The cradle for the hip servo then grows upward
// off the hub as a wall rather than hanging off it as an overhang. Footprint
// about 60 x 74 mm.

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

EPS = 0.02;

// ── Local numbers ────────────────────────────────────────────────────────────
WALL       = THIGH_WALL;                       // one wall standard across the leg
HUB_D      = SERVO_HORN_D + 2 * WALL + 6;      // the X-axis hub
HUB_L      = HORN_BOSS_THICK + WALL;           // its length along X
ARM_Y      = ABDUCT_TO_HIP;                    // hub centre -> hip axis, laterally
ARM_T      = WALL * 2;                         // the reach's web thickness
ARM_H      = SERVO_BODY_H + 2 * WALL;          // its depth in Z
CRADLE_L   = SERVO_BODY_L + 2 * WALL;
CRADLE_W   = SERVO_BODY_W + 2 * WALL;
GUSSET_L   = ARM_Y * 0.65;
CABLE_R    = SERVO_CABLE_D * 1.6;              // service-loop radius, see below

module abduct_checks() {
    assert(HUB_D / 2 >= SERVO_HORN_BOLT_CIRCLE_D / 2 + SERVO_HORN_BOLT_D + WALL,
           str("hub ", HUB_D, " mm is too small for a ", SERVO_HORN_BOLT_CIRCLE_D,
               " mm bolt circle plus ", WALL, " mm of wall"));
    assert(ARM_Y > CRADLE_W / 2,
           str("hip axis at ", ARM_Y, " mm is inside the servo cradle; the arm has ",
               "nowhere to reach"));
    assert(ABDUCT_TO_HIP_DROP == 0,
           str("this bracket assumes a coplanar hip pivot; ABDUCT_TO_HIP_DROP is ",
               ABDUCT_TO_HIP_DROP));
    assert(max(HUB_L + CRADLE_L, ARM_Y + CRADLE_W) <= BED_X,
           "abduction bracket does not fit the bed");
}

// ── The hub: driven about X, in double shear ──────────────────────────────────
//
// Horn pad on -X, idler bearing on +X, coaxial. The STS3215 is a DUAL-SHAFT
// servo -- there is a boss opposite the horn -- so the far-side bearing rides on
// the servo's own casting and is coaxial BY CONSTRUCTION rather than by assembly.
// That distinction matters: a bearing aligned by assembly preloads both itself
// and the servo's own bushing by whatever the build tolerance is, and the whole
// point of the second support is to take load OFF that bushing.
module abduct_hub_solid() {
    translate([-HUB_L / 2, 0, 0])
        rotate([0, 90, 0]) cylinder(d = HUB_D, h = HUB_L, center = true);
    translate([WALL, 0, 0])
        rotate([0, 90, 0])
            cylinder(d = IDLER_BEARING_OD + 6, h = IDLER_BEARING_W + 1.5,
                     center = true);
}

module abduct_hub_cuts() {
    translate([-HUB_L - EPS, 0, 0])
        rotate([0, 90, 0]) sts3215_horn_holes(HUB_L + WALL + 2);
    translate([WALL + IDLER_BEARING_W / 2 + 1, 0, 0])
        rotate([0, -90, 0])
            bearing_pocket(IDLER_BEARING_OD, IDLER_BEARING_W, BEARING_PRESS_FIT,
                           shelf = IDLER_BEARING_SHELF,
                           bore_clear = IDLER_BEARING_ID + 1);
}

// ── The reach: hub to hip cradle ──────────────────────────────────────────────
//
// A pair of webs rather than one, so the section has torsional stiffness. The
// leg's weight arrives at the cradle and leaves through the hub, and between
// them the load is carried in BENDING about X -- in-plane with the layers in the
// stated print orientation, which is the whole reason for that orientation.
module abduct_arm() {
    hull() {
        translate([-HUB_L / 2, 0, 0])
            rotate([0, 90, 0]) cylinder(d = HUB_D, h = ARM_T, center = true);
        translate([-HUB_L / 2, ARM_Y, 0])
            cube([ARM_T, CRADLE_W, ARM_H], center = true);
    }
}

// Gussets where the X-axis structure meets the Y-axis cradle.
//
// THE HIGHEST-STRESS REGION ON THE ROBOT. Everything the leg weighs, times its
// reach, is reacted across this corner. A bare 90-degree corner here would carry
// a stress concentration near 3.0 in bending and roughly half the fatigue life,
// and a leg joint at a 2 Hz gait reaches 1e6 cycles in about 140 hours of
// walking -- so the failure would arrive after the robot had been working fine
// for a week.
module abduct_gussets() {
    for (sz = [-1, 1])
        translate([-HUB_L / 2, HUB_D / 2 - FILLET_R, sz * (ARM_H / 2 - EPS)])
            rotate([0, 0, 0])
                mirror([0, 0, sz > 0 ? 0 : 1])
                    gusset(GUSSET_L, ARM_H * 0.45, ARM_T);
}

// ── The hip servo cradle ─────────────────────────────────────────────────────
module abduct_cradle_solid() {
    translate([-HUB_L / 2, ARM_Y, 0])
        rounded_plate(CRADLE_L, CRADLE_W, ARM_H, r = FILLET_R, c = 0.6);
}

module abduct_cradle_cuts() {
    translate([-HUB_L / 2, ARM_Y, 0])
        rotate([-90, 0, 0]) {
            sts3215_body(0.4);
            sts3215_cable_envelope();
            sts3215_mount_holes(CRADLE_W + 2);
        }
}

// ── Cable routing, which is a moving problem here ────────────────────────────
//
// TWO harnesses pass this bracket -- the abduction servo's daisy chain and the
// hip servo's -- and unlike every other link, THIS ONE ROTATES relative to the
// trunk, through the full +/-0.8 rad abduction range. So the channel is
// deliberately oversized: it has to hold a service loop that can take up 1.6 rad
// of relative rotation without the loom going tight at either end.
//
// A pinched bus cable does not fail one joint. It fails the BUS, and sixteen
// servos go silent at once.
module abduct_cable_route() {
    hull() {
        translate([-HUB_L / 2, HUB_D / 2 - CABLE_R, 0])
            rotate([0, 90, 0]) cylinder(r = CABLE_R, h = ARM_T + 2 * EPS,
                                        center = true);
        translate([-HUB_L / 2, ARM_Y - CRADLE_W / 2, 0])
            rotate([0, 90, 0]) cylinder(r = CABLE_R, h = ARM_T + 2 * EPS,
                                        center = true);
    }
}

// ── The part ─────────────────────────────────────────────────────────────────
module abduct_bracket() {
    abduct_checks();
    difference() {
        union() {
            abduct_hub_solid();
            abduct_arm();
            abduct_gussets();
            abduct_cradle_solid();
        }
        abduct_hub_cuts();
        abduct_cradle_cuts();
        abduct_cable_route();
    }
}

abduct_bracket();
