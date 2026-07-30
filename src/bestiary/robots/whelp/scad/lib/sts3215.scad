// sts3215.scad — the Feetech STS3215 serial bus servo, as geometry and as a
// mounting interface.
//
// Two kinds of module live here and they must not be confused:
//
//   sts3215_body()     the SOLID the servo occupies. Union it into an assembly
//                      preview; SUBTRACT it (grown by a clearance) from a
//                      bracket to make the pocket the servo drops into.
//   sts3215_*_holes()  the HOLES a bracket needs. Subtract from the bracket.
//
// Every dimension comes from params_gen.scad, which is generated from spec.py,
// which carries the provenance for each number. Nothing is typed here. If a
// dimension in this file looks like a number, it is a bug.
//
// WHY THE SERVO IS MODELLED AT ALL, RATHER THAN JUST ITS HOLES
// ------------------------------------------------------------
// Because the thing that goes wrong on a printed robot is not the bolt circle,
// which is easy and which everyone gets right. It is INTERFERENCE: the servo's
// body fouls the thigh at 40 degrees of knee flexion, or the cable exits into
// the space the calf sweeps through, and neither is visible until the plastic
// exists. Modelling the body as a solid means the assembly preview shows the
// collision, and means check.py can ask about clearances instead of trusting
// that somebody looked.
//
// THE OUTPUT SHAFT IS A CANTILEVER, AND THAT IS THE DESIGN PROBLEM
// ---------------------------------------------------------------
// The servo's output bearing is sized to transmit torque, not to carry the
// bending moment of a robot leg landing on it. Hang a 100 mm calf off one side
// of the horn and every impact is a moment on that bearing: the bushing wears
// oval, the joint develops play, the play becomes backlash, and the backlash
// becomes a policy that cannot repeat itself. It is a slow failure, so it gets
// blamed on the controller.
//
// The fix is DOUBLE SHEAR: an idler pivot on the far side of the joint, coaxial
// with the output, so the moment is reacted by two supports and the servo sees
// (almost) pure torque. `sts3215_idler_holes()` places it. Every rotating joint
// on this robot uses it. It costs one bearing and about four grams.

include <params_gen.scad>
// INCLUDE, not use. `use <>` imports modules and functions but NOT variables, so
// with `use` the EPS this file references below was undef -- and undef propagates
// silently through arithmetic into geometry rather than raising. Every CSG
// overshoot in this file was therefore undef, which is the class of bug that
// renders a preview that looks almost right.
include <util.scad>

// ─────────────────────────────────────────────────────────────────────────────
// THE SOLID
// ─────────────────────────────────────────────────────────────────────────────

// Origin is at the CENTRE OF THE OUTPUT SHAFT, on the horn face, with the
// output axis along +Z. That choice matters: a servo positioned by its body
// corner has to be re-positioned every time a body dimension is corrected,
// whereas one positioned by its axis does not, and the axis is the thing the
// kinematics cares about.
//
// +Z  out of the horn face      +X  toward the cable exit (the "back")
module sts3215_body(clearance = 0) {
    c = clearance;
    translate([0, 0, -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H / 2])
        union() {
            // Main case.
            rounded_plate(SERVO_BODY_L + 2 * c, SERVO_BODY_W + 2 * c,
                          SERVO_BODY_H + 2 * c, r = 2, c = 0);
            // Mounting ears, if this variant has them proud of the case.
            if (SERVO_EAR_THICK > 0)
                translate([0, 0, SERVO_EAR_Z_FROM_BODY_MID])
                    cube([SERVO_EAR_SPAN_L + 2 * c, SERVO_BODY_W + 2 * c,
                          SERVO_EAR_THICK + 2 * c], center = true);
        }

    // Output boss and horn, on +Z.
    translate([0, 0, -SERVO_HORN_FACE_TO_BODY])
        cylinder(d = SERVO_OUTPUT_BOSS_D + 2 * c, h = SERVO_HORN_FACE_TO_BODY + c);
    cylinder(d = SERVO_HORN_D + 2 * c, h = SERVO_HORN_THICK + c);

    // Rear idler boss on -Z, if the variant has one. Whether it does is a
    // MEASURED question on the unit in hand, not a datasheet one -- see
    // spec.py's provenance for SERVO_HAS_REAR_BOSS. When it is absent the
    // brackets fall back to a bearing on a through-bolt, which is why the
    // bracket modules take the boss as a parameter rather than assuming it.
    if (SERVO_HAS_REAR_BOSS)
        translate([0, 0, -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H])
            cylinder(d = SERVO_REAR_BOSS_D + 2 * c, h = SERVO_REAR_BOSS_H + c);

    // Cable exit. Modelled as a solid so the assembly preview shows a leg
    // sweeping through it. Cable routing is not decoration: a serial bus servo
    // has a cable at BOTH ends (in and daisy-chain out), and a leg that pinches
    // one takes the whole bus down, not one joint.
    translate([SERVO_BODY_L / 2 - SERVO_CABLE_INSET,
               0,
               -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H / 2])
        rotate([0, 90, 0])
            cylinder(d = SERVO_CABLE_D + 2 * c, h = SERVO_CABLE_STUB);
}

// The swept volume of the servo plus the space its cable needs to leave in.
// Subtract this from a bracket to guarantee the harness has somewhere to go.
module sts3215_cable_envelope(len = undef) {
    l = len == undef ? SERVO_CABLE_STUB : len;
    translate([SERVO_BODY_L / 2 - SERVO_CABLE_INSET,
               0,
               -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H / 2])
        rotate([0, 90, 0])
            hull() {
                cylinder(d = SERVO_CABLE_D, h = 0.01);
                translate([0, 0, l]) cylinder(d = SERVO_CABLE_D * 2, h = 0.01);
            }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE INTERFACES
// ─────────────────────────────────────────────────────────────────────────────

// Holes for the servo's own mounting screws, in the bracket that holds its BODY.
//
// The pattern is a rectangle on the case, and the screws are self-tapping into
// the servo's plastic case rather than into ours -- so these are CLEARANCE holes
// with a counterbore, never insert bosses.
module sts3215_mount_holes(depth = 20, counterbore = true) {
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * SERVO_MOUNT_HOLE_DX / 2, sy * SERVO_MOUNT_HOLE_DY / 2,
                   -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H - EPS])
            if (counterbore)
                cap_screw_hole(SERVO_MOUNT_SCREW_CLEAR_D,
                               SERVO_MOUNT_SCREW_HEAD_D,
                               SERVO_MOUNT_SCREW_HEAD_H, depth);
            else
                cylinder(d = SERVO_MOUNT_SCREW_CLEAR_D, h = depth);
}

// The bolt pattern that couples the servo's HORN to the part it drives.
//
// This is where torque leaves the servo, and it is the interface most likely to
// be wrong in a way that only shows up after printing, so check.py asserts the
// bolt circle and the count against the servo spec independently of this file.
//
// The bosses are deliberately thick (HORN_BOSS_THICK): torque arrives as a
// force on each screw shank BEARING against printed plastic, and bearing area
// is diameter x thickness. Thin bosses do not shear the screws -- steel wins --
// they oval out, and the joint develops exactly the backlash the double-shear
// pivot was installed to prevent.
module sts3215_horn_holes(depth = undef) {
    d = depth == undef ? HORN_BOSS_THICK + 2 : depth;
    for (i = [0 : SERVO_HORN_BOLT_COUNT - 1])
        rotate([0, 0, i * 360 / SERVO_HORN_BOLT_COUNT])
            translate([SERVO_HORN_BOLT_CIRCLE_D / 2, 0, -EPS])
                cylinder(d = SERVO_HORN_BOLT_CLEAR_D, h = d + 2 * EPS);
    // Central relief for the output boss and the horn's own retaining screw.
    translate([0, 0, -EPS])
        cylinder(d = SERVO_HORN_HUB_CLEAR_D, h = SERVO_HORN_HUB_CLEAR_H + EPS);
}

// A raised pad for the horn to bolt against, with the bolt bosses in it.
//
// Union this into the driven part. Sized from the horn, so it cannot drift out
// of agreement with the hole pattern above.
module sts3215_horn_pad(thick = undef, extra_r = 2.5) {
    t = thick == undef ? HORN_BOSS_THICK : thick;
    difference() {
        cylinder(d = SERVO_HORN_BOLT_CIRCLE_D + SERVO_HORN_BOLT_CLEAR_D + 2 * extra_r, h = t);
        sts3215_horn_holes(t);
    }
}

// The far-side pivot that puts the joint in double shear.
//
// Two variants, chosen by SERVO_HAS_REAR_BOSS:
//   true  -> a bearing whose bore rides on the servo's own rear boss. Best:
//            self-aligning with the output shaft by construction, since both
//            are features of one moulded case.
//   false -> a bearing on a shoulder bolt through the bracket, coaxial with the
//            output by assembly rather than by construction. Works, but any
//            misalignment preloads both the bearing and the servo bearing, so
//            the bracket's two halves have to be printed as one part or dowelled.
//
// Subtract from the driven part, coaxial with the servo's output axis.
module sts3215_idler_pocket() {
    translate([0, 0, -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H - IDLER_BEARING_W + EPS])
        bearing_pocket(IDLER_BEARING_OD, IDLER_BEARING_W, BEARING_PRESS_FIT,
                       shelf = IDLER_BEARING_SHELF, bore_clear = IDLER_BEARING_ID + 1);
}

// The stack-up along the output axis, as a single number, so brackets can be
// laid out from it instead of from a sum of remembered thicknesses.
function sts3215_axis_span() =
    SERVO_HORN_THICK + SERVO_HORN_FACE_TO_BODY + SERVO_BODY_H
    + (SERVO_HAS_REAR_BOSS ? SERVO_REAR_BOSS_H : 0);

// Distance from the output axis to the far face of the case, i.e. how much
// clearance a link swinging around this joint needs before it hits the servo.
function sts3215_swing_clearance() =
    sqrt(pow(SERVO_BODY_L / 2, 2) + pow(SERVO_BODY_W / 2, 2));
