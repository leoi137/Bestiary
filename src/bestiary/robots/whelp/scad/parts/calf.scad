// calf.scad — knee axis to wheel axis, and the SWAPPABLE wheel drive interface.
//
//   calf()                    the printed calf
//   wheel_mount_face()        the bolt interface an adapter mates to
//   wheel_adapter_sts3215()   the v1 adapter, carrying an STS3215
//
// FRAME. Origin AT THE KNEE AXIS, along Y. Extends along -Z; the WHEEL AXIS is
// at (0, 0, -CALF_L). Authored LEFT; build.scad mirrors for the right.
//
// FORM. The thigh's channel idiom, shallower: CALF_D deep by CALF_WID wide,
// walls CALF_WALL, opening +Y. Shallower because the moment is lower -- the
// ground reaction acts at the wheel, so the calf's worst bending moment is
// F x CALF_L while the thigh's is F x (THIGH_L + CALF_L).
//
// PRINT ORIENTATION: as the thigh -- lying down, long axis in the bed plane,
// channel opening up, sagittal bending in-plane with the extrusions.
//
// WHY THE WHEEL MOUNT IS AN INTERFACE AND NOT A POCKET
// ---------------------------------------------------
// Because the drive fitted in v1 is the design's headline limitation and it is
// expected to be replaced. An STS3215-C018 turns at 45 rpm, which on a 45 mm
// wheel is 0.205 m/s; comparable wheel-legged machines run 1.5-2.5 m/s, and
// Froude scaling from Go2-W puts a realistic target for a 250 mm body near
// 1.2-1.5 m/s. So the calf terminates in a flat bolted FACE with the pattern
// declared as named constants below, and the servo lives on a separate adapter
// plate. Changing the drive is then a 4 g reprint, not a new leg.
//
// A REPLACEMENT DRIVE MUST BEAT: 300 rpm, 0.20 N.m, 60 g. The rpm sets the
// speed, the torque clears the 20-degree slope demand with margin while staying
// under the friction-cone and wheelie ceilings, and the mass keeps the leg's
// distal inertia from getting worse -- grams at the end of a 210 mm lever from
// the hip cost more than grams anywhere else on the machine.
//
// Note what the torque requirement is NOT set by: acceleration. At 1:345 the
// reflected rotor inertia dominates that term so completely that the machine
// behaves as though it weighed 59 kg longitudinally. A drive with less reduction
// is faster AND accelerates better, for the same output torque.

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

EPS = 0.02;

// ── Local numbers ────────────────────────────────────────────────────────────
HORN_PAD_D  = CALF_D + 4;
BEAR_BOSS_D = IDLER_BEARING_OD + 6;
WINDOW_D    = CALF_D * 0.42;
WINDOW_Z    = [-CALF_L * 0.45];

// The wheel-mount interface. THESE FOUR NUMBERS ARE THE CONTRACT: an adapter
// that matches them fits, and nothing else about the drive is the calf's
// business. Kept together and named so a second adapter is written against a
// stated interface rather than against measurements taken off this file.
MOUNT_PLATE_T  = 4.0;                       // adapter plate thickness
MOUNT_BOLT_DX  = 30.0;                      // bolt rectangle, along the leg
MOUNT_BOLT_DY  = 18.0;                      // ...and across it
MOUNT_BORE_D   = SERVO_OUTPUT_BOSS_D + 3;   // clears any drive's output boss
MOUNT_FACE_W   = MOUNT_BOLT_DX + 2 * (INSERT_OD / 2 + INSERT_WALL + 1);
MOUNT_FACE_H   = MOUNT_BOLT_DY + 2 * (INSERT_OD / 2 + INSERT_WALL + 1);

// The fuse web, if fitted. Sized in spec.py from its shear AREA, so the break
// load is a number with provenance rather than a shape somebody drew.
FUSE_W = FUSE_SHEAR_AREA / FUSE_T;

module calf_checks() {
    assert(CALF_WID >= SERVO_BODY_W + 2 * CALF_WALL,
           str("calf channel ", CALF_WID, " mm cannot straddle a ", SERVO_BODY_W,
               " mm case with ", CALF_WALL, " mm walls"));
    assert(MOUNT_FACE_W <= CALF_L,
           str("wheel mount face ", MOUNT_FACE_W, " mm is longer than the ",
               CALF_L, " mm calf"));
    assert(MOUNT_BORE_D < MOUNT_BOLT_DY,
           str("mount bore ", MOUNT_BORE_D, " mm leaves no web inside a ",
               MOUNT_BOLT_DY, " mm bolt spacing"));
    assert(FUSE_W >= 4,
           str("fuse web ", FUSE_W, " mm wide is not printable; raise ",
               "FUSE_SHEAR_AREA or lower FUSE_T"));
}

// ── The channel ──────────────────────────────────────────────────────────────
module calf_outer() {
    hull() {
        translate([0, 0, -CALF_L / 2])
            rounded_plate(CALF_D, CALF_WID, CALF_L, r = FILLET_R, c = 0.6);
        rotate([90, 0, 0]) cylinder(d = HORN_PAD_D, h = CALF_WID, center = true);
    }
}

module calf_cavity() {
    translate([0, CALF_WALL / 2, -CALF_L / 2])
        cube([CALF_D - 2 * CALF_WALL, CALF_WID - CALF_WALL, CALF_L - 2 * CALF_WALL],
             center = true);
}

module calf_coves() {
    for (sx = [-1, 1])
        translate([sx * (CALF_D / 2 - CALF_WALL), CALF_WALL, -CALF_L + CALF_WALL])
            rotate([0, 0, sx > 0 ? 90 : 0])
                rotate([-90, 0, 0])
                    fillet_in(FILLET_R, CALF_L - 2 * CALF_WALL);
}

// ── The knee joint: driven, in double shear ───────────────────────────────────
module calf_knee_solid() {
    translate([0, -CALF_WID / 2, 0])
        rotate([90, 0, 0]) cylinder(d = HORN_PAD_D, h = HORN_BOSS_THICK);
    translate([0, CALF_WID / 2 - IDLER_BEARING_W - 1.5, 0])
        rotate([-90, 0, 0]) cylinder(d = BEAR_BOSS_D, h = IDLER_BEARING_W + 1.5);
}

module calf_knee_cuts() {
    translate([0, -CALF_WID / 2 - EPS, 0])
        rotate([-90, 0, 0]) sts3215_horn_holes(HORN_BOSS_THICK + CALF_WALL + 2);
    translate([0, CALF_WID / 2 + EPS, 0])
        rotate([90, 0, 0])
            bearing_pocket(IDLER_BEARING_OD, IDLER_BEARING_W, BEARING_PRESS_FIT,
                           shelf = IDLER_BEARING_SHELF,
                           bore_clear = IDLER_BEARING_ID + 1);
}

// ── The swappable wheel-drive interface ──────────────────────────────────────

// The bolt pattern, as its own module so the calf and every adapter cut the same
// holes from one definition. `insert` selects boss-and-insert (the calf side) or
// clearance (the adapter side); an insert on both sides bolts to nothing.
module wheel_mount_face(insert = true) {
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * MOUNT_BOLT_DX / 2, sy * MOUNT_BOLT_DY / 2, 0])
            if (insert)
                insert_hole(INSERT_BORE, INSERT_L, M3_CLEAR);
            else
                cylinder(d = M3_CLEAR, h = MOUNT_PLATE_T + 2 * EPS);
}

// The calf's end pad, plus the fuse web that carries load into it.
//
// THE FUSE IS THE ONLY LOAD PATH HERE, deliberately. Any rib bridging the pad to
// the channel would carry the load around the web and the part would stop being
// a fuse without looking any different. Its job: shear at ~280 N, which is 13x a
// design landing and below the calf's ~473 N lateral bending strength, so a
// sideways impact costs a 2 g printed part instead of a leg or a gearbox.
//
// It does NOT protect a hard vertical landing. Tripping before the gear train's
// ~108 N would need a ~2 mm^2 web: unprintable, and close enough to a real
// landing that it would nuisance-trip -- and a fuse that nuisance-trips gets
// left out, after which there is no fuse. Vertical impact is bounded by the drop
// envelope in the torque report, not by a part.
module calf_foot() {
    z_pad = -CALF_L - MOUNT_PLATE_T / 2 - (FUSE_ENABLE ? FUSE_T : 0);
    // The pad.
    translate([0, 0, z_pad])
        rounded_plate(MOUNT_FACE_W, MOUNT_FACE_H, MOUNT_PLATE_T, r = 3, c = 0.5);
    if (FUSE_ENABLE)
        // The web. Notched on both faces by calf_foot_cuts() so it breaks at a
        // known place and a repeatable load rather than wherever the print
        // happened to be weakest.
        translate([0, 0, -CALF_L - FUSE_T / 2])
            cube([FUSE_W, MOUNT_FACE_H * 0.8, FUSE_T], center = true);
    else
        translate([0, 0, -CALF_L - MOUNT_PLATE_T / 4])
            cube([MOUNT_FACE_W * 0.6, MOUNT_FACE_H * 0.8, MOUNT_PLATE_T / 2],
                 center = true);
}

module calf_foot_cuts() {
    z_pad = -CALF_L - MOUNT_PLATE_T / 2 - (FUSE_ENABLE ? FUSE_T : 0);
    translate([0, 0, z_pad + MOUNT_PLATE_T / 2 + EPS])
        rotate([180, 0, 0]) wheel_mount_face(insert = true);
    translate([0, 0, z_pad]) cylinder(d = MOUNT_BORE_D, h = MOUNT_PLATE_T + 2 * EPS,
                                      center = true);
    if (FUSE_ENABLE)
        // The one place on this robot where a SHARP corner is wanted. Every other
        // internal corner is filleted to keep Kt near 1.5 instead of 3; here the
        // concentration is the feature.
        for (sz = [-1, 1])
            translate([0, 0, -CALF_L - FUSE_T / 2 + sz * FUSE_T / 2])
                rotate([0, 45, 0])
                    cube([FUSE_T * 0.5, MOUNT_FACE_H, FUSE_T * 0.5], center = true);
}

// ── The v1 adapter: an STS3215 on the mount face ─────────────────────────────
//
// A separate printable part. Bolts to the calf's face on one side and carries the
// drive on the other, placing its output axis at (0, 0, -CALF_L) along Y in the
// assembled frame. Printed flat, plate face down.
module wheel_adapter_sts3215() {
    difference() {
        union() {
            rounded_plate(MOUNT_FACE_W, MOUNT_FACE_H, MOUNT_PLATE_T, r = 3, c = 0.5);
            // Cradle walls that take the case's reaction moment in shear rather
            // than through the four case screws in tension.
            for (sy = [-1, 1])
                translate([0, sy * (SERVO_BODY_W / 2 + CALF_WALL / 2),
                           MOUNT_PLATE_T / 2 + SERVO_BODY_H / 2])
                    cube([SERVO_BODY_L, CALF_WALL, SERVO_BODY_H], center = true);
        }
        wheel_mount_face(insert = false);
        translate([0, 0, -MOUNT_PLATE_T / 2 - EPS])
            cylinder(d = MOUNT_BORE_D, h = MOUNT_PLATE_T + 2 * EPS);
        translate([0, 0, MOUNT_PLATE_T / 2]) {
            sts3215_body(0.4);
            sts3215_cable_envelope();
            sts3215_mount_holes(MOUNT_PLATE_T + SERVO_BODY_H + 2);
        }
    }
}

// ── Lightening and cable management ──────────────────────────────────────────
module calf_windows() {
    for (z = WINDOW_Z)
        translate([0, CALF_WID / 2 + EPS, z])
            rotate([90, 0, 0]) teardrop(WINDOW_D, CALF_WID + 2 * EPS);
}

module calf_cable_clips() {
    translate([CALF_D / 2 - CALF_WALL / 2, CALF_WID / 2 - 3, -CALF_L * 0.55])
        rotate([0, 90, 0])
            teardrop(SERVO_CABLE_D + 1.2, CALF_WALL + 2 * EPS, center = true);
}

// ── The part ─────────────────────────────────────────────────────────────────
module calf() {
    calf_checks();
    difference() {
        union() {
            difference() {
                calf_outer();
                calf_cavity();
            }
            calf_coves();
            calf_knee_solid();
            calf_foot();
        }
        calf_knee_cuts();
        calf_foot_cuts();
        calf_windows();
        calf_cable_clips();
    }
}

calf();
