// fitcheck.scad — the coupon that turns a dozen guesses into measurements.
//
// This is the only file in scad/parts/ that is not part of the robot, and it is
// the one to print first. spec.py marks roughly a dozen of the servo interface
// numbers `assumed()`: Feetech publishes torque, speed and current for the
// STS3215 and no dimensioned drawing, so the horn's bolt circle, the number of
// screws in it, the case mounting pattern and the rear boss diameter are all
// educated guesses carried into every bracket the robot has. A wrong bolt
// circle is not a small error -- it is four legs' worth of filament and a day of
// print time producing parts that cannot be bolted to anything.
//
// So: one plate, thirty minutes, every guess placed side by side with its
// neighbours so the real hardware picks the winner. Offer the horn up, read the
// number, edit spec.py, re-export. That is the whole workflow, and it is the
// cheapest possible way to retire an assumption -- cheaper than measuring, even,
// because a caliper reading of a chamfered bolt circle is itself a guess whereas
// "the screws drop through this one and not that one" is not.
//
// The plate also carries the four printer-specific numbers that no datasheet can
// give you, because they belong to YOUR machine and not to ours: the heat-set
// insert bore that neither splits the boss nor spins, the bearing interference
// that presses home without cracking, what your overhangs and bridges really do,
// and your own layer-orientation strength ratio.
//
// PRINT ORIENTATION: exactly as modelled. Everything is flat on the bed except
// the second tensile coupon, which is modelled standing on edge and must stay
// that way -- that is the entire point of it. No supports. If any feature here
// needs support, that is a finding, not a slicer setting.
//
// SIDES: none. This is a coupon, not a leg part, so the left/right mirroring
// convention that governs every other file in scad/parts/ does not apply here.

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

$fn = FN;

EPS = 0.02;         // util.scad's EPS is not visible through `use`, so it is
                    // restated here. Same value, same reason: overshoot every
                    // subtracted solid so no face is ever exactly coincident.

// ─────────────────────────────────────────────────────────────────────────────
// PLATE LAYOUT
//
// The whole coupon must stay inside 120 x 120 mm, and the horn gauge is what
// sets the width: five bolt circles that cannot overlap each other need 112 mm
// of row. Everything else is fitted around that one constraint.
//
// The plate carries features 1-6; the two tensile coupons and the overhang fan
// are SEPARATE bodies in the strip below it, because a tensile coupon that is
// attached to anything cannot be pulled.
// ─────────────────────────────────────────────────────────────────────────────

PLATE_T  = 3.0;     // Also the thickness of the fan pad. One number, so the
                    // engraving depth below is correct on both.
PLATE_W  = 119.0;
PLATE_D  = 98.0;
PLATE_Y0 = 10.0;    // plate centre in Y; the plate spans y = -39 .. +59

ENG_D    = 0.6;     // Engraved, not raised. Raised 3 mm glyphs on a 0.4 mm
                    // nozzle are four loose strands that the nozzle knocks off
                    // on the next pass; a 0.6 mm recess is one contour and
                    // survives being handled with oily fingers.

BOSS_SKIRT = 1.2;   // 45-degree root fillet under every boss. A boss meeting a
                    // plate at a sharp corner is the textbook stress riser, and
                    // here it is loaded exactly where it hurts: pressing an
                    // insert or a bearing in puts a hoop load on the boss and a
                    // prying load on its root. Costs a tenth of a gram.

// ─────────────────────────────────────────────────────────────────────────────
// 1. HORN BOLT CIRCLE GAUGE
//
// Five candidate circles spanning SERVO_HORN_BOLT_CIRCLE_D +/- 4 mm. Offer the
// real horn up; the one whose screws drop through is the answer.
//
// The ORDER is not numeric and that is deliberate: the largest circle needs the
// most room on both sides, so it goes in the middle of the row where there is
// room, and the diameters taper outward. Sorted 13..21 the 21 mm pattern would
// hang 2 mm off the right-hand edge of a plate that is already at the 120 mm
// budget.
//
// Each pattern gets a through-hole for the horn's hub, because a horn whose hub
// is resting on the plate is not seated and will happily "fit" a pattern that is
// a millimetre out. On the 13 mm circle that hub hole leaves only 0.8 mm of
// plastic to the bolt holes; that ring is fragile, and if 13 mm turns out to be
// the right answer, that thin land is itself telling you something about how
// close the real bolt circle runs to the hub.
// ─────────────────────────────────────────────────────────────────────────────

HORN_BCS = [13, 17, 21, 19, 15];
HORN_X   = [-50, -25, 0, 25, 50];
HORN_Y   = 46.0;
HORN_LABEL_Y = 30.5;

// The bolt COUNT is assumed too, so the nominal circle carries twice as many
// holes as we think it has. A 4-hole horn still fits it (every other hole), and
// an 8-hole horn is caught rather than discovered after the legs are printed.
function horn_count(bc) =
    bc == SERVO_HORN_BOLT_CIRCLE_D ? 2 * SERVO_HORN_BOLT_COUNT
                                   : SERVO_HORN_BOLT_COUNT;

module horn_pattern_holes(bc) {
    n = horn_count(bc);
    for (i = [0 : n - 1])
        rotate([0, 0, i * 360 / n])
            translate([bc / 2, 0, -EPS])
                cylinder(d = SERVO_HORN_BOLT_CLEAR_D, h = PLATE_T + 2 * EPS);
    // Hub relief, so the horn seats on the plate rather than on its own boss.
    translate([0, 0, -EPS])
        cylinder(d = SERVO_HORN_HUB_CLEAR_D, h = PLATE_T + 2 * EPS);
}

module horn_bolt_gauge_cuts() {
    for (i = [0 : len(HORN_BCS) - 1]) {
        translate([HORN_X[i], HORN_Y, 0]) horn_pattern_holes(HORN_BCS[i]);
        translate([HORN_X[i], HORN_LABEL_Y, 0])
            engrave(str("BC", HORN_BCS[i],
                        HORN_BCS[i] == SERVO_HORN_BOLT_CIRCLE_D
                            ? str("x", 2 * SERVO_HORN_BOLT_COUNT) : ""),
                    3);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. HEAT-SET INSERT BORE LADDER
//
// Four bosses, bores at INSERT_BORE - 0.2 / +0.0 / +0.2 / +0.4. The insert
// manufacturer's number is a starting point, not an answer: it assumes a hole
// that came out the size it was drawn, and a printed hole does not. Too tight
// and the boss splits as the iron pushes the insert in -- sometimes not visibly,
// which is worse. Too loose and the insert spins in its hole the first time a
// bolt is torqued, and then there is no repair short of reprinting the part.
//
// Press one insert into each, then torque an M3 into each until it stops.
// The right bore is the largest one that does not spin.
// ─────────────────────────────────────────────────────────────────────────────

INS_BOSS_H = INSERT_L + 1.5;    // Same rule insert_boss() uses when h is undef:
                                // 1.5 mm of meat under the insert so the last of
                                // the melt has somewhere to go.
INS_STEPS  = [-0.2, 0.0, 0.2, 0.4];
INS_LABELS = ["I4.0", "I4.2", "I4.4", "I4.6"];
INS_X      = [-50, -37, -24, -11];
INS_Y      = 21.5;
BAND_B_LABEL_Y = 12.0;

module insert_ladder_boss(d_hole) {
    od = d_hole + 2 * INSERT_WALL;
    difference() {
        union() {
            translate([0, 0, PLATE_T]) cylinder(d = od, h = INS_BOSS_H);
            translate([0, 0, PLATE_T])
                cylinder(d1 = od + 2 * BOSS_SKIRT, d2 = od, h = BOSS_SKIRT);
        }
        // Inverted, because insert_hole() is authored with its lead-in at z = 0
        // and its relief above; the insert must enter from the TOP of the boss,
        // not from the plate side.
        translate([0, 0, PLATE_T + INS_BOSS_H]) rotate([180, 0, 0])
            insert_hole(d_hole, INSERT_L, M3_CLEAR);
    }
}

module insert_bore_ladder() {
    for (i = [0 : len(INS_X) - 1])
        translate([INS_X[i], INS_Y, 0])
            insert_ladder_boss(INSERT_BORE + INS_STEPS[i]);
}

module insert_bore_ladder_cuts() {
    for (i = [0 : len(INS_X) - 1]) {
        // The screw relief has to continue through the plate, or a bolt run in
        // from below jacks the insert back out of the boss it was just pressed
        // into -- and it also gives you a punch path to eject a botched insert.
        translate([INS_X[i], INS_Y, -EPS])
            cylinder(d = M3_CLEAR, h = PLATE_T + 2 * EPS);
        translate([INS_X[i], BAND_B_LABEL_Y, 0]) engrave(INS_LABELS[i], 3);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. BEARING POCKET LADDER
//
// Four pockets for the IDLER_BEARING_OD bearing at 0.0 / 0.1 / 0.2 / 0.3 mm
// diametral interference. spec.py records 0.2 mm as needing about 585 N to press
// into PETG -- firm by hand with an arbor press, and it does not fall out. That
// number was measured on somebody else's printer, with their filament, at their
// flow rate, and every one of those changes it.
//
// The number matters in both directions. Too loose and the idler bearing that
// puts each joint in double shear walks out of its pocket under vibration, which
// silently converts the joint back to a cantilever. Too tight and pressing it
// home cracks the boss along a layer line, which you may not see until the leg
// lands on it.
//
// Every pocket gets a punch-out hole through the plate: a bearing you cannot
// remove is a bearing you can only test once, and this ladder is meant to be
// tested with one bearing moved from pocket to pocket.
// ─────────────────────────────────────────────────────────────────────────────

BRG_WALL     = 3.0;
BRG_BOSS_H   = IDLER_BEARING_W + IDLER_BEARING_SHELF + 1.5;
BRG_UNDER    = [0.0, 0.1, 0.2, 0.3];
BRG_LABELS   = ["B0.0", "B0.1", "B0.2", "B0.3"];
BRG_X        = [-49, -29, -9, 11];
BRG_Y        = -1.0;
BRG_LABEL_Y  = -13.5;

module bearing_ladder_boss(under) {
    od = IDLER_BEARING_OD + 2 * BRG_WALL;
    difference() {
        union() {
            translate([0, 0, PLATE_T]) cylinder(d = od, h = BRG_BOSS_H);
            translate([0, 0, PLATE_T])
                cylinder(d1 = od + 2 * BOSS_SKIRT, d2 = od, h = BOSS_SKIRT);
        }
        // Inverted for the same reason as the insert boss: bearing_pocket()
        // builds upward from its mouth, and the mouth belongs at the top face.
        // Flipping it also puts the shelf UNDER the race, which is the whole
        // point of the shelf -- the bearing stops at a known depth instead of
        // being driven until the arbor press feels resistance.
        translate([0, 0, PLATE_T + BRG_BOSS_H]) rotate([180, 0, 0])
            bearing_pocket(IDLER_BEARING_OD, IDLER_BEARING_W, -under,
                           shelf = IDLER_BEARING_SHELF);
        translate([0, 0, -EPS])
            cylinder(d = IDLER_BEARING_ID + 2,
                     h = PLATE_T + BRG_BOSS_H + 2 * EPS);
    }
}

module bearing_fit_ladder() {
    for (i = [0 : len(BRG_X) - 1])
        translate([BRG_X[i], BRG_Y, 0]) bearing_ladder_boss(BRG_UNDER[i]);
}

module bearing_fit_ladder_cuts() {
    for (i = [0 : len(BRG_X) - 1]) {
        translate([BRG_X[i], BRG_Y, -EPS])
            cylinder(d = IDLER_BEARING_ID + 2, h = PLATE_T + 2 * EPS);
        translate([BRG_X[i], BRG_LABEL_Y, 0]) engrave(BRG_LABELS[i], 3);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. M3 CLEARANCE AND NUT POCKET
//
// The two fastener features that appear everywhere else in the robot, once each,
// so M3_CLEAR and M3_NUT_AF are confirmed rather than trusted. A plain M3
// clearance hole and a captive-nut pocket with a side entry slot.
//
// The nut pocket is built with a real roof over it (2.6 mm of plastic) rather
// than opened out to the top face, because the roof is the part that can go
// wrong: it is an unsupported bridge over the nut, and if your machine sags it
// the nut sits proud and the joint never pulls up tight. Checking the across-
// flats fit in an open-topped slot would not test that at all.
// ─────────────────────────────────────────────────────────────────────────────

M3_HOLE_X   = 2.0;
NUT_X       = 18.0;
NUT_PAD_W   = 16.0;
NUT_PAD_D   = 14.0;
NUT_PAD_H   = 6.0;
NUT_FLOOR   = 1.0;      // pad material under the pocket, so the pocket floor is
                        // never coincident with the plate's top face

module nut_pocket_block() {
    difference() {
        translate([0, 0, PLATE_T + NUT_PAD_H / 2])
            rounded_plate(NUT_PAD_W, NUT_PAD_D, NUT_PAD_H, r = 2);
        translate([0, 0, PLATE_T + NUT_FLOOR])
            nut_pocket(M3_NUT_AF, M3_NUT_T, entry_len = 9);
        translate([0, 0, PLATE_T - EPS])
            cylinder(d = M3_CLEAR, h = NUT_PAD_H + 2 * EPS);
    }
}

module m3_gauge_cuts() {
    translate([M3_HOLE_X, INS_Y, -EPS])
        cylinder(d = M3_CLEAR, h = PLATE_T + 2 * EPS);
    translate([NUT_X, INS_Y, -EPS])
        cylinder(d = M3_CLEAR, h = PLATE_T + 2 * EPS);
    translate([M3_HOLE_X, BAND_B_LABEL_Y, 0]) engrave("M3", 3);
    translate([NUT_X, BAND_B_LABEL_Y, 0]) engrave("NUT", 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. SERVO CASE MOUNTING PATTERN GAUGE
//
// Three candidate rectangles of clearance holes, concentric, bracketing
// SERVO_MOUNT_HOLE_DX x SERVO_MOUNT_HOLE_DY. Offer the servo up and see which
// set its ears line up with.
//
// The 6 mm step between candidates is forced, not chosen. Nesting rectangles
// about a common centre moves each corner hole by step/sqrt(2); with 2.9 mm
// clearance holes, anything under a 6 mm step merges adjacent holes into a slot
// and the gauge stops being able to distinguish them. Three candidates at 6 mm
// is what fits, so the brackets are wide: read the nearest fit, then confirm the
// exact pitch with calipers before editing spec.py. This gauge tells you which
// neighbourhood you are in, which is the part that is currently unknown.
// ─────────────────────────────────────────────────────────────────────────────

SRV_PATTERNS = [[32, 12], [38, 18], [44, 24]];
SRV_Y = -24.0;

module servo_mount_gauge_cuts() {
    for (p = SRV_PATTERNS)
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * p[0] / 2, SRV_Y + sy * p[1] / 2, -EPS])
                cylinder(d = SERVO_MOUNT_SCREW_CLEAR_D, h = PLATE_T + 2 * EPS);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6b. TEARDROP WITNESS BLOCK
//
// Two horizontal bores of the same nominal diameter through one block: one a
// plain circle, one a teardrop. Both are measured with the same calipers, on the
// same print, in the same place on the bed, so the difference between them is
// the sag and nothing else.
//
// This is the measurement behind the rule that every horizontal bearing bore on
// this robot is a teardrop. The plain hole's roof goes fully horizontal at the
// top, sags into the bore, and comes out undersize and oval exactly where a
// bearing outer race has to seat -- which is the one place on the part where
// "roughly round" is not good enough. The plain hole here is a deliberate
// violation of the no-overhang rule; it is the control.
//
// Bore diameter is IDLER_BEARING_OD, because that is the horizontal bore the
// robot actually depends on.
// ─────────────────────────────────────────────────────────────────────────────

TD_X   = 40.0;      // block centre
TD_Y   = -1.0;
TD_W   = 34.0;      // 4 mm of wall outboard of each bore and 6 mm between them.
                    // A 1 mm side wall would bow outward as the bore's first
                    // layers went down and the coupon would be measuring the
                    // wall, not the roof.
TD_D   = 14.0;
TD_BORE_X = 8.0;    // bore axis, either side of the block centre
TD_H   = 22.0;      // tall enough to bury the teardrop's point, which reaches a
                    // full bore diameter above the axis, with 4 mm of cover over
                    // it. A point that breaks the top face is a slot, not a bore.
TD_BORE_Z = 8.0;    // bore axis above the block's own base

module teardrop_witness_block() {
    translate([TD_X, TD_Y, 0])
        difference() {
            translate([0, 0, PLATE_T + TD_H / 2])
                rounded_plate(TD_W, TD_D, TD_H, r = 3);
            // Control: a plain horizontal circle.
            translate([-TD_BORE_X, TD_D / 2 + EPS, PLATE_T + TD_BORE_Z])
                rotate([90, 0, 0])
                    cylinder(d = IDLER_BEARING_OD, h = TD_D + 2 * EPS);
            // Treatment: the same bore as a teardrop, point up.
            //
            // truncate = false is deliberate. util.scad's truncated variant caps
            // the profile at 0.35r above the bore centre, which is BELOW the
            // circle's own crown at r, so the cap removes the entire roof and
            // the result is indistinguishable from a plain circle. A coupon
            // whose treatment and control are the same shape measures nothing,
            // so this call takes the full-point form. Worth fixing in util.scad
            // once this print has confirmed the two bores really do differ.
            translate([TD_BORE_X, TD_D / 2 + EPS, PLATE_T + TD_BORE_Z])
                rotate([90, 0, 0])
                    teardrop(IDLER_BEARING_OD, TD_D + 2 * EPS, truncate = false);
            translate([-TD_BORE_X, 0, 0]) engrave("RND", 3, PLATE_T + TD_H);
            translate([TD_BORE_X, 0, 0]) engrave("TD", 3, PLATE_T + TD_H);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. BRIDGING / OVERHANG FAN
//
// A separate body below the plate: three unsupported roofs of 8, 15 and 22 mm
// span, and three overhangs at 45, 60 and 75 degrees from vertical.
//
// The whole robot is designed on the claim that it needs no supports, and that
// claim rests on one number -- the steepest overhang your machine prints
// cleanly. 45 degrees is the figure everyone quotes and it is a property of the
// machine, not of the material: cooling, fan duct, layer height and speed all
// move it. Print this, look at the three undersides, and decide whether the
// no-supports rule is true on YOUR printer before you trust it on a leg.
//
// The bridges are the same argument for MAX_BRIDGE, which util.scad asserts at
// 30 mm for PETG. The three spans here are well inside that, because a fit-check
// coupon should show you a roof that works and a roof that is starting to droop,
// not a guaranteed failure.
//
// The span numbers are engraved on the pad UNDER each opening, where they are
// read through the arch -- there is no room in front of the wall, and a label
// you cannot see is not a label.
// ─────────────────────────────────────────────────────────────────────────────

FAN_X = 25.0;       // pad centre; the pad spans x = -7 .. +57, y = -59 .. -41
FAN_Y = -50.0;
FAN_W = 64.0;
FAN_D = 18.0;

FAN_WALL_T = 4.0;   // wall thickness, i.e. the depth of each bridge
FAN_WALL_H = 14.0;
FAN_OPEN_H = 9.0;   // leaves a 5 mm deck above each bridged roof, so a sagging
                    // roof shows as a droop rather than as a hole

// [centre x in pad coordinates, span]
FAN_OPENINGS = [[-22.75, 8], [-7, 15], [15.75, 22]];

// [wall x in pad coordinates, angle from vertical]
FAN_FINS = [[-28, 45], [-18, 60], [-4, 75]];
FIN_H = 6.0;        // Kept short on purpose: the horizontal run of a 75-degree
                    // overhang is 3.7x its height, so a taller fin would not fit
                    // the plate and would tell you nothing extra.
FIN_D = 6.0;
FIN_BACK = 3.0;     // vertical wall behind each fin, so it has something to be
                    // built on other than a knife edge on the bed

// The overhanging face runs from (0,0) to (run, hgt); material sits above and to
// the left of it, so the face is the underside and `ang` is measured from
// vertical -- the same convention chamfer_under() encodes.
module overhang_fin(ang, hgt, dep) {
    run = hgt * tan(ang);
    rotate([90, 0, 0])
        linear_extrude(height = dep, center = true)
            polygon([[-FIN_BACK, 0], [0, 0], [run, hgt], [-FIN_BACK, hgt]]);
}

module bridge_overhang_fan() {
    translate([FAN_X, FAN_Y, 0])
        difference() {
            union() {
                translate([0, 0, PLATE_T / 2])
                    rounded_plate(FAN_W, FAN_D, PLATE_T, r = 3);
                translate([-31, 3, PLATE_T])
                    cube([62, FAN_WALL_T, FAN_WALL_H]);
                for (f = FAN_FINS)
                    translate([f[0], -5.5, PLATE_T])
                        overhang_fin(f[1], FIN_H, FIN_D);
            }
            for (o = FAN_OPENINGS)
                translate([o[0] - o[1] / 2, 3 - EPS, PLATE_T - EPS])
                    cube([o[1], FAN_WALL_T + 2 * EPS, FAN_OPEN_H + EPS]);
            for (o = FAN_OPENINGS)
                translate([o[0], 3.6, 0]) engrave(str(o[1]), 2.5);
            for (f = FAN_FINS)
                translate([f[0], -1, 0]) engrave(str(f[1]), 2.5);
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. TWO TENSILE COUPONS
//
// Two dogbones, identical in every dimension, differing only in how they lie on
// the bed: one flat, one standing on edge. Pull them both and divide.
//
// Why this cannot be inherited: two vendor PETG datasheets disagree by 60% on
// tensile strength, and neither of them describes your printer. Flat, the
// coupon's 12 mm width is mostly infill and the load crosses whatever angle the
// slicer chose; on edge, the same cross-section is almost entirely perimeters
// running along the pull. That difference is the number every load calculation
// in spec.py's print_* knockdowns is really about, and it is a property of a
// machine and a profile, not of a material.
//
// Honest limit: this pair measures PERIMETER ALIGNMENT, not layer adhesion. The
// worst case -- load pulling layers apart, straight across the Z bonds -- needs a
// third coupon printed standing on its end, and a 48 mm tall bar with a 6 mm
// waist is its own print, not a passenger on this plate. Treat the ratio you get
// here as the in-plane anisotropy and do not read it as the Z-axis knockdown.
//
// The waist is blended with a 12 mm radius on each shoulder. Without it the
// specimen breaks at the shoulder every time and measures the stress riser
// rather than the material -- which is the same fillet argument the rest of this
// robot is built on, showing up in the instrument used to measure it.
// ─────────────────────────────────────────────────────────────────────────────

COUPON_L  = 48.0;
COUPON_W  = 12.0;   // grip width
COUPON_GW = 6.0;    // gauge width
COUPON_X0 = 6.0;    // half the parallel gauge length
COUPON_R  = 12.0;   // shoulder blend radius
COUPON_T  = 3.0;
COUPON_X  = -35.0;
COUPON_FLAT_Y = -47.0;
COUPON_EDGE_Y = -57.0;  // Named rather than inlined because the footprint assert
                        // below has to measure the same number the geometry uses.
                        // A literal in both places is how a plate grows past the
                        // bed while its own assert still passes.

module coupon_waist() {
    hw = COUPON_W / 2;
    gh = COUPON_GW / 2;
    over = 2;       // overshoot past the grip edge, so no coincident face
    union() {
        translate([0, gh + (hw - gh + over) / 2])
            square([2 * COUPON_X0, hw - gh + over], center = true);
        for (s = [-1, 1])
            translate([s * COUPON_X0, gh + COUPON_R]) circle(r = COUPON_R);
    }
}

module coupon_profile() {
    difference() {
        square([COUPON_L, COUPON_W], center = true);
        coupon_waist();
        mirror([0, 1, 0]) coupon_waist();
    }
}

module tensile_coupons() {
    // Flat: layers stacked through the 3 mm thickness, load along the bed X.
    translate([COUPON_X, COUPON_FLAT_Y, 0])
        difference() {
            linear_extrude(height = COUPON_T) coupon_profile();
            // Marked in the grip, never in the gauge -- a 0.6 mm notch in the
            // waist would decide where the specimen breaks and invalidate it.
            translate([20, -1.5, 0]) engrave("FLAT", 3, COUPON_T);
        }

    // On edge: same part rotated a quarter turn about its own long axis, so the
    // 12 mm dimension is now vertical and the section is nearly all perimeter.
    translate([COUPON_X, COUPON_EDGE_Y, COUPON_W / 2])
        rotate([90, 0, 0])
            linear_extrude(height = COUPON_T, center = true) coupon_profile();
}

// ─────────────────────────────────────────────────────────────────────────────
// LETTERING
//
// One helper, used for every label on the coupon. `top` is the z of the face
// being cut into, so the same module works on the 3 mm plate and on the 22 mm
// witness block without anyone having to remember two depths.
// ─────────────────────────────────────────────────────────────────────────────

module engrave(txt, size = 3, top = PLATE_T, align = "center") {
    translate([0, 0, top - ENG_D])
        linear_extrude(height = ENG_D + EPS)
            text(txt, size = size, halign = align, valign = "baseline");
}

// The three legend blocks. They sit in the gaps the gauges leave, which is why
// their coordinates look arbitrary -- they are placed against the features, not
// on a grid. Left-aligned so a long line grows into empty plate rather than
// into a hole pattern.
module plate_legends() {
    LEG = 2.5;

    // Key to the single-letter prefixes, in the space right of the nut block.
    translate([29, 27.0, 0])   engrave("WHELP-16 FITCHECK", LEG, PLATE_T, "left");
    translate([29, 23.5, 0])   engrave("BC = HORN BOLT PCD", LEG, PLATE_T, "left");
    translate([29, 20.0, 0])   engrave("I  = INSERT BORE", LEG, PLATE_T, "left");
    translate([29, 16.5, 0])   engrave("B  = BRG UNDERSIZE", LEG, PLATE_T, "left");
    translate([29, 13.0, 0])   engrave("ALL DIMS mm", LEG, PLATE_T, "left");

    // Which nested rectangle is which, left of the servo gauge.
    translate([-57, -16.0, 0]) engrave("SERVO CASE MOUNT", LEG, PLATE_T, "left");
    translate([-57, -19.5, 0]) engrave("DX x DY mm", LEG, PLATE_T, "left");
    translate([-57, -23.0, 0]) engrave("IN  32x12", LEG, PLATE_T, "left");
    translate([-57, -26.5, 0]) engrave("MID 38x18", LEG, PLATE_T, "left");
    translate([-57, -30.0, 0]) engrave("OUT 44x24", LEG, PLATE_T, "left");

    // What to do with the answer. A coupon that is read and not acted on has
    // cost thirty minutes and retired nothing.
    translate([27, -16.0, 0])  engrave("READ THE ONE", LEG, PLATE_T, "left");
    translate([27, -19.5, 0])  engrave("THAT FITS,", LEG, PLATE_T, "left");
    translate([27, -23.0, 0])  engrave("THEN FIX", LEG, PLATE_T, "left");
    translate([27, -26.5, 0])  engrave("spec.py AND", LEG, PLATE_T, "left");
    translate([27, -30.0, 0])  engrave("RE-EXPORT.", LEG, PLATE_T, "left");
}

// ─────────────────────────────────────────────────────────────────────────────
// ASSEMBLY
// ─────────────────────────────────────────────────────────────────────────────

module fitcheck_plate() {
    union() {
        difference() {
            translate([0, PLATE_Y0, PLATE_T / 2])
                rounded_plate(PLATE_W, PLATE_D, PLATE_T, r = 4);
            // The plate is pressed against a bench while bearings are driven
            // into it, so a first-layer bulge on the underside makes it rock
            // under exactly the load that is supposed to go straight down.
            translate([0, PLATE_Y0, 0]) elephant_relief(PLATE_W, PLATE_D);

            horn_bolt_gauge_cuts();
            insert_bore_ladder_cuts();
            bearing_fit_ladder_cuts();
            m3_gauge_cuts();
            servo_mount_gauge_cuts();
            plate_legends();
        }
        insert_bore_ladder();
        bearing_fit_ladder();
        translate([NUT_X, INS_Y, 0]) nut_pocket_block();
        teardrop_witness_block();
    }
}

module fitcheck() {
    // The labels on the insert ladder are written out rather than computed,
    // because str() of 4.2 + 0.2 is not reliably "4.4". That makes them a copy
    // of a generated number, so the copy is checked here: if spec.py moves
    // INSERT_BORE, this fails at render instead of shipping a plate whose
    // engraving disagrees with its own holes.
    assert(INSERT_BORE == 4.2,
           str("INS_LABELS are written for INSERT_BORE = 4.2 mm but params_gen",
               " now says ", INSERT_BORE, " mm; rewrite the labels"));

    // Same contract for the servo mount legend.
    assert(SERVO_MOUNT_HOLE_DX == 38 && SERVO_MOUNT_HOLE_DY == 18,
           str("the servo mount legend is written for 38 x 18 mm but params_gen",
               " now says ", SERVO_MOUNT_HOLE_DX, " x ", SERVO_MOUNT_HOLE_DY,
               " mm; rewrite the candidates and the legend"));

    // A gauge that does not bracket the value it is testing cannot answer the
    // question it was printed for.
    assert(SERVO_HORN_BOLT_CIRCLE_D >= min(HORN_BCS)
           && SERVO_HORN_BOLT_CIRCLE_D <= max(HORN_BCS),
           str("horn bolt circle ", SERVO_HORN_BOLT_CIRCLE_D,
               " mm is outside the candidates on this gauge; widen HORN_BCS"));

    // The layout is within a millimetre of the 120 mm budget in X, so this is
    // not a formality: one more candidate bolt circle, or one more millimetre of
    // pitch in the horn row, and the plate stops fitting.
    assert(FOOT_X <= 120 && FOOT_Y <= 120,
           str("fitcheck footprint ", FOOT_X, " x ", FOOT_Y,
               " mm exceeds the 120 x 120 mm budget"));
    assert(FOOT_X <= BED_X && FOOT_Y <= BED_Y,
           str("fitcheck footprint ", FOOT_X, " x ", FOOT_Y,
               " mm does not fit the ", BED_X, " x ", BED_Y, " mm bed"));

    fitcheck_plate();
    tensile_coupons();
    bridge_overhang_fan();
}

// Overall footprint: the plate's top edge down to whichever of the two lowest
// bodies actually reaches furthest, taken as a min() rather than named. The
// on-edge coupon looks like the obvious answer and is not: the fan pad's near
// edge sits at FAN_Y - FAN_D/2 = -59, half a millimetre beyond it. Asserting
// against the wrong body is worse than not asserting, because it reports a
// footprint the plate does not have and keeps reporting it as the layout moves.
FOOT_X = PLATE_W;
FOOT_Y = (PLATE_Y0 + PLATE_D / 2)
         - min(COUPON_EDGE_Y - COUPON_T / 2, FAN_Y - FAN_D / 2);

echo(str("fitcheck footprint ", FOOT_X, " x ", FOOT_Y, " mm"));

fitcheck();
