// trunk.scad — the WHELP-16 trunk: two shell halves and two lid pieces.
//
//   trunk_front()          front shell half, carries the two front abduction servos
//   trunk_rear()           rear shell half, carries the two rear abduction servos
//   trunk_lid(half)        one of the two flat covers, "front" or "rear"
//
// FRAME. Every module here is authored in the ASSEMBLED trunk frame: centre at
// the origin, +X forward, +Y left, +Z up. That is deliberate. The two halves are
// cut out of one shell by a single clip solid (lap_clip), so the tongue and the
// recess are complementary by construction rather than by two hand-kept sets of
// numbers agreeing. It also means the assembly preview is just the union of the
// four modules with no transforms to get wrong. To PRINT a half, translate it
// so the part sits on the bed; the shape is already in its print orientation.
//
// SIGN CONVENTION. The trunk is symmetric about y = 0 and is not mirrored for
// the right side; the leg parts are authored left and mirrored, this one is not.
//
// WHY IT IS SPLIT. TRUNK_L is 250 mm against a 220 mm bed. Printing on the bed
// diagonal is real advice and it is fragile advice -- it costs the skirt, the
// brim, most of the part-cooling margin, and it puts the longest unsupported
// span diagonally across the gantry's worst stiffness. So the shell splits at
// the midplane into two 138 mm halves (TRUNK_L/2 + TRUNK_LAP/2 each) joined by a
// LAP of length TRUNK_LAP plus four M3 into heat-set inserts. A lap carries the
// joint load in SHEAR over 26 mm x 53 mm of glued/bolted area on each side wall;
// a butt joint would carry it in tension straight across a layer line, which is
// the weakest direction a printed part has. The lap also cannot hinge: the
// tongue is trapped between the recess's two faces, so the joint has no degree
// of freedom left to open under a landing.
//
// ─────────────────────────────────────────────────────────────────────────────
// A DISCREPANCY THIS FILE FOUND, AND THAT spec.py THEN FIXED
// ─────────────────────────────────────────────────────────────────────────────
// When this file was first written, geometry.abduct_axis_y_mm() evaluated to
// ZERO: the generated stance put both abduction axes of a station on the trunk
// CENTRELINE, separated only in X. Two servos cannot share one line, and no
// bracket geometry makes them so.
//
// The cause was a placeholder in spec.py that collapsed two independent
// quantities into one -- WHERE the pivot sits (which sets the track) and HOW FAR
// the leg steps outboard of it (which sets the abduction moment arm). Move the
// whole leg outboard and the track widens while every torque is unchanged, so
// every number in the torque report was still correct. That is exactly why the
// bug survived: its only visible symptom was a 102 mm track under a 104 mm trunk.
//
// spec.py now carries abduct_y_mm separately from the derived moment arm, and
// check.py asserts that the kinematic pivot lands where the mass model puts the
// abduction servo, that the pivot is inside the shell, and that the leg plane
// clears the trunk's side. This file places its two servos at y = +/- ABDUCT_Y,
// output axes along X, horn faces on the planes x = +/- ABDUCT_X. No Z offset,
// and nothing here diverges from derived_gen.scad.
//
// Kept as a note rather than deleted, because the mechanism is worth seeing: a
// CAD file is a CONSUMER of the kinematics, and a consumer that cannot physically
// be built is evidence ABOUT the kinematics. An earlier draft of this file worked
// around the bug by stacking the two servos 27.2 mm apart in Z and documenting
// the mismatch. That was the right instinct and the wrong fix -- a workaround in
// the consumer leaves the source wrong for every other consumer.
//
// ─────────────────────────────────────────────────────────────────────────────
// PRINT ORIENTATION
// ─────────────────────────────────────────────────────────────────────────────
// trunk_front, trunk_rear: FLOOR DOWN, OPEN SIDE UP, exactly as modelled. Every
//   internal feature -- bulkhead, bay ribs, gussets, battery cradle, SBC posts --
//   is then a vertical wall grown off the floor or off another wall, so there is
//   not one support-needing surface inside the part. The two bosses that DO stick
//   sideways out of a wall, at the lap joint, are placed with wall_boss(), which
//   exists exactly because a horizontal cylinder has a horizontal tangent along
//   its underside and prints onto air without one.
//
//   The abduction servo bays are open to the top, and that is an assembly
//   requirement before it is a printing one: the case's mounting ears span 54 mm
//   against a 53.2 mm interior height, so a servo cannot be introduced into a
//   closed bay from any direction at all -- it has to come in from above, ears
//   last. The lid is that bay's roof, and it arrives already printed flat.
//   Footprint 138 x 104 mm on a 220 x 220 bed, 58 mm tall. Fits with room.
//   The trunk's principal load is the fore-aft bending moment from the four legs,
//   which runs along X -- in-plane with the layers in this orientation, i.e. the
//   layer lines are never the tension path.
//
// trunk_lid: printed INVERTED, rotate([180, 0, 0]), so the lip and the servo
//   hold-down pad point up and the counterbores bridge over a 3.4 mm hole rather
//   than overhang. 125 x 104 mm per piece.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT THE TRUNK OWES THE ABDUCTION JOINT, AND WHAT IT DOES NOT
// ─────────────────────────────────────────────────────────────────────────────
// The double-shear rule says: horn one side, idler bearing the other. For the
// HIP and KNEE that is a bearing pocket in the driven link riding on the servo's
// rear boss (SERVO_HAS_REAR_BOSS is true). For ABDUCTION this file does it
// differently, and the reason is a reach: ABDUCT_X is 92 mm while the trunk's end
// wall is at 125 mm, so the bracket has to come 33 mm INTO the shell to find its
// horn, and its hub therefore passes through the end wall. There is no room left
// on that axis for an IDLER_BEARING_OD race, and a 10 mm ball bearing was never
// the right part to react a leg's bending moment 200 mm out anyway.
//
// So the trunk gives the joint its second support directly, and it is a PLAIN
// JOURNAL rather than a ball bearing: a JOURNAL_D teardrop bore through a boss
// JOURNAL_L long in the end wall, in which the bracket's shaft turns. The shaft
// is inboard of the wall, the leg arm outboard, and they bolt together across it
// -- so the bracket is necessarily two pieces, which is also the only way
// anything larger than the bore gets assembled through it.
//
// STATE IT PLAINLY: this is the one joint on the robot where the double-shear
// rule is met in intent (two supports, the servo shaft relieved of the leg's
// bending moment) but not in letter (no bearing race). A plain PETG journal
// oscillating +/- 46 degrees at walking rate is a reasonable bearing and a poor
// one to be surprised by: watch it for slop, and note that it is the one place
// where the anti-backlash argument in sts3215.scad is satisfied by geometry
// rather than by a bearing.
//
// AN UPGRADE THAT IS NOW POSSIBLE AND WAS NOT. While the two servos were stacked
// in Z, a bracket arm reaching past one case to pick up its rear boss swept an
// annulus straight through the other servo -- straddling and stacking were
// mutually exclusive. At y = +/- ABDUCT_Y the cases span 25.65..48.35 mm on each
// side and leave ~51 mm clear across the centreline, so a straddling bracket with
// a rear-boss bearing now fits. That is the better joint, it is not what this file
// builds, and it is the first thing to try if the journal shows play.
//
// The trunk does not hold the bracket captive: assembly is drop-in from the open
// top, bracket first, then servo, then lid.

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

$fn = FN;

// ─────────────────────────────────────────────────────────────────────────────
// LOCAL CONSTANTS
//
// Anything here is a number params_gen.scad does not carry. Each one says what
// decides it, because the next person to change TRUNK_H needs to know which of
// these move with it and which do not.
// ─────────────────────────────────────────────────────────────────────────────

EPS = 0.02;         // CSG overshoot. util.scad has its own; `use` does not import
                    // constants, so this is a deliberate local copy, same value.
BIG = 600;          // half-space cube edge. Larger than any dimension here, so a
                    // clip cube can never end inside the part by accident.

R_OUT = 8;          // outer corner radius in plan. Not cosmetic: the trunk corner
                    // is where a tipped robot lands, and a square printed corner
                    // is a chisel that splits along the layer it lands on.

ZF = -TRUNK_H / 2 + TRUNK_WALL;             // interior floor, top surface
LI = TRUNK_L - 2 * TRUNK_WALL;              // interior length
WI = TRUNK_W - 2 * TRUNK_WALL;              // interior width
RI = R_OUT - TRUNK_WALL;                    // interior corner radius, 5.6 mm,
                                            // already above FILLET_R so the four
                                            // vertical internal corners need no
                                            // separate fillet.

SERVO_CLR = 0.4;    // pocket clearance on the servo case. Below ~0.3 a PETG
                    // pocket printed at 0.4 nozzle grips the case and the servo
                    // has to be hammered in, which cracks the bay ribs.

LAP_FIT   = 0.25;   // total slip fit across the lap faces. The lap is a sliding
                    // joint during assembly and a bonded one after; zero fit
                    // means the halves jam half-seated and the four M3 then pull
                    // against the jam instead of clamping the joint.
LAP_BAND  = TRUNK_LAP + 12;   // length of the locally-thickened wall. The extra
                    // 12 mm carries the thickening past the lap ends so the step
                    // in section is not at the same station as the joint's end.
SIDE_ZONE_Y = WI / 2 - 3 * TRUNK_WALL;      // 42.4 mm. The lap lives only on the
                    // two side walls: the floor gets a plain butt so the battery
                    // sits on one flat surface instead of rocking on a step.

// The abduction pivots sit at y = +/- ABDUCT_Y, straight from spec.py. The fit
// argument that used to justify stacking them in Z now applies laterally, and it
// passes with 1.25 mm to spare: the dimension across Y is SERVO_BODY_W (24.7), so
// at |y| = ABDUCT_Y each case spans 23.65..48.35 mm against an interior
// half-width of 49.6, and the pair leaves a 47.3 mm channel down the centreline.
// check.py asserts exactly that clearance -- against SERVO_BODY_W and against the
// INTERIOR half-width -- rather than leaving it to this comment, which is also
// what fixes the case's orientation in place: see abduct_servo_at().
BULKHEAD_X = ABDUCT_X - SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H;   // 53.5 mm
BULKHEAD_T = 3.6;   // 1.5 x TRUNK_WALL. It is the servo mount AND the transverse
                    // shear web of the half, so it is thicker than the shell.

BAY_RIB_T = 3;
// The bay ribs run INBOARD of the two servos, not outboard of them, and that is
// forced rather than chosen: the case reaches 48.35 mm against a 49.6 mm interior
// half-width, so outboard of a servo there is 1.25 mm of gap and no rib will fit
// in it. Inboard there is a 47.3 mm channel. This puts each rib's outer face
// tangent to its servo's pocket, so the rib locates the case in Y from the inside
// while the shell wall backs it from the outside, and the two ribs are the
// transverse webs across the middle of a bay whose section the pockets removed.
BAY_RIB_Y = ABDUCT_Y - SERVO_BODY_W / 2 - SERVO_CLR - BAY_RIB_T / 2;   // 21.75 mm

JOURNAL_D = 20;     // the bracket's shaft where it crosses the end wall. Sized
                    // DOWN from the horn pad (24.35 mm), not up to it: at
                    // 2 * ABDUCT_Y axis spacing, a bore wide enough to
                    // pass the horn pad would leave a 1.2 mm web between the two
                    // journals, and that web is the lower bracket's entire
                    // upward bearing surface. 20 mm leaves 7.2 mm of web and
                    // 3.0 mm of wall above the upper bore, and it makes the
                    // bracket a two-piece part joined outboard of the wall --
                    // which it had to be anyway, since nothing bigger than the
                    // bore can be threaded through it during assembly.
JOURNAL_L = 10;     // land length. What stops a plain journal going oval is bore
                    // LENGTH, not diameter; this is 0.5 x bore.

LID_T     = TRUNK_WALL * 1.5;
LID_LIP_H = 2.2;    // the lip that drops into the cavity. Without it the lid is
                    // located only by its screws and every shear load on the top
                    // face is carried by four M3 shanks in oversize holes.
LID_FIT   = 0.3;

RIM_BOSS_Y = WI / 2 - (INSERT_BORE / 2 + INSERT_WALL);   // 45.7 mm: puts the boss
                    // exactly tangent to the inner wall face, so it is a
                    // half-buried column and not a free-standing one.
RIM_X = [22, 62, 108];      // lid screw stations, per half, per side

// The pack this is cradled for. Note the LENGTH: the mass model budgets a
// 105 x 35 x 25 mm pack, and 105 mm does not go in this trunk. The clear span
// between the two abduction bulkheads is 2 * (BULKHEAD_X - BULKHEAD_T) = 99.8 mm
// and the bulkheads cannot be notched to gain it -- a notch wide enough for a
// 35 mm pack lands exactly on the lower servo's mounting screws, which sit at
// |y| = SERVO_MOUNT_HOLE_DX/2 = 19 mm with 5 mm heads. So the constraint falls on
// the battery, not on the shell: buy a pack of 98 mm or less. This is asserted
// in battery_cradle() rather than left as a comment, because the number that
// breaks it is BULKHEAD_X, which moves whenever the servo does.
BATT_L = 98;   BATT_W = 35;  BATT_H = 25;
SBC_DX = 58;   SBC_DY = 49;                 // single-board computer hole pattern
SBC_L  = 85;   SBC_W  = 56;                 // ...and its board outline
SBC_DECK_Z   = ZF + BATT_H + 6;             // board underside, clear of the pack
SBC_SPIGOT_D = 2.4;                         // locates in the board's M2.5 holes
CLAMP_X = 48;  CLAMP_Y = 32;                // retainer posts, outside the board

STRAP_X = 40;  STRAP_Y = 21;                // battery strap anchors

// ─────────────────────────────────────────────────────────────────────────────
// THE SHELL
// ─────────────────────────────────────────────────────────────────────────────

// The outer envelope: one rounded box, the full 250 mm. It is cut in two later.
module trunk_outer() {
    rounded_plate(TRUNK_L, TRUNK_W, TRUNK_H, r = R_OUT, c = 0);
}

// The interior void, as a solid, parametrised by an offset `g` in every
// direction including down through the floor.
//
// One module for the cavity is what makes the lap joint safe. The tongue, the
// recess, the local wall thickening and the lid lip are all defined as the
// material between two offsets of THIS surface, so they cannot drift out of
// agreement the way four separately-typed box sizes would. It also means the
// interior keeps the same rounded corners at every offset -- an offset box with
// square corners would put a stress riser back exactly where RI removed one.
//
// The cavity runs well past the rim because the trunk is open-topped: the lid is
// a separate part and the top face is not a printed surface at all.
module trunk_cavity(g = 0) {
    hc = TRUNK_H + 40;
    translate([0, 0, ZF - g + hc / 2])
        rounded_plate(LI + 2 * g, WI + 2 * g, hc, r = RI + g, c = 0);
}

// The cove where the walls meet the floor.
//
// This is the internal corner that carries the whole trunk's bending load into
// the floor plate, and as modelled it would be a 0.3 mm radius -- Kt near 3.0 in
// bending, against 1.48 for a FILLET_R cove. Halving the fatigue life of the
// most-loaded corner in the robot to save two grams is not a trade.
//
// Built as a 45-degree cove rather than a radius, on purpose: a 45-degree face
// is exactly the steepest overhang FDM prints unsupported, so this fillet is
// free in the print orientation while a true radius would have a near-horizontal
// tangent at the bottom.
module floor_cove() {
    r = FILLET_R;
    difference() {
        translate([0, 0, ZF + r / 2])
            rounded_plate(LI, WI, r, r = RI, c = 0);
        hull() {
            translate([0, 0, ZF - EPS])
                rounded_plate(LI - 2 * r, WI - 2 * r, EPS, r = max(RI - r, 0.5), c = 0);
            translate([0, 0, ZF + r + EPS])
                rounded_plate(LI, WI, EPS, r = RI, c = 0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE LAP JOINT
// ─────────────────────────────────────────────────────────────────────────────

// Extra wall thickness, grown INWARD, over the lap band on the two side walls.
//
// Without this the lap splits a 2.4 mm wall into two 1.2 mm leaves -- three
// extrusion widths each, which is a wall with no core at all and which the
// slicer will happily print as pure perimeter with a void down the middle.
// Doubling the wall locally makes each leaf a full TRUNK_WALL, so the joint is
// as thick as the shell it joins rather than the weakest station on the robot.
//
// It is deliberately NOT applied to the floor: the battery spans the midplane
// and a 2.4 mm step under it would turn the pack into a see-saw.
module lap_thickening() {
    intersection() {
        difference() {
            trunk_cavity(0);
            trunk_cavity(-TRUNK_WALL);
        }
        cube([LAP_BAND, TRUNK_W * 2, TRUNK_H * 2], center = true);
        difference() {
            cube([BIG, BIG, BIG], center = true);
            cube([BIG, 2 * SIDE_ZONE_Y, BIG], center = true);
        }
    }
}

// The solid that defines BOTH halves: front = shell ∩ clip(0), rear = shell −
// clip(LAP_FIT). One definition, so the tongue and the recess are complementary
// by construction and a change to TRUNK_LAP moves both.
//
// `f` is the fit allowance, and it is applied only when the clip is used as a
// SUBTRACTOR, which is why the recess ends up f wider than the tongue on every
// face and f longer than the tongue is long. A tongue that bottoms in its recess
// before the leaves touch carries the joint on its end grain, which is the one
// surface of a printed part with no perimeters running through it.
module lap_clip(f = 0) {
    union() {
        // Everything forward of the split plane.
        translate([TRUNK_LAP / 2 - f + BIG / 2, 0, 0])
            cube([BIG, BIG, BIG], center = true);
        // ...plus the tongue: the inner leaf of the thickened side walls, over
        // the lap band, reaching back into the rear half's territory.
        intersection() {
            difference() {
                trunk_cavity(f / 2);
                trunk_cavity(-TRUNK_WALL - f / 2);
            }
            translate([-f / 2, 0, 0])
                cube([TRUNK_LAP + f, TRUNK_W * 2, TRUNK_H * 2], center = true);
            difference() {
                cube([BIG, BIG, BIG], center = true);
                cube([BIG, 2 * SIDE_ZONE_Y, BIG], center = true);
            }
        }
    }
}

// Places children at one of the four lap bolt stations, oriented so that +Z
// points INWARD from the tongue's outer face -- the direction an insert is
// pressed and the direction a screw travels.
module lap_bolt_at(sy, sz) {
    translate([0, sy * (WI / 2), sz * 11])
        rotate([sy * 90, 0, 0])
            children();
}

// A boss that grows SIDEWAYS off a vertical wall, with a 45-degree cone at its
// root so that no part of it is an overhang.
//
// This is the module that makes a horizontal boss printable at all, and the
// failure without it is worth being concrete about: a 7.8 mm cylinder cantilevered
// off a wall in this print orientation has a fully horizontal tangent along its
// underside, so its lower half prints onto air. It comes out drooped, oval and
// short of round, which for an insert boss means the insert goes in cocked and
// strips on the first bolt.
//
// The cone is symmetric rather than only underneath, and that is deliberate: the
// direction "down" depends on the sign of the rotation the caller used to place
// the boss, and a buttress that has to be handed is a buttress that will one day
// be handed wrong. A frustum cannot be got wrong, costs about a gram, and puts
// the extra material exactly where a boss is loaded, which is at its root.
//
// The bore is NOT cut here. Every caller subtracts insert_hole()/cap_screw_hole()
// after unioning this, so the hole is cut last and cannot be filled by a hull.
module wall_boss(d, l, root_t = 0.8) {
    hull() {
        cylinder(d = d, h = l);
        cylinder(d = d + 2 * l, h = root_t);
    }
}

// Front half: the inserts live in the tongue, entered from its OUTER face -- the
// lap's mating surface, so each insert is pressed flush before the halves go
// together and the rear half's leaf then bears on a steel face rather than on the
// mouth of a plastic hole. Four M3 is not a lot for a 250 mm body, and it does not
// need to be: the bolts only stop the lap from sliding apart, the lap itself
// carries the bending.
module lap_insert_bosses() {
    assert(INSERT_WALL >= 1.2,
           str("lap insert boss wall ", INSERT_WALL, " mm is below the 1.2 mm ",
               "minimum; the boss will split when the insert is pressed"));
    for (sy = [-1, 1], sz = [-1, 1])
        lap_bolt_at(sy, sz)
            wall_boss(INSERT_BORE + 2 * INSERT_WALL, INSERT_L + 1.5);
}

module lap_insert_holes() {
    for (sy = [-1, 1], sz = [-1, 1])
        lap_bolt_at(sy, sz)
            insert_hole(INSERT_BORE, INSERT_L, M3_CLEAR);
}

// Rear half: a pad on the OUTSIDE so the cap head has a flat seat and a full
// counterbore. Bearing an M3 head directly on a 2.3 mm leaf would dish it, and a
// dished leaf is a lap joint that has already started to open.
module lap_head_pads() {
    for (sy = [-1, 1], sz = [-1, 1])
        translate([0, sy * (TRUNK_W / 2), sz * 11])
            rotate([sy * 90, 0, 0])
                // Mirrored so the frustum's wide end lands on the shell and its
                // flat end faces outward under the screw head. Placed with
                // wall_boss() rather than a bare cylinder for the same reason as
                // the inserts opposite it: a 10.3 mm pad standing proud of a
                // vertical wall is a horizontal overhang, and a drooped pad is not
                // a flat seat for a cap head.
                mirror([0, 0, 1])
                    wall_boss(M3_HEAD_D + 2 * 2.4, 3);
}

module lap_bolt_holes() {
    for (sy = [-1, 1], sz = [-1, 1])
        translate([0, sy * (TRUNK_W / 2 + 3), sz * 11])
            rotate([sy * 90, 0, 0])
                cap_screw_hole(M3_CLEAR, M3_HEAD_D, M3_HEAD_H, 14);
}

// ─────────────────────────────────────────────────────────────────────────────
// THE ABDUCTION STATIONS
//
// Each half's outboard bay is almost entirely servo. Two STS3215 cases, 45.2 x
// 24.7 x 35 each, sit in a bay 44 mm long, 55 mm wide and 52 mm tall -- that bay
// is most of the volume of the end of the trunk, and what is left of the shell
// there is a 2.4 mm skin. THE POCKETS ARE STRUCTURE, NOT CAVITIES: the shell
// around them is what reacts the abduction torque and the leg's landing load
// into the body, and it has just had its section gutted. That is the entire
// reason for the bulkhead, the two bay ribs and the four gussets below. Delete
// them and the end of the trunk is a thin-walled box with a hole in each end,
// loaded by a 200 mm lever, which will crack at the bay's inboard corner on some
// landing nobody is watching.
// ─────────────────────────────────────────────────────────────────────────────

// Places children in the frame of one abduction servo: origin at the centre of
// the output shaft on the horn face, output axis along +Z LOCAL, which this
// transform sends to +X global for the front pair and -X for the rear.
//
//   sx  +1 front, -1 rear
//   sy  +1 the left servo of the pair (drives the LEFT leg), -1 the right one
//
// The second rotation rolls the case about its own axis, and which way it rolls
// is FORCED, not chosen. Once the output axis is committed to X, the case's
// remaining two dimensions -- SERVO_BODY_L 45.2 (54 across the mounting ears) and
// SERVO_BODY_W 24.7 -- have to be dealt out to Y and Z, and only one dealing
// exists:
//
//   L along Z, W along Y   <- this. Case spans y = ABDUCT_Y +/- 12.35, i.e.
//                             23.65..48.35 against a 49.6 interior half-width,
//                             and z = +/- 22.6 in a 53.2 interior height. All
//                             four case screws land in the bulkhead, at
//                             y = ABDUCT_Y +/- SERVO_MOUNT_HOLE_DY/2 and
//                             z = +/- SERVO_MOUNT_HOLE_DX/2.
//   L along Y, W along Z      Case spans y = 9..63 against 49.6 -- 13.4 mm of
//                             servo outside the trunk -- and two of its four
//                             screws at y = 55, outside the shell entirely.
//
// An earlier version of this file rolled by sy*90, which is the second row. Its
// stated reason was true and incomplete: 54 mm of ear span really does not fit
// TRUNK_H's 53.2 mm of interior height, so it rolled the ears into Y instead --
// where two cases at |y| = ABDUCT_Y need 126 mm of a 99.2 mm interior. The 54 mm
// is the number to give way on, because the ears are 2.5 mm thick and their tips
// only have to clear the FLOOR: they overhang it by 0.4 mm and the pocket notches
// that much out of a 2.4 mm floor, over 2.5 mm of its length, 27 mm from the end
// wall. A 1.6 mm floor in a 2.5 mm-wide groove nowhere near a load path is the
// price, and it is a bargain against not fitting.
//
// check.py asserts the surviving clearance (`the abduction servo case fits inside
// the trunk cavity`) against SERVO_BODY_W and the interior half-width, so this
// orientation is not merely commented -- reversing it fails a check.
//
// The roll's sign follows sx, not sy, so that BOTH cable exits point at +Z: the
// harness leaves upward, runs over the bulkhead's cropped top edge, and the lid
// closes over it. The alternative roll sends it down onto the floor, where it is
// pinched between a 55 g servo and a printed surface every time the robot lands.
module abduct_servo_at(sx, sy) {
    translate([sx * ABDUCT_X, sy * ABDUCT_Y, 0])
        rotate([0, sx * 90, 0])
            rotate([0, 0, sx > 0 ? 180 : 0])
                children();
}

// The void one servo needs: its case with SERVO_CLR all round, the cone its
// harness needs to leave in, and a through-relief for the rear boss.
//
// The rear boss relief matters more than it looks. SERVO_REAR_BOSS_H is 2 mm and
// the bulkhead behind it is 3.6 mm; without the relief the boss bottoms on the
// bulkhead before the case face does, so all four mounting screws pull the case
// against a 6 mm circle instead of its mounting face and the servo sits cocked
// by whatever the moulding tolerance is. Then the output axis is not where this
// file says it is.
module abduct_pocket(sx, sy) {
    abduct_servo_at(sx, sy)
        union() {
            sts3215_body(SERVO_CLR);
            sts3215_cable_envelope();
            translate([0, 0, -SERVO_HORN_FACE_TO_BODY - SERVO_BODY_H - BULKHEAD_T - 2])
                cylinder(d = SERVO_REAR_BOSS_D + 3, h = BULKHEAD_T + 4);
        }
}

// The servo's own case screws, drilled from the back of the bulkhead forward.
//
// sts3215_mount_holes() drills from the case's rear-face plane in the +axis
// direction, which is where the bracket that holds the servo is assumed to be.
// Here that bracket is the bulkhead, and the bulkhead is BEHIND that plane, so
// the pattern is shifted back by its thickness before drilling, and the 0.6 mm
// extra guarantees the hole breaks cleanly out of the rear face rather than
// leaving a one-layer film for somebody to discover with a screw.
//
// NOT counterbored, deliberately. A 2.5 mm head recess in a 3.6 mm plate leaves
// 1.1 mm of plastic to carry the whole preload of a servo mount; the heads stand
// proud inside the electronics bay instead, where there is nothing for them to
// foul, and bear on the full thickness.
module abduct_mount_holes(sx, sy) {
    abduct_servo_at(sx, sy)
        translate([0, 0, -BULKHEAD_T - 0.6])
            sts3215_mount_holes(depth = BULKHEAD_T + 8, counterbore = false);
}

// The transverse bulkhead: servo mount and shear web in one part.
//
// One bulkhead per half carries both of that half's servos, because both sit at
// the same X station. It closes the end bay against the electronics bay, which
// is also what makes the battery's fore-aft stop free: the pack is captured in X
// between the two bulkheads by structure that was going to be there anyway, and
// needs no printed end stop. It is also what LIMITS the pack -- see BATT_L.
//
// Its top is held LID_LIP_H + 0.4 below the rim so the lid's lip runs over it
// unbroken. A lip notched at the bulkhead would be a lip with a hinge in it.
module abduct_bulkhead(sx) {
    difference() {
        intersection() {
            trunk_cavity(0);
            translate([sx * (BULKHEAD_X - BULKHEAD_T / 2), 0, 0])
                cube([BULKHEAD_T, TRUNK_W * 2, TRUNK_H * 2], center = true);
        }
        translate([0, 0, TRUNK_H / 2 - LID_LIP_H - 0.4 + BIG / 2])
            cube([BIG, BIG, BIG], center = true);
    }
}

// A cable pass-through whose edges are rounded on both faces.
//
// Two failures this avoids. First, a printed edge is a saw: silicone servo wire
// dragged over a 2.4 mm as-printed corner a few thousand times abrades through
// the jacket, and a serial bus does not fail one joint at a time -- a short takes
// the whole chain down. Second, a plain round hole with a horizontal axis prints
// its own roof as a sag; teardrop() replaces that roof with two 45-degree faces.
// The flares are hulls between the bore and a larger teardrop at each face, so
// both mouths get a 45-degree lead and neither is a horizontal ledge.
module cable_port(d, len, ch = 2) {
    union() {
        teardrop(d, len + 2 * EPS, center = true);
        for (s = [-1, 1])
            translate([0, 0, s * len / 2])
                hull() {
                    teardrop(d, EPS, center = true);
                    translate([0, 0, s * ch])
                        teardrop(d + 2 * ch, EPS, center = true);
                }
    }
}

// Harness ports through the bulkhead, INBOARD of everything the servo needs.
//
// The bulkhead is the only transverse web in the half, and its X station is
// crowded: at |y| = ABDUCT_Y there is the rear-boss relief, and at
// |y| = ABDUCT_Y +/- SERVO_MOUNT_HOLE_DY/2 (27 and 45 mm) there are two of the
// four case screws. A port on the servo's own centreline merges with the boss
// relief and turns two small holes into one large one through the web -- and a
// port that merges with a screw hole means the screw has nothing to bear on. So
// the ports sit at |y| = 12, in the channel between the two bay ribs, which is
// where the harness runs anyway.
//
// The double rotation is not decoration: the first lays the bore along X, the
// second rolls the teardrop so its point faces UP. A teardrop pointing sideways
// is a plain hole with a lump on it, and it sags exactly like the plain hole it
// has become.
module bulkhead_ports(sx) {
    for (sy = [-1, 1])
        translate([sx * (BULKHEAD_X - BULKHEAD_T / 2), sy * 12, 0])
            rotate([0, 90, 0])
                rotate([0, 0, 90])
                    cable_port(10, BULKHEAD_T + 4);
}

// The two ribs down the middle of the servo bay, one just inboard of each case.
//
// They locate the servos in Y against the shell wall (0.4 mm of pocket clearance
// on each side of the case, and no more), and they are the only full-height
// vertical webs between the bulkhead and the end wall -- the section the pockets
// removed, put back in the one place it is not in the way, which is the 47.3 mm
// channel between the two cases. The channel between the two RIBS is in turn the
// route the leg harnesses take forward to the bulkhead ports.
//
// They stop short of the rim by 0.4 mm plus the lid's lip, because the lip runs
// the length of the half and a lip with a notch in it is a lip with a hinge.
module abduct_bay_ribs(sx) {
    len = ABDUCT_X + 6 - BULKHEAD_X;
    hgt = TRUNK_H - 2 * TRUNK_WALL - 0.4;
    for (sy = [-1, 1])
        // Rooted ON the floor, not centred on z = 0: a rib that floats even
        // 0.2 mm above the floor prints its first layer in mid-air and is a
        // decoration rather than a web.
        translate([sx * (BULKHEAD_X + len / 2), sy * BAY_RIB_Y, ZF + hgt / 2])
            cube([len, BAY_RIB_T, hgt], center = true);
}

// Gussets tying the bulkhead into the floor.
//
// The bulkhead's job is to take the abduction reaction out of four M3 screws and
// spread it into the shell; a plate standing on a floor takes that load as a
// peel at its root. A gusset turns the peel into shear and its stiffness goes as
// the cube of its height, so 16 mm of triangle is worth far more than 16 mm of
// extra plate thickness would be.
module abduct_gussets(sx) {
    for (sy = [-1, 1], gy = [32, 44])
        translate([sx * (BULKHEAD_X - BULKHEAD_T), sy * gy, ZF])
            rotate([0, 0, sx > 0 ? 180 : 0])
                gusset(18, 16, 3);
}

// The bosses that give the two journals their length, inside the end wall.
//
// TRUNK_WALL alone would be a 2.4 mm land on a 20 mm bore: a knife edge, which
// wears into a taper and hands the abduction joint a few degrees of slop. The
// two bosses merge in the middle, which is wanted -- the material between the
// bores is what the LOWER bracket bears up against, and it is the thinnest
// loaded section at this station.
//
// Clipped to the cavity so it cannot grow through the shell, and clipped below
// the lid's lip, which runs the length of the half and must not be interrupted.
module abduct_journal_boss(sx) {
    for (s = [-1, 1])
        intersection() {
            translate([sx * (TRUNK_L / 2 - TRUNK_WALL - JOURNAL_L), s * ABDUCT_Y, 0])
                rotate([0, sx * 90, 0])
                    cylinder(d = JOURNAL_D + 8, h = JOURNAL_L);
            trunk_cavity(0);
            translate([0, 0, TRUNK_H / 2 - LID_LIP_H - 0.4 - BIG / 2])
                cube([BIG, BIG, BIG], center = true);
        }
}

// The two journals the abduction brackets turn in, through the end wall.
//
// TEARDROPS, not circles, and the reason is worth stating because it looks like
// it costs a bearing surface. A 20 mm horizontal bore printed as a circle sags
// its own roof into the bore and comes out undersize and oval across the top --
// exactly where the bracket bears, because the ground reaction is upward. The
// teardrop replaces that roof with two 45-degree faces, and a shaft pressed
// upward into a 45-degree gable seats in a V: two lines of contact instead of
// one, self-centring, and printed at the one angle FDM makes cleanly. The bore
// is better for being a teardrop, not merely printable.
//
// Both mouths are flared, so nothing the bracket sweeps past is a square edge.
module abduct_window(sx) {
    ch    = 2.5;
    depth = TRUNK_WALL + JOURNAL_L;
    for (s = [-1, 1])
        translate([sx * (TRUNK_L / 2 - depth / 2), s * ABDUCT_Y, 0])
            rotate([0, sx * 90, 0])
                rotate([0, 0, sx * 90])
                    cable_port(JOURNAL_D, depth, ch);
}

// ─────────────────────────────────────────────────────────────────────────────
// WHAT LIVES INSIDE
// ─────────────────────────────────────────────────────────────────────────────

// Cradle walls for the battery, plus one strap anchor per side per half.
//
// The pack is the heaviest single item in the robot and it sits at the bottom of
// the shell where it does the most good for the CoM height. Two things then
// matter: it must not move (a 200 g mass sliding 5 mm during a landing is a
// disturbance the policy never saw in simulation), and it must not be trapped by
// anything printed, because a swollen LiPo has to come out.
//
// So it is captured in Y by two ribs, in X by the two bulkheads (see
// abduct_bulkhead), and in Z by a strap through two M3 anchors -- and nothing
// printed overhangs it, so it lifts straight out.
module battery_cradle() {
    assert(BATT_L + 1.5 <= 2 * (BULKHEAD_X - BULKHEAD_T),
           str("battery ", BATT_L, " mm does not fit the ",
               2 * (BULKHEAD_X - BULKHEAD_T),
               " mm clear span between the abduction bulkheads"));
    len = BULKHEAD_X - BULKHEAD_T - 20;
    for (sx = [-1, 1], sy = [-1, 1]) {
        translate([sx * (20 + len / 2), sy * (BATT_W / 2 + 1.5), ZF + 6])
            cube([len, 3, 12], center = true);
        // Strap anchor. The 45-degree flare at the root is not decoration: a
        // free-standing boss on a floor is a lever with a stress riser at its
        // base, and this one gets pulled on every time the strap is tensioned.
        translate([sx * STRAP_X, sy * STRAP_Y, ZF]) {
            cylinder(d1 = INSERT_BORE + 2 * INSERT_WALL + 5,
                     d2 = INSERT_BORE + 2 * INSERT_WALL, h = 2.5);
            cylinder(d = INSERT_BORE + 2 * INSERT_WALL, h = INSERT_L + 2.5);
        }
    }
}

// The strap inserts, pressed in from ABOVE -- which is the only side an iron can
// reach once the shell exists, and the reason these bosses are not built with
// insert_boss(): that module puts the insert's mouth at the boss's base, which
// here would be under the floor.
//
// Their screw relief runs out through the floor as a 3.4 mm hole, deliberately.
// The anchor is only 8 mm tall, so containing the relief would mean a 17 mm
// tower; letting it through instead means no strap screw can ever bottom out and
// jack its insert back through the boss, and the trunk gets four drain holes at
// its lowest points. The shell is not sealed in any case -- both ends are open
// windows for the abduction brackets.
module battery_strap_holes() {
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * STRAP_X, sy * STRAP_Y, ZF + INSERT_L + 2.5])
            rotate([180, 0, 0])
                insert_hole(INSERT_BORE, INSERT_L, M3_CLEAR);
}

// Posts for the single-board computer.
//
// The board is 85 mm long and neither half has 85 mm of clear floor -- the
// bulkheads are only 107 mm apart and the battery is under it -- so the board
// necessarily STRADDLES the split. That is turned into an advantage: its 58 mm
// hole pattern puts two posts in each half, so the board itself is a fifth
// fastener across the joint, and a half that is not properly seated cannot be
// bolted to the board at all. It is a fit check that happens during assembly
// rather than after.
//
// The locating posts carry a SPIGOT, not a thread. The board's holes are M2.5
// and params_gen.scad carries no M2.5 insert, and a bare tapped M2.5 hole in
// PETG is exactly what insert_boss() exists to prevent -- it holds for three
// assemblies and then spins. So the spigots locate the board, and four M3
// inserts outboard of its outline take a retainer that clamps it. Steel threads
// where there is a thread, plastic only where there is a peg.
module sbc_posts() {
    assert(CLAMP_X > SBC_L / 2 && CLAMP_Y > SBC_W / 2,
           str("SBC retainer posts at ", CLAMP_X, ", ", CLAMP_Y,
               " are underneath a ", SBC_L, " x ", SBC_W, " mm board"));
    h = SBC_DECK_Z - ZF;
    for (sx = [-1, 1], sy = [-1, 1]) {
        translate([sx * SBC_DX / 2, sy * SBC_DY / 2, ZF]) {
            cylinder(d1 = 14, d2 = 8, h = 3.5);
            cylinder(d = 8, h = h);
            translate([0, 0, h - EPS])
                cylinder(d = SBC_SPIGOT_D, h = 3);
        }
        translate([sx * CLAMP_X, sy * CLAMP_Y, ZF]) {
            cylinder(d1 = 14, d2 = INSERT_BORE + 2 * INSERT_WALL, h = 3.5);
            cylinder(d = INSERT_BORE + 2 * INSERT_WALL, h = h);
        }
    }
}

// Entered from the top of each clamp post. The post is 31 mm tall, so the screw
// relief below the insert stays inside it and a long M3 has somewhere to go.
module sbc_holes() {
    h = SBC_DECK_Z - ZF;
    for (sx = [-1, 1], sy = [-1, 1])
        translate([sx * CLAMP_X, sy * CLAMP_Y, ZF + h])
            rotate([180, 0, 0])
                insert_hole(INSERT_BORE, INSERT_L, M3_CLEAR);
}

// Lid screw bosses, hung from the rim on the inside of the side walls.
//
// RIM_BOSS_Y puts each boss tangent to the inner wall face, so it is a
// half-buried column rather than a free-standing one and its own wall is backed
// by the shell. The cone underneath is a 45-degree taper to nothing: a boss
// hanging off a rim has a horizontal underside, which is the one surface this
// print orientation cannot make, and a 45-degree cone is the cheapest way to
// never need support inside a closed box.
//
// They hang from LID_LIP_H below the rim, not from the rim, because the lid's
// lip occupies that band and runs straight over them. The boss tops are what the
// lip lands on, so the lid is located by a continuous ring and seated on six
// pads per piece rather than bearing on a 2.4 mm wall edge.
module rim_bosses() {
    for (sx = [-1, 1], sy = [-1, 1], rx = RIM_X)
        translate([sx * rx, sy * RIM_BOSS_Y, TRUNK_H / 2 - LID_LIP_H])
            rotate([180, 0, 0]) {
                insert_boss(INSERT_BORE, INSERT_L, M3_CLEAR, INSERT_WALL);
                translate([0, 0, INSERT_L + 1.5 - EPS])
                    cylinder(d1 = INSERT_BORE + 2 * INSERT_WALL, d2 = 0.4,
                             h = (INSERT_BORE + 2 * INSERT_WALL) / 2);
            }
}

module rim_boss_holes() {
    for (sx = [-1, 1], sy = [-1, 1], rx = RIM_X)
        translate([sx * rx, sy * RIM_BOSS_Y, TRUNK_H / 2 - LID_LIP_H])
            rotate([180, 0, 0])
                insert_hole(INSERT_BORE, INSERT_L, M3_CLEAR);
}

// ─────────────────────────────────────────────────────────────────────────────
// THE WHOLE SHELL, BEFORE IT IS CUT IN TWO
// ─────────────────────────────────────────────────────────────────────────────
//
// Order is load-bearing here. The cavity is subtracted from the outer box FIRST,
// then everything internal is unioned into the result, then every void is cut
// LAST. Union the bulkhead before subtracting the cavity and the cavity deletes
// it; cut the servo pockets before unioning the ribs and the ribs fill them
// back in. Both mistakes render as a plausible-looking solid.
module trunk_shell_common() {
    difference() {
        union() {
            difference() {
                trunk_outer();
                trunk_cavity(0);
                // Elephant-foot relief on the printed first layer. The two halves
                // are clamped against a flat surface while the lap is bolted, and
                // a 0.2 mm bulge on either bottom edge rocks the half by a few
                // tenths of a degree over 138 mm -- which lands as a fore-aft
                // misalignment of the two abduction axes.
                translate([0, 0, -TRUNK_H / 2])
                    elephant_relief(TRUNK_L, TRUNK_W);
            }
            floor_cove();
            lap_thickening();
            for (sx = [-1, 1]) {
                abduct_bulkhead(sx);
                abduct_bay_ribs(sx);
                abduct_gussets(sx);
                abduct_journal_boss(sx);
            }
            rim_bosses();
            battery_cradle();
            sbc_posts();
        }
        for (sx = [-1, 1]) {
            for (sy = [-1, 1]) {
                abduct_pocket(sx, sy);
                abduct_mount_holes(sx, sy);
            }
            abduct_window(sx);
            bulkhead_ports(sx);
        }
        rim_boss_holes();
        battery_strap_holes();
        sbc_holes();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// THE PARTS
// ─────────────────────────────────────────────────────────────────────────────

// Front half. Print floor down, open side up, no supports, no rotation.
module trunk_front() {
    assert(TRUNK_L / 2 + TRUNK_LAP / 2 <= BED_X && TRUNK_W <= BED_Y,
           str("trunk half ", TRUNK_L / 2 + TRUNK_LAP / 2, " x ", TRUNK_W,
               " mm does not fit the ", BED_X, " x ", BED_Y, " mm bed"));
    difference() {
        union() {
            intersection() {
                trunk_shell_common();
                lap_clip(0);
            }
            lap_insert_bosses();
        }
        lap_insert_holes();
    }
}

// Rear half. Same orientation. It is the mirror of the front only in the lap:
// everything else is symmetric, so the two are separate modules rather than one
// module with a sign, which would have to carry the tongue/recess choice through
// every feature below it.
module trunk_rear() {
    difference() {
        union() {
            difference() {
                trunk_shell_common();
                lap_clip(LAP_FIT);
            }
            lap_head_pads();
        }
        lap_bolt_holes();
    }
}

// The pad on the lid's underside that clamps one abduction servo down.
//
// Its footprint is decided entirely by what it has to miss, which is worth
// spelling out because all three obstacles are invisible from the lid:
//
//   in X, above  RIM_X[1] + boss radius   the shell's middle lid-screw boss hangs
//                                         down into the same band
//   in X, below  the mounting ear         which stands 27 mm off the axis, i.e.
//                                         4.4 mm ABOVE the case's own top face
//   in Y         SERVO_BODY_W - 4         so it lands on the case's top face and
//                                         not on the pocket wall beside it
//
// and in Z it runs from the case's nominal top face to the lid's underside, so it
// bottoms on the servo with no gap. That is the point of it: the case is otherwise
// located in Z only by four self-tapping screws in a 3.6 mm plate, and screws in
// plastic creep. If an assembled lid stands proud of the rim, the material to
// remove is this pad -- never the rim, which is what sets the lid's flatness.
//
// Printed with the lid inverted it is a 6.4 mm block standing on the plate, i.e.
// ordinary walls, and the harness slot through its middle is cut by the servo's
// own cable envelope so it is where the cable is rather than where it was guessed.
module lid_hold_down(sx, sy) {
    ear_x = ABDUCT_X - (SERVO_HORN_FACE_TO_BODY + SERVO_BODY_H / 2
                        - SERVO_EAR_Z_FROM_BODY_MID);
    x0  = RIM_X[1] + (INSERT_BORE + 2 * INSERT_WALL) / 2 + 1.2;
    x1  = ear_x - SERVO_EAR_THICK / 2 - 1.2;
    z0  = SERVO_BODY_L / 2;
    hgt = TRUNK_H / 2 - z0;
    assert(x1 > x0 + 6,
           str("lid hold-down pad has only ", x1 - x0, " mm of x between the rim ", 
               "boss at ", RIM_X[1], " and the servo ear at ", ear_x));
    translate([sx * (x0 + x1) / 2, sy * ABDUCT_Y, z0 + hgt / 2])
        cube([x1 - x0, SERVO_BODY_W - 4, hgt], center = true);
}

// One lid piece, "front" or "rear".
//
// The lid carries no structure -- it is a dust and debris cover over a box whose
// stiffness comes from its walls, its floor and its bulkhead -- with one
// exception, and the exception is load-bearing: a pad on its underside presses on
// each of its half's two abduction servos. Those cases are located in Z by four
// self-tapping case screws each, into a 3.6 mm plate, and screws in plastic creep;
// the pad takes the case's top face and turns the mount into a clamp. Without it
// an abduction axis is free to walk downward by the screw clearance, which is
// backlash appearing in the joint with the longest lever on the robot.
//
// Two pieces, seamed at x = 0, while the SHELL seams at x = TRUNK_LAP/2. The two
// seams are deliberately 13 mm apart: a lid joint stacked directly over a shell
// joint would make one continuous plane of weakness across the whole body.
//
// PRINT: inverted, rotate([180, 0, 0]). The lip and the pad then point up and
// print as ordinary walls, and each screw counterbore becomes a 5.5 mm bridge
// over a 3.4 mm hole, which is well inside anything an FDM machine can span.
module trunk_lid(half = "front") {
    sx    = (half == "front") ? 1 : -1;
    len   = TRUNK_L / 2;
    lip_l = len - 27;
    lip_c = sx * (len / 2 + 8);
    z_top = TRUNK_H / 2 + LID_T;

    difference() {
        union() {
            difference() {
                union() {
                    translate([sx * len / 2, 0, TRUNK_H / 2 + LID_T / 2])
                        rounded_plate(len, TRUNK_W, LID_T, r = R_OUT, c = 0);

                    // The locating lip. A ring rather than a full plate: a plate
                    // would foul the bulkhead, the bay ribs and the journal boss,
                    // and it would seal the harness channels the ribs leave
                    // against the side walls.
                    translate([lip_c, 0, TRUNK_H / 2 - LID_LIP_H / 2])
                        difference() {
                            rounded_plate(lip_l, WI - 2 * LID_FIT, LID_LIP_H,
                                          r = RI, c = 0);
                            rounded_plate(lip_l - 12, WI - 2 * LID_FIT - 12,
                                          LID_LIP_H + 2 * EPS,
                                          r = max(RI - 6, 1), c = 0);
                        }
                }
                // The lip descends into the same 2.2 mm band that the servos'
                // mounting ears reach up into: the ears span +/- 27 mm about the
                // axis, the lip's underside is at 26.8, so without this notch the
                // lid rocks on four ear tips and never seats. Cut with the servo's
                // own solid rather than a hand-placed box, so a corrected ear
                // dimension moves the notch by itself.
                //
                // Subtracted BEFORE the hold-down pads are unioned in, on purpose.
                // The pad has to touch the case, and a pad cut by the same
                // clearance-grown body would stand 0.4 mm off it -- which is
                // precisely the clearance the pad exists to take out.
                for (sy = [-1, 1])
                    abduct_servo_at(sx, sy) sts3215_body(SERVO_CLR);
            }

            // Hold-down pads, one over each servo. They press each case onto its
            // mounting face, so the four case screws are not the only thing
            // resisting the output shaft's reaction moment.
            for (sy = [-1, 1])
                lid_hold_down(sx, sy);
        }

        // The harness slot through each pad. Length is measured to the lid's
        // underside rather than typed, so it opens the pad and stops there: an
        // envelope taken at its natural 18 mm would punch a 10 mm hole clean
        // through the top of the robot, and the harness's route is sideways under
        // the lid and inboard over the bulkhead, not out.
        for (sy = [-1, 1])
            abduct_servo_at(sx, sy)
                sts3215_cable_envelope(TRUNK_H / 2 - SERVO_BODY_L / 2
                                       + SERVO_CABLE_INSET);

        // Screws into the shell's rim bosses, heads flush in the top face.
        for (sy = [-1, 1], rx = RIM_X)
            translate([sx * rx, sy * RIM_BOSS_Y, z_top])
                rotate([180, 0, 0])
                    cap_screw_hole(M3_CLEAR, M3_HEAD_D, M3_HEAD_H,
                                   LID_T + LID_LIP_H + 1);

        // Lightening. Obrounds, never rectangles: a square hole in a panel that
        // gets stood on is a crack starter, and the corner radius is the whole
        // reason the hole is free. The outboard pair also happens to sit over the
        // abduction horns, so the joint can be inspected without unbolting the
        // lid.
        for (sy = [-1, 1], lx = [42, 100])
            translate([sx * lx, sy * 32, TRUNK_H / 2 + LID_T / 2])
                lightening_slot(26, 14, LID_T + 2, r = 7);

        // Printed inverted, so the bed-side face is the lid's OUTER face and the
        // first-layer bulge lands on the visible seam between lid and side wall.
        translate([0, 0, z_top])
            mirror([0, 0, 1])
                elephant_relief(TRUNK_L, TRUNK_W);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PREVIEW
//
// Left commented out on purpose. export.py may `include` this file into an
// assembly, and a live top-level call would silently duplicate the whole trunk
// at the origin -- which renders as a plausible solid and exports as a
// self-intersecting one. Uncomment to look at the part, never to ship it.
// ─────────────────────────────────────────────────────────────────────────────
//
// trunk_front();
// trunk_rear();
// trunk_lid("front");
// trunk_lid("rear");

