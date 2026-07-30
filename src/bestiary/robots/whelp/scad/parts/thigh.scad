// thigh.scad — hip axis to knee axis. One module: thigh().
//
// FRAME. Origin AT THE HIP AXIS, which runs along Y. The leg extends along -Z,
// so the KNEE AXIS is at (0, 0, -THIGH_L) exactly. Authored for the LEFT side;
// build.scad mirrors it for the right, so one printed geometry serves all four.
//
// FORM. A U-channel THIGH_D deep (fore-aft, X) by THIGH_WID wide (lateral, Y),
// walls THIGH_WALL, opening toward +Y (outboard). Depth is in the bending plane
// and bending stiffness goes as depth CUBED, which is the entire reason this is
// a channel and not a plate: at 30 mm deep it is an order of magnitude stiffer
// than a 10 mm plate of the same mass. The open side is also where the knee
// servo drops in, so structure and assembly access are one feature.
//
// PRINT ORIENTATION: lying down, long axis in the bed's XY plane, channel
// opening UP. Then the sagittal bending stress -- the load this part carries
// every step -- runs along the extrusion lines rather than across the layers.
// Nothing in the part overhangs more than 45 degrees in that orientation: the
// channel's interior corners are coved, the bearing bore is a teardrop, and the
// knee cradle's roof is open to the same +Y face the channel is.
//
// WHAT THE THIGH OWES THE TWO JOINTS
// ----------------------------------
// At the HIP end it is DRIVEN: the hip servo's horn bolts to a pad on the -Y
// (inboard) face, and an idler bearing on the +Y face rides the servo's rear
// boss. Two supports, so the servo's own output bushing never sees the leg's
// bending moment -- which is what stops the joint developing the play that
// becomes backlash that becomes a policy which cannot repeat itself.
//
// At the KNEE end it is the MOUNT: the knee servo's BODY bolts here and travels
// with the thigh. That is the direction that matters for the mass model -- a
// servo belongs to the link it is bolted to, not the one it drives -- and
// massmodel.LINK_PARTS says the same thing in Python.

include <../lib/params_gen.scad>
include <../lib/derived_gen.scad>
use <../lib/util.scad>
use <../lib/sts3215.scad>

EPS = 0.02;

// ── Local numbers. Anything params_gen.scad does not carry ───────────────────
HORN_PAD_D   = THIGH_D + 4;              // horn pad diameter at the hip end
BEAR_BOSS_D  = IDLER_BEARING_OD + 6;     // meat around the bearing pocket
CRADLE_H     = SERVO_BODY_W + 2 * THIGH_WALL;   // knee-servo cradle height
WINDOW_D     = THIGH_D * 0.45;           // lightening windows, on the neutral axis
WINDOW_Z     = [-THIGH_L * 0.38, -THIGH_L * 0.62];
CABLE_CLIP_Z = [-THIGH_L * 0.25, -THIGH_L * 0.75];

// Render-time asserts, so a regenerated params_gen.scad cannot quietly
// invalidate the numbers above. A parametric part that silently stops being
// printable when a dimension moves is worse than one that refuses to render.
module thigh_checks() {
    assert(THIGH_WID >= SERVO_BODY_W + 2 * THIGH_WALL,
           str("thigh channel ", THIGH_WID, " mm cannot straddle a ", SERVO_BODY_W,
               " mm servo case with ", THIGH_WALL, " mm walls"));
    assert(HORN_PAD_D / 2 >= SERVO_HORN_BOLT_CIRCLE_D / 2 + SERVO_HORN_BOLT_D,
           str("horn pad ", HORN_PAD_D, " mm is too small for a ",
               SERVO_HORN_BOLT_CIRCLE_D, " mm bolt circle"));
    assert(WINDOW_D + 2 * FILLET_R <= THIGH_D,
           str("lightening window ", WINDOW_D, " mm leaves no ligament in a ",
               THIGH_D, " mm section"));
    assert(THIGH_L + CRADLE_H <= BED_X,
           str("thigh ", THIGH_L + CRADLE_H, " mm does not fit a ", BED_X, " mm bed"));
}

// ── The channel ──────────────────────────────────────────────────────────────

// Outer envelope: the beam, plus the hip-end hub, plus the knee-end cradle,
// hulled so there is no step anywhere along the load path. A step in a bending
// member is a stress concentration in exactly the place the section is already
// working hardest.
module thigh_outer() {
    hull() {
        translate([0, 0, -THIGH_L / 2])
            rounded_plate(THIGH_D, THIGH_WID, THIGH_L, r = FILLET_R, c = 0.6);
        // Hip-end hub: round, because the joint is round and a square corner
        // here would be the crack starter closest to the highest moment.
        rotate([90, 0, 0]) cylinder(d = HORN_PAD_D, h = THIGH_WID, center = true);
    }
    // Knee-end cradle. Not hulled into the beam above: it is deliberately
    // squarer, because the servo case it holds is square and a round pocket in a
    // round boss wastes the wall thickness the mounting screws need.
    translate([0, 0, -THIGH_L])
        rounded_plate(THIGH_D, THIGH_WID, CRADLE_H, r = FILLET_R, c = 0.6);
}

// The cavity, open to +Y. `grow` lets the same profile be used for the coving
// pass, so the inner corners cannot drift out of agreement with the wall.
module thigh_cavity(grow = 0) {
    g = grow;
    translate([0, THIGH_WALL / 2 + g / 2, -THIGH_L / 2])
        cube([THIGH_D - 2 * THIGH_WALL + 2 * g,
              THIGH_WID - THIGH_WALL + g,
              THIGH_L - 2 * THIGH_WALL + 2 * g], center = true);
}

// Coves along the four internal corners of the channel.
//
// FILLET_R everywhere it fits, which is the cheapest strength in the design: a
// 0.3 mm as-modelled corner has a stress concentration factor near 3.0 in
// bending, a 3 mm radius near 1.5. It costs no mass and no print time, and
// skipping it roughly halves fatigue life at the same nominal stress -- while
// costing only ~8% of STATIC strength, which is why static testing makes
// notches look harmless and why this is filleted rather than measured.
module thigh_coves() {
    for (sx = [-1, 1])
        translate([sx * (THIGH_D / 2 - THIGH_WALL), THIGH_WALL, -THIGH_L + THIGH_WALL])
            rotate([0, 0, sx > 0 ? 90 : 0])
                rotate([-90, 0, 0])
                    fillet_in(FILLET_R, THIGH_L - 2 * THIGH_WALL);
}

// ── Joint interfaces ─────────────────────────────────────────────────────────

// Hip end. Horn pad and bolt pattern on -Y; bearing pocket on +Y, coaxial.
//
// The pad is a raised boss rather than a flat face so the horn seats on a
// machined-flat area that is not also the channel wall, and so HORN_BOSS_THICK
// of plastic backs each bolt. That thickness is load-bearing: torque arrives as
// a force on each screw shank BEARING against printed plastic, and bearing area
// is diameter times thickness. Thin bosses do not shear the steel screws, they
// oval out -- and an ovalled bolt hole is exactly the backlash the bearing on
// the far side was fitted to prevent.
module thigh_hip_solid() {
    translate([0, -THIGH_WID / 2, 0])
        rotate([90, 0, 0])
            cylinder(d = HORN_PAD_D, h = HORN_BOSS_THICK);
    translate([0, THIGH_WID / 2 - IDLER_BEARING_W - 1.5, 0])
        rotate([-90, 0, 0])
            cylinder(d = BEAR_BOSS_D, h = IDLER_BEARING_W + 1.5);
}

module thigh_hip_cuts() {
    // Horn bolts, from the inboard face inward.
    translate([0, -THIGH_WID / 2 - EPS, 0])
        rotate([-90, 0, 0])
            sts3215_horn_holes(HORN_BOSS_THICK + THIGH_WALL + 2);
    // Idler bearing pocket, opening outboard. bearing_pocket() carries the
    // interference: BEARING_PRESS_FIT is negative, i.e. the pocket is undersize,
    // because a printed hole comes out undersize by roughly the inward pull of
    // one extrusion and a nominal bore is a bore the race falls out of.
    translate([0, THIGH_WID / 2 + EPS, 0])
        rotate([90, 0, 0])
            bearing_pocket(IDLER_BEARING_OD, IDLER_BEARING_W, BEARING_PRESS_FIT,
                           shelf = IDLER_BEARING_SHELF,
                           bore_clear = IDLER_BEARING_ID + 1);
}

// Knee end. The servo's BODY sits here; its output axis must land exactly on
// (0, 0, -THIGH_L) or the URDF and the plastic disagree about where the knee is.
module thigh_knee_cuts() {
    translate([0, 0, -THIGH_L])
        rotate([-90, 0, 0]) {
            // The pocket, from the servo's own solid plus clearance. Derived
            // from the same module the assembly preview draws, so a corrected
            // case dimension moves both at once.
            sts3215_body(0.4);
            // The harness has to leave without being pinched by the calf's
            // sweep. A serial-bus servo has a cable at BOTH ends, and a leg
            // that pinches one takes the whole bus down, not one joint.
            sts3215_cable_envelope();
            // Case screws.
            sts3215_mount_holes(THIGH_WALL + CRADLE_H + 2);
        }
}

// ── Lightening and cable management ──────────────────────────────────────────

// Windows through the web, on the neutral axis. Removing material where the
// bending stress is zero costs almost no stiffness and buys real mass -- and on
// a 2.2 kg machine where 40% is servos, the structure is where grams come from.
//
// Teardrops, not circles: these are horizontal holes in the print orientation,
// and a circular horizontal hole has a roof that goes fully flat at the top,
// sags into the bore, and leaves a rough undersize hole. Here that only costs
// appearance; the same mistake at the bearing bore would cost the fit.
module thigh_windows() {
    for (z = WINDOW_Z)
        translate([0, THIGH_WID / 2 + EPS, z])
            rotate([90, 0, 0])
                teardrop(WINDOW_D, THIGH_WID + 2 * EPS);
}

// Retention slots for the bus harness along the outboard face. Two cables run
// past this link (the knee servo's and the wheel drive's daisy chain), and an
// unretained loom is a loom that finds the wheel.
module thigh_cable_clips() {
    for (z = CABLE_CLIP_Z)
        translate([THIGH_D / 2 - THIGH_WALL / 2, THIGH_WID / 2 - 3, z])
            rotate([0, 90, 0])
                teardrop(SERVO_CABLE_D + 1.2, THIGH_WALL + 2 * EPS, center = true);
}

// ── The part ─────────────────────────────────────────────────────────────────
module thigh() {
    thigh_checks();
    difference() {
        union() {
            difference() {
                thigh_outer();
                thigh_cavity();
            }
            thigh_coves();
            thigh_hip_solid();
        }
        thigh_hip_cuts();
        thigh_knee_cuts();
        thigh_windows();
        thigh_cable_clips();
        // Bottom-face squish relief, so the printed face seats flat against the
        // knee servo rather than rocking on an elephant foot.
        translate([0, 0, -THIGH_L - CRADLE_H / 2])
            elephant_relief(THIGH_D, THIGH_WID);
    }
}

thigh();
