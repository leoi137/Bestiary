// fuse.scad — the sacrificial link between the calf and the wheel mount.
//
// The cheapest part on the robot and the one with the clearest job: BE THE THING
// THAT BREAKS. Two grams, four minutes to print, two M3 to swap.
//
// WHY IT EXISTS, IN ONE PARAGRAPH
// -------------------------------
// The leg's other failure thresholds, in newtons at the contact patch, are: the
// servo's own torque limit at 26 N (but only for SLOW loads -- a 1:345 gearbox
// reflects so much rotor inertia that it is rigid for anything shorter than
// about 100 ms), the output gear train at ~92 N, and the calf itself at ~470 N
// laterally. A side impact -- tipping over, or driving a wheel into a table leg
// -- lands somewhere in the hundreds of newtons, which is above the calf. So
// without this part, a fall breaks a 40-gram leg that takes three hours to
// reprint. With it, a fall breaks this instead.
//
// SIZING IS A TWO-SIDED CONSTRAINT AND BOTH SIDES MATTER
// -----------------------------------------------------
// Strong enough never to trip in normal use: each wheel carries about 5.6 N
// standing and 24 N in a design landing, and this shears at roughly 280 N, so
// there is better than 10x headroom. A fuse that nuisance-trips is worse than no
// fuse at all, because whoever is building the robot will simply leave it out
// and then have neither.
//
// Weak enough to go before the calf: 280 N against the calf's ~470 N in lateral
// bending. torque.py computes both and check.py asserts the ordering, so
// changing FUSE_SHEAR_AREA in spec.py cannot silently invert it.
//
// WHAT IT DOES *NOT* PROTECT
// --------------------------
// A hard VERTICAL landing. To trip before the gear train's ~92 N this web would
// have to be about 2 mm^2 -- unprintable, and close enough to a real landing
// that it would nuisance-trip. Vertical impact is therefore bounded by the drop
// envelope in the torque report rather than by a part, and that is a real
// conclusion rather than a gap: you cannot fuse a rigid gearbox in series
// without the fuse becoming the thing that fails constantly.
//
// THE SHEAR STRENGTH IS AN ASSUMPTION
// -----------------------------------
// spec.print_shear_strength_mpa is 28 MPa, taken as ~0.6x PETG's tensile yield,
// and it is Kind.ASSUMED. The whole ordering above rests on it. Its
// `replaced_by` is a ten-minute experiment: put one of these in a vice and pull
// it with a luggage scale. Do that before trusting any of the numbers in this
// comment.

include <../lib/params_gen.scad>
use <../lib/util.scad>

EPS = 0.02;

//: Bolt spacing to the calf and to the wheel mount. Two M3 each side.
FUSE_BOLT_PITCH = 22;
FUSE_LEN = 46;
FUSE_TAB_W = 20;

// The shear web's width follows from the area spec.py sizes, so the break load
// is a NUMBER IN THE SPEC rather than a shape somebody drew. Editing the web by
// eye is how a fuse stops being a fuse.
FUSE_WEB_W = FUSE_SHEAR_AREA / FUSE_T;

module fuse() {
    difference() {
        union() {
            // Two mounting tabs, full thickness, with the bolt holes.
            for (s = [-1, 1])
                translate([s * (FUSE_LEN / 2 - FUSE_TAB_W / 2), 0, 0])
                    rounded_plate(FUSE_TAB_W, FUSE_TAB_W, FUSE_T * 2.5, r = 3);

            // The web between them. Deliberately thin, and deliberately the only
            // load path: a rib bridging the two tabs would carry the load around
            // the web and the part would stop being a fuse without looking any
            // different.
            cube([FUSE_LEN, FUSE_WEB_W, FUSE_T], center = true);
        }

        // Bolt holes, counterbored so the heads sit flush against the calf.
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * FUSE_LEN / 2 * 0.72, sy * FUSE_BOLT_PITCH / 4, -FUSE_T * 1.25 - EPS])
                cap_screw_hole(M3_CLEAR, M3_HEAD_D, M3_HEAD_H, FUSE_T * 2.5 + 2 * EPS);

        // A stress-raising notch across the middle of the web, on BOTH faces.
        //
        // This is the one place on the robot where a sharp corner is wanted. Every
        // other internal corner is filleted to keep the stress concentration near
        // 1.5 instead of 3; here the concentration is the feature, because it
        // makes the break happen at a known place and at a repeatable load
        // instead of wherever the print happened to be weakest.
        for (sz = [-1, 1])
            translate([0, 0, sz * FUSE_T / 2])
                rotate([0, 45, 0])
                    cube([FUSE_T * 0.5, FUSE_WEB_W + 2, FUSE_T * 0.5], center = true);
    }
}

fuse();
