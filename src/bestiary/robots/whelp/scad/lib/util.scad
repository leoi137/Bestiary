// util.scad — the printing techniques, expressed as geometry.
//
// Every module here exists because of a specific way an FDM part fails. None
// of them are conveniences. If you delete one, you get the failure back.
//
//   support-free      teardrop(), chamfer_under(), bridge_slot()
//   layer adhesion    print orientation is a per-part decision; these keep the
//                     stressed direction in-plane by making walls, not slabs
//   stress risers     fillet_in(), rounded_plate(), no sharp internal corners
//   fastener pull-out insert_boss(), and the boss wall minimum it enforces
//   first-layer squish elephant_relief()
//
// CONVENTIONS
//   * All dimensions in MILLIMETRES. The exporter divides by 1000 for the URDF,
//     in exactly one place, and check.py asserts the whole robot is under a
//     metre so a missed conversion is caught rather than shipped.
//   * Subtracted solids always overshoot the surface they cut by EPS. A
//     difference() whose faces are exactly coincident produces zero-thickness
//     shells that render fine in preview and export as non-manifold STL. That
//     failure is invisible until the slicer or stl.py complains, so the fix is
//     structural: never author a coincident face.
//   * $fn is set by the caller via the generated params file, not here, so that
//     a preview render and an export render differ only in that one number.

EPS = 0.02;         // CSG overshoot, mm. Larger than float noise, smaller than
                    // any printable feature, so it can never change a fit.

// ─────────────────────────────────────────────────────────────────────────────
// SUPPORT-FREE FEATURES
// ─────────────────────────────────────────────────────────────────────────────

// A horizontal hole that prints without support.
//
// A circular hole whose axis is parallel to the bed has a roof that goes fully
// horizontal at the top, which sags into the bore and leaves the hole undersize
// and rough exactly where a bearing or a screw needs it. A teardrop replaces the
// roof with two 45-degree faces, which every FDM machine bridges cleanly.
//
// `d`   bore diameter
// `h`   length along the bore axis (the bore runs along +Z; rotate to place it)
// `truncate` cuts the point off at 45 degrees so the profile is a "hexagon-ish"
//   teardrop rather than a spike. Costs nothing, and the sharp point of a full
//   teardrop is a stress riser and a place stringing collects.
module teardrop(d, h, truncate = true, center = false) {
    r = d / 2;
    translate([0, 0, center ? -h / 2 : 0])
        linear_extrude(height = h)
            union() {
                circle(r = r);
                // The 45-degree roof: a square rotated 45 degrees, sitting on the
                // circle's centre, reaches sqrt(2)*r above it. Intersecting with a
                // band caps the point.
                intersection() {
                    rotate(45) square([r * sqrt(2), r * sqrt(2)]);
                    if (truncate)
                        translate([-r, -r]) square([2 * r, r * (truncate ? 1.35 : 2)]);
                    else
                        translate([-r, -r]) square([2 * r, 4 * r]);
                }
            }
}

// A 45-degree chamfer that turns an overhang into a printable slope.
//
// Placed under any horizontal face that would otherwise be unsupported. The
// rule this encodes: FDM prints 45 degrees reliably and 60 degrees badly, so a
// part with no face steeper than 45 from vertical needs no support at all --
// and a part that needs no support has no support scars on the surfaces that
// locate a bearing or a servo.
//
// `w`,`d` footprint of the face being supported; `hgt` chamfer height.
module chamfer_under(w, d, hgt) {
    hull() {
        translate([0, 0, hgt - EPS]) cube([w, d, EPS], center = true);
        cube([max(w - 2 * hgt, EPS), max(d - 2 * hgt, EPS), EPS], center = true);
    }
}

// A slot whose flat roof is short enough to bridge unsupported.
//
// `span` is checked against MAX_BRIDGE at render time: an unsupported flat roof
// longer than that sags visibly, and a sagging roof over a servo pocket means
// the servo does not go in. Failing loudly at render beats discovering it after
// a four-hour print.
MAX_BRIDGE = 30;    // mm. Conservative for PETG, which bridges worse than PLA
                    // because it stays soft longer. Raise only after testing
                    // your own machine with a bridging tower.
module bridge_slot(span, d, h) {
    assert(span <= MAX_BRIDGE,
           str("bridge_slot span ", span, " mm exceeds MAX_BRIDGE ", MAX_BRIDGE,
               " mm; split the slot or add a sacrificial centre rib"));
    translate([-span / 2, -d / 2, 0]) cube([span, d, h]);
}

// Relief for first-layer elephant foot.
//
// The first layer is squashed into the bed and spreads by roughly 0.1-0.3 mm
// per side. On a part whose bottom face has to seat flat against another part,
// that bulge becomes a rock. A 0.5 mm x 45-degree chamfer on the bottom edge
// gives the squish somewhere to go.
//
// Subtract this from the bottom of any part with a mating bottom face.
module elephant_relief(w, d, c = 0.5) {
    translate([0, 0, -EPS])
        difference() {
            translate([-w, -d, 0]) cube([2 * w, 2 * d, c + EPS]);
            hull() {
                translate([0, 0, c]) cube([w, d, EPS], center = true);
                translate([0, 0, 0]) cube([w - 2 * c, d - 2 * c, EPS], center = true);
            }
        }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRESS RISERS
// ─────────────────────────────────────────────────────────────────────────────

// The material to ADD at an internal corner to give it a radius.
//
// A sharp internal corner concentrates stress by a factor that rises without
// bound as the radius goes to zero; a radius of even a few times the wall
// thickness brings the concentration factor down near unity. In an FDM part
// the corner is also where the extruder decelerates and where layer bonding is
// weakest, so the geometric riser and the material weak point coincide. This is
// the cheapest strength in the whole design and it costs grams.
//
// Union this into the corner between two faces meeting along +Z.
module fillet_in(r, h) {
    linear_extrude(height = h)
        difference() {
            square([r, r]);
            translate([r, r]) circle(r = r);
        }
}

// A plate with rounded corners in plan and a chamfer on both faces.
//
// Rounded corners are not cosmetic: a square corner on a printed plate is where
// a drop lands, where the layer above it has the least anchorage, and where a
// crack starts. `r` in plan, `c` on the edges.
module rounded_plate(w, d, t, r = 3, c = 0) {
    hull()
        for (sx = [-1, 1], sy = [-1, 1])
            translate([sx * (w / 2 - r), sy * (d / 2 - r), 0])
                if (c > 0)
                    // A chamfered cylinder = hull of two discs of different radius.
                    hull() {
                        cylinder(r = r, h = t - 2 * c, center = true);
                        cylinder(r = r - c, h = t, center = true);
                    }
                else
                    cylinder(r = r, h = t, center = true);
}

// ─────────────────────────────────────────────────────────────────────────────
// FASTENERS
// ─────────────────────────────────────────────────────────────────────────────

// The hole for a heat-set threaded insert, with the lead-in that makes it seat
// square.
//
// Three things are wrong with just putting a cylinder there:
//   1. No lead-in chamfer, so the insert tips as the iron pushes it and ends up
//      cocked. A cocked insert strips on the first bolt.
//   2. No relief below, so displaced plastic has nowhere to go and either bulges
//      the boss or stops the insert short of flush.
//   3. Hole diameter guessed. Too tight splits the boss; too loose spins the
//      insert the first time the bolt is torqued. The number comes from the
//      insert manufacturer and lives in the generated params file.
//
// `d_hole`   manufacturer-specified hole diameter for the insert
// `l_insert` insert length
// `relief_d` diameter of the pocket below the insert (screw clearance is enough)
module insert_hole(d_hole, l_insert, relief_d, lead_in = 0.6) {
    union() {
        // Lead-in chamfer at the mouth.
        translate([0, 0, -EPS])
            cylinder(d1 = d_hole + 2 * lead_in, d2 = d_hole, h = lead_in + EPS);
        // The insert's own bore, slightly deeper than the insert so the last of
        // the melt has somewhere to go rather than jacking the insert back out.
        cylinder(d = d_hole, h = l_insert + 0.4);
        // Screw relief below, so a slightly-long screw does not bottom out and
        // push the insert back through the boss.
        translate([0, 0, l_insert])
            cylinder(d = relief_d, h = l_insert * 2 + EPS);
    }
}

// A cylindrical boss to receive a heat-set insert, with the wall thickness the
// insert needs around it.
//
// `wall` is asserted rather than defaulted, because a boss with too little meat
// around it splits when the insert goes in and there is no visual cue that it
// is about to. Failing at render is the whole point.
module insert_boss(d_hole, l_insert, relief_d, wall, h = undef) {
    // NOTE the comma before the second string. OpenSCAD has NO implicit string
    // concatenation -- two adjacent literals are a PARSE ERROR, and a parse error
    // in a `use`d file is fatal whether or not the module is ever called. This
    // exact line stopped the entire tree from rendering, and because OpenSCAD is
    // not installed here, check.py's lint was the only thing that could have
    // caught it and it did not. Section 8 now looks for the pattern.
    assert(wall >= 1.2,
           str("insert boss wall ", wall, " mm is below the 1.2 mm minimum; ",
               "the boss will split when the insert is pressed"));
    hgt = h == undef ? l_insert + 1.5 : h;
    difference() {
        cylinder(d = d_hole + 2 * wall, h = hgt);
        insert_hole(d_hole, l_insert, relief_d);
    }
}

// Counterbored clearance hole for a socket-head cap screw, printed head-down.
//
// The counterbore roof is a bridge; keeping it under MAX_BRIDGE is automatic
// here because a cap-screw head is a few millimetres across.
module cap_screw_hole(d_clear, d_head, h_head, h_total) {
    union() {
        translate([0, 0, -EPS]) cylinder(d = d_head, h = h_head + EPS);
        translate([0, 0, -EPS]) cylinder(d = d_clear, h = h_total + 2 * EPS);
    }
}

// A hex pocket for a captive nut, entered from the side.
//
// Used where an insert cannot be: through-bolted joints in thin walls. The
// pocket is 0.2 mm oversize on the across-flats so a nut drops in without
// hammering, and the entry slot is one layer taller than the nut so the roof
// bridges rather than resting on the nut.
module nut_pocket(af, thick, entry_len = 0, clearance = 0.2) {
    a = af + clearance;
    union() {
        cylinder(d = a / cos(30), h = thick, $fn = 6);
        if (entry_len > 0)
            translate([-a / 2, 0, 0]) cube([a, entry_len, thick]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BEARINGS
// ─────────────────────────────────────────────────────────────────────────────

// A pocket for a bearing outer race.
//
// `fit` is a diametral allowance, and it is the difference between a bearing
// that presses in and stays, and one that either will not go or falls out.
// Printed holes come out undersize by roughly the extrusion width's worth of
// inward pull, which is why the allowance is positive rather than negative.
// The number is in the generated params file so it can be corrected once from a
// test print rather than in twelve places.
//
// The shelf at the bottom is `shelf` deep and stops the race at a known
// position; the through-hole below it clears the inner race so the shaft turns
// and nothing rubs the seal.
module bearing_pocket(od, w, fit, shelf = 1.2, bore_clear = 1.0) {
    union() {
        translate([0, 0, -EPS]) cylinder(d = od + fit, h = w + EPS);
        translate([0, 0, w - EPS]) cylinder(d = od - 2 * shelf, h = shelf + 2 * EPS);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURE
// ─────────────────────────────────────────────────────────────────────────────

// A triangular gusset between two perpendicular faces.
//
// The cheapest stiffness in a printed part: a rib adds section depth where a
// plate has almost none, and its area moment goes as depth cubed. Printed
// standing up it is also loaded along its layers rather than across them.
module gusset(len, hgt, t) {
    translate([0, -t / 2, 0])
        rotate([90, 0, 0])
            linear_extrude(height = t, center = false)
                polygon([[0, 0], [len, 0], [0, hgt]]);
}

// A rectangular lightening hole with generous corner radii.
//
// Removing material from the neutral axis of a beam costs almost no stiffness
// and buys real mass. The radii are not optional: a square lightening hole in a
// bending member is a crack starter, and the whole benefit is lost the first
// time the part is shock-loaded.
module lightening_slot(l, w, t, r = 2.5) {
    hull()
        for (s = [-1, 1])
            translate([s * (l / 2 - r), 0, 0])
                cylinder(r = r, h = t, center = true);
}
