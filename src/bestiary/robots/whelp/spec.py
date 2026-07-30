"""WHELP-16 — every dimension the machine is built from, and where it came from.

    python -m bestiary.robots.whelp.spec        # print the spec and its provenance

WHAT THIS MACHINE IS
--------------------
A 3D-printed 16-DoF wheel-legged quadruped, about 2.5 kg, driven by twelve
Feetech STS3215 serial bus servos with a swappable wheel drive. It is the
PHYSICAL sibling of robots/hound/, which is the same topology at Unitree Go2
scale and exists only in simulation. Hound answers "what does a wheel-legged
policy look like"; Whelp answers "what survives contact with a floor".

    4 legs x (abduct + hip + knee + wheel) = 16 actuated joints
    +X forward, +Y left, +Z up, MILLIMETRES here, metres at the URDF boundary

THE RULE THIS FILE OBEYS
------------------------
Every attribute below has an entry in SOURCES, and check.py fails if one does
not. Five kinds -- measured, primary, secondary, choice, assumed -- documented
in provenance.py. Derived quantities are @property and are never stored, so no
edit can leave a stale copy of a number behind. ASSUMPTIONS.md is generated
from SOURCES, so it cannot drift from the code.

The point is not tidiness. It is that on a physical robot, a number that was
guessed and a number that was read off a datasheet look identical six weeks
later, and exactly one of them is safe to build a leg out of.

FOUR THINGS THE RESEARCH CHANGED, AND THEY ARE THE WHOLE DESIGN
---------------------------------------------------------------
1. "STS3215" IS NOT A PART NUMBER. Feetech ships at least four variants under
   that string: C018 (12 V, 30 kgf.cm, 1:345), C001 (7.4 V, 19.5 kgf.cm),
   C044 (1:191, 86 rpm) and C046 (1:147). They differ by 2x in torque and 2x in
   speed and they need different bus voltages. This design specifies C018 and
   check.py refuses to proceed against a spec that does not name a variant.

2. STALL TORQUE IS NOT A DESIGN NUMBER. The headline 30 kgf.cm = 2.94 N.m is
   momentary. The manufacturer's RATED load is 10 kgf.cm = 0.98 N.m, a 3x
   derate. Sustained poses are budgeted against 0.98; transients against 2.94.
   Budgeting the whole robot against stall overstates usable torque by 3x, and
   that error propagates into the URDF's effort limits and from there into every
   policy trained against them.

3. THE SOFTWARE TORQUE LIMIT DOES NOT PROTECT AGAINST IMPACT. A 1:345 gear
   train reflects rotor inertia by the square of the ratio: the identified
   armature is ~0.027 kg.m^2 at the joint, roughly a hundred times the thigh's
   own inertia. Backdriving that in the ~25 ms of a landing needs several N.m
   just to accelerate the rotor, so on impact timescales the joint is RIGID no
   matter what the Torque Limit register says. The register protects against
   slow overload -- leaning, pushing, standing wrong -- and nothing else.
   Landing energy therefore has to go somewhere mechanical, which is why the
   tire stroke and the sacrificial fuse are load-bearing parts of this design
   rather than accessories. torque.py computes the crossover.

4. THE WHEEL DRIVE IS THE DESIGN'S REAL LIMITATION. An STS3215-C018 spinning a
   45 mm wheel tops out at 0.21 m/s. Comparable wheel-legged machines run
   1.5-2.5 m/s, and Froude scaling from Go2-W puts a realistic target for a
   250 mm body at 1.2-1.5 m/s. v1 ships the servo anyway -- one part number, one
   bus, no extra electronics, and every number verified -- and the wheel mount
   is parametric so the upgrade is a bracket, not a redesign. What v1 is NOT is
   fast, and pretending otherwise would put a number in the URDF that the
   hardware cannot produce.
"""
from __future__ import annotations

import math

from bestiary.robots.whelp.provenance import Source, assumed, choice, primary, secondary

#: kgf.cm -> N.m. Every servo torque in every Feetech datasheet is in kgf.cm.
KGFCM = 0.0980665


class Spec:
    """Every dimension the machine is built from. Millimetres, radians, kg."""

    # ── Identity ─────────────────────────────────────────────────────────────
    name = "whelp16"
    servo_variant = "STS3215-C018"
    wheel_drive_kind = "sts3215_c018"
    material = "PETG"

    # ── Trunk ────────────────────────────────────────────────────────────────
    # Split fore/aft at the midplane: 250 mm does not fit a 220 mm bed and
    # printing across the diagonal is fragile advice. The two halves lap-joint
    # and share four M3, and the joint sits where the bending moment is lowest
    # because the abduction pivots -- the supports -- are outboard of it.
    trunk_len_mm = 250.0
    trunk_width_mm = 104.0
    trunk_height_mm = 58.0
    trunk_wall_mm = 2.4
    trunk_split_lap_mm = 26.0

    # ── Leg attachment ───────────────────────────────────────────────────────
    abduct_x_mm = 92.0            # abduction pivot, fore/aft from trunk centre
    abduct_y_mm = 36.0            # abduction pivot, laterally from trunk centreline
    abduct_to_hip_mm = 36.0       # abduction pivot -> hip pivot, laterally
    abduct_to_hip_drop_mm = 0.0   # ...and vertically. Zero keeps the abduction
                                  # axis in the plane of the hip pivot, so the
                                  # abduction torque is a pure lateral-offset
                                  # term with no vertical cross-coupling.

    # ── Link lengths ─────────────────────────────────────────────────────────
    thigh_len_mm = 110.0
    calf_len_mm = 100.0
    wheel_radius_mm = 45.0
    wheel_width_mm = 24.0
    calf_to_wheel_plane_mm = 17.0   # calf centreplane -> wheel centreplane

    # ── Leg cross-sections ───────────────────────────────────────────────────
    # Both legs are channel sections, not plates. Bending stiffness goes as the
    # cube of the depth, so a 26 mm channel is an order of magnitude stiffer than
    # a 10 mm plate of the same mass, and the open side is where the servo drops
    # in -- structure and assembly access are the same feature.
    thigh_section_fore_aft_mm = 30.0
    thigh_section_lateral_mm = 32.0
    thigh_wall_mm = 3.2
    calf_section_fore_aft_mm = 26.0
    calf_section_lateral_mm = 24.0
    calf_wall_mm = 3.2
    fillet_r_mm = 3.0             # EVERY internal corner. See print_kt.

    # ── Standing stance (rad) ────────────────────────────────────────────────
    # hip is the input; the knee is SOLVED so the axle lands under the hip pivot
    # (geometry.solve_stance_knee). hip is a real trade and the value is chosen,
    # not inherited: it sets the knee's lever arm, which is L_thigh*sin(hip) and
    # is the single term the whole actuator budget turns on.
    #
    #   hip 0.35 rad -> 38 mm lever, 196 mm drop -- 93% extension, too close to
    #                   the singularity and the knee runs out of range
    #   hip 0.46 rad -> 49 mm lever, 186 mm drop -- 88% extension  <- chosen
    #   hip 0.55 rad -> 57 mm lever, 176 mm drop -- 84% extension, but the
    #                   THREE-WHEEL stance then falls under the 2x margin
    #   hip 0.67 rad -> 68 mm lever, 160 mm drop -- 76% extension, 39% more knee
    #                   torque for travel a 1:345 servo cannot use anyway
    #
    # 0.46 is set by the three-wheel case, not the four-wheel one. Lifting a leg
    # puts HALF the machine's weight on each of the two diagonal wheels -- not a
    # third, which is what an earlier version assumed and which is not even a
    # valid load set. That 1.5x is what makes the stance angle a real constraint
    # rather than a preference.
    stance_abduct_rad = 0.0
    stance_hip_rad = 0.46

    # ── Joint ranges (rad) ───────────────────────────────────────────────────
    abduct_range_rad = (-0.80, 0.80)
    hip_range_rad = (-1.20, 2.60)
    knee_range_rad = (-2.60, -0.60)
    # The wheel has no range. See geometry.py.

    # ── Leg servo: Feetech STS3215-C018 ──────────────────────────────────────
    leg_servo_stall_nm = 30.0 * KGFCM          # 2.942 N.m, MOMENTARY
    leg_servo_rated_nm = 10.0 * KGFCM          # 0.981 N.m, CONTINUOUS
    leg_servo_no_load_rad_s = 2.0 * math.pi * 45.0 / 60.0   # 45 rpm -> 4.712
    leg_servo_gear_ratio = 345.0
    servo_mass_kg = 0.055
    servo_body_l_mm = 45.2
    servo_body_w_mm = 24.7
    servo_body_h_mm = 35.0
    servo_encoder_counts = 4096.0
    servo_backlash_rad = math.radians(0.87)
    servo_deadband_counts = 1.0
    servo_max_temp_c = 60.0
    servo_stall_current_a = 2.7
    servo_rated_current_a = 0.9

    # Reflected rotor inertia at the joint. THE most-omitted parameter in hobby
    # robot URDFs and, at 1:345, the one that dominates: it is ~100x the thigh's
    # own inertia, so a simulator without it models a nearly massless whip that
    # a policy will happily flick at rates the real leg cannot reach. It is also
    # what makes the joint rigid on impact timescales (see the module docstring).
    servo_armature_kgm2 = 0.027
    joint_damping = 0.56
    joint_friction_nm = 0.068

    # ── Wheel drive ──────────────────────────────────────────────────────────
    # Same part as the legs in v1: one SKU, one 12 V bus, no extra drivers, and
    # every number verified. It is also the slowest option by a factor of six.
    wheel_drive_part = "sts3215"
    wheel_drive_stall_nm = 30.0 * KGFCM
    wheel_drive_rated_nm = 10.0 * KGFCM
    wheel_drive_no_load_rad_s = 2.0 * math.pi * 45.0 / 60.0
    wheel_drive_mass_kg = 0.055
    wheel_damping = 0.10
    wheel_friction_nm = 0.030
    #: What an upgrade drive must beat to reach the 1.2-1.5 m/s the envelope
    #: supports. Stated as a requirement rather than a part number, because no
    #: specific motor was verified and inventing one would be the exact failure
    #: this file's provenance rules exist to prevent.
    wheel_upgrade_min_rpm = 300.0
    wheel_upgrade_min_nm = 0.20
    wheel_upgrade_max_mass_kg = 0.060

    # ── Tire ─────────────────────────────────────────────────────────────────
    # TPU 95A over a PETG hub. 95A rather than 85A: 85A yields at 4 MPa and a
    # 2.5 kg robot standing on it takes a permanent flat spot.
    #
    # The tire is a structural element of the impact budget, not a surface
    # finish. Its STROKE is the only mechanical compliance in the leg once the
    # 1:345 gearbox is understood to be rigid on impact timescales, so
    # tire_stroke_mm appears directly in torque.py's landing case.
    tire_thickness_mm = 12.0
    tire_infill_frac = 0.25
    tire_static_sag_mm = 1.4
    tire_stroke_mm = 8.0

    # ── Print material: PETG ─────────────────────────────────────────────────
    print_density_g_cm3 = 1.27
    print_tensile_xy_mpa = 47.0
    print_modulus_gpa = 1.5
    print_z_over_xy = 0.38
    tpu_density_g_cm3 = 1.22

    # The design allowable is NOT the tensile strength. Three knockdowns, all
    # measured, stack multiplicatively and the product is what a part may see:
    #
    #   fatigue   at 1e6 cycles a printed coupon holds ~42% of its static UTS,
    #             and a leg joint at a 2 Hz gait reaches 1e6 cycles in about
    #             140 hours of walking. Sizing to static strength puts the part
    #             exactly on the fatigue line with no margin, and the failure
    #             arrives after a week of the robot "working fine".
    #   Kt        a filleted internal corner concentrates stress ~1.48x in
    #             bending; a sharp one, ~3x. The static penalty for a notch is
    #             only 8%, which is the trap -- static tests make notches look
    #             harmless in printed polymer while they roughly halve fatigue
    #             life at the same nominal stress.
    #   Z         parts are oriented so load runs in-plane, but orientation is
    #             never perfect and this is the allowance for that.
    print_fatigue_knockdown = 0.42
    print_kt = 1.48
    print_orientation_knockdown = 0.85
    print_bearing_strength_mpa = 45.0
    print_shear_strength_mpa = 28.0
    print_effective_solid_frac = 0.50
    print_walls = 5
    print_infill_frac = 0.25

    # ── Fasteners ────────────────────────────────────────────────────────────
    insert_bore_mm = 4.2
    insert_od_mm = 4.6
    insert_len_mm = 5.7
    insert_wall_mm = 1.8
    insert_pullout_n = 1167.0
    insert_torque_fail_nm = 3.0
    insert_mass_kg = 0.0009
    m3_clear_mm = 3.4
    m3_head_d_mm = 5.5
    m3_head_h_mm = 3.0
    m3_nut_af_mm = 5.5
    m3_nut_thick_mm = 2.4

    # ── Bearings: 623ZZ, 3 x 10 x 4 ──────────────────────────────────────────
    idler_bearing_id_mm = 3.0
    idler_bearing_od_mm = 10.0
    idler_bearing_w_mm = 4.0
    idler_bearing_shelf_mm = 1.2
    idler_bearing_mass_kg = 0.0031
    bearing_press_fit_mm = -0.20    # negative = pocket UNDERSIZE, an interference

    # ── Servo horn interface — THE SOFT SPOT OF THIS DESIGN ──────────────────
    # No primary source publishes the STS3215's horn bolt circle or its case
    # mounting-hole pattern. Feetech's product pages give torque, speed and
    # current and no mechanical drawing; the open-hardware projects that use the
    # part ship STEP files rather than dimensioned prints.
    #
    # So these five numbers are ASSUMED, they are load-bearing, and they are the
    # reason scad/parts/fitcheck.scad exists: it prints candidate bolt circles,
    # insert bores and bearing fits on one 30-minute coupon, so the operator
    # replaces every guess here with a measurement before committing a leg set.
    # An assumption with a ten-minute experiment attached is a different object
    # from an assumption without one.
    horn_bolt_circle_mm = 17.0
    horn_bolt_count = 4
    horn_bolt_dia_mm = 2.0
    horn_disc_d_mm = 22.0
    horn_thick_mm = 2.5
    horn_boss_thick_mm = 5.0
    horn_face_to_body_mm = 3.5
    horn_hub_clear_d_mm = 9.0
    horn_hub_clear_h_mm = 4.0
    servo_mount_hole_dx_mm = 38.0
    servo_mount_hole_dy_mm = 18.0
    servo_mount_screw_mm = 2.5
    servo_ear_span_l_mm = 54.0
    servo_ear_thick_mm = 2.5
    servo_ear_z_from_body_mid_mm = 10.0
    servo_has_rear_boss = True
    servo_rear_boss_d_mm = 6.0
    servo_rear_boss_h_mm = 2.0
    servo_output_boss_d_mm = 12.0
    servo_cable_d_mm = 5.0
    servo_cable_inset_mm = 6.0
    servo_cable_stub_mm = 18.0

    # ── Sacrificial fuse ─────────────────────────────────────────────────────
    # A thin printed shear web between the calf and the wheel mount, sized to
    # break before the calf or the servo gearbox does. Two grams, two minutes to
    # print, two M3 to swap. Its whole job is to be the cheapest thing in the
    # yield chain, so that the failure a hard side impact causes is a part swap
    # rather than a stripped gearbox.
    # Sized from the LATERAL case, not the vertical one. Vertically the wheel
    # carries about 6 N standing and 28 N in a design landing, so a fuse strong
    # enough to be safe there is nowhere near breaking. What actually destroys
    # these legs is a sideways impact -- a fall, or driving a wheel into a table
    # leg -- and the calf's lateral bending strength is the thing to get under.
    # torque.py computes both and check.py asserts the ordering.
    fuse_enable = True
    fuse_shear_area_mm2 = 10.0
    fuse_thick_mm = 2.0

    # ── Protection policy ────────────────────────────────────────────────────
    # servo_torque_limit_frac is written to the servo's Torque Limit register at
    # boot. It caps SLOW overload only -- see the module docstring, point 3.
    servo_torque_limit_frac = 0.50
    servo_gear_break_multiple = 1.8
    required_torque_margin = 2.0
    # The drop the leg survives inside its margin ON THREE WHEELS, which is the
    # binding case because a robot that comes down crooked lands on three. This
    # number is NOT chosen for comfort: torque.py derives the envelope from the
    # tire's stroke and the knee's lever, and this is set below it. It is small,
    # and its smallness is the honest consequence of a 1:345 gearbox that cannot
    # backdrive. WHELP-16 is not a jumping robot.
    design_drop_height_mm = 8.0
    yield_knee_rad = 0.35

    # ── Environment and driving ──────────────────────────────────────────────
    ground_friction = 0.9
    rolling_resistance = 0.05
    design_slope_deg = 20.0
    design_accel_m_s2 = 0.5
    # Centre-of-mass height is NOT a Spec number. It is computed from the mass
    # model and the stance by massmodel.cg_height_measured(), because it is an
    # output of the design rather than an input to it, and because the wheelie
    # ceiling depends on it -- a guessed CoM height is a guessed acceleration
    # limit. The first draft of this file did carry it as an assumption; the
    # model puts it at 0.745 of trunk-origin height against a guess of 0.85, a
    # 14% error that would have propagated straight into the wheel budget.

    # ── Electronics and payload ──────────────────────────────────────────────
    battery_mass_kg = 0.200
    compute_mass_kg = 0.075
    wiring_mass_kg = 0.120
    fastener_mass_kg = 0.090

    # ── Manufacturing ────────────────────────────────────────────────────────
    bed_x_mm = 220.0
    bed_y_mm = 220.0
    bed_z_mm = 240.0
    nozzle_mm = 0.4
    layer_mm = 0.2
    scad_fn = 64

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def abduct_axis_to_wheel_plane_mm(self) -> float:
        """The abduction servo's moment arm against a vertical ground reaction.

        The lateral gap between the abduction axis and the wheel's centreplane,
        and the only thing that sets the abduction servo's static holding torque.

        Note what it does NOT depend on: where the abduction axis sits relative
        to the trunk. Moving the whole leg outboard widens the track and changes
        nothing about this torque, because the wheel moves with it. The arm is
        purely how far the leg steps outboard of its own pivot.

        It is not driven to zero. It could be, by hanging the wheel inboard of
        the calf, but at 2.3 kg the payoff is 0.35 N.m against a 0.98 N.m
        continuous rating, and the cost is a wheel that fouls the abduction
        bracket at full abduction. torque.py prints what it actually costs.
        """
        return self.abduct_to_hip_mm + self.calf_to_wheel_plane_mm

    @property
    def wheel_centre_y_mm(self) -> float:
        """Wheel centreplane, from the trunk centreline.

        Derived, and it was a bug when it was not: an earlier draft defined this
        as the moment arm alone, which placed all four abduction axes on the
        trunk's centreline. The torque numbers were unaffected -- the arm is the
        same either way -- but the track came out at 102 mm against a 104 mm
        trunk, and the mass model, which put the abduction servos at the trunk's
        side walls, silently disagreed with the kinematics about where the legs
        were. check.py now asserts the two agree.
        """
        return self.abduct_y_mm + self.abduct_axis_to_wheel_plane_mm

    @property
    def print_design_stress_mpa(self) -> float:
        """What a cyclically-loaded printed part may actually see.

        Static tensile strength times the three measured knockdowns. For PETG at
        47 MPa this lands near 11 MPa -- a quarter of the datasheet number, and
        the single most important correction in this file. A leg sized to 47 MPa
        is not conservative-with-a-safety-factor; it is four times overstressed
        in fatigue and will fail after it has been working.
        """
        return (self.print_tensile_xy_mpa * self.print_fatigue_knockdown
                * self.print_orientation_knockdown / self.print_kt)

    @property
    def print_ultimate_stress_mpa(self) -> float:
        """Allowable for a SINGLE overload event, where fatigue does not apply.

        Used by the yield chain, which asks what breaks in one bad landing
        rather than what wears out over a hundred thousand good ones.
        """
        return self.print_tensile_xy_mpa * self.print_orientation_knockdown / self.print_kt

    @property
    def effective_density_g_cm3(self) -> float:
        """Density to multiply CAD solid volume by, to get printed mass.

        NOT the infill fraction. On a part whose smallest dimension is under
        about 20 mm the walls are most of the volume: five walls at 0.4 mm on a
        typical bracket is ~32% of it, so 25% gyroid gives an effective solid
        fraction near 0.50 rather than 0.25. Getting this wrong by 2x gives a
        URDF whose masses are 2x wrong, and inertia scales with mass.
        """
        return self.print_density_g_cm3 * self.print_effective_solid_frac

    @property
    def impact_rigid_below_ms(self) -> float:
        """Below this contact duration the servo cannot yield at all, ms.

        Time for the reflected rotor to be accelerated through the yield
        rotation by the torque the limit register allows:

            theta = 1/2 (tau/I) t^2   ->   t = sqrt(2 I theta / tau)

        A landing from the design drop height has a contact time of roughly
        2s/v, which is tens of milliseconds. If that is shorter than this
        number -- and it is -- the joint is rigid for the event and every newton
        goes into the structure. This one line is why the tire and the fuse are
        load-bearing.
        """
        tau = self.leg_servo_stall_nm * self.servo_torque_limit_frac
        return 1000.0 * math.sqrt(2.0 * self.servo_armature_kgm2 * self.yield_knee_rad / tau)

    @property
    def trunk_parts(self) -> tuple[str, ...]:
        return ("trunk_front", "trunk_rear", "trunk_lid", "battery", "compute", "wiring")

    @property
    def payload_mass_kg(self) -> float:
        return (self.battery_mass_kg + self.compute_mass_kg
                + self.wiring_mass_kg + self.fastener_mass_kg)


SPEC = Spec()


# ── Provenance ───────────────────────────────────────────────────────────────
# Read provenance.py before editing this. Every numeric attribute of Spec needs
# an entry; check.py fails otherwise, and ASSUMPTIONS.md is generated from here.
_FEETECH_C018 = "https://www.feetechrc.com/525603.html"
_DFROBOT_12V = "https://www.dfrobot.com/product-2962.html"
_SOARM = "https://github.com/TheRobotStudio/SO-ARM100/blob/main/README.md"
_PRUSA_PETG = "https://prusament.com/ (Prusament PETG technical data sheet)"
_ULTIMAKER_TPU = "UltiMaker TPU 95A technical data sheet"
_CNCK = "CNC Kitchen, instrumented heat-set insert and press-fit test series"
_PILKEY = "Pilkey, Peterson's Stress Concentration Factors, filleted flat bar"
_FATIGUE = "Polymers 18(1) 1 — S-N curve for FDM PLA, R=0.05"
_ISO15 = "ISO 15 / bearing catalogue, 623ZZ metric miniature series"
_OPENDUCK = "Open Duck Mini v2 MJCF, STS3215 parameters identified with Rhoban BAM"
_PUPPER = "Stanford Pupper V3 documentation"

SOURCES: dict[str, Source] = {
    # Identity
    "name": choice("Callsign. A whelp is a young hound, and this is hound's hardware sibling."),
    "servo_variant": primary(
        _FEETECH_C018,
        "The EXACT Feetech model code. 'STS3215' alone spans four variants differing 2x in "
        "torque, 2x in speed and needing different bus voltages, so the variant is part of "
        "the spec and not a footnote.",
        load_bearing=True),
    "wheel_drive_kind": choice(
        "Same part as the legs in v1: one SKU, one 12 V bus, no extra drivers. Also the "
        "slowest option available, by roughly six times.", load_bearing=True),
    "material": choice(
        "PETG. PLA and Tough PLA are disqualified by heat: their HDT at 0.45 MPa is 55 degC "
        "and a bracket bolted to a servo that has been holding torque sits at 50-70 degC. "
        "ASA and PETG-CF are better still (HDT 93 and 96) and swap in by changing this and "
        "the three material numbers below.", load_bearing=True),

    # Trunk
    "trunk_len_mm": choice("From the brief. Also sets the wheelbase and so the wheelie limit.",
                           load_bearing=True),
    "trunk_width_mm": choice("Wide enough for a 3S pack between the abduction brackets."),
    "trunk_height_mm": choice("Battery plus a single-board computer, stacked."),
    "trunk_wall_mm": choice("Six perimeters at a 0.4 mm nozzle. Walls carry a printed box, "
                            "not infill: 1->3 perimeters measured +51% tensile while 40->100% "
                            "infill dropped specific strength 38%.", load_bearing=True),
    "trunk_split_lap_mm": choice(
        "250 mm does not fit a 220 mm bed, so the trunk splits at the midplane. A lap joint "
        "this long carries the load in shear across a large area instead of in tension across "
        "a butt line.", load_bearing=True),

    # Leg attachment
    "abduct_x_mm": choice("Sets the 184 mm wheelbase. Longer resists wheelie better and turns "
                          "worse.", load_bearing=True),
    "abduct_y_mm": choice(
        "WHERE the abduction pivot sits laterally, hence the track. Constrained inboard by "
        "the abduction servo fitting inside the trunk shell and outboard by the trunk's own "
        "side wall. Distinct from the abduction MOMENT ARM, which is how far the leg steps "
        "outboard of this pivot and is independent of it.", load_bearing=True),
    "abduct_to_hip_mm": choice("Clears the abduction servo body plus the bracket wall.",
                               load_bearing=True),
    "abduct_to_hip_drop_mm": choice("Zero: keeps the abduction axis coplanar with the hip pivot "
                                    "so abduction torque has no vertical cross-term."),

    # Links
    "thigh_len_mm": choice("From the brief. Its sine at the stance angle IS the knee's lever "
                           "arm, so it sets the actuator budget.", load_bearing=True),
    "calf_len_mm": choice("From the brief. Shorter than the thigh so the stance solve has a "
                          "solution across the whole hip range.", load_bearing=True),
    "wheel_radius_mm": choice("From the brief. Directly sets top speed at a given motor rpm.",
                              load_bearing=True),
    "wheel_width_mm": choice("Contact patch and lateral stability against a tipping moment."),
    "calf_to_wheel_plane_mm": choice("Half the wheel width plus the calf wall plus clearance. "
                                     "It is also the abduction servo's moment arm.",
                                     load_bearing=True),

    # Sections
    "thigh_section_fore_aft_mm": choice("Channel depth in the bending plane; stiffness goes as "
                                        "its cube.", load_bearing=True),
    "thigh_section_lateral_mm": choice("Wide enough to straddle the servo in double shear."),
    "thigh_wall_mm": choice("Six perimeters at 0.4 mm plus a margin for the fillet root.",
                            load_bearing=True),
    "calf_section_fore_aft_mm": choice("As the thigh, shallower because the moment is lower.",
                                       load_bearing=True),
    "calf_section_lateral_mm": choice("Straddles the wheel drive."),
    "calf_wall_mm": choice("As the thigh.", load_bearing=True),
    "fillet_r_mm": primary(
        _PILKEY,
        "Radius on EVERY internal corner. A 0.3 mm as-modelled corner gives Kt = 2.99 in "
        "bending; 3 mm gives 1.48. Halving peak stress for zero mass and zero print time is "
        "the cheapest strength in the design.", load_bearing=True),

    # Stance
    "stance_abduct_rad": choice("Legs vertical in the frontal plane. Widening buys roll "
                                "stability and costs abduction torque all day."),
    "stance_hip_rad": choice(
        "Chosen against the knee lever arm, which is L_thigh*sin(hip). 0.55 rad gives a 57 mm "
        "lever at 84% leg extension -- the knee of the trade between torque and remaining "
        "travel. See the table in the class body.", load_bearing=True),

    # Ranges
    "abduct_range_rad": choice("From the brief. Wide enough to matter for a roll-over recovery, "
                               "which Ascento's paper names as the failure that ends test "
                               "sessions.", load_bearing=True),
    "hip_range_rad": choice("From the brief.", load_bearing=True),
    "knee_range_rad": choice("From the brief. One-sided, so the knee only ever folds one way.",
                             load_bearing=True),

    # Servo
    "leg_servo_stall_nm": primary(
        _FEETECH_C018,
        "'Peak stall torque: 30kg.cm@12V'. MOMENTARY -- the word peak is Feetech's. Used for "
        "transient cases only.", load_bearing=True),
    "leg_servo_rated_nm": secondary(
        _DFROBOT_12V,
        "'Rated load: 10kg.cm'. THE number a sustained-pose budget uses: a 3x derate from "
        "stall. Feetech's own page does not publish a rated load for C018.",
        load_bearing=True,
        conflicts=("Feetech C001 (7.4 V) page states 6.5 kgf.cm rated at 6 V; "
                   "Seeed states 5 kgf.cm; DFRobot states 4 kgf.cm — three sources, three "
                   "numbers, for the sibling variant",)),
    "leg_servo_no_load_rad_s": primary(
        _FEETECH_C018,
        "'No load speed: 0.222sec/60 deg @12V' = 45 rpm = 4.712 rad/s. Under load it is less. "
        "About a sixth of the 30 rad/s every published legged-RL config assumes.",
        load_bearing=True),
    "leg_servo_gear_ratio": primary(_FEETECH_C018, "1:345. The reason the joint is effectively "
                                    "rigid on impact timescales.", load_bearing=True),
    "servo_mass_kg": primary(_FEETECH_C018, "55 g each. Sixteen of them is 880 g, 35% of the "
                             "mass target before any structure exists.", load_bearing=True),
    "servo_body_l_mm": primary(_FEETECH_C018, "Case length, 45.2 mm.", load_bearing=True),
    "servo_body_w_mm": primary(_FEETECH_C018, "Case width, 24.7 mm.", load_bearing=True),
    "servo_body_h_mm": primary(_FEETECH_C018, "Case height, 35 mm.", load_bearing=True),
    "servo_encoder_counts": primary(_FEETECH_C018,
                                    "4096 counts over 360 deg = 0.088 deg resolution."),
    "servo_backlash_rad": secondary(
        _OPENDUCK,
        "0.87 deg measured, against a 0.5 deg published maximum. With the deadband it is "
        "~1.75 deg of total dead motion, which is a floor on how finely any policy can act.",
        load_bearing=True),
    "servo_deadband_counts": secondary(
        "FEETECH SCServo SDK control table, registers 26/27 (CW/CCW dead zone)",
        "Register default is 1 step = 0.088 deg.",
        conflicts=("an independent bench report describes a ~10-count (0.88 deg) effective "
                   "dead zone, 10x the register default — measure yours",)),
    "servo_max_temp_c": primary(_FEETECH_C018,
                                "60 degC operating maximum. A bench test measured +15 degC in "
                                "ten minutes holding 1.47 N.m, so this is reachable in normal "
                                "use, not an abuse limit.", load_bearing=True),
    "servo_stall_current_a": primary(_FEETECH_C018, "2.7 A at 12 V. Sixteen at stall is 43 A."),
    "servo_rated_current_a": secondary(_DFROBOT_12V, "0.9 A at rated load. Sixteen is 14.4 A "
                                       "continuous, which sizes the pack and the wiring."),
    "servo_armature_kgm2": secondary(
        _OPENDUCK,
        "Reflected rotor inertia at the joint, identified with Rhoban BAM on this exact servo. "
        "~100x the thigh's own inertia. Omitting it makes the simulated leg a massless whip.",
        load_bearing=True),
    "joint_damping": secondary(_OPENDUCK, "Identified viscous damping, N.m/(rad/s)."),
    "joint_friction_nm": secondary(_OPENDUCK, "Identified Coulomb friction."),

    # Wheel drive
    "wheel_drive_part": choice("Same servo as the legs in v1."),
    "wheel_drive_stall_nm": primary(_FEETECH_C018, "As the leg servo."),
    "wheel_drive_rated_nm": secondary(_DFROBOT_12V, "As the leg servo."),
    "wheel_drive_no_load_rad_s": primary(
        _FEETECH_C018,
        "4.712 rad/s. On a 45 mm wheel that is 0.21 m/s, and it is this design's headline "
        "limitation.", load_bearing=True),
    "wheel_drive_mass_kg": primary(_FEETECH_C018, "As the leg servo, at the end of the leg."),
    "wheel_damping": assumed("Wheel viscous damping.",
                             "spin a wheel up and log its coast-down from the servo's own "
                             "speed feedback; one command, ten seconds"),
    "wheel_friction_nm": assumed("Wheel Coulomb friction.",
                                 "the same coast-down measurement gives both terms"),
    "wheel_upgrade_min_rpm": secondary(
        "Froude scaling from Unitree Go2-W (Fr = 0.954) and B2-W (Fr = 1.27) to a 0.25 m body",
        "300 rpm on a 45 mm wheel is 1.41 m/s, the envelope a 250 mm machine supports."),
    "wheel_upgrade_min_nm": choice("Above the 20 deg slope demand with margin, and well under "
                                   "the friction-cone ceiling torque.py computes."),
    "wheel_upgrade_max_mass_kg": choice("At or below the servo it replaces, so the leg's "
                                        "distal inertia does not get worse."),

    # Tire
    "tire_thickness_mm": choice("Radial TPU section. It is the only mechanical compliance in "
                                "the leg, so it is a structural number.", load_bearing=True),
    "tire_infill_frac": choice("Gyroid at 25%. Lower is softer and takes a set faster."),
    "tire_static_sag_mm": assumed(
        "Tire squash under the robot's own weight. Sets the LOADED radius, and building the "
        "URDF at the free radius spawns the robot into the floor.",
        "stand the assembled robot on a flat surface and measure hub height with calipers; "
        "compare against the free radius", load_bearing=True),
    "tire_stroke_mm": assumed(
        "Usable radial compression before the tire bottoms on the hub. Appears directly in "
        "torque.py's landing case, where it is most of the impact protection.",
        "press a printed tire in a vice against a scale and record force against deflection "
        "to bottoming; twenty minutes, and it converts the whole impact budget from estimate "
        "to measurement", load_bearing=True),

    # Materials
    "print_density_g_cm3": primary(_PRUSA_PETG, "1.27 g/cm3, ISO 1183. PETG is the heaviest "
                                   "candidate: ASA at 1.07 would save ~190 g of frame.",
                                   load_bearing=True),
    "print_tensile_xy_mpa": primary(
        _PRUSA_PETG, "47 MPa tensile yield, in-plane.",
        load_bearing=True,
        conflicts=("Anycubic PETG TDS states 52 MPa XY / 31 MPa Z; Bambu PETG Basic states "
                   "32 MPa XY / 28 MPa Z — a 60% spread between two vendors on the same ISO "
                   "527 property, which is why the fit-check coupon prints tensile bars too",)),
    "print_modulus_gpa": primary(_PRUSA_PETG, "1.5 GPa. Sets deflection, which is what decides "
                                 "whether the tire or the frame absorbs a landing."),
    "print_z_over_xy": primary(
        _PRUSA_PETG,
        "Prusa's stated anisotropy coefficient, 0.38. Used to check that no part is oriented "
        "with its load across the layers.",
        load_bearing=True,
        conflicts=("UltiMaker measures 0.49 for PETG, Polymaker 0.84, Bembenek 2022 0.74 — "
                   "the ratio is a PROCESS property, not a material constant",)),
    "tpu_density_g_cm3": primary(_ULTIMAKER_TPU, "1.22 g/cm3, ASTM D792."),
    "print_fatigue_knockdown": assumed(
        "Fraction of static UTS a part may see at 1e6 cycles. 0.42 is measured for FDM PLA; "
        "a 2023 review found only ONE published fatigue study on PETG at all, so applying the "
        "PLA number to PETG is an extrapolation.",
        "cycle one representative bracket to 1e5 cycles on a crank rig at the computed stance "
        "load; a weekend, and it is the number the whole structure is sized on",
        load_bearing=True),
    "print_kt": primary(_PILKEY, "Stress concentration at a filleted internal corner, "
                        "evaluated for a representative 20/14 mm step with a 3 mm radius.",
                        load_bearing=True),
    "print_orientation_knockdown": assumed(
        "Allowance for print orientation never being perfectly aligned with the load path.",
        "print the fit-check coupon's tensile bars in both orientations and take the ratio",
        load_bearing=True),
    "print_bearing_strength_mpa": assumed(
        "Bearing stress a bolt shank may apply to printed PETG before the hole ovals. Sets "
        "the horn boss thickness.",
        "pull a single M3 through a printed lug on a luggage scale and record the load at "
        "first visible ovalling", load_bearing=True),
    "print_shear_strength_mpa": assumed(
        "Shear strength across a printed web. Sets the sacrificial fuse's break load, which "
        "is the whole point of the fuse.",
        "shear one printed fuse link in a vice against a luggage scale — ten minutes, and it "
        "is the single most valuable measurement on this list", load_bearing=True),
    "print_effective_solid_frac": secondary(
        "worked example: 5 walls at 0.4 mm on a 110 x 25 x 18 mm part, 25% gyroid",
        "Fraction of CAD solid volume that is actually plastic. NOT the infill fraction: on a "
        "part whose smallest dimension is under ~20 mm the walls are most of the volume.",
        load_bearing=True),
    "print_walls": primary(
        "Polymers 17(13) 1797 — perimeter count vs tensile strength",
        "1 -> 3 perimeters measured +51% tensile strength. Walls are where a printed part's "
        "strength lives; infill is where its mass lives.", load_bearing=True),
    "print_infill_frac": primary(
        "Polymers 14(12) 2446 — specific strength vs infill for PET-G",
        "25% gyroid. Going 0 -> 100% infill raised absolute strength 32% but dropped strength "
        "PER KILOGRAM by 38%, which on a mass-budgeted robot is the wrong direction."),

    # Fasteners
    "insert_bore_mm": primary(_CNCK, "4.2 mm for a vertical M3 x 5.7 insert. Too tight splits "
                              "the boss; too loose spins the insert on first torque.",
                              load_bearing=True),
    "insert_od_mm": primary("Ruthex M3 x 5.7 product data", "4.6 mm outer diameter."),
    "insert_len_mm": primary("Ruthex M3 x 5.7 product data", "5.7 mm."),
    "insert_wall_mm": secondary("Ruthex minimum wall guidance (1.6 mm), rounded up",
                                "Material around the insert boss.", load_bearing=True),
    "insert_pullout_n": primary(_CNCK, "119 kgf in PETG. Comparable to a direct screw, but the "
                                "insert survives repeated assembly and the screw does not."),
    "insert_torque_fail_nm": primary(
        _CNCK,
        "3 N.m in PETG, the best of every method tested (direct screw 1, helicoil 1, captive "
        "nut 2). It also sits above every joint torque in this design, which is what makes "
        "the fastener not the weak link.", load_bearing=True),
    "insert_mass_kg": secondary("Ruthex M3 x 5.7, brass", "~0.9 g each, and there are dozens. "
                                "Concentrated at joints, which is where inertia matters most."),
    "m3_clear_mm": primary("ISO 273 medium fit", "M3 clearance hole."),
    "m3_head_d_mm": primary("ISO 4762 / DIN 912", "M3 socket head cap screw head diameter."),
    "m3_head_h_mm": primary("ISO 4762 / DIN 912", "M3 cap screw head height."),
    "m3_nut_af_mm": primary("ISO 4032", "M3 nut across flats."),
    "m3_nut_thick_mm": primary("ISO 4032", "M3 nut thickness."),

    # Bearings
    "idler_bearing_id_mm": primary(_ISO15, "623ZZ bore."),
    "idler_bearing_od_mm": primary(_ISO15, "623ZZ outer diameter."),
    "idler_bearing_w_mm": primary(_ISO15, "623ZZ width."),
    "idler_bearing_shelf_mm": choice("Shoulder that stops the outer race at a known depth."),
    "idler_bearing_mass_kg": secondary("bearing catalogue", "~3.1 g each, 16 of them."),
    "bearing_press_fit_mm": primary(
        _CNCK,
        "0.2 mm diametral interference. Measured to need ~585 N to press into PETG, which is "
        "firm by hand with an arbor press and does not fall out.", load_bearing=True),

    # Horn — the assumptions
    "horn_bolt_circle_mm": assumed(
        "Diameter of the circle the horn's screws sit on. THE interface torque leaves the "
        "servo through.",
        "measure the horn with calipers, or print scad/parts/fitcheck.scad which carries five "
        "candidate circles on one 30-minute coupon", load_bearing=True),
    "horn_bolt_count": assumed("Number of screws in that circle.",
                               "count them on the horn in your hand", load_bearing=True),
    "horn_bolt_dia_mm": assumed("Their thread diameter.",
                                "measure one, or gauge it on the fit-check coupon",
                                load_bearing=True),
    "horn_disc_d_mm": assumed("Outer diameter of the metal horn disc.",
                              "calipers on the horn"),
    "horn_thick_mm": assumed("Horn thickness; sets the axial stack-up.", "calipers on the horn"),
    "horn_boss_thick_mm": choice(
        "Plastic thickness the horn screws bear against. Torque arrives as a force on each "
        "screw shank bearing against printed plastic, so this times the bolt diameter is the "
        "bearing area. Thin bosses do not shear the screws -- steel wins -- they oval out, and "
        "the joint develops the backlash the double-shear pivot was fitted to prevent.",
        load_bearing=True),
    "horn_face_to_body_mm": assumed("Horn face to case face, along the output axis.",
                                    "calipers on the servo", load_bearing=True),
    "horn_hub_clear_d_mm": assumed("Relief for the output boss and the horn's centre screw.",
                                   "calipers on the servo"),
    "horn_hub_clear_h_mm": assumed("Depth of that relief.", "calipers on the servo"),
    "servo_mount_hole_dx_mm": assumed(
        "Case mounting hole pitch, lengthwise. No primary drawing was located: Feetech "
        "publishes torque, speed and current but no dimensioned print.",
        "calipers on the servo case, or the fit-check coupon's mounting-pattern gauge",
        load_bearing=True),
    "servo_mount_hole_dy_mm": assumed("Case mounting hole pitch, across.",
                                      "calipers on the servo case", load_bearing=True),
    "servo_mount_screw_mm": assumed("Case mounting screw thread.", "gauge one"),
    "servo_ear_span_l_mm": assumed("Mounting flange span including the ears.",
                                   "calipers on the servo case"),
    "servo_ear_thick_mm": assumed("Flange thickness.", "calipers on the servo case"),
    "servo_ear_z_from_body_mid_mm": assumed("Flange height up the case from mid-body.",
                                            "calipers on the servo case"),
    "servo_has_rear_boss": primary(
        _SOARM,
        "TRUE. Feetech describes the STS3215 as a double-shaft servo: there is an idler boss "
        "opposite the output horn. This is what makes a proper double-shear joint possible "
        "without a separate through-bolt, and it is the single most useful mechanical fact "
        "about this servo.", load_bearing=True),
    "servo_rear_boss_d_mm": assumed("Diameter of that rear boss; the idler bearing rides on it.",
                                    "calipers on the servo", load_bearing=True),
    "servo_rear_boss_h_mm": assumed("Its height above the case.", "calipers on the servo"),
    "servo_output_boss_d_mm": assumed("Diameter of the output boss under the horn.",
                                      "calipers on the servo"),
    "servo_cable_d_mm": assumed("Harness bundle diameter at the servo.",
                                "calipers on the supplied cable"),
    "servo_cable_inset_mm": assumed("Cable exit position from the case end.",
                                    "calipers on the servo"),
    "servo_cable_stub_mm": choice("Straight run reserved before the harness may bend. A serial "
                                  "bus servo has a cable at BOTH ends, and a leg that pinches "
                                  "one takes the whole bus down, not one joint."),

    # Fuse
    "fuse_enable": choice("Fitted by default: two grams and two minutes of print time to make "
                          "the cheapest item in the yield chain a part swap.", load_bearing=True),
    "fuse_shear_area_mm2": choice("Sized so the fuse shears below the servo gearbox's estimated "
                                  "strength and below the calf's. torque.py checks the order.",
                                  load_bearing=True),
    "fuse_thick_mm": choice("Web thickness; with the area it sets the break load."),

    # Protection
    "servo_torque_limit_frac": choice(
        "Written to the Torque Limit register at boot. Caps SLOW overload only -- leaning, "
        "pushing, standing wrong. It does nothing for a landing, because the reflected rotor "
        "inertia makes the joint rigid on impact timescales.", load_bearing=True),
    "servo_gear_break_multiple": assumed(
        "Gearbox failure torque as a multiple of stall. No manufacturer publishes it, and no "
        "citable teardown was found. It is the least-known number in the yield chain and the "
        "one that decides whether an overload is reversible.",
        "sacrifice one servo: clamp the horn and load it to failure with a lever and a luggage "
        "scale. One servo and one afternoon buys the number the whole protection argument "
        "rests on", load_bearing=True),
    "required_torque_margin": choice("2x, from the brief. Applied against RATED torque for "
                                     "sustained poses and against STALL for transients.",
                                     load_bearing=True),
    "design_drop_height_mm": choice(
        "The free-fall drop the leg is designed to survive inside its margin. 20 mm is small, "
        "and it is what the tire's stroke supports -- see torque.py, which derives it rather "
        "than assuming it.", load_bearing=True),
    "yield_knee_rad": choice("Knee rotation the design is willing to spend absorbing a SLOW "
                             "overload before something else has to stop it.",
                             load_bearing=True),

    # Environment
    "ground_friction": assumed("TPU on a hard floor.",
                               "drag the powered-down robot on a luggage scale and read the "
                               "breakaway force", load_bearing=True),
    "rolling_resistance": assumed("Rolling resistance coefficient of a printed TPU tire.",
                                  "roll the robot down a shallow ramp and find the angle at "
                                  "which it just keeps moving"),
    "design_slope_deg": choice("Slope the wheel drive must climb."),
    "design_accel_m_s2": choice(
        "Forward acceleration the drive must produce. 0.5 m/s^2 reaches the machine's own "
        "0.21 m/s top speed in 0.4 s, which is ample -- and it has to be modest, because "
        "the reflected rotor inertia makes the apparent mass for acceleration ~59 kg "
        "rather than 2.2. See torque.wheel_budget.", load_bearing=True),

    # Payload
    "battery_mass_kg": choice("3S 2200 mAh LiPo class. 16 servos at 0.9 A rated is 14.4 A, so "
                              "this is a ~10 minute pack at full load.", load_bearing=True),
    "compute_mass_kg": choice("Single-board computer plus the USB-to-TTL bus adapter."),
    "wiring_mass_kg": assumed("Harness, connectors, bus splitters.",
                              "weigh the loom once it is built"),
    "fastener_mass_kg": assumed("All M3 hardware, inserts and bearings together.",
                                "massmodel.py counts them; weigh a bag of each to confirm"),

    # Manufacturing
    "bed_x_mm": choice("Build volume from the brief. Every part is asserted to fit."),
    "bed_y_mm": choice("Build volume from the brief."),
    "bed_z_mm": choice("Assumed typical for a 220 mm bedslinger; not binding on any part here."),
    "nozzle_mm": choice("0.4 mm. A 0.6 mm nozzle is stronger and faster and would change every "
                        "wall thickness here, so it is a spec change and not a slicer setting."),
    "layer_mm": choice("0.2 mm. The literature contradicts itself on layer height, so this is "
                       "held constant across the robot for comparability rather than optimised."),
    "scad_fn": choice("Facets per circle in the CAD. Preview uses fewer; export uses this."),
}


def main() -> int:
    from bestiary.robots.whelp import geometry as geo
    from bestiary.robots.whelp.provenance import audit

    s = SPEC
    print(f"WHELP-16 spec  ({s.servo_variant}, {s.material})")
    print()
    print(f"  trunk        {s.trunk_len_mm:.0f} x {s.trunk_width_mm:.0f} x "
          f"{s.trunk_height_mm:.0f} mm")
    print(f"  thigh/calf   {s.thigh_len_mm:.0f} / {s.calf_len_mm:.0f} mm")
    print(f"  wheel        r = {s.wheel_radius_mm:.0f} mm, w = {s.wheel_width_mm:.0f} mm")
    print(f"  stance       hip {s.stance_hip_rad:+.3f}, knee "
          f"{geo.solve_stance_knee(s):+.3f} rad (solved)")
    print(f"  knee lever   {geo.knee_lever_mm(s):.1f} mm")
    print()
    print("  DERIVED ALLOWABLES")
    print(f"    design stress (cyclic)  {s.print_design_stress_mpa:6.1f} MPa"
          f"   from {s.print_tensile_xy_mpa:.0f} MPa UTS")
    print(f"    ultimate  (one event)   {s.print_ultimate_stress_mpa:6.1f} MPa")
    print(f"    effective density       {s.effective_density_g_cm3:6.3f} g/cm3")
    print(f"    servo rigid below       {s.impact_rigid_below_ms:6.0f} ms of contact")
    print()

    unsourced, orphans, risky = audit()
    print(f"  provenance: {len(SOURCES)} sourced, {len(unsourced)} unsourced, "
          f"{len(orphans)} orphan, {len(risky)} load-bearing assumptions")
    if unsourced:
        print(f"    UNSOURCED: {', '.join(unsourced)}")
    if orphans:
        print(f"    ORPHANS:   {', '.join(orphans)}")
    return 1 if (unsourced or orphans) else 0


if __name__ == "__main__":
    raise SystemExit(main())
