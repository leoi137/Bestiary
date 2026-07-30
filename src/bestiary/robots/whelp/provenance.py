"""Where every number in WHELP-16 came from, and what would replace it.

    python -m bestiary.robots.whelp.provenance          # print the table
    python -m bestiary.robots.whelp.provenance --md     # regenerate ASSUMPTIONS.md

WHY THIS FILE EXISTS
--------------------
This is a *physical* robot. Every number in spec.py eventually becomes a
printed part, a joint limit, or a torque margin, and the ones that are wrong
do not announce themselves in simulation -- they announce themselves when a
leg snaps. The failure mode this file exists to prevent is the one that has
already happened to this project once (research/learnings/013): a number gets
typed, gets used, gets trusted, and nobody can later say whether it was read
off a datasheet or invented on a Tuesday.

Provenance is therefore not documentation. It is a *typed field on every
number*, and check.py fails if a numeric attribute of Spec has no entry here.
You cannot add a dimension to this robot without saying where it came from.

THE FIVE KINDS
--------------
    PRIMARY    a manufacturer datasheet, an official standard, a peer-reviewed
               paper, or reference source in an official repository. Has a URL
               with a part number or a DOI. Trustworthy without re-derivation.

    SECONDARY  a reputable reproduction -- a well-known open-hardware project
               that uses the exact part, a reseller listing. Believable, but a
               transcription error upstream is invisible to us.

    MEASURED   somebody put calipers or a scale on the actual object. The only
               kind that outranks PRIMARY, because it is *this* unit rather
               than the nominal one. Carries a date and who measured it.

    CHOICE     a free design decision. Not right or wrong, just picked, and
               picked for a stated reason. Trunk length is a CHOICE; nothing
               validates it except the requirements it has to meet.

    ASSUMED    a number we needed and could not source. THE DANGEROUS ONE.
               Every ASSUMED entry must state `replaced_by`: the specific
               measurement that would retire it. An ASSUMED number with no
               experiment attached is a guess wearing a lab coat.

DERIVED numbers do not appear here at all. They are `@property` on Spec and
compute themselves from other numbers, so they cannot be stale and cannot be
independently wrong. If you find yourself wanting to register a DERIVED
number, you have written down something a function should have computed.

THE RULE check.py ENFORCES
--------------------------
    1. Every non-underscore numeric/tuple attribute of Spec has an entry here.
    2. Every entry here names a real attribute of Spec (no orphans).
    3. Every ASSUMED entry has a non-empty `replaced_by`.
    4. ASSUMPTIONS.md is byte-identical to what --md would write, so the
       document cannot drift from the code that it documents.

(4) is the one that matters most in six months. A hand-maintained assumptions
list is accurate for about a week.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

SCAD_DIR = Path(__file__).resolve().parent / "scad"
ASSUMPTIONS_MD = Path(__file__).resolve().parent / "ASSUMPTIONS.md"


class Kind(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    MEASURED = "measured"
    CHOICE = "choice"
    ASSUMED = "assumed"


#: Kinds whose numbers may size a structural member or a torque margin without
#: further comment. ASSUMED may still be *used* -- the robot has to be built --
#: but check.py reports the load-bearing ones so they are never a surprise.
TRUSTED = (Kind.PRIMARY, Kind.MEASURED)


@dataclass(frozen=True)
class Source:
    """Provenance for exactly one number in Spec."""

    kind: Kind
    #: URL, DOI, ISO/DIN number, or "calipers, <who>, <YYYY-MM-DD>".
    ref: str
    #: One sentence: what this number *is*, in the physical world.
    why: str
    #: ASSUMED only: the measurement that would retire this guess. Naming it is
    #: what turns an assumption into an experiment somebody can actually run.
    replaced_by: str = ""
    #: True if this number sizes a structural member, a torque margin, a joint
    #: limit, or a clearance -- i.e. if being wrong breaks hardware rather than
    #: just looking odd. Surfaced separately in the report.
    load_bearing: bool = False
    #: Sources that disagree, kept rather than resolved. A silently-chosen
    #: value hides that the world was ambiguous.
    conflicts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind is Kind.ASSUMED and not self.replaced_by:
            raise ValueError(
                f"ASSUMED source {self.ref!r} has no replaced_by. Every assumption must "
                f"name the measurement that retires it, or it is permanent by accident."
            )
        if not self.why:
            raise ValueError(f"source {self.ref!r} has no `why`")


def primary(ref: str, why: str, *, load_bearing: bool = False, conflicts: tuple = ()) -> Source:
    return Source(Kind.PRIMARY, ref, why, load_bearing=load_bearing, conflicts=conflicts)


def secondary(ref: str, why: str, *, load_bearing: bool = False, conflicts: tuple = ()) -> Source:
    return Source(Kind.SECONDARY, ref, why, load_bearing=load_bearing, conflicts=conflicts)


def measured(ref: str, why: str, *, load_bearing: bool = False) -> Source:
    return Source(Kind.MEASURED, ref, why, load_bearing=load_bearing)


def choice(why: str, *, ref: str = "design decision", load_bearing: bool = False) -> Source:
    return Source(Kind.CHOICE, ref, why, load_bearing=load_bearing)


def assumed(why: str, replaced_by: str, *, ref: str = "no source found",
            load_bearing: bool = False) -> Source:
    return Source(Kind.ASSUMED, ref, why, replaced_by=replaced_by, load_bearing=load_bearing)


# ── Report ───────────────────────────────────────────────────────────────────
def _spec_numbers() -> dict[str, object]:
    """Every attribute of Spec that is a number or a tuple of numbers.

    Imported lazily: provenance.py must not import spec.py at module scope, or
    the two files form an import cycle the moment spec.py wants a Kind.
    """
    from bestiary.robots.whelp.spec import Spec

    out: dict[str, object] = {}
    for name in dir(Spec):
        if name.startswith("_"):
            continue
        value = getattr(Spec, name)
        if isinstance(value, bool):          # bool is an int; a flag is not a dimension
            out[name] = value
        elif isinstance(value, (int, float)):
            out[name] = value
        elif isinstance(value, tuple) and value and all(
            isinstance(v, (int, float)) for v in value
        ):
            out[name] = value
        elif isinstance(value, str):
            out[name] = value
    return out


def audit() -> tuple[list[str], list[str], list[str]]:
    """(unsourced Spec attributes, orphan SOURCES keys, load-bearing assumptions)."""
    from bestiary.robots.whelp.spec import SOURCES

    numbers = _spec_numbers()
    unsourced = sorted(set(numbers) - set(SOURCES))
    orphans = sorted(set(SOURCES) - set(numbers))
    risky = sorted(
        k for k, s in SOURCES.items()
        if s.kind is Kind.ASSUMED and s.load_bearing
    )
    return unsourced, orphans, risky


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def to_markdown() -> str:
    """ASSUMPTIONS.md, generated. Never hand-edit the output of this function."""
    from bestiary.robots.whelp.spec import SOURCES

    numbers = _spec_numbers()
    lines: list[str] = []
    w = lines.append

    w("# WHELP-16 — assumptions")
    w("")
    w("**Generated by `python -m bestiary.robots.whelp.provenance --md`. Do not hand-edit:**")
    w("`check.py` fails if this file differs from what that command writes, so an edit here")
    w("is reverted by the next regeneration and a stale edit is caught rather than trusted.")
    w("")
    w("Every number the robot is built from appears below with its provenance. The five kinds,")
    w("and what each is worth, are documented at the top of `provenance.py`. The short version:")
    w("")
    w("| kind | meaning | trust |")
    w("|---|---|---|")
    w("| `measured` | calipers or a scale on the actual object | highest — it is *this* unit |")
    w("| `primary` | datasheet, standard, paper, official source | high — but it is the nominal unit |")
    w("| `secondary` | a reputable project or reseller reproducing the number | medium |")
    w("| `choice` | a free design decision, made for a stated reason | n/a — not right or wrong |")
    w("| `assumed` | needed, unsourced | **lowest — each one names the measurement that retires it** |")
    w("")

    unsourced, orphans, risky = audit()
    if unsourced:
        w(f"> ⚠ **{len(unsourced)} Spec attributes have no provenance entry:** "
          + ", ".join(f"`{u}`" for u in unsourced))
        w("")
    if risky:
        w(f"> ⚠ **{len(risky)} load-bearing assumptions** — these size a structural member, a")
        w("> torque margin, a joint limit or a clearance, and are *not* sourced. They are the")
        w("> shortlist of things to measure first, before printing anything expensive:")
        w("> " + ", ".join(f"`{r}`" for r in risky))
        w("")

    order = [Kind.ASSUMED, Kind.CHOICE, Kind.SECONDARY, Kind.PRIMARY, Kind.MEASURED]
    titles = {
        Kind.ASSUMED: ("Assumed — unsourced, each with the measurement that would retire it",
                       "These are the robot's soft spots. `replaced_by` is not aspirational: it is a\n"
                       "list of experiments, most of which take under ten minutes with a scale, a\n"
                       "caliper, or a luggage scale and a lever arm."),
        Kind.CHOICE: ("Chosen — free design decisions",
                      "Nothing external validates these. They are judged only by whether the\n"
                      "requirements they exist to satisfy are met, which `check.py` tests."),
        Kind.SECONDARY: ("Secondary — reputable but not authoritative",
                         "Believable, but an upstream transcription error would be invisible here.\n"
                         "Worth promoting to `measured` when the part is in hand."),
        Kind.PRIMARY: ("Primary — datasheet, standard, paper, or official source",
                       "Trustworthy for the *nominal* part. The unit on your bench may differ, which\n"
                       "is why some of these are still worth re-measuring."),
        Kind.MEASURED: ("Measured — calipers or a scale on the actual object",
                        "The only kind that describes the robot being built rather than the robot on\n"
                        "paper. Every other row is a candidate to become one of these."),
    }

    for kind in order:
        keys = sorted(k for k, s in SOURCES.items() if s.kind is kind)
        if not keys:
            continue
        title, blurb = titles[kind]
        w(f"## {title}")
        w("")
        w(blurb)
        w("")
        if kind is Kind.ASSUMED:
            w("| attribute | value | what it is | how to replace it |")
            w("|---|---|---|---|")
            for k in keys:
                s = SOURCES[k]
                flag = " ⚠" if s.load_bearing else ""
                val = _fmt(numbers.get(k, "?"))
                w(f"| `{k}`{flag} | `{val}` | {s.why} | {s.replaced_by} |")
        else:
            w("| attribute | value | what it is | source |")
            w("|---|---|---|---|")
            for k in keys:
                s = SOURCES[k]
                flag = " ⚠" if s.load_bearing else ""
                val = _fmt(numbers.get(k, "?"))
                ref = s.ref
                if s.conflicts:
                    ref += "<br>*disagrees with:* " + "; ".join(s.conflicts)
                w(f"| `{k}`{flag} | `{val}` | {s.why} | {ref} |")
        w("")

    w("---")
    w("")
    w("⚠ marks a **load-bearing** number: one that sizes a structural member, a torque margin, a")
    w("joint limit or a clearance. Being wrong about an unmarked number looks odd; being wrong")
    w("about a marked one breaks hardware.")
    w("")
    n_lb = sum(1 for s in SOURCES.values() if s.load_bearing)
    w(f"{len(SOURCES)} numbers, {n_lb} load-bearing, {len(risky)} both load-bearing and assumed.")
    w("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if "--md" in argv:
        ASSUMPTIONS_MD.write_text(to_markdown(), encoding="utf-8")
        print(f"wrote {ASSUMPTIONS_MD}")
        return 0

    from bestiary.robots.whelp.spec import SOURCES

    numbers = _spec_numbers()
    unsourced, orphans, risky = audit()
    by_kind: dict[Kind, int] = {}
    for s in SOURCES.values():
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1

    print("WHELP-16 provenance")
    print(f"  {len(numbers)} numbers in Spec, {len(SOURCES)} with a source")
    for kind in Kind:
        print(f"    {kind.value:<10} {by_kind.get(kind, 0):>3}")
    if unsourced:
        print(f"  UNSOURCED ({len(unsourced)}): {', '.join(unsourced)}")
    if orphans:
        print(f"  ORPHAN SOURCES ({len(orphans)}): {', '.join(orphans)}")
    if risky:
        print(f"  LOAD-BEARING ASSUMPTIONS ({len(risky)}):")
        for k in risky:
            print(f"    {k:<28} {_fmt(numbers.get(k, '?')):>12}   replace by: {SOURCES[k].replaced_by}")
    return 1 if (unsourced or orphans) else 0


if __name__ == "__main__":
    # DELEGATE to the canonically-imported module rather than calling main() from
    # this one. `python -m bestiary.robots.whelp.provenance` loads this file as
    # `__main__`, and spec.py then imports `bestiary.robots.whelp.provenance` --
    # a SECOND, distinct module object with its own `Kind` enum. Every
    # `source.kind is Kind.ASSUMED` in this file then compares members of two
    # different enum classes and is silently False, so `to_markdown()` produced a
    # document with the header and no rows, and `--md` wrote it out looking fine.
    #
    # It was caught only because check.py asserts ASSUMPTIONS.md equals what
    # to_markdown() returns, and that check runs under a normal import where the
    # two copies collapse to one. That is the whole argument for the freshness
    # assertions in section 9: a generated file that is quietly empty is worse
    # than one that is missing, because it still looks authoritative.
    #
    # Comparing `.value` instead would also work and is the usual advice, but it
    # only fixes the comparisons somebody remembers to write that way. Making the
    # entry point use the canonical module fixes every present and future
    # identity comparison in this file at once.
    from bestiary.robots.whelp.provenance import main as _main

    raise SystemExit(_main(sys.argv[1:]))
