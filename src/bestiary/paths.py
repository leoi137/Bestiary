"""Every filesystem path in the project resolves from here.

Modules must never build paths with `Path(__file__).parent.parent` chains.
Those encode how deep a file happens to sit in the tree, so they break the
moment the file moves -- which is exactly what happened during the refactor
that created this module. Import the constant instead.

Anything that runs unattended (the research loop) must also never depend on
the current working directory. `RUNS` below is absolute for that reason:
`Path("runs")` silently wrote to the wrong place when a script was launched
from anywhere other than the repo root.
"""
from __future__ import annotations

from pathlib import Path

# src/bestiary/paths.py -> src/bestiary -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ASSETS = REPO_ROOT / "assets"      # generated models, meshes, terrain, figures
RUNS = REPO_ROOT / "runs"          # per-run dirs; gitignored, tens of GB
RESEARCH = REPO_ROOT / "research"  # learnings, decisions, episodes, ledger
DOCS = REPO_ROOT / "docs"

LEDGER = RESEARCH / "ledger.jsonl"  # append-only, one row per finished run

# --- Model XMLs -------------------------------------------------------------
# MuJoCo resolves <mesh file="meshes/..."> and <hfield file="terrain/..."> as
# paths RELATIVE TO THE XML'S OWN DIRECTORY. Every model below therefore has
# to sit in ASSETS, next to meshes/ and terrain/. Moving a model XML into its
# robot folder breaks asset resolution at load time with a bare file-not-found
# and no hint about why. The robot folders hold source (build, check, card);
# ASSETS holds generated output. Keep that split.
TERRAIN_DIR = ASSETS / "terrain"
MESH_DIR = ASSETS / "meshes"

HOUND_XML = ASSETS / "hound16.xml"
HOUND_DESERT_XML = ASSETS / "hound16_desert.xml"
SPYDER_XML = ASSETS / "spyder12.xml"
SPYDER_DESERT_XML = ASSETS / "spyder12_desert.xml"
