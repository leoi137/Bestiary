"""Terrain: generating heightfields, and reading them back at runtime.

`generate` writes the binary heightfield and its matching texture from a
fractal noise seed. `field` is the runtime side -- looking up ground height
under a robot so an env can tell "fell over" from "walked downhill". `spec`
is the provenance side -- the hash that says WHICH ground, pinned into a run's
config.json so a terrain swap cannot happen silently underneath it.
"""
from bestiary.terrain.field import HeightField, ground_height_at
from bestiary.terrain.spec import TERRAIN_HASH_VERSION, TerrainSpec

__all__ = ["HeightField", "TERRAIN_HASH_VERSION", "TerrainSpec", "ground_height_at"]
