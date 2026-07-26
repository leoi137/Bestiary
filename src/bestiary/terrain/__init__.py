"""Terrain: generating heightfields, and reading them back at runtime.

`generate` writes the binary heightfield and its matching texture from a
fractal noise seed. `field` is the runtime side -- looking up ground height
under a robot so an env can tell "fell over" from "walked downhill".
"""
from bestiary.terrain.field import HeightField, ground_height_at

__all__ = ["HeightField", "ground_height_at"]
