"""Bestiary -- legged robots built, trained, and documented in simulation.

Importing this package deliberately does NOT register the Gymnasium
environments; that would make a cheap `import bestiary` pull in MuJoCo. Do
`import bestiary.envs` when you need the env ids.
"""

__all__ = ["paths"]
