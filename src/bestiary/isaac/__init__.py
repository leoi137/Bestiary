"""Isaac Lab bridge: Bestiary assets and terrain, served to a batched GPU sim.

Everything in this package runs under the **Isaac Lab interpreter**, not
`Bestiary/venv`. Isaac Sim is built against one specific Python (3.12 for Isaac
Sim 6.x) and pins its own torch, so it gets its own environment; this package
is imported into it by path rather than installed alongside the MuJoCo stack.

That is why nothing here may import `mujoco`, `stable_baselines3`, or
`bestiary.envs`. The only Bestiary modules safe to import from this side are
ones with no heavy dependencies — `bestiary.paths` (pathlib only) and
`bestiary.terrain.isaac_hf` (struct + numpy + scipy). Keep it that way: the
moment this package needs the MuJoCo stack, the two environments have to be
reconciled, and that is a much larger problem than it looks.

Run anything here as:

    PYTHONPATH=<repo>/src ~/IsaacLab/isaaclab.sh -p -m bestiary.isaac.<module>
"""
