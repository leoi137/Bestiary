"""Every number in `research/episodes/015` and in the P6-P11 calibration rows.

    venv/bin/python research/scripts/018_ladder_overnight_reads.py

The three-rung reward ladder (`spyder_ladder_env_cfg.py`, 1500 iterations each)
and the 15,000-iteration run that spent its answer (`spyder_overnight_env_cfg.py`)
produced four TensorBoard event files and two console logs. This script is the
only thing allowed to turn those into numbers: the number rule says a figure
enters the record because code computed it, not because someone read it off a
terminal, so the episode quotes this script's output and the calibration rows
cite this script by name.

What it does, in order:

1. **Cross-checks tb against the console log.** The two are written by different
   code paths in rsl_rl -- the event file by the summary writer, the terminal
   block by the printer -- so they are an independent transcription of the same
   quantity. If they disagree, one of them is being misread and neither should
   be published. Console prints 2 decimals, so the tolerance is half a unit in
   the last place.

2. **Resolves P6, P9, P10, P11**, each against the threshold exactly as it was
   pre-registered, at the ITERATION IT WAS PRE-REGISTERED FOR. The overnight run
   peaked at a higher reward than it finished at; scoring P9 on the peak would be
   the peak-versus-final error this record is built to make impossible, so the
   peak is printed next to the final and the verdict uses the final.

3. **Shows why P7 cannot be resolved at all.** Its claim compares
   `Episode_Reward/action_rate_l2` between the actionrate rung and the bare rung.
   A reward term that is not in a task's reward table is never computed by the
   reward manager and therefore never logged, so that tag exists in exactly the
   arm that pays the tax and in none of the others. The script prints both arms'
   full tag inventories rather than asserting this.

4. **Shows what P8 would need.** Its claim is a base_ang_vel_xy rms over a fixed
   deterministic drive, and no such quantity is in any training log (the training
   metrics are a 2-D linear error and a yaw error, neither of which is roll/pitch
   rate) or in any committed measurement file. The script prints the inventory of
   what the ladder run directory actually holds.

5. **Prices what 10x the training bought**, by comparing the winning rung's final
   iteration to the overnight run's final iteration on the metrics both logged.

6. **Establishes that strafe was commanded and that its quality was not
   measured** -- the first from the env config each run dumped at launch, the
   second from the absence of any per-axis velocity metric in the tag inventory.

Reads only committed-alongside artifacts under `runs/`: event files, the
`params/env.yaml` each run dumps at launch, and the console logs. No GPU, no
Isaac, no network. `runs/` is gitignored, so this script is the durable form of
those numbers and it fails loudly rather than quietly if an artifact is missing.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from bestiary import paths

# --------------------------------------------------------------------------
# The artifacts. Named once, here, so a moved run directory is one edit and a
# missing one is an immediate crash with the path in the message.
# --------------------------------------------------------------------------

LADDER_ROOT = paths.RUNS / "spyder_ladder_s1"
OVERNIGHT_ROOT = paths.RUNS / "spyder_overnight_s1"

#: rung id -> its rsl_rl log directory under the ladder root. The timestamped
#: leaf is resolved at runtime rather than hardcoded: there is exactly one per
#: rung, and asserting that is better than trusting a copied timestamp.
LADDER_RUNGS = ("bare", "actionrate", "tilt")

LADDER_CONSOLE = LADDER_ROOT / "box_console.log"
OVERNIGHT_CONSOLE = OVERNIGHT_ROOT / "box_console.log"

#: Marker the ladder driver wrote around each rung, used to split one console
#: log into three. `=== RUNG START Bestiary-Ladder-Bare-Spyder-v0 02:41:08 ===`
RUNG_START = re.compile(r"^=== RUNG START (\S+) .*===$", re.M)

#: The two lines rsl_rl prints per iteration, at two decimals.
CONSOLE_REWARD = re.compile(r"^\s*Mean reward:\s*(-?[\d.]+)\s*$", re.M)
CONSOLE_LENGTH = re.compile(r"^\s*Mean episode length:\s*(-?[\d.]+)\s*$", re.M)

#: Console rounds to 2 dp, so agreement means "inside half a unit in the last
#: place". Anything wider means tb and the terminal are not the same run.
CONSOLE_TOL = 0.005

# --- The pre-registered thresholds, copied from research/calibration.jsonl ---
# Copied, not re-derived: a threshold that drifts between the prediction and its
# resolution is the one failure mode a calibration record cannot survive.
P6_MIN_REWARD = 12.0          # all three rungs, at iteration 1500
P7_REQUIRED_DROP = 0.30       # actionrate's |action delta| >= 30% below bare's
P9_MIN_REWARD = 20.0          # overnight final mean reward
P10_MIN_LENGTH = 900.0        # overnight final mean episode length, of 1000
P11_MAX_ERROR_VEL_XY = 0.35   # overnight final Metrics/base_velocity/error_vel_xy

#: Iteration indices called out in the episode's trajectory table. rsl_rl indexes
#: from 0, so a "1500 iteration" run's last index is 1499.
OVERNIGHT_MILESTONES = (1707, 4023, 7635, 11517, 14193)

#: How many trailing iterations the "last-N mean" uses. Ten is what the overnight
#: task's own docstring quotes, and it exists so a single noisy final iteration
#: cannot decide an ordering by itself.
LAST_N = 10

# --------------------------------------------------------------------------


def _log_dir(root: Path, *parts: str) -> Path:
    """The single timestamped rsl_rl directory under `root/parts...`.

    Raises when there is not exactly one. Two would mean a rung was launched
    twice and the numbers below would silently be from whichever sorted first.
    """
    parent = root.joinpath(*parts)
    if not parent.is_dir():
        raise FileNotFoundError(
            f"{parent} does not exist. The runs/ tree is gitignored; this script "
            "reads artifacts that must be present on the machine that pulled them."
        )
    leaves = sorted(p for p in parent.iterdir() if p.is_dir())
    if len(leaves) != 1:
        raise AssertionError(
            f"{parent} holds {len(leaves)} timestamped directories {[p.name for p in leaves]}, "
            "expected exactly 1. Two launches under one rung name cannot be told apart "
            "by this script and must not be averaged."
        )
    return leaves[0]


def _scalars(event_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """`tag -> [(step, value), ...]` for every scalar in an event directory."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    if not any(event_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"no tfevents file under {event_dir}")
    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return {
        tag: [(e.step, e.value) for e in acc.Scalars(tag)]
        for tag in acc.Tags()["scalars"]
    }


def _final(series: list[tuple[int, float]]) -> tuple[int, float]:
    return series[-1]


def _at(series: list[tuple[int, float]], step: int) -> float:
    lookup = dict(series)
    if step not in lookup:
        raise KeyError(f"iteration {step} is not in this series (it runs "
                       f"{series[0][0]}..{series[-1][0]})")
    return lookup[step]


def _last_n_mean(series: list[tuple[int, float]], n: int = LAST_N) -> float:
    return statistics.fmean(v for _, v in series[-n:])


def _console_finals(text: str) -> tuple[float, float]:
    """The last `Mean reward` / `Mean episode length` printed in a log segment."""
    rewards = CONSOLE_REWARD.findall(text)
    lengths = CONSOLE_LENGTH.findall(text)
    if not rewards or not lengths:
        raise AssertionError(
            f"found {len(rewards)} reward lines and {len(lengths)} length lines in this "
            "console segment; the printer's format has changed and this parser is stale."
        )
    return float(rewards[-1]), float(lengths[-1])


def _ladder_console_segments() -> dict[str, str]:
    """Split the ladder's single console log into one segment per rung."""
    text = LADDER_CONSOLE.read_text(errors="replace")
    marks = list(RUNG_START.finditer(text))
    if len(marks) != len(LADDER_RUNGS):
        raise AssertionError(
            f"{LADDER_CONSOLE} has {len(marks)} RUNG START markers, expected "
            f"{len(LADDER_RUNGS)}. The log is not the three-rung ladder this script reads."
        )
    segments = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        task = m.group(1)
        rung = next((r for r in LADDER_RUNGS if r.lower() in task.lower().replace("-", "")), None)
        if rung is None:
            raise AssertionError(f"console marker {task!r} matches no rung in {LADDER_RUNGS}")
        segments[rung] = text[m.start():end]
    if set(segments) != set(LADDER_RUNGS):
        raise AssertionError(f"split the console into {sorted(segments)}, expected {sorted(LADDER_RUNGS)}")
    return segments


def _reward_weights(env_yaml: Path) -> dict[str, float]:
    """`term -> weight` for the LIVE reward terms in a launched run's config dump.

    A tiny indentation parser rather than a yaml dependency: the dump contains
    `!!python/object/apply` tags that only `yaml.unsafe_load` accepts, and this
    script must not need a loader that can execute what it reads.
    """
    lines = env_yaml.read_text().splitlines()
    try:
        start = lines.index("rewards:")
    except ValueError as exc:
        raise AssertionError(f"{env_yaml} has no top-level `rewards:` block") from exc

    weights: dict[str, float] = {}
    term: str | None = None
    for line in lines[start + 1:]:
        if line and not line.startswith(" "):
            break
        head = re.match(r"^  (\w+):(.*)$", line)
        if head:
            term = head.group(1) if head.group(2).strip() != "null" else None
            continue
        w = re.match(r"^    weight: (-?[\d.eE+-]+)$", line)
        if w and term is not None:
            weights[term] = float(w.group(1))
    if not weights:
        raise AssertionError(f"{env_yaml} yielded no live reward weights; parser is stale")
    return weights


def _scalar_field(env_yaml: Path, pattern: str) -> str:
    m = re.search(pattern, env_yaml.read_text(), re.M)
    if m is None:
        raise AssertionError(f"{env_yaml} has no line matching {pattern!r}")
    return m.group(1).strip()


def _lin_vel_y_range(env_yaml: Path) -> tuple[float, float]:
    """The commanded lateral velocity range the run actually launched with."""
    text = env_yaml.read_text()
    m = re.search(r"^      lin_vel_y: !!python/tuple\n      - (-?[\d.]+)\n      - (-?[\d.]+)$",
                  text, re.M)
    if m is None:
        raise AssertionError(f"{env_yaml} has no parseable lin_vel_y command range")
    return float(m.group(1)), float(m.group(2))


def _verdict(name: str, measured: float, bar: float, op: str) -> bool:
    got = measured >= bar if op == ">=" else measured <= bar
    print(f"  {name:<12s} measured {measured:10.4f} {op} {bar:<8.4f} -> "
          f"{'TRUE' if got else 'FALSE'}")
    return got


# --------------------------------------------------------------------------


def main() -> None:
    ladder_dirs = {r: _log_dir(LADDER_ROOT, f"spyder_ladder_{r}") for r in LADDER_RUNGS}
    overnight_dir = _log_dir(OVERNIGHT_ROOT, "box_logs")

    ladder = {r: _scalars(d) for r, d in ladder_dirs.items()}
    overnight = _scalars(overnight_dir)
    segments = _ladder_console_segments()

    print("=" * 78)
    print("0. THE RUNS, from each run's own launch-time config dump")
    print("=" * 78)
    rows = [(f"ladder/{r}", ladder_dirs[r]) for r in LADDER_RUNGS] + [("overnight", overnight_dir)]
    for label, d in rows:
        env_yaml = d / "params" / "env.yaml"
        agent_yaml = d / "params" / "agent.yaml"
        seed = _scalar_field(env_yaml, r"^seed: (.*)$")
        envs = _scalar_field(env_yaml, r"^  num_envs: (.*)$")
        ep_s = _scalar_field(env_yaml, r"^episode_length_s: (.*)$")
        iters = _scalar_field(agent_yaml, r"^max_iterations: (.*)$")
        live = sorted(_reward_weights(env_yaml))
        print(f"  {label:<18s} seed {seed}  envs {envs}  episode {ep_s} s  "
              f"max_iterations {iters}")
        print(f"  {'':<18s} live reward terms ({len(live)}): {', '.join(live)}")
    print("  ^ one seed per arm, everywhere. Nothing below is a finding.")
    print()

    print("=" * 78)
    print("1. TENSORBOARD vs CONSOLE  (independent transcriptions of one run)")
    print("=" * 78)
    for r in LADDER_RUNGS:
        step, tb_r = _final(ladder[r]["Train/mean_reward"])
        _, tb_l = _final(ladder[r]["Train/mean_episode_length"])
        c_r, c_l = _console_finals(segments[r])
        for what, tb_v, c_v in (("reward", tb_r, c_r), ("ep length", tb_l, c_l)):
            gap = abs(tb_v - c_v)
            ok = gap <= CONSOLE_TOL if what == "reward" else gap <= CONSOLE_TOL
            print(f"  ladder/{r:<11s} iter {step:<6d} {what:<9s} tb {tb_v:9.4f}  "
                  f"console {c_v:8.2f}  |gap| {gap:.4f}  {'agree' if ok else 'MISMATCH'}")
            if not ok:
                raise AssertionError(
                    f"ladder/{r} {what}: tb says {tb_v}, console says {c_v}, "
                    f"{gap:.4f} apart against a {CONSOLE_TOL} tolerance. One of the two "
                    "is not this run and neither may be published."
                )
    step, tb_r = _final(overnight["Train/mean_reward"])
    _, tb_l = _final(overnight["Train/mean_episode_length"])
    c_r, c_l = _console_finals(OVERNIGHT_CONSOLE.read_text(errors="replace"))
    for what, tb_v, c_v in (("reward", tb_r, c_r), ("ep length", tb_l, c_l)):
        gap = abs(tb_v - c_v)
        print(f"  overnight          iter {step:<6d} {what:<9s} tb {tb_v:9.4f}  "
              f"console {c_v:8.2f}  |gap| {gap:.4f}  "
              f"{'agree' if gap <= CONSOLE_TOL else 'MISMATCH'}")
        if gap > CONSOLE_TOL:
            raise AssertionError(f"overnight {what}: tb {tb_v} vs console {c_v}")
    print()

    print("=" * 78)
    print("2. THE LADDER, at its final iteration and over its last 10")
    print("=" * 78)
    print(f"  {'rung':<12s} {'final rew':>10s} {'last10 rew':>11s} {'final len':>10s} "
          f"{'last10 len':>11s} {'err_vel_xy':>11s} {'terrain':>8s}")
    ladder_final = {}
    for r in LADDER_RUNGS:
        fr = _final(ladder[r]["Train/mean_reward"])[1]
        fl = _final(ladder[r]["Train/mean_episode_length"])[1]
        ladder_final[r] = (fr, fl)
        print(f"  {r:<12s} {fr:10.4f} {_last_n_mean(ladder[r]['Train/mean_reward']):11.4f} "
              f"{fl:10.4f} {_last_n_mean(ladder[r]['Train/mean_episode_length']):11.4f} "
              f"{_final(ladder[r]['Metrics/base_velocity/error_vel_xy'])[1]:11.4f} "
              f"{_final(ladder[r]['Curriculum/terrain_levels'])[1]:8.4f}")

    print()
    print("  Income, decomposed. rsl_rl logs each term as an episode sum normalised")
    print("  by the maximum episode length, so the per-episode figure is term x 20 s.")
    print(f"  {'rung':<12s} {'income/s':>9s} {'income/ep':>10s} {'tax/s':>9s} "
          f"{'tax/ep':>8s} {'net/ep':>8s} {'reported':>9s} {'ratio':>7s}")
    income_tags = ("Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/track_ang_vel_z_exp")
    ep_len_s = float(_scalar_field(ladder_dirs["bare"] / "params" / "env.yaml",
                                   r"^episode_length_s: (.*)$"))
    for r in LADDER_RUNGS:
        terms = {t: _final(s)[1] for t, s in ladder[r].items() if t.startswith("Episode_Reward/")}
        income = sum(v for t, v in terms.items() if t in income_tags)
        tax = sum(v for t, v in terms.items() if t not in income_tags)
        net = income + tax
        reported = ladder_final[r][0]
        print(f"  {r:<12s} {income:9.4f} {income * ep_len_s:10.4f} {tax:9.4f} "
              f"{tax * ep_len_s:8.4f} {net * ep_len_s:8.4f} {reported:9.4f} "
              f"{reported / net:7.3f}")
    weights = _reward_weights(ladder_dirs["bare"] / "params" / "env.yaml")
    ceiling = sum(weights[t] for t in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")) * ep_len_s
    print(f"  income ceiling = ({weights['track_lin_vel_xy_exp']} + "
          f"{weights['track_ang_vel_z_exp']}) x {ep_len_s} s = {ceiling:.1f} per episode")
    print("  ^ the `ratio` column is reported mean reward / summed terms per second.")
    print("    It lands near the episode length in seconds for every arm, which is")
    print("    what pins the normalisation; it is not exactly 20 because the two are")
    print("    averaged over different episode populations, so treat the decomposed")
    print("    figures as accurate to a couple of percent and never as an identity.")
    print()
    bare_income, ar_income = (
        sum(_final(ladder[r][t])[1] for t in income_tags) for r in ("bare", "actionrate")
    )
    print(f"  actionrate income/s {ar_income:.4f} vs bare {bare_income:.4f} = "
          f"{(ar_income / bare_income - 1) * 100:+.1f}%")
    print("  ^ the winner earns MORE income while paying a tax the control does not,")
    print("    so its margin on net reward understates its margin on tracking.")
    print()

    print("=" * 78)
    print("3. P6 -- all three rungs reach mean reward >= 12 by iteration 1500")
    print("=" * 78)
    p6 = all(_verdict(r, ladder_final[r][0], P6_MIN_REWARD, ">=") for r in LADDER_RUNGS)
    worst = min(LADDER_RUNGS, key=lambda r: ladder_final[r][0])
    print(f"  P6 -> {'TRUE' if p6 else 'FALSE'}   (weakest rung {worst} at "
          f"{ladder_final[worst][0]:.4f}, {ladder_final[worst][0] - P6_MIN_REWARD:+.4f} "
          f"against the bar)")
    print()

    print("=" * 78)
    print("4. P7 -- actionrate's |action delta| >= 30% below bare's, from tb")
    print("=" * 78)
    tag = "Episode_Reward/action_rate_l2"
    for r in LADDER_RUNGS:
        have = tag in ladder[r]
        value = f"{_final(ladder[r][tag])[1]:.6f}" if have else "TAG ABSENT"
        print(f"  {r:<12s} {tag:<32s} {value}")
    print("  Full Episode_Reward inventory per rung:")
    for r in LADDER_RUNGS:
        present = sorted(t for t in ladder[r] if t.startswith("Episode_Reward/"))
        print(f"    {r:<12s} {', '.join(t.split('/', 1)[1] for t in present)}")
    print(f"  bare's reward table (launch dump): {sorted(_reward_weights(ladder_dirs['bare'] / 'params' / 'env.yaml'))}")
    print()
    print(f"  P7 -> UNRESOLVABLE. The claim needs both arms' {tag.split('/')[1]}")
    print("  and the control arm has none. A term absent from a reward table is never")
    print("  evaluated by the reward manager, so it is logged by exactly the arm that")
    print(f"  pays it. The bar was a {P7_REQUIRED_DROP:.0%} drop against a quantity that")
    print("  was never computed for the arm it had to be compared against.")
    print("  What DOES exist, and is not the claim:")
    ar = _final(ladder["actionrate"][tag])[1]
    w = _reward_weights(ladder_dirs["actionrate"] / "params" / "env.yaml")["action_rate_l2"]
    print(f"    actionrate {tag} = {ar:.6f} per second at iteration 1499")
    print(f"    at weight {w}, the raw penalised quantity is {ar / w:.4f} "
          f"sum-of-squares per second")
    print(f"    over {ep_len_s:.0f} s that is {ar * ep_len_s:.4f} reward per episode, "
          f"{abs(ar) / ar_income * 100:.2f}% of the arm's own income")
    print("  Resolving P7 now needs a NEW measurement -- both checkpoints rolled out")
    print("  under one script that computes |action delta| itself -- which is a")
    print("  different instrument from the one the claim named and would be a new")
    print("  prediction, not a resolution of this one.")
    print()

    print("=" * 78)
    print("5. P8 -- bare is the wildest in playback (largest base_ang_vel_xy rms)")
    print("=" * 78)
    everywhere = sorted(set.intersection(*(set(ladder[r]) for r in LADDER_RUNGS)))
    rate_like = [t for t in everywhere if "ang_vel" in t and "track" not in t]
    print(f"  scalar tags logged by all three rungs ({len(everywhere)}):")
    for t in everywhere:
        print(f"    {t}")
    print(f"  of those, roll/pitch-rate metrics: {rate_like if rate_like else 'NONE'}")
    print("  ^ the training metrics are a 2-D linear-velocity error and a yaw error.")
    print("    Neither is torso roll/pitch rate, and the tilt rung's own penalty term")
    print("    is a reward line for one arm, not a metric for three.")
    print("  Files in the ladder run root that are not per-rung log directories:")
    for p in sorted(LADDER_ROOT.iterdir()):
        if p.is_file():
            print(f"    {p.name:<28s} {p.stat().st_size / 1e6:8.2f} MB")
    measurements = paths.RESEARCH / "measurements"
    hits = sorted(p.name for p in measurements.glob("*")
                  if "ladder" in p.name or "spyder_ladder" in p.name) if measurements.is_dir() else []
    print(f"  committed measurement files matching the ladder: {hits if hits else 'NONE'}")
    print()
    print("  P8 -> UNRESOLVED, and NOT scoreable from what exists. The claim names a")
    print("  number (base_ang_vel_xy rms over a fixed 8 s drive) produced by a")
    print("  committed measurement snippet; that snippet was never written and the")
    print("  quantity appears in no log or measurement file. What exists is video,")
    print("  which supports an impression and not an rms. The three checkpoints are")
    print("  on disk, so the measurement remains MAKEABLE on a GPU -- this row is")
    print("  pending an instrument, not void for lack of a subject.")
    print()

    print("=" * 78)
    print("6. THE OVERNIGHT RUN")
    print("=" * 78)
    o_rew = overnight["Train/mean_reward"]
    o_len = overnight["Train/mean_episode_length"]
    fin_step, fin_rew = _final(o_rew)
    fin_len = _final(o_len)[1]
    print(f"  {'iteration':>10s} {'mean reward':>12s} {'mean ep length':>15s}")
    for s in OVERNIGHT_MILESTONES:
        print(f"  {s:>10d} {_at(o_rew, s):12.4f} {_at(o_len, s):15.4f}")
    print(f"  {fin_step:>10d} {fin_rew:12.4f} {fin_len:15.4f}   <- final")
    peak_rew = max(v for _, v in o_rew)
    peak_at = next(s for s, v in o_rew if v == peak_rew)
    peak_len = max(v for _, v in o_len)
    peak_len_at = next(s for s, v in o_len if v == peak_len)
    print(f"  peak reward     {peak_rew:.4f} at iteration {peak_at} "
          f"({peak_rew - fin_rew:+.4f} against the final)")
    print(f"  peak ep length  {peak_len:.4f} at iteration {peak_len_at}")
    print(f"  last-{LAST_N} mean  reward {_last_n_mean(o_rew):.4f}  "
          f"length {_last_n_mean(o_len):.4f}")
    print("  Final term decomposition, per second and per episode:")
    o_terms = {t.split("/", 1)[1]: _final(s)[1]
               for t, s in overnight.items() if t.startswith("Episode_Reward/")}
    for name, v in sorted(o_terms.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<24s} {v:9.6f}/s   {v * ep_len_s:9.4f}/episode")
    o_income = sum(v for k, v in o_terms.items() if k.startswith("track_"))
    o_tax = sum(v for k, v in o_terms.items() if not k.startswith("track_"))
    print(f"    {'income':<24s} {o_income:9.6f}/s   {o_income * ep_len_s:9.4f}/episode  "
          f"({o_income * ep_len_s / ceiling * 100:.1f}% of the {ceiling:.0f} ceiling)")
    print(f"    {'tax':<24s} {o_tax:9.6f}/s   {o_tax * ep_len_s:9.4f}/episode")
    print("  Other final readings:")
    for t in ("Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
              "Metrics/success_rate", "Curriculum/terrain_levels",
              "Episode_Termination/time_out", "Episode_Termination/base_contact",
              "Policy/mean_std"):
        print(f"    {t:<42s} {_final(overnight[t])[1]:9.4f}")
    tl = overnight["Curriculum/terrain_levels"]
    tl_max = max(v for _, v in tl)
    tl_at = next(s for s, v in tl if v == tl_max)
    print(f"    terrain level: start {tl[0][1]:.4f}, max {tl_max:.4f} at iteration {tl_at}, "
          f"final {_final(tl)[1]:.4f}")
    print()

    print("=" * 78)
    print("7. P9 / P10 / P11 -- the overnight run's three pre-registered claims")
    print("=" * 78)
    _verdict("P9", fin_rew, P9_MIN_REWARD, ">=")
    print(f"        NOTE: the peak {peak_rew:.4f} at iteration {peak_at} DOES clear "
          f"{P9_MIN_REWARD:.0f}.")
    print("        The claim says 'final mean reward at iteration 15000'. Scoring it")
    print("        on the peak would be the peak-versus-final substitution this")
    print("        record exists to prevent. Resolved on the final: FALSE.")
    _verdict("P10", fin_len, P10_MIN_LENGTH, ">=")
    _verdict("P11", _final(overnight["Metrics/base_velocity/error_vel_xy"])[1],
             P11_MAX_ERROR_VEL_XY, "<=")
    print()

    print("=" * 78)
    print("8. WHAT 10x THE TRAINING BOUGHT  (actionrate@1499 -> overnight@14999)")
    print("=" * 78)
    print("  DIFFERENT REWARD TABLES: the overnight run pays two terms the rung did")
    print("  not, so net reward is not a like-for-like comparison and income is.")
    pairs = (
        ("mean reward (net)", ladder_final["actionrate"][0], fin_rew, "higher"),
        ("mean episode length", ladder_final["actionrate"][1], fin_len, "higher"),
        ("income per second", ar_income, o_income, "higher"),
        ("error_vel_xy",
         _final(ladder["actionrate"]["Metrics/base_velocity/error_vel_xy"])[1],
         _final(overnight["Metrics/base_velocity/error_vel_xy"])[1], "lower"),
        ("error_vel_yaw",
         _final(ladder["actionrate"]["Metrics/base_velocity/error_vel_yaw"])[1],
         _final(overnight["Metrics/base_velocity/error_vel_yaw"])[1], "lower"),
        ("base_contact share",
         _final(ladder["actionrate"]["Episode_Termination/base_contact"])[1],
         _final(overnight["Episode_Termination/base_contact"])[1], "lower"),
        ("terrain level",
         _final(ladder["actionrate"]["Curriculum/terrain_levels"])[1],
         _final(overnight["Curriculum/terrain_levels"])[1], "higher"),
    )
    print(f"  {'metric':<22s} {'rung@1499':>11s} {'overnight':>11s} {'change':>9s}  better is")
    for name, a, b, better in pairs:
        print(f"  {name:<22s} {a:11.4f} {b:11.4f} {(b / a - 1) * 100:+8.1f}%  {better}")
    thr = ladder_final["actionrate"][0]
    first = next((s for s, v in o_rew if v >= thr), None)
    above = sum(1 for _, v in o_rew if v >= thr) / len(o_rew)
    print(f"  the overnight run first matched the rung's final reward ({thr:.4f}) at "
          f"iteration {first},")
    print(f"  and spent {above * 100:.1f}% of its iterations at or above it.")
    print()

    print("=" * 78)
    print("9. STRAFE -- commanded in all four runs, quality measured in none")
    print("=" * 78)
    for label, d in rows:
        lo, hi = _lin_vel_y_range(d / "params" / "env.yaml")
        print(f"  {label:<18s} commanded lin_vel_y = ({lo}, {hi}) m/s")
    # `vel_y` not followed by another letter: `error_vel_yaw` is a YAW metric and
    # a substring match would count it as a lateral one, which is exactly the
    # confusion this section exists to rule out.
    lateral = re.compile(r"vel_y(?![a-z])")
    per_axis = [t for t in sorted(set(everywhere) | set(overnight)) if lateral.search(t)]
    print(f"  metrics naming the lateral axis, in any run: {per_axis if per_axis else 'NONE'}")
    print("  ^ error_vel_xy is the norm of a 2-D error. A policy that tracks v_x")
    print("    perfectly and never strafes and one that splits the error between the")
    print("    axes produce the same number, so no reading above says anything about")
    print("    how well the machine sidesteps. Strafe quality is UNMEASURED.")
    print()

    print("=" * 78)
    print("10. WALL CLOCK, SAMPLES AND THROUGHPUT")
    print("=" * 78)
    print(f"  {'run':<18s} {'iters':>7s} {'samples':>15s} {'console steps':>15s} "
          f"{'elapsed':>10s} {'mean fps':>10s}")
    for label, d in rows:
        agent_yaml = d / "params" / "agent.yaml"
        env_yaml = d / "params" / "env.yaml"
        iters = int(_scalar_field(agent_yaml, r"^max_iterations: (.*)$"))
        per_env = int(_scalar_field(agent_yaml, r"^num_steps_per_env: (.*)$"))
        envs = int(_scalar_field(env_yaml, r"^  num_envs: (.*)$"))
        samples = iters * envs * per_env
        scal = ladder[label.split("/", 1)[1]] if label.startswith("ladder/") else overnight
        seg = segments[label.split("/", 1)[1]] if label.startswith("ladder/") \
            else OVERNIGHT_CONSOLE.read_text(errors="replace")
        console_steps = int(re.findall(r"^\s*Total steps:\s*(\d+)\s*$", seg, re.M)[-1])
        elapsed = re.findall(r"^\s*Time elapsed:\s*(\d+:\d\d:\d\d)\s*$", seg, re.M)[-1]
        fps = statistics.fmean(v for _, v in scal["Perf/total_fps"])
        print(f"  {label:<18s} {iters:>7,d} {samples:>15,d} {console_steps:>15,d} "
              f"{elapsed:>10s} {fps:>10,.0f}")
        if console_steps != samples:
            raise AssertionError(
                f"{label}: iters x envs x num_steps_per_env = {samples:,} but the console's "
                f"last 'Total steps' says {console_steps:,}. The config dump and the run "
                "disagree about how many samples were collected."
            )
    print("  ^ `samples` is derived (iterations x envs x num_steps_per_env) and")
    print("    `console steps` is what the run printed; they must be equal, and the")
    print("    script raises rather than prints if they are not.")


if __name__ == "__main__":
    main()
