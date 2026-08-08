"""Every number in `research/episodes/017` and in the P15-P17 calibration rows.

    venv/bin/python research/scripts/019_forward_v5_paired_reads.py

The paired forward-v5 probes produced THREE rsl_rl runs and one console log:

    spyder_forward_v5_s1/box_logs/<ts>   fine-tune of the v4 forward probe,
                                         iterations 1499 -> 2998 on v5 ground
    hound_forward_v5_s1/box_logs/<ts1>   Hound from scratch, 0 -> 1499
    hound_forward_v5_s1/box_logs/<ts2>   Hound fine-tune, 1499 -> 2298

The single console log lives under the HOUND run root and carries BOTH probe
segments (`=== SPIDER-V5 START ... ===` / `=== HOUND-V5 START ... ===`); the
Hound fine-tune was launched afterwards and has no console log at all, which
this script states rather than hides.

Same contract as `018_ladder_overnight_reads.py`: the number rule says a figure
enters the record because code computed it, so the episode quotes this script's
output and the calibration rows cite this script by name.

What it does, in order:

1. **The runs, from each run's own launch-time config dump.** One seed each.

2. **Cross-checks tb against the console log** for the two runs that have one.
   The event file and the terminal block are written by different code paths in
   rsl_rl, so they are an independent transcription of the same quantity.

3. **Prints peak beside final, everywhere.** All three runs peak well above
   where they finish; every verdict below is scored on the FINAL, which is what
   each claim was pre-registered against. Scoring on the peak is the
   substitution this record exists to make impossible (episodes/015, P9).

4. **The Spyder fine-tune against its own parent** (`spyder_forward_s1`,
   the v4 forward probe whose `model_1499.pt` it resumed from) on the metrics
   both logged. Two variables moved -- terrain v4 -> v5 AND 1500 more
   iterations -- so this is a description of what the fine-tune produced, never
   an attribution to either.

5. **Resolves P15 / P16 / P17** against the thresholds exactly as
   pre-registered, at the iteration each was pre-registered for. P16 carries
   its own arithmetic: the pure-rolling ceiling is a derived quantity, so the
   derivation is recomputed here rather than quoted.

6. **The Hound fine-tune's deltas** against the scratch run it resumed from,
   final-versus-final AND trailing-mean-versus-trailing-mean, because those two
   comparisons do not agree and only one of them is quotable as a headline.

7. **The speed each training log implies**, against the speed the deterministic
   playback of the same checkpoint showed. On the Spyder the two are consistent;
   on the Hound they are not, by a factor this script computes and does not
   explain.

8. **Wall clock, samples and throughput**, with the derived sample count
   asserted equal to what each run printed.

Reads only artifacts under `runs/`: event files, the `params/*.yaml` each run
dumps at launch, and the console log. No GPU, no Isaac, no network. `runs/` is
gitignored, so this script is the durable form of those numbers and it fails
loudly rather than quietly if an artifact is missing.
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

SPYDER_V4_ROOT = paths.RUNS / "spyder_forward_s1"       # the parent, v4 ground
SPYDER_V5_ROOT = paths.RUNS / "spyder_forward_v5_s1"    # the fine-tune, v5
HOUND_V5_ROOT = paths.RUNS / "hound_forward_v5_s1"      # scratch + fine-tune

#: The console log the paired probes wrote. It sits under the HOUND root
#: because the driver appended both segments to one file.
PROBE_CONSOLE = HOUND_V5_ROOT / "box_console.log"

#: Markers the probe driver wrote around each segment.
#: `=== SPIDER-V5 START 04:35:32 ===`
SEGMENT_START = re.compile(r"^=== (SPIDER-V5|HOUND-V5) START .*===$", re.M)

#: The two lines rsl_rl prints per iteration, at two decimals.
CONSOLE_REWARD = re.compile(r"^\s*Mean reward:\s*(-?[\d.]+)\s*$", re.M)
CONSOLE_LENGTH = re.compile(r"^\s*Mean episode length:\s*(-?[\d.]+)\s*$", re.M)

#: Console rounds to 2 dp, so agreement means "inside half a unit in the last
#: place". Anything wider means tb and the terminal are not the same run.
CONSOLE_TOL = 0.005

# --- The pre-registered thresholds, copied from research/calibration.jsonl ---
# Copied, not re-derived: a threshold that drifts between the prediction and
# its resolution is the one failure mode a calibration record cannot survive.
P15_MIN_REWARD = 35.0     # Spyder v5 final mean reward, at iteration 1500
P16_MIN_REWARD = 18.1     # Hound final mean reward, at iteration 1500
P17_MIN_REWARD = 10.0     # Hound mean reward, BY iteration 500
P17_BY_ITER = 500

# --- The pure-rolling ceiling P16 was built on, recomputed not quoted --------
# hound_cfg.wheel_velocity_gain() docstring: at the 3.0 N.m effort ceiling the
# velocity drive saturates at 3.0 / 0.28128 = 10.665 rad/s of commanded wheel
# speed. Rim speed is that times the wheel radius.
WHEEL_EFFORT_LIMIT_NM = 3.0
WHEEL_VELOCITY_GAIN = 0.28128   # N.m per rad/s, hound_cfg.wheel_velocity_gain()
WHEEL_RADIUS_M = 0.08496        # rim speed 0.906 m/s at 10.665 rad/s

#: How many trailing iterations the "last-N mean" uses, matching 018.
LAST_N = 10

#: A second, wider trailing window. Ten iterations of a 4096-env run is still a
#: small episode sample; 100 is what the "best100" peak is measured over, so the
#: same window is used at the end of the run to make the two comparable.
LAST_WIDE = 100

#: Playback speeds, m/s, read off the deterministic single-robot rollouts of
#: these same checkpoints. NOT produced by this script and NOT reproducible from
#: anything under `runs/` -- the player prints telemetry to a terminal and the
#: only artifact kept is the video. They are carried here so section 8 can put
#: them beside the number the training log implies; every use of them in the
#: record must say where they came from. Parent figure: episodes/014.
PLAYBACK_MS = {
    "spyder v4 (parent)": (4.2, 5.4),
    "spyder v5 (fine-tune)": (5.5, 6.5),
    # The Hound figure is the SCRATCH checkpoint (model_1499): its clip,
    # runs/spyder_forward_v5_s1/hound_v5_run.mp4, was written before the
    # fine-tune started, so it cannot be the fine-tuned policy. The fine-tuned
    # checkpoint has strip footage but no speed reading.
    "hound v5 (scratch)": (8.0, 10.0),
}

#: How far up the demo strip each deterministic attempt got, world-frame x in
#: metres, read off the player's telemetry. Observed, like PLAYBACK_MS, and
#: carried here only so section 10 can turn them into strip fractions from the
#: strip's OWN committed geometry instead of from arithmetic done in prose.
STRIP_ATTEMPT_X_M = {"model_1499.pt": 20.0, "model_2298.pt": 26.4}

#: The demo strip's geometry, parsed out of its config module rather than
#: retyped. `spyder_demo_env_cfg` imports isaaclab at module scope and this
#: script must run in the MuJoCo venv, so the constants are read from the source
#: text; a rename or a moved number makes the parse fail loudly.
DEMO_CFG = Path(__file__).resolve().parents[2] / "src" / "bestiary" / "isaac" / "spyder_demo_env_cfg.py"

#: Physics steps per second of simulated time: `episode_length_s` is 20.0 and a
#: full episode is 1000 steps, so a logged episode length in steps times this is
#: seconds. Asserted against each run's own config dump rather than trusted.
CONTROL_DT_S = 0.02

# --------------------------------------------------------------------------


def _log_dirs(root: Path) -> list[Path]:
    """Every timestamped rsl_rl directory under `root/box_logs`, sorted."""
    parent = root / "box_logs"
    if not parent.is_dir():
        raise FileNotFoundError(
            f"{parent} does not exist. The runs/ tree is gitignored; this script "
            "reads artifacts that must be present on the machine that pulled them."
        )
    leaves = sorted(p for p in parent.iterdir() if p.is_dir())
    if not leaves:
        raise AssertionError(f"{parent} holds no timestamped directories")
    return leaves


def _one_log_dir(root: Path) -> Path:
    leaves = _log_dirs(root)
    if len(leaves) != 1:
        raise AssertionError(
            f"{root / 'box_logs'} holds {len(leaves)} timestamped directories "
            f"{[p.name for p in leaves]}, expected exactly 1."
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


def _peak(series: list[tuple[int, float]]) -> tuple[int, float]:
    value = max(v for _, v in series)
    return next(s for s, v in series if v == value), value


def _last_n_mean(series: list[tuple[int, float]], n: int = LAST_N) -> float:
    return statistics.fmean(v for _, v in series[-n:])


def _rolling_max(series: list[tuple[int, float]], window: int) -> tuple[int, float]:
    """(iteration, value) of the highest `window`-iteration trailing mean.

    A single rsl_rl iteration averages over whatever episodes happened to end
    in it, so the per-iteration series is very noisy: quoting its maximum as
    "the run reached X" overstates by the noise amplitude. The trailing mean is
    what a reader means by "mid-run it was running at about X".
    """
    if len(series) < window:
        raise AssertionError(f"series has {len(series)} points, window is {window}")
    best = (series[window - 1][0], statistics.fmean(v for _, v in series[:window]))
    for i in range(window - 1, len(series)):
        m = statistics.fmean(v for _, v in series[i - window + 1:i + 1])
        if m > best[1]:
            best = (series[i][0], m)
    return best


def _console_finals(text: str) -> tuple[float, float]:
    """The last `Mean reward` / `Mean episode length` printed in a segment."""
    rewards = CONSOLE_REWARD.findall(text)
    lengths = CONSOLE_LENGTH.findall(text)
    if not rewards or not lengths:
        raise AssertionError(
            f"found {len(rewards)} reward lines and {len(lengths)} length lines in this "
            "console segment; the printer's format has changed and this parser is stale."
        )
    return float(rewards[-1]), float(lengths[-1])


def _console_segments() -> dict[str, str]:
    """Split the paired-probe console log into one segment per robot."""
    text = PROBE_CONSOLE.read_text(errors="replace")
    marks = list(SEGMENT_START.finditer(text))
    if len(marks) != 2:
        raise AssertionError(
            f"{PROBE_CONSOLE} has {len(marks)} START markers, expected 2 "
            "(SPIDER-V5 and HOUND-V5). This is not the paired-probe log."
        )
    segments = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        segments[m.group(1)] = text[m.start():end]
    if set(segments) != {"SPIDER-V5", "HOUND-V5"}:
        raise AssertionError(f"split the console into {sorted(segments)}")
    return segments


def _scalar_field(env_yaml: Path, pattern: str) -> str:
    m = re.search(pattern, env_yaml.read_text(), re.M)
    if m is None:
        raise AssertionError(f"{env_yaml} has no line matching {pattern!r}")
    return m.group(1).strip()


def _reward_weights(env_yaml: Path) -> dict[str, float]:
    """`term -> weight` for the LIVE reward terms in a launched run's dump.

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


def _verdict(name: str, measured: float, bar: float, op: str) -> bool:
    got = measured >= bar if op == ">=" else measured <= bar
    print(f"  {name:<6s} measured {measured:10.4f} {op} {bar:<8.4f} -> "
          f"{'TRUE' if got else 'FALSE'}")
    return got


# --------------------------------------------------------------------------


def main() -> None:
    spyder_v4_dir = _one_log_dir(SPYDER_V4_ROOT)
    spyder_v5_dir = _one_log_dir(SPYDER_V5_ROOT)
    hound_dirs = _log_dirs(HOUND_V5_ROOT)
    if len(hound_dirs) != 2:
        raise AssertionError(
            f"{HOUND_V5_ROOT / 'box_logs'} holds {len(hound_dirs)} run directories, "
            "expected 2 (the scratch run and the fine-tune that resumed it)."
        )
    hound_scratch_dir, hound_ft_dir = hound_dirs

    runs = {
        "spyder v4 (parent)": spyder_v4_dir,
        "spyder v5 (fine-tune)": spyder_v5_dir,
        "hound v5 (scratch)": hound_scratch_dir,
        "hound v5 (fine-tune)": hound_ft_dir,
    }
    tb = {label: _scalars(d) for label, d in runs.items()}
    segments = _console_segments()

    print("=" * 78)
    print("0. THE RUNS, from each run's own launch-time config dump")
    print("=" * 78)
    for label, d in runs.items():
        env_yaml = d / "params" / "env.yaml"
        agent_yaml = d / "params" / "agent.yaml"
        rew = tb[label]["Train/mean_reward"]
        print(f"  {label:<22s} {d.name}")
        print(f"  {'':<22s} seed {_scalar_field(env_yaml, r'^seed: (.*)$')}  "
              f"envs {_scalar_field(env_yaml, r'^  num_envs: (.*)$')}  "
              f"episode {_scalar_field(env_yaml, r'^episode_length_s: (.*)$')} s  "
              f"max_iterations {_scalar_field(agent_yaml, r'^max_iterations: (.*)$')}")
        print(f"  {'':<22s} logged iterations {rew[0][0]}..{rew[-1][0]} ({len(rew)} points)  "
              f"reward terms: {sorted(_reward_weights(env_yaml))}")
    print()
    print("  ^ ONE SEED per arm, and two of the four are fine-tunes that inherit")
    print("    every gradient of the run they resumed. Nothing below is a finding.")
    print()

    print("=" * 78)
    print("1. TENSORBOARD vs CONSOLE  (independent transcriptions of one run)")
    print("=" * 78)
    for label, seg_key in (("spyder v5 (fine-tune)", "SPIDER-V5"),
                           ("hound v5 (scratch)", "HOUND-V5")):
        step, tb_r = _final(tb[label]["Train/mean_reward"])
        _, tb_l = _final(tb[label]["Train/mean_episode_length"])
        c_r, c_l = _console_finals(segments[seg_key])
        for what, tb_v, c_v in (("reward", tb_r, c_r), ("ep length", tb_l, c_l)):
            gap = abs(tb_v - c_v)
            print(f"  {label:<22s} iter {step:<5d} {what:<9s} tb {tb_v:9.4f}  "
                  f"console {c_v:8.2f}  |gap| {gap:.4f}  "
                  f"{'agree' if gap <= CONSOLE_TOL else 'MISMATCH'}")
            if gap > CONSOLE_TOL:
                raise AssertionError(
                    f"{label} {what}: tb says {tb_v}, console says {c_v}, {gap:.4f} apart "
                    f"against a {CONSOLE_TOL} tolerance. One of the two is not this run "
                    "and neither may be published."
                )
    print("  hound v5 (fine-tune)   NO CONSOLE LOG EXISTS. It was launched after the")
    print("    paired-probe driver exited, so its numbers have ONE transcription and")
    print("    no cross-check. Stated, not hidden.")
    print()

    print("=" * 78)
    print("2. FINAL vs PEAK vs LAST-10  (every verdict below is scored on FINAL)")
    print("=" * 78)
    print(f"  {'run':<22s} {'final it':>9s} {'final rew':>10s} {'last10':>9s} "
          f"{'last100':>9s} {'peak it':>8s} {'peak rew':>9s} {'best100':>9s} {'final len':>10s}")
    for label in runs:
        r = tb[label]["Train/mean_reward"]
        length = tb[label]["Train/mean_episode_length"]
        fin_it, fin_r = _final(r)
        pk_it, pk_r = _peak(r)
        roll_it, roll_v = _rolling_max(r, LAST_WIDE)
        print(f"  {label:<22s} {fin_it:>9d} {fin_r:>10.4f} {_last_n_mean(r):>9.4f} "
              f"{_last_n_mean(r, LAST_WIDE):>9.4f} {pk_it:>8d} {pk_r:>9.4f} {roll_v:>9.4f} "
              f"{_final(length)[1]:>10.4f}")
    print()
    print("  `peak rew` is the single best ITERATION -- an average over whatever")
    print("  episodes happened to end in one iteration, so it carries the full")
    print("  sampling noise. `best100` is the best 100-iteration trailing mean, which")
    print("  is what 'mid-run it was running at about X' actually means. Both are")
    print("  printed so nobody has to reach for the peak to make a run sound better,")
    print("  and NEITHER decides a verdict: the claims name the final iteration.")
    for label in runs:
        r = tb[label]["Train/mean_reward"]
        pk_it, pk_r = _peak(r)
        roll_it, roll_v = _rolling_max(r, LAST_WIDE)
        fin_r = _final(r)[1]
        print(f"    {label:<22s} peak/final {pk_r / fin_r:5.2f}x   "
              f"best100 {roll_v:8.3f} at iter {roll_it}  ({roll_v / fin_r:.2f}x final)   "
              f"last100/final {_last_n_mean(r, LAST_WIDE) / fin_r:.2f}x")
    print()
    print("  READ THE `last100/final` COLUMN BEFORE QUOTING ANY FINAL. It is below 1")
    print("  where the final iteration flatters the run and above 1 where the final")
    print("  UNDERSTATES it. Both directions are present in this table, which is why a")
    print("  final is a verdict-scoring instrument and not a description of a policy.")
    print()

    print("=" * 78)
    print("3. THE SPYDER FINE-TUNE AGAINST ITS OWN PARENT")
    print("=" * 78)
    print("  TWO VARIABLES MOVED: terrain v4 -> v5 AND 1500 further iterations from")
    print("  the parent's model_1499. This table describes what came out; it")
    print("  attributes nothing to either variable, and cannot.")
    pairs = (
        ("mean reward (m/ep)", "Train/mean_reward", "higher"),
        ("mean episode length", "Train/mean_episode_length", "higher"),
        ("forward_velocity /s", "Episode_Reward/forward_velocity", "higher"),
        ("time_out share", "Episode_Termination/time_out", "higher"),
        ("base_contact share", "Episode_Termination/base_contact", "lower"),
        ("terrain level", "Curriculum/terrain_levels", "n/a -- different ground"),
    )
    print(f"  {'metric':<22s} {'v4 @1499':>11s} {'v5 @2998':>11s} {'change':>9s}  better is")
    for name, tag, better in pairs:
        a = _final(tb["spyder v4 (parent)"][tag])[1]
        b = _final(tb["spyder v5 (fine-tune)"][tag])[1]
        print(f"  {name:<22s} {a:11.4f} {b:11.4f} {(b / a - 1) * 100:+8.1f}%  {better}")
    print()
    print("  Terrain levels are NOT comparable across the two: v4 and v5 are different")
    print("  assets with different curricula, so the column is printed and not read.")
    print()

    print("=" * 78)
    print("4. P15 -- Spyder v5 final mean reward >= 35 m/episode at iteration 1500")
    print("=" * 78)
    sp_r = tb["spyder v5 (fine-tune)"]["Train/mean_reward"]
    sp_fin_it, sp_fin = _final(sp_r)
    print(f"  The fine-tune's 1500 iterations run 1499..{sp_fin_it}, so 'at iteration")
    print("  1500' is its final iteration: 1500 further iterations were requested and")
    print(f"  {len(sp_r)} were logged.")
    print("  THE LITERAL READING IS A TRAP, and it is printed so nobody re-derives it")
    print("  by hand later. rsl_rl's reward and length buffers are EMPTY at a resume and")
    print("  refill only as episodes complete, so the iterations numbered near 1500 in")
    print("  this log are an instrument warming up, not a policy:")
    for s in (1499, 1500, 1501, 1550, 1600):
        print(f"    iteration {s:>4d}  reward {_at(sp_r, s):9.4f}  "
              f"ep length {_at(tb['spyder v5 (fine-tune)']['Train/mean_episode_length'], s):8.2f}")
    print("  A 16.74-step mean episode length is not a machine that forgot how to walk;")
    print("  it is two or three short episodes in a buffer sized for a hundred. The")
    print("  claim is scored at the FINAL of the 1500 requested iterations (2998), which")
    print("  is what 'at iteration 1500' meant when it was written for a run that was")
    print("  then planned from scratch.")
    _verdict("P15", sp_fin, P15_MIN_REWARD, ">=")
    print(f"        margin {sp_fin - P15_MIN_REWARD:+.4f} on the final; the last-10 mean "
          f"{_last_n_mean(sp_r):.4f} clears it too.")
    print(f"        parent (v4, iteration 1499) finished at "
          f"{_final(tb['spyder v4 (parent)']['Train/mean_reward'])[1]:.4f}.")
    print()

    print("=" * 78)
    print("5. P16 -- Hound final mean reward > 18.1 m/episode at iteration 1500")
    print("=" * 78)
    rad_s = WHEEL_EFFORT_LIMIT_NM / WHEEL_VELOCITY_GAIN
    rim = rad_s * WHEEL_RADIUS_M
    ep_s = float(_scalar_field(hound_scratch_dir / "params" / "env.yaml",
                               r"^episode_length_s: (.*)$"))
    ceiling = rim * ep_s
    print("  The bar is DERIVED, so it is recomputed here rather than quoted:")
    print(f"    commanded wheel speed saturates at {WHEEL_EFFORT_LIMIT_NM} N.m / "
          f"{WHEEL_VELOCITY_GAIN} N.m per rad/s = {rad_s:.3f} rad/s")
    print(f"    rim speed = {rad_s:.3f} rad/s x {WHEEL_RADIUS_M} m = {rim:.4f} m/s")
    print(f"    a {ep_s:.0f} s episode of pure rolling at that speed earns "
          f"{ceiling:.3f} m, which is the pre-registered {P16_MIN_REWARD}")
    if abs(ceiling - P16_MIN_REWARD) > 0.05:
        raise AssertionError(
            f"recomputed rolling ceiling {ceiling:.4f} is not the pre-registered "
            f"{P16_MIN_REWARD}; the bar and its derivation have drifted apart."
        )
    ho_r = tb["hound v5 (scratch)"]["Train/mean_reward"]
    ho_fin_it, ho_fin = _final(ho_r)
    _verdict("P16", ho_fin, P16_MIN_REWARD, ">=")
    print(f"        iteration {ho_fin_it}, margin x{ho_fin / P16_MIN_REWARD:.2f} over the "
          f"ceiling ({ho_fin - P16_MIN_REWARD:+.2f} m/episode).")
    first_over = next((s for s, v in ho_r if v > P16_MIN_REWARD), None)
    frac_over = sum(1 for _, v in ho_r if v > P16_MIN_REWARD) / len(ho_r)
    print(f"        first iteration above the ceiling: {first_over}; "
          f"{frac_over * 100:.1f}% of all iterations sit above it.")
    print(f"        implied mean forward speed = {ho_fin:.2f} m / {ep_s:.0f} s = "
          f"{ho_fin / ep_s:.3f} m/s, x{ho_fin / ep_s / rim:.2f} the drive's saturation.")
    print("        NOTE the episode length: the machine does not survive the full")
    print(f"        episode ({_final(tb['hound v5 (scratch)']['Train/mean_episode_length'])[1]:.2f} "
          "steps of 1000), so the metres were earned in LESS than 20 s and the")
    print("        speed above is a floor on the speed, not an estimate of it.")
    print()

    print("=" * 78)
    print(f"6. P17 -- Hound mean reward >= 10 BY iteration {P17_BY_ITER}")
    print("=" * 78)
    print("  'By' is a deadline, not a point: the claim is satisfied if the series")
    print(f"  reaches {P17_MIN_REWARD} at ANY iteration <= {P17_BY_ITER}. Both readings "
          "are printed.")
    at_500 = _at(ho_r, P17_BY_ITER)
    first_ten = next((s for s, v in ho_r if v >= P17_MIN_REWARD), None)
    if first_ten is None:
        raise AssertionError("the Hound never reached mean reward 10 -- P17 is FALSE")
    best_by = max(v for s, v in ho_r if s <= P17_BY_ITER)
    best_by_it = next(s for s, v in ho_r if s <= P17_BY_ITER and v == best_by)
    print(f"  value AT iteration {P17_BY_ITER}          {at_500:10.4f}")
    print(f"  first iteration >= {P17_MIN_REWARD:.0f}            {first_ten:10d}  "
          f"(value {_at(ho_r, first_ten):.4f})")
    print(f"  best iteration <= {P17_BY_ITER}           {best_by:10.4f} at iteration {best_by_it}")
    print(f"  mean of iterations {P17_BY_ITER - 49}..{P17_BY_ITER}     "
          f"{statistics.fmean(v for s, v in ho_r if P17_BY_ITER - 49 <= s <= P17_BY_ITER):10.4f}")
    _verdict("P17", at_500, P17_MIN_REWARD, ">=")
    print(f"        TRUE on both readings, and not narrowly: x{at_500 / P17_MIN_REWARD:.1f} "
          f"the bar at the deadline, and the bar was cleared at iteration {first_ten} --")
    print(f"        {P17_BY_ITER - first_ten} iterations early. The early-verdict")
    print("        heuristic transfers to the heavier, higher-DoF body.")
    print("  Trajectory through the deadline:")
    for s in (50, 100, 150, 200, 300, 400, 500):
        print(f"    iteration {s:>4d}  reward {_at(ho_r, s):9.4f}  "
              f"ep length {_at(tb['hound v5 (scratch)']['Train/mean_episode_length'], s):8.2f}")
    print()

    print("=" * 78)
    print("7. THE HOUND FINE-TUNE  (+800 iterations from the scratch run's final)")
    print("=" * 78)
    ft = tb["hound v5 (fine-tune)"]
    print(f"  {'metric':<22s} {'scratch@1499':>13s} {'fine-tune@2298':>15s} {'change':>9s}")
    for name, tag in (("mean reward (m/ep)", "Train/mean_reward"),
                      ("mean episode length", "Train/mean_episode_length"),
                      ("forward_velocity /s", "Episode_Reward/forward_velocity"),
                      ("time_out share", "Episode_Termination/time_out"),
                      ("base_contact share", "Episode_Termination/base_contact"),
                      ("terrain level", "Curriculum/terrain_levels")):
        a = _final(tb["hound v5 (scratch)"][tag])[1]
        b = _final(ft[tag])[1]
        print(f"  {name:<22s} {a:13.4f} {b:15.4f} {(b / a - 1) * 100:+8.1f}%")
    print()
    print("  THE SAME COMPARISON ON TRAILING MEANS, which does not agree with it:")
    print(f"  {'window':<22s} {'scratch':>13s} {'fine-tune':>15s} {'change':>9s}")
    for name, n in (("final iteration only", 1), (f"last-{LAST_N} mean", LAST_N),
                    (f"last-{LAST_WIDE} mean", LAST_WIDE)):
        a = _last_n_mean(tb["hound v5 (scratch)"]["Train/mean_reward"], n)
        b = _last_n_mean(ft["Train/mean_reward"], n)
        print(f"  {name:<22s} {a:13.4f} {b:15.4f} {(b / a - 1) * 100:+8.1f}%")
    print("  ^ the headline +46% is final-versus-final, and the fine-tune's final")
    print("    iteration sits ABOVE its own trailing mean while the scratch run's sits")
    print("    BELOW its own. On the wider window the gain is roughly a third of the")
    print("    headline. The honest sentence is the trailing one; the final-versus-final")
    print("    figure is quotable only with this table attached.")
    print()

    print("=" * 78)
    print("8. IMPLIED SPEED FROM THE LOG vs SPEED SEEN IN PLAYBACK")
    print("=" * 78)
    print("  reward = base-frame v_x at weight 1.0, times the control period, so an")
    print("  episode's return IS metres of forward travel. Mean return over mean")
    print("  episode duration is therefore a time-weighted mean forward speed for the")
    print("  episodes that ended in that iteration -- approximate, because it is a")
    print("  ratio of two separately-averaged quantities.")
    print(f"  {'run':<22s} {'m/ep':>9s} {'steps':>8s} {'sec':>7s} {'implied m/s':>12s} "
          f"{'mph':>6s} {'playback m/s':>13s}")
    for label, d in runs.items():
        ep_len_s = float(_scalar_field(d / "params" / "env.yaml", r"^episode_length_s: (.*)$"))
        steps = _final(tb[label]["Train/mean_episode_length"])[1]
        secs = steps * CONTROL_DT_S
        if abs(ep_len_s / CONTROL_DT_S - 1000.0) > 1e-6:
            raise AssertionError(
                f"{label}: episode_length_s {ep_len_s} over CONTROL_DT_S {CONTROL_DT_S} is not "
                "the 1000-step episode the record's 'of 1000' language assumes."
            )
        metres = _final(tb[label]["Train/mean_reward"])[1]
        pb = PLAYBACK_MS.get(label)
        pb_s = f"{pb[0]:.1f}-{pb[1]:.1f}" if pb else "not measured"
        print(f"  {label:<22s} {metres:9.2f} {steps:8.2f} {secs:7.2f} {metres / secs:12.3f} "
              f"{metres / secs * 2.23694:6.1f} {pb_s:>13s}")
    print()
    print("  The Spyder rows are consistent: the deterministic single-robot playback is")
    print("  FASTER than the population's time-weighted mean, which is what a rollout")
    print("  with no action noise, no pushes and no early falls should be.")
    print("  The Hound row is NOT, and it is the wrong way round -- the log implies a")
    print("  speed well ABOVE what the playback showed. Three candidates, none tested:")
    print("    (a) base-frame v_x is not ground speed. A pitching, airborne gallop")
    print("        projects velocity onto a nose axis that swings; |v_x^b| <= |v| holds")
    print("        instantaneously, so this direction alone cannot explain an EXCESS.")
    print("    (b) the population is heterogeneous in a way one seeded rollout is not:")
    print("        an episode that tumbles down a v5 slope earns metres fast and dies,")
    print("        and time-weighting does not remove it, it up-weights it.")
    print("    (c) the playback figure is eyeballed telemetry, not an instrument.")
    print("  Recorded as an open discrepancy, not resolved. Anything that quotes a")
    print("  Hound speed must say which of the two numbers it is quoting.")
    print()

    print("=" * 78)
    print("9. WALL CLOCK, SAMPLES AND THROUGHPUT")
    print("=" * 78)
    print(f"  {'run':<22s} {'iters':>6s} {'samples':>14s} {'elapsed (tb)':>13s} "
          f"{'mean fps':>10s}")
    for label, d in runs.items():
        agent_yaml = d / "params" / "agent.yaml"
        env_yaml = d / "params" / "env.yaml"
        iters = int(_scalar_field(agent_yaml, r"^max_iterations: (.*)$"))
        per_env = int(_scalar_field(agent_yaml, r"^num_steps_per_env: (.*)$"))
        envs = int(_scalar_field(env_yaml, r"^  num_envs: (.*)$"))
        r = tb[label]["Train/mean_reward"]
        logged = len(r)
        fps = statistics.fmean(v for _, v in tb[label]["Perf/total_fps"])
        # Wall clock from the event file's own timestamps: the fine-tune has no
        # console, so the console's `Time elapsed` is not available for all four.
        minutes = _elapsed_minutes(d)
        print(f"  {label:<22s} {logged:>6d} {logged * envs * per_env:>14,d} "
              f"{minutes:>12.1f}m {fps:>10,.0f}")
        if label in ("spyder v5 (fine-tune)", "hound v5 (scratch)"):
            seg = segments["SPIDER-V5" if label.startswith("spyder") else "HOUND-V5"]
            console_steps = int(re.findall(r"^\s*Total steps:\s*(\d+)\s*$", seg, re.M)[-1])
            derived = iters * envs * per_env
            if console_steps != derived:
                raise AssertionError(
                    f"{label}: max_iterations x envs x num_steps_per_env = {derived:,} but "
                    f"the console's last 'Total steps' says {console_steps:,}. The config "
                    "dump and the run disagree about how many samples were collected."
                )
            secs = float(re.findall(r"^Training time: ([\d.]+) seconds$", seg, re.M)[-1])
            print(f"  {'':<22s} console 'Training time' {secs:.2f} s = {secs / 60:.1f} min, "
                  f"total steps {console_steps:,} (matches the derived count)")
    print()
    print("  `elapsed (tb)` is first-to-last scalar timestamp in the event file, so it")
    print("  excludes Kit's startup and is slightly SHORTER than the console's own")
    print("  'Training time'. For the fine-tunes it is the only wall clock that exists.")
    print()

    print("=" * 78)
    print("10. THE DEMO-STRIP ATTEMPTS, against the strip's own committed geometry")
    print("=" * 78)
    length = _demo_constant("DEMO_LENGTH_M")
    spawn = _demo_constant("SPAWN_X_M")
    start_x = -length / 2.0
    print(f"  strip runs x = {start_x:+.1f} .. {start_x + length:+.1f} m ({length:.0f} m long), "
          f"spawn pinned at x = {spawn:+.1f}")
    print(f"  strip length {length:.0f} m = {length * 3.28084:.0f} ft end to end")
    print(f"  {'checkpoint':<16s} {'reached x':>10s} {'travelled':>10s} {'ft':>7s} "
          f"{'% of strip':>11s} {'to summit':>10s} {'ft':>6s}")
    for ckpt, x in STRIP_ATTEMPT_X_M.items():
        travelled = x - spawn
        left = start_x + length - x
        print(f"  {ckpt:<16s} {x:>+10.1f} {travelled:>10.1f} {travelled * 3.28084:>7.0f} "
              f"{(x - start_x) / length * 100:>10.1f}% {left:>8.1f} m {left * 3.28084:>6.0f}")
    print("  ^ `% of strip` measures from the strip's START, not from the spawn, which")
    print("    is the number the clip filenames use. Neither attempt reached the summit;")
    print("    both x values are observed telemetry, not this script's product.")
    print()

    print("=" * 78)
    print("11. DUAL UNITS for every figure the episode quotes in prose")
    print("=" * 78)
    print("  CLAUDE.md: an explanation carries BOTH units, never one. The code and the")
    print("  logs stay SI; this section is the conversion, done by code so the episode")
    print("  never does arithmetic in prose.")
    print(f"  {'quantity':<40s} {'SI':>14s} {'US customary':>20s}")
    metres = [
        ("Hound final, per episode", ho_fin),
        ("Hound fine-tune final, per episode", _final(ft["Train/mean_reward"])[1]),
        ("pure-rolling ceiling, per episode", ceiling),
        ("Spyder v5 final, per episode", sp_fin),
        ("Spyder v4 parent final, per episode",
         _final(tb["spyder v4 (parent)"]["Train/mean_reward"])[1]),
    ]
    for name, m in metres:
        print(f"  {name:<40s} {m:11.2f} m {m * 3.28084:17.0f} ft")
    speeds = [
        ("wheel drive saturation", rim),
        ("Spyder v4 playback, low", PLAYBACK_MS["spyder v4 (parent)"][0]),
        ("Spyder v4 playback, high", PLAYBACK_MS["spyder v4 (parent)"][1]),
        ("Spyder v5 playback, low", PLAYBACK_MS["spyder v5 (fine-tune)"][0]),
        ("Spyder v5 playback, high", PLAYBACK_MS["spyder v5 (fine-tune)"][1]),
        ("Hound playback (scratch ckpt), low", PLAYBACK_MS["hound v5 (scratch)"][0]),
        ("Hound playback (scratch ckpt), high", PLAYBACK_MS["hound v5 (scratch)"][1]),
        ("Hound implied by the scratch log",
         ho_fin / (_final(tb["hound v5 (scratch)"]["Train/mean_episode_length"])[1] * CONTROL_DT_S)),
    ]
    for name, v in speeds:
        print(f"  {name:<40s} {v:9.3f} m/s {v * 2.23694:16.1f} mph")


def _demo_constant(name: str) -> float:
    """A module-level float constant read out of the demo strip's config source.

    Parsed rather than imported: that module imports isaaclab at module scope
    and this script runs in the MuJoCo venv. A rename fails here loudly instead
    of leaving a stale number hardcoded in a research script.
    """
    m = re.search(rf"^{name} = (-?[\d.]+)$", DEMO_CFG.read_text(), re.M)
    if m is None:
        raise AssertionError(
            f"{DEMO_CFG} has no module-level `{name} = <float>`. The demo strip's "
            "geometry moved or was renamed; this script's strip arithmetic is stale."
        )
    return float(m.group(1))


def _elapsed_minutes(event_dir: Path) -> float:
    """Minutes between the first and last scalar written to an event file."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    events = acc.Scalars("Train/mean_reward")
    return (events[-1].wall_time - events[0].wall_time) / 60.0


if __name__ == "__main__":
    main()
