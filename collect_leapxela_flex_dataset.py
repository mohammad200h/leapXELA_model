"""Dataset generator for the LeapXELA flex taxel hand.

For each object shape, sweeps the object count 1..5 at a *fixed* size and
collects N accepted samples per (shape, count) for the shape- and
quantity-inference models.

Acceptance is strict: a sample counts only if every spawned object was still in
the hand at the end, so the count label is never ambiguous. Episodes that lose
an object are retried (up to --max-attempts) rather than kept with a corrected
label. Buckets that cannot reach N are reported, not silently shortened.

Object size is calibrated first unless --scale is given: strict retention is the
binding constraint, and the size that survives count 5 is not obvious a priori.

    conda run --no-capture-output -n leapxela python \
        collect_leapxela_flex_dataset.py --samples 1
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from grasp_touch_test import (
    ABORT_CHECK_TIME,
    GRASP_PATTERNS,
    OBJECT_ADHESION_GAP,
    OBJECT_SIZES,
    PALM_PITCH_DEG,
    PLACEMENT_HEIGHT,
    PLACEMENT_JITTER,
    PREGRIP_FRACTION,
    SETTLE_DURATION,
    THUMB_DELAY,
    THUMB_GRIP_FRACTION,
    ObjectSpec,
    object_mass,
    run_episode,
)

CALIB_SCALES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
CALIB_COUNTS = (1, 3, 5)
CALIB_TRIALS = 3
# Retention is decided by the close and the first squeeze cycles, so calibration
# does not need the full episode length.
CALIB_DURATION = 6.0
STAGING_NAME = "_attempt"


def episode_args(args: argparse.Namespace, duration: float, video: bool):
    """The Namespace `run_episode` expects, with dataset-level knobs applied."""
    return argparse.Namespace(
        duration=duration,
        # Never abort a video run: those re-simulate an episode that was already
        # accepted, and an abort would leave the sample folder without its npz.
        abort_check_time=0.0 if video else args.abort_check_time,
        sample_hz=args.sample_hz,
        grip_fraction=args.grip_fraction,
        thumb_grip_fraction=args.thumb_grip_fraction,
        thumb_delay=args.thumb_delay,
        pregrip_fraction=args.pregrip_fraction,
        palm_pitch=args.palm_pitch,
        object_adhesion=args.object_adhesion,
        object_adhesion_gap=args.object_adhesion_gap,
        placement_jitter=args.placement_jitter,
        placement_height=args.placement_height,
        settle_duration=args.settle_duration,
        orientation_random=True,
        object_mass=0.0,
        seed=args.seed,
        pulse_hz=args.pulse_hz,
        pulse_amplitude=args.pulse_amplitude,
        shear_amplitude=args.shear_amplitude,
        squeeze_steps=args.squeeze_steps,
        orbit_revolutions=args.orbit_revolutions,
        heatmap_channel=args.heatmap_channel,
        force_scale=args.force_scale,
        cloud_view=args.cloud_view,
        cloud_grid=args.cloud_grid,
        arrow_scale=args.arrow_scale,
        video=video,
    )


def uniform_specs(shape: str, scale: float, count: int) -> list:
    """`count` identical objects.

    Built directly rather than through `sample_specs`, whose multi-object path
    deliberately randomises shape and size from the --multi-* pools; this
    dataset fixes both so only the count varies.
    """
    return [ObjectSpec(shape, scale, object_mass(scale, 0.0))] * count


def episode_rng(seed: int, shape: str, count: int, sample: int, attempt: int):
    """Deterministic stream per attempt.

    zlib.crc32 rather than hash(): Python salts string hashes per process, so
    hash() would make --seed non-reproducible across runs and across workers.
    """
    return np.random.default_rng(
        [seed, zlib.crc32(shape.encode()), count, sample, attempt]
    )


def is_accepted(results: dict, criterion: str) -> bool:
    """Strict: every spawned object accounted for, and no divergence.

    `held` is the original criterion and is purely kinematic -- an object lying
    in the unsensed pocket between the palm pads and the finger bases satisfies
    it while contributing nothing to the tactile signal, so the episode keeps a
    count label the taxels cannot support. `sensed` requires each object to
    actually be touching a membrane, which is the guarantee the quantity task
    needs. Default stays `held` so existing yields do not shift silently.
    """
    counted = results["objects_held"] if criterion == "held" else results["objects_sensed"]
    return (
        results["divergences"] == 0
        and counted == results["object_count"]
        and results["passed"]
    )


def run_attempt(
    shape: str, scale: float, count: int, pattern: str, output_dir: Path,
    args: argparse.Namespace, duration: float, video: bool,
    sample: int, attempt: int,
) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    return run_episode(
        uniform_specs(shape, scale, count),
        pattern,
        episode_args(args, duration, video),
        output_dir,
        episode_rng(args.seed, shape, count, sample, attempt),
    )


def calib_trial(job: dict) -> dict:
    """One calibration episode, in its own staging dir so workers cannot collide."""
    args = job["args"]
    results = run_attempt(
        job["shape"], job["scale"], job["count"], args.grasp_pattern[0],
        job["staging"], args, args.calib_duration, False, 0, job["trial"],
    )
    if job["staging"].exists():
        shutil.rmtree(job["staging"])
    return {
        "key": (job["scale"], job["shape"], job["count"]),
        "accepted": int(is_accepted(results, args.acceptance)),
        "peak": results["peak_total_normal_n"],
        "aborted": results["aborted"],
    }


def calibrate(args: argparse.Namespace, staging_root: Path, pool) -> dict:
    """Strict-retention rate per (scale, shape, count), and the scale to use.

    Sweeps every shape the dataset will contain, not a single representative
    one: shapes differ enormously here -- measured at scale 0.3, box holds 5
    objects on the first attempt every time while spheres roll out of the hand,
    so calibrating on box alone picks a size that cannot fill the sphere
    buckets.

    Picks the scale with the best *worst-case* rate over every (shape, count)
    cell, since the hardest cell is what decides whether generation terminates.
    Ties break on peak force, preferring the size that reads more strongly.

    Trials are independent episodes, so they run on the same pool as collection.
    `episode_rng` keys off (shape, count, trial) alone, so the table -- and the
    scale it chooses -- does not depend on the order they complete in.
    """
    print(f"calibrating object size over scales {list(args.calib_scales)} "
          f"x shapes {list(args.shapes)} x counts {list(args.calib_counts)} "
          f"x {args.calib_trials} trials ({args.calib_duration:g} s, no video)")
    jobs = [
        {"args": args, "scale": scale, "shape": shape, "count": count,
         "trial": trial,
         "staging": staging_root / f"{STAGING_NAME}_calib_{index:04d}"}
        for index, (scale, shape, count, trial) in enumerate(
            (s, sh, c, t)
            for s in args.calib_scales
            for sh in args.shapes
            for c in args.calib_counts
            for t in range(args.calib_trials)
        )
    ]
    outcomes = {}
    for outcome in pool(calib_trial, jobs):
        outcomes.setdefault(outcome["key"], []).append(outcome)

    table = []
    for scale in args.calib_scales:
        cells = {}
        peaks = []
        for shape in args.shapes:
            for count in args.calib_counts:
                trials = outcomes[(scale, shape, count)]
                cells[f"{shape}_{count}"] = (
                    sum(o["accepted"] for o in trials) / args.calib_trials
                )
                # Aborted trials only saw the first ~2 s, so their peak is not
                # comparable to a full episode's -- averaging them in drags the
                # tie-break down in proportion to how often a scale drops an
                # object. This is therefore the mean peak over trials that ran
                # to completion, which the retention rates above already select
                # for. With --abort-check-time 0 nothing aborts and it is the
                # same mean as before the abort path existed; with abort on it
                # is a conditional mean and will read higher. Only ever used to
                # break a tie between scales with equal worst-case retention.
                peaks.extend(o["peak"] for o in trials if not o["aborted"])
        worst = min(cells.values())
        worst_cell = min(cells, key=cells.get)
        row = {
            "scale": scale,
            "rate_by_shape_count": cells,
            "worst_rate": worst,
            "worst_cell": worst_cell,
            # Every trial aborted: the scale held nothing, so worst_rate is 0
            # and this tie-break never gets consulted.
            "mean_peak_normal_n": float(np.mean(peaks)) if peaks else 0.0,
        }
        table.append(row)
        print(f"  scale {scale:4.2f} | worst {worst * 100:3.0f}% "
              f"at {worst_cell:<14} | mean peak "
              f"{row['mean_peak_normal_n']:5.2f} N")
    best = max(table, key=lambda r: (r["worst_rate"], r["mean_peak_normal_n"]))
    if best["worst_rate"] == 0.0:
        print("  WARNING: no scale in the sweep retained every object even "
              f"once at its hardest cell ({best['worst_cell']}). Those buckets "
              "will exhaust --max-attempts. Drop that count or shape from the "
              "run -- do NOT raise --pregrip-fraction to compensate: measured, "
              "0.50 and above makes the fingers diverge (NaN QACC at t~0.03) "
              "rather than gripping harder.")
    print(f"  chosen scale {best['scale']:g} (worst-case strict retention "
          f"{best['worst_rate'] * 100:.0f}% at {best['worst_cell']})")
    return {"table": table, "chosen_scale": best["scale"]}


def collect_sample(job: dict) -> dict:
    """Collect one accepted sample, retrying until --max-attempts.

    One sample -- not one bucket -- is the unit of work, because bucket costs
    differ by ~30x (box count 1 accepts first try; sphere count 5 exhausts 30
    attempts), and a pool given whole buckets is decided by its slowest one.
    Samples are independent: `episode_rng` keys off (shape, count, sample,
    attempt) and each writes its own `sample_NNN` directory.
    """
    args = job["args"]
    shape, count, scale, sample = (
        job["shape"], job["count"], job["scale"], job["sample"]
    )
    bucket = job["run_dir"] / shape / f"count_{count}"
    bucket.mkdir(parents=True, exist_ok=True)
    staging = bucket / f"{STAGING_NAME}_{sample:03d}"
    pattern = args.grasp_pattern[sample % len(args.grasp_pattern)]
    accepted = None
    attempts_used = 0
    for attempt in range(args.max_attempts):
        attempts_used += 1
        results = run_attempt(
            shape, scale, count, pattern, staging, args, args.duration,
            False, sample, attempt,
        )
        if is_accepted(results, args.acceptance):
            accepted = (attempt, results)
            break
    if accepted is None:
        if staging.exists():
            shutil.rmtree(staging)
        print(f"  {shape} count={count} sample {sample}: no accepted "
              f"episode in {args.max_attempts} attempts")
        return {"shape": shape, "count": count, "sample": sample,
                "attempts_used": attempts_used, "record": None}
    attempt, results = accepted
    sample_dir = bucket / f"sample_{sample:03d}"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    staging.rename(sample_dir)
    # Video is re-run rather than rendered up front: rejected attempts far
    # outnumber accepted ones under strict acceptance, and the episode is
    # deterministic in its seed, so this reproduces the kept episode exactly
    # while paying render cost only for samples that survive.
    wants_video = args.video_samples < 0 or sample < args.video_samples
    if wants_video:
        run_attempt(
            shape, scale, count, pattern, sample_dir, args, args.duration,
            True, sample, attempt,
        )
    record = {
        "path": str(sample_dir.relative_to(job["run_dir"])),
        "shape": shape,
        "count": count,
        "scale": scale,
        "pattern": pattern,
        "objects_held": results["objects_held"],
        "peak_total_normal_n": results["peak_total_normal_n"],
        "contact_fraction": results["contact_fraction"],
        "attempts": attempt + 1,
        "video": wants_video,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset"] = record
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  {shape} count={count} sample {sample:03d}: accepted on "
          f"attempt {attempt + 1}, peak {results['peak_total_normal_n']:.2f} N")
    return {"shape": shape, "count": count, "sample": sample,
            "attempts_used": attempts_used, "record": record}


def group_buckets(outcomes: list, shapes: list, counts: list,
                  requested: int) -> list:
    """Per-(shape, count) summaries, in the shapes x counts order of the args.

    Rebuilt from sample outcomes so `dataset.json` and `index.csv` keep the
    layout they had when a bucket was the unit of work, whatever order the pool
    finished in.
    """
    buckets = []
    for shape in shapes:
        for count in counts:
            mine = sorted(
                (o for o in outcomes
                 if o["shape"] == shape and o["count"] == count),
                key=lambda o: o["sample"],
            )
            samples = [o["record"] for o in mine if o["record"] is not None]
            buckets.append({
                "shape": shape,
                "count": count,
                "requested": requested,
                "accepted": len(samples),
                "attempts_used": sum(o["attempts_used"] for o in mine),
                "samples": samples,
            })
    return buckets


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=_MODEL_DIR,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a LeapXELA shape x count tactile dataset.",
    )
    parser.add_argument(
        "--shapes", nargs="+", choices=sorted(OBJECT_SIZES),
        default=["sphere", "box", "cylinder"],
    )
    parser.add_argument(
        "--counts", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="object counts to collect; one folder per count",
    )
    parser.add_argument(
        "--samples", type=int, default=1,
        help="accepted samples per (shape, count)",
    )
    parser.add_argument(
        "--scale", type=float, default=0.0,
        help="fixed object size multiplier; 0 calibrates it first",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--max-attempts", type=int, default=30,
        help="episodes tried per sample before giving up on it",
    )
    parser.add_argument(
        "--video-samples", type=int, default=1,
        help="accepted samples per bucket that also get grasp.mp4 "
             "(-1 renders every sample, 0 renders none)",
    )
    parser.add_argument(
        # Measured on a 10-core M-series: aggregate stepping throughput is 2.90x
        # at 3 workers, 7.29x at 8, and regresses to 7.00x at 10. Episodes are
        # single-threaded, so 8 is the knee -- lower it on a machine in use.
        "--workers", type=int, default=8,
        help="parallel worker processes; samples are independent",
    )
    parser.add_argument(
        "--abort-check-time", type=float, default=ABORT_CHECK_TIME,
        help="seconds after which an attempt that has already lost an object "
             "is abandoned rather than simulated to the end; 0 disables",
    )
    parser.add_argument(
        "--grasp-pattern", nargs="+", choices=GRASP_PATTERNS, default=["pulse"],
        help="pattern(s) cycled across the samples of each bucket",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calib-scales", nargs="+", type=float, default=list(CALIB_SCALES),
    )
    parser.add_argument(
        "--calib-counts", nargs="+", type=int, default=list(CALIB_COUNTS),
    )
    parser.add_argument("--calib-trials", type=int, default=CALIB_TRIALS)
    parser.add_argument("--calib-duration", type=float, default=CALIB_DURATION)
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--grip-fraction", type=float, default=0.55)
    parser.add_argument(
        "--thumb-grip-fraction", type=float, default=THUMB_GRIP_FRACTION,
    )
    parser.add_argument("--thumb-delay", type=float, default=THUMB_DELAY)
    parser.add_argument(
        "--pregrip-fraction", type=float, default=PREGRIP_FRACTION,
    )
    parser.add_argument(
        "--acceptance", choices=("held", "sensed"), default="held",
        help="what every spawned object must satisfy. 'held' is kinematic and "
             "counts objects lying in the unsensed pocket; 'sensed' requires "
             "each to actually touch a taxel membrane",
    )
    parser.add_argument("--palm-pitch", type=float, default=PALM_PITCH_DEG)
    parser.add_argument(
        "--object-adhesion", type=float, default=0.0,
        help="adhesive force in N between OBJECTS ONLY, holding a multi-object "
             "cluster together (0 = off). Never applied to object-hand or "
             "object-skin contacts, so the taxel signal is not biased. "
             "Needs MuJoCo >= 3.11",
    )
    parser.add_argument(
        "--object-adhesion-gap", type=float, default=OBJECT_ADHESION_GAP,
        help="distance in metres over which object adhesion keeps acting; "
             "must be > 0 or adhesion has no effect at any strength",
    )
    parser.add_argument(
        "--placement-jitter", type=float, default=PLACEMENT_JITTER,
    )
    parser.add_argument(
        "--placement-height", type=float, default=PLACEMENT_HEIGHT,
    )
    parser.add_argument(
        "--settle-duration", type=float, default=SETTLE_DURATION,
    )
    parser.add_argument("--pulse-hz", type=float, default=1.0)
    parser.add_argument("--pulse-amplitude", type=float, default=0.15)
    parser.add_argument("--shear-amplitude", type=float, default=0.15)
    parser.add_argument("--squeeze-steps", type=int, default=4)
    parser.add_argument("--orbit-revolutions", type=float, default=1.0)
    parser.add_argument("--heatmap-channel", default="normal")
    parser.add_argument("--force-scale", type=float, default=0.5)
    parser.add_argument("--cloud-view", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--cloud-grid", default="surface")
    parser.add_argument("--arrow-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-root", type=Path,
        default=_MODEL_DIR.parent / "data" / "leapxela_data",
        help="a timestamped run folder is created inside this root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    run_dir = args.output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"run folder: {run_dir}")

    # One pool for both phases. chunksize=1 because episode costs are wildly
    # uneven -- batching hands a worker several hard jobs while others idle.
    executor = (
        ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1
        else None
    )

    def pool(fn, jobs):
        """Run jobs, reporting progress -- a phase can be hundreds of episodes."""
        stream = (
            executor.map(fn, jobs, chunksize=1) if executor is not None
            else (fn(job) for job in jobs)
        )
        step = max(1, len(jobs) // 10)
        outcomes = []
        for outcome in stream:
            outcomes.append(outcome)
            if len(outcomes) % step == 0 or len(outcomes) == len(jobs):
                print(f"  [{len(outcomes)}/{len(jobs)} episodes, "
                      f"{time.time() - started:.0f}s elapsed]", flush=True)
        return outcomes

    try:
        calibration = None
        scale = args.scale
        if scale <= 0.0:
            calibration = calibrate(args, run_dir, pool)
            scale = calibration["chosen_scale"]
            (run_dir / "calibration.json").write_text(
                json.dumps(calibration, indent=2)
            )
        else:
            print(f"using fixed object scale {scale:g} (calibration skipped)")

        # Hardest first: high counts need the most attempts, so starting them
        # early keeps them off the tail where they would decide the wall clock.
        jobs = sorted(
            [
                {"args": args, "shape": shape, "count": count, "scale": scale,
                 "sample": sample, "run_dir": run_dir}
                for shape in args.shapes
                for count in args.counts
                for sample in range(args.samples)
            ],
            key=lambda job: -job["count"],
        )
        print(f"collecting {len(args.shapes) * len(args.counts)} buckets x "
              f"{args.samples} samples ({len(jobs)} episodes to fill) at scale "
              f"{scale:g}, {args.duration:g} s each, {args.workers} workers")
        outcomes = pool(collect_sample, jobs)
    finally:
        if executor is not None:
            executor.shutdown()

    buckets = group_buckets(outcomes, args.shapes, args.counts, args.samples)
    rows = [s for bucket in buckets for s in bucket["samples"]]
    with (run_dir / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "shape", "count", "scale", "pattern",
                        "objects_held", "peak_total_normal_n",
                        "contact_fraction", "attempts", "video"],
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    (run_dir / "dataset.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        # Flex forces are not comparable across MuJoCo versions (3.11 moved flex
        # elasticity into the CG solver), so the version is part of a dataset's
        # identity, not incidental environment detail.
        "mujoco_version": mujoco.__version__,
        "elapsed_seconds": round(elapsed, 1),
        "object_scale": scale,
        "acceptance": f"strict: objects_{args.acceptance} == object_count",
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "calibration": calibration,
        "buckets": buckets,
        "total_samples": len(rows),
    }, indent=2))

    short = [b for b in buckets if b["accepted"] < b["requested"]]
    print(f"\n{len(rows)} samples in {elapsed / 60:.1f} min -> {run_dir}")
    print(" shape      count  accepted  attempts  accept rate")
    for bucket in buckets:
        rate = bucket["accepted"] / max(bucket["attempts_used"], 1)
        print(f"  {bucket['shape']:<9} {bucket['count']:>4}  "
              f"{bucket['accepted']:>7}/{bucket['requested']:<3} "
              f"{bucket['attempts_used']:>7}  {rate * 100:>7.0f}%")
    if short:
        print("\nbuckets that did not reach the requested sample count:")
        for bucket in short:
            print(f"  {bucket['shape']} count={bucket['count']}: "
                  f"{bucket['accepted']}/{bucket['requested']} "
                  f"(hit --max-attempts {args.max_attempts})")


if __name__ == "__main__":
    main()
