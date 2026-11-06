"""lidarv2 parallel arena — use processes, because the probe showed GIL-bound.

`run_parallel_arena` farms each fuser and size estimator out to a process pool,
then reuses the serial illusion and triangulator stages. Falls back to serial if
multiprocessing is restricted on the device.
"""

import json
import multiprocessing
import os
import time

from arena.arena import (
    evaluate_illusions,
    evaluate_triangulators,
    eval_fuser,
    eval_size,
    snip,
)
from arena.depth_fusers import FUSERS
from arena.size_estimators import SIZE_ESTIMATORS, SLOW_ESTIMATORS
from fixtures.premade_objects import PREMADE


def _fuser_task(args):
    name, fixtures, depths, noises, level = args
    return eval_fuser(name, fixtures, depths, noises, level)


def _size_task(args):
    name, fixtures, depths, noises, fuser, level = args
    return eval_size(name, fixtures, depths, noises, fuser, level)


def run_parallel_arena(workers=None, quick=False, top_k=3, include_slow=False):
    workers = workers or (os.cpu_count() or 1)
    fixtures = PREMADE[:3] if quick else PREMADE
    depths = (2.0, 3.0) if quick else (1.0, 2.0, 3.0, 5.0)
    noises = (0.0, 0.02) if quick else (0.0, 0.02, 0.05)
    size_names = [n for n in SIZE_ESTIMATORS if include_slow or n not in SLOW_ESTIMATORS]

    start = time.perf_counter()
    parallel_error = None
    try:
        with multiprocessing.Pool(workers) as pool:
            fusers = pool.map(
                _fuser_task,
                [(n, fixtures, depths, noises, 0.95) for n in FUSERS],
            )
            sizes = pool.map(
                _size_task,
                [(n, fixtures, depths, noises, "inverse_variance", 0.95)
                 for n in size_names],
            )
    except Exception as exc:  # restricted multiprocessing -> serial fallback
        parallel_error = "%s: %s" % (type(exc).__name__, exc)
        fusers = [eval_fuser(n, fixtures, depths, noises, 0.95) for n in FUSERS]
        sizes = [eval_size(n, fixtures, depths, noises, "inverse_variance", 0.95)
                 for n in size_names]
    wall_ms = (time.perf_counter() - start) * 1000.0

    illusions = evaluate_illusions(fixtures, depths, noises)
    tris = evaluate_triangulators()

    fusers.sort(key=lambda r: r["score"])
    sizes.sort(key=lambda r: r["score"])
    report = {
        "fusers": fusers,
        "sizes": sizes,
        "illusions": illusions,
        "triangulators": tris,
        "workers": workers,
        "wall_ms": round(wall_ms, 2),
        "parallel_error": parallel_error,
    }
    cut = snip(report, top_k=top_k)
    report["survivors"] = cut["keep"]
    report["pruned"] = cut["prune"]
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="lidarv2 process-pool arena")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--out", default="arena_parallel_report.json")
    args = parser.parse_args(argv)

    report = run_parallel_arena(workers=args.workers, quick=args.quick, top_k=args.top)
    print("workers      :", report["workers"])
    print("wall_ms      :", report["wall_ms"])
    if report.get("parallel_error"):
        print("parallel_error:", report["parallel_error"])
    print("survivors    :", json.dumps(report["survivors"]))
    print("pruned       :", json.dumps(report["pruned"]))
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out)
    return report


if __name__ == "__main__":
    main()
