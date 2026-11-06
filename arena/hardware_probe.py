"""lidarv2 hardware probe — measure the phone, then budget the math.

Feeds the arena with real throughput numbers: CPU iterations/s (single and
threaded), fusion ops/s, peak memory, and a recommended algorithm budget.
Pure stdlib. `stress()` is intentionally bounded so it does not cook the device.
"""

import json
import math
import multiprocessing
import os
import platform
import threading
import time
import tracemalloc

from arena.depth_fusers import FUSERS
from depth_engine import DepthEngine
from fixtures.premade_objects import by_tag, synthesize


def _meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        pass
    return info


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                low = line.lower()
                if low.startswith("model name") or low.startswith("hardware"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _burn(seconds):
    end = time.perf_counter() + seconds
    n = 0
    x = 0.0
    while time.perf_counter() < end:
        x += math.sin(n) * math.cos(n)
        n += 1
    return n


def stress(seconds=2.0, workers=None):
    """Push every core for `seconds`. Bounded by design."""
    workers = workers or (os.cpu_count() or 1)
    counts = []
    lock = threading.Lock()

    def worker():
        n = _burn(seconds)
        with lock:
            counts.append(n)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = max(1e-9, time.perf_counter() - start)
    return {
        "workers": workers,
        "elapsed_s": round(elapsed, 4),
        "iterations": sum(counts),
        "ops_per_sec": round(sum(counts) / elapsed, 1),
    }


def _mp_burn(seconds):
    return _burn(seconds)


def stress_processes(seconds=1.0, workers=None):
    """True multi-core stress via processes (bypasses the GIL)."""
    workers = workers or (os.cpu_count() or 1)
    start = time.perf_counter()
    with multiprocessing.Pool(workers) as pool:
        counts = pool.map(_mp_burn, [seconds] * workers)
    elapsed = max(1e-9, time.perf_counter() - start)
    return {
        "workers": workers,
        "elapsed_s": round(elapsed, 4),
        "iterations": sum(counts),
        "ops_per_sec": round(sum(counts) / elapsed, 1),
    }


def probe(seconds=0.5, workers=None, level=0.95, processes=False):
    cpu = os.cpu_count() or 1
    workers = workers or cpu
    mem = _meminfo()

    start = time.perf_counter()
    single = _burn(seconds)
    single_elapsed = max(1e-9, time.perf_counter() - start)
    single_ops = single / single_elapsed

    multi = stress(seconds, workers)
    scaling = multi["ops_per_sec"] / single_ops if single_ops else 0.0

    process_result = None
    if processes:
        try:
            process_result = stress_processes(seconds, workers)
        except Exception as exc:  # multiprocessing can be restricted on Android
            process_result = {"error": str(exc)}

    engine = DepthEngine()
    obs = synthesize(by_tag("2x4"), 2.0)
    cues = engine.cues(obs)
    tracemalloc.start()
    start = time.perf_counter()
    iters = 0
    while time.perf_counter() - start < seconds:
        FUSERS["inverse_variance"](cues, level)
        iters += 1
    fusion_elapsed = max(1e-9, time.perf_counter() - start)
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    fusion_ops = iters / fusion_elapsed
    total_mb = mem.get("MemTotal", 0) / 1024.0
    avail_mb = mem.get("MemAvailable", 0) / 1024.0

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": cpu,
        "cpu_model": _cpu_model(),
        "mem_total_mb": round(total_mb, 1),
        "mem_available_mb": round(avail_mb, 1),
        "single_core_ops_per_sec": round(single_ops, 1),
        "multi_core_ops_per_sec": multi["ops_per_sec"],
        "thread_scaling": round(scaling, 3),
        "process_ops_per_sec": (process_result.get("ops_per_sec")
                                if process_result and "ops_per_sec" in process_result else None),
        "process_scaling": (round(process_result["ops_per_sec"] / single_ops, 3)
                            if process_result and "ops_per_sec" in process_result and single_ops else None),
        "process_stress": process_result,
        "fusion_ops_per_sec": round(fusion_ops, 1),
        "fusion_peak_mem_mb": round(peak_mb, 4),
        "budget": {
            "max_fusions_per_second": int(fusion_ops),
            "max_arena_evals_per_second": int(fusion_ops * 0.5),
            "memory_headroom_mb": round(max(0.0, avail_mb * 0.25), 1),
            "recommended_threads": 1 if scaling < 1.0 else workers,
            "recommended_processes": workers,
            "gil_bound": scaling < 1.0,
        },
    }


def suggest(report, top_k=3):
    """Pick the arena survivors that fit the measured budget."""
    budget = report.get("budget", {})
    keep = {}
    for stage in ("fusers", "sizes", "illusions", "triangulators"):
        rows = report.get(stage) or []
        keep[stage] = [r["method"] for r in rows[:top_k]]
    return {
        "budget": budget,
        "keep": keep,
        "note": "top-%d per stage fits the measured fusion/CPU budget" % top_k,
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="lidarv2 hardware probe")
    parser.add_argument("--seconds", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-seconds", type=float, default=2.0)
    parser.add_argument("--processes", action="store_true",
                        help="also stress with multiprocessing (bypasses the GIL)")
    parser.add_argument("--json", default="hardware_probe.json")
    args = parser.parse_args(argv)

    result = probe(seconds=args.seconds, workers=args.workers, processes=args.processes)
    if args.stress:
        result["stress"] = stress(args.stress_seconds, args.workers)
    for key, value in result.items():
        if key != "budget":
            print("%-28s %s" % (key, value))
    print("%-28s %s" % ("budget", json.dumps(result["budget"])))
    with open(args.json, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", args.json)
    return result


if __name__ == "__main__":
    main()
