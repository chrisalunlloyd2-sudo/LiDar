"""lidarv2 arena — benchmark every algorithm, rank it, snip the losers.

Stages
------
1. depth fusers     (17 methods)
2. size estimators  (11 methods; fit_rect is slow, opt-in)
3. illusion models  (none / canonical5 / extended10) under a synthetic bias
4. triangulators    (5 two-view methods)
5. survivors        top-K per stage by composite score

Score = MAPE + 2·max(0, 95 − coverage) + 0.01·runtime_ms   (lower is better)
"""

import json
import math
import time
import tracemalloc

from accuracy import percent_error
from arena.depth_fusers import FUSERS
from arena.illusion_models import ILLUSION_MODELS
from arena.size_estimators import SIZE_ESTIMATORS, SLOW_ESTIMATORS
from arena.triangulators import TRIANGULATORS
from depth_engine import DepthEngine
from fixtures.premade_objects import PREMADE, synthesize
from geospatial import enu_to_geodetic
from models import CameraIntrinsics, CameraPose
from triangulate import enu_direction_to_pixel

ILLUSION_STRESS = dict(convergence=0.6, arrowhead=0.5, specular=0.4, contrast=0.2)


def _score(mape, coverage, runtime_ms):
    # Accuracy first: the user's target is the % error to reality. Runtime is
    # only a tie-breaker, and coverage below 95 % is penalised.
    return mape + 2.0 * max(0.0, 95.0 - coverage) + 0.0001 * runtime_ms


def _apparent_bias(obs):
    conv = obs.convergence or 0.0
    arrow = obs.arrowhead or 0.0
    return (1.0 + 0.30 * conv + 0.20 * arrow, 1.0 + 0.15 * conv)


def _grid(fixtures, depths, noise_levels):
    i = 0
    for obj in fixtures:
        for depth in depths:
            for noise in noise_levels:
                i += 1
                yield i, obj, depth, noise


# ── stage 1: depth fusers ──────────────────────────────────────────────
def eval_fuser(name, fixtures=None, depths=(1.0, 2.0, 3.0, 5.0),
               noise_levels=(0.0, 0.02, 0.05), level=0.95):
    """Evaluate one fuser. Module-level and picklable for the process pool."""
    fixtures = fixtures or PREMADE
    engine = DepthEngine()
    fn = FUSERS[name]
    tracemalloc.start()
    start = time.perf_counter()
    errors, covered, total = [], 0, 0
    for seed, obj, depth, noise in _grid(fixtures, depths, noise_levels):
        obs = synthesize(obj, depth, noise_frac=noise, seed=seed)
        cues = engine.cues(obs)
        est = fn(cues, level)
        f = obs.intrinsics.f_px
        w = est["depth_m"] * obs.object_px_width / f * 1000.0
        h = est["depth_m"] * obs.object_px_height / f * 1000.0
        errors += [percent_error(w, obj["w_mm"]), percent_error(h, obj["h_mm"])]
        covered += 1 if est["ci_low_m"] <= depth <= est["ci_high_m"] else 0
        total += 1
    elapsed = (time.perf_counter() - start) * 1000.0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    mape = sum(errors) / len(errors) if errors else 0.0
    coverage = 100.0 * covered / max(1, total)
    return {
        "method": name, "mape": round(mape, 4), "coverage": round(coverage, 2),
        "runtime_ms": round(elapsed, 2), "peak_mem_mb": round(peak, 3),
        "score": round(_score(mape, coverage, elapsed), 4),
    }


def evaluate_fusers(fixtures=None, depths=(1.0, 2.0, 3.0, 5.0),
                    noise_levels=(0.0, 0.02, 0.05), level=0.95):
    fixtures = fixtures or PREMADE
    results = [eval_fuser(name, fixtures, depths, noise_levels, level) for name in FUSERS]
    return sorted(results, key=lambda r: r["score"])


# ── stage 2: size estimators ───────────────────────────────────────────
def eval_size(name, fixtures=None, depths=(1.0, 2.0, 3.0, 5.0),
              noise_levels=(0.0, 0.02, 0.05), fuser="inverse_variance",
              level=0.95):
    """Evaluate one size estimator. Module-level and picklable."""
    fixtures = fixtures or PREMADE
    engine = DepthEngine()
    fuse = FUSERS[fuser]
    fn = SIZE_ESTIMATORS[name]
    tracemalloc.start()
    start = time.perf_counter()
    errors = []
    for seed, obj, depth, noise in _grid(fixtures, depths, noise_levels):
        obs = synthesize(obj, depth, noise_frac=noise, seed=seed)
        cues = engine.cues(obs)
        est = fuse(cues, level)
        dims = fn(obs, est["depth_m"], level)
        errors += [percent_error(dims["w_mm"], obj["w_mm"]),
                   percent_error(dims["h_mm"], obj["h_mm"])]
    elapsed = (time.perf_counter() - start) * 1000.0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    mape = sum(errors) / len(errors) if errors else 0.0
    return {
        "method": name, "mape": round(mape, 4), "coverage": 100.0,
        "runtime_ms": round(elapsed, 2), "peak_mem_mb": round(peak, 3),
        "score": round(_score(mape, 100.0, elapsed), 4),
    }


def evaluate_sizes(fixtures=None, depths=(1.0, 2.0, 3.0, 5.0),
                   noise_levels=(0.0, 0.02, 0.05), fuser="inverse_variance",
                   level=0.95, include_slow=False):
    fixtures = fixtures or PREMADE
    names = [n for n in SIZE_ESTIMATORS if include_slow or n not in SLOW_ESTIMATORS]
    results = [eval_size(n, fixtures, depths, noise_levels, fuser, level) for n in names]
    return sorted(results, key=lambda r: r["score"])


# ── stage 3: illusion models ───────────────────────────────────────────
def evaluate_illusions(fixtures=None, depths=(1.0, 2.0, 3.0, 5.0),
                       noise_levels=(0.0, 0.02), fuser="inverse_variance",
                       size="pixel_scale", level=0.95):
    fixtures = fixtures or PREMADE
    engine = DepthEngine()
    fuse = FUSERS[fuser]
    size_fn = SIZE_ESTIMATORS[size]
    results = []
    for name, model in ILLUSION_MODELS.items():
        tracemalloc.start()
        start = time.perf_counter()
        errors = []
        for seed, obj, depth, noise in _grid(fixtures, depths, noise_levels):
            obs = synthesize(obj, depth, noise_frac=noise, seed=seed, **ILLUSION_STRESS)
            cues = engine.cues(obs)
            est = fuse(cues, level)
            dims = size_fn(obs, est["depth_m"], level)
            bw, bh = _apparent_bias(obs)
            apparent_w = dims["w_mm"] * bw
            apparent_h = dims["h_mm"] * bh
            report = model(obs, cues)
            f = report["combined_factor"]
            errors += [percent_error(apparent_w * f, obj["w_mm"]),
                       percent_error(apparent_h * f, obj["h_mm"])]
        elapsed = (time.perf_counter() - start) * 1000.0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        mape = sum(errors) / len(errors) if errors else 0.0
        results.append({
            "method": name, "mape": round(mape, 4), "coverage": 100.0,
            "runtime_ms": round(elapsed, 2), "peak_mem_mb": round(peak, 3),
            "score": round(_score(mape, 100.0, elapsed), 4),
        })
    return sorted(results, key=lambda r: r["score"])


# ── stage 4: triangulators ─────────────────────────────────────────────
def evaluate_triangulators(depths=(3.0, 5.0, 8.0), baseline_m=0.5, level=0.95):
    intr = CameraIntrinsics.from_fov(640, 480, 60.0)
    pose1 = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                       pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
    lat2, lon2, alt2 = enu_to_geodetic(0.0, 0.0, 0.0, baseline_m, 0.0, 0.0)
    pose2 = CameraPose(lat=lat2, lon=lon2, alt_m=alt2, heading_deg=0.0,
                       pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)

    results = []
    for name, fn in TRIANGULATORS.items():
        tracemalloc.start()
        start = time.perf_counter()
        errors = []
        for obj in PREMADE:
            w = obj["w_mm"] / 1000.0
            h = obj["h_mm"] / 1000.0
            for depth in depths:
                corners = [
                    (-w / 2, depth, h / 2), (w / 2, depth, h / 2),
                    (w / 2, depth, -h / 2), (-w / 2, depth, -h / 2),
                ]
                c1 = [enu_direction_to_pixel(*c, pose1, intr) for c in corners]
                c2 = [enu_direction_to_pixel(c[0] - baseline_m, c[1], c[2], pose2, intr)
                      for c in corners]
                if any(p is None for p in c1 + c2):
                    continue
                size = fn(pose1, c1, pose2, c2, intr)
                if size is None:
                    continue
                errors += [percent_error(size["w_mm"], obj["w_mm"]),
                           percent_error(size["h_mm"], obj["h_mm"])]
        elapsed = (time.perf_counter() - start) * 1000.0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        mape = sum(errors) / len(errors) if errors else 0.0
        results.append({
            "method": name, "mape": round(mape, 4), "coverage": 100.0,
            "runtime_ms": round(elapsed, 2), "peak_mem_mb": round(peak, 3),
            "score": round(_score(mape, 100.0, elapsed), 4),
        })
    return sorted(results, key=lambda r: r["score"])


# ── orchestration + snip ───────────────────────────────────────────────
def run_arena(quick=False, top_k=3, include_slow=False):
    fixtures = PREMADE[:3] if quick else PREMADE
    depths = (2.0, 3.0) if quick else (1.0, 2.0, 3.0, 5.0)
    noises = (0.0, 0.02) if quick else (0.0, 0.02, 0.05)
    fusers = evaluate_fusers(fixtures, depths, noises)
    sizes = evaluate_sizes(fixtures, depths, noises, include_slow=include_slow)
    illusions = evaluate_illusions(fixtures, depths, noises)
    tris = evaluate_triangulators()
    report = {"fusers": fusers, "sizes": sizes,
              "illusions": illusions, "triangulators": tris}
    cut = snip(report, top_k=top_k)
    report["survivors"] = cut["keep"]
    report["pruned"] = cut["prune"]
    return report


BASELINES = {
    "fusers": "inverse_variance",
    "sizes": "pixel_scale",
    "illusions": "none",
    "triangulators": "midpoint",
}


def snip(report, top_k=3, mape_slack=0.01, runtime_factor=5.0):
    """Keep the accuracy leaders, the top-K by score, and the baseline.

    Prune anything more than 1 MAPE point worse than the baseline, and anything
    more than 5x slower than the baseline unless it is a top-K scorer.
    """
    keep, prune = {}, {}
    for stage, baseline in BASELINES.items():
        rows = report.get(stage) or []
        if not rows:
            continue
        best_mape = min(r["mape"] for r in rows)
        base = next((r for r in rows if r["method"] == baseline), None)
        base_mape = base["mape"] if base else best_mape
        base_runtime = base["runtime_ms"] if base else 0.0
        top_names = {r["method"] for r in rows[:top_k]}

        keep_set = set(top_names)
        if base:
            keep_set.add(baseline)
        for r in rows:
            if r["mape"] <= best_mape + mape_slack:
                keep_set.add(r["method"])

        for r in rows:
            if r["method"] not in keep_set:
                continue
            much_worse = r["mape"] > base_mape + 1.0
            too_slow = (base_runtime > 0 and r["runtime_ms"] > base_runtime * runtime_factor
                        and r["method"] != baseline)
            if much_worse or too_slow:
                keep_set.discard(r["method"])

        keep[stage] = [r["method"] for r in rows if r["method"] in keep_set]
        prune[stage] = [r["method"] for r in rows if r["method"] not in keep_set]
    return {"keep": keep, "prune": prune}


def _print_stage(name, rows, limit=8):
    print("\n== %s ==" % name)
    print("  %-20s %8s %9s %10s %10s" % ("method", "MAPE%", "cover%", "runtime_ms", "score"))
    for r in rows[:limit]:
        print("  %-20s %8.3f %9.1f %10.2f %10.3f"
              % (r["method"], r["mape"], r["coverage"], r["runtime_ms"], r["score"]))
    if len(rows) > limit:
        print("  ... %d more" % (len(rows) - limit))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="lidarv2 algorithm arena")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--out", default="arena_report.json")
    parser.add_argument("--include-slow", action="store_true")
    args = parser.parse_args(argv)

    report = run_arena(quick=args.quick, top_k=args.top, include_slow=args.include_slow)
    _print_stage("depth fusers", report["fusers"])
    _print_stage("size estimators", report["sizes"])
    _print_stage("illusion models", report["illusions"])
    _print_stage("triangulators", report["triangulators"])
    print("\nSURVIVORS:", json.dumps(report["survivors"], indent=2))
    print("PRUNED:", json.dumps(report["pruned"], indent=2))
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    surv_path = args.out.replace(".json", "_survivors.json")
    with open(surv_path, "w") as fh:
        json.dump({"keep": report["survivors"], "prune": report["pruned"]}, fh, indent=2)
    print("\nwrote", args.out, "and", surv_path)
    return report


if __name__ == "__main__":
    main()
