"""lidarv2 accuracy — the "% to reality" number we want to minimise.

`run_benchmark` projects each premade object at a chosen depth, runs the real
depth + illusion-correction pipeline, and reports per-dimension percent error,
mean absolute percentage error (MAPE), and confidence-interval coverage.
"""

from depth_engine import DepthEngine
from fixtures.premade_objects import PREMADE, synthesize, truth_dims
from illusions import correct_size, evaluate
from inventory import emulated_dimensions_mm
from shapes import classify_shape


def percent_error(emulated, truth):
    if truth == 0:
        return float("inf")
    return abs(float(emulated) - float(truth)) / abs(float(truth)) * 100.0


def mape(pairs):
    errs = [percent_error(e, t) for e, t in pairs if t]
    return sum(errs) / len(errs) if errs else 0.0


def coverage_pct(intervals, truths):
    if not intervals:
        return 0.0
    hits = 0
    for (lo, hi), truth in zip(intervals, truths):
        if lo <= truth <= hi:
            hits += 1
    return 100.0 * hits / len(intervals)


def run_benchmark(fixtures=None, depth_m=2.0, noise_frac=0.0, level=0.95,
                  intrinsics=None):
    engine = DepthEngine()
    results = []
    for i, obj in enumerate(fixtures or PREMADE):
        obs = synthesize(obj, depth_m, intrinsics=intrinsics,
                         noise_frac=noise_frac, seed=i)
        cues = engine.cues(obs)
        est = engine.estimate(obs, level=level)
        raw = emulated_dimensions_mm(obs, est.depth_m)
        ev = evaluate(obs, cues)
        factor = ev["combined_factor"]
        corrected = correct_size(raw, factor)

        raw_lo = emulated_dimensions_mm(obs, est.ci_low_m)
        raw_hi = emulated_dimensions_mm(obs, est.ci_high_m)
        ci_lo = correct_size(raw_lo, factor)
        ci_hi = correct_size(raw_hi, factor)

        truth = truth_dims(obj)
        results.append({
            "tag": obj["tag"],
            "label": obj["label"],
            "truth_mm": {"w": truth[0], "h": truth[1], "d": truth[2]},
            "raw_mm": {"w": raw[0], "h": raw[1], "d": raw[2]},
            "corrected_mm": {"w": corrected[0], "h": corrected[1], "d": corrected[2]},
            "ci_mm": {
                "w": [ci_lo[0], ci_hi[0]],
                "h": [ci_lo[1], ci_hi[1]],
            },
            "error_pct": {
                "w": round(percent_error(corrected[0], truth[0]), 3),
                "h": round(percent_error(corrected[1], truth[1]), 3),
                "d": round(percent_error(corrected[2], truth[2]), 3)
                if corrected[2] is not None else None,
            },
            "depth_m": round(est.depth_m, 4),
            "depth_ci_m": [round(est.ci_low_m, 4), round(est.ci_high_m, 4)],
            "confidence": round(est.confidence, 4),
            "correction_pct": ev["correction_percent"],
            "shape": classify_shape(corrected[0], corrected[1], corrected[2]),
        })

    w_pairs = [(r["corrected_mm"]["w"], r["truth_mm"]["w"]) for r in results]
    h_pairs = [(r["corrected_mm"]["h"], r["truth_mm"]["h"]) for r in results]
    all_pairs = w_pairs + h_pairs
    all_errors = [percent_error(e, t) for e, t in all_pairs]

    w_ci = [(r["ci_mm"]["w"][0], r["ci_mm"]["w"][1]) for r in results]
    h_ci = [(r["ci_mm"]["h"][0], r["ci_mm"]["h"][1]) for r in results]
    w_truth = [r["truth_mm"]["w"] for r in results]
    h_truth = [r["truth_mm"]["h"] for r in results]

    return {
        "depth_m": depth_m,
        "noise_frac": noise_frac,
        "level": level,
        "results": results,
        "mape_w_pct": round(mape(w_pairs), 3),
        "mape_h_pct": round(mape(h_pairs), 3),
        "mean_mape_pct": round(mape(all_pairs), 3),
        "max_error_pct": round(max(all_errors), 3) if all_errors else 0.0,
        "coverage_w_pct": round(coverage_pct(w_ci, w_truth), 2),
        "coverage_h_pct": round(coverage_pct(h_ci, h_truth), 2),
    }
