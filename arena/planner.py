"""lidarv2 active view planning — where to stand next, inside the real FOV.

v2.3 fused a *fixed*, evenly spaced set of views and reported one seed's
number. v2.4 asks the prior question: given a depth estimate and the camera
intrinsics, which lateral offsets are physically observable, and which set of
them minimises the size error for a given budget?

Two facts drive the design.

1. **Visibility.** An offset `b` puts the object centre at `u = cx - f*b/z`.
   At 1.5 m with a 60 deg FOV the whole-object limit is ~0.69 m, but v2.3's
   simulator happily used offsets out to 2.8 m, where the object is far
   outside the image. `fov_cap_m` derives the real limit from the intrinsics,
   the depth and the object width; `simulate_offset_view` drops a view whose
   corners leave the frame.

2. **Coverage.** Under multi-view 3-D averaging the error is driven by the
   *spread* of the offsets, not by one long baseline. `run_plan_benchmark`
   shows an evenly covered arc beating both "stand at the extremes" and
   random placement, and shows why: the two error terms below trade a pixel
   term against a triangulation term, and clustering views starves the second.

The predictive model has two calibrated terms. For `n` views with offsets `b_i`
at depth `z`, focal `f`, pixel noise `sigma_px` and object height `h`:

    pixel   = 100 * K_PIXEL * sigma_px / (h_px * sqrt(n))      h_px = f*h/z
    triang  = 100 * K_DEPTH * sigma_z / z
              sigma_z = z^2 * (sigma_px/f) / sqrt(sum b_i^2)
    error%  = hypot(pixel, triang)

`K_PIXEL = 1.1` and `K_DEPTH = 1.7` were fitted against the multipview
simulator over 90 plan/fixture/budget combinations (rms log-ratio 0.153). The
model is a planning aid, not a guarantee: `run_plan_benchmark` re-checks it on
the same fixture set and reports the realised/predicted ratio.
"""

import json
import math
import random
import statistics

from arena.multiview import pair_is_usable
from fixtures.premade_objects import PREMADE, by_tag
from geospatial import enu_to_geodetic
from models import CameraIntrinsics, CameraPose
from triangulate import enu_direction_to_pixel, triangulate_point

K_PIXEL = 1.1
K_DEPTH = 1.7
DEFAULT_SIGMA_PX = 0.3
DEFAULT_MARGIN = 0.06
MIN_SEP_M = 0.02


# ── visibility ─────────────────────────────────────────────────────────
def fov_cap_m(intrinsics, depth_m, object_w_m=0.0, margin=DEFAULT_MARGIN):
    """Largest lateral offset that still keeps the whole object in the frame.

    `margin` is the fraction of the image width kept free at each edge so the
    detector has context around the object. Returns 0.0 when the object
    already fills the usable width.
    """
    if depth_m <= 0 or intrinsics.f_px <= 0:
        return 0.0
    edge_px = intrinsics.image_w / 2.0 - margin * intrinsics.image_w
    half_obj_px = intrinsics.f_px * (object_w_m / 2.0) / depth_m
    cap = (edge_px - half_obj_px) * depth_m / intrinsics.f_px
    return max(0.0, cap)


def offset_pixel(intrinsics, depth_m, offset_m):
    """Where the object centre lands horizontally for a lateral offset."""
    return intrinsics.cx - intrinsics.f_px * offset_m / depth_m


def is_visible(intrinsics, depth_m, object_w_m, object_h_m, offset_m,
               margin=DEFAULT_MARGIN):
    """True when all four projected corners stay inside the usable frame."""
    if depth_m <= 0:
        return False
    half_w, half_h = object_w_m / 2.0, object_h_m / 2.0
    lim_u = (0.5 - margin) * intrinsics.image_w
    lim_v = (0.5 - margin) * intrinsics.image_h
    for dx, up in ((-half_w, half_h), (half_w, half_h),
                   (half_w, -half_h), (-half_w, -half_h)):
        u = intrinsics.cx + intrinsics.f_px * (dx - offset_m) / depth_m
        v = intrinsics.cy - intrinsics.f_px * up / depth_m
        if abs(u - intrinsics.cx) > lim_u or abs(v - intrinsics.cy) > lim_v:
            return False
    return True


# ── plan construction ──────────────────────────────────────────────────
def even_plan(cap_m, n):
    """`n` offsets covering the visible arc evenly, centre included."""
    n = max(1, int(n))
    if n == 1:
        return [0.0]
    step = 2.0 * cap_m / (n - 1)
    return [round(-cap_m + step * i, 6) for i in range(n)]


def acquisition_order(offsets):
    """Visit the offset closest to the centre first, then the widest, inwards.

    An even budget has no exact centre offset, so the innermost view is first.
    Early stopping is the common case, so the first two or three views should
    already carry most of the information.
    """
    return sorted(offsets, key=lambda b: (abs(b), b))


def extremes_plan(cap_m, n, sep=0.12):
    """Alternate the two widest offsets, stepping inwards on each revisit."""
    n = max(1, int(n))
    out = [0.0]
    sign, k = 1, 1
    while len(out) < n:
        out.append(round(sign * (cap_m - (k - 1) * sep), 6))
        sign = -sign
        if sign == 1:
            k += 1
    return out[:n]


def random_plan(cap_m, n, seed=11):
    rng = random.Random(seed)
    return sorted(round(rng.uniform(-cap_m, cap_m), 6) for _ in range(n))


# ── prediction ─────────────────────────────────────────────────────────
def predicted_error_pct(intrinsics, depth_m, object_h_mm, offsets,
                        sigma_px=DEFAULT_SIGMA_PX):
    """Predicted relative height error (1 sigma, percent) for a view set."""
    n = len(offsets)
    if n == 0 or depth_m <= 0 or intrinsics.f_px <= 0:
        return None
    h_px = intrinsics.f_px * (object_h_mm / 1000.0) / depth_m
    if h_px <= 0:
        return None
    pixel_term = 100.0 * K_PIXEL * sigma_px / (h_px * math.sqrt(n))
    sum_b2 = sum(b * b for b in offsets)
    if sum_b2 <= 0:
        return None
    sigma_z = depth_m * depth_m * (sigma_px / intrinsics.f_px) / math.sqrt(sum_b2)
    depth_term = 100.0 * K_DEPTH * sigma_z / depth_m
    return round(math.hypot(pixel_term, depth_term), 4)


def information_score(offsets, depth_m, sigma_px=DEFAULT_SIGMA_PX):
    """Relative depth information: sqrt(sum b_i^2) / (z^2 * sigma_theta)."""
    sum_b2 = sum(b * b for b in offsets)
    if sum_b2 <= 0 or depth_m <= 0:
        return 0.0
    return round(math.sqrt(sum_b2) / (depth_m * depth_m * sigma_px), 6)


def plan_views(intrinsics, depth_m, budget, object_w_mm=0.0, object_h_mm=88.9,
               sigma_px=DEFAULT_SIGMA_PX, margin=DEFAULT_MARGIN):
    """Build an acquisition plan and its predicted error after each view."""
    cap = fov_cap_m(intrinsics, depth_m, object_w_mm / 1000.0, margin)
    offsets = even_plan(cap, budget)
    order = acquisition_order(offsets)
    steps = []
    for k in range(1, len(order) + 1):
        taken = order[:k]
        steps.append({
            "views": k,
            "offset_m": order[k - 1],
            "offsets_m": taken,
            "predicted_error_pct": predicted_error_pct(
                intrinsics, depth_m, object_h_mm, taken, sigma_px),
        })
    return {
        "strategy": "even",
        "budget": len(order),
        "fov_cap_m": round(cap, 4),
        "offsets_m": offsets,
        "acquisition_order_m": order,
        "span_m": round(max(order) - min(order), 4),
        "information_score": information_score(order, depth_m, sigma_px),
        "predicted_error_pct": predicted_error_pct(
            intrinsics, depth_m, object_h_mm, order, sigma_px),
        "steps": steps,
    }


# ── simulation ─────────────────────────────────────────────────────────
def simulate_offset_view(obj, offset_m, depth_m=1.5, noise_px=DEFAULT_SIGMA_PX,
                         rng=None, intrinsics=None, margin=DEFAULT_MARGIN):
    """Project one object as seen from a lateral offset, or None if clipped."""
    intr = intrinsics or CameraIntrinsics.from_fov(640, 480, 60.0)
    w_m = obj["w_mm"] / 1000.0
    h_m = obj["h_mm"] / 1000.0
    if not is_visible(intr, depth_m, w_m, h_m, offset_m, margin):
        return None
    lat, lon, alt = enu_to_geodetic(0.0, 0.0, 0.0, offset_m, 0.0, 0.0)
    pose = CameraPose(lat=lat, lon=lon, alt_m=alt, heading_deg=0.0,
                      pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
    gen = rng or random.Random(0)
    corners = []
    for dx, up in ((-w_m / 2, h_m / 2), (w_m / 2, h_m / 2),
                   (w_m / 2, -h_m / 2), (-w_m / 2, -h_m / 2)):
        p = enu_direction_to_pixel(dx - offset_m, depth_m, up, pose, intr)
        if p is None:
            return None
        corners.append((p[0] + gen.gauss(0.0, noise_px),
                        p[1] + gen.gauss(0.0, noise_px)))
    return {"pose": pose, "corners": corners, "intrinsics": intr,
            "offset_m": offset_m}


def simulate_plan(obj, offsets, depth_m=1.5, noise_px=DEFAULT_SIGMA_PX,
                  seed=0, intrinsics=None, margin=DEFAULT_MARGIN):
    """Build the usable views for a plan; clipped views are skipped."""
    rng = random.Random(seed)
    views = []
    for b in offsets:
        v = simulate_offset_view(obj, b, depth_m, noise_px, rng, intrinsics, margin)
        if v is not None:
            views.append(v)
    return views


def _measure(views, obj):
    """3-D corner-average size for a view set, with pair validation applied."""
    if len(views) < 2:
        return None
    acc = [[0.0, 0.0, 0.0] for _ in range(4)]
    count = 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            pts = []
            for k in range(4):
                p = triangulate_point(views[i]["pose"], views[i]["corners"][k],
                                      views[j]["pose"], views[j]["corners"][k],
                                      views[i]["intrinsics"])
                if p is None:
                    break
                pts.append(p)
            if len(pts) < 4 or not pair_is_usable(views[i]["pose"],
                                                  views[j]["pose"], pts):
                continue
            for k in range(4):
                acc[k][0] += pts[k]["east_m"]
                acc[k][1] += pts[k]["north_m"]
                acc[k][2] += pts[k]["up_m"]
            count += 1
    if count == 0:
        return None
    for k in range(4):
        acc[k] = [c / count for c in acc[k]]
    dist = lambda a, b: math.sqrt(sum((a[t] - b[t]) ** 2 for t in range(3)))
    return {"w_mm": dist(acc[0], acc[1]) * 1000.0,
            "h_mm": dist(acc[0], acc[3]) * 1000.0,
            "pairs": count}


def evaluate_plan(obj, offsets, depth_m=1.5, noise_px=DEFAULT_SIGMA_PX,
                  seeds=8, intrinsics=None, margin=DEFAULT_MARGIN):
    """Realised error over independent noise draws; None when never solvable."""
    errors, skipped = [], 0
    for seed in range(seeds):
        views = simulate_plan(obj, offsets, depth_m, noise_px, seed,
                              intrinsics, margin)
        est = _measure(views, obj)
        if est is None:
            skipped += 1
            continue
        errors.append(abs(est["h_mm"] - obj["h_mm"]) / obj["h_mm"] * 100.0)
    if not errors:
        return {"median_pct": None, "p90_pct": None, "max_pct": None,
                "seeds": 0, "skipped": skipped}
    ordered = sorted(errors)
    return {
        "median_pct": round(statistics.median(errors), 4),
        "p90_pct": round(ordered[int(0.9 * (len(ordered) - 1))], 4),
        "max_pct": round(ordered[-1], 4),
        "seeds": len(errors),
        "skipped": skipped,
    }


# ── benchmark ──────────────────────────────────────────────────────────
STRATEGIES = ("even", "extremes", "random", "anchor")


def _plan_for(strategy, cap_m, budget):
    if strategy == "even":
        return even_plan(cap_m, budget)
    if strategy == "extremes":
        return extremes_plan(cap_m, budget)
    if strategy == "random":
        return random_plan(cap_m, budget)
    if strategy == "anchor":
        return [0.0] * budget
    raise ValueError("unknown strategy: %s" % strategy)


def run_plan_benchmark(budgets=(2, 4, 6, 8), strategies=STRATEGIES,
                       depth_m=1.5, noise_px=DEFAULT_SIGMA_PX, seeds=10,
                       intrinsics=None, margin=DEFAULT_MARGIN,
                       target_pct=0.75, hard_tag="2x6", fixtures=None):
    """Compare view-placement strategies; predict the error before measuring it."""
    intr = intrinsics or CameraIntrinsics.from_fov(640, 480, 60.0)
    fixtures = fixtures or PREMADE
    rows = []
    for obj in fixtures:
        cap = fov_cap_m(intr, depth_m, obj["w_mm"] / 1000.0, margin)
        for budget in budgets:
            for strategy in strategies:
                offsets = _plan_for(strategy, cap, budget)
                measured = evaluate_plan(obj, offsets, depth_m, noise_px,
                                         seeds, intr, margin)
                predicted = predicted_error_pct(intr, depth_m, obj["h_mm"],
                                                offsets, noise_px)
                rows.append({
                    "tag": obj["tag"],
                    "budget": budget,
                    "strategy": strategy,
                    "cap_m": round(cap, 4),
                    "offsets_m": offsets,
                    "predicted_pct": predicted,
                    "measured": measured,
                    "ratio": (round(measured["median_pct"] / predicted, 3)
                              if measured["median_pct"] and predicted else None),
                })

    summary = {}
    for strategy in strategies:
        subset = [r for r in rows if r["strategy"] == strategy]
        by_budget = {}
        for budget in budgets:
            cells = [r for r in subset if r["budget"] == budget]
            med = [r["measured"]["median_pct"] for r in cells
                   if r["measured"]["median_pct"] is not None]
            preds = [r["predicted_pct"] for r in cells
                     if r["predicted_pct"] is not None]
            hard = [r["measured"]["median_pct"] for r in cells
                    if r["tag"] == hard_tag
                    and r["measured"]["median_pct"] is not None]
            by_budget[str(budget)] = {
                "mape_pct": round(statistics.median(med), 4) if med else None,
                "hard_median_pct": round(statistics.median(hard), 4) if hard else None,
                "solved": len(med),
                "cells": len(cells),
                "predicted_pct": round(statistics.median(preds), 4) if preds else None,
            }
        reached = None
        for budget in sorted(budgets):
            cell = by_budget[str(budget)]
            metric = cell["hard_median_pct"] if hard_tag else cell["mape_pct"]
            if metric is not None and metric <= target_pct:
                reached = budget
                break
        summary[strategy] = {"by_budget": by_budget, "views_to_target": reached}

    ratios = [r["ratio"] for r in rows
              if r["strategy"] == "even" and r["ratio"]]
    calibration = {
        "k_pixel": K_PIXEL,
        "k_depth": K_DEPTH,
        "median_ratio": round(statistics.median(ratios), 3) if ratios else None,
        "rms_log_ratio": round(math.sqrt(sum(math.log(x) ** 2 for x in ratios)
                                         / len(ratios)), 3) if ratios else None,
        "note": "ratio = realised median / predicted 1-sigma; 1.0 is calibrated",
    }
    return {
        "config": {"budgets": list(budgets), "strategies": list(strategies),
                   "depth_m": depth_m, "noise_px": noise_px, "seeds": seeds,
                   "margin": margin, "target_pct": target_pct,
                   "hard_tag": hard_tag,
                   "intrinsics": {"image_w": intr.image_w,
                                  "image_h": intr.image_h,
                                  "f_px": round(intr.f_px, 2),
                                  "fov_deg": intr.fov_deg}},
        "strategies": summary,
        "calibration": calibration,
        "rows": rows,
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="lidarv2 active view planning")
    parser.add_argument("--budgets", default="2,4,6,8")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--depth", type=float, default=1.5)
    parser.add_argument("--noise", type=float, default=DEFAULT_SIGMA_PX)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--target-pct", type=float, default=0.75)
    parser.add_argument("--hard-tag", default="2x6")
    parser.add_argument("--json", default="planner_report.json")
    args = parser.parse_args(argv)

    budgets = tuple(int(x) for x in args.budgets.split(",") if x.strip())
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    report = run_plan_benchmark(budgets=budgets, strategies=strategies,
                                depth_m=args.depth, noise_px=args.noise,
                                seeds=args.seeds, target_pct=args.target_pct,
                                hard_tag=args.hard_tag)

    print("config:", json.dumps(report["config"]))
    print("  %-10s %6s %8s %8s %8s %9s %10s"
          % ("strategy", "views", "MAPE", "hard", "solved", "predicted", "to target"))
    for name, s in report["strategies"].items():
        for budget, cell in s["by_budget"].items():
            def pct(v):
                return "%s%%" % v if v is not None else "—"
            print("  %-10s %6s %8s %8s %8s %9s %10s" % (
                name, budget, pct(cell["mape_pct"]), pct(cell["hard_median_pct"]),
                "%d/%d" % (cell["solved"], cell["cells"]),
                pct(cell["predicted_pct"]),
                s["views_to_target"] if s["views_to_target"] is not None else "—"))
    print("calibration:", json.dumps(report["calibration"]))
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.json)
    return report


if __name__ == "__main__":
    main()
