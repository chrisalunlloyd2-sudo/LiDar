"""lidarv2 multi-view refinement — drive the 2x6 height error below 1%.

Simulate N slightly offset views of the same premade object, then compare:

  temporal      median of per-frame pixel-scale sizes (noise averaging)
  triangulated  median of pairwise RANSAC triangulations (the surviving method)

The 2x6's 38.1 mm height is only a few pixels at distance, so this is the hard
case the single-view benchmark could not beat.
"""

import json
import math
import random
from statistics import median

from accuracy import percent_error
from arena.triangulators import TRIANGULATORS
from fixtures.premade_objects import PREMADE
from geospatial import enu_to_geodetic
from models import CameraIntrinsics, CameraPose
from triangulate import (
    enu_direction_to_pixel,
    geodetic_to_enu,
    triangulate_point,
)

MIN_PAIR_BASELINE_M = 0.02
MAX_PAIR_RESIDUAL_M = 0.05
DEPTH_PLAUSIBLE_M = (0.2, 20.0)


def simulate_views(obj, depth_m=1.5, n_views=9, baseline_m=0.2, noise_px=0.3,
                   seed=0, intrinsics=None):
    """Project the object's four corners from N offset camera poses."""
    intr = intrinsics or CameraIntrinsics.from_fov(640, 480, 60.0)
    rng = random.Random(seed)
    w = obj["w_mm"] / 1000.0
    h = obj["h_mm"] / 1000.0
    views = []
    for i in range(max(2, n_views)):
        offset = i * baseline_m
        lat, lon, alt = enu_to_geodetic(0.0, 0.0, 0.0, offset, 0.0, 0.0)
        pose = CameraPose(lat=lat, lon=lon, alt_m=alt, heading_deg=0.0,
                          pitch_deg=0.0, roll_deg=0.0, accuracy_m=1.0, has_gps=True)
        corners = [(-w / 2, depth_m, h / 2), (w / 2, depth_m, h / 2),
                   (w / 2, depth_m, -h / 2), (-w / 2, depth_m, -h / 2)]
        pix = []
        for c in corners:
            p = enu_direction_to_pixel(c[0] - offset, c[1], c[2], pose, intr)
            if p is None:
                return []
            pix.append((p[0] + rng.gauss(0.0, noise_px),
                        p[1] + rng.gauss(0.0, noise_px)))
        views.append({"pose": pose, "corners": pix, "intrinsics": intr,
                      "offset_m": offset})
    return views


def refine_temporal(views, obj_w_mm, depth_m):
    """Median of per-view pixel-scale sizes."""
    ws, hs = [], []
    for v in views:
        us = [c[0] for c in v["corners"]]
        vs = [c[1] for c in v["corners"]]
        px_w = max(us) - min(us)
        px_h = max(vs) - min(vs)
        if px_w <= 0:
            continue
        ws.append(obj_w_mm)
        hs.append(obj_w_mm * px_h / px_w)
    if not hs:
        return None
    return {"w_mm": median(ws), "h_mm": median(hs), "n": len(hs), "method": "temporal"}


def refine_aligned(views, obj_w_mm, depth_m):
    """Average each corner's offset across views, then measure.

    Taking max-min of four noisy corners inflates the extent (an order-statistic
    bias that does not average out). Aligning corners to their centroid first,
    then averaging each corner across views, removes both the bias and the noise.
    """
    n = len(views[0]["corners"])
    acc = [[0.0, 0.0] for _ in range(n)]
    for v in views:
        cu = sum(c[0] for c in v["corners"]) / n
        cv = sum(c[1] for c in v["corners"]) / n
        for k, c in enumerate(v["corners"]):
            acc[k][0] += c[0] - cu
            acc[k][1] += c[1] - cv
    for k in range(n):
        acc[k][0] /= len(views)
        acc[k][1] /= len(views)
    us = [a[0] for a in acc]
    vs = [a[1] for a in acc]
    px_w = max(us) - min(us)
    px_h = max(vs) - min(vs)
    if px_w <= 0:
        return None
    return {"w_mm": obj_w_mm, "h_mm": obj_w_mm * px_h / px_w,
            "n": len(views), "method": "aligned"}


def pair_is_usable(pose_a, pose_b, pts):
    """Reject a view pair before it can poison the 3-D average.

    Two views taken from almost the same spot give a near-singular ray
    intersection: `denom` is tiny but not zero, so the midpoint lands far away
    and a single pair can drag the average and collapse the measured size. A
    pair is usable only when the baseline is real, every corner triangulates
    inside the plausible depth range, the rays actually meet, and both views
    agree on each corner's depth.
    """
    dx, dy, dz = geodetic_to_enu(pose_a.lat, pose_a.lon, pose_a.alt_m,
                                 pose_b.lat, pose_b.lon, pose_b.alt_m)
    if math.sqrt(dx * dx + dy * dy + dz * dz) < MIN_PAIR_BASELINE_M:
        return False
    lo, hi = DEPTH_PLAUSIBLE_M
    for p in pts:
        if p is None or p["residual_m"] > MAX_PAIR_RESIDUAL_M:
            return False
        d1, d2 = p["depth1_m"], p["depth2_m"]
        if not (lo <= d1 <= hi and lo <= d2 <= hi):
            return False
        if abs(d1 - d2) > 0.2 * (d1 + d2) / 2.0:
            return False
    return True


def refine_3d(views, method="ransac"):
    """Average each corner's 3-D position across all view pairs, then measure."""
    n = 4
    acc = [[0.0, 0.0, 0.0] for _ in range(n)]
    count = 0
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            pts = []
            for k in range(n):
                p = triangulate_point(views[i]["pose"], views[i]["corners"][k],
                                      views[j]["pose"], views[j]["corners"][k],
                                      views[i]["intrinsics"])
                if p is None:
                    break
                pts.append(p)
            if len(pts) < n:
                continue
            if not pair_is_usable(views[i]["pose"], views[j]["pose"], pts):
                continue
            for k in range(n):
                acc[k][0] += pts[k]["east_m"]
                acc[k][1] += pts[k]["north_m"]
                acc[k][2] += pts[k]["up_m"]
            count += 1
    if count == 0:
        return None
    for k in range(n):
        acc[k] = [c / count for c in acc[k]]

    def dist(a, b):
        return math.sqrt(sum((a[t] - b[t]) ** 2 for t in range(3)))

    return {
        "w_mm": dist(acc[0], acc[1]) * 1000.0,
        "h_mm": dist(acc[0], acc[3]) * 1000.0,
        "n": count,
        "method": "3d_" + method,
    }


def refine_triangulated(views, method="ransac"):
    """Median of pairwise triangulated sizes."""
    fn = TRIANGULATORS[method]
    sizes = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            size = fn(views[i]["pose"], views[i]["corners"],
                      views[j]["pose"], views[j]["corners"],
                      views[i]["intrinsics"])
            if size is not None:
                sizes.append(size)
    if not sizes:
        return None
    ws = sorted(s["w_mm"] for s in sizes)
    hs = sorted(s["h_mm"] for s in sizes)
    return {
        "w_mm": median(ws), "h_mm": median(hs), "n": len(sizes),
        "method": method, "w_spread": ws[-1] - ws[0], "h_spread": hs[-1] - hs[0],
    }


def _errors(est, obj):
    return {"w": round(percent_error(est["w_mm"], obj["w_mm"]), 3),
            "h": round(percent_error(est["h_mm"], obj["h_mm"]), 3)}


def run_multiview_benchmark(n_views=9, baseline_m=0.2, noise_px=0.3,
                            depth_m=1.5, method="ransac", fixtures=None,
                            seed=0):
    """One noise draw. Seed 0 reproduces v2.3; sweep seeds for a median."""
    results = []
    temporal_errors, triangulated_errors, aligned_errors, three_d_errors = [], [], [], []
    for obj in (fixtures or PREMADE):
        views = simulate_views(obj, depth_m=depth_m, n_views=n_views,
                               baseline_m=baseline_m, noise_px=noise_px,
                               seed=seed)
        if not views:
            continue
        temporal = refine_temporal(views, obj["w_mm"], depth_m)
        aligned = refine_aligned(views, obj["w_mm"], depth_m)
        three_d = refine_3d(views, method)
        triangulated = refine_triangulated(views, method)
        te = _errors(temporal, obj) if temporal else None
        ae = _errors(aligned, obj) if aligned else None
        tde = _errors(three_d, obj) if three_d else None
        tre = _errors(triangulated, obj) if triangulated else None
        if te:
            temporal_errors += [te["w"], te["h"]]
        if ae:
            aligned_errors += [ae["w"], ae["h"]]
        if tde:
            three_d_errors += [tde["w"], tde["h"]]
        if tre:
            triangulated_errors += [tre["w"], tre["h"]]
        results.append({
            "tag": obj["tag"],
            "truth_mm": {"w": obj["w_mm"], "h": obj["h_mm"]},
            "temporal_mm": {"w": round(temporal["w_mm"], 2),
                            "h": round(temporal["h_mm"], 2)} if temporal else None,
            "aligned_mm": {"w": round(aligned["w_mm"], 2),
                           "h": round(aligned["h_mm"], 2)} if aligned else None,
            "three_d_mm": {"w": round(three_d["w_mm"], 2),
                           "h": round(three_d["h_mm"], 2)} if three_d else None,
            "triangulated_mm": {"w": round(triangulated["w_mm"], 2),
                                "h": round(triangulated["h_mm"], 2)} if triangulated else None,
            "temporal_error_pct": te,
            "aligned_error_pct": ae,
            "three_d_error_pct": tde,
            "triangulated_error_pct": tre,
            "views": len(views),
            "pairs": triangulated["n"] if triangulated else 0,
        })
    return {
        "config": {"n_views": n_views, "baseline_m": baseline_m,
                   "noise_px": noise_px, "depth_m": depth_m, "method": method,
                   "seed": seed},
        "results": results,
        "temporal_mape_pct": round(sum(temporal_errors) / len(temporal_errors), 3)
        if temporal_errors else None,
        "aligned_mape_pct": round(sum(aligned_errors) / len(aligned_errors), 3)
        if aligned_errors else None,
        "three_d_mape_pct": round(sum(three_d_errors) / len(three_d_errors), 3)
        if three_d_errors else None,
        "triangulated_mape_pct": round(sum(triangulated_errors) / len(triangulated_errors), 3)
        if triangulated_errors else None,
        "max_temporal_error_pct": round(max(temporal_errors), 3) if temporal_errors else None,
        "max_aligned_error_pct": round(max(aligned_errors), 3) if aligned_errors else None,
        "max_three_d_error_pct": round(max(three_d_errors), 3) if three_d_errors else None,
        "max_triangulated_error_pct": round(max(triangulated_errors), 3)
        if triangulated_errors else None,
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="lidarv2 multi-view refinement")
    parser.add_argument("--views", type=int, default=9)
    parser.add_argument("--baseline", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--depth", type=float, default=1.5)
    parser.add_argument("--method", default="ransac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default="multiview_report.json")
    args = parser.parse_args(argv)

    report = run_multiview_benchmark(
        n_views=args.views, baseline_m=args.baseline, noise_px=args.noise,
        depth_m=args.depth, method=args.method, seed=args.seed)
    print("config:", json.dumps(report["config"]))
    print("  %-12s %14s %14s %14s %14s %14s"
          % ("tag", "truth mm", "temporal", "aligned", "3d-avg", "triangulated"))
    for r in report["results"]:
        t = r["temporal_mm"] or {}
        a = r["aligned_mm"] or {}
        d = r["three_d_mm"] or {}
        tri = r["triangulated_mm"] or {}
        print("  %-12s %14s %14s %14s %14s %14s" % (
            r["tag"],
            "%sx%s" % (r["truth_mm"]["w"], r["truth_mm"]["h"]),
            "%sx%s" % (t.get("w", "?"), t.get("h", "?")),
            "%sx%s" % (a.get("w", "?"), a.get("h", "?")),
            "%sx%s" % (d.get("w", "?"), d.get("h", "?")),
            "%sx%s" % (tri.get("w", "?"), tri.get("h", "?")),
        ))
    print("temporal MAPE     :", report["temporal_mape_pct"], "%")
    print("aligned MAPE      :", report["aligned_mape_pct"], "%")
    print("3d-avg MAPE       :", report["three_d_mape_pct"], "%")
    print("triangulated MAPE :", report["triangulated_mape_pct"], "%")
    for r in report["results"]:
        if r["tag"] == "2x6":
            print("2x6 height error  : temporal", r["temporal_error_pct"]["h"],
                  "% | aligned", r["aligned_error_pct"]["h"],
                  "% | 3d-avg", r["three_d_error_pct"]["h"],
                  "% | triangulated", r["triangulated_error_pct"]["h"], "%")
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.json)
    return report


if __name__ == "__main__":
    main()
