"""lidarv2 triangulators — five two-view size strategies to compete in the arena."""

import math
from statistics import median

from fit_engine import nelder_mead
from geospatial import pixel_to_ray, ray_camera_to_enu
from triangulate import (
    enu_direction_to_pixel,
    geodetic_to_enu,
    triangulate_point,
    triangulate_size,
)


def _world_ray(pose, u, v, intr):
    return ray_camera_to_enu(pixel_to_ray(u, v, intr),
                             pose.heading_deg, pose.pitch_deg, pose.roll_deg)


def _norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist(p, q):
    return math.sqrt((p["east_m"] - q["east_m"]) ** 2
                     + (p["north_m"] - q["north_m"]) ** 2
                     + (p["up_m"] - q["up_m"]) ** 2)


def _robust_size(pts):
    tl, tr, br, bl = pts
    widths = [_dist(tl, tr), _dist(bl, br)]
    heights = [_dist(tl, bl), _dist(tr, br)]
    return (median(widths) * 1000.0, median(heights) * 1000.0)


def _pack(pts, w_mm, h_mm, residual):
    return {
        "w_mm": round(w_mm, 2),
        "h_mm": round(h_mm, 2),
        "d_mm": None,
        "residual_m": round(residual, 6),
        "corners": pts,
    }


# ── midpoint (classical) ───────────────────────────────────────────────
def midpoint(pose1, corners1, pose2, corners2, intr):
    return triangulate_size(pose1, corners1, pose2, corners2, intr)


# ── parallax ───────────────────────────────────────────────────────────
def _parallax_point(pose1, uv1, pose2, uv2, intr):
    o2 = geodetic_to_enu(pose1.lat, pose1.lon, pose1.alt_m,
                         pose2.lat, pose2.lon, pose2.alt_m)
    baseline = _norm(o2)
    d1 = _world_ray(pose1, uv1[0], uv1[1], intr)
    d2 = _world_ray(pose2, uv2[0], uv2[1], intr)
    cosang = max(-1.0, min(1.0, _dot(d1, d2)))
    angle = math.acos(cosang)
    if angle < 1e-6 or baseline < 1e-6:
        return triangulate_point(pose1, uv1, pose2, uv2, intr)
    z = baseline / (2.0 * math.tan(angle / 2.0))
    east, north, up = z * d1[0], z * d1[1], z * d1[2]
    from geospatial import enu_to_geodetic
    lat, lon, alt = enu_to_geodetic(pose1.lat, pose1.lon, pose1.alt_m, east, north, up)
    return {"east_m": east, "north_m": north, "up_m": up,
            "depth1_m": z, "depth2_m": z, "residual_m": 0.0,
            "lat": lat, "lon": lon, "alt_m": alt}


def parallax(pose1, corners1, pose2, corners2, intr):
    pts = [_parallax_point(pose1, a, pose2, b, intr) for a, b in zip(corners1, corners2)]
    w, h = _robust_size(pts)
    return _pack(pts, w, h, max(p["residual_m"] for p in pts))


# ── non-linear least squares ───────────────────────────────────────────
def _ls_point(pose1, uv1, pose2, uv2, intr):
    o2 = geodetic_to_enu(pose1.lat, pose1.lon, pose1.alt_m,
                         pose2.lat, pose2.lon, pose2.alt_m)
    seed = triangulate_point(pose1, uv1, pose2, uv2, intr)
    if seed is None:
        return None

    def project(p, pose, origin):
        rel = (p[0] - origin[0], p[1] - origin[1], p[2] - origin[2])
        return enu_direction_to_pixel(rel[0], rel[1], rel[2], pose, intr)

    def objective(p):
        a = project(p, pose1, (0.0, 0.0, 0.0))
        b = project(p, pose2, o2)
        if a is None or b is None:
            return 1e12
        return ((a[0] - uv1[0]) ** 2 + (a[1] - uv1[1]) ** 2
                + (b[0] - uv2[0]) ** 2 + (b[1] - uv2[1]) ** 2)

    best, cost = nelder_mead(objective,
                             [seed["east_m"], seed["north_m"], seed["up_m"]],
                             [0.2, 0.2, 0.2], max_iter=120)
    return {"east_m": best[0], "north_m": best[1], "up_m": best[2],
            "depth1_m": seed["depth1_m"], "depth2_m": seed["depth2_m"],
            "residual_m": math.sqrt(max(0.0, cost)),
            "lat": seed["lat"], "lon": seed["lon"], "alt_m": seed["alt_m"]}


def least_squares(pose1, corners1, pose2, corners2, intr):
    pts = [_ls_point(pose1, a, pose2, b, intr) for a, b in zip(corners1, corners2)]
    if any(p is None for p in pts):
        return midpoint(pose1, corners1, pose2, corners2, intr)
    w, h = _robust_size(pts)
    return _pack(pts, w, h, max(p["residual_m"] for p in pts))


# ── RANSAC / robust median ─────────────────────────────────────────────
def ransac(pose1, corners1, pose2, corners2, intr):
    pts = [triangulate_point(pose1, a, pose2, b, intr)
           for a, b in zip(corners1, corners2)]
    if any(p is None for p in pts):
        return midpoint(pose1, corners1, pose2, corners2, intr)
    residuals = [p["residual_m"] for p in pts]
    cutoff = max(0.05, 2.0 * median(residuals))
    outliers = sum(1 for r in residuals if r > cutoff)
    w, h = _robust_size(pts)
    result = _pack(pts, w, h, max(residuals))
    result["outliers"] = outliers
    return result


# ── weighted midpoint ──────────────────────────────────────────────────
def weighted(pose1, corners1, pose2, corners2, intr):
    pts = [triangulate_point(pose1, a, pose2, b, intr)
           for a, b in zip(corners1, corners2)]
    if any(p is None for p in pts):
        return midpoint(pose1, corners1, pose2, corners2, intr)
    weights = [1.0 / (p["residual_m"] + 1e-3) for p in pts]
    wsum = sum(weights)
    centroid = {
        "east_m": sum(p["east_m"] * w for p, w in zip(pts, weights)) / wsum,
        "north_m": sum(p["north_m"] * w for p, w in zip(pts, weights)) / wsum,
        "up_m": sum(p["up_m"] * w for p, w in zip(pts, weights)) / wsum,
    }
    # translate each corner toward the weighted centroid by its confidence
    adjusted = []
    for p, w in zip(pts, weights):
        k = min(0.5, w / (wsum / len(pts)) * 0.05)
        adjusted.append({
            "east_m": p["east_m"] + k * (centroid["east_m"] - p["east_m"]),
            "north_m": p["north_m"] + k * (centroid["north_m"] - p["north_m"]),
            "up_m": p["up_m"] + k * (centroid["up_m"] - p["up_m"]),
        })
    w_mm, h_mm = _robust_size(adjusted)
    return _pack(pts, w_mm, h_mm, max(p["residual_m"] for p in pts))


TRIANGULATORS = {
    "midpoint": midpoint,
    "parallax": parallax,
    "least_squares": least_squares,
    "ransac": ransac,
    "weighted": weighted,
}
