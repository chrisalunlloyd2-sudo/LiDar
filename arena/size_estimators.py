"""lidarv2 size estimators — 11 ways to turn pixels + depth into millimetres.

Deliberately redundant. The arena measures which ones actually reduce error;
the rest are snipped.
"""

import math

from fit_engine import bbox_corners, fit_rectangle


def _intr(obs):
    return obs.intrinsics


def _pixel_dims(obs, depth_m):
    f = _intr(obs).f_px or 1.0
    return (depth_m * obs.object_px_width / f * 1000.0,
            depth_m * obs.object_px_height / f * 1000.0)


def _result(w, h, method, sigma_w=0.0, sigma_h=0.0):
    return {
        "w_mm": max(1e-6, float(w)),
        "h_mm": max(1e-6, float(h)),
        "d_mm": None,
        "method": method,
        "sigma_w_mm": float(sigma_w),
        "sigma_h_mm": float(sigma_h),
    }


def pixel_scale(obs, depth_m, level=0.95):
    w, h = _pixel_dims(obs, depth_m)
    f = _intr(obs).f_px or 1.0
    sw = depth_m / f * 1000.0 * 0.5  # half-pixel quantisation
    return _result(w, h, "pixel_scale", sw, sw)


def angular(obs, depth_m, level=0.95):
    fov = _intr(obs).fov_deg or 60.0
    f_eff = (_intr(obs).image_w / 2.0) / math.tan(math.radians(fov) / 2.0)
    w = depth_m * obs.object_px_width / f_eff * 1000.0
    h = depth_m * obs.object_px_height / f_eff * 1000.0
    return _result(w, h, "angular", w * 0.03, h * 0.03)


def trig(obs, depth_m, level=0.95):
    f = _intr(obs).f_px or 1.0
    tw = 2.0 * math.atan(obs.object_px_width / (2.0 * f))
    th = 2.0 * math.atan(obs.object_px_height / (2.0 * f))
    return _result(2.0 * depth_m * math.tan(tw / 2.0) * 1000.0,
                   2.0 * depth_m * math.tan(th / 2.0) * 1000.0,
                   "trig", 0.0, 0.0)


def similar_triangles(obs, depth_m, level=0.95):
    f = _intr(obs).f_px or 1.0
    w, h = _pixel_dims(obs, depth_m)
    return _result(w * (f / f), h, "similar_triangles")


def ground_plane(obs, depth_m, level=0.95):
    if obs.pose is None:
        return pixel_scale(obs, depth_m, level)
    f = _intr(obs).f_px or 1.0
    v_bottom = obs.center_y + obs.object_px_height / 2.0
    elevation = math.atan2(v_bottom - _intr(obs).cy, f)
    depression = elevation - math.radians(obs.pose.pitch_deg)
    if depression <= math.radians(0.5):
        return pixel_scale(obs, depth_m, level)
    d_gp = obs.camera_height_m / math.tan(depression)
    w, h = _pixel_dims(obs, d_gp)
    return _result(w, h, "ground_plane", w * 0.10, h * 0.10)


def cross_ratio(obs, depth_m, level=0.95):
    """Projective cross-ratio correction against the frame edges."""
    W = _intr(obs).image_w or 1
    l = obs.center_x - obs.object_px_width / 2.0
    r = obs.center_x + obs.object_px_width / 2.0
    cr = ((r - l) * (W - 0.0)) / max(1e-6, (r - 0.0) * (W - l))
    w, h = _pixel_dims(obs, depth_m)
    factor = 1.0 + 0.10 * (cr - 0.5)
    return _result(w * factor, h * factor, "cross_ratio")


def vanishing_point(obs, depth_m, level=0.95):
    """Perspective correction from the horizon implied by the pitch."""
    if obs.pose is None:
        return pixel_scale(obs, depth_m, level)
    f = _intr(obs).f_px or 1.0
    horizon = _intr(obs).cy - f * math.tan(math.radians(obs.pose.pitch_deg))
    v_top = obs.center_y - obs.object_px_height / 2.0
    v_bottom = obs.center_y + obs.object_px_height / 2.0
    if (v_top - horizon) * (v_bottom - horizon) <= 0:
        return pixel_scale(obs, depth_m, level)
    ratio = abs(v_bottom - horizon) / max(1e-6, abs(v_top - horizon))
    factor = 2.0 * ratio / (1.0 + ratio)
    w, h = _pixel_dims(obs, depth_m)
    return _result(w * factor, h * factor, "vanishing_point")


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(matrix)
    a = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        pv = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor:
                for j in range(col, n + 1):
                    a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def _homography(src, dst):
    rows, rhs = [], []
    for (u, v), (x, y) in zip(src, dst):
        rows.append([u, v, 1, 0, 0, 0, -u * x, -v * x]); rhs.append(x)
        rows.append([0, 0, 0, u, v, 1, -u * y, -v * y]); rhs.append(y)
    h = _solve(rows, rhs)
    return h + [1.0] if h else None


def homography(obs, depth_m, level=0.95):
    """4-point DLT from the bbox to a unit square, then local Jacobian scale."""
    f = _intr(obs).f_px or 1.0
    l = obs.center_x - obs.object_px_width / 2.0
    r = obs.center_x + obs.object_px_width / 2.0
    t = obs.center_y - obs.object_px_height / 2.0
    b = obs.center_y + obs.object_px_height / 2.0
    src = [(l, t), (r, t), (r, b), (l, b)]
    dst = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    h = _homography(src, dst)
    if not h:
        return pixel_scale(obs, depth_m, level)
    u, v = obs.center_x, obs.center_y
    den = h[6] * u + h[7] * v + h[8]
    if abs(den) < 1e-12:
        return pixel_scale(obs, depth_m, level)
    j11 = (h[0] * den - (h[0] * u + h[1] * v + h[2]) * h[6]) / (den * den)
    j22 = (h[4] * den - (h[3] * u + h[4] * v + h[5]) * h[7]) / (den * den)
    w = depth_m * abs(1.0 / j11) / f * 1000.0 if abs(j11) > 1e-12 else obs.object_px_width
    hgt = depth_m * abs(1.0 / j22) / f * 1000.0 if abs(j22) > 1e-12 else obs.object_px_height
    return _result(w, hgt, "homography")


def fit_rect(obs, depth_m, level=0.95):
    fit = fit_rectangle(bbox_corners(obs), _intr(obs), depth_m)
    if fit is None:
        return pixel_scale(obs, depth_m, level)
    return _result(fit["w_mm"], fit["h_mm"], "fit_rect")


def _multi(obs, depth_m, level, how):
    cands = [pixel_scale, angular, ground_plane, trig]
    vals = [fn(obs, depth_m, level) for fn in cands]
    ws = sorted(v["w_mm"] for v in vals)
    hs = sorted(v["h_mm"] for v in vals)
    if how == "median":
        w = ws[len(ws) // 2]
        h = hs[len(hs) // 2]
    else:
        w = sum(ws) / len(ws)
        h = sum(hs) / len(hs)
    return _result(w, h, "%s_multi" % how)


def median_multi(obs, depth_m, level=0.95):
    return _multi(obs, depth_m, level, "median")


def weighted_multi(obs, depth_m, level=0.95):
    return _multi(obs, depth_m, level, "weighted")


SIZE_ESTIMATORS = {
    "pixel_scale": pixel_scale,
    "angular": angular,
    "trig": trig,
    "similar_triangles": similar_triangles,
    "ground_plane": ground_plane,
    "cross_ratio": cross_ratio,
    "vanishing_point": vanishing_point,
    "homography": homography,
    "fit_rect": fit_rect,
    "median_multi": median_multi,
    "weighted_multi": weighted_multi,
}

SLOW_ESTIMATORS = {"fit_rect"}
