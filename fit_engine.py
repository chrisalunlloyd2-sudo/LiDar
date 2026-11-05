"""lidarv2 fit engine — fit a 3-D rectangle/cuboid model to observed corners.

The fit minimises reprojection error: project the parametric model through the
camera intrinsics and compare against the observed image corners. A stdlib
Nelder-Mead optimiser handles the small parameter space, so no numpy/scipy.

`fit_rectangle` solves (width, height, yaw) at a fixed depth.
`fit_cuboid`    solves (width, height, thickness, yaw) at a fixed distance.
The model is translated to the observed corner centroid, so off-centre
bounding boxes (e.g. floor-contact objects) fit correctly.
"""

import math

from models import CameraIntrinsics


# ── Nelder-Mead ────────────────────────────────────────────────────────
def nelder_mead(func, x0, steps, max_iter=300, tol=1e-9):
    """Minimise `func` over a small real vector. Returns (x, f(x))."""
    n = len(x0)
    simplex = [list(map(float, x0))]
    for i in range(n):
        x = list(simplex[0])
        x[i] += steps[i]
        simplex.append(x)
    fvals = [func(x) for x in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) <= tol * (abs(fvals[0]) + tol):
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]

        xr = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = func(xr)
        if fr < fvals[0]:
            xe = [centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(n)]
            fe = func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])
                                  for j in range(n)]
                    fvals[i] = func(simplex[i])

    best = min(range(n + 1), key=lambda i: fvals[i])
    return simplex[best], fvals[best]


# ── projection models ──────────────────────────────────────────────────
def _project_local(points_m, yaw_deg, translate, intr):
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    tx, ty, tz = translate
    out = []
    for x, y, z in points_m:
        xr = x * cy + z * sy + tx
        yr = y + ty
        zr = -x * sy + z * cy + tz
        if zr <= 1e-6:
            zr = 1e-6
        u = intr.cx + intr.f_px * xr / zr
        v = intr.cy + intr.f_px * yr / zr
        out.append((u, v))
    return out


def project_rectangle(w_mm, h_mm, yaw_deg, depth_m, intr, center_m=(0.0, 0.0)):
    w, h = w_mm / 1000.0, h_mm / 1000.0
    local = [(-w / 2, -h / 2, 0.0), (w / 2, -h / 2, 0.0),
             (w / 2, h / 2, 0.0), (-w / 2, h / 2, 0.0)]
    return _project_local(local, yaw_deg, (center_m[0], center_m[1], depth_m), intr)


def project_cuboid(w_mm, h_mm, d_mm, yaw_deg, distance_m, intr, center_m=(0.0, 0.0)):
    w, h, d = w_mm / 1000.0, h_mm / 1000.0, d_mm / 1000.0
    face = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    local = [(x, y, 0.0) for x, y in face] + [(x, y, d) for x, y in face]
    return _project_local(local, yaw_deg, (center_m[0], center_m[1], distance_m), intr)


def bbox_corners(obs):
    """The four corners of an axis-aligned detected bounding box."""
    l = obs.center_x - obs.object_px_width / 2.0
    r = obs.center_x + obs.object_px_width / 2.0
    t = obs.center_y - obs.object_px_height / 2.0
    b = obs.center_y + obs.object_px_height / 2.0
    return [(l, t), (r, t), (r, b), (l, b)]


# ── fitters ────────────────────────────────────────────────────────────
def _rms(projected, observed):
    n = min(len(projected), len(observed))
    total = 0.0
    for (pu, pv), (ou, ov) in zip(projected[:n], observed[:n]):
        total += (pu - ou) ** 2 + (pv - ov) ** 2
    return math.sqrt(total / max(1, n))


def _centroid(corners):
    n = max(1, len(corners))
    return (sum(c[0] for c in corners) / n, sum(c[1] for c in corners) / n)


def _center_m(corners, intr, depth_m):
    cu, cv = _centroid(corners)
    return ((cu - intr.cx) / intr.f_px * depth_m,
            (cv - intr.cy) / intr.f_px * depth_m)


def fit_rectangle(corners_px, intr, depth_m, init=(100.0, 100.0, 0.0)):
    """Fit (width, height, yaw) to four observed corners at a fixed depth."""
    if len(corners_px) < 4:
        return None
    center = _center_m(corners_px, intr, depth_m)

    def objective(p):
        w, h, yaw = p
        if w <= 1.0 or h <= 1.0:
            return 1e12
        proj = project_rectangle(w, h, yaw, depth_m, intr, center)
        return sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                   for a, b in zip(proj, corners_px[:4]))

    best, _cost = nelder_mead(objective, init, steps=[10.0, 10.0, 5.0])
    w, h, yaw = best
    proj = project_rectangle(w, h, yaw, depth_m, intr, center)
    rms = _rms(proj, corners_px[:4])
    return {
        "kind": "rectangle",
        "w_mm": round(max(1.0, w), 2),
        "h_mm": round(max(1.0, h), 2),
        "d_mm": None,
        "yaw_deg": round(yaw, 3),
        "depth_m": depth_m,
        "rms_px": round(rms, 4),
        "corners_px": [{"u": round(p[0], 2), "v": round(p[1], 2)} for p in proj],
        "ok": rms < 50.0,
    }


def fit_cuboid(corners_px, intr, distance_m, init=(100.0, 100.0, 100.0, 0.0)):
    """Fit (width, height, thickness, yaw) to eight observed cuboid corners."""
    if len(corners_px) < 8:
        return None
    # front face = first four corners
    center = _center_m(corners_px[:4], intr, distance_m)

    def objective(p):
        w, h, d, yaw = p
        if w <= 1.0 or h <= 1.0 or d <= 0.0:
            return 1e12
        proj = project_cuboid(w, h, d, yaw, distance_m, intr, center)
        return sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                   for a, b in zip(proj, corners_px[:8]))

    best, _cost = nelder_mead(objective, init, steps=[10.0, 10.0, 10.0, 5.0])
    w, h, d, yaw = best
    proj = project_cuboid(w, h, d, yaw, distance_m, intr, center)
    rms = _rms(proj, corners_px[:8])
    return {
        "kind": "cuboid",
        "w_mm": round(max(1.0, w), 2),
        "h_mm": round(max(1.0, h), 2),
        "d_mm": round(max(0.1, d), 2),
        "yaw_deg": round(yaw, 3),
        "distance_m": distance_m,
        "rms_px": round(rms, 4),
        "corners_px": [{"u": round(p[0], 2), "v": round(p[1], 2)} for p in proj],
        "ok": rms < 50.0,
    }


def fit_observation(obs, depth_m):
    """Fit a fronto-parallel rectangle to an observation's bbox."""
    f = obs.intrinsics.f_px or 1.0
    w0 = depth_m * obs.object_px_width / f * 1000.0
    h0 = depth_m * obs.object_px_height / f * 1000.0
    return fit_rectangle(bbox_corners(obs), obs.intrinsics, depth_m,
                         init=(max(1.0, w0), max(1.0, h0), 0.0))
