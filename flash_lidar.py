"""LiDar flash-photometric depth — active-light inverse-square ranging.

This merges the differential-flash pipeline into the emulative LiDAR engine.
The camera torch is pulsed; the ambient frame is subtracted, and the additive
flux per grid cell is converted to range by the inverse-square law. The result
is a *depth grid*, not a single number, and it feeds the same inverse-variance
fusion and confidence-interval machinery as the monocular cues.

Pipeline
--------
    ambient I0 + torch I1  ->  D = max(0, I1 - I0)      (differential map)
    grid cell (u, v)       ->  S = mean(D over the cell)  (additive flux)
    reference patch d0, S0 ->  d = d0 * sqrt(S0 / S)      (inverse square)
    N frames               ->  inverse-variance depth per cell + spread
    grid + intrinsics      ->  3-D points in ENU / WGS84
    connected plane        ->  normal, RMS, principal-axis extents
    extents + catalog      ->  [TARGET DETECTED] with a 95% interval

Honest limits
-------------
Reflectance is not observable from one flash/ambient pair. The map is relative
to the calibrated reference patch: a cell whose albedo differs by factor `a`
reads `sqrt(a)` too far. `FlashCalibration.albedo_rel_sigma` turns that prior
uncertainty into extra depth sigma instead of pretending it away, and the CLI
reports the mixed-albedo case separately. Specular and saturated cells are
dropped rather than clipped.

Pure Python stdlib. No numpy, no PIL, no native library: the browser already
computes cell means on a canvas, so the backend only sees numbers.
"""

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from inventory import TAG_CATALOG
from models import CameraIntrinsics, CameraPose, CueEstimate, Observation
from uncertainty import fuse_cues

SATURATED_DN = 250.0
NOISE_FLOOR_DN = 1.5
MIN_PLANE_CELLS = 6
ASCII_RAMP = " .:-=+*#%@"
# Catalog entries whose w_mm/h_mm describe the whole observed face, so the
# major extent is constrained too. Lumber is excluded: its face length is the
# board length, which the catalog does not know.
FACE_IS_OBJECT = frozenset({"credit_card", "a4_paper", "brick", "paint_can", "box"})


# ── calibration ────────────────────────────────────────────────────────
@dataclass
class FlashCalibration:
    """Reference measurement for the inverse-square gauge.

    Point the torch at a flat surface of known distance, read the differential
    flux there, and pass both numbers. `sigma_delta_dn` is the per-cell noise
    on the differential map (estimate it from two ambient frames);
    `albedo_rel_sigma` is how much the scene's reflectance is expected to
    differ from the reference patch.
    """
    reference_depth_m: float
    reference_delta_dn: float
    sigma_delta_dn: float = 1.0
    albedo_rel_sigma: float = 0.15
    min_depth_m: float = 0.3
    max_depth_m: float = 8.0
    noise_floor_dn: float = NOISE_FLOOR_DN

    def constant(self):
        """k = S0 * d0^2, the lumped source/albedo constant."""
        return self.reference_delta_dn * self.reference_depth_m ** 2

    def as_dict(self):
        return {
            "reference_depth_m": self.reference_depth_m,
            "reference_delta_dn": round(self.reference_delta_dn, 3),
            "sigma_delta_dn": self.sigma_delta_dn,
            "albedo_rel_sigma": self.albedo_rel_sigma,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "constant": round(self.constant(), 3),
        }


# ── differential map + range ───────────────────────────────────────────
def differential_map(ambient_cells, flash_cells):
    """Net additive intensity per cell, sensor noise clipped at zero."""
    if len(ambient_cells) != len(flash_cells):
        raise ValueError("ambient and flash grids differ in length")
    return [max(0.0, float(f) - float(a))
            for a, f in zip(ambient_cells, flash_cells)]


def cell_centres(cols, rows, intrinsics):
    """Pixel coordinate of every cell centre, row-major."""
    out = []
    for r in range(rows):
        v = (r + 0.5) * intrinsics.image_h / rows
        for c in range(cols):
            u = (c + 0.5) * intrinsics.image_w / cols
            out.append((u, v))
    return out


def range_from_flux(flux_dn, cal):
    """Inverse-square range and its 1-sigma from one cell's flux."""
    if flux_dn < cal.noise_floor_dn:
        return None, None, "dark"
    d = cal.reference_depth_m * math.sqrt(cal.reference_delta_dn / flux_dn)
    if d < cal.min_depth_m:
        return d, None, "too_near"
    if d > cal.max_depth_m:
        return d, None, "too_far"
    # d ~ S^-1/2 -> relative sigma is half the flux sigma; albedo adds a prior
    rel = 0.5 * math.hypot(cal.sigma_delta_dn / flux_dn, cal.albedo_rel_sigma)
    return d, d * rel, "ok"


def analyse_frame(ambient_cells, flash_cells, cols, rows, cal, intrinsics):
    """One flash/ambient pair -> per-cell depth estimates."""
    diff = differential_map(ambient_cells, flash_cells)
    centres = cell_centres(cols, rows, intrinsics)
    cells = []
    for idx, flux in enumerate(diff):
        u, v = centres[idx]
        cell = {"index": idx, "u": round(u, 2), "v": round(v, 2),
                "flux_dn": round(flux, 3)}
        if flash_cells[idx] >= SATURATED_DN:
            cell.update(depth_m=None, sigma_m=None, valid=False,
                        reason="saturated")
        else:
            d, sigma, reason = range_from_flux(flux, cal)
            cell.update(depth_m=None if d is None else round(d, 6),
                        sigma_m=None if sigma is None else round(sigma, 6),
                        valid=(sigma is not None), reason=reason)
        cells.append(cell)
    return cells


def valid_fraction(cells):
    if not cells:
        return 0.0
    return sum(1 for c in cells if c["valid"]) / len(cells)


# ── calibration from a captured pair ───────────────────────────────────
def calibrate_from_grids(ambient_cells, flash_cells, cols, rows, intrinsics,
                         known_range_m=None, inner=0.25):
    """Suggest a `FlashCalibration` from one ambient/torch pair.

    The reference flux is the median differential of the centre box (the
    reticle), because that is what the user is aiming at when calibrating.
    `known_range_m` is the *range* to that patch; without it the result is
    unanchored and cannot produce metres.

    Returns plain numbers plus a health verdict, so the caller can refuse a bad
    calibration instead of silently ranging with it.
    """
    if len(ambient_cells) != cols * rows or len(flash_cells) != cols * rows:
        raise ValueError("ambient and flash grids must each hold cols*rows cells")
    diff = differential_map(ambient_cells, flash_cells)
    c0 = max(0, int(cols * inner))
    c1 = max(c0 + 1, cols - c0)
    r0 = max(0, int(rows * inner))
    r1 = max(r0 + 1, rows - r0)
    centre = sorted(diff[r * cols + c]
                    for r in range(r0, r1) for c in range(c0, c1))
    med = statistics.median(centre)
    p90 = centre[int(0.9 * (len(centre) - 1))]
    lit = sum(1 for v in diff if v >= NOISE_FLOOR_DN) / len(diff)
    warnings = []
    if med > 200.0:
        warnings.append("centre differential above 200 DN — stop down or move "
                        "back, cells are close to clipping")
    elif med < 20.0:
        warnings.append("centre differential below 20 DN — move closer or "
                        "brighten; range noise will dominate")
    if lit < 0.5:
        warnings.append("less than half the frame receives torch light")
    if med > 0 and p90 / med > 1.5:
        warnings.append("uneven illumination: p90/median = %.2f" % (p90 / med))
    status = "ok"
    if med <= 0.0:
        status = "dark"
    elif med > 200.0:
        status = "saturated"
    elif warnings:
        status = "marginal"
    out = {
        "cols": cols,
        "rows": rows,
        "centre_median_dn": round(med, 3),
        "centre_p90_dn": round(p90, 3),
        "full_median_dn": round(statistics.median(sorted(diff)), 3),
        "lit_fraction": round(lit, 4),
        "status": status,
        "warnings": warnings,
        "anchored": known_range_m is not None,
        "suggested_calibration": None,
    }
    if known_range_m is not None and med > 0:
        cal = FlashCalibration(reference_depth_m=float(known_range_m),
                               reference_delta_dn=round(med, 3),
                               sigma_delta_dn=1.4)
        out["suggested_calibration"] = cal.as_dict()
        out["suggested_calibration"]["albedo_rel_sigma"] = cal.albedo_rel_sigma
    return out


# ── temporal fusion ────────────────────────────────────────────────────
def fuse_depth_frames(frames, min_frames=3, converge_sigma_m=0.02):
    """Fuse N independent depth grids cell-by-cell (inverse variance)."""
    if not frames:
        return []
    n_cells = len(frames[0])
    fused = []
    for idx in range(n_cells):
        samples = [(c["depth_m"], c["sigma_m"]) for f in frames
                   for c in (f[idx],) if c["valid"] and c["depth_m"]]
        if not samples:
            fused.append({"index": idx, "valid": False, "n": 0,
                          "depth_m": None, "sigma_m": None,
                          "spread_m": None, "converged": False})
            continue
        w = [1.0 / (s * s) for _, s in samples]
        wsum = sum(w)
        mu = sum(d * wi for (d, _), wi in zip(samples, w)) / wsum
        sigma = math.sqrt(1.0 / wsum)
        spread = math.sqrt(max(0.0, sum(wi * (d - mu) ** 2
                                        for (d, _), wi in zip(samples, w)) / wsum))
        converged = (len(samples) >= min_frames
                     and sigma <= converge_sigma_m
                     and spread <= max(3.0 * sigma, 0.01))
        fused.append({
            "index": idx, "valid": True, "n": len(samples),
            "depth_m": round(mu, 6), "sigma_m": round(sigma, 6),
            "spread_m": round(spread, 6), "converged": converged,
        })
    return fused


# ── geometry ───────────────────────────────────────────────────────────
def undistort_pixel(u, v, intrinsics, k1=0.0, k2=0.0, iterations=5):
    """Invert radial lens distortion by fixed-point iteration."""
    if k1 == 0.0 and k2 == 0.0:
        return u, v
    xd = (u - intrinsics.cx) / intrinsics.f_px
    yd = (v - intrinsics.cy) / intrinsics.f_px
    x, y = xd, yd
    for _ in range(max(1, iterations)):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x, y = xd / radial, yd / radial
    return intrinsics.cx + x * intrinsics.f_px, intrinsics.cy + y * intrinsics.f_px


def _enu_point(pose, intrinsics, u, v, depth_m):
    from geospatial import pixel_to_ray, ray_camera_to_enu, enu_to_geodetic

    ray = pixel_to_ray(u, v, intrinsics)
    direction = ray_camera_to_enu(ray, pose.heading_deg, pose.pitch_deg,
                                  pose.roll_deg)
    east, north, up = (depth_m * direction[0], depth_m * direction[1],
                       depth_m * direction[2])
    lat, lon, alt = enu_to_geodetic(pose.lat, pose.lon, pose.alt_m,
                                    east, north, up)
    return {"east_m": east, "north_m": north, "up_m": up,
            "lat": lat, "lon": lon, "alt_m": alt, "depth_m": depth_m}


def cell_points(cells, cols, rows, intrinsics, pose=None, k1=0.0, k2=0.0):
    """Valid depth cells -> 3-D ENU/WGS84 points (row-major, None where invalid)."""
    pose = pose or CameraPose()
    points = []
    for cell in cells:
        if not cell["valid"] or not cell["depth_m"]:
            points.append(None)
            continue
        u, v = undistort_pixel(cell["u"], cell["v"], intrinsics, k1, k2)
        p = _enu_point(pose, intrinsics, u, v, cell["depth_m"])
        p.update(u=u, v=v, sigma_m=cell.get("sigma_m") or 0.0,
                 index=cell["index"])
        points.append(p)
    return points


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return (0.0, 0.0, 0.0) if n <= 0 else (a[0] / n, a[1] / n, a[2] / n)


def _vec(p):
    return (p["east_m"], p["north_m"], p["up_m"])


def _jacobi_eigen_3x3(matrix, iterations=24):
    """Eigenvalues/vectors of a symmetric 3x3 (cyclic Jacobi, pure stdlib)."""
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(iterations):
        if abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2]) < 1e-12:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-15:
                continue
            theta = 0.5 * math.atan2(2.0 * a[p][q], a[q][q] - a[p][p])
            c, s = math.cos(theta), math.sin(theta)
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    values = [a[i][i] for i in range(3)]
    vectors = [[v[0][j], v[1][j], v[2][j]] for j in range(3)]
    return values, vectors


def _plane_normal(vecs):
    """Normal = eigenvector of the smallest covariance eigenvalue, toward camera."""
    n = len(vecs)
    cx = sum(v[0] for v in vecs) / n
    cy = sum(v[1] for v in vecs) / n
    cz = sum(v[2] for v in vecs) / n
    cxx = cyy = czz = cxy = cxz = cyz = 0.0
    for v in vecs:
        dx, dy, dz = v[0] - cx, v[1] - cy, v[2] - cz
        cxx += dx * dx
        cyy += dy * dy
        czz += dz * dz
        cxy += dx * dy
        cxz += dx * dz
        cyz += dy * dz
    m = [[cxx / n, cxy / n, cxz / n],
         [cxy / n, cyy / n, cyz / n],
         [cxz / n, cyz / n, czz / n]]
    values, vectors = _jacobi_eigen_3x3(m)
    normal = _norm(tuple(vectors[values.index(min(values))]))
    if _dot(normal, (cx, cy, cz)) > 0:   # point at the camera (origin)
        normal = (-normal[0], -normal[1], -normal[2])
    return normal, (cx, cy, cz)


def segment_planes(points, cols, rows, max_depth_jump_m=None, min_cells=12):
    """Grow regions by depth continuity, fit a plane, then merge fragments.

    Local normals are useless at flash ranges: at 3.4 m the depth noise
    (~0.1 m) is larger than the cell spacing, so a cross-product normal is
    noise. Neighbour-to-neighbour depth continuity is robust to that, and the
    normal comes from a PCA fit over the whole region.
    """
    sigmas = [p["sigma_m"] for p in points if p is not None and p.get("sigma_m")]
    med_sigma = statistics.median(sigmas) if sigmas else 0.02
    jump = (max_depth_jump_m if max_depth_jump_m is not None
            else max(0.08, 4.0 * med_sigma))
    visited = [False] * len(points)
    planes = []
    for start in range(len(points)):
        if visited[start] or points[start] is None:
            continue
        queue = [start]
        visited[start] = True
        members = []
        while queue:
            idx = queue.pop()
            members.append(idx)
            r, c = divmod(idx, cols)
            d_here = points[idx]["depth_m"]
            for nb, ok in ((idx - 1, c > 0), (idx + 1, c < cols - 1),
                           (idx - cols, r > 0), (idx + cols, r < rows - 1)):
                if not ok or visited[nb] or points[nb] is None:
                    continue
                if abs(points[nb]["depth_m"] - d_here) > jump:
                    continue
                visited[nb] = True
                queue.append(nb)
        if len(members) < min_cells:
            continue
        plane = _fit_plane(points, members)
        if plane is not None:
            planes.append(plane)

    planes = _merge_coplanar(planes, points, med_sigma, min_cells)
    planes.sort(key=lambda p: p["cells"], reverse=True)
    for i, p in enumerate(planes):
        p["plane_id"] = i
    return planes


def _merge_coplanar(planes, points, med_sigma, min_cells):
    """Collapse fragments of the same physical face into one plane."""
    cos_tol = math.cos(math.radians(8.0))
    offset_tol = max(0.03, 3.0 * med_sigma)
    changed = True
    while changed:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                na = (a["normal"]["east"], a["normal"]["north"], a["normal"]["up"])
                nb = (b["normal"]["east"], b["normal"]["north"], b["normal"]["up"])
                if _dot(na, nb) < cos_tol:
                    continue
                if abs(a["offset_m"] - b["offset_m"]) > offset_tol:
                    continue
                members = sorted(set(a["members"]) | set(b["members"]))
                merged = _fit_plane(points, members)
                if merged is None:
                    continue
                planes[i] = merged
                planes.pop(j)
                changed = True
                break
            if changed:
                break
    return [p for p in planes if p["cells"] >= min_cells]


def _fit_plane(points, members):
    """PCA normal, plane offset, RMS, and principal-axis extents."""
    vecs_all = [_vec(points[i]) for i in members]
    sigmas = [points[i].get("sigma_m") or 0.0 for i in members]
    med_sigma = statistics.median(sigmas) if sigmas else 0.0
    inliers = list(range(len(members)))
    for _ in range(3):
        normal, centroid = _plane_normal([vecs_all[k] for k in inliers])
        dists = [_dot(_sub(v, centroid), normal) for v in vecs_all]
        med = statistics.median(dists)
        mad = statistics.median(abs(d - med) for d in dists) * 1.4826
        tol = max(3.0 * mad, 0.02, 2.0 * med_sigma)
        new = [k for k, d in enumerate(dists) if abs(d - med) <= tol]
        if not new:
            break
        inliers = new
        if len(inliers) == len(members):
            break
    if len(inliers) < MIN_PLANE_CELLS:
        return None

    vecs = [vecs_all[k] for k in inliers]
    normal, centroid = _plane_normal(vecs)
    cx, cy, cz = centroid
    dists = [_dot(_sub(v, centroid), normal) for v in vecs]
    rms = math.sqrt(sum(d * d for d in dists) / len(dists))
    offset = _dot(centroid, normal)

    ref = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    t1 = _norm(_cross(ref, normal))
    t2 = _cross(normal, t1)
    pts2 = [(_dot(_sub(v, centroid), t1), _dot(_sub(v, centroid), t2)) for v in vecs]
    extents = _principal_extents(pts2)
    area = max(1e-12, extents[0] * extents[1])
    cell_m = math.sqrt(area / len(vecs))
    # hull extents are measured between cell centres: add one cell of missing
    # boundary (half at each edge) so a 139.7 mm board does not read 125 mm
    length_m = max(extents) + cell_m
    width_m = min(extents) + cell_m
    kept_sigmas = [sigmas[k] for k in inliers]
    median_sigma = statistics.median(kept_sigmas) if kept_sigmas else 0.0
    return {
        "plane_id": -1,
        "cells": len(vecs),
        "centre": {"east_m": cx, "north_m": cy, "up_m": cz},
        "normal": {"east": normal[0], "north": normal[1], "up": normal[2]},
        "offset_m": round(offset, 5),
        "rms_m": round(rms, 5),
        "median_sigma_m": round(median_sigma, 5),
        "cell_m": round(cell_m, 5),
        "length_m": round(length_m, 4),
        "width_m": round(width_m, 4),
        "area_m2": round(length_m * width_m, 4),
        "aspect": round(length_m / max(width_m, 1e-9), 3),
        "members": [members[k] for k in inliers],
    }


def _principal_extents(pts2):
    """Bounding extents after rotating onto the point cloud's principal axis."""
    n = len(pts2)
    mx = sum(p[0] for p in pts2) / n
    my = sum(p[1] for p in pts2) / n
    cxx = sum((p[0] - mx) ** 2 for p in pts2) / n
    cyy = sum((p[1] - my) ** 2 for p in pts2) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in pts2) / n
    angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    ca, sa = math.cos(angle), math.sin(angle)
    us = [p[0] * ca + p[1] * sa for p in pts2]
    vs = [-p[0] * sa + p[1] * ca for p in pts2]
    return [max(us) - min(us), max(vs) - min(vs)]


# ── target matching ────────────────────────────────────────────────────
def match_target(plane, max_error=0.15, min_gap=0.05, min_aspect=2.0):
    """Match a plane to the catalog, and abstain when the evidence is weak.

    A planar face gives one cross-section. Width quantisation is about one
    cell (~2 cm), so at 2 m a 184 mm board can read 168 mm. Three guards keep
    the output honest: a shape prior (an elongated face cannot be a paint-can
    lid), a whole-face rule (a sheet, card or brick must match in both extents,
    so a 30 cm strip cannot pass as a credit card), and a separation rule (the
    best candidate must beat the runner-up by `min_gap`, otherwise the plane is
    reported as a measurement only).
    """
    minor_mm = min(plane["length_m"], plane["width_m"]) * 1000.0
    major_mm = max(plane["length_m"], plane["width_m"]) * 1000.0
    aspect = plane.get("aspect") or (major_mm / max(minor_mm, 1e-9))
    elongated = aspect >= min_aspect
    width_sigma_m = math.sqrt(plane["median_sigma_m"] ** 2
                              + (plane["cell_m"] / 2.0) ** 2)
    width_ci95_mm = 2.0 * width_sigma_m * 1000.0

    candidates = []
    for entry in TAG_CATALOG:
        a, b = entry["w_mm"], entry["h_mm"]
        cross_aspect = max(a, b) / max(min(a, b), 1e-9)
        if elongated and cross_aspect < 1.4:
            continue                      # a board is not a can lid
        if not elongated and cross_aspect >= 1.4:
            continue
        err = min(abs(minor_mm - a) / max(minor_mm, a),
                  abs(minor_mm - b) / max(minor_mm, b))
        if entry["tag"] in FACE_IS_OBJECT and \
                abs(major_mm - max(a, b)) / max(major_mm, max(a, b)) > max_error:
            continue                      # a 30 cm strip is not a credit card
        if err <= max_error:
            candidates.append({"tag": entry["tag"], "label": entry["label"],
                               "cross_mm": [a, b], "error": round(err, 4)})
    candidates.sort(key=lambda c: c["error"])
    head = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    decisive = bool(head) and (second is None
                               or second["error"] - head["error"] >= min_gap)
    if not decisive:
        head = None
    quality = 1.0 / (1.0 + plane["rms_m"] / 0.02)
    support = min(1.0, plane["cells"] / 200.0)
    agreement = 1.0 - (head["error"] if head else 0.5)
    confidence = 0.45 * quality + 0.30 * support + 0.25 * max(0.0, agreement)
    if head is None:
        confidence = min(confidence, 0.5)
    return {
        "tag": head["tag"] if head else None,
        "label": head["label"] if head else "unidentified",
        "reason": ("ok" if head else
                   ("ambiguous" if candidates else "no_catalog_match")),
        "candidates": [c["tag"] for c in candidates[:3]],
        "ambiguous": len(candidates) > 1,
        "cross_mm": head["cross_mm"] if head else None,
        "measured_width_mm": round(minor_mm, 1),
        "width_ci95_mm": round(width_ci95_mm, 1),
        "length_mm": round(major_mm, 1),
        "length_ci95_cm": round(width_sigma_m * 100.0, 2),
        "aspect": round(aspect, 3),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
    }


# ── flash as a depth cue (the merge point) ─────────────────────────────
def flash_cue(cells, cols, rows, intrinsics, u=None, v=None, radius_px=80.0):
    """Collapse the flash depth map near the reticle into one `CueEstimate`.

    The reticle is the measurement, so the cells nearest it pick the target:
    everything within `radius_px` is clustered around the depth of the closest
    valid cell, and only that cluster is fused. Without the gate a close board
    inside the radius is averaged with the wall behind it and reads too far.
    """
    cu = intrinsics.cx if u is None else float(u)
    cv = intrinsics.cy if v is None else float(v)
    nearest = min(cells, key=lambda c: math.hypot(c["u"] - cu, c["v"] - cv),
                  default=None)
    if nearest is None or not (nearest["valid"] and nearest["depth_m"]
                               and nearest["sigma_m"]):
        # the reticle cell itself is unmeasurable (saturated, dark, too far).
        # Falling back to the nearest valid cell would report the background.
        return None
    picked = [c for c in cells
              if c["valid"] and c["depth_m"] and c["sigma_m"]
              and math.hypot(c["u"] - cu, c["v"] - cv) <= radius_px]
    if len(picked) < 3:
        return None
    picked.sort(key=lambda c: math.hypot(c["u"] - cu, c["v"] - cv))
    gate = max(0.12, 3.0 * statistics.median(c["sigma_m"] for c in picked))
    anchor = picked[0]["depth_m"]
    picked = [c for c in picked if abs(c["depth_m"] - anchor) <= gate]
    if len(picked) < 3:
        return None
    w = [1.0 / (c["sigma_m"] ** 2) for c in picked]
    wsum = sum(w)
    mu = sum(c["depth_m"] * wi for c, wi in zip(picked, w)) / wsum
    spread = math.sqrt(max(0.0, sum(wi * (c["depth_m"] - mu) ** 2
                                    for c, wi in zip(picked, w)) / wsum))
    sigma = math.sqrt(1.0 / wsum + spread ** 2)
    return CueEstimate("flash_falloff", mu, max(sigma, 1e-4), 1.0)


def estimate_with_flash(obs: Observation, ambient_cells, flash_cells, cal,
                        cols, rows, level=0.95):
    """Full merge: flash grid -> depth cue -> fusion with the monocular cues."""
    cells = analyse_frame(ambient_cells, flash_cells, cols, rows, cal,
                          obs.intrinsics)
    cue = flash_cue(cells, cols, rows, obs.intrinsics,
                    u=obs.center_x, v=obs.center_y)
    cues = [cue] if cue else []
    from depth_engine import DepthEngine

    cues.extend(DepthEngine().cues(obs))
    estimate = fuse_cues(cues, level=level)
    return {"estimate": estimate, "cells": cells, "flash_cue": cue}


# ── terminal visualiser ────────────────────────────────────────────────
def ascii_depth(cells, cols, rows, width=None, ansi=False):
    """Render the depth grid; dense glyphs (or bright ANSI) are near."""
    width = width or cols
    height = max(1, int(round(width * rows / max(1, cols) * 0.5)))
    valid = [c["depth_m"] for c in cells if c["valid"] and c["depth_m"]]
    if not valid:
        return ["(no valid flash cells)"]
    lo = min(valid)
    hi = max(valid)
    span = max(hi - lo, 1e-6)
    lines = []
    for r in range(height):
        row = []
        for c in range(width):
            src_c = min(cols - 1, int(c * cols / width))
            src_r = min(rows - 1, int(r * rows / height))
            cell = cells[src_r * cols + src_c]
            if not cell["valid"] or not cell["depth_m"]:
                row.append(" ")
                continue
            near = 1.0 - (cell["depth_m"] - lo) / span
            if ansi:
                level = int(round(lerp(30, 235, near)))
                row.append("\x1b[38;2;%d;%d;%dm\u2588\x1b[0m"
                           % (level, level, level))
            else:
                idx = int(round(near * (len(ASCII_RAMP) - 1)))
                row.append(ASCII_RAMP[idx])
        lines.append("".join(row))
    return lines


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


# ── synthetic scene (tests + demo) ─────────────────────────────────────
@dataclass
class ScenePatch:
    """A fronto-parallel rectangle with real metric size, in camera coords."""
    tag: str
    label: str
    centre_x_m: float
    centre_y_m: float
    width_m: float
    height_m: float
    depth_m: float
    albedo: float = 0.55


@dataclass
class FlashScene:
    name: str
    patches: List[ScenePatch]
    wall_depth_m: float = 3.4
    wall_albedo: float = 0.55
    notes: str = ""


DEFAULT_SCENE = FlashScene(
    name="workshop — 2x6 at 1.5 m, 2x8 at 2.3 m, wall at 3.4 m",
    patches=[
        ScenePatch("2x6", "2x6 lumber", -0.20, 0.02, 0.1397, 0.62, 1.5),
        ScenePatch("2x8", "2x8 lumber", 0.30, -0.05, 0.1842, 0.55, 2.3),
    ],
)


def scene_ray(scene, u, v, intrinsics):
    """Nearest fronto-parallel patch hit by a pixel ray, else the wall.

    The returned distance is the *Euclidean range* to the surface point, not
    its z-depth: that is what the inverse-square law measures, and reporting
    z-depth instead makes a flat wall render as a sphere of constant range.
    """
    x = (u - intrinsics.cx) / intrinsics.f_px
    y = (v - intrinsics.cy) / intrinsics.f_px
    scale = math.sqrt(1.0 + x * x + y * y)
    hits = []
    for p in scene.patches:
        x_m, y_m = x * p.depth_m, y * p.depth_m
        if (abs(x_m - p.centre_x_m) <= p.width_m / 2.0
                and abs(y_m - p.centre_y_m) <= p.height_m / 2.0):
            hits.append((p.depth_m * scale, p.albedo, p.tag))
    if hits:
        return min(hits)
    return scene.wall_depth_m * scale, scene.wall_albedo, "wall"


def render_scene(scene, intrinsics, cols=64, rows=36, gain=490.0,
                 ambient_gain=35.0, ambient_offset=8.0, noise_dn=1.0,
                 seed=0, mixed_albedo=False):
    """Forward-render ambient + torch frames; return grids and truth range."""
    rng = random.Random(seed)
    truth, ambient, flash, tags = [], [], [], []
    for r in range(rows):
        for c in range(cols):
            u = (c + 0.5) * intrinsics.image_w / cols
            v = (r + 0.5) * intrinsics.image_h / rows
            d, rho, tag = scene_ray(scene, u, v, intrinsics)
            if mixed_albedo and tag != "wall":
                rho = 0.30
            flux = gain * rho / (d * d)
            amb = ambient_gain * rho + ambient_offset + rng.gauss(0.0, noise_dn)
            fl = amb + flux + rng.gauss(0.0, noise_dn)
            truth.append(d)
            ambient.append(round(max(0.0, min(255.0, amb)), 3))
            flash.append(round(max(0.0, min(255.0, fl)), 3))
            tags.append(tag)
    return {"cols": cols, "rows": rows, "truth_depth_m": truth,
            "ambient": ambient, "flash": flash, "tags": tags,
            "scene": scene.name}


def calibrate_scene(frame, intrinsics, depth_m=None, sample="wall"):
    """Calibrate from a known-range flat patch (default: the wall).

    `depth_m` overrides the reference *range* (not z-depth); by default the
    median truth range of the sampled cells is used, which is what the median
    differential flux corresponds to.
    """
    idx = [i for i, t in enumerate(frame["tags"]) if t == sample]
    if not idx:
        idx = list(range(len(frame["tags"])))
    deltas = sorted(max(0.0, frame["flash"][i] - frame["ambient"][i])
                    for i in idx)
    ref_delta = statistics.median(deltas)
    known = depth_m
    if known is None:
        known = statistics.median(sorted(frame["truth_depth_m"][i] for i in idx))
    return FlashCalibration(reference_depth_m=known,
                            reference_delta_dn=max(ref_delta, 1e-3),
                            sigma_delta_dn=1.4, albedo_rel_sigma=0.05)


def ground_truth_cells(frame, intrinsics):
    centres = cell_centres(frame["cols"], frame["rows"], intrinsics)
    return [{"index": i, "u": u, "v": v, "depth_m": frame["truth_depth_m"][i]}
            for i, (u, v) in enumerate(centres)]


def depth_mape(cells, truth_cells):
    errs = [abs(c["depth_m"] - t["depth_m"]) / t["depth_m"] * 100.0
            for c, t in zip(cells, truth_cells)
            if c["valid"] and c["depth_m"] and t["depth_m"]]
    if not errs:
        return {"mape_pct": None, "median_pct": None, "max_pct": None,
                "cells": 0}
    errs.sort()
    return {"mape_pct": round(sum(errs) / len(errs), 3),
            "median_pct": round(statistics.median(errs), 3),
            "p90_pct": round(errs[int(0.9 * (len(errs) - 1))], 3),
            "max_pct": round(errs[-1], 3), "cells": len(errs)}


def analyse_scene(frame, intrinsics, pose=None, cal=None, mixed_albedo=False):
    """Full pipeline on a rendered scene; returns measured vs truth."""
    pose = pose or CameraPose()
    cal = cal or calibrate_scene(frame, intrinsics)
    cells = analyse_frame(frame["ambient"], frame["flash"],
                          frame["cols"], frame["rows"], cal, intrinsics)
    truth = ground_truth_cells(frame, intrinsics)
    points = cell_points(cells, frame["cols"], frame["rows"], intrinsics, pose)
    planes = segment_planes(points, frame["cols"], frame["rows"])
    targets = []
    for plane in planes:
        match = match_target(plane)
        if match["tag"]:
            targets.append(dict(match, plane_id=plane["plane_id"],
                                cells=plane["cells"], rms_m=plane["rms_m"]))
    return {
        "cells": cells, "truth": truth, "points": points, "planes": planes,
        "targets": targets, "accuracy": depth_mape(cells, truth),
        "calibration": cal.as_dict(), "valid_pct": round(valid_fraction(cells) * 100, 2),
        "ascii": ascii_depth(cells, frame["cols"], frame["rows"], width=64),
        "scene": frame["scene"], "mixed_albedo": mixed_albedo,
    }


def analyse_grid(ambient_cells, flash_cells, cols, rows, intrinsics, cal,
                 pose=None, level=0.95):
    """Analyse a caller-supplied ambient/flash pair (no synthetic truth).

    This is the browser path: the frontend already averages pixels into cells,
    so the backend only sees two flat grids. `cal` must come from a known-depth
    reference patch, otherwise every range is unanchored.
    """
    if len(ambient_cells) != cols * rows or len(flash_cells) != cols * rows:
        raise ValueError("ambient and flash grids must each hold cols*rows cells")
    pose = pose or CameraPose()
    cells = analyse_frame(ambient_cells, flash_cells, cols, rows, cal, intrinsics)
    points = cell_points(cells, cols, rows, intrinsics, pose)
    planes = segment_planes(points, cols, rows)
    targets = []
    for plane in planes:
        match = match_target(plane)
        if match["tag"]:
            targets.append(dict(match, plane_id=plane["plane_id"],
                                cells=plane["cells"], rms_m=plane["rms_m"]))
    cue = flash_cue(cells, cols, rows, intrinsics)
    return {
        "cells": cells, "points": points, "planes": planes, "targets": targets,
        "accuracy": None, "truth": None, "calibration": cal.as_dict(),
        "valid_pct": round(valid_fraction(cells) * 100, 2),
        "ascii": ascii_depth(cells, cols, rows, width=cols),
        "scene": None, "mixed_albedo": False,
        "flash_cue": asdict(cue) if cue else None,
        "level": level,
        "cols": cols, "rows": rows,
    }


def run_demo(cols=96, rows=54, mixed_albedo=False, seed=0, width=96):
    intr = CameraIntrinsics.from_fov(640, 480, 60.0)
    frame = render_scene(DEFAULT_SCENE, intr, cols=cols, rows=rows, seed=seed,
                         mixed_albedo=mixed_albedo)
    result = analyse_scene(frame, intr, mixed_albedo=mixed_albedo)
    result["ascii"] = ascii_depth(result["cells"], cols, rows, width=width)
    result["flash_cue"] = asdict(flash_cue(result["cells"], cols, rows, intr))
    result.update(cols=cols, rows=rows, level=0.95)
    return result


# ── CLI ────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="LiDar flash-photometric depth")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--rows", type=int, default=54)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-albedo", action="store_true")
    parser.add_argument("--ascii", action="store_true", help="print the depth map")
    parser.add_argument("--ansi", action="store_true", help="colour depth map")
    parser.add_argument("--json", default="flash_report.json")
    parser.add_argument("--full", action="store_true",
                        help="include the per-cell grid and truth in the report")
    args = parser.parse_args(argv)

    if not args.demo:
        parser.error("only --demo is supported on device (frames come from the browser)")

    result = run_demo(args.cols, args.rows, args.mixed_albedo, args.seed,
                      width=args.cols)
    acc = result["accuracy"]
    print("scene :", result["scene"])
    print("cal   :", json.dumps(result["calibration"]))
    print("cells : %d  valid %.1f %%  depth MAPE %s %%  median %s %%  p90 %s %%"
          % (len(result["cells"]), result["valid_pct"], acc["mape_pct"],
             acc["median_pct"], acc["p90_pct"]))
    print("planes: %d" % len(result["planes"]))
    for t in result["targets"]:
        amb = " (candidates %s)" % "/".join(t["candidates"]) if t["ambiguous"] else ""
        print("[TARGET DETECTED]: %s%s | length %.1f cm | +/- %.1f cm (95%% CI)"
              % (t["label"], amb, t["length_mm"] / 10.0, t["length_ci95_cm"]))
    if args.ansi:
        print("\n".join(ascii_depth(result["cells"], args.cols, args.rows,
                                    width=args.cols, ansi=True)))
    elif args.ascii:
        print("\n".join(result["ascii"]))
    report = {k: v for k, v in result.items()
              if k not in ("points", "cells", "truth")}
    if args.full:
        report.update(cells=result["cells"], truth=result["truth"])
    with open(args.json, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.json)
    return result


if __name__ == "__main__":
    main()
