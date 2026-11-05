"""lidarv2 triangulation — two-view ray intersection for size and position.

Given the same object seen from two camera poses, back-project each pixel to a
world ray and intersect the rays. The midpoint of the closest approach is the
3-D point; the distance between triangulated corners is the object's size.
"""

import math

from geospatial import (
    enu_to_geodetic,
    pixel_to_ray,
    ray_camera_to_enu,
    wgs84_radii,
)


def geodetic_to_enu(lat0, lon0, alt0, lat, lon, alt):
    r_north, r_east = wgs84_radii(lat0)
    north = math.radians(lat - lat0) * r_north
    east = math.radians(lon - lon0) * r_east * math.cos(math.radians(lat0))
    return east, north, alt - alt0


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _world_ray(pose, u, v, intr):
    ray = pixel_to_ray(u, v, intr)
    return ray_camera_to_enu(ray, pose.heading_deg, pose.pitch_deg, pose.roll_deg)


def closest_points(o1, d1, o2, d2):
    """Closest approach of two rays (returns t1, t2, midpoint, residual)."""
    w0 = (o1[0] - o2[0], o1[1] - o2[1], o1[2] - o2[2])
    a = _dot(d1, d1)
    b = _dot(d1, d2)
    c = _dot(d2, d2)
    d = _dot(d1, w0)
    e = _dot(d2, w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return None
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    p1 = (o1[0] + t1 * d1[0], o1[1] + t1 * d1[1], o1[2] + t1 * d1[2])
    p2 = (o2[0] + t2 * d2[0], o2[1] + t2 * d2[1], o2[2] + t2 * d2[2])
    mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
    residual = math.dist(p1, p2)
    return t1, t2, mid, residual


def triangulate_point(pose1, uv1, pose2, uv2, intr):
    """Triangulate one pixel correspondence into a georeferenced point."""
    o1 = (0.0, 0.0, 0.0)
    d1 = _world_ray(pose1, uv1[0], uv1[1], intr)
    o2 = geodetic_to_enu(pose1.lat, pose1.lon, pose1.alt_m,
                         pose2.lat, pose2.lon, pose2.alt_m)
    d2 = _world_ray(pose2, uv2[0], uv2[1], intr)
    cp = closest_points(o1, d1, o2, d2)
    if cp is None:
        return None
    t1, t2, mid, residual = cp
    lat, lon, alt = enu_to_geodetic(pose1.lat, pose1.lon, pose1.alt_m,
                                    mid[0], mid[1], mid[2])
    return {
        "east_m": mid[0], "north_m": mid[1], "up_m": mid[2],
        "depth1_m": t1, "depth2_m": t2,
        "residual_m": residual,
        "lat": lat, "lon": lon, "alt_m": alt,
    }


def _dist3(p, q):
    return math.sqrt((p["east_m"] - q["east_m"]) ** 2
                     + (p["north_m"] - q["north_m"]) ** 2
                     + (p["up_m"] - q["up_m"]) ** 2)


def triangulate_size(pose1, corners1, pose2, corners2, intr):
    """Triangulate four corner correspondences and recover width/height."""
    if len(corners1) != 4 or len(corners2) != 4:
        return None
    pts = []
    for a, b in zip(corners1, corners2):
        p = triangulate_point(pose1, a, pose2, b, intr)
        if p is None:
            return None
        pts.append(p)
    tl, tr, br, bl = pts
    return {
        "w_mm": round(_dist3(tl, tr) * 1000.0, 2),
        "h_mm": round(_dist3(tl, bl) * 1000.0, 2),
        "d_mm": None,
        "residual_m": round(max(p["residual_m"] for p in pts), 6),
        "corners": pts,
    }


def enu_direction_to_pixel(east, north, up, pose, intr):
    """Inverse projection: an ENU direction -> pixel, for constructing tests."""
    h = math.radians(pose.heading_deg)
    fwd = east * math.sin(h) + north * math.cos(h)
    right = east * math.cos(h) - north * math.sin(h)

    r = math.radians(pose.roll_deg)
    right, up = right * math.cos(r) + up * math.sin(r), -right * math.sin(r) + up * math.cos(r)

    p = math.radians(pose.pitch_deg)
    fwd, up = fwd * math.cos(p) + up * math.sin(p), -fwd * math.sin(p) + up * math.cos(p)

    x, y, z = right, -up, fwd
    if z <= 1e-9:
        return None
    return (intr.cx + intr.f_px * x / z, intr.cy + intr.f_px * y / z)
