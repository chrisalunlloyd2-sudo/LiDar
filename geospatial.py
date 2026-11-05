"""lidarv2 geospatial engine — pixel rays -> ENU -> WGS84.

Pure stdlib. The projection is a local tangent-plane (ENU) approximation,
accurate to a few centimetres over the tens-of-metres range a phone camera
can actually resolve.
"""

import math

from models import CameraIntrinsics, CameraPose, GeoPoint, GeoScanPoint

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_MEAN_R = 6371008.8


def wgs84_radii(lat_deg):
    """Return (meridional, prime-vertical) radii of curvature in metres."""
    phi = math.radians(lat_deg)
    s2 = math.sin(phi) ** 2
    denom = 1.0 - WGS84_E2 * s2
    r_north = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)
    r_east = WGS84_A / math.sqrt(denom)
    return r_north, r_east


def enu_to_geodetic(lat0, lon0, alt0, east_m, north_m, up_m):
    """Local ENU offset (m) -> WGS84 (lat, lon, alt)."""
    r_north, r_east = wgs84_radii(lat0)
    lat = lat0 + math.degrees(north_m / r_north)
    coslat = max(math.cos(math.radians(lat0)), 1e-9)
    lon = lon0 + math.degrees(east_m / (r_east * coslat))
    alt = alt0 + up_m
    return lat, lon, alt


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_MEAN_R * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2 (0 = north, clockwise)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def pixel_to_ray(u, v, intr):
    """Pixel -> camera-frame ray (x right, y down, z forward)."""
    x = (u - intr.cx) / intr.f_px
    y = (v - intr.cy) / intr.f_px
    return (x, y, 1.0)


def _normalize(vec):
    n = math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
    if n <= 0.0:
        return (0.0, 0.0, -1.0)
    return (vec[0] / n, vec[1] / n, vec[2] / n)


def ray_camera_to_enu(ray, heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Rotate a camera-frame ray into the ENU world frame.

    heading 0 = north, clockwise positive; pitch positive = looking up;
    roll positive = clockwise about the forward axis.
    """
    x, y, z = ray
    # camera -> vehicle-local (forward, right, up)
    fwd, right, up = z, x, -y

    # pitch about the right axis
    p = math.radians(pitch_deg)
    cp, sp = math.cos(p), math.sin(p)
    fwd, up = fwd * cp - up * sp, fwd * sp + up * cp

    # roll about the forward axis
    r = math.radians(roll_deg)
    cr, sr = math.cos(r), math.sin(r)
    right, up = right * cr - up * sr, right * sr + up * cr

    # heading about the up axis
    h = math.radians(heading_deg)
    ch, sh = math.cos(h), math.sin(h)
    north = fwd * ch - right * sh
    east = fwd * sh + right * ch
    return _normalize((east, north, up))


def enu_direction_to_bearing_deg(east, north):
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def uncertainty_radius_m(depth_m, sigma_depth_m, sigma_angle_deg, gps_accuracy_m):
    """Combine depth, angular and GPS error into a 1-sigma horizontal radius."""
    sigma_ang = math.radians(sigma_angle_deg)
    return math.sqrt(
        max(sigma_depth_m, 0.0) ** 2
        + (max(depth_m, 0.0) * sigma_ang) ** 2
        + max(gps_accuracy_m, 0.0) ** 2
    )


def project_point(pose, intr, u, v, depth_m, sigma_depth_m=0.0, sigma_angle_deg=2.0):
    """Project a pixel at a known depth to a georeferenced point."""
    ray = pixel_to_ray(u, v, intr)
    e, n, up = ray_camera_to_enu(ray, pose.heading_deg, pose.pitch_deg, pose.roll_deg)
    east = depth_m * e
    north = depth_m * n
    up_m = depth_m * up
    lat, lon, alt = enu_to_geodetic(pose.lat, pose.lon, pose.alt_m, east, north, up_m)
    distance = math.sqrt(east * east + north * north)
    bearing = enu_direction_to_bearing_deg(east, north)
    radius = uncertainty_radius_m(depth_m, sigma_depth_m, sigma_angle_deg, pose.accuracy_m)
    return GeoPoint(
        lat=lat, lon=lon, alt_m=alt,
        east_m=east, north_m=north, up_m=up_m,
        distance_m=distance, bearing_deg=bearing,
        radius_m=radius, has_gps=pose.has_gps,
    )


def ground_plane_point(pose, intr, u, v, camera_height_m, sigma_angle_deg=2.0):
    """Intersect a pixel ray with the flat ground plane below the camera.

    Returns None when the ray points at or above the horizon, or when the
    intersection is implausibly far away.
    """
    ray = pixel_to_ray(u, v, intr)
    e, n, up = ray_camera_to_enu(ray, pose.heading_deg, pose.pitch_deg, pose.roll_deg)
    if up >= -1e-6:
        return None
    t = camera_height_m / (-up)  # distance along the unit ray
    if t <= 0.0 or t > 200.0:
        return None
    east = t * e
    north = t * n
    up_m = -camera_height_m
    lat, lon, alt = enu_to_geodetic(pose.lat, pose.lon, pose.alt_m, east, north, up_m)
    distance = math.sqrt(east * east + north * north)
    radius = uncertainty_radius_m(t, t * 0.05, sigma_angle_deg, pose.accuracy_m)
    return GeoScanPoint(
        lat=lat, lon=lon, alt_m=alt,
        east_m=east, north_m=north, up_m=up_m,
        distance_m=distance, u=int(u), v=int(v), radius_m=radius,
    )


def scan_ground(pose, intr, camera_height_m=1.5, cols=24, rows=16, sigma_angle_deg=2.0):
    """Emulated LiDAR scan: a grid of rays intersected with the ground plane."""
    pts = []
    for r in range(max(1, rows)):
        v = (r + 0.5) * intr.image_h / rows
        for c in range(max(1, cols)):
            u = (c + 0.5) * intr.image_w / cols
            gp = ground_plane_point(pose, intr, u, v, camera_height_m, sigma_angle_deg)
            if gp is not None:
                pts.append(gp)
    return pts
