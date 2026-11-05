"""lidarv2 premade fixtures — known objects rendered as synthetic observations.

Each fixture is a physical object with ground-truth dimensions. `synthesize`
projects it at a chosen depth to produce the exact bounding box a perfect
detector would report — the "known premade object picture". The benchmark then
runs the real pipeline on that observation and compares the emulated size to
the truth.
"""

import math
import random

from models import CameraIntrinsics, CameraPose, Observation

# tag, label, width_mm, height_mm, thickness_mm
PREMADE = [
    {"tag": "2x4", "label": "2x4 stud", "w_mm": 88.9, "h_mm": 38.1, "d_mm": 38.1},
    {"tag": "2x6", "label": "2x6 board", "w_mm": 139.7, "h_mm": 38.1, "d_mm": 38.1},
    {"tag": "4x4", "label": "4x4 post", "w_mm": 88.9, "h_mm": 88.9, "d_mm": 88.9},
    {"tag": "square_100", "label": "100 mm square tile", "w_mm": 100.0, "h_mm": 100.0, "d_mm": 10.0},
    {"tag": "cube_100", "label": "100 mm cube", "w_mm": 100.0, "h_mm": 100.0, "d_mm": 100.0},
    {"tag": "rect_200x100", "label": "200x100 mm panel", "w_mm": 200.0, "h_mm": 100.0, "d_mm": 20.0},
]


def by_tag(tag):
    for obj in PREMADE:
        if obj["tag"] == tag:
            return obj
    return None


def truth_dims(obj):
    return (obj["w_mm"], obj["h_mm"], obj["d_mm"])


def synthesize(obj, depth_m, intrinsics=None, pose=None, camera_height_m=1.5,
               noise_frac=0.0, seed=0, **features):
    """Project a known object at `depth_m` into a perfect bounding box."""
    intr = intrinsics or CameraIntrinsics.from_fov(720, 1280, 60.0)
    rng = random.Random(seed)
    f = intr.f_px

    px_w = obj["w_mm"] * f / (depth_m * 1000.0)
    px_h = obj["h_mm"] * f / (depth_m * 1000.0)
    if noise_frac:
        px_w *= 1.0 + rng.uniform(-noise_frac, noise_frac)
        px_h *= 1.0 + rng.uniform(-noise_frac, noise_frac)
    px_w = max(2.0, px_w)
    px_h = max(2.0, px_h)

    # Put the object's bottom edge on the floor so the ground cue agrees.
    depression = math.atan(camera_height_m / depth_m)
    v_bottom = intr.cy + f * math.tan(depression)
    center_y = max(px_h / 2.0, min(intr.image_h - px_h / 2.0, v_bottom - px_h / 2.0))

    if pose is None:
        pose = CameraPose(lat=45.0, lon=10.0, alt_m=50.0, heading_deg=0.0,
                          pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True)

    base = dict(
        texture_density=0.45, sharpness=0.45,
        convergence=0.0, arrowhead=0.0, specular=0.0, contrast=0.45,
        room="premade",
    )
    base.update(features)
    return Observation(
        object_px_width=px_w,
        object_px_height=px_h,
        center_x=intr.cx,
        center_y=center_y,
        intrinsics=intr,
        pose=pose,
        known_width_mm=obj["w_mm"],
        object_class=obj["tag"],
        camera_height_m=camera_height_m,
        **base,
    )
