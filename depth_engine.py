"""lidarv2 depth engine — multi-cue monocular depth.

Cues implemented (each returns a CueEstimate with its own uncertainty):

  pixel_scaling   d = K * f / p          (known physical width + calibrated f)
  angular_size    d = K / (2 tan(theta/2)) with theta from the horizontal FOV
  ground_plane    d = h_cam / tan(depression angle of the ground-contact edge)
  texture_prior   weak log-texture prior (advisory; deliberately high sigma)

The ground-plane cue is what makes the emulated LiDAR scan possible: for any
pixel whose ray points below the horizon we can intersect the flat-ground
assumption and get a depth even without knowing the object's size.
"""

import math

from models import CueEstimate, Observation
from uncertainty import fuse_cues


class DepthEngine:
    def __init__(self, fov_deg=60.0, camera_height_m=1.5):
        self.fov_deg = float(fov_deg)
        self.camera_height_m = float(camera_height_m)

    # ── individual cues ────────────────────────────────────────────────
    def _pixel_scaling_cue(self, obs):
        p = obs.object_px_width
        f = obs.intrinsics.f_px
        if p <= 0 or f <= 0 or not obs.known_width_mm:
            return None
        d_m = (obs.known_width_mm * f) / p / 1000.0
        # half-pixel quantisation + 2% calibration/rounding drift
        sigma = max(0.01, d_m * (0.5 / p + 0.02))
        return CueEstimate("pixel_scaling", d_m, sigma, 1.0)

    def _angular_cue(self, obs):
        p = obs.object_px_width
        if p <= 0 or not obs.known_width_mm:
            return None
        fov = obs.intrinsics.fov_deg or self.fov_deg
        f_eff = (obs.intrinsics.image_w / 2.0) / math.tan(math.radians(fov) / 2.0)
        if f_eff <= 0:
            return None
        theta = 2.0 * math.atan(p / (2.0 * f_eff))
        if theta <= 0:
            return None
        d_m = (obs.known_width_mm / (2.0 * math.tan(theta / 2.0))) / 1000.0
        sigma = max(0.01, d_m * 0.05)
        return CueEstimate("angular_size", d_m, sigma, 0.8)

    def _ground_cue(self, obs):
        if obs.pose is None:
            return None
        # ground-contact edge = bottom of the detected bounding box
        v_bottom = obs.center_y + obs.object_px_height / 2.0
        f = obs.intrinsics.f_px
        if f <= 0:
            return None
        elevation = math.atan2(v_bottom - obs.intrinsics.cy, f)
        depression = elevation - math.radians(obs.pose.pitch_deg)
        if depression <= math.radians(0.5):
            return None  # at or above the horizon
        d_m = obs.camera_height_m / math.tan(depression)
        if d_m <= 0 or d_m > 120.0:
            return None
        return CueEstimate("ground_plane", d_m, max(0.05, 0.25 * d_m), 0.6)

    def _texture_cue(self, obs):
        if obs.texture_density is None:
            return None
        t = max(0.0, min(1.0, float(obs.texture_density)))
        d_m = 0.8 + 6.0 * t
        return CueEstimate("texture_prior", d_m, 0.6 * d_m, 0.15)

    # ── public API ─────────────────────────────────────────────────────
    def cues(self, obs: Observation):
        out = []
        for cue in (
            self._pixel_scaling_cue(obs),
            self._angular_cue(obs),
            self._ground_cue(obs),
            self._texture_cue(obs),
        ):
            if cue is not None:
                out.append(cue)
        return out

    def estimate(self, obs: Observation, level=0.95):
        return fuse_cues(self.cues(obs), level=level)

    def grid_scan(self, pose, intr, camera_height_m=None, cols=24, rows=16):
        """Ground-plane point cloud for the emulated LiDAR scan."""
        from geospatial import scan_ground

        ch = self.camera_height_m if camera_height_m is None else float(camera_height_m)
        return scan_ground(pose, intr, ch, cols=cols, rows=rows)
