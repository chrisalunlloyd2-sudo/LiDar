"""lidarv2 data models — plain dataclasses, stdlib only.

Coordinate conventions
----------------------
Camera frame : +x right, +y down, +z forward (pinhole).
World frame  : ENU — +E east, +N north, +U up (metres, local tangent plane).
Geodetic     : WGS84 latitude/longitude in degrees, altitude in metres.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics. `f_px` is focal length in pixels."""
    f_px: float
    cx: float
    cy: float
    image_w: int
    image_h: int
    fov_deg: float = 60.0

    @classmethod
    def from_fov(cls, image_w, image_h, fov_deg=60.0):
        f = (image_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        return cls(
            f_px=f,
            cx=image_w / 2.0,
            cy=image_h / 2.0,
            image_w=int(image_w),
            image_h=int(image_h),
            fov_deg=float(fov_deg),
        )


@dataclass
class CameraPose:
    """Camera pose from browser Geolocation + DeviceOrientation."""
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    heading_deg: float = 0.0   # 0 = north, clockwise, degrees
    pitch_deg: float = 0.0     # positive = looking up
    roll_deg: float = 0.0
    accuracy_m: float = 8.0    # GPS horizontal accuracy (1-sigma-ish)
    has_gps: bool = False


@dataclass
class Observation:
    """One detected object in one frame, with optional visual features."""
    object_px_width: float
    object_px_height: float
    center_x: float
    center_y: float
    intrinsics: CameraIntrinsics
    pose: Optional[CameraPose] = None
    known_width_mm: Optional[float] = None
    object_class: str = "unknown"
    camera_height_m: float = 1.5
    # Optional frontend-computed visual cues (normalised 0..1 unless noted).
    texture_density: Optional[float] = None
    sharpness: Optional[float] = None
    convergence: Optional[float] = None   # Ponzo / perspective-convergence evidence
    arrowhead: Optional[float] = None     # Müller-Lyer evidence
    specular: Optional[float] = None      # specular / mirror evidence
    contrast: Optional[float] = None
    room: str = "unknown"
    timestamp: Optional[float] = None


@dataclass
class CueEstimate:
    """A single monocular depth cue with its standard uncertainty."""
    name: str
    depth_m: float
    sigma_m: float
    weight: float = 1.0


@dataclass
class DepthEstimate:
    """Fused depth with a confidence interval."""
    depth_m: float
    ci_low_m: float
    ci_high_m: float
    level: float
    sigma_m: float
    confidence: float
    cues: List[CueEstimate] = field(default_factory=list)
    method: str = "inverse_variance"

    @property
    def ci_half_m(self):
        return (self.ci_high_m - self.ci_low_m) / 2.0


@dataclass
class IllusionFlag:
    code: str
    label: str
    severity: float


@dataclass
class IllusionReport:
    risk: float
    flags: List[IllusionFlag]
    rejected: bool
    reason: str = ""


@dataclass
class GeoPoint:
    """Georeferenced point in WGS84 plus its local ENU offset."""
    lat: float
    lon: float
    alt_m: float
    east_m: float
    north_m: float
    up_m: float
    distance_m: float
    bearing_deg: float
    radius_m: float
    has_gps: bool = True


@dataclass
class GeoScanPoint:
    """One ground-plane point from an emulated LiDAR scan."""
    lat: float
    lon: float
    alt_m: float
    east_m: float
    north_m: float
    up_m: float
    distance_m: float
    u: int
    v: int
    radius_m: float
