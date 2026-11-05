import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from depth_engine import DepthEngine  # noqa: E402
from models import CameraIntrinsics, CameraPose, Observation  # noqa: E402


def make_obs(depth_m=1.0, known_mm=85.6, with_pose=False, **kwargs):
    intr = CameraIntrinsics.from_fov(640, 480, 60.0)
    px_w = known_mm * intr.f_px / (depth_m * 1000.0) if known_mm else 40.0
    pose = CameraPose(lat=0.0, lon=0.0, alt_m=0.0, heading_deg=0.0,
                      pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True) if with_pose else None
    return Observation(
        object_px_width=px_w,
        object_px_height=px_w * 0.45,
        center_x=intr.cx,
        center_y=intr.cy,
        intrinsics=intr,
        pose=pose,
        known_width_mm=known_mm,
        **kwargs,
    )


class TestPixelAndAngularCues(unittest.TestCase):
    def test_pixel_scaling_recovers_depth(self):
        obs = make_obs(depth_m=1.0)
        engine = DepthEngine()
        cues = {c.name: c for c in engine.cues(obs)}
        self.assertIn("pixel_scaling", cues)
        self.assertAlmostEqual(cues["pixel_scaling"].depth_m, 1.0, places=3)

    def test_angular_cue_agrees(self):
        obs = make_obs(depth_m=2.0)
        cues = {c.name: c for c in DepthEngine().cues(obs)}
        self.assertAlmostEqual(cues["angular_size"].depth_m, 2.0, places=2)

    def test_estimate_fuses_to_target(self):
        obs = make_obs(depth_m=3.0)
        est = DepthEngine().estimate(obs)
        self.assertAlmostEqual(est.depth_m, 3.0, places=2)
        self.assertLess(est.ci_low_m, est.depth_m)
        self.assertGreater(est.ci_high_m, est.depth_m)
        self.assertGreater(est.confidence, 0.5)

    def test_no_cues_raises(self):
        obs = make_obs(depth_m=1.0, known_mm=None)
        with self.assertRaises(ValueError):
            DepthEngine().estimate(obs)


class TestGroundCue(unittest.TestCase):
    def test_ground_plane_recovers_depth(self):
        # target 5 m on a flat floor, camera 1.5 m high
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        camera_height = 1.5
        target = 5.0
        depression = math.atan(camera_height / target)
        v_bottom = intr.cy + intr.f_px * math.tan(depression)
        height = 40.0
        obs = Observation(
            object_px_width=60.0, object_px_height=height,
            center_x=intr.cx, center_y=v_bottom - height / 2.0,
            intrinsics=intr,
            pose=CameraPose(lat=0, lon=0, alt_m=0, heading_deg=0,
                            pitch_deg=0, roll_deg=0, accuracy_m=5, has_gps=True),
            known_width_mm=None, camera_height_m=camera_height,
        )
        cues = {c.name: c for c in DepthEngine(camera_height_m=camera_height).cues(obs)}
        self.assertIn("ground_plane", cues)
        self.assertAlmostEqual(cues["ground_plane"].depth_m, target, delta=0.2)

    def test_no_ground_cue_above_horizon(self):
        intr = CameraIntrinsics.from_fov(640, 480, 60.0)
        obs = Observation(
            object_px_width=60.0, object_px_height=20.0,
            center_x=intr.cx, center_y=intr.cy - 100.0,
            intrinsics=intr,
            pose=CameraPose(pitch_deg=0.0),
            known_width_mm=None,
        )
        names = [c.name for c in DepthEngine().cues(obs)]
        self.assertNotIn("ground_plane", names)


class TestGridScan(unittest.TestCase):
    def test_scan_returns_ground_points(self):
        intr = CameraIntrinsics.from_fov(320, 240, 60.0)
        pose = CameraPose(lat=45.0, lon=10.0, alt_m=50.0, heading_deg=0.0,
                          pitch_deg=0.0, roll_deg=0.0, accuracy_m=5.0, has_gps=True)
        pts = DepthEngine(camera_height_m=1.5).grid_scan(pose, intr, cols=8, rows=6)
        self.assertGreater(len(pts), 0)
        for p in pts:
            self.assertLess(p.distance_m, 200.0)
            self.assertAlmostEqual(p.up_m, -1.5, places=6)
            self.assertAlmostEqual(p.alt_m, 48.5, places=4)


if __name__ == "__main__":
    unittest.main()
