# lidarv2 — emulative LiDAR + spatial inventory

`lidarv2` turns a phone camera into an **emulated LiDAR**: it estimates monocular
depth for detected objects, fuses several independent cues into a **confidence
interval**, projects the result to **WGS84 latitude / longitude / altitude**,
**filters optical illusions**, and builds a queryable **geotagged inventory**
(e.g. "2×4 in basement B, 88.9 × 38.1 mm, at 45.000000, 10.000000").

The frontend is a camera (`getUserMedia` + canvas). The backend is a pure-stdlib
Python inference service — no pip installs, no neural weights, no external
services.

## Why not a neural depth net?

This device is 32-bit ARM Termux with no `torch`/`tensorflow`/`PIL`/`OpenCV`.
`lidarv2` therefore uses an explicit **multi-cue statistical inference engine**
instead of a learned model. It is honest about that: every estimate carries an
interval, and cues that disagree are treated as evidence of illusion rather than
averaged away.

## Architecture

```
camera frame ──► browser canvas analysis ──► POST /api/observe
                                                   │
                        ┌──────────────────────────┼──────────────────────────┐
                        ▼                          ▼                          ▼
                 depth_engine.py           illusion_filter.py           geospatial.py
              pixel / angular /            cue conflict, Ponzo,       pixel ray → ENU →
              ground / texture             Müller-Lyer, mirror,       WGS84 + ± radius
                        │                  haze, foreshortening
                        ▼
                 uncertainty.py
              inverse-variance fusion +
              between-cue variance →
              depth ± CI + confidence
                        │
                        ▼
                 inventory.py ──► store.py (SQLite) ──► GET /api/inventory
```

### Modules

| File | Responsibility |
|------|----------------|
| `models.py` | dataclasses for intrinsics, pose, observations, estimates, flags, geo points |
| `depth_engine.py` | four depth cues + ground-plane grid scan |
| `uncertainty.py` | inverse-variance fusion, between-cue variance, CI, bootstrap |
| `illusion_filter.py` | scores and ranks optical-illusion signatures; accepts/rejects |
| `geospatial.py` | pixel→ray→ENU→WGS84, haversine, uncertainty radius, ground scan |
| `inventory.py` | physical tag catalog, dimension matching, emulated dimensions |
| `store.py` | SQLite map points + aggregated inventory |
| `server.py` | stdlib `ThreadingHTTPServer` JSON API + static frontend |

## Run

```bash
cd ~/lidarv2
python3 server.py                 # http://0.0.0.0:8790, db ./lidarv2.db
# or
python3 server.py --memory --port 8790
```

Open `http://<phone-ip>:8790` in Chrome. Allow camera + location + motion.

1. Pick an object preset (or type a known width in mm).
2. Hold the object inside the reticle and tap **Calibrate**.
3. The depth card updates live with `depth ± half-width (95%)`.
4. **Scan ground** sweeps a ground-plane grid into the local ENU map.
5. **Seed demo** injects four synthetic items (2×4, 2×6, paint can, box) so the
   inventory and map are non-empty without a physical camera.

## API

| Method | Path | Body / query | Returns |
|--------|------|--------------|---------|
| GET | `/api/health` | — | status + db |
| GET | `/api/config` | — | version + tag catalog |
| POST | `/api/observe` | observation JSON | depth + CI, illusions, geo, item |
| POST | `/api/scan` | intrinsics + pose + grid | ground point cloud |
| POST | `/api/demo` | `{room}` | seeded inventory |
| GET | `/api/map` | `?room=&tag=&limit=` | accepted map points |
| GET | `/api/inventory` | `?room=&tag=` | items + summary |
| POST | `/api/reset` | `{}` | clears map + inventory |

### Observation payload

```json
{
  "object_px_width": 49.3, "object_px_height": 22.2,
  "center_x": 320, "center_y": 240,
  "intrinsics": {"f_px": 554.3, "cx": 320, "cy": 240,
                 "image_w": 640, "image_h": 480, "fov_deg": 60},
  "pose": {"lat": 45.0, "lon": 10.0, "alt_m": 50.0,
           "heading_deg": 0, "pitch_deg": 0, "roll_deg": 0,
           "accuracy_m": 5, "has_gps": true},
  "known_width_mm": 88.9, "object_class": "2x4",
  "camera_height_m": 1.5, "room": "basement B", "level": 0.95,
  "texture_density": 0.4, "convergence": 0.05, "specular": 0.02, "contrast": 0.4
}
```

## Confidence intervals

For cues `d_i ± σ_i`, the fused estimate uses inverse-variance weights
`w_i = 1/σ_i²`. The interval is

```
σ² = 1/Σw_i  +  Var_between(informative cues)
CI = μ ± z(level) · σ
```

The between-cue term is a method-of-moments random effect: genuine disagreement
widens the interval. Cues whose inverse-variance weight is negligible (e.g. a
far-ground cue when an object is held above the floor) are excluded from both the
point estimate's conflict term and the illusion test, so they cannot poison the
result.

## Optical illusions filtered

| Code | Signature |
|------|-----------|
| `cue_conflict` | cues disagree beyond their error bars (hard reject past tolerance) |
| `size_constancy` | known-size cue fights the angular cue |
| `ponzo` | converging perspective lines inflate apparent size |
| `muller_lyer` | arrowhead terminators distort length |
| `reflection` | specular/mirror surface shows a false 3-D image |
| `aerial_perspective` | low-contrast haze flattens depth |
| `foreshortening` | object nearly edge-on |

Flags are ranked by severity; the noisy-OR risk rejects an observation when it
exceeds `0.65` (or on a hard cue conflict).

## Tests

```bash
cd ~/lidarv2
python3 -m pytest tests -q
# or
python3 -m unittest discover -s tests -v
```

## Known limitations

- Monocular depth is ill-posed; accuracy is best for known-size objects and
  floor-contact objects within a few metres.
- The ground scan assumes a flat floor at `camera_height_m` below the camera.
- GPS altitude and compass heading vary by device; calibrate heading against a
  known bearing for georeferencing work.
- Thickness is not observable from a single view and is taken from the catalog
  when the tag matches.
- This is an inference *engine*, not a trained depth network (see above).
