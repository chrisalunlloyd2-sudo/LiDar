# lidarv2 — emulative LiDAR + spatial inventory

`lidarv2` turns a phone camera into an **emulated LiDAR**: it estimates monocular
depth, fuses several cues into a **confidence interval**, corrects the **5 optical
illusions**, classifies **squares / rectangles / cubes / cuboids**, fits a 3-D model
to the observed corners, triangulates across two views, projects to **WGS84
lat/lon/alt**, and builds a queryable **geotagged inventory** (e.g. "2×4 in
basement B, 88.9 × 38.1 mm"). The GUI is fully modular: every page is a separate
file memory-mapped from `gui_pages/`.

Pure Python stdlib. No pip installs, no neural weights, no external services.

## v2.1 highlights

- **5-illusion registry + correction math** (`illusions.py`), applied to the
  emulated size before it is stored.
- **Shape classifier** (`shapes.py`) for square · rectangle · cube · cuboid ·
  lumber.
- **Model fitting** (`fit_engine.py`): Nelder-Mead reprojection fit for
  rectangles (width, height, yaw) and cuboids (width, height, thickness, yaw).
- **Two-view triangulation** (`triangulate.py`) for points and sizes.
- **Accuracy benchmark** (`accuracy.py` + `fixtures/`): the mean absolute
  percentage error to reality is the number to minimise.
- **mmap modular GUI** (`page_store.py` + `gui_pages/`): the shell loads one
  fragment per page from `/page/<name>`.

## Architecture

```
camera frame ──► browser canvas analysis ──► POST /api/observe
                                                   │
        ┌──────────────────┬───────────────────────┼───────────────────────┐
        ▼                  ▼                       ▼                       ▼
 depth_engine.py    illusion_filter.py        illusions.py            geospatial.py
 pixel / angular /  reject on cue conflict    5-illusion correction    pixel ray → ENU →
 ground / texture   Ponzo, mirror, haze…      applied to dimensions    WGS84 + ± radius
        │                                          │
        ▼                                          ▼
 uncertainty.py                            shapes.py / inventory.py
 inverse-variance fusion                   shape + tag + emulated dims
 + between-cue variance                    fit_engine.py (optional)
        │                                  triangulate.py (two views)
        └──────────────────► store.py (SQLite) ──► GET /api/inventory
```

### Modules

| File | Responsibility |
|------|----------------|
| `models.py` | dataclasses: intrinsics, pose, observation, estimate, flags, geo |
| `depth_engine.py` | 4 depth cues + ground-plane grid scan |
| `uncertainty.py` | inverse-variance fusion, between-cue variance, CI, bootstrap |
| `illusion_filter.py` | accept/reject on illusion risk (cue conflict, mirror, haze…) |
| `illusions.py` | the canonical **5 illusions** + multiplicative correction math |
| `shapes.py` | square / rectangle / cube / cuboid / lumber classifier |
| `fit_engine.py` | Nelder-Mead rectangle/cuboid reprojection fit |
| `triangulate.py` | two-view ray intersection (point + size) |
| `accuracy.py` | percent error, MAPE, CI coverage, benchmark |
| `fixtures/premade_objects.py` | known objects projected into perfect observations |
| `geospatial.py` | pixel→ray→ENU→WGS84, haversine, uncertainty radius, ground scan |
| `inventory.py` | physical tag catalog, dimension matching, emulated dimensions |
| `store.py` | SQLite map points + aggregated inventory |
| `page_store.py` | **mmap** page store for the modular GUI |
| `server.py` | stdlib `ThreadingHTTPServer` JSON API + page routes |
| `gui_pages/` | shell + one fragment per page + `app.js` |

## The 5 optical illusions

| id | kind | correction |
|----|------|------------|
| `ponzo` | geometry | `factor = 1 − 0.30 · convergence` |
| `muller_lyer` | geometry | `factor = 1 − 0.20 · arrowhead` |
| `vertical_horizontal` | geometry | `factor = 1 − 0.12 · clamp((aspect−1)/2)` |
| `ebbinghaus` | geometry | `factor = 1 − 0.15 · clamp((|d_pixel−d_angular|/max − 0.10)/0.40)` |
| `checker_shadow` | light | `factor = 1 − 0.10 · (0.5·(1−contrast) + 0.5·specular)` |

The factors multiply into one `combined_factor` (clamped 0.30–1.20). The server
reports raw dimensions, corrected dimensions, and the correction percent.

## Confidence intervals

For cues `d_i ± σ_i`, inverse-variance weights `w_i = 1/σ_i²` give the point
estimate; a method-of-moments between-cue variance widens the interval when
informative cues disagree:

```
σ² = 1/Σw_i + Var_between(informative cues)
CI = μ ± z(level) · σ
```

Cues with negligible inverse-variance weight (e.g. a far-ground cue when an
object is held above the floor) are excluded so they cannot poison the interval.

## Accuracy benchmark

`POST /api/benchmark` runs the full pipeline over all premade fixtures and
returns the **mean absolute percentage error** and **CI coverage**:

```
mean MAPE 0.446 % · max error 2.887 % · CI coverage 100 % / 100 %
```

The premade 2×4 emulates to **88.9 × 38.1 × 38.1 mm at 0.0 % error** with a
depth CI of `[1.878, 2.123] m` at 2 m. Use `noise_frac` to stress the estimator.

## Run

```bash
cd ~/lidarv2
python3 server.py                 # http://0.0.0.0:8790, db ./lidarv2.db
# or
python3 server.py --memory --port 8790
# or
bash run.sh
```

Open `http://<phone-ip>:8790` in Chrome. Allow camera + location + motion.

1. Pick a preset (or type a known width in mm).
2. Hold the object in the reticle and tap **Calibrate**.
3. The depth card shows `depth ± half-width (95%)`, the corrected dimensions and
   the illusion correction percent.
4. **Fit model** fits a rectangle/cuboid to the detected corners.
5. **Scan ground** sweeps a ground-plane grid into the local ENU map.
6. **Premade object** emulates a known fixture and reports its % error.
7. **Benchmark** reports the MAPE over all fixtures.

## API

| Method | Path | Body / query | Returns |
|--------|------|--------------|---------|
| GET | `/` | — | mmap shell |
| GET | `/page/<name>` | — | one mmap page fragment |
| GET | `/api/pages` | — | mapped page list |
| GET | `/api/health` | — | status + db + page count |
| GET | `/api/config` | — | version + tag catalog |
| GET | `/api/illusions` | — | the 5-illusion registry |
| POST | `/api/observe` | observation JSON | depth+CI, 5 illusions, geo, item |
| POST | `/api/scan` | intrinsics + pose + grid | ground point cloud |
| POST | `/api/fit` | `corners_px`, `depth_m`, `kind` | rectangle/cuboid fit |
| POST | `/api/triangulate` | `pose1/pose2`, `uv1/uv2` or `corners1/2` | point / size |
| POST | `/api/premade` | `{tag, depth_m}` | emulated vs truth + error % |
| POST | `/api/benchmark` | `{depth_m, noise_frac}` | MAPE + coverage |
| POST | `/api/demo` | `{room}` | seeded inventory |
| GET | `/api/map` | `?room=&tag=&limit=` | accepted map points |
| GET | `/api/inventory` | `?room=&tag=` | items + summary |
| POST | `/api/reset` | `{}` | clears map + inventory |

## Tests

```bash
cd ~/lidarv2
python3 -m pytest tests -q      # 129 tests
```

## Known limitations

- Monocular depth is ill-posed; accuracy is best for known-size and
  floor-contact objects within a few metres.
- The ground scan assumes a flat floor at `camera_height_m`.
- Thickness is not observable from one view; it comes from the catalog or a
  cuboid fit.
- GPS altitude and compass heading vary by device; calibrate heading against a
  known bearing for georeferencing work.
- This is a statistical inference engine, not a trained depth network.
