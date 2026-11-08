# lidarv2 — emulative LiDAR + spatial inventory

[![CI](https://github.com/chrisalunlloyd2-sudo/LiDar/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisalunlloyd2-sudo/LiDar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ffaa.svg)](LICENSE)

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

## v2.2 — push the boundaries, then snip

`arena/` deliberately over-develops the math: **17 depth fusers**, **11 size
estimators**, **3 illusion models** (none / canonical5 / extended10) and **5
triangulators**. `python3 -m arena.arena` benchmarks every method across the
premade fixtures, ranks by MAPE (accuracy first), and writes
`arena_report.json` + `arena_report_survivors.json`.

Arena winners (kept):

| stage | survivors |
|-------|-----------|
| fusers | cauchy · inverse_variance · tukey_biweight · student_t · huber |
| sizes | angular · similar_triangles · trig · pixel_scale · median_multi · homography |
| illusions | canonical5 · none |
| triangulators | ransac · midpoint |

Snipped (much worse than baseline, or more than 5× slower):

| stage | pruned |
|-------|--------|
| fusers | kalman · dempster_shafer · ransac · bayesian · particle · weighted_mean · bootstrap · harmonic_mean · geometric_mean · trimmed_mean · winsorized · median |
| sizes | cross_ratio · vanishing_point · weighted_multi · ground_plane |
| illusions | extended10 (over-corrects: 27.9 % vs 15.1 % MAPE) |
| triangulators | least_squares (1.1 s) · parallax · weighted |

The production pipeline already uses survivors: inverse-variance fusion,
canonical 5 illusions, pixel-scale sizing, midpoint triangulation.

## v2.3 — parallel arena + multi-view refinement

`python3 -m arena.parallel_arena --workers 4` farms each fuser and size
estimator to a process pool (threads are useless under the GIL). Quick arena:
**525 ms vs 1,145 ms serial = 2.18×** on 4 workers.

`python3 -m arena.multiview --views 15 --baseline 0.2 --noise 0.3 --depth 1.5`
simulates 15 offset views and compares refinement strategies. The hard case is
the 2×6's 38.1 mm height:

| method | 2×6 height error |
|--------|------------------|
| temporal median | 2.370 % |
| aligned corners | 0.907 % |
| **3-D corner average** | **0.358 %** (seed 0) |
| RANSAC triangulation | 1.391 % |

3-D MAPE over all fixtures: **0.317 %** at seed 0. The 3-D average triangulates
each corner across every view pair, then averages before measuring — removing
both the max/min order-statistic bias and the per-view noise.

That table is **one noise draw**, and a lucky one: over 16 seeds the same
configuration gives a median 2×6-height error of **0.747 %** (p90 1.498 %,
3-D MAPE median 0.583 %). v2.4 therefore reports medians over seeds rather
than a single draw, and adds a pair guard that drops degenerate view pairs
before they can drag the average (`pair_is_usable`).

## v2.4 — active view planning (where to stand)

v2.3 fused a *fixed* set of views. v2.4 asks the prior question: which lateral
offsets can the camera actually observe, and which of them should it visit?

`fov_cap_m` derives the visible arc from the intrinsics, the depth and the
object width. For a 2×6 at 1.5 m with a 60° FOV it is **±0.692 m** — the v2.3
sweep used offsets out to 2.8 m, where the object is far outside the image.
`simulate_offset_view` drops any view whose corners leave the frame, so a view
you cannot take no longer counts.

`python3 -m arena.planner --seeds 16` compares four placement strategies at
equal budgets and reports the median over 16 noise draws. 2×6 height error:

| strategy | 2 views | 4 views | 6 views | 8 views | views to ≤0.75 % |
|----------|---------|---------|---------|---------|------------------|
| **even arc** | **1.804 %** | **1.186 %** | **0.780 %** | **0.702 %** | **8** |
| alternating extremes | 1.738 % | 1.582 % | 0.845 % | 0.731 % | 8 |
| random offsets | 5.230 % | 2.051 % | 0.987 % | 0.761 % | never |
| anchor only | — | — | — | — | never |

All-fixture MAPE for the even arc: 0.709 / 0.451 / 0.330 / 0.332 %. Two
findings matter:

- **Coverage beats one long baseline.** Under multi-view averaging the error is
  driven by the spread of the offsets. Alternating the two extremes wastes
  views on near-duplicate baselines (33 % worse at 4 views), and random
  placement clusters by chance — up to 73 % worse than an even arc at 4 views,
  still behind at 8.
- **Zero baseline cannot triangulate at all.** Repeating the anchor view leaves
  the ray intersection singular — every anchor-only cell fails to solve.

The planner also predicts the error before measuring it. With `n` views at
offsets `b_i`, depth `z`, focal `f` and pixel noise `sigma_px`:

```
pixel  = 100 * 1.1 * sigma_px / (h_px * sqrt(n))
triang = 100 * 1.7 * sigma_z / z,   sigma_z = z^2 * (sigma_px/f) / sqrt(sum b_i^2)
error% = hypot(pixel, triang)
```

The two constants were fitted over 90 plan/fixture/budget combinations.
On the benchmark the realised/predicted ratio has median **0.934** and rms
log-ratio **0.127** — the model is calibrated, and `/api/plan` reports it per
view so the frontend can show the expected improvement before you move.

Views are acquired from the inside out (`acquisition_order`), so an early stop
after two or three views still carries most of the information.

## v2.5 — torch-pulse depth (flash photometric)

`flash_lidar.py` adds the active-light path: pulse the torch, subtract the
ambient frame, and invert the square law.

```
ambient I0 + torch I1  ->  D = max(0, I1 - I0)   differential map
cell mean              ->  S                     additive flux
reference d0, S0       ->  d = d0 * sqrt(S0/S)   inverse-square range
N frames               ->  inverse-variance per cell + spread
grid + intrinsics      ->  3-D ENU/WGS84, PCA planes, catalog match
```

The flash enters the engine as a first-class `CueEstimate` named
`flash_falloff`, so `uncertainty.fuse_cues` and its confidence intervals apply
unchanged. `POST /api/flash/observe` returns the fused depth beside the
monocular cues.

Measured on the demo scene (96 × 54 = 5,184 cells, 16 seeds):

| metric | value |
|--------|-------|
| depth MAPE | **1.93 %** (p90 1.96 %) |
| per-cell median | 1.62 % |
| per-cell p90 | 4.12 % |
| targets found | 2×6 **and** 2×8, 16/16 seeds |
| runtime | ~0.49 s per frame |

Face widths carry the cell quantisation honestly: the 2×6 measures
148.6 mm against a true 139.7 mm (**+6.4 %**, 95 % CI ±80.7 mm) and the 2×8
189.9 mm against 184.2 mm (**+3.1 %**, ±136.8 mm). Matching abstains instead of
guessing: a face must have a decisive catalog width, an elongated face cannot
be a can lid, and a card / sheet / brick must match in **both** extents — a
30 cm strip no longer passes as a credit card.

Mixed albedo is the documented failure. Setting the object faces to 0.30 albedo
raises MAPE to **3.94 %** (2× the clean run) and the near 2×6 drops out of the
catalog 16/16, because its 140 mm face reads ~103 mm. The far 2×8 still
matches — but its range is 35 % too far, so a label is not a measurement.

The forward model had a real bug worth recording: it fed z-depth into the
square law, so a flat wall rendered as a **sphere of constant range** (plane RMS
89 mm, offset 1.85 m). The renderer now uses Euclidean range along the ray;
the same wall fits at 2.000 m with 17 mm RMS.

Calibration needs the reference range and differential flux of a flat patch
plus a per-cell noise estimate. Reflectance is unobservable from one
flash/ambient pair, so `albedo_rel_sigma` widens the depth sigma rather than
pretending the ambiguity away. `python3 flash_lidar.py --demo` prints the grid
and writes `flash_report.json`.

### Live capture (browser)

The **Torch-pulse depth** card runs the ping from the camera. **1 · Ambient**
scrubs the chosen number of frames (default 3, selectable 1/3/5/7) and keeps the
per-cell median, so one auto-exposure step or hand shake cannot move the grid;
**2 · Torch** repeats it with the torch on; **Solve ping** anchors the
inverse-square gauge on the median differential of the centre box at the
reference distance you type.

Where the browser exposes `torch` on the video track the app switches it; most
Chrome/Android builds do not, so toggle the phone torch by hand between the two
captures. Auto-exposure is the enemy — if the gain changes between phases the
differential collapses, and the solver refuses a ping whose centre differential
is under 0.5 DN instead of inventing a range. The scrubbed grids go to
`/api/flash` for the map and, with a live monocular observation, to
`/api/flash/observe` for the fused depth ± CI.

The reference number is the whole gauge: see [CALIBRATION.md](CALIBRATION.md) for
the healthy DN window, the synthetic sweep, the device procedure and the failure
modes.

## Hardware limits

`python3 -m arena.hardware_probe --processes` measured on the Android 13 /
armv8l 32-bit device:

| metric | value |
|--------|-------|
| cores | 8 |
| RAM | 3.6 GB total / ~575 MB available |
| single-core | ~0.5–0.66 M ops/s |
| thread scaling | 0.54–0.87× (GIL-bound) |
| process scaling | 1.52× |
| fusion throughput | ~3.6–4.0 k/s |
| fusion peak memory | ~0.001 MB |

Budget: `recommended_threads = 1`, `recommended_processes = 8`,
`gil_bound = true` — use processes (or native code) for parallel work, not
Python threads.

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
| `arena/depth_fusers.py` | 17 depth-fusion algorithms |
| `arena/size_estimators.py` | 11 size estimators |
| `arena/illusion_models.py` | none / canonical5 / extended10 |
| `arena/triangulators.py` | 5 two-view triangulators |
| `arena/arena.py` | rank every method, score, snip |
| `arena/parallel_arena.py` | process-pool arena (GIL bypass) |
| `arena/multiview.py` | temporal / aligned / 3-D multi-view refinement |
| `arena/planner.py` | FOV-capped active view planning + strategy benchmark |
| `flash_lidar.py` | torch-pulse inverse-square depth, temporal fusion, PCA planes, catalog match |
| `arena/hardware_probe.py` | CPU/memory/GIL/fusion limits + budget |
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

## Install on Termux

```bash
pkg install -y python curl
curl -fsSL https://raw.githubusercontent.com/chrisalunlloyd2-sudo/LiDar/main/scripts/termux-install.sh | bash
lidar                  # GUI + API on http://127.0.0.1:8790
```

The installer downloads the latest release `.deb`, verifies it against the
published `SHA256SUMS`, and installs it with `pkg`. `lidar --memory` uses a
throwaway database; the default lives in `~/.lidar/lidarv2.db`.

Manual route if you prefer to see every step:

```bash
curl -fsSLO https://github.com/chrisalunlloyd2-sudo/LiDar/releases/download/v2.5.1/lidar_2.5.1_all.deb
curl -fsSLO https://github.com/chrisalunlloyd2-sudo/LiDar/releases/download/v2.5.1/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
pkg install -y ./lidar_2.5.1_all.deb
```

Build the package yourself with `bash scripts/build_termux_package.sh` — it
writes `lidar_<version>_all.deb`, a tarball and checksums into `dist/`
(`Architecture: all`, `Depends: python`). Remove it with `pkg uninstall lidar`;
your database in `~/.lidar/` is left alone.

## Run

```bash
cd ~/LiDar
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
8. **Plan views** derives the visible arc from the current depth and lists the
   offsets to visit plus the predicted error after each one.
9. **Flash depth** renders the torch-pulse scene, reports depth MAPE and the
   catalog target, and draws the ASCII depth map.

## API

| Method | Path | Body / query | Returns |
|--------|------|--------------|---------|
| GET | `/` | — | mmap shell |
| GET | `/page/<name>` | — | one mmap page fragment |
| GET | `/api/pages` | — | mapped page list |
| GET | `/api/health` | — | status + db + page count |
| GET | `/api/config` | — | version + tag catalog |
| GET | `/api/illusions` | — | the 5-illusion registry |
| GET | `/api/hardware` | — | CPU/mem/GIL/fusion limits |
| GET | `/api/survivors` | — | arena keep/prune |
| POST | `/api/arena` | `{}` | run the quick arena |
| POST | `/api/observe` | observation JSON | depth+CI, 5 illusions, geo, item |
| POST | `/api/scan` | intrinsics + pose + grid | ground point cloud |
| POST | `/api/fit` | `corners_px`, `depth_m`, `kind` | rectangle/cuboid fit |
| POST | `/api/triangulate` | `pose1/pose2`, `uv1/uv2` or `corners1/2` | point / size |
| POST | `/api/premade` | `{tag, depth_m}` | emulated vs truth + error % |
| POST | `/api/benchmark` | `{depth_m, noise_frac}` | MAPE + coverage |
| POST | `/api/plan` | `{depth_m, budget, tag, sigma_px}` | FOV cap + offsets + predicted error per view |
| POST | `/api/flash` | `{cols, rows, seed}` or `{ambient[], flash[], calibration}` | depth grid, planes, targets, ASCII map |
| POST | `/api/flash/calibrate` | `{ambient[], flash[], reference_m}` | centre DN, health verdict, suggested `k` |
| POST | `/api/flash/observe` | observation JSON + grids + `calibration` | flash cue fused with the monocular cues |
| POST | `/api/demo` | `{room}` | seeded inventory |
| GET | `/api/map` | `?room=&tag=&limit=` | accepted map points |
| GET | `/api/inventory` | `?room=&tag=` | items + summary |
| POST | `/api/reset` | `{}` | clears map + inventory |

## Security: secret scrub

`scripts/scrub_secrets.py` is a three-pass, stdlib-only scanner for a repo that
is about to go public:

```bash
python3 scripts/scrub_secrets.py                  # tree + history + metadata
python3 scripts/scrub_secrets.py --history --rev origin/main   # what is published
python3 scripts/scrub_secrets.py --json scan.json
```

It flags GitHub / AWS / Slack / Google tokens, private keys, bearer tokens,
assigned secrets and credential URLs, plus emails, phone numbers and absolute
home paths. Every match is redacted before printing, so the report itself is
safe to paste into an issue. Exit code 1 means something was found.

`--rev` scopes the history pass to what is actually on the remote; the default
`--all` also covers tags and other local-only refs, which is where old
monorepo blobs hide. Run it before publishing and again after pulling outside
contributions.

## Tests

```bash
cd ~/LiDar
python3 -m pytest tests -q      # 210 tests
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

## Support

If this saved you some time, send coffee money to
`chrisalunlloyd2@gmail.com`.

## Changelog

Release history with the measured numbers behind each decision lives in
[CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE). Contributions are welcome under the same terms.
