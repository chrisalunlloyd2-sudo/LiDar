# Changelog

All notable changes to lidarv2 / LiDar. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Accuracy figures are
medians over noise seeds, with `n` stated; a single draw is always labelled.

## [Unreleased]

Nothing yet.

## [2.5.0] — 2026-11-07

### Added

- **Torch-pulse depth** (`flash_lidar.py`): differential flash/ambient map →
  inverse-square range → inverse-variance temporal fusion → 3-D ENU/WGS84
  points → PCA plane segmentation → catalog match, plus ASCII/ANSI depth maps.
- `flash_falloff` enters the engine as a first-class `CueEstimate`, so
  `uncertainty.fuse_cues` and its confidence intervals apply unchanged.
- `POST /api/flash` (demo scene or browser-supplied grids) and
  `POST /api/flash/observe` (flash cue fused with the monocular cues).
- mmap GUI page `flash.html`: demo run plus live two-phase capture — ambient,
  torch, solve — with 1/3/5/7-frame scrub (default 3, per-cell median) and
  torch control through `ImageCapture` where the browser exposes it.
- `scripts/scrub_secrets.py`: three-pass (tree / history / metadata) secret and
  PII scan with redacted output and `--fail-on` for CI.
- CI workflow running `compileall`, the test suite and the secret scrub on every
  push and PR; `CONTRIBUTING.md`; MIT `LICENSE`; CI and license badges.

### Fixed

- The forward renderer fed **z-depth** into the square law, so a flat wall
  rendered as a sphere of constant range (plane RMS 89 mm, offset 1.85 m). It
  now uses Euclidean range along the ray; the same wall fits at 2.000 m with
  17 mm RMS.
- `match_target` matched a 30 cm strip to a credit card's short side. Entries
  whose catalog dimensions describe the whole face (card, A4, brick, paint can,
  box) must now match in both extents; lumber is exempt because the catalog does
  not know board length.
- `flash_cue` averaged background wall inside its 80 px radius, so a 1.5 m board
  read 1.67 m. Cells are now gated around the depth of the cell nearest the
  reticle before fusion.

### Measured

- 96 × 54 = 5,184 cells, 16 seeds: depth MAPE **1.93 %** (p90 1.96 %),
  per-cell median 1.62 %, p90 4.12 %, ~0.49 s per frame; both boards found
  16/16.
- Face widths: 2×6 148.6 mm vs 139.7 mm (+6.4 %, 95 % CI ±80.7 mm); 2×8
  189.9 mm vs 184.2 mm (+3.1 %, ±136.8 mm).
- Mixed albedo (0.30 vs the 0.55 reference): **3.94 %**, and the near 2×6 drops
  out of the catalog 16/16 (its 140 mm face reads ~103 mm). The far 2×8 still
  matches, but its range is 35 % too far — a label is not a measurement.

### Changed

- Test suite: 210 tests (was 183).

## [2.4.0] — 2026-11-07

### Added

- **Active view planning** (`arena/planner.py`): `fov_cap_m` derives the
  observable arc from intrinsics, depth and object width — ±0.692 m for a 2×6
  at 1.5 m with a 60° FOV. Views whose corners leave the frame are dropped.
- Strategy benchmark at equal budgets; the planner predicts error before
  measuring it (`pixel` + `triangulation` terms fitted over 90 combinations,
  realised/predicted median 0.934, rms log-ratio 0.127).
- `POST /api/plan` and the `plan.html` GUI page.

### Measured

- 2×6 height error, 16 seeds — even arc: 1.804 / 1.186 / 0.780 / 0.702 % at
  2/4/6/8 views; alternating extremes 1.738 / 1.582 / 0.845 / 0.731 %; random
  5.230 / 2.051 / 0.987 / 0.761 %. Anchor-only plans cannot triangulate.
- All-fixture MAPE for the even arc: 0.709 / 0.451 / 0.330 / 0.332 %.

### Changed

- Extracted to the standalone `LiDar` repository (v2.0–v2.4 history preserved).

## [2.3.0] — 2026-11-05

### Added

- `arena/parallel_arena.py`: process-pool arena. Threads are useless under the
  GIL on this device; **525 ms vs 1,145 ms = 2.18×** on 4 workers.
- `arena/multiview.py`: 3-D corner averaging across views, with `pair_is_usable`
  rejecting degenerate pairs before they drag the average.

### Fixed

- Accuracy reporting: v2.3's headline 0.290 % was **seed 0**. The same
  configuration over 16 seeds gives a median of **0.747 %** (p90 1.498 %), so
  seeds became an explicit parameter and single draws are labelled.

## [2.2.0] — 2026-11-05

### Added

- `arena/`: 17 depth fusers, 11 size estimators, 3 illusion models and 5
  triangulators, ranked by MAPE and snipped — survivors (`inverse_variance`,
  `pixel_scale`, `canonical5`, `midpoint`, …) are what production uses.
- `arena/hardware_probe.py`: 8 cores, 3.6 GB RAM, thread scaling 0.54–0.87×
  (GIL-bound), process scaling 1.52×, fusion ~3.6–4.0 k/s → budget
  `threads=1`, `processes=8`.

## [2.1.0] — 2026-11-05

### Added

- Canonical **5-illusion** registry and multiplicative correction.
- Shape classifier (square / rectangle / cube / cuboid / lumber).
- Nelder-Mead rectangle and cuboid reprojection fit.
- Two-view triangulation for points and sizes.
- Accuracy benchmark: mean MAPE 0.446 %, max error 2.887 %, CI coverage 100 %.
- mmap modular GUI (`page_store.py` + one fragment per page).

## [2.0.0] — 2026-11-05

### Added

- Emulative LiDAR backend with confidence intervals: pixel-scale, angular,
  ground-plane and texture depth cues fused by inverse variance, with a
  between-cue variance term widening the interval when cues disagree.
- Illusion filter (cue conflict, mirror, haze, specular) and accept/reject gate.
- Geotagged inventory: pixel → ray → ENU → WGS84 with an uncertainty radius,
  SQLite map points and aggregated items.
- Stdlib `ThreadingHTTPServer` JSON API (`/api/observe`, `/api/scan`,
  `/api/map`, `/api/inventory`) plus a camera frontend.
