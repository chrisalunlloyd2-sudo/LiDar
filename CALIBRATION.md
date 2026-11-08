# Calibration

The torch-pulse gauge turns differential flux into metres through one lumped
constant:

```
k = S0 · d0²          S0 = median differential DN of the calibration patch
d  = d0 · sqrt(S0/S)  d0 = range to that patch, S = a cell's differential DN
```

A calibration is therefore two numbers: a **reference range** and the
**differential flux at that range**. Everything below is about choosing those
two numbers so the gauge is trustworthy, and about recognising when it is not.

> **Status of the numbers.** The tables in *Synthetic gauge pass* were produced
> by `scripts/flash_calibrate.py --sweep` on this device (renderer gain 490 or
> 120, 96 × 54 cells, seed 0, albedo 0.55). They characterise the **model**, not
> your camera. The *Device pass* section is the procedure that produces the
> real table; its results are left blank until someone runs it on a phone.

## What the gauge needs

| quantity | healthy window | why |
|---|---|---|
| centre differential `S0` | **20–200 DN** | below 20 DN sensor noise dominates; above 200 DN cells clip |
| lit fraction | **≥ 80 %** | the torch must actually light the frame |
| p90 / median of the centre box | **≤ 1.5** | a hot spot means the reference is not a flat patch |
| cue error at a known range | **≤ 3 %** | the number the ping reports should match the tape measure |

`saturated` (>200 DN), `dark` (≤0 DN) and `marginal` (lit or evenness) are
refusals, not warnings to ignore. `calibrate_from_grids` returns them in the
`status` field; the browser solver refuses a ping under 0.5 DN outright.

## Synthetic gauge pass

`python3 scripts/flash_calibrate.py --sweep --markdown` — each wall is
calibrated at its own distance, then the whole grid is ranged with that
calibration. Reference range is the median **range** to the wall (slightly
longer than its z-distance, because ranges are Euclidean).

### Gain 490 — the demo default

| wall z (m) | ref range (m) | centre DN | lit % | status | depth MAPE % | plane z error % |
|---|---|---|---|---|---|---|
| 0.50 | 0.538 | 227.8 | 100 | saturated | — | — |
| 0.75 | 0.807 | 227.8 | 100 | saturated | — | — |
| 1.00 | 1.076 | 226.7 | 100 | saturated | 1.27 | 1.27 |
| 1.25 | 1.345 | 148.8 | 100 | ok | 0.27 | 0.03 |
| 1.50 | 1.614 | 103.3 | 100 | ok | 0.39 | 0.03 |
| 2.00 | 2.153 | 58.1 | 100 | ok | 0.70 | 0.01 |
| 2.50 | 2.691 | 37.2 | 100 | ok | 1.09 | 0.01 |
| 3.00 | 3.229 | 25.8 | 100 | ok | 1.57 | 0.07 |
| 4.00 | 4.305 | 14.5 | 100 | marginal | 2.80 | 0.14 |

At this gain the usable band is **1.25–3 m**. Closer than ~1.1 m the flash
clips: the differential pins at 227.8 DN and the reported depth is biased
(1.27 % at 1.0 m, no valid cells at 0.75 m and below). At 4 m the signal is
14.5 DN — still solvable, but 2.8 % noisy, and the status says `marginal`.

### Gain 120 — shifted to close range

| wall z (m) | ref range (m) | centre DN | lit % | status | depth MAPE % | plane z error % |
|---|---|---|---|---|---|---|
| 0.50 | 0.538 | 226.1 | 100 | saturated | 0.39 | 0.39 |
| 0.75 | 0.807 | 101.2 | 100 | ok | 0.40 | 0.03 |
| 1.00 | 1.076 | 56.9 | 100 | ok | 0.71 | 0.01 |
| 1.25 | 1.345 | 36.4 | 100 | ok | 1.11 | 0.02 |
| 1.50 | 1.614 | 25.3 | 100 | ok | 1.60 | 0.07 |
| 2.00 | 2.153 | 14.2 | 100 | marginal | 2.86 | 0.17 |
| 2.50 | 2.691 | 9.1 | 100 | marginal | 4.51 | 0.37 |
| 3.00 | 3.229 | 6.3 | 100 | marginal | 6.62 | 0.56 |
| 4.00 | 4.305 | 3.5 | 97 | marginal | 11.39 | 1.32 |

Lowering the gain moves the healthy band to **0.75–1.5 m**. There is no single
gain for 0.5–4 m; the exposure/gain must be chosen for the working range, and
`k` recomputed when it changes.

### Does one calibration transfer to another distance?

Calibrated once on a 2.15 m wall, then measuring a centred 2×6 board at other
ranges (gain 490). `cue` is the reticle measurement, `width` is the segmented
face — they fail differently, which is the point.

| 2×6 at (m) | cue range (m) | cue error % | width (mm) | width error % | plane matched |
|---|---|---|---|---|---|
| 0.75 | — | — | — | — | no (target clipped) |
| 1.00 | — | — | — | — | no (target clipped) |
| 1.25 | 1.255 | **+0.37** | 152.4 | +9.09 | yes |
| 1.50 | 1.505 | **+0.35** | 146.8 | +5.08 | yes |
| 2.00 | 2.006 | **+0.31** | 147.3 | +5.44 | yes |
| 2.50 | 2.507 | **+0.29** | 145.8 | +4.37 | yes |
| 3.00 | 3.025 | +0.82 | — | — | no (merged with wall) |
| 4.00 | 4.026 | +0.65 | — | — | no (merged with wall) |

The range gauge transfers well: **0.3–0.8 % from 1.25 m to 4 m** with one
calibration. Widths are a different story — the one-cell boundary correction
runs +4 to +9 %, and beyond ~2.5 m the board no longer separates from the wall:
at 4 m the per-cell depth sigma is ~0.23 m, and flood-fill needs a standoff of
roughly **5σ**, so a 1 m gap leaks. The cue is unaffected because it clusters
around the reticle instead of segmenting the full grid.

A clipped target used to be worse than a missing one: the cue fell back to the
nearest *valid* cell — the background — and reported ~4.03 m for a 0.75 m board.
`flash_cue` now returns `None` when the reticle cell itself is unmeasurable.

## Device pass

Do this before trusting a live range on your phone. It takes about ten minutes.
The card keeps a log of every ping and copies out rows in the table format, so
the device table fills itself. **Guided pass** walks the four distances for you.

1. `python3 server.py` (or the installed `lidar`) and open the GUI on the phone;
   start the camera.
2. Tape-measure a flat, evenly lit wall/patch at **1.0 m**. Press **Guided
   pass** — it clears any locked `k`, sets the `reference (true range now)` box
   to 1.0 m and tells you the step.
3. **1 · Ambient** (scrub 3 frames), torch **on**, **2 · Torch**, **Solve ping**.
   The card asks `/api/flash/calibrate` for the verdict (`ok`/`marginal`/
   `saturated`/`dark`, centre DN, `k`), refuses a dark pair, and **locks `k`**.
4. Move to **1.5 m**, measure, and press the same three buttons — the pass has
   already put `1.5` in the box, so this ping uses the locked `k` and the logged
   error is a genuine transfer error, not a self-fulfilling one.
5. Repeat at **2.0 m** and **3.0 m**; the pass marks itself complete after the
   fourth row. If you press **Guided pass** again mid-run it cancels; running
   unlocked pings instead would re-anchor each time and read ~0 %.
6. Press **Copy rows** and paste them under *Device results* below. **Recalibrate**
   clears the locked `k` when you change exposure, zoom or camera.
7. Optional: **Export grids** at each distance and replay the analysis from the
   shell — same verdict, reproducible:

   ```bash
   python3 scripts/flash_calibrate.py --grids flash_grids.json --range 1.5
   python3 scripts/flash_calibrate.py --grids flash_grids.json --range 1.5 --markdown
   ```

   The second form prints one table row (`self-anchored`: it calibrates and
   measures the same pair, so expect ~0 % — it validates the pipeline, not the
   transfer). The endpoint behind it is `POST /api/flash/calibrate` with
   `{cols, rows, ambient[], flash[], reference_m}`.

### Device results

Filled from **Copy rows** on the phone — do not copy the synthetic table.

| target range | centre DN | k = S0·d0² | cue range | error % | status | notes |
|---|---|---|---|---|---|---|
| 1.0 m | | | | | | |
| 1.5 m | | | | | | |
| 2.0 m | | | | | | |
| 3.0 m | | | | | | |

## Failure modes

| symptom | cause | what to do |
|---|---|---|
| solve refuses, differential < 0.5 DN | torch off, or auto-exposure stepped between phases | keep exposure fixed; capture ambient then torch quickly; scrub ≥ 3 |
| `saturated`, no valid cells on the target | target too close/bright for the gain | move back past ~1.2 m at gain 490, or lower exposure gain |
| cue returns nothing at the reticle | reticle cell clipped or dark (intentional refusal) | reframe or change distance; do not lower the noise floor |
| width wrong by +5–10 % | one-cell boundary correction at coarse grids | refine the grid (more cols/rows) or prefer the cue for range |
| board missing beyond ~2.5 m | depth sigma ≈ 0.2 m vs a 1 m standoff; segmentation merges | increase the standoff behind the target, or measure range with the cue |
| every range 35 % too far | mixed albedo: `d ∝ sqrt(ρ_ref/ρ)` | calibrate on the same material, or accept the bias explicitly |
| `marginal`, lit < 80 % | torch beam narrower than the frame or hand shadow | centre the reticle, remove occlusion, re-scrub |

Changing zoom, exposure, torch level or camera (front/back) invalidates `k`.
`albedo_rel_sigma` is the honest escape hatch: it widens the depth sigma for
the part of the error calibration cannot remove.
