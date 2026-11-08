#!/usr/bin/env python3
"""flash_calibrate — check and derive the torch-pulse range gauge.

Two modes
---------
  --grids FILE.json   analyse a captured ambient/torch pair (from the GUI's
                      "Export grids" button) and suggest a calibration
  --sweep             synthetic gauge check: calibrate at each distance, then
                      test whether a calibration transfers to another distance

Pure stdlib. Run from the repo root so `flash_lidar` imports:

  python3 scripts/flash_calibrate.py --sweep
  python3 scripts/flash_calibrate.py --grids flash_grids.json --range 1.5
  python3 scripts/flash_calibrate.py --grids flash_grids.json --json out.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import flash_lidar as fl  # noqa: E402
from models import CameraIntrinsics  # noqa: E402

DEFAULT_DISTANCES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)


def intrinsics_from(data):
    intr = data.get("intrinsics") or {}
    if intr.get("f_px"):
        return CameraIntrinsics(
            f_px=float(intr["f_px"]),
            cx=float(intr.get("cx", intr.get("image_w", 640) / 2.0)),
            cy=float(intr.get("cy", intr.get("image_h", 480) / 2.0)),
            image_w=int(intr.get("image_w", 640)),
            image_h=int(intr.get("image_h", 480)),
            fov_deg=float(intr.get("fov_deg", 60.0)),
        )
    return CameraIntrinsics.from_fov(int(intr.get("image_w", 640)),
                                     int(intr.get("image_h", 480)),
                                     float(intr.get("fov_deg", 60.0)))


def load_grids(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cols = int(data.get("cols") or 0)
    rows = int(data.get("rows") or 0)
    ambient = data.get("ambient") or []
    flash = data.get("flash") or []
    if cols <= 0 or rows <= 0:
        raise SystemExit("grid file needs cols and rows")
    if len(ambient) != cols * rows or len(flash) != cols * rows:
        raise SystemExit("grid file: ambient/flash must each hold cols*rows values")
    return data, cols, rows, [float(v) for v in ambient], [float(v) for v in flash]


def report_grids(path, known_range):
    data, cols, rows, ambient, flash = load_grids(path)
    intr = intrinsics_from(data)
    known = known_range if known_range is not None else data.get("reference_m")
    result = fl.calibrate_from_grids(ambient, flash, cols, rows, intr,
                                     known_range_m=known)
    result["source"] = os.path.basename(path)
    result["samples"] = data.get("samples")
    print("source      : %s (%d x %d cells)" % (path, cols, rows))
    print("centre      : median %.1f DN  p90 %.1f DN  lit %.1f %%"
          % (result["centre_median_dn"], result["centre_p90_dn"],
             result["lit_fraction"] * 100.0))
    print("status      : %s" % result["status"])
    for warning in result["warnings"]:
        print("warning     : %s" % warning)
    if result["suggested_calibration"]:
        cal = result["suggested_calibration"]
        print("calibration : reference %.2f m at %.1f DN (k = %.1f)"
              % (cal["reference_depth_m"], cal["reference_delta_dn"],
                 cal["constant"]))
    else:
        print("calibration : unanchored — pass --range <metres to the patch>")
    return result


def sweep(distances, cols, rows, gain, calibration_range, seed, markdown):
    intr = CameraIntrinsics.from_fov(640, 480, 60.0)
    rows_out = []
    for z in distances:
        frame = fl.render_scene(fl.FlashScene("wall", [], wall_depth_m=z), intr,
                                cols=cols, rows=rows, gain=gain, seed=seed)
        cal = fl.calibrate_scene(frame, intr)
        analysed = fl.analyse_grid(frame["ambient"], frame["flash"],
                                   cols, rows, intr, cal)
        truth = fl.ground_truth_cells(frame, intr)
        accuracy = fl.depth_mape(analysed["cells"], truth)
        plane = analysed["planes"][0] if analysed["planes"] else None
        plane_err = None
        if plane is not None:
            plane_err = abs(abs(plane["offset_m"]) - z) / z * 100.0
        health = fl.calibrate_from_grids(
            frame["ambient"], frame["flash"], cols, rows, intr,
            known_range_m=cal.reference_depth_m)
        rows_out.append({
            "distance_m": z,
            "reference_range_m": round(cal.reference_depth_m, 3),
            "delta_dn": round(cal.reference_delta_dn, 2),
            "lit_pct": round(health["lit_fraction"] * 100.0, 1),
            "status": health["status"],
            "range_mape_pct": accuracy["mape_pct"],
            "plane_z_error_pct": None if plane_err is None else round(plane_err, 3),
            "valid_pct": analysed["valid_pct"],
        })

    transfer = []
    for z in distances:
        frame = fl.render_scene(fl.FlashScene("wall", [], wall_depth_m=calibration_range),
                                intr, cols=cols, rows=rows, gain=gain, seed=seed)
        cal = fl.calibrate_scene(frame, intr)
        scene = fl.FlashScene("board", [
            fl.ScenePatch("2x6", "2x6 lumber", 0.0, 0.0, 0.1397, 0.62, z),
        ], wall_depth_m=calibration_range + 2.0)
        board = fl.render_scene(scene, intr, cols=cols, rows=rows,
                                gain=gain, seed=seed)
        analysed = fl.analyse_grid(board["ambient"], board["flash"],
                                   cols, rows, intr, cal)
        target = next((t for t in analysed["targets"] if t["tag"] == "2x6"), None)
        cue = analysed["flash_cue"]
        cue_error = None
        if cue and cue.get("depth_m"):
            cue_error = round((cue["depth_m"] - z) / z * 100.0, 2)
        transfer.append({
            "target_m": z,
            "calibrated_at_m": round(cal.reference_depth_m, 3),
            "cue_range_m": None if not cue else round(cue["depth_m"], 3),
            "cue_error_pct": cue_error,
            "measured_width_mm": target["measured_width_mm"] if target else None,
            "width_error_pct": None if not target else round(
                (target["measured_width_mm"] - 139.7) / 139.7 * 100.0, 2),
            "matched": bool(target),
        })

    if markdown:
        print("| wall z (m) | ref range (m) | centre DN | lit % | gauge status "
              "| depth MAPE % | plane z error % |")
        print("|---|---|---|---|---|---|---|")
        for r in rows_out:
            print("| %.2f | %.3f | %.1f | %.0f | %s | %s | %s |" % (
                r["distance_m"], r["reference_range_m"], r["delta_dn"],
                r["lit_pct"], r["status"],
                "—" if r["range_mape_pct"] is None else "%.2f" % r["range_mape_pct"],
                "—" if r["plane_z_error_pct"] is None else "%.2f" % r["plane_z_error_pct"]))
        print()
        print("| 2x6 at (m) | calibrated at (m) | cue range (m) | cue error % "
              "| width (mm) | width error % | matched |")
        print("|---|---|---|---|---|---|---|")
        for r in transfer:
            print("| %.2f | %.3f | %s | %s | %s | %s | %s |" % (
                r["target_m"], r["calibrated_at_m"],
                "—" if r["cue_range_m"] is None else "%.3f" % r["cue_range_m"],
                "—" if r["cue_error_pct"] is None else "%+.2f" % r["cue_error_pct"],
                "—" if r["measured_width_mm"] is None else "%.1f" % r["measured_width_mm"],
                "—" if r["width_error_pct"] is None else "%+.2f" % r["width_error_pct"],
                "yes" if r["matched"] else "no"))
    else:
        for r in rows_out:
            print("z=%.2f m  ref=%.3f m  DN=%.1f  lit=%.0f%%  %-9s  "
                  "range MAPE=%s  plane z err=%s"
                  % (r["distance_m"], r["reference_range_m"], r["delta_dn"],
                     r["lit_pct"], r["status"], r["range_mape_pct"],
                     r["plane_z_error_pct"]))
        print()
        for r in transfer:
            print("2x6 at %.2f m (cal at %.3f m): cue %s m (%s %%), width %s mm "
                  "(%s %%), matched=%s"
                  % (r["target_m"], r["calibrated_at_m"], r["cue_range_m"],
                     r["cue_error_pct"], r["measured_width_mm"],
                     r["width_error_pct"], r["matched"]))
    return {"gauge": rows_out, "transfer": transfer,
            "config": {"cols": cols, "rows": rows, "gain": gain,
                       "calibration_range_m": calibration_range, "seed": seed}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="torch-pulse calibration check")
    parser.add_argument("--grids", help="exported ambient/torch grid JSON")
    parser.add_argument("--range", type=float, default=None,
                        help="range in metres to the calibration patch")
    parser.add_argument("--sweep", action="store_true",
                        help="synthetic gauge + transfer sweep")
    parser.add_argument("--distances", default=None,
                        help="comma-separated metres for --sweep")
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--rows", type=int, default=54)
    parser.add_argument("--gain", type=float, default=490.0)
    parser.add_argument("--calibration-range", type=float, default=2.0,
                        help="distance the transfer test calibrates at")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if args.grids:
        result = report_grids(args.grids, args.range)
    elif args.sweep or not args.grids:
        distances = DEFAULT_DISTANCES
        if args.distances:
            distances = tuple(float(v) for v in args.distances.split(","))
        result = sweep(distances, args.cols, args.rows, args.gain,
                       args.calibration_range, args.seed, args.markdown)
    else:
        parser.error("nothing to do")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
