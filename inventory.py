"""lidarv2 inventory — auto-tagging and emulated dimensions.

Every accepted observation is classified against a small physical catalog
(lumber, sheet goods, containers) using the *emulated* dimensions derived
from depth + pixels. Items are then aggregated by tag + room + location so
"find the 2x4 in basement B" returns geotagged stock rather than raw frames.

Dimension emulation
-------------------
    real_width_mm  = depth_m * px_width  / f_px * 1000
    real_height_mm = depth_m * px_height / f_px * 1000
Thickness is not observable from a single view, so it is taken from the
catalog when the tag matches, otherwise left unknown (None).
"""

from models import Observation

TAG_CATALOG = [
    {"tag": "2x4", "label": "2x4 lumber", "w_mm": 88.9, "h_mm": 38.1, "d_mm": 38.1,
     "aliases": ["2x4", "2×4", "two by four"]},
    {"tag": "2x6", "label": "2x6 lumber", "w_mm": 139.7, "h_mm": 38.1, "d_mm": 38.1,
     "aliases": ["2x6", "2×6"]},
    {"tag": "2x8", "label": "2x8 lumber", "w_mm": 184.2, "h_mm": 38.1, "d_mm": 38.1,
     "aliases": ["2x8", "2×8"]},
    {"tag": "4x4", "label": "4x4 post", "w_mm": 88.9, "h_mm": 88.9, "d_mm": 88.9,
     "aliases": ["4x4", "4×4"]},
    {"tag": "1x4", "label": "1x4 board", "w_mm": 88.9, "h_mm": 19.05, "d_mm": 19.05,
     "aliases": ["1x4", "1×4"]},
    {"tag": "credit_card", "label": "Credit card", "w_mm": 85.6, "h_mm": 53.98, "d_mm": 0.76,
     "aliases": ["credit card", "card"]},
    {"tag": "a4_paper", "label": "A4 sheet", "w_mm": 210.0, "h_mm": 297.0, "d_mm": 0.1,
     "aliases": ["a4", "paper"]},
    {"tag": "brick", "label": "Standard brick", "w_mm": 200.0, "h_mm": 65.0, "d_mm": 95.0,
     "aliases": ["brick"]},
    {"tag": "paint_can", "label": "Paint can (1 gal)", "w_mm": 168.0, "h_mm": 190.0, "d_mm": 168.0,
     "aliases": ["paint", "paint can"]},
    {"tag": "box", "label": "Cardboard box", "w_mm": 400.0, "h_mm": 400.0, "d_mm": 400.0,
     "aliases": ["box"]},
]

_CATALOG_BY_TAG = {e["tag"]: e for e in TAG_CATALOG}


def _match(a, b, tol):
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


def classify_dimensions(w_mm, h_mm, tolerance=0.15):
    """Best catalog match for an (width, height) pair, orientation-agnostic."""
    if not w_mm or not h_mm or w_mm <= 0 or h_mm <= 0:
        return None
    best = None
    best_err = float("inf")
    for entry in TAG_CATALOG:
        for a, b in ((entry["w_mm"], entry["h_mm"]), (entry["h_mm"], entry["w_mm"])):
            if _match(w_mm, a, tolerance) and _match(h_mm, b, tolerance):
                err = abs(w_mm - a) / max(w_mm, a) + abs(h_mm - b) / max(h_mm, b)
                if err < best_err:
                    best_err = err
                    best = entry
    return best


def classify_observation(obs: Observation, dims_mm=None):
    """Resolve a tag: explicit preset first, then dimension matching."""
    name = (obs.object_class or "").strip()
    if name and name not in ("unknown", "custom"):
        for entry in TAG_CATALOG:
            if name == entry["tag"] or name == entry["label"] or name in entry["aliases"]:
                return entry
        return {"tag": name, "label": name, "w_mm": None, "h_mm": None, "d_mm": None, "aliases": []}
    if dims_mm:
        match = classify_dimensions(dims_mm[0], dims_mm[1])
        if match:
            return match
    return {"tag": "unknown", "label": "Unknown object", "w_mm": None, "h_mm": None,
            "d_mm": None, "aliases": []}


def emulated_dimensions_mm(obs: Observation, depth_m):
    """Emulate (width, height, thickness) in mm from depth + pixel extents."""
    f = obs.intrinsics.f_px or 1.0
    w = depth_m * obs.object_px_width / f * 1000.0
    h = depth_m * obs.object_px_height / f * 1000.0
    return (round(w, 1), round(h, 1), None)


def fill_thickness(dims_mm, tag):
    """Attach catalog thickness when the tag is known and thickness unknown."""
    w, h, d = dims_mm
    if d is None:
        entry = _CATALOG_BY_TAG.get((tag or {}).get("tag"))
        if entry:
            d = entry.get("d_mm")
    return (w, h, d)
