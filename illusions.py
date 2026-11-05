"""lidarv2 illusions — the canonical 5-illusion list + correction math.

Focus: squares, rectangles and cubes. Each illusion exposes a severity in
[0,1] and a multiplicative correction factor applied to the *emulated size*.
The factors multiply into one combined correction, which the server reports
alongside the corrected dimensions.

The five (geometry + light):

  1. ponzo              converging perspective inflates apparent size
  2. muller_lyer        arrowhead terminators distort length
  3. vertical_horizontal vertical extents look longer than equal horizontals
  4. ebbinghaus         size-constancy / relative-size contrast
  5. checker_shadow     luminance contrast misreads surface lightness/depth
"""

from models import IllusionFlag


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


ILLUSION_REGISTRY = [
    {
        "id": "ponzo",
        "name": "Ponzo",
        "kind": "geometry",
        "applies_to": "both",
        "formula": "factor = 1 - 0.30 * convergence",
        "max_correction": 0.30,
    },
    {
        "id": "muller_lyer",
        "name": "Müller-Lyer",
        "kind": "geometry",
        "applies_to": "rectangle",
        "formula": "factor = 1 - 0.20 * arrowhead",
        "max_correction": 0.20,
    },
    {
        "id": "vertical_horizontal",
        "name": "Vertical-horizontal",
        "kind": "geometry",
        "applies_to": "rectangle",
        "formula": "factor = 1 - 0.12 * clamp((aspect-1)/2)",
        "max_correction": 0.12,
    },
    {
        "id": "ebbinghaus",
        "name": "Ebbinghaus",
        "kind": "geometry",
        "applies_to": "both",
        "formula": "factor = 1 - 0.15 * clamp((|d_pixel-d_angular|/max - 0.10)/0.40)",
        "max_correction": 0.15,
    },
    {
        "id": "checker_shadow",
        "name": "Checker-shadow",
        "kind": "light",
        "applies_to": "both",
        "formula": "factor = 1 - 0.10 * (0.5*(1-contrast) + 0.5*specular)",
        "max_correction": 0.10,
    },
]


def _ponzo(obs, cues):
    return _clamp(obs.convergence or 0.0)


def _muller_lyer(obs, cues):
    return _clamp(obs.arrowhead or 0.0)


def _vertical_horizontal(obs, cues):
    if obs.object_px_width <= 0:
        return 0.0
    aspect = obs.object_px_height / obs.object_px_width
    return _clamp((aspect - 1.0) / 2.0) if aspect > 1.0 else 0.0


def _ebbinghaus(obs, cues):
    by_name = {c.name: c for c in cues}
    if "pixel_scaling" not in by_name or "angular_size" not in by_name:
        return 0.0
    a = by_name["pixel_scaling"].depth_m
    b = by_name["angular_size"].depth_m
    rel = abs(a - b) / max(a, b, 1e-9)
    return _clamp((rel - 0.10) / 0.40)


def _checker_shadow(obs, cues):
    spec = _clamp(obs.specular or 0.0)
    contrast = 0.5 if obs.contrast is None else _clamp(obs.contrast)
    low_contrast = _clamp((0.25 - contrast) / 0.25)
    return _clamp(0.5 * low_contrast + 0.5 * spec)


_SEVERITY = {
    "ponzo": _ponzo,
    "muller_lyer": _muller_lyer,
    "vertical_horizontal": _vertical_horizontal,
    "ebbinghaus": _ebbinghaus,
    "checker_shadow": _checker_shadow,
}


def evaluate(obs, cues):
    """Score all five illusions and return the combined size correction."""
    entries = []
    combined = 1.0
    for spec in ILLUSION_REGISTRY:
        severity = _clamp(_SEVERITY[spec["id"]](obs, cues))
        factor = 1.0 - spec["max_correction"] * severity
        combined *= factor
        entries.append({
            "id": spec["id"],
            "name": spec["name"],
            "kind": spec["kind"],
            "applies_to": spec["applies_to"],
            "severity": round(severity, 4),
            "factor": round(factor, 4),
            "formula": spec["formula"],
        })
    combined = _clamp(combined, 0.30, 1.20)
    ranked = sorted(entries, key=lambda e: e["severity"], reverse=True)
    return {
        "illusions": entries,
        "top": ranked[:3],
        "combined_factor": round(combined, 4),
        "correction_percent": round((1.0 - combined) * 100.0, 2),
    }


def as_flags(report):
    """Adapt the five-illusion report to the IllusionFlag model."""
    return [IllusionFlag(e["id"], e["name"], e["severity"]) for e in report["illusions"]
            if e["severity"] > 0.15]


def correct_size(dims_mm, factor):
    """Scale an emulated (w, h, d) tuple by the combined correction factor."""
    w, h, d = dims_mm
    return (
        round(w * factor, 1),
        round(h * factor, 1),
        None if d is None else round(d * factor, 1),
    )
