"""lidarv2 illusion models — baseline, the canonical 5, and an extended 10.

The arena compares "no correction", the user's canonical five, and a deliberately
over-built ten. The extra five are pruned if they do not improve accuracy.
"""

from illusions import ILLUSION_REGISTRY, evaluate as evaluate_canonical


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def _contrast(obs):
    return 0.5 if obs.contrast is None else _clamp(obs.contrast)


def _spec(obs):
    return _clamp(obs.specular or 0.0)


def _conv(obs):
    return _clamp(obs.convergence or 0.0)


def _arrow(obs):
    return _clamp(obs.arrowhead or 0.0)


EXTRA_REGISTRY = [
    {
        "id": "delboeuf",
        "name": "Delboeuf",
        "kind": "geometry",
        "applies_to": "both",
        "formula": "factor = 1 - 0.08 * clamp((0.30 - contrast)/0.30)",
        "max_correction": 0.08,
        "_sev": lambda obs, cues: _clamp((0.30 - _contrast(obs)) / 0.30),
    },
    {
        "id": "jastrow",
        "name": "Jastrow",
        "kind": "geometry",
        "applies_to": "rectangle",
        "formula": "factor = 1 - 0.07 * convergence",
        "max_correction": 0.07,
        "_sev": lambda obs, cues: _conv(obs),
    },
    {
        "id": "zollner",
        "name": "Zöllner",
        "kind": "geometry",
        "applies_to": "rectangle",
        "formula": "factor = 1 - 0.06 * mean(convergence, arrowhead)",
        "max_correction": 0.06,
        "_sev": lambda obs, cues: _clamp(0.5 * (_conv(obs) + _arrow(obs))),
    },
    {
        "id": "poggendorff",
        "name": "Poggendorff",
        "kind": "geometry",
        "applies_to": "rectangle",
        "formula": "factor = 1 - 0.06 * arrowhead",
        "max_correction": 0.06,
        "_sev": lambda obs, cues: _arrow(obs),
    },
    {
        "id": "cafe_wall",
        "name": "Café wall",
        "kind": "light",
        "applies_to": "both",
        "formula": "factor = 1 - 0.05 * (0.5*(1-contrast) + 0.5*specular)",
        "max_correction": 0.05,
        "_sev": lambda obs, cues: _clamp(0.5 * (1.0 - _contrast(obs)) + 0.5 * _spec(obs)),
    },
]


def _apply(registry, obs, cues):
    entries = []
    combined = 1.0
    for spec in registry:
        severity = _clamp(spec["_sev"](obs, cues))
        factor = 1.0 - spec["max_correction"] * severity
        combined *= factor
        entries.append({
            "id": spec["id"], "name": spec["name"], "kind": spec["kind"],
            "applies_to": spec["applies_to"], "severity": round(severity, 4),
            "factor": round(factor, 4), "formula": spec["formula"],
        })
    return entries, combined


def none(obs, cues):
    return {"illusions": [], "top": [], "combined_factor": 1.0,
            "correction_percent": 0.0, "model": "none"}


def canonical5(obs, cues):
    report = evaluate_canonical(obs, cues)
    report["model"] = "canonical5"
    return report


def extended10(obs, cues):
    base = evaluate_canonical(obs, cues)
    extra, extra_factor = _apply(EXTRA_REGISTRY, obs, cues)
    entries = base["illusions"] + extra
    combined = _clamp(base["combined_factor"] * extra_factor, 0.2, 1.2)
    ranked = sorted(entries, key=lambda e: e["severity"], reverse=True)
    return {
        "illusions": entries,
        "top": ranked[:3],
        "combined_factor": round(combined, 4),
        "correction_percent": round((1.0 - combined) * 100.0, 2),
        "model": "extended10",
    }


ILLUSION_MODELS = {
    "none": none,
    "canonical5": canonical5,
    "extended10": extended10,
}

ALL_ILLUSIONS = ILLUSION_REGISTRY + EXTRA_REGISTRY
