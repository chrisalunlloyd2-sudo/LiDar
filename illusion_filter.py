"""lidarv2 illusion filter — reject the top optical illusions.

Monocular depth is ambiguous, and several classical illusions make a naive
estimator confidently wrong. This module scores independent illusion
signatures, combines them with a noisy-OR, and rejects observations whose
risk is too high. The flags are ranked so the UI can show *which* illusions
were filtered.

Signatures
----------
cue_conflict       cues disagree far beyond their own error bars
size_constancy     known-size cue fights the angular cue
ponzo              converging perspective lines inflate apparent size
muller_lyer        arrowhead terminators distort length
reflection         specular / mirror surface (shows a false 3-D image)
aerial_perspective low-contrast haze flattens depth
foreshortening     extreme aspect ratio (object nearly edge-on)
"""

from models import IllusionFlag, IllusionReport
from uncertainty import informative_cues


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def analyze(obs, cues, accept_threshold=0.65):
    """Score illusions for one observation and decide accept/reject."""
    flags = []

    # 1. Cross-cue depth conflict (informative cues only).
    useful = informative_cues(cues)
    if len(useful) >= 2:
        depths = [c.depth_m for c in useful]
        sigmas = [max(c.sigma_m, 1e-6) for c in useful]
        spread = max(depths) - min(depths)
        mean_sigma = sum(sigmas) / len(sigmas)
        z = spread / (2.0 * mean_sigma + 1e-9)
        severity = _clamp((z - 1.0) / 3.0)
        if severity > 0.05:
            flags.append(IllusionFlag("cue_conflict", "Cross-cue depth conflict", severity))

    # 2. Size-constancy conflict (known-size vs angular).
    by_name = {c.name: c for c in cues}
    if "pixel_scaling" in by_name and "angular_size" in by_name:
        a = by_name["pixel_scaling"].depth_m
        b = by_name["angular_size"].depth_m
        rel = abs(a - b) / max(a, b, 1e-9)
        severity = _clamp((rel - 0.10) / 0.40)
        if severity > 0.05:
            flags.append(IllusionFlag("size_constancy", "Size-constancy conflict", severity))

    # 3. Ponzo / perspective convergence.
    if obs.convergence is not None:
        severity = _clamp(obs.convergence)
        if severity > 0.2:
            flags.append(IllusionFlag("ponzo", "Ponzo perspective convergence", severity))

    # 4. Müller-Lyer arrowheads.
    if obs.arrowhead is not None:
        severity = _clamp(obs.arrowhead)
        if severity > 0.2:
            flags.append(IllusionFlag("muller_lyer", "Muller-Lyer arrowheads", severity))

    # 5. Specular / mirror surface: bright, smooth patch fakes a solid object.
    if obs.specular is not None:
        spec = _clamp(obs.specular)
        texture = 0.5 if obs.texture_density is None else _clamp(obs.texture_density)
        severity = _clamp(spec * (1.0 - texture))
        if severity > 0.15:
            flags.append(IllusionFlag("reflection", "Specular/mirror illusion", severity))

    # 6. Aerial perspective: very low contrast flattens depth.
    if obs.contrast is not None:
        severity = _clamp((0.12 - _clamp(obs.contrast)) / 0.12)
        if severity > 0.2:
            flags.append(IllusionFlag("aerial_perspective", "Low-contrast haze", severity))

    # 7. Foreshortening: object nearly edge-on to the camera.
    if obs.object_px_width > 0:
        aspect = obs.object_px_height / obs.object_px_width
        if aspect > 6.0:
            flags.append(IllusionFlag("foreshortening", "Extreme foreshortening", _clamp((aspect - 6.0) / 10.0)))
        elif aspect < 0.08:
            flags.append(IllusionFlag("foreshortening", "Extreme foreshortening", _clamp((0.08 - aspect) / 0.08)))

    flags.sort(key=lambda f: f.severity, reverse=True)

    # Noisy-OR risk; each flag contributes at most 0.75 so a single weak flag
    # cannot reject on its own.
    survive = 1.0
    for f in flags:
        survive *= (1.0 - 0.75 * f.severity)
    risk = _clamp(1.0 - survive)

    hard_reject = any(f.code == "cue_conflict" and f.severity >= 0.75 for f in flags)
    rejected = hard_reject or risk >= accept_threshold
    if hard_reject:
        reason = "cue conflict exceeds tolerance"
    elif rejected:
        reason = "illusion risk %.2f >= %.2f" % (risk, accept_threshold)
    else:
        reason = ""
    return IllusionReport(risk=risk, flags=flags, rejected=rejected, reason=reason)


def top_flags(report, limit=3):
    return report.flags[:limit]
