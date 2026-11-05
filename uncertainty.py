"""lidarv2 uncertainty engine — inverse-variance cue fusion + confidence intervals.

The backend never reports a bare depth. Every estimate carries an interval
whose width is driven by:

  * within-cue measurement noise (pixel quantisation, calibration error), and
  * between-cue disagreement (a random-effect term), so that cues which
    disagree inflate the interval instead of silently averaging away.

Pure stdlib (`statistics.NormalDist`, `random`).
"""

import math
import random
from statistics import NormalDist

from models import CueEstimate, DepthEstimate


def z_for_level(level):
    """Two-sided z score for a confidence level (e.g. 0.95 -> 1.96)."""
    level = min(max(float(level), 0.01), 0.9999)
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def _confidence_from_sigma(mu, sigma):
    if mu <= 0.0:
        return 0.0
    relative = sigma / mu
    return max(0.0, min(1.0, 1.0 - relative))


def _prepare(cues):
    out = []
    for c in cues or []:
        if c is None:
            continue
        if not (c.depth_m > 0.0) or not (c.sigma_m > 0.0):
            continue
        out.append(c)
    return out


def informative_cues(cues, rel=1e-2):
    """Cues whose inverse-variance weight is not negligible.

    A far-ground cue can fire with a huge sigma when an object is held above
    the floor. It carries almost no weight, so it must not be allowed to
    inflate the interval or be mistaken for a genuine cross-cue conflict.
    """
    usable = _prepare(cues)
    if not usable:
        return []
    weights = [c.weight / (c.sigma_m ** 2) for c in usable]
    wmax = max(weights)
    return [c for c, w in zip(usable, weights) if w >= wmax * rel]


def fuse_cues(cues, level=0.95, between_scale=1.0, method="inverse_variance"):
    """Fuse independent depth cues into a DepthEstimate.

    Inverse-variance weighting gives the point estimate. A method-of-moments
    between-cue variance term is added when two or more informative cues
    genuinely disagree, which is what makes disagreement widen the interval.
    """
    cues = _prepare(cues)
    if not cues:
        raise ValueError("no usable depth cues (need depth_m > 0 and sigma_m > 0)")

    weights = [c.weight / (c.sigma_m ** 2) for c in cues]
    wsum = sum(weights)
    mu = sum(c.depth_m * w for c, w in zip(cues, weights)) / wsum
    var_within = 1.0 / wsum

    useful = informative_cues(cues)
    if len(useful) >= 2:
        w_inf = [c.weight / (c.sigma_m ** 2) for c in useful]
        wsum_inf = sum(w_inf)
        scatter = sum(w * (c.depth_m - mu) ** 2 for c, w in zip(useful, w_inf)) / wsum_inf
        mean_var = sum(c.sigma_m ** 2 for c in useful) / len(useful)
        var_between = max(0.0, scatter - mean_var)
    else:
        var_between = 0.0

    var = var_within + between_scale * var_between
    sigma = math.sqrt(var)
    z = z_for_level(level)
    lo = max(0.0, mu - z * sigma)
    hi = mu + z * sigma
    return DepthEstimate(
        depth_m=mu,
        ci_low_m=lo,
        ci_high_m=hi,
        level=level,
        sigma_m=sigma,
        confidence=_confidence_from_sigma(mu, sigma),
        cues=cues,
        method=method,
    )


def bootstrap_ci(cues, level=0.95, n=500, seed=0):
    """Percentile bootstrap interval over the cue set.

    Useful as a robustness check on the analytic interval; a large gap between
    the two is itself a signal that the cues disagree.
    """
    cues = _prepare(cues)
    if not cues:
        raise ValueError("no usable depth cues")
    rng = random.Random(seed)
    values = []
    for _ in range(max(1, n)):
        sample = [rng.choice(cues) for _ in cues]
        w = [1.0 / (c.sigma_m ** 2) for c in sample]
        values.append(sum(c.depth_m * wi for c, wi in zip(sample, w)) / sum(w))
    values.sort()
    lo_idx = int((1.0 - level) / 2.0 * n)
    hi_idx = min(n - 1, int((1.0 + level) / 2.0 * n))
    return values[max(0, lo_idx)], values[hi_idx]


def interval_width_ratio(estimate):
    """Half-width / depth — a scale-free precision score."""
    if estimate.depth_m <= 0:
        return float("inf")
    return estimate.ci_half_m / estimate.depth_m
