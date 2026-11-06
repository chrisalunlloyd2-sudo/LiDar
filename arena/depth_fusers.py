"""lidarv2 depth fusers — 16 ways to turn cues into depth + a CI.

Over-development on purpose: the arena benchmarks every method on the premade
fixtures and `arena.snip()` keeps the ones that actually help. All pure stdlib.
"""

import math
import random
from dataclasses import asdict
from statistics import median

from models import CueEstimate
from uncertainty import fuse_cues, z_for_level


def _clean(cues):
    return [c for c in cues if c is not None and c.depth_m > 0 and c.sigma_m > 0]


def _result(mu, sigma, level, method, n):
    mu = max(1e-9, float(mu))
    sigma = max(1e-9, float(sigma))
    z = z_for_level(level)
    lo = max(0.0, mu - z * sigma)
    hi = mu + z * sigma
    conf = max(0.0, min(1.0, 1.0 - sigma / mu))
    return {
        "depth_m": mu,
        "ci_low_m": lo,
        "ci_high_m": hi,
        "sigma_m": sigma,
        "confidence": conf,
        "level": level,
        "method": method,
        "n_cues": n,
    }


def _mad_scale(values, center):
    dev = [abs(v - center) for v in values]
    mad = median(dev) if dev else 0.0
    return max(1e-9, 1.4826 * mad)


# ── classical ──────────────────────────────────────────────────────────
def inverse_variance(cues, level=0.95):
    return asdict(fuse_cues(_clean(cues), level=level))


def weighted_mean(cues, level=0.95):
    cs = _clean(cues)
    w = [c.weight for c in cs]
    wsum = sum(w) or 1.0
    mu = sum(c.depth_m * wi for c, wi in zip(cs, w)) / wsum
    var = sum((wi ** 2) * (c.sigma_m ** 2) for c, wi in zip(cs, w)) / (wsum ** 2)
    return _result(mu, math.sqrt(var), level, "weighted_mean", len(cs))


def median_fuser(cues, level=0.95):
    cs = _clean(cues)
    d = [c.depth_m for c in cs]
    mu = median(d)
    return _result(mu, _mad_scale(d, mu) / math.sqrt(max(1, len(d))),
                   level, "median", len(cs))


def trimmed_mean(cues, level=0.95, trim=0.2):
    cs = _clean(cues)
    d = sorted(c.depth_m for c in cs)
    k = int(len(d) * trim)
    core = d[k:len(d) - k] or d
    mu = sum(core) / len(core)
    var = sum((x - mu) ** 2 for x in core) / max(1, len(core) - 1)
    return _result(mu, math.sqrt(var) / math.sqrt(len(core)), level, "trimmed_mean", len(cs))


def winsorized(cues, level=0.95, pct=0.10):
    cs = _clean(cues)
    d = sorted(c.depth_m for c in cs)
    k = max(1, int(len(d) * pct))
    lo, hi = d[k - 1], d[len(d) - k]
    clipped = [min(max(x, lo), hi) for x in d]
    mu = sum(clipped) / len(clipped)
    var = sum((x - mu) ** 2 for x in clipped) / max(1, len(clipped) - 1)
    return _result(mu, math.sqrt(var) / math.sqrt(len(clipped)), level, "winsorized", len(cs))


def geometric_mean(cues, level=0.95):
    cs = _clean(cues)
    logs = [math.log(c.depth_m) for c in cs]
    mu_log = sum(logs) / len(logs)
    mu = math.exp(mu_log)
    var = sum((x - mu_log) ** 2 for x in logs) / max(1, len(logs) - 1)
    return _result(mu, mu * math.sqrt(var / len(logs)), level, "geometric_mean", len(cs))


def harmonic_mean(cues, level=0.95):
    cs = _clean(cues)
    inv = sum(1.0 / c.depth_m for c in cs)
    mu = len(cs) / inv
    var = sum((c.sigma_m / c.depth_m ** 2) ** 2 for c in cs)
    sigma = (mu ** 2) * math.sqrt(var)
    return _result(mu, sigma, level, "harmonic_mean", len(cs))


# ── robust M-estimators ────────────────────────────────────────────────
def _irls(cues, weight_fn, method, level=0.95, iters=30):
    cs = _clean(cues)
    d = [c.depth_m for c in cs]
    s = [c.sigma_m for c in cs]
    mu = median(d)
    scale = _mad_scale(d, mu)
    w = [1.0 / (si * si) for si in s]
    for _ in range(iters):
        w = [weight_fn((x - mu) / scale) / (si * si) for x, si in zip(d, s)]
        wsum = sum(w)
        if wsum <= 0:
            break
        mu_new = sum(wi * x for wi, x in zip(w, d)) / wsum
        if abs(mu_new - mu) < 1e-12:
            mu = mu_new
            break
        mu = mu_new
    wsum = sum(w) or 1.0
    return _result(mu, math.sqrt(1.0 / wsum), level, method, len(cs))


def huber(cues, level=0.95):
    def w(r):
        a = abs(r)
        return 1.0 if a <= 1.345 else 1.345 / a
    return _irls(cues, w, "huber", level)


def tukey_biweight(cues, level=0.95):
    def w(r):
        a = abs(r)
        return (1.0 - (a / 4.685) ** 2) ** 2 if a < 4.685 else 0.0
    return _irls(cues, w, "tukey_biweight", level)


def cauchy(cues, level=0.95):
    def w(r):
        return 1.0 / (1.0 + (r / 2.385) ** 2)
    return _irls(cues, w, "cauchy", level)


def student_t(cues, level=0.95):
    def w(r):
        return 4.0 / (3.0 + (r / 2.0) ** 2)
    return _irls(cues, w, "student_t", level)


# ── Bayesian / sequential / consensus ──────────────────────────────────
def bayesian(cues, level=0.95):
    cs = _clean(cues)
    d = [c.depth_m for c in cs]
    mu0 = median(d)
    sigma0 = max(d) * 0.5
    prec = 1.0 / (sigma0 ** 2)
    num = mu0 / (sigma0 ** 2)
    for c in cs:
        prec += 1.0 / (c.sigma_m ** 2)
        num += c.depth_m / (c.sigma_m ** 2)
    return _result(num / prec, math.sqrt(1.0 / prec), level, "bayesian", len(cs))


def kalman(cues, level=0.95):
    cs = _clean(cues)
    mu = cs[0].depth_m
    var = cs[0].sigma_m ** 2
    for c in cs[1:]:
        k = var / (var + c.sigma_m ** 2)
        mu += k * (c.depth_m - mu)
        var *= (1.0 - k)
    return _result(mu, math.sqrt(var), level, "kalman", len(cs))


def ransac(cues, level=0.95):
    cs = _clean(cues)
    best, best_n = None, -1
    for hyp in cs:
        inliers = [c for c in cs if abs(c.depth_m - hyp.depth_m) <= 2.5 * hyp.sigma_m]
        if len(inliers) > best_n:
            best, best_n = inliers, len(inliers)
    best = best or cs
    wsum = sum(1.0 / (c.sigma_m ** 2) for c in best)
    mu = sum(c.depth_m / (c.sigma_m ** 2) for c in best) / wsum
    return _result(mu, math.sqrt(1.0 / wsum), level, "ransac", len(cs))


def dempster_shafer(cues, level=0.95):
    cs = _clean(cues)
    w = [1.0 / (c.sigma_m ** 2) for c in cs]
    wsum = sum(w)
    mu = sum(wi * c.depth_m for wi, c in zip(w, cs)) / wsum
    conflict = 1.0 - sum(wi * wi for wi in w) / (wsum * wsum)
    sigma = math.sqrt(1.0 / wsum) * (1.0 + max(0.0, conflict))
    return _result(mu, sigma, level, "dempster_shafer", len(cs))


def bootstrap(cues, level=0.95, n=300, seed=7):
    cs = _clean(cues)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [rng.choice(cs) for _ in cs]
        wsum = sum(1.0 / (c.sigma_m ** 2) for c in sample)
        means.append(sum(c.depth_m / (c.sigma_m ** 2) for c in sample) / wsum)
    means.sort()
    mu = median(means)
    lo = means[max(0, int((1 - level) / 2 * n))]
    hi = means[min(n - 1, int((1 + level) / 2 * n))]
    sigma = (hi - lo) / (2 * 1.96)
    return _result(mu, sigma, level, "bootstrap", len(cs))


def particle(cues, level=0.95, n=600, seed=11):
    cs = _clean(cues)
    rng = random.Random(seed)
    samples, weights = [], []
    for _ in range(n):
        c = rng.choice(cs)
        samples.append(rng.gauss(c.depth_m, c.sigma_m))
        weights.append(1.0 / (c.sigma_m ** 2))
    wsum = sum(weights)
    mu = sum(s * w for s, w in zip(samples, weights)) / wsum
    var = sum(w * (s - mu) ** 2 for s, w in zip(samples, weights)) / wsum
    return _result(mu, math.sqrt(var), level, "particle", len(cs))


FUSERS = {
    "inverse_variance": inverse_variance,
    "weighted_mean": weighted_mean,
    "median": median_fuser,
    "trimmed_mean": trimmed_mean,
    "winsorized": winsorized,
    "geometric_mean": geometric_mean,
    "harmonic_mean": harmonic_mean,
    "huber": huber,
    "tukey_biweight": tukey_biweight,
    "cauchy": cauchy,
    "student_t": student_t,
    "bayesian": bayesian,
    "kalman": kalman,
    "ransac": ransac,
    "dempster_shafer": dempster_shafer,
    "bootstrap": bootstrap,
    "particle": particle,
}
