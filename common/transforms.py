# -*- coding: utf-8 -*-
"""Transforms of the context target, each with the map that brings a quantile back.

`transform(arm, c, y)` returns the target an arm fits on, the inverse map for the
model's quantiles, and how many context values the arm changed. The arms are the
repairs of part three: a hard cap, logarithms, an outlier test, a derived cap, and
rank maps with and without a Pareto tail.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto, norm, rankdata

from common import metrics

ARMS = ("raw", "clip", "log", "tail_log", "evt_trim", "sd_cap",
        "rank_emp", "rank_gpd", "rank_gpd_trim", "rank_exp")

# Cap of `tail_log` in robust sd above the median, fixed before its first run at the
# value where a hard clip was seen to cost.
TAIL_LOG_C = 50.0

# The sd shift `sd_cap` brings a context down to: the top of the lowest leverage bin
# of part two, fixed before any repair was measured.
TAU = 1.2

# The rank map's GPD threshold and minimum exceedances, and the guards of the trimmed
# arms. Not part of any resume key: changing one needs a new OUTPUT.
RANK_U_SHARE = 0.9
RANK_K_MIN = 30
RANK_TRIM = 1
RANK_XI_CAP = 0.95


def robust_sd(y) -> float:
    """IQR / 1.349: consistent for the sd of a normal, insensitive to outliers."""
    q75, q25 = np.percentile(y, [75, 25])
    return float(max((q75 - q25) / 1.349, 1e-12))


def sd_cap(y, tau=TAU, iters=80):
    """The largest cap whose capped context has an sd shift of at most `tau`, and how
    many values lie above it.

    A context already at or below `tau` comes back untouched, so the rule costs nothing
    where there is nothing to repair. Otherwise bisection between the median, where the
    shift of a capped context is 1, and the maximum, where it is not below `tau`.
    """
    y = np.asarray(y, dtype=float)
    if metrics.sd_shift(y) <= tau:
        return float(y.max()), 0
    lo, hi = float(np.median(y)), float(y.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if metrics.sd_shift(np.minimum(y, mid)) <= tau:
            lo = mid
        else:
            hi = mid
    return lo, int((y > lo).sum())


def evt_trim(y, k_share=0.05, k_min=50, j_max=5, alpha=0.01):
    """Pull down the top values that the tail of the rest cannot have produced.

    Under a Pareto tail the scaled log spacings i * (log X(i) - log X(i+1)), with X(1)
    the largest, are independent exponentials with mean xi (Renyi). xi is estimated
    from the spacings below the `j_max` candidates, a candidate is flagged when its
    spacing exceeds the Bonferroni threshold at `alpha`, and flagged values are capped
    at the largest one not flagged. A simplified form of the trimmed-Hill test of
    Bhattacharya, Kallitsis and Stoev (2019). Returns the new target and how many
    values changed.
    """
    y = np.asarray(y, dtype=float)
    pos = np.sort(y[y > 0])[::-1]
    k = max(k_min, int(k_share * len(pos)))
    if len(pos) < k + 2:
        return y.copy(), 0
    logs = np.log(pos[:k + 1])
    spacings = np.arange(1, k + 1) * (logs[:-1] - logs[1:])
    xi_hat = float(spacings[j_max:].mean())
    if xi_hat <= 0:
        return y.copy(), 0
    threshold = -np.log(alpha / j_max)
    flagged = [i for i in range(1, j_max + 1) if spacings[i - 1] / xi_hat > threshold]
    if not flagged:
        return y.copy(), 0
    cap = pos[max(flagged)]
    return np.minimum(y, cap), int((y > cap).sum())


def rank_map(y, u_share=RANK_U_SHARE, k_min=RANK_K_MIN, with_tail=True,
             scores="normal", trim=0, xi_cap=None):
    """Rank scores into the context, and an empirical inverse spliced to a GPD back out.

    Ranks cannot inflate an sd, so a leverage row stretches nothing. The empirical
    inverse alone saturates at the largest observed value; above the `u_share` quantile
    a GPD fitted to the context's own exceedances replaces it (`with_tail=False` keeps
    the saturation). The GPD is fitted on the margin, not per x, so the top of the range
    comes back smoothed.

    `scores` is "normal" or "exponential". An error in the model's quantile comes back
    amplified like exp(xi * z**2 / 2) through normal scores and like exp(xi * z) through
    exponential ones. `trim` drops that many of the largest exceedances before the fit,
    and `xi_cap` bounds the fitted shape.

    The two branches meet where the empirical inverse itself returns u. With the
    exceedance rate 1 - len(exc) / n instead, the empirical branch ended one order
    statistic above u, the map stepped down at the seam, and `np.searchsorted` on the
    back-transformed borders read garbage.

    Returns the scores, the inverse map, and n, since every value is rewritten.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    p = rankdata(y, method="average") / (n + 1)

    if scores == "normal":
        z, to_p = norm.ppf(p), norm.cdf
    elif scores == "exponential":
        z = -np.log1p(-p)

        def to_p(q):
            return 1.0 - np.exp(-np.maximum(q, 0.0))
    else:
        raise ValueError(scores)

    order = np.argsort(p, kind="mergesort")
    xp, fp = p[order], y[order]
    u = float(np.quantile(y, u_share))
    exc = np.sort(y[y > u]) - u
    zeta = max(1.0 - float(np.interp(u, fp, xp)), 1e-12)
    xi = scale = np.nan
    if with_tail and len(exc) >= k_min:
        try:
            xi, _, scale = genpareto.fit(exc[:-trim] if trim else exc, floc=0.0)
            if xi_cap is not None:
                xi = min(float(xi), float(xi_cap))
        except Exception:
            xi = np.nan

    def back(q):
        pr = np.clip(to_p(np.asarray(q, dtype=float)), 1e-12, 1 - 1e-12)
        out = np.interp(pr, xp, fp)
        if np.isfinite(xi):
            hi = pr > 1 - zeta
            if hi.any():
                r = (1 - pr[hi]) / zeta
                out[hi] = (u + scale * np.log(1 / r) if abs(xi) < 1e-8
                           else u + scale / xi * (r ** (-xi) - 1))
        return out

    return z, back, n


def _identity(q):
    return q


def transform(arm, c, y):
    """The target `arm` fits on, the inverse for its quantiles, and how many values it
    changed. `c` is the cap in robust sd for clip and tail_log, and tau for sd_cap."""
    y = np.asarray(y, dtype=float)
    med, rsd = float(np.median(y)), robust_sd(y)
    if arm == "raw":
        return y, _identity, 0
    if arm == "clip":
        cap = med + c * rsd
        return np.minimum(y, cap), _identity, int((y > cap).sum())
    if arm == "log":
        if (y <= 0).any():
            raise ValueError("log needs a positive target")
        return np.log(y), np.exp, 0
    if arm == "tail_log":
        cap, d = med + c * rsd, c * rsd
        over = y > cap
        z = y.copy()
        z[over] = cap + d * np.log1p((y[over] - cap) / d)

        def back(q):
            return np.where(q > cap, cap + d * np.expm1((q - cap) / d), q)
        return z, back, int(over.sum())
    if arm == "evt_trim":
        z, n = evt_trim(y)
        return z, _identity, n
    if arm == "sd_cap":
        # `c is None`, not `c or TAU`: a tau of 0.0 is a real request.
        cap, n = sd_cap(y, TAU if c is None else c)
        return np.minimum(y, cap), _identity, n
    if arm == "rank_emp":
        return rank_map(y, with_tail=False)
    if arm == "rank_gpd":
        return rank_map(y)
    if arm == "rank_gpd_trim":
        return rank_map(y, trim=RANK_TRIM, xi_cap=RANK_XI_CAP)
    if arm == "rank_exp":
        return rank_map(y, scores="exponential", trim=RANK_TRIM, xi_cap=RANK_XI_CAP)
    raise ValueError(arm)
