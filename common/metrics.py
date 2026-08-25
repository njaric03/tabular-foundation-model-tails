# -*- coding: utf-8 -*-
"""Tail-index estimators and scoring rules, in one place.

These bodies existed in up to 19 copies across the experiment scripts, which is
how two of them drifted apart:

* the GPD quantile function guarded its denominator in only half the copies, so
  the rest divided by zero at xi = 0 and produced a warning that the
  filterwarnings("ignore") at the top of every script swallowed;
* the Hill estimator used the smallest INCLUDED order statistic as its
  threshold in two of three copies, which makes one term of the average
  identically zero and biases xi low by about xi/k. See `hill` below.

All estimators here take and return plain arrays; nothing reads the filesystem.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import genpareto

# Quantile levels most experiments ask a model for.
LEVELS = [0.5, 0.9, 0.99]


# --------------------------------------------------------------- GPD quantiles

def gpd_quantile(a, xi):
    """Quantile function of the standard GPD (scale 1), with the Gumbel limit at xi -> 0.

    The denominator is guarded rather than trusting np.where: `where` evaluates
    BOTH branches before selecting, so at xi = 0 the rejected branch divides by
    zero. The result is still correct, but the RuntimeWarning that would be the
    only warning sign is suppressed repo-wide.
    """
    a = np.asarray(a, dtype=float)
    xi = np.asarray(xi, dtype=float)
    near_zero = np.abs(xi) < 1e-9
    xi_safe = np.where(near_zero, 1.0, xi)
    return np.where(near_zero, -np.log(1 - a), ((1 - a) ** (-xi_safe) - 1) / xi_safe)


def xi_from_ratio(ratio, hi=0.99, lo=0.9, bracket=(-0.9, 6.0)):
    """Invert the raw quantile ratio Q(hi)/Q(lo) into a tail index.

    Returns NaN when the ratio falls outside the bracket, which happens exactly
    on the heaviest tails. Callers must filter on the success flag of the
    measurement, not on notna() of this value, or they drop the heavy tails
    preferentially.
    """
    def f(x):
        return float(gpd_quantile(hi, x) / gpd_quantile(lo, x)) - ratio
    try:
        return brentq(f, *bracket)
    except Exception:
        return np.nan


def xi_from_residual_ratio(ratio, hi=0.99, lo=0.9, mid=0.5, bracket=(-0.9, 6.0)):
    """Invert the location-invariant ratio (Q(hi)-Q(mid)) / (Q(lo)-Q(mid)).

    The raw ratio above is biased low whenever the distribution has a location
    component, because the location does not cancel. Here it does. Validated
    against a known xi: median absolute error 0.071 -> 0.006
    (findings/h1/xi_residual.md).
    """
    def f(x):
        return float((gpd_quantile(hi, x) - gpd_quantile(mid, x)) /
                     (gpd_quantile(lo, x) - gpd_quantile(mid, x))) - ratio
    try:
        return brentq(f, *bracket)
    except Exception:
        return np.nan


def truncated_mean_std(t, xi):
    """E[C | C <= t] for the standard GPD(xi, 1). t may be a vector.

    Separates a structural mean deficit, where the model cannot extrapolate past
    the largest training value, from a model-specific one.
    """
    t = np.asarray(t, dtype=float)
    S = (1 + xi * t) ** (-1.0 / xi)
    E_min = (1 - (1 + xi * t) ** (1 - 1.0 / xi)) / (1 - xi)
    F = 1 - S
    return np.where(F > 1e-12, (E_min - t * S) / np.maximum(F, 1e-12), t)


# ------------------------------------------------------- estimators from a sample

def hill(y, k_share=0.10, k_min=30):
    """Hill tail-index estimator on the top k observations.

    The threshold is the (k+1)-th largest value, y[-k-1], NOT the smallest
    included one. Using y[-k] makes one term of the average identically zero and
    biases xi low by about xi/k: at the k = 30 floor that is -0.017 at xi = 0.5
    and -0.030 at xi = 0.9; at a typical k = 500 it is -0.001. Two of the three
    copies in the repo had the biased form.
    """
    y = np.asarray(y, dtype=float)
    y = np.sort(y[y > 0])
    k = max(int(len(y) * k_share), k_min)
    if len(y) <= k:
        return np.nan
    return float(np.mean(np.log(y[-k:]) - np.log(y[-k - 1])))


def gpd_mle(z, frac=0.10, min_exceedances=50):
    """Maximum-likelihood xi of a GPD fitted to peaks over the (1-frac) quantile."""
    z = np.asarray(z, dtype=float)
    u = np.quantile(z, 1 - frac)
    excess = z[z > u] - u
    if len(excess) < min_exceedances:
        return np.nan
    return float(genpareto.fit(excess, floc=0)[0])


def residuals(X, y):
    """y divided by a fitted log-linear conditional scale exp(X @ b).

    Removes the part of the spread the covariates explain through the scale, so
    what is left carries the shape. Only positive y take part.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = y > 0
    D = np.c_[np.ones(ok.sum()), X[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(y[ok]), rcond=None)
    return y[ok] / np.exp(D @ b)


def sd_shift(sample):
    """How much the largest point inflates the sd: sd(all) / sd(without max).

    This is the causal quantity behind the leverage finding, as opposed to the
    max-over-second ratio, which conditions on the wrong thing; see section 4.2b
    of findings/findings/NALAZI.md. ddof=0 throughout, consistently with every other use.
    """
    s = np.sort(np.asarray(sample, dtype=float))
    return float(np.std(s) / max(np.std(s[:-1]), 1e-12))


# ------------------------------------------------------------- scoring rules

def pinball_all(q, y, levels=LEVELS):
    """Per-point, per-level pinball loss. Shape (n, len(levels))."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y[:, None] - q
    a = np.asarray(levels, dtype=float)[None, :]
    return np.where(d >= 0, a * d, (a - 1.0) * d)


def pinball_at(q, y, level, levels=LEVELS):
    """Mean pinball loss at the level closest to the requested one."""
    i = int(np.argmin(np.abs(np.asarray(levels) - level)))
    return float(pinball_all(q[:, [i]], y, [levels[i]]).mean())


def crps(q, y, levels=LEVELS):
    """CRPS approximated from a quantile grid: twice the mean pinball loss."""
    return float(2.0 * pinball_all(q, y, levels).mean())


def twcrps(q, y, threshold, levels=LEVELS):
    """Threshold-weighted CRPS with chaining function v(y) = max(y, t)."""
    return crps(np.maximum(q, threshold), np.maximum(y, threshold), levels)


def mean_from_quantiles(q, levels=LEVELS):
    """Mean as the trapezoidal integral of the quantile function over levels."""
    a = np.asarray(levels, dtype=float)
    a_full = np.concatenate([[0.0], a, [1.0]])
    q_full = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)
    # numpy 2 renamed trapz to trapezoid; the venvs run different majors.
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trapz(q_full, a_full, axis=1)


def gamma_deviance(mu, y):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    return float(2.0 * np.mean((y - mu) / mu - np.log(y / mu)))


def rel_change(before, after):
    """Median relative change, as a fraction. The response to an intervention."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    ok = np.abs(before) > 1e-12
    if not ok.any():
        return np.nan
    return float(np.median((after[ok] - before[ok]) / before[ok]))


def share_tracked(model_lo, model_hi, true_lo, true_hi, min_range=0.05):
    """Fraction of a true gradient the model reproduces, across two groups.

    Returns NaN when the reference gradient is too small to divide by. Without
    that guard the ratio explodes: on real data the shape gradient is weak and
    the quotient ranged from -2.80 to 6.16, which is why the real-data claim is
    made by rank rather than as a percentage.
    """
    span = float(true_hi) - float(true_lo)
    if not np.isfinite(span) or abs(span) < min_range:
        return np.nan
    return float((float(model_hi) - float(model_lo)) / span)
