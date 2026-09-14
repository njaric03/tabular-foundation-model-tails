# -*- coding: utf-8 -*-
"""Tail-index estimators and scoring rules. Plain arrays in and out, no file access."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import genpareto

# The quantile levels most experiments ask a model for.
LEVELS = [0.5, 0.9, 0.99]


# ------------------------------------------------------------------ tail index

def gpd_quantile(a, xi):
    """Quantile function of the standard GPD (scale 1), with the exponential limit at xi = 0.

    The denominator is guarded explicitly: `np.where` evaluates both branches, so at
    xi = 0 the rejected one would still divide by zero and warn.
    """
    a = np.asarray(a, dtype=float)
    xi = np.asarray(xi, dtype=float)
    near_zero = np.abs(xi) < 1e-9
    xi_safe = np.where(near_zero, 1.0, xi)
    return np.where(near_zero, -np.log(1 - a), ((1 - a) ** (-xi_safe) - 1) / xi_safe)


def xi_from_ratio(ratio, hi=0.99, lo=0.9, bracket=(-0.9, 6.0)):
    """The xi whose GPD has Q(hi) / Q(lo) = ratio, or NaN outside the bracket.

    NaN comes exactly on the heaviest tails, so filter rows on their success flag and
    not on notna() of this value.
    """
    def f(x):
        return float(gpd_quantile(hi, x) / gpd_quantile(lo, x)) - ratio
    try:
        return brentq(f, *bracket)
    except Exception:
        return np.nan


def xi_from_residual_ratio(ratio, hi=0.99, lo=0.9, mid=0.5, bracket=(-0.9, 6.0)):
    """The xi whose GPD has (Q(hi) - Q(mid)) / (Q(lo) - Q(mid)) = ratio, or NaN.

    A location shift cancels in this ratio and biases the raw one low. On samples with
    a known xi the median absolute error falls from 0.071 to 0.006.
    """
    def f(x):
        return float((gpd_quantile(hi, x) - gpd_quantile(mid, x)) /
                     (gpd_quantile(lo, x) - gpd_quantile(mid, x))) - ratio
    try:
        return brentq(f, *bracket)
    except Exception:
        return np.nan


def implied_xi(q_hi, q_lo, q_mid=None) -> float:
    """Tail index implied by predicted Q(0.99) and Q(0.9), from the median over rows.

    Without `q_mid` the ratio Q(0.99) / Q(0.9), over rows with Q(0.9) > 1e-9. With it
    the location-invariant (Q(0.99) - Q(0.5)) / (Q(0.9) - Q(0.5)), over rows where the
    denominator exceeds 1e-12. NaN when no row qualifies.
    """
    q_hi, q_lo = np.asarray(q_hi, float), np.asarray(q_lo, float)
    if q_mid is None:
        ok = q_lo > 1e-9
        return xi_from_ratio(float(np.median(q_hi[ok] / q_lo[ok]))) if ok.any() else np.nan
    q_mid = np.asarray(q_mid, float)
    den = q_lo - q_mid
    ok = den > 1e-12
    if not ok.any():
        return np.nan
    return xi_from_residual_ratio(float(np.median((q_hi[ok] - q_mid[ok]) / den[ok])))


def truncated_mean_std(t, xi):
    """E[C | C <= t] for C ~ GPD(xi, 1); `t` may be a vector."""
    t = np.asarray(t, dtype=float)
    S = (1 + xi * t) ** (-1.0 / xi)
    E_min = (1 - (1 + xi * t) ** (1 - 1.0 / xi)) / (1 - xi)
    F = 1 - S
    return np.where(F > 1e-12, (E_min - t * S) / np.maximum(F, 1e-12), t)


def hill(y, k_share=0.10, k_min=30):
    """Hill estimator on the top k = max(k_share * n, k_min) positive values.

    The threshold is the (k+1)-th largest value. Taking the k-th makes one term of the
    mean zero and biases xi low by about xi / k, which two earlier copies did.
    """
    y = np.asarray(y, dtype=float)
    y = np.sort(y[y > 0])
    k = max(int(len(y) * k_share), k_min)
    if len(y) <= k:
        return np.nan
    return float(np.mean(np.log(y[-k:]) - np.log(y[-k - 1])))


def gpd_mle(z, frac=0.10, min_exceedances=50):
    """Maximum-likelihood xi of a GPD fitted to the peaks over the (1 - frac) quantile."""
    z = np.asarray(z, dtype=float)
    u = np.quantile(z, 1 - frac)
    excess = z[z > u] - u
    if len(excess) < min_exceedances:
        return np.nan
    return float(genpareto.fit(excess, floc=0)[0])


def residuals(X, y):
    """The positive y divided by a fitted log-linear scale exp(a + X @ b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = y > 0
    D = np.c_[np.ones(ok.sum()), X[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(y[ok]), rcond=None)
    return y[ok] / np.exp(D @ b)


# ------------------------------------------------------------------ leverage

def sd_shift(sample):
    """sd(sample) / sd(sample without its largest value), both with ddof=0.

    The size of a leverage row. The ratio of the largest to the second largest value
    conditions on the wrong thing.
    """
    s = np.sort(np.asarray(sample, dtype=float))
    return float(np.std(s) / max(np.std(s[:-1]), 1e-12))


def y0_for_sd_shift(y, target, hi_factor=1e9):
    """The added value that makes sd_shift(y + [y0]) equal `target`.

    A dose of `dose * max(y)` is not a fixed treatment, because max(y) is itself
    heavy-tailed: at xi = 0.9 one nominal dose of 100 gave shifts from 50 to 97 across
    five seeds. Dosing on the shift fixes the treatment and puts the generator on the
    axis of the prevalence survey. NaN when a duplicate of the maximum already exceeds
    the target, since that is the smallest shift one added value can cause.
    """
    y = np.asarray(y, dtype=float)
    lo = float(y.max())
    if sd_shift(np.append(y, lo)) > target:
        return np.nan
    hi = lo if lo > 0 else 1.0
    for _ in range(64):
        hi *= 10.0
        if sd_shift(np.append(y, hi)) >= target:
            break
        if hi > abs(lo) * hi_factor:
            return np.nan
    return float(brentq(lambda v: sd_shift(np.append(y, v)) - target, lo, hi))


# ------------------------------------------------------------------ scoring rules

def pinball(y, q, level) -> float:
    """Mean pinball loss of one vector of quantiles at one level."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


def pinball_all(q, y, levels=LEVELS):
    """Per-row, per-level pinball loss, shape (n, len(levels))."""
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y[:, None] - q
    a = np.asarray(levels, dtype=float)[None, :]
    return np.where(d >= 0, a * d, (a - 1.0) * d)


def pinball_at(q, y, level, levels=LEVELS):
    """Mean pinball loss at the level of `levels` closest to `level`."""
    i = int(np.argmin(np.abs(np.asarray(levels) - level)))
    return float(pinball_all(q[:, [i]], y, [levels[i]]).mean())


def crps(q, y, levels=LEVELS):
    """CRPS approximated on a quantile grid: twice the mean pinball loss."""
    return float(2.0 * pinball_all(q, y, levels).mean())


def twcrps(q, y, threshold, levels=LEVELS):
    """Threshold-weighted CRPS with the chaining function v(y) = max(y, threshold)."""
    return crps(np.maximum(q, threshold), np.maximum(y, threshold), levels)


def mean_from_quantiles(q, levels=LEVELS):
    """The mean as the trapezoidal integral of the quantile function over `levels`."""
    a_full = np.concatenate([[0.0], np.asarray(levels, dtype=float), [1.0]])
    q_full = np.concatenate([q[:, :1], q, q[:, -1:]], axis=1)
    # numpy 2 renamed trapz to trapezoid, and the venvs run different majors.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(q_full, a_full, axis=1)


def gamma_deviance(mu, y):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    return float(2.0 * np.mean((y - mu) / mu - np.log(y / mu)))


def rel_change_by_level(q_ref, q, positive=False) -> list[float]:
    """Median relative change of each column of `q` against the same column of `q_ref`.

    Rows whose reference is within 1e-9 of zero are skipped, and with `positive` every
    reference at or below 1e-9.
    """
    q_ref, q = np.asarray(q_ref, dtype=float), np.asarray(q, dtype=float)
    out = []
    for a, b in zip(q_ref.T, q.T):
        ok = a > 1e-9 if positive else np.abs(a) > 1e-9
        out.append(float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan)
    return out
