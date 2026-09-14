# -*- coding: utf-8 -*-
"""Tail and scale gradients across the terciles of a feature, for the dissociation (H1).

A gradient is the change of a quantity from the bottom tercile of a feature to the top
one: the tail index for the shape, the log median for the scale. On the generator the
truth is known; on real data the reference is estimated from held-out rows.
"""
from __future__ import annotations

import numpy as np

from common import generator, metrics

N_TERCILES = 3
# A tercile with fewer usable rows gives no Hill estimate.
MIN_PER_TERCILE = 400
LEVELS = [0.5, 0.9, 0.99]
# The fit/test split and bootstrap count of the real-data dissociation, shared with the
# calibration on the generator so both measure the same procedure.
FIT_ROWS, TEST_ROWS, N_BOOT = 3000, 6000, 100


def tercile_edges(values, k=N_TERCILES) -> np.ndarray:
    return np.quantile(values, np.linspace(0, 1, k + 1))


def tercile_of(values, edges) -> np.ndarray:
    """Index 0..k-1 of the tercile each value falls in."""
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


def usable_edges(values, k=N_TERCILES):
    """Tercile edges of a feature, or None when it has too few distinct values to split."""
    if len(np.unique(values)) < k * 3:
        return None
    edges = tercile_edges(values, k)
    return edges if len(np.unique(edges)) == k + 1 else None


def generator_terciles(X):
    """Terciles of a generator design by x4, which carries the shape, and by the linear
    predictor of the scale, which carries the level."""
    lin = X @ generator.W / np.linalg.norm(generator.W)
    return (tercile_of(X[:, 4], tercile_edges(X[:, 4])),
            tercile_of(lin, tercile_edges(lin)))


def generator_shares(q, X, shape_truth, median_truth, levels=LEVELS) -> dict:
    """Shape and scale shares of predicted quantiles on a generator test set.

    Shape: the implied-xi change across the outer x4 terciles over the true change,
    where `shape_truth(rows)` is the true tail index of a boolean row selection. Scale:
    the log ratio of the outer terciles' median predictions over that of
    `median_truth`, the true conditional median per row, across terciles of the
    scale's linear predictor.
    """
    shape_t, scale_t = generator_terciles(X)
    top = N_TERCILES - 1
    hi, lo, mid = levels.index(0.99), levels.index(0.9), levels.index(0.5)
    xi_model = [metrics.implied_xi(q[shape_t == t, hi], q[shape_t == t, lo]) for t in (0, top)]
    xi_true = [shape_truth(shape_t == t) for t in (0, top)]
    m_lo, m_hi = np.median(q[scale_t == 0, mid]), np.median(q[scale_t == top, mid])
    t_lo, t_hi = np.median(median_truth[scale_t == 0]), np.median(median_truth[scale_t == top])
    return dict(shape_share=(xi_model[1] - xi_model[0]) / (xi_true[1] - xi_true[0]),
                scale_share=(np.log(m_hi / m_lo) / np.log(t_hi / t_lo)
                             if m_lo > 0 and m_hi > 0 else np.nan),
                xi_t1=xi_model[0], xi_t3=xi_model[1],
                xi_true_t1=xi_true[0], xi_true_t3=xi_true[1])


def log_linear_scale(X, y):
    """exp(a + X @ b) fitted to log(y) on the positive y, as a function of any X."""
    ok = y > 0
    D = np.c_[np.ones(ok.sum()), X[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(y[ok]), rcond=None)
    return lambda Z: np.clip(np.exp(np.c_[np.ones(len(Z)), Z] @ b), 1e-9, None)


def hill_by_tercile(y, s_hat, terc, terciles=range(N_TERCILES), k_share=0.10) -> list:
    """Hill index of the residuals y / s_hat per tercile, NaN below MIN_PER_TERCILE rows."""
    out = []
    for t in terciles:
        z = y[terc == t] / s_hat[terc == t]
        z = z[np.isfinite(z) & (z > 0)]
        out.append(metrics.hill(z, k_share=k_share) if len(z) >= MIN_PER_TERCILE else np.nan)
    return out


def steepest_feature(Xf, yf, s_hat_f, k_share=0.10):
    """The feature whose outer terciles differ most in Hill index, chosen on the fit rows.

    Returns (column, tercile edges, number of usable features). A maximum over columns
    is large even without any dependence, which `reference_instrument.py` measures.
    """
    best, best_range, best_edges, scanned = None, -np.inf, None, 0
    for j in range(Xf.shape[1]):
        edges = usable_edges(Xf[:, j])
        if edges is None:
            continue
        xs = hill_by_tercile(yf, s_hat_f, tercile_of(Xf[:, j], edges), k_share=k_share)
        if not all(np.isfinite(xs)):
            continue
        scanned += 1
        if abs(xs[2] - xs[0]) > best_range:
            best, best_range, best_edges = j, abs(xs[2] - xs[0]), edges
    return best, best_edges, scanned


def reference_gradient(y, s_hat, terc, k_share=0.10) -> float:
    """Hill index on residuals, top tercile minus bottom."""
    lo, hi = hill_by_tercile(y, s_hat, terc, (0, N_TERCILES - 1), k_share)
    return hi - lo


def model_gradient(q, terc, levels=LEVELS) -> float:
    """Location-invariant implied xi of predicted quantiles, top tercile minus bottom."""
    xs = []
    for t in (0, N_TERCILES - 1):
        sel = terc == t
        if sel.sum() < MIN_PER_TERCILE:
            return np.nan
        xs.append(metrics.implied_xi(q[sel, levels.index(0.99)], q[sel, levels.index(0.9)],
                                     q[sel, levels.index(0.5)]))
    return xs[1] - xs[0]


def scale_gradient(lo, hi) -> float:
    """log(hi / lo), NaN unless both are positive."""
    return float(np.log(hi / lo)) if lo > 0 and hi > 0 else np.nan


def slope_through_origin(x, y, var_x=None):
    """Slope of y on x through the origin, optionally corrected for known error in x.

    Returns (raw, corrected, r, error share). The correction divides by
    sum(x^2) - sum(var_x), the spread of the true regressor, and stays NaN when that
    removes more than 95% of it.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan
    den = float(np.sum(x ** 2))
    raw = float(np.sum(x * y) / den) if den > 0 else np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    corrected = var_share = np.nan
    if var_x is not None:
        v = float(np.nansum(np.asarray(var_x, float)[ok]))
        var_share = v / den if den > 0 else np.nan
        if den - v > 0.05 * den:
            corrected = float(np.sum(x * y) / (den - v))
    return raw, corrected, r, var_share


def feature_gradients(Xf, Xte, yte, s_te, q, rng, n_boot, k_share=0.10, xi_te=None):
    """Shape and scale gradients through the terciles of every usable feature.

    Features are split on the fit rows and measured on the test rows. Per feature: the
    reference and model shape gradients with the bootstrap variance of the reference,
    the same for the log median, and the true shape gradient when `xi_te` is given.
    Returns a dict of equally long lists per quantity (the scale lists may be shorter).
    """
    g = {k: [] for k in ("ref_shape", "model_shape", "var_shape", "true_shape",
                         "ref_scale", "model_scale", "var_scale")}
    mid, top = LEVELS.index(0.5), N_TERCILES - 1
    for j in range(Xf.shape[1]):
        edges = usable_edges(Xf[:, j])
        if edges is None:
            continue
        terc = tercile_of(Xte[:, j], edges)
        if min((terc == t).sum() for t in range(N_TERCILES)) < MIN_PER_TERCILE:
            continue
        g_ref = reference_gradient(yte, s_te, terc, k_share)
        g_mod = model_gradient(q, terc)
        if not (np.isfinite(g_ref) and np.isfinite(g_mod)):
            continue

        boot = []
        for _ in range(n_boot):
            k = rng.integers(0, len(yte), len(yte))
            boot.append(reference_gradient(yte[k], s_te[k], terc[k], k_share))
        boot = np.array([b for b in boot if np.isfinite(b)])
        if len(boot) < n_boot // 2:
            continue
        g["ref_shape"].append(g_ref)
        g["model_shape"].append(g_mod)
        g["var_shape"].append(float(np.var(boot)))
        if xi_te is not None:
            g["true_shape"].append(float(xi_te[terc == top].mean() - xi_te[terc == 0].mean()))

        s_ref = scale_gradient(float(np.median(yte[terc == 0])),
                               float(np.median(yte[terc == top])))
        s_mod = scale_gradient(float(np.median(q[terc == 0, mid])),
                               float(np.median(q[terc == top, mid])))
        boot = []
        for _ in range(n_boot):
            k = rng.integers(0, len(yte), len(yte))
            t = terc[k]
            if min((t == u).sum() for u in (0, top)) < 2:
                continue
            boot.append(scale_gradient(float(np.median(yte[k][t == 0])),
                                       float(np.median(yte[k][t == top]))))
        boot = np.array([b for b in boot if np.isfinite(b)])
        if np.isfinite(s_ref) and np.isfinite(s_mod) and len(boot) > 2:
            g["ref_scale"].append(s_ref)
            g["model_scale"].append(s_mod)
            g["var_scale"].append(float(np.var(boot)))
    return g
