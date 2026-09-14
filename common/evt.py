# -*- coding: utf-8 -*-
"""Post-hoc extreme-value graft: keep the model's body, replace its tail by a GPD.

    1. Residuals z = y / m(x) on the training rows, m the model's conditional median.
    2. A GPD fitted to the exceedances of z over its ALPHA0 quantile.
    3. Above ALPHA0, Q(a | x) = Q_model(ALPHA0 | x) + m(x) * GPD increment.

The tail takes its shape from the data and its level from the model, and the splice is
continuous. `scores` returns the metrics the graft experiments compare.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto

from common import metrics

ALPHA0 = 0.90
TAIL_LEVELS = [0.995, 0.999, 0.9995]
LEVELS = sorted(set(np.round(np.arange(1, 100) / 100.0, 4).tolist()
                    + TAIL_LEVELS + [0.5, ALPHA0, 0.99]))
I50, I90, I99, I999 = (LEVELS.index(0.5), LEVELS.index(ALPHA0),
                       LEVELS.index(0.99), LEVELS.index(0.999))


def residuals(y_tr, med_tr):
    """Multiplicative residuals y / m(x), or None when fewer than 200 rows survive.

    Rows with a non-positive median, or one below 1% of the median positive prediction,
    are dropped rather than floored. Models predict negative medians on a positive target
    (TabICLv2 on 11 of 2000 generator rows), and dividing by a floored one gives a
    residual near 1e13 that alone sets the fitted shape above 3.
    """
    y = np.asarray(y_tr, dtype=float)
    m = np.asarray(med_tr, dtype=float)
    pos = np.isfinite(m) & (m > 0) & np.isfinite(y) & (y > 0)
    if pos.sum() < 200:
        return None
    ok = pos & (m > 0.01 * float(np.median(m[pos])))
    if ok.sum() < 200:
        return None
    z = y[ok] / m[ok]
    return z[np.isfinite(z) & (z > 0)]


def fit_tail(y_tr, med_tr, alpha0=ALPHA0):
    """GPD shape, scale and threshold of the residuals above their alpha0 quantile."""
    z = residuals(y_tr, med_tr)
    if z is None:
        return np.nan, np.nan, np.nan
    u = float(np.quantile(z, alpha0))
    ex = z[z > u] - u
    if len(ex) < 50:
        return np.nan, np.nan, np.nan
    xi_hat, _, sigma_hat = genpareto.fit(ex, floc=0)
    return float(xi_hat), float(sigma_hat), u


def fit_tail_regularised(y_tr, med_tr, thresholds=(0.05, 0.10, 0.20), xi_min=0.0,
                         xi_max=0.95):
    """A steadier fit: the median over several thresholds, with the shape clipped.

    The single fit on 2000 rows put the shape above 1 in four of 30 fits. The median
    over thresholds removes the dependence on one threshold, and `xi_max` below 1 keeps
    a finite mean. Each scale is moved to the ALPHA0 threshold before the median.
    """
    z = residuals(y_tr, med_tr)
    if z is None:
        return np.nan, np.nan, np.nan
    u0 = float(np.quantile(z, ALPHA0))
    xis, sigmas = [], []
    for p in thresholds:
        u_p = float(np.quantile(z, 1 - p))
        ex = z[z > u_p] - u_p
        if len(ex) < 50:
            continue
        xi_p, _, sg_p = genpareto.fit(ex, floc=0)
        if abs(xi_p) > 1e-8:
            sg_p = sg_p + xi_p * (u0 - u_p)
        xis.append(float(xi_p))
        sigmas.append(float(max(sg_p, 1e-9)))
    if not xis:
        return np.nan, np.nan, np.nan
    return (float(np.clip(np.median(xis), xi_min, xi_max)), float(np.median(sigmas)), u0)


def splice(q, med_te, xi_hat, sigma_hat, levels=LEVELS, alpha0=ALPHA0):
    """Quantiles with the tail above alpha0 replaced by the graft; the body stays the model's."""
    if not np.isfinite(xi_hat) or not np.isfinite(sigma_hat):
        return q.copy()
    q = np.asarray(q, dtype=float).copy()
    levels = np.asarray(levels, dtype=float)
    base = q[:, int(np.argmin(np.abs(levels - alpha0)))]
    med = np.maximum(np.asarray(med_te, dtype=float), 1e-12)
    upper = levels > alpha0
    p = (1.0 - levels[upper]) / (1.0 - alpha0)
    if abs(xi_hat) < 1e-8:
        step = -np.log(p) * sigma_hat
    else:
        step = (p ** (-xi_hat) - 1.0) / xi_hat * sigma_hat
    q[:, upper] = base[:, None] + med[:, None] * step[None, :]
    return np.maximum.accumulate(q, axis=1)


def scores(q, y, threshold):
    """The metrics the graft is judged on, for quantiles on LEVELS. `threshold` is the
    lower bound of the tail twCRPS weights."""
    mu = metrics.mean_from_quantiles(q, LEVELS)
    return dict(
        crps=metrics.crps(q, y, LEVELS),
        twcrps=metrics.twcrps(q, y, threshold, LEVELS),
        pinball99=metrics.pinball_at(q, y, 0.99, LEVELS),
        pinball999=metrics.pinball_at(q, y, 0.999, LEVELS),
        cov99=float(np.mean(y <= q[:, I99])),
        cov999=float(np.mean(y <= q[:, I999])),
        gamma_dev=metrics.gamma_deviance(mu, y),
        xi_implied=metrics.implied_xi(q[:, I99], q[:, I90], q[:, I50]),
        mean=float(mu.mean()),
    )
