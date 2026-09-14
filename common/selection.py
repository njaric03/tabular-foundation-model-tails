# -*- coding: utf-8 -*-
"""The entry test for tables whose tail index is identified, shared by the dataset surveys.

Three estimators on the residuals of a log-linear scale must agree, and the GPD fit must
be stable across thresholds; otherwise a model is being scored against a reference that
itself moves.
"""
from __future__ import annotations

import numpy as np

from common import metrics

MAX_ROWS = 30000
MIN_N = 3000
MIN_XI = 0.15
MAX_RANGE = 0.25
FRACS = (0.05, 0.10, 0.20)


def subsample(X, y, max_rows=MAX_ROWS):
    """At most `max_rows` rows, drawn with a fixed seed."""
    if len(y) <= max_rows:
        return X, y
    idx = np.random.default_rng(0).choice(len(y), max_rows, replace=False)
    return X[idx], y[idx]


def estimates(X, y) -> dict:
    """The three estimates and the entry test.

    A table passes with at least MIN_N positive targets, a largest estimate of at least
    MIN_XI, the GPD MLE at 10%, Hill and the quantile inversion within MAX_RANGE of each
    other, and the GPD MLE within MAX_RANGE across the thresholds in FRACS.
    """
    z = metrics.residuals(X, y)
    gpd = [metrics.gpd_mle(z, f) for f in FRACS]
    hill = metrics.hill(z)
    q50, q90, q99 = np.quantile(z, [0.5, 0.9, 0.99])
    quantile = (metrics.xi_from_residual_ratio((q99 - q50) / (q90 - q50))
                if (q90 - q50) > 1e-12 else np.nan)
    three = np.array([gpd[1], hill, quantile], dtype=float)
    estimate_range = float(np.nanmax(three) - np.nanmin(three))
    range_threshold = float(np.nanmax(gpd) - np.nanmin(gpd))
    n_pos = int((y > 0).sum())
    _, counts = np.unique(y[y > 0], return_counts=True)
    return dict(n=n_pos, gpd05=gpd[0], gpd10=gpd[1], gpd20=gpd[2], hill=hill,
                quantile=quantile, estimate_range=estimate_range,
                range_threshold=range_threshold, tied_share=float(counts.max() / n_pos),
                passes=bool(n_pos >= MIN_N and np.nanmax(three) >= MIN_XI
                            and estimate_range <= MAX_RANGE and range_threshold <= MAX_RANGE))
