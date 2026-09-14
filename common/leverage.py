# -*- coding: utf-8 -*-
"""Leverage on real tables: subsamples binned by their sd shift, and the shift a single
corrupted row would cause."""
from __future__ import annotations

import numpy as np

# Bins of the natural sd shift of a subsample.
BINS = [(0.0, 1.2), (1.2, 2.0), (2.0, 4.0), (4.0, 1e9)]

# The sd shift at which part two first sees a loss. A unit error that stays below it
# tests nothing.
VISIBLE_SHIFT = 2.0


def add_row(X, y, y0, x0=None):
    """The context with one more row: target `y0` at `x0`, the centre of x by default."""
    x0 = np.zeros((1, X.shape[1])) if x0 is None else np.atleast_2d(x0)
    return np.vstack([X, x0]), np.concatenate([y, np.atleast_1d(y0)])


def mean_influence(before, after) -> float:
    """Median relative change of a mean prediction, against |before| floored at 1e-9."""
    return float(np.median((after - before) / np.maximum(np.abs(before), 1e-9)))


def bin_name(k: int) -> str:
    lo, hi = BINS[k]
    return f"{lo}-{hi if hi < 1e8 else 'inf'}"


def subsamples(y, rng, n_fit, n_test, per_bin, max_attempts):
    """Rejection sampling of up to `per_bin` subsamples into every sd-shift bin.

    Returns {bin index: [(row indices, sd shift, largest over second largest), ...]}.
    The first `n_fit` indices are the context and the rest the test rows. The draws
    depend only on `rng`, so the same seed gives the same subsamples in every script.
    """
    n = min(n_fit + n_test, len(y))
    found = {k: [] for k in range(len(BINS))}
    for _ in range(max_attempts):
        if all(len(v) >= per_bin for v in found.values()):
            break
        idx = rng.permutation(len(y))[:n]
        yf = y[idx[:n_fit]]
        p = np.sort(yf)
        shift = np.std(yf) / max(np.std(p[:-1]), 1e-12)
        lev = abs(p[-1]) / max(abs(p[-2]), 1e-12)
        for k, (lo, hi) in enumerate(BINS):
            if lo <= shift < hi and len(found[k]) < per_bin:
                found[k].append((idx, shift, lev))
                break
    return found


def shift_if_corrupted(yf, factor):
    """The sd shift of the context if row i alone were multiplied by `factor`, for every
    i at once."""
    n = len(yf)
    s, q = yf.sum(), (yf ** 2).sum()
    new = yf * factor
    s1, q1 = s + new - yf, q + new ** 2 - yf ** 2
    sd_all = np.sqrt(np.maximum(q1 / n - (s1 / n) ** 2, 0))
    top = yf.max()
    # The largest value after the change: the corrupted row if it overtook the maximum.
    drop = np.where(new >= top, new, top)
    s2, q2 = s1 - drop, q1 - drop ** 2
    sd_wo = np.sqrt(np.maximum(q2 / (n - 1) - (s2 / (n - 1)) ** 2, 1e-24))
    return sd_all / sd_wo
