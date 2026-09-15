# -*- coding: utf-8 -*-
"""Arm A1 of the pre-training experiment: heavy tails and contaminated contexts.

Applied per table to the base dump, so A0 and A1 see the same base tables and differ only
in these two treatments. Plain numpy, one table at a time:

    y = prior_arms.heavy_tail(X, y, k, rng)            # context and test alike
    y, i, shift = prior_arms.contaminate(y, split, rng)  # context only

`heavy_tail` reads each row's within-table rank level u = (rank + 1/2) / n through a
standard GPD quantile whose shape depends on one randomly chosen active feature,
xi = XI_LO + (XI_HI - XI_LO) * sigmoid(w * z_j) with z_j that feature standardised. With a
shape that varies by row the order of the rows is not kept, only each row's level. It is
the data-generating distribution, so the test targets get it too.

`contaminate` rescales one context row so that sd(context) / sd(context without that row)
lands on a log-uniform draw in [SHIFT_LO, SHIFT_HI]. Both sds use ddof=0, as
`metrics.sd_shift` does. With the other n - 1 values of mean m and variance v, a value c
gives shift^2 = ((n - 1) / n) (1 + (c - m)^2 / (n v)), so c is solved in closed form and
placed above m, the direction of a leverage row in the upper tail. Test targets are never
touched: the loss has to teach the clean conditional distribution.
"""
from __future__ import annotations

import numpy as np

from common.metrics import gpd_quantile

P_TAIL, P_CONTAM = 0.5, 0.3
XI_LO, XI_HI = 0.0, 1.0
W_SD = 2.0
SHIFT_LO, SHIFT_HI = 4.0, 50.0


def heavy_tail(X, y, k: int, rng, xi_lo: float = XI_LO, xi_hi: float = XI_HI,
               w_sd: float = W_SD):
    """(new y, feature index j, xi per row) for one table with `k` active features."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    j = int(rng.integers(k))
    w = float(rng.normal(0.0, w_sd))
    xj = X[:, j]
    z = (xj - xj.mean()) / max(xj.std(), 1e-12)
    xi = xi_lo + (xi_hi - xi_lo) / (1.0 + np.exp(-w * z))
    ranks = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    u = (ranks + 0.5) / n
    return gpd_quantile(u, xi), j, xi


def value_for_shift(others, target: float) -> float:
    """The value c that, appended to `others`, gives sd(all) / sd(others) = target."""
    others = np.asarray(others, float)
    n = len(others) + 1
    m, v = others.mean(), others.var()
    if v <= 0 or target <= 1.0:
        raise ValueError(f"no finite value reaches a shift of {target} here")
    return float(m + np.sqrt(n * v * (target ** 2 * n / (n - 1) - 1.0)))


def shift_of_row(y_context, i: int) -> float:
    """sd(context) / sd(context without row i), both ddof=0."""
    y_context = np.asarray(y_context, float)
    rest = np.delete(y_context, i)
    return float(y_context.std() / max(rest.std(), 1e-300))


def contaminate(y, split: int, rng, lo: float = SHIFT_LO, hi: float = SHIFT_HI):
    """(new y, row index, realised shift). Only rows before `split` can change."""
    y = np.array(y, dtype=float)
    target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    i = int(rng.integers(split))
    y[i] = value_for_shift(np.delete(y[:split], i), target)
    return y, i, shift_of_row(y[:split], i)


def apply_a1(X, y, k: int, split: int, rng, p_tail: float = P_TAIL,
             p_contam: float = P_CONTAM, **kw):
    """One table under arm A1. Returns (y, info) where info records what was drawn."""
    info = dict(tail=False, contam=False, shift=1.0)
    if rng.random() < p_tail:
        y, info["feature"], _ = heavy_tail(X, y, k, rng,
                                           **{a: kw[a] for a in ("xi_lo", "xi_hi", "w_sd")
                                              if a in kw})
        info["tail"] = True
    if rng.random() < p_contam:
        y, info["row"], info["shift"] = contaminate(
            y, split, rng, **{a: kw[a] for a in ("lo", "hi") if a in kw})
        info["contam"] = True
    return np.asarray(y, float), info
