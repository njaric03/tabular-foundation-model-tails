# -*- coding: utf-8 -*-
"""One synthetic data generator, replacing 25 copies.

The copies carried two incompatible signatures, make_data(xi, n, rng) and
podaci(n, xi, rng), so the first two arguments were swapped between them and
passing a call from one script to another was a silent error whenever xi was a
round number. Only half of them handled the xi -> 0 branch; the other half
divided by zero and returned all-NaN quietly, because every script starts with
filterwarnings("ignore").

Here xi is keyword-only, so the swap cannot happen, and the return is a named
tuple with always the same four fields, so the arity cannot drift.

    from common import generator

    p = generator.gpd(2000, rng, xi=0.7)              # constant shape
    p = generator.gpd(2000, rng, xi=generator.XI_OF_X)  # shape driven by x4
    p.X, p.y, p.s, p.xi
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm

from common.metrics import gpd_quantile

# Weights of the linear predictor of the scale. x3 and x4 carry ZERO: the scale
# does not pass through them, so the tail shape can be tied to x4 while the
# level stays put.
W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])

XI_LO, XI_HI = 0.15, 0.90

# Sentinel: the shape depends on x4 instead of being a constant.
XI_OF_X = "xi(x)"


class Data(NamedTuple):
    """Always four fields, so the return arity cannot drift between scripts."""
    X: np.ndarray
    y: np.ndarray
    s: np.ndarray          # GPD scale per point
    xi: np.ndarray         # GPD shape per point (a constant is broadcast)


def xi_of_x(X: np.ndarray) -> np.ndarray:
    """Tail shape rises with x4, which has zero weight in the scale.

    That is what separates shape from level: anything a model captures about the
    tail through x4 is genuine shape adaptation, not a by-product of a larger scale.
    """
    return XI_LO + (XI_HI - XI_LO) * norm.cdf(X[:, 4])


def gpd(n: int, rng, *, xi, d: int = 5, w: np.ndarray | None = None,
        clip: bool = False) -> Data:
    """n draws from y | x ~ GPD(xi(x), s(x)), with s(x) = exp(0.6 * x.w / norm(w)).

    xi is keyword-only on purpose: in the old signatures it was sometimes the
    first and sometimes the second positional argument.

    clip caps the uniform draw at 1 - 1e-9, which caps the largest value the
    tail can produce. The xi(x) experiments never clipped and the constant-xi
    ones always did, so the flag exists to reproduce both exactly rather than
    quietly changing published numbers. Pass clip=True for a constant xi.
    """
    w = W if w is None else np.asarray(w, dtype=float)
    X = rng.normal(size=(n, d))
    s = np.exp(0.6 * (X @ w) / np.linalg.norm(w))
    xi_v = (xi_of_x(X) if isinstance(xi, str) and xi == XI_OF_X
            else np.full(n, float(xi)))
    # rng.random draws from [0, 1), so 1 - U is never zero.
    U = rng.random(n)
    if clip:
        U = np.clip(U, 1e-9, 1 - 1e-9)
    return Data(X=X, y=s * gpd_quantile(U, xi_v), s=s, xi=xi_v)


def true_mean(p: Data) -> np.ndarray:
    """E[Y|x] = s / (1 - xi), defined only for xi < 1."""
    return np.where(p.xi < 1, p.s / np.maximum(1 - p.xi, 1e-12), np.inf)


def true_median(p: Data) -> np.ndarray:
    return p.s * gpd_quantile(0.5, p.xi)
