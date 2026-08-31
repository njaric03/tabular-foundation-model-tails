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


class Covariates(NamedTuple):
    """The design without the draw: X, the linear predictor, the scale, the shape."""
    X: np.ndarray
    lin: np.ndarray        # x.w / norm(w), the standardised linear predictor
    s: np.ndarray
    xi: np.ndarray


def xi_of_x(X: np.ndarray) -> np.ndarray:
    """Tail shape rises with x4, which has zero weight in the scale.

    That is what separates shape from level: anything a model captures about the
    tail through x4 is genuine shape adaptation, not a by-product of a larger scale.
    """
    return XI_LO + (XI_HI - XI_LO) * norm.cdf(X[:, 4])


def covariates(n: int, rng, *, xi, d: int = 5,
               w: np.ndarray | None = None) -> Covariates:
    """X, the linear predictor, s(x) and xi(x), without drawing the target.

    Split out of `gpd` for the scripts that need this design but a different
    tail family or a different link: `tail_families.py` draws Frechet, Burr and
    Student-t on top of it, and `coherence.py` needs an additive control where
    the log is not the canonical scale. Both used to carry their own copy of
    W, of s(x) and of xi(x).

    The draw order is the same as in `gpd` -- X first, the target's uniform
    afterwards -- so a script that calls this and then draws its own U gets the
    same stream it had when the formula was inline.
    """
    w = W if w is None else np.asarray(w, dtype=float)
    X = rng.normal(size=(n, d))
    lin = (X @ w) / np.linalg.norm(w)
    # Written exactly as the inline copies had it. Associating the multiplication
    # the other way (0.6 * lin) moves the last bit and changes published numbers;
    # tests/test_generator.py holds that line down.
    s = np.exp(0.6 * (X @ w) / np.linalg.norm(w))
    xi_v = (xi_of_x(X) if isinstance(xi, str) and xi == XI_OF_X
            else np.full(n, float(xi)))
    return Covariates(X=X, lin=lin, s=s, xi=xi_v)


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
    c = covariates(n, rng, xi=xi, d=d, w=w)
    X, s, xi_v = c.X, c.s, c.xi
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
