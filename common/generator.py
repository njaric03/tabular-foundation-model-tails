# -*- coding: utf-8 -*-
"""The synthetic generator: y | x ~ GPD(xi(x), s(x)).

    p = generator.gpd(2000, rng, xi=0.7, clip=True)      # constant shape
    p = generator.gpd(2000, rng, xi=generator.XI_OF_X)   # shape driven by x4
    p.X, p.y, p.s, p.xi

s(x) = exp(0.6 * x.w / |w|) does not depend on x3 or x4, while xi(x) rises with x4, so
whatever a model captures about the tail through x4 is shape and not level. `xi` is
keyword-only: earlier copies took it as the first argument in some scripts and the
second in others.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm

from common.metrics import gpd_quantile, truncated_mean_std

# Weights of the scale's linear predictor. x3 and x4 carry none.
W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])

XI_LO, XI_HI = 0.15, 0.90

# Pass as `xi` for a shape that depends on x4.
XI_OF_X = "xi(x)"


class Data(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    s: np.ndarray          # GPD scale per row
    xi: np.ndarray         # GPD shape per row, a constant broadcast


class Covariates(NamedTuple):
    X: np.ndarray
    lin: np.ndarray        # x.w / |w|, the standardised linear predictor
    s: np.ndarray
    xi: np.ndarray


def xi_of_x(X: np.ndarray) -> np.ndarray:
    """Tail shape rising with x4, from XI_LO to XI_HI."""
    return XI_LO + (XI_HI - XI_LO) * norm.cdf(X[:, 4])


def covariates(n: int, rng, *, xi, d: int = 5, w: np.ndarray | None = None) -> Covariates:
    """X, the linear predictor, s(x) and xi(x), without drawing the target.

    For scripts that put another tail family or link on the same design. X is drawn
    first, as in `gpd`, so a caller that draws its own uniforms next gets the stream
    the inline formula had.
    """
    w = W if w is None else np.asarray(w, dtype=float)
    X = rng.normal(size=(n, d))
    lin = (X @ w) / np.linalg.norm(w)
    # Written as the inline copies had it. Associating the product the other way,
    # 0.6 * lin, moves the last bit; tests/test_generator.py holds it down.
    s = np.exp(0.6 * (X @ w) / np.linalg.norm(w))
    xi_v = (xi_of_x(X) if isinstance(xi, str) and xi == XI_OF_X
            else np.full(n, float(xi)))
    return Covariates(X=X, lin=lin, s=s, xi=xi_v)


def gpd(n: int, rng, *, xi, d: int = 5, w: np.ndarray | None = None,
        clip: bool = False) -> Data:
    """n draws of y | x ~ GPD(xi(x), s(x)).

    `clip` caps the uniform draw at 1 - 1e-9 and so the largest value the tail can
    produce. The constant-xi experiments were measured with it and the xi(x) ones
    without, so the flag reproduces both.
    """
    c = covariates(n, rng, xi=xi, d=d, w=w)
    U = rng.random(n)
    if clip:
        U = np.clip(U, 1e-9, 1 - 1e-9)
    return Data(X=c.X, y=c.s * gpd_quantile(U, c.xi), s=c.s, xi=c.xi)


def true_mean(p: Data) -> np.ndarray:
    """E[Y | x] = s / (1 - xi), infinite at xi >= 1."""
    return np.where(p.xi < 1, p.s / np.maximum(1 - p.xi, 1e-12), np.inf)


def truncated_mean(p: Data, top: float) -> np.ndarray:
    """E[Y | x, Y <= top] per row, for a constant xi: the most a model that cannot
    predict above the largest training value can get right."""
    return p.s * truncated_mean_std(top / p.s, float(p.xi[0]))
