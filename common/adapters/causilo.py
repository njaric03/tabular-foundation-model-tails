# -*- coding: utf-8 -*-
"""Causilo adapter (Nums AI, September 2026): quantiles from its 999-level head.

Causilo 1.0.3 standardises the target with sklearn's `StandardScaler`
(`data/dataset.py:70`), the plain mean and sd, while its features go through a
`none / rank2gaussian / robust / power` cycle over ensemble members, after which tails
beyond four two-pass sds are compressed with arcsinh (`data/normalization.py`). Member 0
is `none`, so a single member sees z-scored, tail-compressed features.

The head predicts tau_i = i / 1000 for i = 1..999. Levels between are interpolated
linearly; outside [0.001, 0.999] the package extrapolates an exponential tail from the
two outermost quantiles (`quantiles.py`), which is xi = 0, as in TabICL. The 0.999 used
here is the last native level, so no extrapolation enters.

    Q = causilo.quantiles(Xtr, ytr, Xte, seed=7000, levels=[0.5, 0.99])
"""
from __future__ import annotations

import numpy as np


def create(seed: int, n_est: int = 1, device: str = "cpu"):
    """An unfitted CausiloRegressor. `device="auto"` would pick CUDA when present, so
    CPU is named, as for every other model here."""
    from causilo import CausiloRegressor
    return CausiloRegressor(n_estimators=n_est, random_state=seed, device=device)


def quantiles(Xtr, ytr, Xte, seed, levels, n_est=1) -> np.ndarray:
    """(n_test, len(levels)) on the original target scale."""
    m = create(seed, n_est)
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return np.asarray(m.predict(np.asarray(Xte, float), output_type="quantiles",
                                quantiles=list(levels)), dtype=float)


def mean(Xtr, ytr, Xte, seed, n_est=1) -> np.ndarray:
    """(n_test,) the package's own mean, the average of its 999 quantiles."""
    m = create(seed, n_est)
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return np.asarray(m.predict(np.asarray(Xte, float)), dtype=float)
