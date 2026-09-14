# -*- coding: utf-8 -*-
"""TabDPT adapter: quantiles recovered from the internal bin head.

TabDPT publishes a point prediction, but internally holds a distribution over 2048 bins
fixed on [-10, 10] in normalised space. The method that collapses those logits to an
expectation is intercepted for one call, so the logits are kept and the original still
runs. The package does not expose the affine map back to target units, but the
intercepted expectation and the returned prediction differ by exactly that map, so one
least-squares fit on the pairs recovers it.
"""
from __future__ import annotations

import numpy as np


def quantiles(Xtr, ytr, Xte, seed, levels):
    """(n_test, len(levels)) quantiles on the original target scale, from one member."""
    import torch
    from tabdpt import TabDPTRegressor

    m = TabDPTRegressor(device="cpu")
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))

    captured = []
    original = type(m)._expectation_from_regression_logits

    def intercept(self, reg_logits):
        captured.append(reg_logits.detach().float().cpu())
        return original(self, reg_logits)

    type(m)._expectation_from_regression_logits = intercept
    try:
        raw = np.asarray(m.predict(np.asarray(Xte, float), n_ensembles=1, seed=seed),
                         dtype=float)
    finally:
        type(m)._expectation_from_regression_logits = original

    logits = torch.cat(captured, dim=0)
    if len(logits) != len(raw):
        raise RuntimeError(f"{len(logits)} logit rows but {len(raw)} predictions")

    mm = m.model
    edges = torch.linspace(mm.regression_bin_min, mm.regression_bin_max,
                           mm.regression_bin_count + 1, dtype=logits.dtype)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = torch.softmax(logits, dim=-1)

    # raw = a * y_norm + b, fitted on the model's own expectation.
    y_hat = (w * centres).sum(dim=-1).numpy()
    a, b = np.linalg.lstsq(np.c_[y_hat, np.ones(len(y_hat))], raw, rcond=None)[0]

    cdf = torch.cumsum(w, dim=-1).numpy()
    ev = edges.numpy()
    rows = np.arange(len(raw))
    q = np.empty((len(raw), len(levels)))
    for j, alpha in enumerate(levels):
        idx = np.clip(np.sum(cdf < alpha, axis=1), 0, len(centres) - 1)
        c_lo = np.where(idx > 0, cdf[rows, idx - 1], 0.0)
        c_hi = cdf[rows, idx]
        share = np.clip((alpha - c_lo) / np.maximum(c_hi - c_lo, 1e-12), 0.0, 1.0)
        q[:, j] = ev[idx] + share * (ev[idx + 1] - ev[idx])
    return q * a + b


def mean(Xtr, ytr, Xte, seed, n_est=1):
    """(n_test,) the published point prediction, seeded, with `n_est` members."""
    from tabdpt import TabDPTRegressor
    m = TabDPTRegressor(device="cpu")
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return np.asarray(m.predict(np.asarray(Xte, float), n_ensembles=n_est, seed=seed),
                      dtype=float)
