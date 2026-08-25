# -*- coding: utf-8 -*-
"""TabDPT adapter: quantiles pulled out of the internal bin head.

TabDPT publishes only a point prediction, but internally it holds a
distribution over 2048 bins fixed on [-10, 10] in z-space. The quantiles are
recovered by intercepting the method that collapses those logits to an
expectation, keeping the logits, and letting the original run.

The bin edges live in normalised space and the package does not expose the
affine map back. It is reconstructed from the model own mean: the intercepted
expectation and the returned prediction differ by exactly that affine map, so
one least-squares fit on (expectation, prediction) recovers it.
"""
from __future__ import annotations

import numpy as np


def quantiles(Xtr, ytr, Xte, seed, levels):
    """(n_test, len(levels)) quantiles on the original target scale."""
    import torch
    from tabdpt import TabDPTRegressor

    Xtr = np.asarray(Xtr, float)
    ytr = np.asarray(ytr, float)
    Xte = np.asarray(Xte, float)

    m = TabDPTRegressor(device="cpu")
    m.fit(Xtr, ytr)

    captured = []
    original = type(m)._expectation_from_regression_logits

    def intercept(self, reg_logits):
        captured.append(reg_logits.detach().float().cpu())
        return original(self, reg_logits)

    type(m)._expectation_from_regression_logits = intercept
    try:
        raw = np.asarray(m.predict(Xte, n_ensembles=1, seed=seed), dtype=float)
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

    # Affine map raw = a * y_norm + b, reconstructed from the model own mean.
    y_hat = (w * centres).sum(dim=-1).numpy()
    A = np.c_[y_hat, np.ones(len(y_hat))]
    a, b = np.linalg.lstsq(A, raw, rcond=None)[0]

    cdf = torch.cumsum(w, dim=-1).numpy()
    q = np.empty((len(raw), len(levels)))
    ev = edges.numpy()
    for j, alpha in enumerate(levels):
        idx = np.clip(np.sum(cdf < alpha, axis=1), 0, len(centres) - 1)
        c_lo = np.where(idx > 0, cdf[np.arange(len(idx)), idx - 1], 0.0)
        c_hi = cdf[np.arange(len(idx)), idx]
        share = np.clip((alpha - c_lo) / np.maximum(c_hi - c_lo, 1e-12), 0.0, 1.0)
        q[:, j] = ev[idx] + share * (ev[idx + 1] - ev[idx])
    return q * a + b


def mean(Xtr, ytr, Xte, seed, n_est=1):
    """(n_test,) point prediction, the published output."""
    from tabdpt import TabDPTRegressor
    m = TabDPTRegressor(device="cpu")
    m.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return np.asarray(m.predict(np.asarray(Xte, float)), dtype=float)
