# -*- coding: utf-8 -*-
"""EXAONE-Tabular adapter: quantiles from its 999-level head, and a loader that works.

Two patches, each active for one call only:

1. `predict` returns a trimmed mean and discards the quantile bank inside
   `_collapse_members`. `quantiles` swaps that method for one returning the chosen
   levels, so de-normalisation and member weighting stay the model's own.
2. `from_pretrained` fails on torch 2.13 with "checkpoint quantile levels do not match
   the model": 240 of the 999 stored levels differ from `torch.linspace` by one ULP,
   and `checkpoint.py` compares them with the bitwise `torch.equal`. `_lenient_equal`
   accepts a difference up to 1e-6 at the same shape and dtype while loading.

The head predicts tau_i = (i + 1) / 1000 for i = 0..998, so Q(0.99) is index 989. Like
the other packages, `regressor.py:341` standardises the target by its plain mean and sd
while the features go through a QuantileTransformer.

    m = exaone.create(seed=0)
    m.fit(Xtr, ytr)
    Q = exaone.quantiles(m, Xte, [0.5, 0.9, 0.99])   # (n_test, 3)
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch

from common import env

QUANT_COUNT = 999


def level_index(tau: float) -> int:
    """Index of tau in the bank tau_i = (i + 1) / 1000."""
    i = int(round(tau * (QUANT_COUNT + 1))) - 1
    if not 0 <= i < QUANT_COUNT:
        raise ValueError(f"tau={tau} outside the grid of {QUANT_COUNT} levels")
    return i


@contextlib.contextmanager
def _lenient_equal(atol: float = 1e-6):
    """`torch.equal` with a one-ULP tolerance, for the duration of the load only."""
    original = torch.equal

    def lenient(a, b):
        if original(a, b):
            return True
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype or a.dtype == torch.bool:
            return False
        return bool((a - b).abs().max() <= atol)

    torch.equal = lenient
    try:
        yield
    finally:
        torch.equal = original


def create(seed: int = 0, n_est: int | None = None, device: str = "cpu",
           dtype: str | None = None):
    """A loaded EXAONETabularRegressor.

    The seed fixes the permutations of the ensemble members. The manifest pins
    `compute_dtype="float16"` for CUDA; torch emulates fp16 on CPU an order of magnitude
    slower, so the CPU default here is float32, a documented departure from the shipped
    configuration (DTYPE=float16 restores it). The manifest's `ensemble_count=8` is
    overridden by `n_est` so EXAONE is comparable with the 1 and 4 used elsewhere.
    """
    from exaonetabular import EXAONETabularRegressor
    if dtype is None:
        dtype = env.text("DTYPE", "float32" if str(device) == "cpu" else "") or None
    with _lenient_equal():
        return EXAONETabularRegressor.from_pretrained(
            device=device, seed=seed, ensemble_count=n_est, compute_dtype=dtype)


def quantiles(model, X, taus):
    """(n_test, len(taus)) predicted quantiles on the original target scale.

    The bank is sorted before indexing, as the model's own trimmed path does. Without
    fitted member weights, which NNLS only fits above about 12,000 context rows, members
    are joined by `.mean(dim=0)`, which works on (members, rows, K) and gives every level
    in one pass. With weights the blend does not broadcast, so each level takes a pass.
    PER_LEVEL=1 forces the slow path.
    """
    import exaonetabular.regressor as R

    X = np.asarray(X, dtype=np.float64)
    idx = [level_index(t) for t in taus]
    fast = model._state().get("member_weights") is None and not env.flag("PER_LEVEL")
    original = R.EXAONETabularRegressor._collapse_members

    def make_collapse(pick):
        def collapse(self, output, query_count):
            expected = (self.manifest.runtime.ensemble_count, query_count,
                        self.manifest.output_width)
            if tuple(output.shape) != expected or not bool(torch.isfinite(output).all()):
                raise RuntimeError("the model returned invalid quantiles")
            return torch.sort(output.float(), dim=-1).values[..., pick]
        return collapse

    try:
        if fast:
            R.EXAONETabularRegressor._collapse_members = make_collapse(idx)
            Q = np.asarray(model.predict(X), dtype=np.float64)
            return Q.reshape(len(X), len(idx))
        out = []
        for i in idx:
            R.EXAONETabularRegressor._collapse_members = make_collapse(i)
            out.append(np.asarray(model.predict(X), dtype=np.float64))
        return np.column_stack(out)
    finally:
        R.EXAONETabularRegressor._collapse_members = original


def n_parameters(model) -> int:
    return int(sum(p.numel() for p in model.model.parameters()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = int(os.environ.get("N", "800"))
    X = rng.normal(size=(n, 5))
    y = np.exp(0.6 * X[:, 0]) * ((1 - rng.random(n)) ** (-0.7) - 1) / 0.7
    Xte = rng.normal(size=(200, 5))

    m = create()
    print(f"parameters: {n_parameters(m):,}")
    print(f"ensemble_count={m.manifest.runtime.ensemble_count}  "
          f"quantile_count={m.manifest.regression.quantile_count}  "
          f"point_estimate={m.manifest.regression.point_estimate}")
    m.fit(X, y)
    Q = quantiles(m, Xte, [0.5, 0.9, 0.99, 0.999])
    point = m.predict(Xte)
    print("monotone Q50<=Q90<=Q99<=Q999:", bool(np.all(np.diff(Q, axis=1) >= -1e-9)))
    print("medians:", np.round(np.median(Q, axis=0), 3),
          " point:", round(float(np.median(point)), 3))
    print("empirical y:", np.round(np.quantile(y, [0.5, 0.9, 0.99, 0.999]), 3))
