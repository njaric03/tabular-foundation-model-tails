# -*- coding: utf-8 -*-
"""Adapter for EXAONE-Tabular: reading quantiles out of its 999-quantile head.

WHY
---
EXAONE-Tabular (LG AI Research, August 2026) was, on 17 August 2026, SECOND on
the TabArena leaderboard (Elo 1755 overall, 1883 on regression), at about 21.1M
parameters. There is no published paper, only a model card and the code. Its
`regressor.py:341-349` does the same thing as the other four packages:

    center_tensor = selected_y.mean()
    scale_tensor  = selected_y.std(correction=1)
    transformed   = (selected_y - center_tensor) / (scale_tensor + 1e-8)

A non-robust standardisation of the target, while the features go through a
QuantileTransformer, a rank transform that is immune to outliers by
construction. The model card says as much in words: "Targets are standardized
against the fitted support set".

This file exposes the quantiles so the model can be measured with the same
instrument as TabICLv2, TabPFN and TabDPT.

TWO PROBLEMS SOLVED HERE
------------------------
1. `predict()` returns a point only, the mean trimmed over the central 99.8% of
   the quantile function. The bank of 999 levels is thrown away in
   `_collapse_members`. The fix is to replace `_collapse_members` for the
   duration of the call so that it returns the selected level instead. The rest
   of the path (`* scale + center`, the ensemble member weights) is untouched,
   so the output is de-normalised exactly the way the model does it.

2. `from_pretrained()` FAILS on torch 2.13:

       ValueError: checkpoint quantile levels do not match the model

   The cause is not a mismatched model but float32 rounding: the checkpoint
   carries `quantile_levels`, and the head builds them with
   `torch.linspace(1/1000, 999/1000, 999)`. 240 of the 999 levels differ by
   **one ULP** (max |difference| = 5.96e-08), and `checkpoint.py:295` compares
   with `torch.equal`, which is bitwise exact. The fix replaces `torch.equal`
   for the duration of the load with a version that, when the exact comparison
   fails, allows a difference of up to 1e-6 at the same shape. Nothing else
   changes.

LEVELS
------
The checkpoint carries tau_i = (i+1)/1000 for i = 0..998, verified by reading the
safetensors file. So Q(0.5) -> 499, Q(0.9) -> 899, Q(0.99) -> 989,
Q(0.999) -> 998.

USAGE
-----
    from common.adapters import exaone

    m = exaone.create(seed=0)
    m.fit(Xtr, ytr)
    Q = exaone.quantiles(m, Xte, [0.5, 0.9, 0.99])   # (n_test, 3)
"""
import contextlib
import os

import numpy as np
import torch


QUANT_COUNT = 999


def level_index(tau: float) -> int:
    """tau_i = (i+1)/1000 -> index into the bank of 999 quantiles."""
    i = int(round(tau * (QUANT_COUNT + 1))) - 1
    if not 0 <= i < QUANT_COUNT:
        raise ValueError(f"tau={tau} outside the grid of {QUANT_COUNT} levels")
    return i


@contextlib.contextmanager
def _lenient_equal(atol: float = 1e-6):
    """`torch.equal` with a one ULP tolerance, for the duration of the load only.

    See the module docstring: without this `from_pretrained` fails on torch 2.13.
    The exact comparison is still tried first; the tolerance is used only when
    that fails, and only for tensors of the same shape and dtype.
    """
    original = torch.equal

    def lenient(a, b):
        if original(a, b):
            return True
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
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

    The seed does not enter the weights, the model is pre-trained; it fixes the
    permutations of the ensemble members, the same role `random_state` has in the
    other packages.

    DTYPE. The manifest pins `compute_dtype="float16"`, because the model is
    written for CUDA ("a CUDA GPU is strongly recommended -- the model uses fused
    attention kernels and half precision"). On CPU torch has no fast fp16 kernels
    and emulates them, so inference is an order of magnitude slower. The default
    on CPU is therefore **float32**: that is more precise rather than less, but
    it IS a departure from the shipped configuration and has to be written down
    as such in the thesis. To measure in the shipped configuration:
    `DTYPE=float16`.

    N_EST. The manifest pins `ensemble_count=8`. The other measurements in this
    repository use 1 or 4 (`findings/h1/vincentization.md`), so the knob is
    exposed for comparability.
    """
    from exaonetabular import EXAONETabularRegressor
    if dtype is None:
        dtype = os.environ.get("DTYPE") or ("float32" if str(device) == "cpu" else None)
    with _lenient_equal():
        return EXAONETabularRegressor.from_pretrained(
            device=device, seed=seed, ensemble_count=n_est, compute_dtype=dtype)


def quantiles(model, X, taus):
    """(n_test, len(taus)) predicted quantiles, on the original target scale.

    Works by having `_collapse_members` return the selected levels instead of the
    trimmed mean, for the duration of the call. The bank is sorted before it is
    indexed, which is what the model does in its own `trimmed` branch, as a guard
    against crossing quantiles. The rest of the `predict` path (`* scale +
    center`, joining the members) is untouched.

    FAST PATH. When the member weights are `None`, and they are below about
    12,000 context rows, since NNLS asks for 2,000 held-out rows, `predict` joins
    the members with `.mean(dim=0)`, which also works on a tensor of shape
    (members, rows, K). All levels then come out of ONE pass. With fitted weights
    the blend does not broadcast over a third axis, so it falls back to one pass
    per level.
    """
    import exaonetabular.regressor as R

    X = np.asarray(X, dtype=np.float64)
    idx = [level_index(t) for t in taus]
    fast = (model._state().get("member_weights") is None
            and not os.environ.get("PER_LEVEL"))
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
    print("monotone Q50<=Q90<=Q99<=Q999:",
          bool(np.all(np.diff(Q, axis=1) >= -1e-9)))
    print("medians:", np.round(np.median(Q, axis=0), 3),
          " point:", round(float(np.median(point)), 3))
    print("empirical y:", np.round(np.quantile(y, [0.5, 0.9, 0.99, 0.999]), 3))
