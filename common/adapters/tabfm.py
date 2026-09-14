# -*- coding: utf-8 -*-
"""TabFM adapter: point prediction only.

TabFM returns one number per row; every quantile in the package is a QuantileTransformer
on the features. It takes part in the mean experiments and in no tail-shape measurement.
The 6.6 GB bfloat16 checkpoint is loaded once per process. Fetch it with
`run/download_tabfm.sh`, which needs about 10 GB free.
"""
from __future__ import annotations

import os

import numpy as np

_model = None


def load(checkpoint=None, device="cpu"):
    """The TabFM regression model, loaded on the first call and reused after it."""
    global _model
    if _model is not None:
        return _model
    import tabfm
    import torch

    ckpt = checkpoint or os.path.expanduser("~/tabfm-regression")
    if not os.path.isfile(os.path.join(ckpt, "model.safetensors")):
        raise FileNotFoundError(f"missing {ckpt}/model.safetensors; run run/download_tabfm.sh")
    _model = tabfm.tabfm_v1_0_0_pytorch.load(
        model_type="regression", checkpoint_path=ckpt, device=device,
        dtype=torch.bfloat16)
    return _model


def mean(Xtr, ytr, Xte, seed, n_est=1):
    """(n_test,) point prediction on the original target scale."""
    from tabfm import TabFMRegressor
    r = TabFMRegressor(load(), n_estimators=n_est, random_state=seed)
    r.fit(np.asarray(Xtr, float), np.asarray(ytr, float))
    return np.asarray(r.predict(np.asarray(Xte, float)), dtype=float)
