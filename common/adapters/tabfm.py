# -*- coding: utf-8 -*-
"""TabFM adapter: point prediction only.

TabFM exposes no conditional distribution. Its predict returns one number per
row, and every occurrence of the word quantile in the package refers to a
QuantileTransformer applied to the FEATURES. So TabFM can take part in the mean
experiments but not in any shape-of-the-tail measurement; that is a property of
the model, not a gap in the work.

The weights are 6.6 GB in bfloat16 and load in tens of seconds, so they are
cached at module level: a script that fits many times pays the load once.
Download them with run/download_tabfm.sh; about 10 GB of free disk is needed.
"""
from __future__ import annotations

import os

import numpy as np

_model = None


def load(checkpoint=None, device="cpu"):
    """Load the TabFM regression checkpoint once and keep it."""
    global _model
    if _model is not None:
        return _model
    import torch
    import tabfm

    ckpt = checkpoint or os.path.expanduser("~/tabfm-regression")
    if not os.path.isfile(os.path.join(ckpt, "model.safetensors")):
        raise SystemExit(
            f"missing {ckpt}/model.safetensors -- run run/download_tabfm.sh "
            f"(6.6 GB, needs about 10 GB free)")
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
