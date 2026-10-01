# -*- coding: utf-8 -*-
"""LimiX-2 adapter (Stable AI, September 2026, 1st on TabArena): quantiles from its bar head.

LimiX-2 publishes only a point prediction. Its regression head is a bar distribution
with half-normal outer tails, the design of TabPFN's `FullSupportBarDistribution`, and
`inference/v2_0/inference_utils.py` decodes nothing from it but the mean. `quantiles`
therefore reads the member-averaged bucket probabilities at the one place the package
decodes them (`logits_to_output`, called from `get_reg_pred_result`) and inverts their
CDF here: linearly inside a bucket and through the half-normal in the two outer ones,
with the scale the package's own `mean` uses (median at the bucket width). That is the
law the head defines; unlike TabPFN <= 9.0.0, the reported quantile is not stopped at
the last border. Everything before the decode is the package's.

The target is standardised by its plain mean and sd (`predictor.py:3181`,
`_prepare_reg_context`), and the shipped regression config has no `TargetTransform`, so
nothing protects it. `outlier_remove_std=12` is stored and never used on this path.

The ensemble is the config's list of preprocessing pipelines; `n_est` keeps the first
`n_est` of them. The checkpoint is pinned to a Hub revision and loaded once per process.
LIMIX_DEVICE picks the device (default cuda when there is one): 400M parameters are
slow on CPU, and GPU kernels differ from CPU ones in the last bits, so the device is a
provenance knob. Mixed precision (fp16 autocast) is the package default on CUDA.

    Q = limix.quantiles(Xtr, ytr, Xte, seed=7000, levels=[0.5, 0.99])
"""
from __future__ import annotations

import contextlib
import json
from importlib import resources

import numpy as np

from common import env

REPO, FILE = "stable-ai/LimiX-2", "LimiX-2.ckpt"
REVISION = "de07b679e74a41b50b9de18251a8fa245e537440"
CONFIG = "reg_default_noretrieval_v2.json"
# HalfNormal(1).icdf(0.5): the package sets each outer scale so the median is the width.
HALF_NORMAL_MEDIAN = 0.6744897501960817

_ckpt = None


def device():
    import torch
    name = env.text("LIMIX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _checkpoint():
    global _ckpt
    if _ckpt is None:
        import torch
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=REPO, filename=FILE, revision=REVISION)
        _ckpt = (path, torch.load(path, map_location="cpu", weights_only=False))
    return _ckpt


def config(n_est: int) -> dict:
    """The shipped regression config with its first n_est pipelines."""
    with resources.files("config").joinpath(CONFIG).open(encoding="utf-8") as fh:
        cfg = json.load(fh)
    if any("TargetTransform" in p for p in cfg["pipelines"]):
        raise RuntimeError(f"{CONFIG} now transforms the target; the adapter assumes not")
    if not 1 <= n_est <= len(cfg["pipelines"]):
        raise ValueError(f"n_est={n_est}; the config has {len(cfg['pipelines'])} pipelines")
    cfg["pipelines"] = cfg["pipelines"][:n_est]
    return cfg


def create(seed: int, n_est: int = 1):
    from inference.predictor import LimiXPredictor
    path, ckpt = _checkpoint()
    with contextlib.redirect_stdout(None):
        return LimiXPredictor(device=device(), model_path=path, inference_config=config(n_est),
                              seed=seed, ckpt=ckpt)


@contextlib.contextmanager
def _captured_decode():
    """Record the averaged log-probabilities and borders of each regression decode."""
    import inference.v2_0.predictor as pred
    original, seen = pred.logits_to_output, []

    def capture(output_type, logits, quantiles, borders, bucket_widths):
        seen.append((logits.float().cpu().numpy(), borders.double().cpu().numpy()))
        return original(output_type=output_type, logits=logits, quantiles=quantiles,
                        borders=borders, bucket_widths=bucket_widths)

    pred.logits_to_output = capture
    try:
        yield seen
    finally:
        pred.logits_to_output = original


def bar_quantiles(logp: np.ndarray, borders: np.ndarray, levels) -> np.ndarray:
    """(n, len(levels)) quantiles of a bar distribution with half-normal outer buckets."""
    from scipy.special import erfinv
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    widths = np.diff(borders)
    n, b = p.shape
    rows = np.arange(n)
    cdf = np.cumsum(p, axis=1)
    out = np.empty((n, len(levels)))
    for k, tau in enumerate(levels):
        j = np.minimum((cdf < tau).sum(axis=1), b - 1)
        pj = np.maximum(p[rows, j], 1e-300)
        r = np.clip((tau - (cdf[rows, j] - p[rows, j])) / pj, 0.0, 1.0 - 1e-12)
        v = borders[j] + r * widths[j]
        left, right = j == 0, j == b - 1
        s0, s1 = widths[0] / HALF_NORMAL_MEDIAN, widths[-1] / HALF_NORMAL_MEDIAN
        v[left] = borders[1] - s0 * np.sqrt(2) * erfinv(1.0 - r[left])
        v[right] = borders[-2] + s1 * np.sqrt(2) * erfinv(r[right])
        out[:, k] = v
    return out


def quantiles(Xtr, ytr, Xte, seed, levels, n_est=1) -> np.ndarray:
    """(n_test, len(levels)) on the original target scale."""
    import torch
    m = create(seed, n_est)
    with _captured_decode() as seen, contextlib.redirect_stdout(None):
        m.predict(np.asarray(Xtr, float), np.asarray(ytr, float), np.asarray(Xte, float),
                  task_type="Regression")
    if len(seen) != 1:
        raise RuntimeError(f"expected one regression decode, saw {len(seen)}")
    logp, borders = seen[0]
    q = bar_quantiles(logp, borders, list(levels)) * m.y_std + m.y_mean
    del m
    torch.cuda.empty_cache()
    return q


def mean(Xtr, ytr, Xte, seed, n_est=1) -> np.ndarray:
    """(n_test,) the package's own point prediction, the mean of its bar distribution."""
    import torch
    m = create(seed, n_est)
    with contextlib.redirect_stdout(None):
        out = m.predict(np.asarray(Xtr, float), np.asarray(ytr, float), np.asarray(Xte, float),
                        task_type="Regression")
    del m
    torch.cuda.empty_cache()
    return np.asarray(out, dtype=float)
