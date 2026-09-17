# -*- coding: utf-8 -*-
"""nanoTabPFN adapter: quantiles from the bar distribution of a small pre-trained model.

The models are pre-trained in this repository by
`experiments/h3_repair/prior_pretraining_train.py` on TFM-Playground's architecture, one
per arm of prior (A0, A1) and target encoding (B0 mean and sd, B1 median and IQR / 1.349),
and saved as one file each under `.cache/tfmp/checkpoints/`:

    architecture, model        as TFM-Playground's `train.py` writes them
    bucket_edges               the borders in encoded space
    encoding                   "meansd" or "robust"
    target_clip                the cap on the encoded target, TARGET_CLIP
    max_context, max_features  the largest context and feature count seen in pre-training

Keeping the edges and the encoding in the checkpoint means a model cannot be read with the
borders or the encoding of another arm.

TFM-Playground's own `NanoTabPFNRegressor.predict` returns only the mean. Quantiles come
from `pfns.bar_distribution.BarDistribution.icdf`, which `FullSupportBarDistribution`
inherits in pfns 0.3.0: linear inside every bar, including the two outer ones, so no
quantile lies beyond the outermost border. The half-normal tails enter only the loss and
the mean. This is the same reading as TabPFN's, whose quantile function never reads its
half-normal tail either.

A test row attends only to the context, so splitting test rows into chunks changes nothing
but memory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from common import paths

NAME = re.compile(r"^nanoTabPFN-A[01TC]B[01]-s\d+$")
CHECKPOINTS = paths.ROOT / ".cache" / "tfmp" / "checkpoints"
ENCODINGS = ("meansd", "robust")
# The constant TFM-Playground adds to the sd; the robust scale gets the same floor.
EPS = 1e-8
# The encoded target is capped here in training, in the edges and at inference. On TabICL's
# prior the robust scale is below a hundredth of the sd in 2.7% of contexts, the encoded
# values reach 1e9, and fp16 turns them into NaN. The cap touches 0.7% of contexts under
# A0B1, 2.6% under A1B1 and one table of 56000 under A0B0.
TARGET_CLIP = 1e4
TEST_CHUNK = 512

_loaded: dict = {}


def encode_stats(y, encoding: str) -> tuple[float, float]:
    """(centre, scale) of a context target under one encoding.

    `meansd` is TFM-Playground's: mean and sd with ddof=1, plus 1e-8. `robust` is the
    median and IQR / 1.349 with the same constant, the scale of `transforms.robust_sd`.
    """
    y = np.asarray(y, dtype=float)
    if encoding == "meansd":
        return float(y.mean()), float(y.std(ddof=1)) + EPS
    if encoding == "robust":
        q75, q25 = np.percentile(y, [75, 25])
        return float(np.median(y)), float((q75 - q25) / 1.349) + EPS
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def encode(y, centre: float, scale: float, clip: float = TARGET_CLIP) -> np.ndarray:
    """The target in encoded units, capped at +-clip."""
    return np.clip((np.asarray(y, dtype=float) - centre) / scale, -clip, clip)


def device() -> str:
    """TFMP_DEVICE, or cuda when torch sees a GPU. GPU and CPU kernels differ in the
    last bits, so the value is recorded in provenance."""
    import torch

    from common import env
    return env.text("TFMP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(name: str) -> Path:
    if not NAME.match(name):
        raise ValueError(f"not a nanoTabPFN arm name: {name!r}")
    return CHECKPOINTS / f"{name}.pth"


def legacy_architecture(state) -> tuple[dict, dict]:
    """(architecture, renamed weights) of a bare state dict in the layout TFM-Playground
    released its regressor in, before commit e88b8d9 stored the architecture.

    The released file cannot be read by TFM-Playground's own loader at 7b37681. Its layer
    code is the current one under other names (`transformer_encoder.transformer_blocks`,
    `self_attn_between_*`). The head count is not in the weights; the loader of that time
    set it to embedding size // 64, and so does this one.
    """
    weights = {k.replace("transformer_encoder.", "", 1)
                .replace("self_attn_between_", "self_attention_between_"): v
               for k, v in state.items()}
    e = weights["feature_encoder.linear_layer.weight"].shape[0]
    arch = dict(embedding_size=e, num_attention_heads=e // 64,
                mlp_hidden_size=weights["decoder.linear1.weight"].shape[0],
                num_layers=sum(k.endswith("self_attention_between_datapoints.in_proj_weight")
                               for k in weights),
                num_outputs=weights["decoder.linear2.weight"].shape[0])
    return arch, weights


def load(path, dev: str):
    """(model, bar distribution, meta) from a checkpoint of this repository, or from the
    released TFM-Playground regressor with its buckets file next to it (smoke test only).
    """
    import torch
    from pfns.bar_distribution import FullSupportBarDistribution
    from tfmplayground.models.nanotabpfn import NanoTabPFNModel

    path = Path(path)
    key = (str(path.resolve()), os.path.getmtime(path), dev)
    if key in _loaded:
        return _loaded[key]
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "architecture" in state:
        arch, weights = state["architecture"], state["model"]
    else:
        arch, weights = legacy_architecture(state)
    model = NanoTabPFNModel(embedding_size=arch["embedding_size"],
                            num_attention_heads=arch["num_attention_heads"],
                            mlp_hidden_size=arch["mlp_hidden_size"],
                            num_layers=arch["num_layers"],
                            num_outputs=arch["num_outputs"])
    model.load_state_dict(weights)
    model.to(dev).eval()

    if "bucket_edges" in state:
        edges = state["bucket_edges"]
        # nanoTabPFN-A0B0-s1 was trained before the cap existed; it would have touched one
        # of its 56000 tables. It is read with the cap like every other arm.
        meta = dict(encoding=state["encoding"], max_context=int(state["max_context"]),
                    max_features=int(state["max_features"]),
                    target_clip=float(state.get("target_clip", TARGET_CLIP)))
    else:
        # The released regressor: buckets in a sibling file, trained on 50-row tables.
        edges = torch.load(path.with_name(path.stem + "_buckets.pth"), map_location="cpu")
        meta = dict(encoding="meansd", max_context=None, max_features=None,
                    target_clip=TARGET_CLIP)
    if meta["encoding"] not in ENCODINGS:
        raise ValueError(f"{path.name}: unknown encoding {meta['encoding']!r}")
    dist = FullSupportBarDistribution(edges.float().to(dev))
    _loaded.clear()
    _loaded[key] = (model, dist, meta)
    return _loaded[key]


def predict(path, Xtr, ytr, Xte, levels, dev: str | None = None):
    """(quantiles of shape (n_test, len(levels)), raw-space borders) on the target scale.

    Refuses a context or a feature count beyond what pre-training saw rather than
    truncating: the model would still answer, and the answer would not be the arm's.
    """
    import torch

    dev = dev or device()
    model, dist, meta = load(path, dev)
    Xtr, Xte = np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)
    ytr = np.asarray(ytr, float)
    if Xtr.ndim != 2 or Xte.ndim != 2 or Xtr.shape[1] != Xte.shape[1]:
        raise ValueError(f"feature shapes differ: {Xtr.shape} and {Xte.shape}")
    if not (np.isfinite(Xtr).all() and np.isfinite(Xte).all() and np.isfinite(ytr).all()):
        raise ValueError("non-finite values in the input; nanoTabPFN has no imputer here")
    if meta["max_context"] is not None and len(Xtr) > meta["max_context"]:
        raise ValueError(f"context of {len(Xtr)} rows exceeds the pre-training maximum "
                         f"{meta['max_context']}")
    if meta["max_features"] is not None and Xtr.shape[1] > meta["max_features"]:
        raise ValueError(f"{Xtr.shape[1]} features exceed the pre-training maximum "
                         f"{meta['max_features']}")

    centre, scale = encode_stats(ytr, meta["encoding"])
    y_enc = torch.tensor(encode(ytr, centre, scale, meta["target_clip"]),
                         dtype=torch.float32, device=dev)[None]
    x_ctx = torch.tensor(Xtr, device=dev)
    q = np.empty((len(Xte), len(levels)))
    with torch.no_grad():
        for start in range(0, len(Xte), TEST_CHUNK):
            x_te = torch.tensor(Xte[start:start + TEST_CHUNK], device=dev)
            x = torch.cat([x_ctx, x_te])[None]
            logits = model((x, y_enc), train_test_split_index=len(Xtr)).squeeze(0).float()
            for j, a in enumerate(levels):
                q[start:start + len(x_te), j] = dist.icdf(logits, float(a)).double().cpu().numpy()
    borders = dist.borders.double().cpu().numpy()
    return q * scale + centre, borders * scale + centre


def quantiles(name: str, Xtr, ytr, Xte, seed: int, levels, n_est: int = 1) -> np.ndarray:
    """(n_test, len(levels)) from the checkpoint of arm `name`.

    `seed` is accepted for the common signature: inference is deterministic, and the
    pre-training seed is part of the name. There is no ensemble, so n_est must be 1.
    """
    if n_est != 1:
        raise ValueError(f"nanoTabPFN has no ensemble; n_est={n_est} would be recorded "
                         f"but not run")
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; pre-train the arm first")
    return predict(path, Xtr, ytr, Xte, levels)[0]
