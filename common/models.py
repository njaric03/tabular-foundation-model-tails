# -*- coding: utf-8 -*-
"""The one place a model is built and asked for quantiles or a mean.

    Q = models.quantiles("TabPFN-V3", Xtr, ytr, Xte, seed=7000, levels=[0.5, 0.9, 0.99])
    m = models.mean("TabDPT", Xtr, ytr, Xte, seed=7000)

Every model gets `random_state=seed` (rule 4). An unknown name raises instead of
falling through to a default model. The tree controls run at fixed settings on every
dataset while the foundation models run at their defaults; for the robustness claim
that is irrelevant, for the H1 table it is a limitation to state.
"""
from __future__ import annotations

import numpy as np

from common import env

# Other spellings that older MODELS= invocations used.
SYNONYMS = {
    "TabPFNv3": "TabPFN-V3", "TabPFN-v3": "TabPFN-V3",
    "TabPFNv2": "TabPFN-v2.5", "TabICL": "TabICLv2",
    "GradientBoosting": "GBM", "XGBoost": "XGB", "CatBoost": "CB",
}

# TabPFN generations other than V3, selected through `model_path`; 8.4.0 ships them.
TABPFN_PATHS = {
    "TabPFN-v2.5": "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    "TabPFN-v2.6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
}

# Small models pre-trained here, one per arm of prior (A) and target encoding (B) and
# pre-training seed; see common/adapters/nanotabpfn.py. Seeds 1 to 3 are the first
# setting, 1024-row tables; 11 and 12 the second, 256-row tables; 41 and 42 the third,
# 50-row tables at a learning rate of 3e-4, the first setting whose models learn anything;
# 61 to 71 later local attempts; 81 and 82 the runs on a Colab GPU in bf16.
# The seed says which setting a checkpoint and a result row belong to.
NANOTABPFN = [f"nanoTabPFN-A{a}B{b}-s{s}" for a in (0, 1) for b in (0, 1)
              for s in (1, 2, 3, 11, 12, 41, 42, 61, 62, 63, 71, 72, 81, 82, 91, 92)]

# TabICLv2 after continued pre-training on the prior arms of
# experiments/h3_repair/tabicl_prior_finetune.py, read from local checkpoints.
TABICL_FT = [f"TabICLv2-FT-A{a}-s{s}" for a in (0, 1) for s in (1, 2)]
TABICL_FT_DIR = ".cache/tfmp/tabicl_ft"

SUPPORTED = ["GBM", "XGB", "CB", "TabICLv2", "TabPFN-V3", "TabPFN-v2.5",
             "TabPFN-v2.6", "EXAONE", "TabDPT", "TabFM"] + NANOTABPFN + TABICL_FT

# CATEGORICAL=native passes the columns `datasets.prepare` coded from categories to
# TabPFN as `categorical_features_indices`; the script sets CATEGORICAL_INDICES from
# `datasets.LAST`. TabICL 2.1.1 has no such parameter.
CATEGORICAL = env.text("CATEGORICAL", "") == "native"
CATEGORICAL_INDICES: list = []


def parse_list(value: str) -> list[str]:
    """Split a `MODELS=` value and return every name under its canonical spelling (rule 7)."""
    return [normalise(x.strip()) for x in value.split(",") if x.strip()]


def normalise(name: str) -> str:
    name = SYNONYMS.get(name, name)
    if name not in SUPPORTED:
        raise ValueError(f"unknown model {name!r}. Supported: {', '.join(SUPPORTED)}.")
    return name


def tabpfn_regressor(name: str, seed: int, n_est: int):
    """An unfitted TabPFNRegressor of generation `name`, seeded, with this repo's settings."""
    from tabpfn import TabPFNRegressor
    kw = dict(n_estimators=n_est, device="cpu", random_state=seed,
              ignore_pretraining_limits=True)
    if name in TABPFN_PATHS:
        kw["model_path"] = TABPFN_PATHS[name]
    if CATEGORICAL and CATEGORICAL_INDICES:
        kw["categorical_features_indices"] = list(CATEGORICAL_INDICES)
    return TabPFNRegressor(**kw)


def tabicl_regressor(seed: int, n_est: int, name: str = "TabICLv2"):
    """An unfitted TabICLRegressor, seeded; a fine-tuned arm loads its local checkpoint
    and never downloads another in its place."""
    from tabicl import TabICLRegressor
    kw = dict(n_estimators=n_est, device="cpu", random_state=seed)
    if name in TABICL_FT:
        from common import paths
        path = paths.ROOT / TABICL_FT_DIR / f"{name}.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; fine-tune the arm first")
        kw.update(model_path=str(path), allow_auto_download=False)
    return TabICLRegressor(**kw)


def tabpfn_quantiles(m, X, levels) -> np.ndarray:
    """(n, len(levels)) from a fitted TabPFNRegressor, which returns one array per level."""
    return np.stack([np.asarray(a, dtype=float) for a in
                     m.predict(X, output_type="quantiles", quantiles=list(levels))],
                    axis=1)


def quantiles(name: str, Xtr, ytr, Xte, seed: int, levels, n_est: int = 1) -> np.ndarray:
    """(n_test, len(levels)) predicted quantiles on the original target scale."""
    name = normalise(name)
    levels = list(levels)
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)

    if name == "EXAONE":
        from common.adapters import exaone
        m = exaone.create(seed=seed, n_est=n_est)
        m.fit(Xtr, ytr)
        return np.asarray(exaone.quantiles(m, Xte, levels), dtype=float)

    if name == "TabDPT":
        # The bin head is read from one forward pass. Accepting n_est > 1 would put
        # that value in the CSV while a single member was measured.
        if n_est != 1:
            raise ValueError(f"TabDPT quantiles are read from one member; n_est={n_est}")
        from common.adapters import tabdpt
        return tabdpt.quantiles(Xtr, ytr, Xte, seed, levels)

    if name in NANOTABPFN:
        from common.adapters import nanotabpfn
        return nanotabpfn.quantiles(name, Xtr, ytr, Xte, seed, levels, n_est)

    if name == "TabICLv2" or name in TABICL_FT:
        m = tabicl_regressor(seed, n_est, name)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=levels),
                          dtype=float)

    if name.startswith("TabPFN"):
        m = tabpfn_regressor(name, seed, n_est)
        m.fit(Xtr, ytr)
        return tabpfn_quantiles(m, Xte, levels)

    if name == "XGB":
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:quantileerror",
                             quantile_alpha=np.array(levels),
                             n_estimators=300, max_depth=4, learning_rate=0.1,
                             random_state=seed, verbosity=0)
        m.fit(Xtr, ytr)
        q = np.asarray(m.predict(Xte), dtype=float)
        return q if q.ndim == 2 else q.reshape(len(Xte), -1)

    if name == "CB":
        # MultiQuantile fits every level in one model.
        from catboost import CatBoostRegressor
        alphas = ",".join(str(a) for a in levels)
        m = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}",
                              iterations=500, depth=6, learning_rate=0.1,
                              random_seed=seed, verbose=False, allow_writing_files=False)
        m.fit(Xtr, ytr)
        q = np.asarray(m.predict(Xte), dtype=float)
        return q if q.ndim == 2 else q.reshape(len(Xte), -1)

    if name == "GBM":
        # random_state matters even with subsample=1.0: the splitter uses it to order
        # features and break ties.
        from sklearn.ensemble import GradientBoostingRegressor
        return np.column_stack([
            GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200,
                                      max_depth=3, random_state=seed)
            .fit(Xtr, ytr).predict(Xte)
            for a in levels])

    raise NotImplementedError(f"{name} has no quantile head; use mean() instead.")


def mean(name: str, Xtr, ytr, Xte, seed: int, n_est: int = 1) -> np.ndarray:
    """(n_test,) point prediction. For models with a quantile head, the median."""
    name = normalise(name)
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)

    if name == "TabFM":
        from common.adapters import tabfm
        return tabfm.mean(Xtr, ytr, Xte, seed, n_est)

    if name == "TabDPT":
        from common.adapters import tabdpt
        return tabdpt.mean(Xtr, ytr, Xte, seed, n_est)

    if name == "EXAONE":
        from common.adapters import exaone
        m = exaone.create(seed=seed, n_est=n_est)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte), dtype=float)

    if name == "GBM":
        from sklearn.ensemble import GradientBoostingRegressor
        return np.asarray(
            GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      random_state=seed).fit(Xtr, ytr).predict(Xte),
            dtype=float)

    if name == "XGB":
        import xgboost as xgb
        m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                             random_state=seed, verbosity=0)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte), dtype=float)

    if name == "CB":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1,
                              random_seed=seed, verbose=False, allow_writing_files=False)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte), dtype=float)

    return quantiles(name, Xtr, ytr, Xte, seed, [0.5], n_est)[:, 0]


def predictive_mean(name: str, Xtr, ytr, Xte, seed: int, n_est: int = 1) -> np.ndarray:
    """(n_test,) the model's own mean, where it publishes one.

    Unlike `mean`, TabICL and TabPFN are asked for `output_type="mean"` instead of the
    median. `mean` keeps the median because truncated_support, repair_mean and
    credibility were measured with it.
    """
    name = normalise(name)
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)

    if name == "TabICLv2" or name in TABICL_FT:
        m = tabicl_regressor(seed, n_est, name)
    elif name.startswith("TabPFN"):
        m = tabpfn_regressor(name, seed, n_est)
    else:
        return mean(name, Xtr, ytr, Xte, seed, n_est)
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte, output_type="mean"), dtype=float)
