# -*- coding: utf-8 -*-
"""One place where a model is constructed. Replaces 20 hand-copied dispatchers.

The repo held 20 functions doing the same thing under 13 different names, about
1068 lines, instantiating a regressor in 65 places across 31 files. That is what
made the seeding bug of 25 August possible: one script passed random_state=0 to
TabPFN while giving the other models the seed, so two measurements being
compared were not seeded alike, and there was no single place where that could
be seen. It hit 278 TabPFN rows across 9 CSV files.

RULES BUILT IN
  - EVERY model gets random_state=seed. No exceptions and no silent zero.
  - describe() returns the run parameters as a dict, so they enter the CSV as
    columns. Rule 2 of RULES.md: a knob that is not a column cannot be
    reconstructed afterwards. What the environment sets rather than the script --
    package versions, the commit, the virtualenv -- goes to
    `results/provenance.csv`, written by `common/append.py` itself.
  - HYPERPARAMETERS OF THE CONTROLS ARE FIXED, NOT TUNED. The tree models run at
    the settings below on every dataset, while the foundation models run at their
    library defaults. For the robustness claim that is irrelevant: the controls
    are there to show that a model which does not standardise the target does not
    react, and no tuning changes that. For the H1 table, where GBM captures more
    of the shape gradient than the foundation models do, it is a limitation worth
    stating rather than defending.
  - An unknown model name fails immediately instead of silently falling through
    to the GBM branch, which is what most of the copies did.

    from common import models

    Q = models.quantiles("TabPFN-V3", Xtr, ytr, Xte, seed=7000,
                         levels=[0.5, 0.9, 0.99], n_est=4)   # (n_test, 3)
    m = models.mean("TabDPT", Xtr, ytr, Xte, seed=7000, n_est=1)   # (n_test,)
    row.update(models.describe("TabPFN-V3", seed=7000, n_est=4))
"""
from __future__ import annotations

import numpy as np

# Names as they appear in the existing CSV files. Synonyms are mapped so older
# MODELS= invocations keep working.
SYNONYMS = {
    "TabPFNv3": "TabPFN-V3", "TabPFN-v3": "TabPFN-V3",
    "TabPFNv2": "TabPFN-v2.5", "TabICL": "TabICLv2",
    "GradientBoosting": "GBM", "XGBoost": "XGB", "CatBoost": "CB",
}

# TabPFN generations are selected through model_path; package 8.4.0 ships
# v2.5, v2.6 and v3.
TABPFN_PATHS = {
    "TabPFN-v2.5": "tabpfn-v2.5-regressor-v2.5_real.ckpt",
    "TabPFN-v2.6": "tabpfn-v2.6-regressor-v2.6_default.ckpt",
}

SUPPORTED = ["GBM", "XGB", "CB", "TabICLv2", "TabPFN-V3", "TabPFN-v2.5",
             "TabPFN-v2.6", "EXAONE", "TabDPT", "TabFM"]

# Models with no conditional distribution at all. They can only be measured on
# the mean, which is a property of the model, not a gap in the measurements.
MEAN_ONLY = ["TabFM"]


def parse_list(value: str) -> list[str]:
    """Split a `MODELS=` value and canonicalise every name in it.

    Scripts used to write the caller's spelling straight into the `model`
    column, so `MODELS=XGBoost` and `MODELS=XGB` produced two labels for one
    model and `prevalence_models.csv` disagrees with `coverage.csv` to this
    day. Parsing through here means the CSV carries the canonical name whatever
    the caller typed.
    """
    return [normalise(x.strip()) for x in value.split(",") if x.strip()]


def normalise(name: str) -> str:
    name = SYNONYMS.get(name, name)
    if name not in SUPPORTED:
        raise ValueError(
            f"unknown model {name!r}. Supported: {', '.join(SUPPORTED)}.\n"
            f"Unknown names used to fall through to the GBM branch silently, so a "
            f"run looked successful while measuring the wrong model.")
    return name


def describe(name: str, seed: int, n_est: int, **extra) -> dict:
    """Run parameters as CSV columns. See rule 2 in RULES.md.

    Kept because it names the minimum every row must carry, but nothing calls it:
    the scripts build their own row dicts, which is why it was written and then
    forgotten. What cannot be forgotten now is the environment: `common/append.py`
    records package versions, the commit and the environment knobs into
    `results/provenance.csv` on the first write of every process, without any
    script having to remember to ask.
    """
    return dict(model=normalise(name), seed=seed, n_est=n_est, **extra)


# ------------------------------------------------------------------- quantiles

def quantiles(name: str, Xtr, ytr, Xte, seed: int, levels, n_est: int = 1) -> np.ndarray:
    """(n_test, len(levels)) predicted quantiles on the original target scale.

    seed goes into random_state of EVERY model. If some model ever has to be
    seeded differently, that must be explicit and argued here, not a forgotten
    zero in one of twenty copies.
    """
    name = normalise(name)
    levels = list(levels)
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    ytr = np.asarray(ytr, float)

    if name == "EXAONE":
        from common.adapters import exaone
        m = exaone.napravi(seed=seed, n_est=n_est)
        m.fit(Xtr, ytr)
        return np.asarray(exaone.kvantili(m, Xte, levels), dtype=float)

    if name == "TabDPT":
        from common.adapters import tabdpt
        return tabdpt.quantiles(Xtr, ytr, Xte, seed, levels)

    if name == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=n_est, device="cpu", random_state=seed)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=levels),
                          dtype=float)

    if name.startswith("TabPFN"):
        from tabpfn import TabPFNRegressor
        kw = dict(n_estimators=n_est, device="cpu", random_state=seed,
                  ignore_pretraining_limits=True)
        if name in TABPFN_PATHS:
            kw["model_path"] = TABPFN_PATHS[name]
        m = TabPFNRegressor(**kw)
        m.fit(Xtr, ytr)
        return np.stack([np.asarray(a, dtype=float) for a in
                         m.predict(Xte, output_type="quantiles", quantiles=levels)],
                        axis=1)

    if name == "XGB":
        # The serious tree baseline; sklearn GBM is the pedagogical one.
        import xgboost as xgb
        m = xgb.XGBRegressor(objective="reg:quantileerror",
                             quantile_alpha=np.array(levels),
                             n_estimators=300, max_depth=4, learning_rate=0.1,
                             random_state=seed, verbosity=0)
        m.fit(Xtr, ytr)
        q = np.asarray(m.predict(Xte), dtype=float)
        return q if q.ndim == 2 else q.reshape(len(Xte), -1)

    if name == "CB":
        # Best tree model in regression on TabArena, so the fair contrast for
        # the foundation models. MultiQuantile fits all levels in one model.
        from catboost import CatBoostRegressor
        alphas = ",".join(str(a) for a in levels)
        m = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}",
                              iterations=500, depth=6, learning_rate=0.1,
                              random_seed=seed, verbose=False, allow_writing_files=False)
        m.fit(Xtr, ytr)
        q = np.asarray(m.predict(Xte), dtype=float)
        return q if q.ndim == 2 else q.reshape(len(Xte), -1)

    if name == "GBM":
        # GBM is NOT deterministic in random_state despite subsample=1.0 and
        # max_features=None: the sklearn splitter uses it to order features and
        # break ties. Measured on identical data, the median relative prediction
        # difference is 9.6%. At the reported sample sizes the effect is noise
        # (28% vs 29% shape share, sd 0.25 over 6 seeds), but it is still seeded
        # properly here so the control arm follows the same rule as the rest.
        from sklearn.ensemble import GradientBoostingRegressor
        return np.column_stack([
            GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=200,
                                      max_depth=3, random_state=seed)
            .fit(Xtr, ytr).predict(Xte)
            for a in levels])

    raise NotImplementedError(
        f"{name} has no quantile head in this module; use mean() instead.")


# ------------------------------------------------------------------------ mean

def mean(name: str, Xtr, ytr, Xte, seed: int, n_est: int = 1) -> np.ndarray:
    """(n_test,) point prediction. Also for models with no quantile head."""
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
        m = exaone.napravi(seed=seed, n_est=n_est)
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

    # The rest have a quantile head; the median is the closest point summary
    # without extra assumptions.
    return quantiles(name, Xtr, ytr, Xte, seed, [0.5], n_est)[:, 0]
