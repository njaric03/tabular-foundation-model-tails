# -*- coding: utf-8 -*-
"""Swap the scale estimator for a robust one after the fit, without touching the data.

Standardisation undoes any affine map of the input, so the estimator itself is patched:
the sd the model divides by is replaced by IQR / 1.349 after `fit`, and TabPFN's
raw-space grid is rebuilt.

This does not test a robust scale. `clip_context.py` found that `fit` has already
normalised the context by the contaminated sd, so after the swap the network conditions
on one scale and decodes with another, 0.053 times smaller. That mismatch alone explains
the +925% response and the worse pinball at every level recorded here.

    XI=0.9 SEEDS=3 MODELS=TabICLv2 python -u experiments/h3_repair/robust_scale.py
"""
import numpy as np
import pandas as pd

from common import (
    env,
    generator,
    leverage,
    metrics,
    models,
    paths,
    quiet,
    runner,
    tables,
    transforms,
)

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
XI = env.floats("XI", [0.7, 0.9])
VARIANTS = env.names("VARIANTS", "raw,robust")
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3")
# The patch is written against a single estimator.
N_EST = 1
OUTPUT = env.text("OUTPUT", "robust_scale.csv")

COLUMNS = ["n_est", "xi", "model", "variant", "seed", "response_q99", "response_q999",
           "pb50", "pb90", "pb99", "pb999", "scale_ratio", "seconds", "reason"]
KEY = ["xi", "model", "variant", "seed", "n_est"]


def quantiles(variant, name, X, y, Xte, seed):
    """Quantiles, with the fitted target scale replaced by the robust one for `robust`.
    Returns the quantiles and the ratio of the new scale to the old one."""
    ratio = np.nan
    if name == "TabICLv2":
        m = models.tabicl_regressor(seed, N_EST)
        m.fit(X, y)
        if variant == "robust":
            scaler = getattr(m, "y_scaler_", None) or getattr(m, "scaler_", None)
            if scaler is None or not hasattr(scaler, "scale_"):
                raise RuntimeError("TabICL target scaler not found")
            old, new = float(np.ravel(scaler.scale_)[0]), transforms.robust_sd(y)
            scaler.scale_ = np.full_like(np.asarray(scaler.scale_, dtype=float), new)
            ratio = new / old
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=LEVELS)), ratio
    if name.startswith("TabPFN"):
        m = models.tabpfn_regressor(name, seed, N_EST)
        m.fit(X, y)
        if variant == "robust":
            old, new = float(m.y_train_std_), transforms.robust_sd(y)
            m.y_train_std_ = new
            m._rebuild_raw_space_bardist()          # the borders depend on the sd
            ratio = new / old
        return models.tabpfn_quantiles(m, Xte, LEVELS), ratio
    raise ValueError(f"{name}: the scale patch exists for TabICL and TabPFN only")


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(N_TRAIN, rng, xi=cell["xi"], clip=True)
    test = generator.gpd(N_TEST, rng, xi=cell["xi"], clip=True)
    args = (cell["variant"], cell["model"])
    q_clean, ratio = quantiles(*args, train.X, train.y, test.X, cell["seed"])
    q_row, _ = quantiles(*args, *leverage.add_row(train.X, train.y, 100 * train.y.max()),
                         test.X, cell["seed"])
    response = metrics.rel_change_by_level(q_clean, q_row, positive=True)
    return dict(response_q99=response[2], response_q999=response[3], scale_ratio=ratio,
                **{f"pb{t}": metrics.pinball(test.y, q_clean[:, i], a)
                   for i, (a, t) in enumerate(zip(LEVELS, TAGS))})


def main():
    cells = runner.grid(xi=XI, model=MODELS, variant=VARIANTS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"response q99 {r['response_q99']:+7.1%}  "
                                        f"q999 {r['response_q999']:+7.1%}  pb99 {r['pb99']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for col in ("response_q99", "response_q999"):
        print(f"\n=== benefit: {col}, median ===")
        print(d.pivot_table(index=["xi", "model"], columns="variant", values=col,
                            aggfunc="median").round(3).to_string())
    for col in ("pb50", "pb90", "pb99", "pb999"):
        p = d.pivot_table(index=["xi", "model"], columns="variant", values=col, aggfunc="median")
        print(f"\n=== cost: {col} on the clean context, relative to raw ===")
        print(p.div(p["raw"], axis=0).sub(1).round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
