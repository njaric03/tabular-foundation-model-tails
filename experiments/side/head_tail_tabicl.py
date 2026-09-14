# -*- coding: utf-8 -*-
"""The tail TabICL's head can express, against the one it is configured to express.

TabICL predicts 999 quantile levels and extrapolates past them with a parametric tail.
`tabicl` 2.1.1 `_model/quantile_dist.py` implements both an exponential tail (xi = 0) and
a GPD one, with the shape bounded to [-0.49, 0.49]. The switch is
`QuantileDistribution(..., tail_type="exp")`, the head is built with that default, and
`TabICLRegressor` exposes no parameter for it.

The model is fitted once on a context with a known xi. Its bank of 999 quantiles is read
out and wrapped in `QuantileDistribution` once per tail type, so only the extrapolation
past 0.999 differs between the two columns. The implied xi is read far enough out that
the tail decides, from Q(0.99999) and Q(0.9999).

Predictions written before the run: `exp` gives an implied xi near zero at every true xi;
`gpd` rises with the true xi and saturates near 0.49. If `gpd` does not recover more than
`exp`, the branch is dead code and nothing should be filed upstream.

    XI=0.2,0.5,0.8 SEEDS=3 python -u experiments/side/head_tail_tabicl.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import env, generator, metrics, models, paths, quiet, runner, tables

quiet.silence()

OUTPUT = env.text("OUTPUT", "head_tail_tabicl.csv")
XI = env.floats("XI", [0.0, 0.2, 0.5, 0.8])
SEEDS = list(range(env.integer("SEEDS", 3)))
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = env.integer("N_TEST", 200)
N_EST = env.integer("N_EST", 1)
# 0.999 is the last level the bank carries; past it the tail family decides.
LEVELS = [0.5, 0.999, 0.9999, 0.99999]
ALPHAS = [(i + 1) / 1000.0 for i in range(999)]

COLUMNS = ["xi_true", "tail_type", "seed", "n_est", "n_train",
           "q50", "q999", "q9999", "q99999", "ratio", "xi_implied",
           "eta_r_median", "seconds", "reason"]
KEY = ["xi_true", "tail_type", "seed", "n_est", "n_train"]


def native_bank(Xtr, ytr, Xte, seed, n_est):
    """TabICL's own 999 quantile levels, shape (rows, levels)."""
    m = models.tabicl_regressor(seed, n_est)
    m.fit(Xtr, ytr)
    q = np.asarray(m.predict(Xte, output_type="quantiles", alphas=ALPHAS), dtype=float)
    return q.T if q.shape[0] == len(ALPHAS) and q.shape[0] != len(Xte) else q


def measure(cell):
    from tabicl._model.quantile_dist import QuantileDistribution

    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(cell["n_train"], rng, xi=cell["xi_true"], clip=True)
    Xte = generator.gpd(N_TEST, rng, xi=cell["xi_true"], clip=True).X
    dist = QuantileDistribution(
        torch.tensor(native_bank(train.X, train.y, Xte, cell["seed"], cell["n_est"]),
                     dtype=torch.float64),
        alpha_levels=torch.tensor(ALPHAS, dtype=torch.float64), tail_type=cell["tail_type"])
    q = {a: float(np.median(np.asarray(dist.icdf(torch.tensor(a, dtype=torch.float64)),
                                       dtype=float)))
         for a in LEVELS}
    ratio = ((q[0.99999] - q[0.5]) / (q[0.9999] - q[0.5]) if q[0.9999] > q[0.5] else np.nan)
    eta = getattr(dist, "eta_r", None)
    return dict(q50=q[0.5], q999=q[0.999], q9999=q[0.9999], q99999=q[0.99999], ratio=ratio,
                xi_implied=(metrics.xi_from_ratio(ratio, hi=0.99999, lo=0.9999)
                            if np.isfinite(ratio) else np.nan),
                eta_r_median=float(np.median(np.asarray(eta, dtype=float)))
                if eta is not None else np.nan)


def main() -> None:
    cells = runner.grid(xi_true=XI, tail_type=["exp", "gpd"], seed=SEEDS, n_est=[N_EST],
                        n_train=[N_TRAIN])
    runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
               show=lambda r: f"xi_implied {r['xi_implied']:.3f}  eta {r['eta_r_median']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(d.pivot_table(index="xi_true", columns="tail_type", values="xi_implied",
                        aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    main()
