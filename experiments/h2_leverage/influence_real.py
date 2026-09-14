# -*- coding: utf-8 -*-
"""The influence of one real row, with nothing injected.

`influence.py` injects y = 100 * max, which real tables do not contain. Here a subsample
of a real table is fitted with and without its own largest value, and the change of the
predicted quantiles is that observation's influence. Control: the same with a random row
removed, so the change in sample size alone is measured too.

    DATASETS=freMTPL2sev SEEDS=3 MODELS=TabICLv2 python -u experiments/h2_leverage/influence_real.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = [0.5, 0.9, 0.99]
N_FIT, N_TEST = 2000, 1500
DATASETS = env.names("DATASETS", "freMTPL2sev,OnlineNewsPopularity,"
                                 "Buzzinsocialmedia_Twitter,Allstate_Claims_Severity")
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
SEEDS = env.seeds(env.integer("SEEDS", 10))
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "influence_real.csv")

COLUMNS = ["n_est", "dataset", "model", "seed", "leverage", "max_y",
           "d_q50", "d_q90", "d_q99", "k_q50", "k_q90", "k_q99", "seconds", "reason"]
KEY = ["dataset", "model", "seed", "n_est"]


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"], positive_only=True)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    seed = cell["seed"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    Xf, yf, Xte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]]

    def without(j):
        keep = np.ones(len(yf), bool)
        keep[j] = False
        return models.quantiles(cell["model"], Xf[keep], yf[keep], Xte, seed=seed,
                                levels=LEVELS, n_est=cell["n_est"])

    top = int(np.argmax(yf))
    ordered = np.sort(yf)
    q_without_max = without(top)
    q_full = models.quantiles(cell["model"], Xf, yf, Xte, seed=seed, levels=LEVELS,
                              n_est=cell["n_est"])
    other = int(rng.integers(len(yf)))
    while other == top:
        other = int(rng.integers(len(yf)))
    d = metrics.rel_change_by_level(q_without_max, q_full)
    k = metrics.rel_change_by_level(without(other), q_full)
    return dict(leverage=float(ordered[-1] / max(ordered[-2], 1e-12)), max_y=float(ordered[-1]),
                d_q50=d[0], d_q90=d[1], d_q99=d[2], k_q50=k[0], k_q90=k[1], k_q99=k[2])


def main():
    cells = runner.grid(dataset=DATASETS, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"leverage {r['leverage']:5.1f}x | largest: q99 "
                                        f"{r['d_q99']:+6.1%} | random: q99 {r['k_q99']:+5.1%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for col, label in (("d_q99", "largest row removed"), ("k_q99", "random row removed")):
        print(f"\n=== change of Q(0.99), {label}, median ===")
        print(d.pivot_table(index="dataset", columns="model", values=col,
                            aggfunc="median").round(3).to_string())
    high = d[d.leverage >= 3]
    print(f"\n=== subsamples with leverage >= 3x: {len(high)} ===")
    if len(high):
        print(high.pivot_table(index="model", values=["d_q99", "k_q99"],
                               aggfunc="median").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
