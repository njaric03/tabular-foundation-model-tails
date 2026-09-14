# -*- coding: utf-8 -*-
"""Does a truncated support explain the deficit in the conditional mean?

A head that cannot place mass above what the context supports should, if it is otherwise
right, match E[Y | x, Y <= max(y_train)] rather than the true E[Y | x] = s / (1 - xi):

    ratio_truncated ~ 1, ratio_true < 1   the deficit is the truncation
    both below 1                          truncation explains part of it
    ratio_truncated > 1                   the explanation fails

    XI=0.7,0.9 SEEDS=3 python -u experiments/h1_shape_vs_scale/truncated_support.py
"""
import numpy as np
import pandas as pd

from common import env, generator, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
XI = env.floats("XI", [0.5, 0.7, 0.9])
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
N_EST = env.integer("N_EST", 4)
OUTPUT = env.text("OUTPUT", "truncated_support.csv")

COLUMNS = ["n_est", "xi", "model", "seed", "ratio_true", "ratio_truncated", "threshold_std",
           "share_above_threshold", "seconds", "reason"]
KEY = ["xi", "model", "seed", "n_est"]


def measure(cell):
    xi, seed = cell["xi"], cell["seed"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    test = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    top = float(train.y.max())
    true_mean = test.s / (1 - xi)
    truncated_mean = test.s * metrics.truncated_mean_std(top / test.s, xi)
    mu = models.mean(cell["model"], train.X, train.y, test.X, seed=seed, n_est=cell["n_est"])
    return dict(ratio_true=float(np.mean(mu) / np.mean(true_mean)),
                ratio_truncated=float(np.mean(mu) / np.mean(truncated_mean)),
                threshold_std=float(np.median(top / test.s)),
                share_above_threshold=float((test.y > top).mean()))


def main():
    cells = runner.grid(xi=XI, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"model/true {r['ratio_true']:.3f}  "
                                        f"model/truncated {r['ratio_truncated']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    g = d.groupby(["xi", "model"])[["ratio_true", "ratio_truncated"]].mean()
    # The share of the gap to the true mean that the truncation accounts for.
    g["explained"] = (g.ratio_truncated - g.ratio_true) / (1 - g.ratio_true)
    print("\n=== model mean against the true and the truncated mean (1.00 is exact) ===")
    print(g.round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
