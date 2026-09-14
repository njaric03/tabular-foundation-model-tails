# -*- coding: utf-8 -*-
"""Does the architecture of the output head predict tail behaviour? TabDPT's mean.

From the source: TabPFN scales a fixed z-space bin grid by the empirical sd; TabDPT does
the same with bins fixed on mean +- 10 sd; TabICL predicts 999 quantiles with no grid.
Hence three predictions that can fail: TabPFN and TabDPT show a truncated support (the
mean closer to the truncated than to the true mean), both lose resolution under a leverage
row, and TabDPT cannot predict above mean + 10 sd of the training target.

Only the public mean is used:

    ratio_true        model mean / true conditional mean
    ratio_truncated   model mean / E[Y | Y <= max(y_train)]
    mean_influence    change of the mean after one added row at y = 100 * max
    bound_coverage    share of rows whose true mean lies below mean + 10 sd

    venv-tabdpt/Scripts/python.exe -u experiments/h2_leverage/architecture_tabdpt.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
XI = env.floats("XI", [0.5, 0.7, 0.9])
SEEDS = env.seeds(env.integer("SEEDS", 3))
OUTPUT = env.text("OUTPUT", "architecture_tabdpt.csv")

COLUMNS = ["xi", "model", "seed", "ratio_true", "ratio_truncated", "mean_influence",
           "bound_coverage", "seconds", "reason"]
KEY = ["xi", "model", "seed"]


def measure(cell):
    xi, seed = cell["xi"], cell["seed"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    test = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    top = float(train.y.max())
    true_mean = test.s / (1 - xi)

    def mean(X, y):
        return models.mean("TabDPT", X, y, test.X, seed=seed)

    mu = mean(train.X, train.y)
    mu_row = mean(*leverage.add_row(train.X, train.y, 100 * top))
    bound = train.y.mean() + 10.0 * train.y.std()
    return dict(ratio_true=float(np.mean(mu) / np.mean(true_mean)),
                ratio_truncated=float(np.mean(mu) / np.mean(generator.truncated_mean(test, top))),
                mean_influence=leverage.mean_influence(mu, mu_row),
                bound_coverage=float(np.mean(true_mean <= bound)))


def main():
    cells = runner.grid(xi=XI, model=["TabDPT"], seed=SEEDS)
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"model/true {r['ratio_true']:.3f}  model/truncated "
                                        f"{r['ratio_truncated']:.3f}  influence "
                                        f"{r['mean_influence']:+.1%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== TabDPT, median over seeds ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence",
                           "bound_coverage"]].median().round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
