# -*- coding: utf-8 -*-
"""TabFM: truncated support and the influence of one row, on the mean it publishes.

`TabFMRegressor.__init__` carries `outlier_threshold: float = 4.0`, an explicit outlier
clip in preprocessing that neither TabPFN nor TabDPT has. Predictions written before the
run: TabFM is robust to the added row, and because it clips the values the mean consists
of, it falls short of the true mean under a heavy tail.

    venv-tabfm/Scripts/python.exe -u experiments/h2_leverage/architecture_tabfm.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, models, paths, quiet, runner, tables

quiet.silence()

XI = env.floats("XI", [0.5, 0.7, 0.9])
SEEDS = env.seeds(env.integer("SEEDS", 3))
# TabFM has about 1.6 billion parameters, so the CPU profile is small.
N_TRAIN = env.integer("N_TRAIN", 1000)
N_TEST = env.integer("N_TEST", 200)
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "architecture_tabfm.csv")

COLUMNS = ["n_est", "n_train", "n_test", "xi", "model", "seed", "ratio_true",
           "ratio_truncated", "mean_influence", "seconds", "reason"]
KEY = ["xi", "seed", "n_est", "n_train", "n_test"]


def measure(cell):
    xi, seed = cell["xi"], cell["seed"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
    test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)
    top = float(train.y.max())

    def mean(X, y):
        return models.mean("TabFM", X, y, test.X, seed=seed, n_est=cell["n_est"])

    mu = mean(train.X, train.y)
    mu_row = mean(*leverage.add_row(train.X, train.y, 100 * top))
    return dict(ratio_true=float(np.mean(mu) / np.mean(test.s / (1 - xi))),
                ratio_truncated=float(np.mean(mu) / np.mean(generator.truncated_mean(test, top))),
                mean_influence=leverage.mean_influence(mu, mu_row))


def main():
    cells = runner.grid(xi=XI, seed=SEEDS, n_est=[N_EST], n_train=[N_TRAIN],
                        n_test=[N_TEST], model=["TabFM"])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"model/true {r['ratio_true']:.3f}  model/truncated "
                                        f"{r['ratio_truncated']:.3f}  influence "
                                        f"{r['mean_influence']:+.1%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(f"\n=== TabFM, {len(d)} runs, median ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence"]]
          .median().round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
