# -*- coding: utf-8 -*-
"""The mean's response to one contaminated row, for EXAONE, comparable with TabFM.

`influence.py` measures Q(0.5), while the TabFM figure is a change in the mean; under a
heavy tail the two move differently. This is `architecture_tabfm.py` with EXAONE in place
of TabFM, in the same profile (N_TRAIN=1000, N_TEST=200, N_EST=1), chosen for TabFM's
size and kept for comparability.

    mean_influence    median (mean_with_row - mean) / |mean|, the row at x = 0 and
                      y = DOSE * max(y_train)
    ratio_true        mean(model) / mean(s / (1 - xi))
    ratio_truncated   mean(model) / mean(E[Y | Y <= max(y_train)])

    XI=0.9 SEEDS=5 N_TRAIN=2000 python -u experiments/h2_leverage/mean_exaone.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, metrics, models, paths, quiet, runner, tables

quiet.silence()

XI = env.floats("XI", [0.5, 0.7, 0.9])
SEEDS = env.seeds(env.integer("SEEDS", 3))
N_TRAIN = env.integer("N_TRAIN", 1000)
N_TEST = env.integer("N_TEST", 200)
N_EST = env.integer("N_EST", 1)
DOSE = env.number("DOSE", 100.0)
OUTPUT = env.text("OUTPUT", "mean_exaone.csv")

KEY = ["xi", "seed", "n_train", "n_test", "n_est", "dose"]
COLUMNS = ["xi", "model", "seed", "n_train", "n_test", "n_est", "dose",
           # the measured size of the treatment; the nominal dose is not a fixed one
           "sd_shift",
           "ratio_true", "ratio_truncated", "mean_influence", "seconds", "reason"]


def measure(cell):
    xi, seed = cell["xi"], cell["seed"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi)
    test = generator.gpd(cell["n_test"], rng, xi=xi)
    top = float(train.y.max())
    y0 = cell["dose"] * top

    def mean(X, y):
        return models.mean("EXAONE", X, y, test.X, seed=seed, n_est=cell["n_est"])

    mu = mean(train.X, train.y)
    mu_row = mean(*leverage.add_row(train.X, train.y, y0))
    return dict(ratio_true=float(np.mean(mu) / np.mean(test.s / (1 - xi))),
                ratio_truncated=float(np.mean(mu) / np.mean(generator.truncated_mean(test, top))),
                mean_influence=leverage.mean_influence(mu, mu_row),
                sd_shift=metrics.sd_shift(np.append(train.y, y0)))


def main():
    cells = runner.grid(xi=XI, seed=SEEDS, n_train=[N_TRAIN], n_test=[N_TEST], n_est=[N_EST],
                        dose=[DOSE], model=["EXAONE"])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"model/true {r['ratio_true']:.3f}  model/truncated "
                                        f"{r['ratio_truncated']:.3f}  influence "
                                        f"{r['mean_influence']:+.1%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== EXAONE, mean, median over seeds ===")
    print(d.groupby("xi")[["ratio_true", "ratio_truncated", "mean_influence"]]
          .median().round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
