# -*- coding: utf-8 -*-
"""Does the shape share grow with the size of the context?

At n_train = 2000 the models capture about 94% of the scale change and under half of the
shape change, while an extreme-value oracle recovers 98% of the shape. If the share
climbs towards the oracle with n_train, the deficit is a matter of data; if it saturates
while the scale stays high, it is what the model brought from pre-training.

    N_GRID=500,1000,2000 SEEDS=3 python -u experiments/h1_shape_vs_scale/sample_complexity.py
"""
import numpy as np
import pandas as pd

from common import env, generator, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TEST = 900
N_EST = env.integer("N_EST", 4)
N_GRID = env.integers("N_GRID", [500, 1000, 2000, 4000])
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("GBM,TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "sample_complexity.csv")

COLUMNS = ["n_train", "model", "seed", "n_est", "shape_share", "scale_share",
           "xi_t1", "xi_t3", "xi_true_t1", "xi_true_t3", "seconds", "reason"]
KEY = ["n_train", "model", "seed", "n_est"]


def measure(cell):
    seed = cell["seed"]
    # Train and test come from separate streams, so the test set is the same at every
    # n_train. The published CSV drew both from one stream; the notebook says so.
    train = generator.gpd(cell["n_train"], np.random.default_rng(seed), xi=generator.XI_OF_X)
    test = generator.gpd(N_TEST, np.random.default_rng(seed + 1_000_000), xi=generator.XI_OF_X)
    q = models.quantiles(cell["model"], train.X, train.y, test.X, seed=seed,
                         levels=gradients.LEVELS, n_est=cell["n_est"])
    return gradients.generator_shares(q, test.X, lambda rows: float(test.xi[rows].mean()),
                                      test.s * metrics.gpd_quantile(0.5, test.xi))


def main():
    cells = runner.grid(n_train=N_GRID, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"shape {r['shape_share']:6.0%}  "
                                        f"scale {r['scale_share']:6.0%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for col in ("shape_share", "scale_share"):
        print(f"\n=== {col} by n_train ===")
        print(d.pivot_table(index="n_train", columns="model", values=col,
                            aggfunc="mean").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
