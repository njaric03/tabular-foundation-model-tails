# -*- coding: utf-8 -*-
"""Does ensemble aggregation compress the tail?

TabPFN and TabICL average their members quantile by quantile (Vincentization), which
gives a lighter tail than a mixture of the members would. On a generator with a constant
xi, the implied xi is measured at 1, 2, 4 and 8 members with everything else fixed.

    XI=0.7 SEEDS=5 python -u experiments/h1_shape_vs_scale/vincentization.py
"""
import numpy as np
import pandas as pd

from common import env, generator, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
XI = env.floats("XI", [0.3, 0.7, 0.9])
N_EST = env.integers("N_EST", [1, 2, 4, 8])
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "vincentization.csv")

COLUMNS = ["xi_true", "model", "n_estimators", "seed", "xi_implied", "seconds", "reason"]
KEY = ["xi_true", "model", "n_estimators", "seed"]


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(N_TRAIN, rng, xi=cell["xi_true"], clip=True)
    test = generator.gpd(N_TEST, rng, xi=cell["xi_true"], clip=True)
    q = models.quantiles(cell["model"], train.X, train.y, test.X, seed=cell["seed"],
                         levels=gradients.LEVELS, n_est=cell["n_estimators"])
    return dict(xi_implied=metrics.implied_xi(q[:, 2], q[:, 1]))


def main():
    cells = runner.grid(xi_true=XI, model=MODELS, n_estimators=N_EST, seed=SEEDS)
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"implied xi {r['xi_implied']:+.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== implied xi by number of members ===")
    print(d.pivot_table(index=["model", "xi_true"], columns="n_estimators",
                        values="xi_implied").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
