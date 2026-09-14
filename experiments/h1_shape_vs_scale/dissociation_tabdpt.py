# -*- coding: utf-8 -*-
"""The shape/scale dissociation for TabDPT, from quantiles read out of its bin head.

TabDPT publishes only a mean; `common/adapters/tabdpt.py` recovers its internal
distribution. The measure is the one `sample_complexity.py` applies to the other models,
at n_train = 2000 and one ensemble member.

    venv-tabdpt/Scripts/python.exe -u experiments/h1_shape_vs_scale/dissociation_tabdpt.py
"""
import numpy as np
import pandas as pd

from common import env, generator, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
SEEDS = env.seeds(env.integer("SEEDS", 10))
OUTPUT = env.text("OUTPUT", "dissociation_tabdpt.csv")

COLUMNS = ["seed", "shape_share", "scale_share", "xi_t1", "xi_t3",
           "xi_true_t1", "xi_true_t3", "seconds", "reason"]
KEY = ["seed"]


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
    test = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
    q = models.quantiles("TabDPT", train.X, train.y, test.X, seed=cell["seed"],
                         levels=gradients.LEVELS)
    return gradients.generator_shares(q, test.X, lambda rows: float(test.xi[rows].mean()),
                                      test.s * metrics.gpd_quantile(0.5, test.xi))


def main():
    seconds = runner.run(OUTPUT, COLUMNS, KEY, runner.grid(seed=SEEDS), measure,
                         show=lambda r: f"shape {r['shape_share']:6.0%}  "
                                        f"scale {r['scale_share']:6.0%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(f"\n=== TabDPT, {len(d)} seeds: mean (sd) ===")
    for col in ("scale_share", "shape_share"):
        print(f"  {col}  {d[col].mean():.0%}  ({d[col].std():.2f})")
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
