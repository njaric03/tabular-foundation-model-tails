# -*- coding: utf-8 -*-
"""Does winsorising the target help the models that publish only a mean (TabDPT, TabFM)?

Log and rank transforms do not map a mean back exactly, since E[g^-1(Z)] != g^-1(E[Z]);
winsorising stays in the original units. Measured:

    benefit   the change of the mean after one added row at y = 100 * max
    cost      the ratio to the truncated mean on the clean context

    venv-tabdpt/Scripts/python.exe -u experiments/h3_repair/repair_mean.py
    MODEL=TabFM venv-tabfm/Scripts/python.exe -u experiments/h3_repair/repair_mean.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, models, paths, quiet, runner, tables

quiet.silence()

WINSOR_SHARE = 0.005
MODEL = env.models("TabDPT", name="MODEL")[0]
N_EST = env.integer("N_EST", 4)
XI = env.floats("XI", [0.7, 0.9])
SEEDS = env.seeds(env.integer("SEEDS", 5))
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = env.integer("N_TEST", 900)
OUTPUT = env.text("OUTPUT", f"repair_mean_{MODEL.lower()}.csv")

COLUMNS = ["n_train", "n_test", "n_est", "xi", "model", "variant", "seed", "mean_influence",
           "ratio_truncated", "seconds", "reason"]
KEY = ["xi", "model", "variant", "seed", "n_est", "n_train", "n_test"]


def target(variant, y):
    return y if variant == "raw" else np.minimum(y, np.quantile(y, 1 - WINSOR_SHARE))


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(cell["n_train"], rng, xi=cell["xi"], clip=True)
    test = generator.gpd(cell["n_test"], rng, xi=cell["xi"], clip=True)
    top = float(train.y.max())

    def mean(X, y):
        return models.mean(cell["model"], X, target(cell["variant"], y), test.X,
                           seed=cell["seed"], n_est=cell["n_est"])

    mu = mean(train.X, train.y)
    mu_row = mean(*leverage.add_row(train.X, train.y, 100 * top))
    return dict(mean_influence=leverage.mean_influence(mu, mu_row),
                ratio_truncated=float(np.mean(mu) / np.mean(generator.truncated_mean(test, top))))


def main():
    cells = runner.grid(xi=XI, model=[MODEL], variant=["raw", "winsor"], seed=SEEDS,
                        n_est=[N_EST], n_train=[N_TRAIN], n_test=[N_TEST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"influence {r['mean_influence']:+8.1%}  "
                                        f"model/truncated {r['ratio_truncated']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(f"\n=== {MODEL}, median: influence of one row, and ratio to the truncated mean ===")
    print(d.pivot_table(index="xi", columns="variant", values=["mean_influence", "ratio_truncated"],
                        aggfunc="median").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
