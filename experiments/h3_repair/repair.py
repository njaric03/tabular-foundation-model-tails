# -*- coding: utf-8 -*-
"""Does transforming the target remove the response to one outlier, and at what cost?

Winsorising removes the injected row by definition, so a smaller response proves nothing
alone. Two things are measured, and a transform is a repair only if it passes both:

    benefit   the response of Q(0.99) and Q(0.999) to one added row at y = 100 * max
    cost      pinball on the clean context

Variants, all monotone, so quantiles map back exactly:

    raw      as is
    log      fit on log(y), exponentiate
    winsor   the top 0.5% capped, in the original units
    rank     normal scores, inverted through the empirical quantiles
    asinh    median + IQR * asinh((y - median) / IQR), inverted by sinh

    XI=0.9 SEEDS=3 MODELS=TabICLv2 python -u experiments/h3_repair/repair.py
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import env, generator, leverage, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
WINSOR_SHARE = 0.005
XI = env.floats("XI", [0.7, 0.9])
VARIANTS = env.names("VARIANTS", "raw,log,winsor,rank,asinh")
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3")
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "repair.csv")

COLUMNS = ["n_est", "xi", "model", "variant", "seed", "response_q99", "response_q999",
           "pb50", "pb90", "pb99", "pb999", "twcrps", "seconds", "reason"]
KEY = ["xi", "model", "variant", "seed", "n_est"]


def quantiles(variant, name, X, y, Xte, seed, n_est):
    """Fit under a target transform and return quantiles in the original units."""
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)

    def predict(target):
        return models.quantiles(name, X, target, Xte, seed=seed, levels=LEVELS, n_est=n_est)

    if variant == "raw":
        return predict(y)
    if variant == "log":
        return np.exp(predict(np.log(y)))
    if variant == "winsor":
        return predict(np.minimum(y, np.quantile(y, 1 - WINSOR_SHARE)))
    if variant == "asinh":
        med = np.median(y)
        iqr = np.subtract(*np.percentile(y, [75, 25])) or 1.0
        return med + iqr * np.sinh((predict(med + iqr * np.arcsinh((y - med) / iqr)) - med) / iqr)
    if variant == "rank":
        n = len(y)
        q_z = predict(norm.ppf((np.argsort(np.argsort(y)) + 0.5) / n))
        return np.interp(q_z, norm.ppf((np.arange(n) + 0.5) / n), np.sort(y))
    raise ValueError(variant)


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(N_TRAIN, rng, xi=cell["xi"], clip=True)
    test = generator.gpd(N_TEST, rng, xi=cell["xi"], clip=True)
    args = (cell["variant"], cell["model"])
    q_clean = quantiles(*args, train.X, train.y, test.X, cell["seed"], cell["n_est"])
    q_row = quantiles(*args, *leverage.add_row(train.X, train.y, 100 * train.y.max()), test.X,
                      cell["seed"], cell["n_est"])
    response = metrics.rel_change_by_level(q_clean, q_row, positive=True)
    r = dict(response_q99=response[2], response_q999=response[3])
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        r[f"pb{t}"] = metrics.pinball(test.y, q_clean[:, i], a)
    # The mean pinball of the thresholded quantiles, half of metrics.twcrps; kept so new
    # rows match the published ones.
    threshold = float(np.quantile(test.y, 0.90))
    r["twcrps"] = float(np.mean([metrics.pinball(np.maximum(test.y, threshold),
                                                 np.maximum(q_clean[:, i], threshold), a)
                                 for i, a in enumerate(LEVELS)]))
    return r


def main():
    cells = runner.grid(xi=XI, model=MODELS, variant=VARIANTS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"response {r['response_q99']:+7.1%}  pb99 {r['pb99']:.4f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== benefit: response of Q(0.99) to one added row, median ===")
    print(d.pivot_table(index=["xi", "model"], columns="variant", values="response_q99",
                        aggfunc="median").round(3).to_string())
    for col in ("pb99", "pb50"):
        p = d.pivot_table(index=["xi", "model"], columns="variant", values=col, aggfunc="median")
        print(f"\n=== cost: {col} on the clean context, relative to raw ===")
        print(p.div(p["raw"], axis=0).sub(1).round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
