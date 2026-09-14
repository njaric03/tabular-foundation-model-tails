# -*- coding: utf-8 -*-
"""The dissociation on real data without a chosen feature and without a fragile quotient.

`dissociation_real.py` takes one feature, the maximum over columns, and divides the
model's tail gradient by the reference one along it. The maximum is mostly selection
noise (`reference_instrument.csv`) and a small denominator explodes.

Here the model is fitted once per (dataset, model, seed) and its predictions are read
through the terciles of every usable feature. The share becomes the slope of the model's
gradient on the reference gradient across features, through the origin, reported raw and
corrected for measurement error in the reference:

    slope = sum(ref * mod) / (sum(ref^2) - sum(var_ref))

with var_ref bootstrapped over the test rows per feature. The shape reference is far
noisier than the scale one, so without the correction a worse instrument alone could
produce the dissociation. `instrument_ceiling.py` measures what a perfect model scores.

    MODELS=TabICLv2 DATASETS=houses SEEDS=1 \
      python -u experiments/h1_shape_vs_scale/dissociation_real_slope.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, gradients, models, paths, quiet, runner, stats, tables

quiet.silence()

LEVELS = gradients.LEVELS
N_FIT, N_TEST = gradients.FIT_ROWS, gradients.TEST_ROWS
N_BOOT = env.integer("N_BOOT", gradients.N_BOOT)
N_EST = env.integer("N_EST", 4)
SEEDS = env.seeds(env.integer("SEEDS", 3))
DATASETS = env.names("DATASETS", datasets.DISSOCIATION)
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
OUTPUT = env.text("OUTPUT", "dissociation_real_slope.csv")

COLUMNS = ["dataset", "model", "seed", "n_est", "n_features",
           "slope_shape", "slope_shape_corrected", "r_shape",
           "slope_scale", "slope_scale_corrected", "r_scale",
           "var_share_shape", "var_share_scale", "seconds", "reason"]
KEY = ["dataset", "model", "seed", "n_est"]


def slopes(g) -> dict:
    """The row fields from `gradients.feature_gradients` output."""
    if len(g["ref_shape"]) < 3:
        raise RuntimeError(f"only {len(g['ref_shape'])} usable features")
    a = gradients.slope_through_origin(g["ref_shape"], g["model_shape"], g["var_shape"])
    b = gradients.slope_through_origin(g["ref_scale"], g["model_scale"], g["var_scale"])
    return dict(n_features=len(g["ref_shape"]),
                slope_shape=a[0], slope_shape_corrected=a[1], r_shape=a[2], var_share_shape=a[3],
                slope_scale=b[0], slope_scale_corrected=b[1], r_scale=b[2], var_share_scale=b[3])


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"], positive_only=True)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    rng = np.random.default_rng(cell["seed"])
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    Xf, yf, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
    s_te = gradients.log_linear_scale(Xf, yf)(Xte)
    # One fit; the terciles regroup the same predictions, so the cost does not grow
    # with the number of features.
    q = models.quantiles(cell["model"], Xf, yf, Xte, seed=cell["seed"], levels=LEVELS,
                         n_est=cell["n_est"])
    return slopes(gradients.feature_gradients(Xf, Xte, yte, s_te, q, rng, N_BOOT))


def main():
    cells = runner.grid(dataset=DATASETS, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"shape {r['slope_shape']:+.2f} "
                                        f"(corrected {r['slope_shape_corrected']:+.2f}) | "
                                        f"scale {r['slope_scale']:+.2f} "
                                        f"(corrected {r['slope_scale_corrected']:+.2f}) | "
                                        f"{r['n_features']} features")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    print("\n=== slope by model, median over datasets and seeds ===")
    print(d.groupby("model")[["slope_scale", "slope_scale_corrected", "slope_shape",
                              "slope_shape_corrected"]].median().round(3).to_string())
    print("\n=== share of the reference variance that is measurement error ===")
    print(d.groupby("model")[["var_share_scale", "var_share_shape"]]
          .median().round(3).to_string())
    r = stats.paired_by_target(d, "slope_scale_corrected", "slope_shape_corrected")
    if r["clusters"]:
        print(f"\nscale closer to 1 than shape: {stats.describe(r, unit='target vectors')}")


if __name__ == "__main__":
    main()
