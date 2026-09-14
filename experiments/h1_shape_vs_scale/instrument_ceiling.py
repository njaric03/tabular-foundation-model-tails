# -*- coding: utf-8 -*-
"""What `dissociation_real_slope.py` reports for a model that is right.

The text reads that slope as a share, 1.00 meaning the model tracks the reference. That
assumes a perfect model scores 1.00, and it need not: the reference is a noisy Hill
estimate while the model side is nearly exact, and noise in the reference rescales the
slope. On the generator xi(x) is known, so the model can be replaced by the true
conditional quantiles; whatever then differs from 1.00 is the instrument.

The same code path as the real-data script (`gradients.feature_gradients`), on generator
designs padded with zero-weight noise features to the widths the real tables have, plus:

    instrument_slope   slope of the reference on the true gradient

First pass, 10 seeds per width, k = 0.10: the scale reads 1.00 (corrected 0.95 to 1.01)
at every width. The raw shape slope falls from 1.23 at d = 6 to 0.37 at d = 80; the
corrected one does not depend on d but sits near 1.5, so a shape share has to be divided
by the ceiling at its own width rather than read against 1.00.

    D=6,80 SEEDS=3 python -u experiments/h1_shape_vs_scale/instrument_ceiling.py
"""
import numpy as np
import pandas as pd

from common import env, generator, gradients, metrics, paths, quiet, runner, tables

quiet.silence()

LEVELS = gradients.LEVELS
N_FIT, N_TEST = gradients.FIT_ROWS, gradients.TEST_ROWS
# The feature counts that occur in dissociation_real_slope.csv, so every dataset meets
# the ceiling at its own width.
D_LIST = env.integers("D", [6, 7, 8, 14, 25, 37, 51, 80])
SEEDS = env.seeds(env.integer("SEEDS", 10))
N_BOOT = env.integer("N_BOOT", gradients.N_BOOT)
K_SHARE = env.number("K_SHARE", 0.10)
# A second, deliberately noisy oracle would enter as another value of this column.
ARM = env.text("ARM", "oracle")
OUTPUT = env.text("OUTPUT", "instrument_ceiling.csv")

COLUMNS = ["arm", "d", "seed", "k_share", "n_fit", "n_test", "n_boot", "n_features",
           "slope_shape", "slope_shape_corrected", "r_shape", "var_share_shape",
           "slope_scale", "slope_scale_corrected", "r_scale", "var_share_scale",
           "instrument_slope", "instrument_r", "true_grad_median",
           "seconds", "reason"]
KEY = ["arm", "d", "seed", "k_share", "n_fit", "n_test", "n_boot"]


def measure(cell):
    d, seed = cell["d"], cell["seed"]
    rng = np.random.default_rng(seed)
    # Zero-weight extra features carry neither scale nor shape: the worst case.
    w = np.concatenate([generator.W, np.zeros(d - len(generator.W))])
    p = generator.gpd(N_FIT + N_TEST, rng, xi=generator.XI_OF_X, d=d, w=w)
    Xf, yf = p.X[:N_FIT], p.y[:N_FIT]
    Xte, yte, s_te_true, xi_te = p.X[N_FIT:], p.y[N_FIT:], p.s[N_FIT:], p.xi[N_FIT:]

    s_te = gradients.log_linear_scale(Xf, yf)(Xte)
    oracle = np.column_stack([s_te_true * metrics.gpd_quantile(np.full(len(xi_te), a), xi_te)
                              for a in LEVELS])
    g = gradients.feature_gradients(Xf, Xte, yte, s_te, oracle, rng, cell["n_boot"],
                                    k_share=cell["k_share"], xi_te=xi_te)
    if len(g["ref_shape"]) < 3:
        raise RuntimeError(f"only {len(g['ref_shape'])} usable features")
    a = gradients.slope_through_origin(g["ref_shape"], g["model_shape"], g["var_shape"])
    b = gradients.slope_through_origin(g["ref_scale"], g["model_scale"], g["var_scale"])
    # The regressor here is the true gradient, which has no measurement error.
    i_raw, _, i_r, _ = gradients.slope_through_origin(g["true_shape"], g["ref_shape"])
    return dict(n_features=len(g["ref_shape"]),
                slope_shape=a[0], slope_shape_corrected=a[1], r_shape=a[2], var_share_shape=a[3],
                slope_scale=b[0], slope_scale_corrected=b[1], r_scale=b[2], var_share_scale=b[3],
                instrument_slope=i_raw, instrument_r=i_r,
                true_grad_median=float(np.median(g["true_shape"])))


def main():
    widths = [d for d in D_LIST if d >= len(generator.W)]
    if len(widths) < len(D_LIST):
        print(f"skipping d below {len(generator.W)}: {sorted(set(D_LIST) - set(widths))}")
    cells = runner.grid(arm=[ARM], d=widths, seed=SEEDS, k_share=[K_SHARE],
                        n_fit=[N_FIT], n_test=[N_TEST], n_boot=[N_BOOT])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"shape {r['slope_shape']:+.3f} "
                                        f"(corrected {r['slope_shape_corrected']:+.3f}) | "
                                        f"scale {r['slope_scale']:+.3f} | "
                                        f"instrument {r['instrument_slope']:+.3f}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    ceiling = d.groupby("d")[["n_features", "slope_shape", "slope_shape_corrected",
                              "slope_scale", "slope_scale_corrected",
                              "var_share_shape", "instrument_slope"]].median()
    print("\n=== ceiling: what a perfect model scores through the same measure ===")
    print(ceiling.round(3).to_string())

    measured = paths.result("dissociation_real_slope.csv")
    if not measured.exists():
        return
    m = tables.ok_rows(pd.read_csv(measured))
    m = m[m.n_features.notna()].assign(d=lambda f: f.n_features.astype(int))
    missing = sorted(set(m.d) - set(ceiling.index))
    m = m.join(ceiling.add_prefix("ceiling_"), on="d")
    m["shape_calibrated"] = m.slope_shape_corrected / m.ceiling_slope_shape_corrected
    m["scale_calibrated"] = m.slope_scale_corrected / m.ceiling_slope_scale_corrected
    print("\n=== the real-data slopes divided by the ceiling at their own width ===")
    print(m.groupby("model")[["slope_shape_corrected", "shape_calibrated",
                              "slope_scale_corrected", "scale_calibrated"]]
          .median().round(3).to_string())
    if missing:
        print(f"\nno ceiling yet for d = {missing}; run with D={','.join(map(str, missing))}")


if __name__ == "__main__":
    main()
