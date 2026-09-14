# -*- coding: utf-8 -*-
"""The shape/scale dissociation on real data, through the steepest feature.

On real data xi(x) is unknown, so the reference comes from held-out rows. Per
(dataset, model, seed), 3000 fit rows and 6000 test rows:

  1. On the fit rows only: a log-linear scale, and the feature whose outer terciles differ
     most in Hill index of the residuals.
  2. On the test rows, in terciles with edges from the fit rows:
       shape   reference Hill on residuals; model the implied xi of its quantiles;
               share = model change / reference change
       scale   log ratio of the outer terciles' medians, model over empirical
  3. Control: the same measure on a random feature.

The feature is a maximum over columns, which `reference_instrument.py` shows is mostly
selection noise; `dissociation_real_slope.py` replaces this measure.

    DATASETS=diamonds,houses SEEDS=1 MODELS=TabICLv2 \
      python -u experiments/h1_shape_vs_scale/dissociation_real.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = gradients.LEVELS
N_FIT, N_TEST = gradients.FIT_ROWS, gradients.TEST_ROWS
N_EST = env.integer("N_EST", 4)
DATASETS = env.names("DATASETS", datasets.DISSOCIATION)
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
SEEDS = env.seeds(env.integer("SEEDS", 3))
OUTPUT = env.text("OUTPUT", "dissociation_real.csv")

KEY = ["dataset", "model", "seed", "n_est"]
COLUMNS = ["dataset", "model", "seed", "n_est", "feature", "shape_share", "scale_share",
           "ref_xi_t1", "ref_xi_t3", "mod_xi_t1", "mod_xi_t3",
           "shape_share_control", "ref_range", "seconds", "reason",
           # The location-invariant form, since the raw ratio is biased low under a
           # location component and the Hill reference is not.
           "shape_share_residual", "mod_xi_r_t1", "mod_xi_r_t3"]


def shares(j, edges, Xte, yte, s_te, q):
    """Both shares along feature j, or None when a test tercile is too thin."""
    terc = gradients.tercile_of(Xte[:, j], edges)
    if min((terc == t).sum() for t in range(3)) < gradients.MIN_PER_TERCILE:
        return None
    ref = gradients.hill_by_tercile(yte, s_te, terc)
    raw = [metrics.implied_xi(q[terc == t, 2], q[terc == t, 1]) for t in range(3)]
    res = [metrics.implied_xi(q[terc == t, 2], q[terc == t, 1], q[terc == t, 0])
           for t in range(3)]
    span = ref[2] - ref[0]
    usable = abs(span) > 0.05

    med_ref = [float(np.median(yte[terc == t])) for t in (0, 2)]
    med_mod = [float(np.median(q[terc == t, 0])) for t in (0, 2)]
    scale_share = np.nan
    if min(med_ref + med_mod) > 0:
        d = np.log(med_ref[1] / med_ref[0])
        if abs(d) > 0.05:
            scale_share = np.log(med_mod[1] / med_mod[0]) / d
    return dict(shape_share=(raw[2] - raw[0]) / span if usable else np.nan,
                shape_share_residual=(res[2] - res[0]) / span if usable else np.nan,
                scale_share=scale_share, ref_range=span,
                ref_xi_t1=ref[0], ref_xi_t3=ref[2], mod_xi_t1=raw[0], mod_xi_t3=raw[2],
                mod_xi_r_t1=res[0], mod_xi_r_t3=res[2])


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"])
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    rng = np.random.default_rng(cell["seed"])
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    Xf, yf, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
    if (yf > 0).sum() < N_FIT * 0.5:
        shift = min(yf.min(), yte.min())
        yf, yte = yf - shift + 1e-6, yte - shift + 1e-6

    scale = gradients.log_linear_scale(Xf, yf)
    j, edges, _ = gradients.steepest_feature(Xf, yf, scale(Xf))
    if j is None:
        raise RuntimeError("no feature passed the selection")
    q = models.quantiles(cell["model"], Xf, yf, Xte, seed=cell["seed"], levels=LEVELS,
                         n_est=cell["n_est"])
    r = shares(j, edges, Xte, yte, scale(Xte), q)
    if r is None:
        raise RuntimeError("test terciles too thin")

    jj = int(rng.integers(X.shape[1]))
    control_edges = gradients.tercile_edges(Xf[:, jj])
    control = (shares(jj, control_edges, Xte, yte, scale(Xte), q)
               if len(np.unique(control_edges)) == 4 else None)
    return dict(r, feature=j,
                shape_share_control=control["shape_share"] if control else np.nan)


def main():
    cells = runner.grid(dataset=DATASETS, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"shape {r['shape_share']:7.0%}  "
                                        f"scale {r['scale_share']:7.0%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(f"\n=== shares on real data, {len(d)} runs ===")
    print(d.pivot_table(index="dataset", columns="model", values=["shape_share", "scale_share"],
                        aggfunc="mean").round(2).to_string())
    print(d.groupby("model")[["scale_share", "shape_share", "shape_share_control"]]
          .median().round(2).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
