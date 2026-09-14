# -*- coding: utf-8 -*-
"""The real-data shape reference: the model's own instrument, and the noise of the selection.

Two gaps in `dissociation_real.py`:

1. The reference xi per tercile comes from Hill on residuals, the model's from inverting
   Q(0.99)/Q(0.9). Here the reference is also computed by inverting the same ratios of the
   empirical quantiles in each tercile, raw and location-invariant.
2. The feature is the maximum of |xi(t3) - xi(t1)| over all columns, and a maximum of
   noisy statistics is large without any dependence. Its null distribution is measured
   by repeating the whole selection on permuted targets.

No model is called.

    ref_range_hill above null_p95        the gradient beats its own selection noise
    ref_range_hill inside the null       the rank claim rests on the selection alone

    DATASETS=diamonds,houses SEEDS=1 python -u experiments/h1_shape_vs_scale/reference_instrument.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, gradients, metrics, paths, quiet, runner, tables

quiet.silence()

N_FIT, N_TEST = gradients.FIT_ROWS, gradients.TEST_ROWS
N_PERM = env.integer("N_PERM", 50)
N_BOOT = env.integer("N_BOOT", 200)
# Hill's share of top observations, on which the whole real-data reference depends.
K_SHARE = env.number("K_SHARE", 0.10)
SEEDS = env.seeds(env.integer("SEEDS", 3))
DATASETS = env.names("DATASETS", datasets.DISSOCIATION)
OUTPUT = env.text("OUTPUT", "reference_instrument.csv")

COLUMNS = ["dataset", "seed", "k_share", "n_perm", "n_boot", "feature", "n_features",
           "ref_range_hill", "ref_range_ratio", "ref_range_ratio_residual",
           "boot_lo", "boot_hi", "null_p50", "null_p95", "above_null",
           "seconds", "reason"]
KEY = ["dataset", "seed", "k_share", "n_perm", "n_boot"]


def ratio_by_tercile(y, terc, residual=False):
    """xi per tercile by the model-side instrument, applied to empirical quantiles."""
    out = []
    for t in range(3):
        z = y[terc == t]
        z = z[np.isfinite(z)]
        if len(z) < gradients.MIN_PER_TERCILE:
            out.append(np.nan)
            continue
        hi, lo, mid = np.quantile(z, [0.99, 0.9, 0.5])
        if residual:
            out.append(metrics.xi_from_residual_ratio((hi - mid) / (lo - mid))
                       if lo - mid > 1e-12 else np.nan)
        else:
            out.append(metrics.xi_from_ratio(hi / lo) if lo > 1e-12 else np.nan)
    return out


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"], positive_only=True)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    rng = np.random.default_rng(cell["seed"])
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    Xf, yf, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
    k = cell["k_share"]

    scale = gradients.log_linear_scale(Xf, yf)
    j, edges, scanned = gradients.steepest_feature(Xf, yf, scale(Xf), k_share=k)
    if j is None:
        raise RuntimeError("no feature has three usable terciles")
    terc = gradients.tercile_of(Xte[:, j], edges)
    s_te = scale(Xte)
    hill = gradients.hill_by_tercile(yte, s_te, terc, k_share=k)
    ratio = ratio_by_tercile(yte, terc)
    residual = ratio_by_tercile(yte, terc, residual=True)

    boot = []
    for _ in range(cell["n_boot"]):
        b = rng.integers(0, len(yte), len(yte))
        hb = gradients.hill_by_tercile(yte[b], s_te[b], terc[b], k_share=k)
        boot.append(hb[2] - hb[0])
    boot = np.array([v for v in boot if np.isfinite(v)])

    # The selection on permuted targets, where the tail depends on no feature.
    null = []
    for _ in range(cell["n_perm"]):
        yp = rng.permutation(yf)
        jp, ep, _ = gradients.steepest_feature(
            Xf, yp, gradients.log_linear_scale(Xf, yp)(Xf), k_share=k)
        if jp is None:
            continue
        hp = gradients.hill_by_tercile(yte, s_te, gradients.tercile_of(Xte[:, jp], ep),
                                       k_share=k)
        if np.isfinite(hp[2]) and np.isfinite(hp[0]):
            null.append(abs(hp[2] - hp[0]))
    null = np.array(null)

    ref = hill[2] - hill[0]
    return dict(
        feature=j, n_features=scanned, ref_range_hill=ref,
        ref_range_ratio=ratio[2] - ratio[0],
        ref_range_ratio_residual=residual[2] - residual[0],
        boot_lo=float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
        boot_hi=float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
        null_p50=float(np.median(null)) if len(null) else np.nan,
        null_p95=float(np.quantile(null, 0.95)) if len(null) else np.nan,
        above_null=bool(abs(ref) > np.quantile(null, 0.95)) if len(null) else None)


def main():
    cells = runner.grid(dataset=DATASETS, seed=SEEDS, k_share=[K_SHARE], n_perm=[N_PERM],
                        n_boot=[N_BOOT])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"hill {r['ref_range_hill']:+.3f} "
                                        f"[{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}] | "
                                        f"ratio {r['ref_range_ratio']:+.3f} | "
                                        f"null p95 {r['null_p95']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== reference gradient by instrument, median per dataset ===")
    print(d.groupby("dataset")[["ref_range_hill", "ref_range_ratio",
                                "ref_range_ratio_residual", "null_p95"]]
          .median().round(3).to_string())
    print("\n=== share of runs whose gradient beats the selection null ===")
    print(d.groupby("dataset").above_null.mean().round(2).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
