# -*- coding: utf-8 -*-
"""Empirical coverage of the predicted quantiles: the number a user of an interval gets.

Every other measure is indirect: an implied tail index, pinball, twCRPS. Coverage says
whether a bound holds its nominal level. Per dataset, model and seed: fit on N_FIT rows,
predict on N_TEST, count the test outcomes above each predicted quantile.

    dose = 0    the clean context
    dose > 0    plus one row with y = dose * max(y_fit), the treatment of influence.py

`sd_shift` records the measured size of the treatment. `n_above*` travels with every
coverage because at 6000 test rows the 0.999 level expects six exceedances, where four
against eight is noise. The dataset `sint` is the generator with XI, where the true
coverage is known.

    MODELS=TabICLv2 DATASETS=freMTPL2sev DOSES=0,100 \
      python -u experiments/h1_shape_vs_scale/empirical_coverage.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, generator, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
N_FIT, N_TEST = 3000, 6000
N_EST = env.integer("N_EST", 1)
SEEDS = env.seeds(env.integer("SEEDS", 3))
DOSES = env.floats("DOSES", [0.0, 100.0])
MODELS = env.models("TabICLv2,TabPFN-V3,GBM,XGB")
XI_SYNTHETIC = env.number("XI", 0.7)
# freMTPL2sev, the one table with both leverage and a heavy tail, plus the dissociation
# tables except Allstate_Claims_Severity.
DEFAULT_DATASETS = ["freMTPL2sev"] + [d for d in datasets.DISSOCIATION
                                      if d != "Allstate_Claims_Severity"]
DATASETS = [f"sint xi={XI_SYNTHETIC:g}" if name == "sint" else name
            for name in env.names("DATASETS", DEFAULT_DATASETS)]
OUTPUT = env.text("OUTPUT", "coverage.csv")

COLUMNS = (["dataset", "model", "seed", "dose", "n_est", "n_fit", "n_test", "sd_shift"]
           + [f"cov{t}" for t in TAGS] + [f"n_above{t}" for t in TAGS]
           + [f"pb{t}" for t in TAGS] + ["seconds", "reason"])
KEY = ["dataset", "model", "seed", "dose", "n_est"]


def split(name, seed):
    """Fit and test rows of a dataset, or of the generator for `sint xi=...`."""
    if name.startswith("sint"):
        rng = np.random.default_rng(seed)
        fit = generator.gpd(N_FIT, rng, xi=XI_SYNTHETIC, clip=True)
        test = generator.gpd(N_TEST, rng, xi=XI_SYNTHETIC, clip=True)
        return fit.X, fit.y, test.X, test.y
    X, y = datasets.load_cached(name, positive_only=True)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    idx = np.random.default_rng(seed).permutation(len(y))[: N_FIT + N_TEST]
    return X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]


def measure(cell):
    seed, dose = cell["seed"], cell["dose"]
    Xf, yf, Xte, yte = split(cell["dataset"], seed)
    shift = 1.0
    if dose > 0:
        j = int(np.random.default_rng(seed).integers(len(yf)))
        y0 = dose * float(yf.max())
        shift = metrics.sd_shift(np.append(yf, y0))
        Xf, yf = np.vstack([Xf, Xf[[j]]]), np.concatenate([yf, [y0]])
    q = models.quantiles(cell["model"], Xf, yf, Xte, seed=seed, levels=LEVELS,
                         n_est=cell["n_est"])
    r = dict(sd_shift=shift)
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        above = int((yte > q[:, i]).sum())
        r[f"cov{t}"] = 1.0 - above / len(yte)
        r[f"n_above{t}"] = above
        r[f"pb{t}"] = metrics.pinball_at(q, yte, a, LEVELS)
    return r


def main():
    cells = runner.grid(dataset=DATASETS, model=MODELS, dose=DOSES, seed=SEEDS,
                        n_est=[N_EST], n_fit=[N_FIT], n_test=[N_TEST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"cov 0.9 {r['cov90']:.3f} | 0.99 {r['cov99']:.4f} "
                                        f"({r['n_above99']} above) | 0.999 {r['cov999']:.4f} "
                                        f"({r['n_above999']} above)")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for t in TAGS:
        print(f"\n=== coverage at 0.{t}, clean context, median ===")
        print(d[d.dose == 0].pivot_table(index="dataset", columns="model", values=f"cov{t}",
                                         aggfunc="median").round(4).to_string())
    if (d.dose > 0).any():
        print("\n=== coverage at 0.99 by dose ===")
        print(d.pivot_table(index=["dataset", "model"], columns="dose", values="cov99",
                            aggfunc="median").round(4).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
