# -*- coding: utf-8 -*-
"""Post-hoc correction of the conditional mean, across the model family.

The deficit in the mean is large and stable where the 0.999 quantile is not: TabPFN-V3
returns 0.304 of the true conditional mean at xi = 0.9 (sd 0.012 over five seeds)
against an attainable 0.549, and the gamma deviance reproduces across runs with r = 0.91.
The correction needs only a point prediction, so it applies to mean-only models too.

    y_corrected = c * y_predicted

`c` is chosen on a held-out validation split. Minimising the gamma deviance gives
c = mean(y / yhat) in closed form; `c_ratio` = mean(y) / mean(yhat) weights by size and is
robust to tiny predictions. An isotonic map yhat -> y is measured as the richer variant.
The mean is the model's own (`models.predictive_mean`), not the median.

    MODEL=TabICLv2 python -u experiments/h3_repair/mean_correction.py
    MODEL=TabFM venv-tabfm/Scripts/python -u experiments/h3_repair/mean_correction.py
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from common import datasets, env, generator, metrics, models, quiet, runner

quiet.silence()

MODEL = env.models("TabICLv2", name="MODEL")[0]
# TabFM takes about two and a half minutes per fit on CPU, so it runs a smaller profile
# with one member, as it was measured before.
TABFM = MODEL == "TabFM"
N_EST = env.integer("N_EST", 1 if TABFM else 4)
N_FIT, N_VAL, N_TEST = (1000, 500, 600) if TABFM else (2000, 800, 1500)
XI_SYNTHETIC = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
OUTPUT = env.text("OUTPUT", "mean_correction.csv")

VARIANTS = ["raw", "constant", "mean_ratio", "isotonic"]
COLUMNS = (["dataset", "model", "seed", "n_est", "c_val", "c_test", "c_ratio", "unknown_share"]
           + [f"{p}_{v}" for p in ("gdev", "rmse", "ratio") for v in VARIANTS]
           + ["seconds", "reason"])
KEY = ["dataset", "model", "seed", "n_est"]


def floor(mu, yf):
    """A positive floor tied to the target's scale. A floor at 1e-9 once let one tiny
    prediction make y / yhat about 1e9 and set `c` on its own."""
    return np.maximum(np.asarray(mu, dtype=float), 1e-3 * float(np.median(yf)))


def measure(cell):
    (Xf, yf), (Xv, yv), (Xt, yt) = cell["_parts"]
    true_mean = cell["_true_mean"]
    # One fit for both the validation and the test rows.
    both = models.predictive_mean(MODEL, Xf, yf, np.vstack([Xv, Xt]), seed=cell["seed"],
                                  n_est=cell["n_est"])
    raw_v, raw_t = both[:len(Xv)], both[len(Xv):]
    mv, mt = floor(raw_v, yf), floor(raw_t, yf)

    sel = raw_v > 0
    if sel.sum() < 50:
        sel = np.ones(len(mv), dtype=bool)
    c = float(np.mean(yv[sel] / mv[sel]))
    c_ratio = float(np.mean(yv[sel]) / np.mean(mv[sel]))
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(mv[sel], yv[sel])
    variants = {"raw": mt, "constant": c * mt, "mean_ratio": c_ratio * mt,
                "isotonic": np.maximum(iso.predict(mt), 1e-9)}

    r = dict(c_val=c, c_test=float(np.mean(yt / mt)), c_ratio=c_ratio,
             unknown_share=float(np.mean(raw_t <= 0)))
    for name, mu in variants.items():
        r[f"gdev_{name}"] = metrics.gamma_deviance(mu, yt)
        r[f"rmse_{name}"] = float(np.sqrt(np.mean((mu - yt) ** 2)))
        if true_mean is not None and np.all(np.isfinite(true_mean)):
            r[f"ratio_{name}"] = float(mu.mean() / true_mean.mean())
    return r


def cells():
    for xi in XI_SYNTHETIC:
        for seed in SEEDS:
            rng = np.random.default_rng(3000 + seed * 100 + int(xi * 10))
            fit, val, test = (generator.gpd(n, rng, xi=xi) for n in (N_FIT, N_VAL, N_TEST))
            yield dict(dataset=f"sint xi={xi}", model=MODEL, seed=seed, n_est=N_EST,
                       _parts=((fit.X, fit.y), (val.X, val.y), (test.X, test.y)),
                       _true_mean=generator.true_mean(test))
    for name in datasets.selected_names():
        try:
            X, y = datasets.load(name, positive_only=True)
        except Exception as e:
            print(f"  {name}: skipped, {runner.reason(e)}", flush=True)
            continue
        for seed in SEEDS:
            Xr, Xt, yr, yt = train_test_split(X, y, test_size=min(N_TEST, int(0.25 * len(y))),
                                              random_state=seed)
            Xf, Xv, yf, yv = train_test_split(Xr, yr, train_size=min(N_FIT, int(0.6 * len(yr))),
                                              test_size=min(N_VAL, int(0.25 * len(yr))),
                                              random_state=seed)
            yield dict(dataset=name, model=MODEL, seed=seed, n_est=N_EST,
                       _parts=((Xf, yf), (Xv, yv), (Xt, yt)), _true_mean=None)


def main():
    print(f"MODEL={MODEL} N_EST={N_EST}", flush=True)
    seconds = runner.run(
        OUTPUT, COLUMNS, KEY, cells(), measure,
        show=lambda r: f"c {r['c_val']:5.2f} (ideal {r['c_test']:5.2f})  gdev: constant "
                       f"{100 * (r['gdev_constant'] / r['gdev_raw'] - 1):+6.1f}%  mean ratio "
                       f"{100 * (r['gdev_mean_ratio'] / r['gdev_raw'] - 1):+6.1f}%  isotonic "
                       f"{100 * (r['gdev_isotonic'] / r['gdev_raw'] - 1):+6.1f}%")
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
