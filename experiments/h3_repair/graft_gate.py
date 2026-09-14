# -*- coding: utf-8 -*-
"""When to apply the EVT graft: a fair oracle, a validation gate, and candidate signals.

One pass answers three questions with the same fits:

  1. How large is a fair oracle? Picking the better option on the test set overfits, so
     the test set is split in halves: decide on one, measure on the other.
  2. Does a validation gate work? The graft is applied when it lowers pinball at 0.999 on
     held-out validation rows the model has not seen; no threshold on xi is involved.
  3. What predicts a helpful graft? Features computed from the fit rows only, most
     importantly on the pair Buzz (xi ~ 0.63, the graft worsens by 44%) against
     freMTPL2sev (xi ~ 0.62, it improves by 30%).

Splits: fit (model and tail fit), val (the gate), test (evaluation, in two halves). N_EST=1
and LOG_SCALE=1 remove the aggregation effect and fit on log(y); the defaults reproduce
the published run.

    N_EST=1 LOG_SCALE=1 OUTPUT=graft_gate_log.csv python -u experiments/h3_repair/graft_gate.py
"""
import time

import numpy as np
from scipy.stats import genpareto
from sklearn.model_selection import train_test_split

from common import datasets, env, evt, generator, metrics, models, quiet, runner

quiet.silence()

XI_SYNTHETIC = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_FIT, N_VAL, N_TEST = 2000, 800, 1000
MODELS = ["TabICLv2", "TabPFN-V3"]
N_EST = env.integer("N_EST", 4)
LOG_SCALE = env.flag("LOG_SCALE")
THRESHOLDS = (0.05, 0.10, 0.20)
OUTPUT = env.text("OUTPUT", "graft_gate.csv")

COLUMNS = ["dataset", "model", "seed", "n_est", "log_scale", "xi_evt",
           "xi_ref", "xi_ref_range", "xi_hill", "xi_quantile", "agreement_range",
           "n_exceedances", "ad_gpd", "tied_share", "unique_share", "xi_model",
           "median_unknown_share",
           "val_raw_pinball999", "val_grafted_pinball999", "val_raw_twcrps",
           "val_grafted_twcrps", "val_says_graft",
           "test_raw_pinball999", "test_grafted_pinball999", "test_raw_pinball99",
           "test_grafted_pinball99", "test_raw_twcrps", "test_grafted_twcrps",
           "test_raw_gamma_dev", "test_grafted_gamma_dev", "test_raw_cov999",
           "test_grafted_cov999", "A_raw", "A_grafted", "B_raw", "B_grafted", "seconds",
           "reason"]
KEY = ["dataset", "model", "seed", "n_est", "log_scale"]


def anderson_darling(ex, xi, sigma):
    """Anderson-Darling statistic of exceedances against a fitted GPD. The parameters come
    from the same data, so it is a feature for the gate, not a formal test."""
    u = np.clip(np.sort(genpareto.cdf(ex, xi, loc=0, scale=sigma)), 1e-12, 1 - 1e-12)
    i = np.arange(1, len(u) + 1)
    return float(-len(u) - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))))


def features(Xfit, yfit, q_fit):
    """Everything computable when the decision is made, so from the fit rows only."""
    o = {}
    D = np.c_[np.ones(len(yfit)), Xfit]
    b, *_ = np.linalg.lstsq(D, np.log(yfit), rcond=None)
    z = yfit / np.exp(D @ b)
    xis = [metrics.gpd_mle(z, p) for p in THRESHOLDS]
    xis = [x for x in xis if np.isfinite(x)]
    o["xi_ref"] = float(np.median(xis)) if xis else np.nan
    o["xi_ref_range"] = float(np.max(xis) - np.min(xis)) if len(xis) > 1 else np.nan

    o["xi_hill"] = metrics.hill(z)
    q50, q90, q99 = np.quantile(z, [0.5, 0.9, 0.99])
    o["xi_quantile"] = (metrics.xi_from_residual_ratio((q99 - q50) / (q90 - q50))
                        if (q90 - q50) > 1e-12 else np.nan)
    three = np.array([o["xi_ref"], o["xi_hill"], o["xi_quantile"]], dtype=float)
    o["agreement_range"] = float(np.nanmax(three) - np.nanmin(three))

    u = np.quantile(z, 0.90)
    ex = z[z > u] - u
    o["n_exceedances"] = int(len(ex))
    if len(ex) >= 50:
        xi_h, _, sg_h = genpareto.fit(ex, floc=0)
        o["ad_gpd"] = anderson_darling(ex, xi_h, sg_h)
    else:
        o["ad_gpd"] = np.nan

    # Discreteness of the target, the hypothesis for Buzz.
    _, counts = np.unique(yfit, return_counts=True)
    o["tied_share"] = float(counts.max() / len(yfit))
    o["unique_share"] = float(len(counts) / len(yfit))

    o["xi_model"] = metrics.implied_xi(q_fit[:, evt.I99], q_fit[:, evt.I90], q_fit[:, evt.I50])
    o["median_unknown_share"] = float(np.mean(q_fit[:, evt.I50] <= 0))
    return o


def measure(cell):
    (Xf, yf), (Xv, yv), (Xt, yt) = cell["_parts"]
    y_fit = np.log(np.maximum(yf, 1e-12)) if LOG_SCALE else yf
    q = models.quantiles(cell["model"], Xf, y_fit, np.vstack([Xf, Xv, Xt]), seed=cell["seed"],
                         levels=evt.LEVELS, n_est=cell["n_est"])
    if LOG_SCALE:
        # Quantiles are equivariant under a monotone map, so exp brings them back exactly.
        q = np.exp(q)
    q_f, q_v, q_t = np.split(q, [len(Xf), len(Xf) + len(Xv)])

    xi_e, sg_e, _ = evt.fit_tail_regularised(yf, q_f[:, evt.I50])
    grafted_v = evt.splice(q_v, q_v[:, evt.I50], xi_e, sg_e)
    grafted_t = evt.splice(q_t, q_t[:, evt.I50], xi_e, sg_e)
    r = dict(xi_evt=xi_e, **features(Xf, yf, q_f))

    raw_v = evt.scores(q_v, yv, float(np.quantile(yv, 0.90)))
    graft_v = evt.scores(grafted_v, yv, float(np.quantile(yv, 0.90)))
    for m in ("pinball999", "twcrps"):
        r[f"val_raw_{m}"], r[f"val_grafted_{m}"] = raw_v[m], graft_v[m]
    r["val_says_graft"] = bool(graft_v["pinball999"] < raw_v["pinball999"])

    raw_t = evt.scores(q_t, yt, float(np.quantile(yt, 0.90)))
    graft_t = evt.scores(grafted_t, yt, float(np.quantile(yt, 0.90)))
    for m in ("pinball999", "pinball99", "twcrps", "gamma_dev", "cov999"):
        r[f"test_raw_{m}"], r[f"test_grafted_{m}"] = raw_t[m], graft_t[m]

    half = len(yt) // 2
    for label, rows in (("A", slice(0, half)), ("B", slice(half, None))):
        threshold = float(np.quantile(yt[rows], 0.90))
        r[f"{label}_raw"] = evt.scores(q_t[rows], yt[rows], threshold)["pinball999"]
        r[f"{label}_grafted"] = evt.scores(grafted_t[rows], yt[rows], threshold)["pinball999"]
    return r


def cells():
    def cell(label, model, seed, parts):
        return dict(dataset=label, model=model, seed=seed, n_est=N_EST,
                    log_scale=int(LOG_SCALE), _parts=parts)

    for xi in XI_SYNTHETIC:
        for seed in SEEDS:
            rng = np.random.default_rng(5000 + seed * 100 + int(xi * 10))
            parts = tuple((p.X, p.y) for p in (generator.gpd(n, rng, xi=xi)
                                               for n in (N_FIT, N_VAL, N_TEST)))
            for model in MODELS:
                yield cell(f"sint xi={xi}", model, seed, parts)

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
            for model in MODELS:
                yield cell(name, model, seed, ((Xf, yf), (Xv, yv), (Xt, yt)))


def main():
    t0 = time.time()
    def show(r):
        change = 100 * (r["test_grafted_pinball999"] / r["test_raw_pinball999"] - 1)
        return (f"test {change:+6.1f}%  val says {'graft' if r['val_says_graft'] else 'keep'}"
                f"  xi_ref {r['xi_ref']:.2f}  AD {r['ad_gpd']:.1f}")

    runner.run(OUTPUT, COLUMNS, KEY, cells(), measure, show=show)
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
