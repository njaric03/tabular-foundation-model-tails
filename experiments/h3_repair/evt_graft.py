# -*- coding: utf-8 -*-
"""Post-hoc EVT graft against the raw model, on the generator and on selected tables.

The tables are those `experiments/datasets/dataset_selection.py` passes, where the tail
index is identified, from heavy (freMTPL2sev) to light (Allstate): the graft has to help
where the tail is heavy and not hurt where it is not. On the generator the true xi is
known. twCRPS is the metric that weights the tail. The graft is in `common/evt.py`.

    python -u experiments/h3_repair/evt_graft.py
    PARTS=sint python -u experiments/h3_repair/evt_graft.py
"""
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.model_selection import train_test_split

from common import append, datasets, env, evt, generator, models, paths, quiet, runner

quiet.silence()

XI_SYNTHETIC = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_TRAIN_SYNTHETIC, N_TEST_SYNTHETIC = 2000, 800
N_TRAIN_REAL, N_TEST_REAL = 3000, 1000
N_EST = 4
MODELS = ["TabICLv2", "TabPFN-V3"]
PARTS = env.text("PARTS", "all")          # all | sint | real
OUTPUT = env.text("OUTPUT", "evt_graft.csv")

VARIANTS = ("raw", "EVT simple", "EVT regularised")
COLUMNS = ["crps", "twcrps", "pinball99", "pinball999", "cov99", "cov999", "gamma_dev",
           "xi_implied", "mean", "dataset", "model", "seed", "variant", "xi_true",
           "xi_evt_simple", "xi_evt_reg"]
KEY = ["dataset", "model", "seed", "variant"]


def done_units():
    """(dataset, model, seed) units that carry all three variants. A partial unit, left by
    an older version of this script, is measured again."""
    counts = Counter(k[:3] for k in append.done(OUTPUT, KEY) if k[3] in VARIANTS)
    return {unit for unit, n in counts.items() if n == len(VARIANTS)}


def reference_xi(X, y):
    """GPD shape of the residuals of a log-linear scale, over their 90th percentile."""
    D = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(D, np.log(y), rcond=None)
    z = y / np.exp(D @ b)
    u = np.quantile(z, 0.90)
    return float(genpareto.fit(z[z > u] - u, floc=0)[0])


def measure(label, Xtr, ytr, Xte, yte, model, seed, xi_true):
    q = models.quantiles(model, Xtr, ytr, np.vstack([Xtr, Xte]), seed=seed,
                         levels=evt.LEVELS, n_est=N_EST)
    q_tr, q_te = q[:len(Xtr)], q[len(Xtr):]
    med_tr, med_te = q_tr[:, evt.I50], q_te[:, evt.I50]
    xi_s, sg_s, _ = evt.fit_tail(ytr, med_tr)
    xi_r, sg_r, _ = evt.fit_tail_regularised(ytr, med_tr)
    variants = {"raw": q_te,
                "EVT simple": evt.splice(q_te, med_te, xi_s, sg_s),
                "EVT regularised": evt.splice(q_te, med_te, xi_r, sg_r)}
    threshold = float(np.quantile(yte, 0.90))
    rows = [dict(evt.scores(qv, yte, threshold), dataset=label, model=model, seed=seed,
                 variant=v, xi_true=xi_true, xi_evt_simple=xi_s, xi_evt_reg=xi_r)
            for v, qv in variants.items()]
    for row in rows:                       # all three or none, so a unit is never partial
        append.write(OUTPUT, row, COLUMNS)
    raw = rows[0]["twcrps"]
    change = [100 * (r["twcrps"] - raw) / raw if raw else np.nan for r in rows[1:]]
    print(f"  {label:24s} {model:10s} s={seed}  twCRPS {raw:.4g} | simple {change[0]:+.1f}% "
          f"(xi {xi_s:.2f}) | regularised {change[1]:+.1f}% (xi {xi_r:.2f})", flush=True)


def main():
    t0 = time.time()
    done = done_units()
    if done:
        print(f"resuming, {len(done)} units already measured", flush=True)

    if PARTS in ("all", "sint"):
        print("=== generator, true xi known ===", flush=True)
        for xi in XI_SYNTHETIC:
            for seed in SEEDS:
                rng = np.random.default_rng(7000 + seed * 1000 + int(xi * 10))
                train = generator.gpd(N_TRAIN_SYNTHETIC, rng, xi=xi)
                test = generator.gpd(N_TEST_SYNTHETIC, rng, xi=xi)
                label = f"sint xi={xi}"
                for model in MODELS:
                    if (label, model, seed) not in done:
                        measure(label, train.X, train.y, test.X, test.y, model, seed, xi)

    if PARTS in ("all", "real"):
        print("\n=== selected real tables ===", flush=True)
        for name in datasets.selected_names():
            try:
                X, y = datasets.load(name, positive_only=True)
            except Exception as e:
                print(f"  {name}: skipped, {runner.reason(e)}", flush=True)
                continue
            for seed in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, train_size=min(N_TRAIN_REAL, int(0.6 * len(y))),
                    test_size=min(N_TEST_REAL, int(0.3 * len(y))), random_state=seed)
                ref = reference_xi(Xtr, ytr)
                for model in MODELS:
                    if (name, model, seed) in done:
                        continue
                    try:
                        measure(name, Xtr, ytr, Xte, yte, model, seed, ref)
                    except Exception as e:
                        print(f"  {name} {model} s={seed}: failed, {runner.reason(e)}",
                              flush=True)

    d = pd.read_csv(paths.result(OUTPUT))
    for col in ("twcrps", "pinball999"):
        print(f"\n=== {col}, raw against grafted (lower is better) ===")
        p = d.pivot_table(index=["dataset", "model"], columns="variant", values=col)
        for v in VARIANTS[1:]:
            if v in p.columns:
                p[v + " %"] = 100 * (p[v] - p["raw"]) / p["raw"]
        print(p.round(3).to_string())
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
