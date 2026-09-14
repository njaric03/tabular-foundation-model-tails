# -*- coding: utf-8 -*-
"""Can xi(x) be recovered from 2000 rows at all? The control that H1 rests on.

If an estimator that knows more than any model cannot recover the shape gradient at the
models' sample size, their deficit says nothing about them. Over a grid of n, per
tercile of x4:

    oracle_s    GPD MLE on y / s(x), with the true scale: an upper bound
    oracle_r    GPD MLE on y / s_hat(x), with a fitted log-linear scale
    empirical   the quantile-ratio inversion a model is scored with, on the raw y

    python -u experiments/h1_shape_vs_scale/oracle_xi_of_x.py
"""
import time

import numpy as np
import pandas as pd

from common import append, env, generator, gradients, metrics, quiet

quiet.silence()

N_GRID = [500, 1000, 2000, 5000, 20000, 100000]
# The seed of repetition r depends on (r, n) only, so more repetitions keep the first
# ones identical.
N_REPEATS = env.integer("N_REPEATS", 20)
OUTPUT = env.text("OUTPUT", "oracle_xi_of_x.csv")
ESTIMATORS = ("true", "oracle_s", "oracle_r", "empirical")


def repeat(n, rng):
    p = generator.gpd(n, rng, xi=generator.XI_OF_X)
    tercile, _ = gradients.generator_terciles(p.X)
    s_hat = gradients.log_linear_scale(p.X, p.y)(p.X)
    out = {}
    for t in range(3):
        sel = tercile == t
        out[f"true_{t}"] = float(p.xi[sel].mean())
        out[f"oracle_s_{t}"] = metrics.gpd_mle(p.y[sel] / p.s[sel])
        out[f"oracle_r_{t}"] = metrics.gpd_mle(p.y[sel] / s_hat[sel])
        q_lo, q_hi = np.quantile(p.y[sel], [0.9, 0.99])
        out[f"empirical_{t}"] = metrics.xi_from_ratio(q_hi / q_lo) if q_lo > 1e-12 else np.nan
    return out


def main():
    t0 = time.time()
    rows = []
    for n in N_GRID:
        for r in range(N_REPEATS):
            # The seed depends on n too; one that ignored n nested the draws across the grid.
            rows.append(dict(repeat(n, np.random.default_rng(10_000 + 97 * r + 7 * n)),
                             n=n, repeat=r))
        print(f"  n={n:>7}  [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    append.replace(OUTPUT, df)

    slopes = pd.DataFrame({e: df[f"{e}_2"] - df[f"{e}_0"] for e in ESTIMATORS}).assign(n=df.n)
    g = slopes.groupby("n")
    print("\n=== slope, xi in the top tercile minus the bottom: mean (sd over repeats) ===")
    table = g.mean().round(3).astype(str) + " (" + g.std(ddof=1).round(2).astype(str) + ")"
    table["oracle_s above 0.25"] = g.oracle_s.apply(lambda v: f"{(v > 0.25).mean():.0%}")
    print(table.to_string())
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
