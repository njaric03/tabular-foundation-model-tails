# -*- coding: utf-8 -*-
"""Can the conditional mean be estimated from 2000 rows at all?

At xi >= 0.5 the variance is infinite and the sample mean is itself a poor estimator. If
the training mean returns 55% of the true mean at xi = 0.9, a model returning 57% sits
at the limit of the data rather than below it. Two references, as ratios to the true mean:

    r_sample   the plain mean of the training target
    r_oracle   a well-specified EVT estimate: log-linear scale, GPD MLE on the top 10% of
               the residuals, the mean as body plus GPD tail

    python -u experiments/h1_shape_vs_scale/mean_attainability.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from common import append, env, generator, quiet

quiet.silence()

XI = [0.0, 0.3, 0.5, 0.7, 0.9]
N_TRAIN = 2000
N_REP_SAMPLE = 400        # the sample mean is cheap
N_REP_ORACLE = 120        # the GPD fit is not
OUTPUT = env.text("OUTPUT", "mean_attainability.csv")


def oracle_mean(X, y, frac=0.10):
    """EVT estimate of the mean conditional mean; NaN when the fitted shape reaches 1."""
    D = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(D, np.log(y), rcond=None)
    scale_hat = np.exp(D @ b)
    z = y / scale_hat
    u = np.quantile(z, 1 - frac)
    shape, _, sigma = genpareto.fit(z[z > u] - u, floc=0)
    if shape >= 0.999:
        return np.nan
    m_z = (z[z <= u].sum() + (z > u).sum() * (u + sigma / (1 - shape))) / len(z)
    return float((m_z * scale_hat).mean())


def main():
    t0, rows = time.time(), []
    for xi in XI:
        sample, oracle = [], []
        for k in range(N_REP_SAMPLE):
            p = generator.gpd(N_TRAIN, np.random.default_rng(10000 + k), xi=xi)
            true = float((p.s / (1 - xi)).mean())
            sample.append(p.y.mean() / true)
            if k < N_REP_ORACLE:
                m = oracle_mean(p.X, p.y)
                if np.isfinite(m):
                    oracle.append(m / true)
        sample = np.array(sample)
        rows.append(dict(xi=xi,
                         r_sample_median=float(np.median(sample)),
                         r_sample_q10=float(np.quantile(sample, 0.10)),
                         r_sample_q90=float(np.quantile(sample, 0.90)),
                         share_below_060=float(np.mean(sample < 0.60)),
                         r_oracle_median=float(np.median(oracle)) if oracle else np.nan,
                         n_rep=N_REP_SAMPLE))
        print(f"  xi={xi}  r_sample {rows[-1]['r_sample_median']:.3f}  "
              f"r_oracle {rows[-1]['r_oracle_median']:.3f}  [{time.time() - t0:.0f}s]",
              flush=True)
    df = pd.DataFrame(rows)
    append.replace(OUTPUT, df)
    print("\n=== the mean attainable from 2000 rows, as a ratio to the true mean ===")
    print(df.round(3).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
