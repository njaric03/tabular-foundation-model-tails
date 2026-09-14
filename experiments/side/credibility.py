# -*- coding: utf-8 -*-
"""How much do tabular foundation models trust the context? Against Buhlmann's optimum.

Credibility theory (Buhlmann 1967) weighs group experience against the collective. For

    theta_i ~ (mu, tau^2),   y_ij | theta_i ~ (theta_i, sigma^2)

the best linear predictor of group i is Z * mean_i + (1 - Z) * mu with Z = n / (n + k),
k = sigma^2 / tau^2. A PFN mixes context and prior too, with an implicit Z, measured here
as the slope through the origin of (prediction_i - mu_hat) on (mean_i - mu_hat):

    Z_hat ~ Z_opt   the classical optimal shrinkage
    Z_hat < Z_opt   under-credits the context, trusts the prior too much
    Z_hat > Z_opt   over-credits the context

Shrinking too much towards the prior is the same statement as compressing the tail, in
the language actuaries use.

    N_PER_GROUP=5,20 K=4 N_REPEATS=5 MODELS=TabICLv2 python -u experiments/side/credibility.py
"""
import numpy as np
import pandas as pd

from common import env, models, paths, quiet, runner, tables

quiet.silence()

N_PER_GROUP = env.integers("N_PER_GROUP", [2, 5, 10, 20, 50])
K = env.floats("K", [1, 4, 25])
N_REPEATS = env.integer("N_REPEATS", 20)
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
N_GROUPS = env.integer("N_GROUPS", 40)
MU, TAU2 = 100.0, 25.0
OUTPUT = env.text("OUTPUT", "credibility.csv")

COLUMNS = ["n_groups", "n_per_group", "k", "model", "repeat", "Z_opt", "Z_hat",
           "difference", "r2", "seconds", "reason"]
KEY = ["n_per_group", "k", "model", "repeat", "n_groups"]


def measure(cell):
    n, k, rep, groups = cell["n_per_group"], cell["k"], cell["repeat"], cell["n_groups"]
    rng = np.random.default_rng(10_000 * rep + 100 * n + int(k))
    theta = rng.normal(MU, np.sqrt(TAU2), groups)
    gid = np.repeat(np.arange(groups), n)
    y = rng.normal(theta[gid], np.sqrt(TAU2 * k))
    # The group id is the only informative feature; the second column is noise, since
    # some wrappers refuse a single column.
    X = np.c_[gid.astype(float), rng.normal(size=len(gid))]
    Xq = np.c_[np.arange(groups, dtype=float), np.zeros(groups)]

    pred = models.mean(cell["model"], X, y, Xq, seed=rep)
    mu_hat = float(y.mean())
    a = np.array([y[gid == i].mean() for i in range(groups)]) - mu_hat
    b = pred - mu_hat
    den = float(np.sum(a * a))
    if den < 1e-12:
        raise RuntimeError("no variation between groups")
    z_hat = float(np.sum(a * b) / den)
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    z_opt = n / (n + k)
    return dict(Z_opt=z_opt, Z_hat=z_hat, difference=z_hat - z_opt,
                r2=1.0 - float(np.sum((b - z_hat * a) ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan)


def main():
    cells = runner.grid(n_per_group=N_PER_GROUP, k=K, model=MODELS,
                        repeat=list(range(N_REPEATS)), n_groups=[N_GROUPS])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"Z_opt {r['Z_opt']:.3f}  Z_hat {r['Z_hat']:.3f}  "
                                        f"r2 {r['r2']:.2f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== implicit credibility: median Z_hat, and Z_opt ===")
    table = d.pivot_table(index=["k", "n_per_group"], columns="model", values="Z_hat",
                          aggfunc="median")
    table["Z_opt"] = [n / (n + k) for k, n in table.index]
    print(table.round(3).to_string())
    print("\n=== how well the linear credibility form describes the predictions: R^2 ===")
    print(d.pivot_table(index="model", columns="k", values="r2", aggfunc="median")
          .round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
