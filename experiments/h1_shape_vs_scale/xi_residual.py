# -*- coding: utf-8 -*-
"""A location-invariant tail index, and whether it changes the H1 conclusion.

The implied xi from Q(0.99)/Q(0.9) assumes a pure scale family. A location component
pulls the ratio towards 1 and the xi low. The ratio of differences from the median,

    (Q(0.99) - Q(0.5)) / (Q(0.9) - Q(0.5)),

is invariant to location by construction.

Part 1, no model: both estimators on GPD samples with a known xi and an added location.
Part 2: the tracked share of the xi(x) gradient with both estimators. Prediction written
before the run: the share moves by less than 10 points and the conclusion (scale about
94%, shape well below) stands.

    PARTS=1 python -u experiments/h1_shape_vs_scale/xi_residual.py
"""
import time

import numpy as np
import pandas as pd

from common import append, env, generator, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = gradients.LEVELS
N_EST = env.integer("N_EST", 4)
PARTS = env.text("PARTS", "12")
N_SEEDS = env.integer("SEEDS", 10)
N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
MODELS = env.models("GBM,TabICLv2,TabPFN-V3")
OUTPUT1 = env.text("OUTPUT1", "xi_residual_part1.csv")
OUTPUT2 = env.text("OUTPUT2", "xi_residual_part2.csv")

COLUMNS2 = ["model", "seed", "n_est", "n_fit", "n_test", "d_true", "seconds", "reason",
            "xi_raw_t0", "xi_residual_t0", "xi_raw_t1", "xi_residual_t1",
            "xi_raw_t2", "xi_residual_t2", "xi_true_t0", "xi_true_t2"]
KEY2 = ["model", "seed", "n_est", "n_fit", "n_test"]


def part1():
    rows = []
    for xi in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for mu_ratio in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for seed in range(N_SEEDS):
                u = np.random.default_rng(9000 + seed).random(200000)
                y0 = ((1 - u) ** (-xi) - 1) / xi
                hi, lo, mid = np.quantile(y0 + mu_ratio * np.median(y0), [0.99, 0.9, 0.5])
                rows.append(dict(
                    xi=xi, mu_ratio=mu_ratio, seed=seed,
                    xi_raw=metrics.xi_from_ratio(hi / lo) if lo > 0 else np.nan,
                    xi_residual=(metrics.xi_from_residual_ratio((hi - mid) / (lo - mid))
                                 if lo - mid > 1e-12 else np.nan)))
    d = pd.DataFrame(rows)
    d["error_raw"] = d.xi_raw - d.xi
    d["error_residual"] = d.xi_residual - d.xi
    append.replace(OUTPUT1, d)
    for col in ("error_raw", "error_residual"):
        print(f"\n=== part 1: {col} by xi and location ===")
        print(d.pivot_table(index="xi", columns="mu_ratio", values=col).round(3).to_string())


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(cell["n_fit"], rng, xi=generator.XI_OF_X)
    test = generator.gpd(cell["n_test"], rng, xi=generator.XI_OF_X)
    tercile, _ = gradients.generator_terciles(test.X)
    true = [float(test.xi[tercile == t].mean()) for t in range(3)]
    q = models.quantiles(cell["model"], train.X, train.y, test.X, seed=cell["seed"],
                         levels=LEVELS, n_est=cell["n_est"])
    r = dict(d_true=true[2] - true[0], xi_true_t0=true[0], xi_true_t2=true[2])
    for t in range(3):
        sel = tercile == t
        r[f"xi_raw_t{t}"] = metrics.implied_xi(q[sel, 2], q[sel, 1])
        r[f"xi_residual_t{t}"] = metrics.implied_xi(q[sel, 2], q[sel, 1], q[sel, 0])
    return r


def part2():
    cells = runner.grid(seed=list(range(N_SEEDS)), model=MODELS, n_est=[N_EST],
                        n_fit=[N_FIT], n_test=[N_TEST])
    runner.run(OUTPUT2, COLUMNS2, KEY2, cells, measure,
               show=lambda r: f"raw t0/t2 {r['xi_raw_t0']:.2f}/{r['xi_raw_t2']:.2f}  "
                              f"residual t0/t2 {r['xi_residual_t0']:.2f}/"
                              f"{r['xi_residual_t2']:.2f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT2)))
    if d.empty:
        return
    true = float(d.xi_true_t2.mean() - d.xi_true_t0.mean())
    d = d.assign(slope_raw=d.xi_raw_t2 - d.xi_raw_t0,
                 slope_residual=d.xi_residual_t2 - d.xi_residual_t0)
    print(f"\n=== part 2: true slope {true:.3f}; share by model, averaged over seeds ===")
    print((d.groupby("model")[["slope_raw", "slope_residual"]].mean() / true)
          .round(3).to_string())
    print("\n=== per-seed slope ===")
    print(d.groupby("model")[["slope_raw", "slope_residual"]]
          .agg(["median", "min", "max", "std"]).round(3).to_string())
    rng = np.random.default_rng(0)
    print("\n=== bootstrap 95% interval of the raw share over seeds ===")
    for model, g in d.groupby("model"):
        v = g.slope_raw.to_numpy()
        boot = [rng.choice(v, len(v), replace=True).mean() / true for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {model:10s} share {v.mean() / true:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    t0 = time.time()
    if "1" in PARTS:
        part1()
    if "2" in PARTS:
        part2()
    print(f"\ntotal {time.time() - t0:.0f}s")
