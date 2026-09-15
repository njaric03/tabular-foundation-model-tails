# -*- coding: utf-8 -*-
"""H1, scale half: does a model track the conditional scale s(x)?

The generator of `shape_of_x.py`. Test rows are split into terciles of the scale's linear
predictor, and in each the median predicted Q(0.5) is set against the median true
conditional median. `tables.scale_share_by_model` turns the rows into the share of the
true log change a model reproduces, the number the shape share is compared with.

    MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=scale_of_x_exaone_nest1.csv \
      python -u experiments/h1_shape_vs_scale/scale_of_x.py
"""
import time

import numpy as np
import pandas as pd

from common import append, env, generator, gradients, metrics, models, paths, quiet, tables

quiet.silence()

# A knob since nanoTabPFN; see shape_of_x.py. Files without n_train hold 2000.
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = 900
LEVELS = gradients.LEVELS
N_EST = env.integer("N_EST", 4)
SEEDS = env.seeds(env.integer("SEEDS", 20))
MODELS = env.models("GBM,TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "scale_of_x.csv")

COLUMNS = ["model", "seed", "tercile", "n_est", "n_train", "median_true", "median_model",
           "xi_mean"]
KEY = ["model", "seed", "tercile", "n_est", "n_train"]


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
        test = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
        _, tercile = gradients.generator_terciles(test.X)
        median_true = test.s * metrics.gpd_quantile(0.5, test.xi)
        for name in MODELS:
            rows = [dict(model=name, seed=seed, tercile=t + 1, n_est=N_EST, n_train=N_TRAIN)
                    for t in range(3)]
            if all(append.key(r, KEY) in done for r in rows):
                continue
            q = models.quantiles(name, train.X, train.y, test.X, seed=seed,
                                 levels=LEVELS, n_est=N_EST)
            for t, row in enumerate(rows):
                sel = tercile == t
                row.update(median_true=float(np.median(median_true[sel])),
                           median_model=float(np.median(q[sel, 0])),
                           xi_mean=float(test.xi[sel].mean()))
                if append.key(row, KEY) not in done:
                    append.write(OUTPUT, row, COLUMNS)
            shown = " / ".join(f"{r['median_model']:.2f} ({r['median_true']:.2f})" for r in rows)
            print(f"  {name:10s} seed={seed}  median by tercile, model (true): {shown}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    print("\n=== share of the true log change in the median ===")
    print(tables.scale_share_by_model(pd.read_csv(paths.result(OUTPUT))).round(3).to_string())
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
