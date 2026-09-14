# -*- coding: utf-8 -*-
"""H1, shape half: does a model change the tail shape of its prediction with x?

On the generator xi(x) rises with x4 while the scale ignores x4. Test rows are split into
terciles of x4; in each, the implied xi is inverted from the median ratio Q(0.99) / Q(0.9)
and set against the true mean xi there. `tables.slope_by_model` turns the rows into the
share of the true slope a model reproduces; `scale_of_x.py` measures the scale half the
same way.

    MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv \
      python -u experiments/h1_shape_vs_scale/shape_of_x.py
"""
import time

import numpy as np
import pandas as pd

from common import append, env, generator, gradients, metrics, models, paths, quiet, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
LEVELS = gradients.LEVELS
# The ensemble size moves TabPFN's shape share from 19% at 4 members to 42% at 1.
N_EST = env.integer("N_EST", 4)
SEEDS = env.seeds(env.integer("SEEDS", 20))
MODELS = env.models("GBM,TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "shape_of_x_20.csv")

COLUMNS = ["model", "seed", "tercile", "n_est", "xi_true", "ratio_model", "xi_implied"]
KEY = ["model", "seed", "tercile", "n_est"]


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
        test = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
        tercile, _ = gradients.generator_terciles(test.X)
        for name in MODELS:
            rows = [dict(model=name, seed=seed, tercile=t + 1, n_est=N_EST) for t in range(3)]
            if all(append.key(r, KEY) in done for r in rows):
                continue
            q = models.quantiles(name, train.X, train.y, test.X, seed=seed,
                                 levels=LEVELS, n_est=N_EST)
            for t, row in enumerate(rows):
                sel = tercile == t
                num, den = q[sel, 2], q[sel, 1]
                ok = den > 1e-9
                ratio = float(np.median(num[ok] / den[ok])) if ok.any() else np.nan
                row.update(xi_true=float(test.xi[sel].mean()), ratio_model=ratio,
                           xi_implied=(metrics.xi_from_ratio(ratio) if np.isfinite(ratio)
                                       else np.nan))
                if append.key(row, KEY) not in done:
                    append.write(OUTPUT, row, COLUMNS)
            shown = " / ".join(f"{r['xi_implied']:.2f}" for r in rows)
            print(f"  {name:10s} seed={seed}  implied xi by tercile: {shown}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    report(t0)


def report(t0=None):
    d = pd.read_csv(paths.result(OUTPUT))
    print("\n=== true and implied xi by tercile ===")
    print(d.groupby("tercile").xi_true.mean().round(3).to_string())
    print(d.pivot_table(index="tercile", columns="model", values="xi_implied",
                        aggfunc="mean").round(3).to_string())
    print("\n=== share of the true slope ===")
    print(tables.slope_by_model(d).round(3).to_string())
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
