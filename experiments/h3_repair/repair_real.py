# -*- coding: utf-8 -*-
"""Winsorising on real tables, with and without a corrupted row.

One real extreme value barely moves the sd (1.13 to 1.36 times when removed), while the
effect needs 4 times or more. A spike of 10 to 100 times the maximum is a data error: a
wrong unit, a broken sensor, a duplicated record. So one corrupted row is injected, a copy
of a random context row with its target raised `dose` times, and measured:

    benefit   how much winsorising cuts the damage of that row
    cost      what winsorising costs on the clean table

since a user who does not know whether the table has a bad row would winsorise always.

    DATASETS=freMTPL2sev DOSES=10 SEEDS=3 python -u experiments/h3_repair/repair_real.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_FIT, N_TEST = 2000, 1500
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
WINSOR_SHARE = 0.005
DATASETS = env.names("DATASETS", "freMTPL2sev,OnlineNewsPopularity,Allstate_Claims_Severity")
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
DOSES = env.floats("DOSES", [0, 10, 100])        # 0 is the clean table
SEEDS = env.seeds(env.integer("SEEDS", 5))
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "repair_real.csv")

COLUMNS = ["n_est", "dataset", "model", "dose", "variant", "seed",
           "pb50", "pb90", "pb99", "pb999", "sd_shift", "seconds", "reason"]
KEY = ["dataset", "model", "dose", "variant", "seed", "n_est"]


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"], positive_only=True)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"too few rows ({len(y)})")
    rng = np.random.default_rng(cell["seed"])
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    Xf, yf, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]

    sd_clean = float(yf.std())
    if cell["dose"] > 0:
        j = int(rng.integers(len(yf)))
        Xf, yf = np.vstack([Xf, Xf[[j]]]), np.concatenate([yf, [cell["dose"] * yf.max()]])
    shift = float(yf.std() / sd_clean)
    if cell["variant"] == "winsor":
        yf = np.minimum(yf, np.quantile(yf, 1 - WINSOR_SHARE))

    q = models.quantiles(cell["model"], Xf, yf, Xte, seed=cell["seed"], levels=LEVELS,
                         n_est=cell["n_est"])
    return dict(sd_shift=shift, **{f"pb{t}": metrics.pinball(yte, q[:, i], a)
                                   for i, (a, t) in enumerate(zip(LEVELS, TAGS))})


def main():
    cells = runner.grid(dataset=DATASETS, model=MODELS, dose=DOSES, variant=["raw", "winsor"],
                        seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"pb99 {r['pb99']:.4g}  pb999 {r['pb999']:.4g}  "
                                        f"sd x{r['sd_shift']:.2f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== sd shift by dose, median ===")
    print(d.groupby("dose").sd_shift.median().round(2).to_string())
    for col in ("pb99", "pb999"):
        p = d.pivot_table(index=["dataset", "model", "dose"], columns="variant", values=col,
                          aggfunc="median")
        print(f"\n=== {col}: winsor relative to raw (negative means winsor is better) ===")
        print((p["winsor"] / p["raw"] - 1).unstack("dose").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
