# -*- coding: utf-8 -*-
"""Conditional on the leverage a real subsample carries, does Q(0.99) move?

Only the tables `prevalence_data.py` flags (sd shift of 2 or more somewhere). In each,
subsamples of N_FIT + N_TEST rows are rejection-sampled into four bins of their natural
sd shift, and every model is fitted with and without the largest context value.

Predictions written before the run: at a shift of 4 and above, a Q(0.99) change of 10 to
30% for TabICL and a bimodal one for TabPFN; below 1.2, under 5%; GBM and XGBoost flat
in every bin. If the bins do not separate, the mechanism does not hold on real data.

    DATASETS=freMTPL2sev MODELS=GBM N_PER_BIN=5 \
      python -u experiments/h2_leverage/prevalence_models.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, leverage, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = [0.5, 0.9, 0.99]
N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
N_EST = env.integer("N_EST", 1)
N_PER_BIN = env.integer("N_PER_BIN", 8)
# Only bounds the search: a repeat's subsample does not depend on it.
MAX_ATTEMPTS = env.integer("MAX_ATTEMPTS", 4000)
MODELS = env.models("GBM,XGB,TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "prevalence_models.csv")

COLUMNS = ["n_est", "n_fit", "n_test", "dataset", "model", "bin", "repeat", "sd_shift",
           "leverage", "d_q50", "d_q90", "d_q99", "seconds", "reason"]
KEY = ["dataset", "model", "bin", "repeat", "n_est", "n_fit", "n_test"]


def default_datasets() -> list[str]:
    path = paths.result("prevalence_data.csv")
    if not path.exists():
        return ["freMTPL2sev"]
    d = pd.read_csv(path)
    d = tables.ok_rows(d)
    return list(d[d.sd_shift_max >= 2.0].sort_values("sd_shift_max", ascending=False).dataset)


def cells():
    for name in env.names("DATASETS", default_datasets()):
        try:
            _, y = datasets.load_cached(name, positive_only=True)
        except Exception as e:
            print(f"{name}: does not load, {runner.reason(e)}", flush=True)
            continue
        if len(y) < N_FIT + N_TEST:
            print(f"{name}: too few rows", flush=True)
            continue
        found = leverage.subsamples(y, np.random.default_rng(31337), N_FIT, N_TEST,
                                    N_PER_BIN, MAX_ATTEMPTS)
        print(f"\n{name}: subsamples per bin " + ", ".join(
            f"{leverage.bin_name(k)}: {len(v)}" for k, v in found.items()), flush=True)
        for k, subsamples in found.items():
            for j, (idx, shift, lev) in enumerate(subsamples):
                for model in MODELS:
                    yield dict(dataset=name, model=model, bin=leverage.bin_name(k), repeat=j,
                               n_est=N_EST, n_fit=N_FIT, n_test=N_TEST,
                               sd_shift=shift, leverage=lev, _idx=idx)


def measure(cell):
    X, y = datasets.load_cached(cell["dataset"], positive_only=True)
    idx = cell["_idx"]
    Xf, yf, Xte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]]
    keep = np.ones(len(yf), bool)
    keep[int(np.argmax(yf))] = False
    seed = 7000 + cell["repeat"]
    q_without = models.quantiles(cell["model"], Xf[keep], yf[keep], Xte, seed=seed,
                                 levels=LEVELS, n_est=cell["n_est"])
    q_with = models.quantiles(cell["model"], Xf, yf, Xte, seed=seed, levels=LEVELS,
                              n_est=cell["n_est"])
    d = metrics.rel_change_by_level(q_without, q_with)
    return dict(d_q50=d[0], d_q90=d[1], d_q99=d[2])


def report():
    path = paths.result(OUTPUT)
    if not path.exists():
        return
    d = tables.ok_rows(pd.read_csv(path))
    for col in ("d_q99", "d_q50"):
        print(f"\n=== change of {col}, median by sd-shift bin ===")
        print(d.pivot_table(index="bin", columns="model", values=col,
                            aggfunc="median").round(3).to_string())
    top = d[d.bin.astype(str).str.startswith("4.0")]
    if len(top):
        print("\n=== spread in the top bin ===")
        print(top.groupby("model").d_q99.describe()[["count", "min", "50%", "max"]]
              .round(3).to_string())


def main():
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells(), measure,
                         show=lambda r: f"sd {r['sd_shift']:5.2f}  q99 {r['d_q99']:+7.1%}  "
                                        f"q50 {r['d_q50']:+6.1%}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
