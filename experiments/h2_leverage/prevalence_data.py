# -*- coding: utf-8 -*-
"""How often a real regression table contains a row that dominates the sd. No model.

The easy objection to part two is that an injected row of 100 * max is artificial. The
answer is the base rate. Per ScoringBench table (plus freMTPL2sev), N_SUBSAMPLES
subsamples of N_FIT rows, and in each:

    leverage   largest / second largest
    sd shift   sd with the largest value / sd without it, the quantity that matters

The shares above 1.5, 2, 4 and 10 are reported, and the tables with a shift of 4 or more
go to `data/prevalence_at_risk.json` for `prevalence_models.py`.

    N_SUBSAMPLES=50 DATASETS=freMTPL2sev,Allstate_Claims_Severity \
      python -u experiments/h2_leverage/prevalence_data.py
"""
import json

import numpy as np
import pandas as pd

from common import datasets, env, metrics, paths, quiet, runner, tables

quiet.silence()

# 200 subsamples per table is what the cited file was measured with (99 x 200).
N_SUBSAMPLES = env.integer("N_SUBSAMPLES", 200)
N_FIT = env.integer("N_FIT", 2000)
THRESHOLDS = [1.5, 2.0, 4.0, 10.0]
LEVERAGE_THRESHOLDS = (3, 5, 10, 100)
OUTPUT = env.text("OUTPUT", "prevalence_data.csv")
# freMTPL2sev is not in ScoringBench, and the cited file carries it.
_SUITE = datasets.scoringbench_names()
DATASETS = env.names("DATASETS", _SUITE + [n for n in ["freMTPL2sev"] if n not in _SUITE])

COLUMNS = (["dataset", "n_rows", "n_subsamples", "leverage_median", "leverage_p90",
            "sd_shift_median", "sd_shift_max"]
           + [f"p_sd_{p}" for p in THRESHOLDS]
           + [f"p_leverage_{p}" for p in LEVERAGE_THRESHOLDS]
           + ["hill_xi", "n_fit", "seconds", "reason"])
KEY = ["dataset", "n_fit", "n_subsamples"]


def measure(cell):
    _, y = datasets.load(cell["dataset"])
    y = y[np.isfinite(y)]
    if len(y) < 300:
        raise RuntimeError(f"too few rows ({len(y)})")
    n = min(cell["n_fit"], len(y) // 2)
    rng = np.random.default_rng(20260824)
    leverage, shift = [], []
    for _ in range(cell["n_subsamples"]):
        u = y[rng.permutation(len(y))[:n]]
        p = np.sort(u)
        leverage.append(abs(p[-1]) / max(abs(p[-2]), 1e-12))
        shift.append(np.std(u) / max(np.std(p[:-1]), 1e-12))
    leverage, shift = np.asarray(leverage), np.asarray(shift)
    r = dict(n_rows=len(y), leverage_median=float(np.median(leverage)),
             leverage_p90=float(np.quantile(leverage, 0.9)),
             sd_shift_median=float(np.median(shift)), sd_shift_max=float(shift.max()),
             hill_xi=metrics.hill(y))
    r.update({f"p_sd_{p}": float((shift >= p).mean()) for p in THRESHOLDS})
    r.update({f"p_leverage_{p}": float((leverage >= p).mean()) for p in LEVERAGE_THRESHOLDS})
    return r


def report(d):
    d = tables.ok_rows(d)
    if d.empty:
        return
    print(f"\n=== prevalence over {len(d)} tables ===")
    for p in THRESHOLDS:
        col = f"p_sd_{p}"
        print(f"  sd shift >= {p:>4}x in some subsample: {(d[col] > 0).sum():>3}/{len(d)} "
              f"tables | in >= 5% of subsamples: {(d[col] >= 0.05).sum():>3}")
    for p in LEVERAGE_THRESHOLDS:
        print(f"  leverage >= {p:>4}x: {(d[f'p_leverage_{p}'] > 0).sum():>3}/{len(d)} tables")

    at_risk = d[d["p_sd_4.0"] > 0].sort_values("p_sd_4.0", ascending=False)
    print("\n=== tables with an sd shift of 4 or more, the input of prevalence_models.py ===")
    print(at_risk[["dataset", "n_rows", "sd_shift_median", "sd_shift_max", "p_sd_4.0",
                   "hill_xi"]].to_string(index=False))
    out = paths.data("prevalence_at_risk.json")
    out.write_text(json.dumps(list(at_risk.dataset), ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  -> {len(at_risk)} tables in {out.relative_to(paths.ROOT)}")


def main():
    cells = runner.grid(dataset=DATASETS, n_fit=[N_FIT], n_subsamples=[N_SUBSAMPLES])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"n={r['n_rows']} sd shift median "
                                        f"{r['sd_shift_median']:.2f} max {r['sd_shift_max']:.2f}"
                                        f" | xi_hill {r['hill_xi']:.2f}")
    report(pd.read_csv(paths.result(OUTPUT)))
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
