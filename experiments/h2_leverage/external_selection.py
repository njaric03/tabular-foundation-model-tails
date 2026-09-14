# -*- coding: utf-8 -*-
"""Which of the external heavy-tailed candidates enter the real-data pool.

The candidates are in `common/external.py`. The gate, written before any was loaded:

  G1. At least N_FIT + N_TEST = 3000 rows with a finite, positive target.
  G2. At least 3 covariates after the drops fixed in `common/external.py`.
  G3. A Hill tail index of at least 0.3 on the positive target (k = 10%), the level
      below which part one treats a tail as light.
  G4. A target vector not already recorded under another name.

Also measured, and not part of the gate: the natural sd shift over 200 subsamples of 2000,
as in `prevalence_data.py`. Whether a table carries leverage is an outcome, not a reason
to pick it. Writes `external_selection.csv` and `data/external_selected.txt`.

    venv-data/Scripts/python -m common.external       # once, converts
    python -u experiments/h2_leverage/external_selection.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, external, metrics, paths, quiet, runner

quiet.silence()

N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
N_SUBSAMPLES = env.integer("N_SUBSAMPLES", 200)
MIN_COVARIATES = 3
MIN_HILL = 0.3
THRESHOLDS = [2.0, 4.0, 10.0]
OUTPUT = env.text("OUTPUT", "external_selection.csv")

COLUMNS = (["dataset", "domain", "n_rows", "n_pos", "n_cov", "n_categorical", "hill_xi",
            "duplicate_of", "sd_shift_median", "sd_shift_max"]
           + [f"p_sd_{t}" for t in THRESHOLDS]
           + ["passed", "failed_gates", "n_fit", "n_subsamples", "seconds", "reason"])
KEY = ["dataset", "n_fit", "n_subsamples"]


def measure(cell):
    name = cell["dataset"]
    X, y = datasets.load(name)
    sha = datasets.LAST["fingerprint"]["y_sha1"]
    duplicates = sorted(n for n, s in datasets.target_groups().items() if s == sha and n != name)
    yp = y[np.isfinite(y) & (y > 0)]
    r = dict(domain=external.REGISTRY[name]["domain"], n_rows=len(y), n_pos=len(yp),
             n_cov=int(X.shape[1]), n_categorical=datasets.LAST.get("n_categorical", 0),
             hill_xi=float(metrics.hill(yp)) if len(yp) > 100 else np.nan,
             duplicate_of=",".join(duplicates))

    if len(yp) >= 600:
        n = min(cell["n_fit"], len(yp) // 2)
        rng = np.random.default_rng(20260824)          # the seed of prevalence_data.py
        shifts = np.array([metrics.sd_shift(yp[rng.permutation(len(yp))[:n]])
                           for _ in range(cell["n_subsamples"])])
        r.update(sd_shift_median=float(np.median(shifts)), sd_shift_max=float(shifts.max()))
        for t in THRESHOLDS:
            r[f"p_sd_{t}"] = float((shifts >= t).mean())

    failed = [gate for gate, fails in [("G1", r["n_pos"] < N_FIT + N_TEST),
                                       ("G2", r["n_cov"] < MIN_COVARIATES),
                                       ("G3", not r["hill_xi"] >= MIN_HILL),
                                       ("G4", bool(duplicates))] if fails]
    return dict(r, passed=not failed, failed_gates=",".join(failed))


def main():
    cells = runner.grid(dataset=list(external.REGISTRY), n_fit=[N_FIT],
                        n_subsamples=[N_SUBSAMPLES])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"n_pos {r['n_pos']}  hill {r['hill_xi']:.2f}  "
                                        f"{'PASS' if r['passed'] else 'fail ' + r['failed_gates']}")
    d = pd.read_csv(paths.result(OUTPUT))
    keep = list(d[d.passed.astype(str) == "True"].dataset)
    out = paths.data("external_selected.txt")
    out.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"\n{len(keep)} of {len(d)} passed -> {out.relative_to(paths.ROOT)}  [{seconds:.0f}s]")


if __name__ == "__main__":
    main()
