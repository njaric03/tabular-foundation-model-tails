# -*- coding: utf-8 -*-
"""Which of the external heavy-tailed candidates enter the real-data pool.

WHY
---
Parts two and three stand on one real table, freMTPL2sev, and the OpenML pool
cannot give a second: benchmark tables are curated, and only 2 of 103 surveyed
datasets carry both a leverage of 2 sd and a tail index above 0.5. The
candidates in `common/external.py` come from the fields where heavy tails with
covariates are the use case: insurance severity, health expenditure, online
popularity. This script decides which of them count.

THE GATE, WRITTEN BEFORE ANY CANDIDATE WAS LOADED
-------------------------------------------------
  G1. At least N_FIT + N_TEST = 3000 rows with a finite, positive target. Those
      are the rows every real-data script keeps, and 3000 is what one subsample
      of the leverage sweep needs.
  G2. At least 3 covariates after the drops fixed in `common/external.py`.
  G3. A Hill tail index of the positive target of at least 0.3 (k = 10%), the
      level below which part one treats a tail as light.
  G4. A target vector not already in the pool: its `y_sha1` differs from every
      other recorded fingerprint.

Also measured, and deliberately not part of the gate: the natural sd shift over
200 subsamples of 2000, the statistic of `prevalence_data.py`, here on the
positive target that the models see. Whether a dataset carries leverage is an
outcome to report, not a reason to pick it.

Writes `external_selection.csv`, one row per candidate with the reason it failed
if it did, and `data/external_selected.txt`, the names that passed.

RUNNING
-------
    venv-data/Scripts/python -m common.external           # once, converts
    python -u experiments/h2_leverage/external_selection.py
"""
import os
import time

import numpy as np

from common import append, datasets, external, metrics, paths, quiet

quiet.silence()

N_FIT = int(os.environ.get("N_FIT", "2000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
N_SUBSAMPLES = int(os.environ.get("N_SUBSAMPLES", "200"))
MIN_COVARIATES = 3
MIN_HILL = 0.3
THRESHOLDS = [2.0, 4.0, 10.0]
OUT = os.environ.get("OUTPUT", "external_selection.csv")

COLUMNS = (["dataset", "domain", "n_rows", "n_pos", "n_cov", "n_categorical", "hill_xi",
            "duplicate_of", "sd_shift_median", "sd_shift_max"]
           + [f"p_sd_{t}" for t in THRESHOLDS]
           + ["passed", "failed_gates", "n_fit", "n_subsamples", "seconds", "reason"])
KEY = ["dataset", "n_fit", "n_subsamples"]


def one(name):
    X, y = datasets.load(name)
    n_cat = datasets.LAST.get("n_categorical", 0)
    sha = datasets.LAST["fingerprint"]["y_sha1"]
    dup = sorted(n for n, s in datasets.target_groups().items() if s == sha and n != name)
    ok = np.isfinite(y) & (y > 0)
    yp = y[ok]
    r = dict(dataset=name, domain=external.REGISTRY[name]["domain"], n_rows=len(y),
             n_pos=int(ok.sum()), n_cov=int(X.shape[1]), n_categorical=n_cat,
             hill_xi=float(metrics.hill(yp)) if len(yp) > 100 else np.nan,
             duplicate_of=",".join(dup), n_fit=N_FIT, n_subsamples=N_SUBSAMPLES)

    if len(yp) >= 2 * 300:
        n = min(N_FIT, len(yp) // 2)
        rng = np.random.default_rng(20260824)          # prevalence_data's seed
        shifts = np.array([metrics.sd_shift(yp[rng.permutation(len(yp))[:n]])
                           for _ in range(N_SUBSAMPLES)])
        r.update(sd_shift_median=float(np.median(shifts)), sd_shift_max=float(shifts.max()))
        for t in THRESHOLDS:
            r[f"p_sd_{t}"] = float((shifts >= t).mean())

    failed = []
    if r["n_pos"] < N_FIT + N_TEST:
        failed.append("G1")
    if r["n_cov"] < MIN_COVARIATES:
        failed.append("G2")
    if not (r["hill_xi"] >= MIN_HILL):
        failed.append("G3")
    if dup:
        failed.append("G4")
    r.update(passed=not failed, failed_gates=",".join(failed), reason="")
    return r


def main():
    done = append.done(OUT, KEY)
    t0 = time.time()
    for name in external.REGISTRY:
        if append.key(dict(dataset=name, n_fit=N_FIT, n_subsamples=N_SUBSAMPLES), KEY) in done:
            continue
        t1 = time.time()
        try:
            r = one(name)
        except Exception as e:
            r = dict(dataset=name, passed=False, reason=f"{type(e).__name__}: {e}"[:110],
                     n_fit=N_FIT, n_subsamples=N_SUBSAMPLES)
        r["seconds"] = round(time.time() - t1, 1)
        append.write(OUT, r, COLUMNS)
        print(f"  {name:18s} n_pos {r.get('n_pos', 0):>7}  cov {r.get('n_cov', 0):>3}  "
              f"hill {r.get('hill_xi', np.nan):5.2f}  sd_shift max {r.get('sd_shift_max', np.nan):6.2f}"
              f"  P(>=2) {r.get('p_sd_2.0', np.nan):5.1%}  "
              f"{'PASS' if r['passed'] else 'fail ' + (r.get('failed_gates') or r.get('reason', ''))}",
              flush=True)
    import pandas as pd
    d = pd.read_csv(paths.result(OUT))
    keep = list(d[d.passed.astype(str) == "True"].dataset)
    out = paths.data("external_selected.txt")
    out.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"\n{len(keep)} of {len(d)} passed -> {out.relative_to(paths.ROOT)}  "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
