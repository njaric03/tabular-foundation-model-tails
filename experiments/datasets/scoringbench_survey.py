# -*- coding: utf-8 -*-
"""The tail index of every ScoringBench table, with no model run.

ScoringBench (arXiv:2603.29928) publishes 51 models on 102 tables, including a right-tail
weighted CRPS, but tail weight is not an axis of it. This applies the estimators and the
entry test of `dataset_selection.py` to its tables, so its results can be stratified by
tail. `source`, `target_source` and `n_categorical` record how each table was loaded,
since the survey's base rate of leverage depends on which column became the target.

    python -u experiments/datasets/scoringbench_survey.py
"""
import pandas as pd

from common import datasets, env, paths, quiet, runner, selection

quiet.silence()

# Below this many positive values no estimate is computed.
MIN_N_MEASURE = 500
OUTPUT = env.text("OUTPUT", "scoringbench_survey.csv")

COLUMNS = ["dataset", "n", "d", "shifted", "gpd05", "gpd10", "gpd20", "hill", "quantile",
           "estimate_range", "range_threshold", "tied_share",
           "source", "target_source", "n_categorical", "reason", "passes"]
KEY = ["dataset"]


def measure(cell):
    X, y = datasets.load(cell["dataset"])
    loaded = dict(source=datasets.LAST.get("source", ""),
                  target_source=datasets.LAST.get("target_source", ""),
                  n_categorical=datasets.LAST.get("n_categorical", 0))
    X, y = selection.subsample(X, y)
    # Non-positive targets are dropped, as in dataset_selection.py. Shifting is only a
    # rescue for tables that would otherwise keep too few points (Buzz has many zeros).
    shifted = False
    if (y > 0).sum() < MIN_N_MEASURE <= len(y):
        y = y - y.min() + 1e-6 * (y.max() - y.min() or 1.0)
        shifted = True
    if (y > 0).sum() < MIN_N_MEASURE:
        return dict(loaded, n=int((y > 0).sum()), reason="too few values for an estimate",
                    passes=False)
    return dict(selection.estimates(X, y), loaded, d=X.shape[1], shifted=shifted)


def main():
    cells = runner.grid(dataset=datasets.scoringbench_names())
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"GPD {r['gpd10']:+.2f} Hill {r['hill']:+.2f} "
                                        f"quantile {r['quantile']:+.2f} | "
                                        f"{'PASS' if r['passes'] else '-'}")
    df = pd.read_csv(paths.result(OUTPUT))
    print(f"\ndone in {seconds:.0f}s | estimated {df.gpd10.notna().sum()} of {len(df)} | "
          f"passing {int(df.passes.fillna(False).astype(bool).sum())}")


if __name__ == "__main__":
    main()
