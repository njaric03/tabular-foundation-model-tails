# -*- coding: utf-8 -*-
"""Strict selection of the tables the post-hoc tail repair is evaluated on.

Most public tables fail to identify their tail index: three estimators disagree, so the
reference a model would be scored against moves with the estimator. The entry test is in
`common/selection.py`. The share of tied values is reported too, since tables like
freMTPL2sev put much of their mass on a few points. Writes `dataset_selection.csv` and
the names that pass to `data/selected_datasets.txt` (or SELECTED).

    python -u experiments/datasets/dataset_selection.py
"""
import time

import pandas as pd

from common import append, datasets, env, paths, quiet, runner, selection

quiet.silence()

OUTPUT = env.text("OUTPUT", "dataset_selection.csv")
# A knob, so a check run does not overwrite the list three experiments read.
SELECTED = env.text("SELECTED", "selected_datasets.txt")

CANDIDATES = [
    ("freMTPL2sev", "claim severity, French motor insurance"),
    ("delays_zurich_transport", "transit delay"),
    ("Airlines_DepDelay_1M", "departure delay"),
    ("Buzzinsocialmedia_Twitter", "interaction count"),
    ("medical_charges", "medical cost"),
    ("nyc-taxi-green-dec-2016", "fare amount"),
    ("black_friday", "purchase amount"),
    ("diamonds", "price"),
    ("SGEMM_GPU_kernel_performance", "runtime"),
    ("Allstate_Claims_Severity", "claim severity, Allstate"),
    ("video_transcoding", "transcoding time"),
    ("Bike_Sharing_Demand", "rental count"),
    ("house_sales", "house price"),
    ("Yolanda", "synthetic regression"),
    ("wine_quality", "wine score, light-tailed control"),
]
COLUMNS = ["dataset", "description", "n", "d", "gpd10", "hill", "quantile",
           "estimate_range", "range_threshold", "tied_share", "passes"]


def main():
    t0, rows = time.time(), []
    for name, description in CANDIDATES:
        try:
            X, y = selection.subsample(*datasets.load(name))
        except Exception as e:
            print(f"  {name:30s} skipped, {runner.reason(e)}", flush=True)
            continue
        if (y > 0).sum() < selection.MIN_N:
            print(f"  {name:30s} too few positive values ({(y > 0).sum()})", flush=True)
            continue
        r = dict(selection.estimates(X, y), dataset=name, description=description, d=X.shape[1])
        rows.append(r)
        print(f"  {name:30s} GPD {r['gpd10']:+.2f}  Hill {r['hill']:+.2f}  quantile "
              f"{r['quantile']:+.2f} | range {r['estimate_range']:.2f} / threshold "
              f"{r['range_threshold']:.2f} | tied {r['tied_share']:.1%}  "
              f"{'PASS' if r['passes'] else '-'}", flush=True)

    df = pd.DataFrame(rows)[COLUMNS].sort_values(["passes", "gpd10"], ascending=[False, False])
    append.replace(OUTPUT, df)
    selected = list(df[df.passes].dataset)
    paths.data(SELECTED).write_text("\n".join(selected), encoding="utf-8")
    print(f"\npass ({len(selected)}): {selected}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
