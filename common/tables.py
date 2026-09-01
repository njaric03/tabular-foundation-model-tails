# -*- coding: utf-8 -*-
"""Aggregations the findings quote, so a notebook cell never carries its own logic.

Each function takes a result frame and returns the table exactly as the findings
report it. Keeping them here means the notebook and any script agree by
construction, and a changed aggregation shows up in one place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ok_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Rows whose measurement succeeded.

    Filtering on the success flag rather than on notna() of an estimate: the
    tail-index inversion returns NaN exactly on the heaviest tails, so filtering
    on the estimate would drop heavy tails preferentially.
    """
    if "reason" not in d.columns:
        return d
    return d[d.reason.fillna("") == ""]


def slope_by_model(d: pd.DataFrame, value="xi_implied", truth="xi_true",
                   group="tercile") -> pd.DataFrame:
    """Change from the first to the last group, per model, against the true change.

    This is the shape half of the dissociation: what fraction of the true
    gradient the model reproduces.
    """
    d = ok_rows(d)
    piv = d.pivot_table(index=group, columns="model", values=value, aggfunc="mean")
    truth_by = d.groupby(group)[truth].mean()
    true_slope = float(truth_by.iloc[-1] - truth_by.iloc[0])
    rows = []
    for m in piv.columns:
        slope = float(piv[m].iloc[-1] - piv[m].iloc[0])
        rows.append(dict(model=m, model_slope=slope, true_slope=true_slope,
                         share=slope / true_slope if true_slope else np.nan))
    return pd.DataFrame(rows).set_index("model").sort_values("share", ascending=False)


def scale_share_by_model(d: pd.DataFrame) -> pd.DataFrame:
    """The scale half: the model median against the true median, across terciles.

    Measured on the log of the ratio between the outer terciles, which is what
    scale_of_x.py records, so the two halves of the dissociation are produced by
    the same procedure on the same data.
    """
    d = ok_rows(d)
    out = []
    for m, g in d.groupby("model"):
        piv = g.pivot_table(index="tercile", values=["median_model", "median_true"],
                            aggfunc="mean")
        num = np.log(piv.median_model.iloc[-1] / piv.median_model.iloc[0])
        den = np.log(piv.median_true.iloc[-1] / piv.median_true.iloc[0])
        out.append(dict(model=m, share=float(num / den) if den else np.nan))
    return pd.DataFrame(out).set_index("model").sort_values("share", ascending=False)


def share_by_run(shape: pd.DataFrame, scale: pd.DataFrame) -> pd.DataFrame:
    """Both shares per (model, seed), for the paired test on the synthetic data.

    `slope_by_model` and `scale_share_by_model` average over seeds first and are
    what the headline table quotes. The paired test needs the two shares run by
    run instead, and the free unit there is the seed: one seed is one draw of the
    data, while the three models share it. See `common/stats.py`.
    """
    def per_run(d, f, name):
        out = []
        for (m, s), g in ok_rows(d).groupby(["model", "seed"]):
            g = g.sort_values("tercile")
            if len(g) < 3:
                continue
            out.append({"model": m, "seed": s, name: f(g)})
        return pd.DataFrame(out)

    a = per_run(shape, lambda g: ((g.xi_implied.iloc[-1] - g.xi_implied.iloc[0])
                                  / (g.xi_true.iloc[-1] - g.xi_true.iloc[0])),
                "shape_share")
    b = per_run(scale, lambda g: (np.log(g.median_model.iloc[-1] / g.median_model.iloc[0])
                                  / np.log(g.median_true.iloc[-1] / g.median_true.iloc[0])),
                "scale_share")
    return a.merge(b, on=["model", "seed"]).dropna()


def median_response(d: pd.DataFrame, col: str, by=("model",)) -> pd.DataFrame:
    """Median response with the spread that a single median would hide.

    The findings record three separate failures caused by reporting a median of
    a few seeds over a bimodal distribution, so min, max and n travel with it.
    """
    d = ok_rows(d)
    g = d.groupby(list(by))[col]
    return pd.DataFrame({"median": g.median(), "min": g.min(), "max": g.max(),
                         "n": g.count()}).sort_values("median")


def paired_rank_test(d: pd.DataFrame, a="scale_share", b="shape_share",
                     cluster="dataset"):
    """How often a is closer to 1 than b, at the row level and per cluster.

    The real-data claim is made by rank rather than as a percentage because the
    reference shape gradient there is weak, so the ratio explodes.

    Two p-values come back, and the conservative one is the one to quote. The
    rows are nested -- 9 datasets x 3 models x 3 seeds -- so a Wilcoxon over all
    of them counts 59 dependent comparisons as if they were 59 independent ones.
    `cluster_p` reduces each dataset to one median difference first and runs an
    exact sign test on those. See `common/stats.py`.
    """
    from common import stats
    d = ok_rows(d)
    r = stats.paired_by_cluster(d, a=a, b=b, cluster=cluster)
    return dict(n=r["n"], a_closer=r["wins"], share=r["share"], p=r["naive_p"],
                clusters=r["clusters"], cluster_wins=r["cluster_wins"],
                cluster_p=r["cluster_p"])
