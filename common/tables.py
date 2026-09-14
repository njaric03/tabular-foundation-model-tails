# -*- coding: utf-8 -*-
"""The aggregations the notebooks print, kept here so a notebook cell carries no logic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ok_rows(d: pd.DataFrame) -> pd.DataFrame:
    """Rows whose measurement succeeded, by the `reason` column.

    Not notna() of an estimate: the tail-index inversion returns NaN exactly on the
    heaviest tails, so that filter would drop them preferentially.
    """
    if "reason" not in d.columns:
        return d
    return d[d.reason.fillna("") == ""]


def slope_by_model(d: pd.DataFrame, value="xi_implied", truth="xi_true",
                   group="tercile") -> pd.DataFrame:
    """Per model, the change of `value` from the first group to the last against the true
    change: the shape half of the dissociation."""
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
    """Per model, the log ratio of the outer terciles' predicted medians against the true
    one: the scale half, by the same procedure as the shape half."""
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
    """Both shares per (model, seed), for the paired test on the generator, where the seed
    is the free unit."""
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
    """Median response with min, max and n, since a median of few seeds over a bimodal
    response has already misled three times."""
    g = ok_rows(d).groupby(list(by))[col]
    return pd.DataFrame({"median": g.median(), "min": g.min(), "max": g.max(),
                         "n": g.count()}).sort_values("median")


def paired_rank_test(d: pd.DataFrame, a="scale_share", b="shape_share",
                     column="dataset"):
    """How often `a` is closer to 1 than `b`, per row and per target vector.

    Real-data claims are rank claims because the reference shape gradient is weak and
    the ratio explodes. Quote `cluster_p`, clustered by target vector (rule 6); the
    row-level `p` treats nested seeds and models as independent.
    """
    from common import stats
    r = stats.paired_by_target(ok_rows(d), a=a, b=b, column=column)
    return dict(n=r["n"], a_closer=r["wins"], share=r["share"], p=r["naive_p"],
                clusters=r["clusters"], cluster_wins=r["cluster_wins"],
                cluster_p=r["cluster_p"])


def canonical_models(d: pd.DataFrame, column="model") -> pd.DataFrame:
    """The `model` column mapped to canonical names, for frames written before
    `models.parse_list` normalised them."""
    from common import models

    if column not in d.columns:
        return d
    out = d.copy()
    out[column] = out[column].map(
        lambda n: models.SYNONYMS.get(n, n) if isinstance(n, str) else n)
    return out
