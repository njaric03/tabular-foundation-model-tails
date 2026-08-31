# -*- coding: utf-8 -*-
"""Tests that respect how the measurements are nested.

THE PROBLEM THIS FIXES

`dissociation_real.csv` holds 59 usable comparisons: 9 datasets x 3 models x 3
seeds. A paired Wilcoxon over those 59 rows assumes 59 independent pairs, and
they are not. Three seeds on the same dataset with the same model are three
looks at nearly the same quantity, and three models on one dataset share the
dataset. The unit that varies freely is the dataset, of which there are nine.

The consequence is only about precision, not about direction: the row-level
p = 3.4e-11 is an overstatement of evidence that is already visible by eye, and
at the dataset level the same claim stands at p = 0.002. The honest number is
the second one, and it is the one a committee will ask for.

The same applies to the synthetic families: 60 runs are 4 families x 5 seeds x 3
models, where the free unit is (family, seed) -- 20 of them, not 60.

WHAT IS REPORTED

`paired_by_cluster` returns both, side by side, so the text can quote the
conservative number and still show the row-level one:

    naive_p     paired Wilcoxon over all rows, pseudoreplicated
    cluster_p   exact sign test over cluster medians, one-sided
    clusters    how many free units the second number rests on
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_by_cluster(d: pd.DataFrame, a: str, b: str, cluster="dataset",
                      target: float = 1.0) -> dict:
    """Is `a` closer to `target` than `b`, counted once per cluster?

    Rows are reduced to one median difference per cluster, and the sign test
    runs on those. `cluster` may be a column name or a list of them.
    """
    from scipy.stats import binomtest, wilcoxon

    keys = [cluster] if isinstance(cluster, str) else list(cluster)
    d = d.dropna(subset=[a, b]).copy()
    if d.empty:
        return dict(n=0, clusters=0, wins=0, cluster_wins=0,
                    naive_p=np.nan, cluster_p=np.nan)

    d["_da"] = (d[a] - target).abs()
    d["_db"] = (d[b] - target).abs()
    d["_diff"] = d._db - d._da            # positive when a is closer

    naive_p = float(wilcoxon(d._da, d._db).pvalue) if len(d) > 1 else np.nan

    per = d.groupby(keys)._diff.median()
    per = per[per != 0]
    k, n = int((per > 0).sum()), int(len(per))
    cluster_p = (float(binomtest(k, n, 0.5, alternative="greater").pvalue)
                 if n else np.nan)

    return dict(n=int(len(d)), wins=int((d._diff > 0).sum()),
                share=float((d._diff > 0).mean()),
                clusters=n, cluster_wins=k,
                naive_p=naive_p, cluster_p=cluster_p)


def describe(res: dict, unit="datasets") -> str:
    """One line for a findings document."""
    return (f"{res['wins']}/{res['n']} rows (naive p = {res['naive_p']:.2g}, "
            f"pseudoreplicated); {res['cluster_wins']}/{res['clusters']} {unit} "
            f"(sign test p = {res['cluster_p']:.2g})")
