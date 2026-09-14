# -*- coding: utf-8 -*-
"""Paired tests that count each free unit once.

The rows of a result file are nested: seeds within models within datasets. A Wilcoxon
over all rows treats them as independent and overstates the evidence (p = 3.4e-11 on
the real-data dissociation). `paired_by_cluster` reduces each cluster to one median
difference and runs an exact sign test on those; `paired_by_target` clusters by target
vector, since two dataset names can serve one target (rule 6). Both p-values come back,
and the text quotes `cluster_p`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_by_cluster(d: pd.DataFrame, a: str, b: str, cluster="dataset",
                      target: float = 1.0) -> dict:
    """Is `a` closer to `target` than `b`, counted once per cluster?

    `cluster` is a column name or a list of them. Clusters whose median difference is
    exactly zero drop out of the sign test.
    """
    from scipy.stats import binomtest, wilcoxon

    keys = [cluster] if isinstance(cluster, str) else list(cluster)
    d = d.dropna(subset=[a, b]).copy()
    if d.empty:
        return dict(n=0, wins=0, share=np.nan, clusters=0, cluster_wins=0,
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
    """One line with both levels of the test."""
    return (f"{res['wins']}/{res['n']} rows (naive p = {res['naive_p']:.2g}, "
            f"pseudoreplicated); {res['cluster_wins']}/{res['clusters']} {unit} "
            f"(sign test p = {res['cluster_p']:.2g})")


def add_target_group(d: pd.DataFrame, column="dataset") -> pd.DataFrame:
    """A copy with a `target_group` column: the target vector's hash, or the name when
    no fingerprint is recorded."""
    from common import datasets

    groups = datasets.target_groups()
    out = d.copy()
    out["target_group"] = out[column].map(lambda n: groups.get(n, n))
    return out


def paired_by_target(d: pd.DataFrame, a: str, b: str, column="dataset",
                     target: float = 1.0) -> dict:
    """`paired_by_cluster` with the target vector, not the dataset name, as the unit."""
    return paired_by_cluster(add_target_group(d, column), a, b,
                             cluster="target_group", target=target)
