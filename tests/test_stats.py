# -*- coding: utf-8 -*-
"""The clustered test in `common/stats.py`.

The point of the module is that a nested design must not be counted as if it
were flat, so the tests are about exactly that: the same effect, told once per
free unit, must give a weaker p-value than the same effect told three times per
free unit.
"""
import numpy as np
import pandas as pd
import pytest

from common import stats


def frame(n_datasets=9, n_models=3, n_seeds=3, effect=0.3, seed=0):
    """A nested design: one true effect per dataset, repeated by model and seed."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_datasets):
        for m in range(n_models):
            for s in range(n_seeds):
                rows.append(dict(dataset=f"d{d}", model=f"m{m}", seed=s,
                                 scale_share=1 + 0.02 * rng.normal(),
                                 shape_share=1 - effect + 0.02 * rng.normal()))
    return pd.DataFrame(rows)


def test_clustering_costs_significance_and_keeps_direction():
    r = stats.paired_by_cluster(frame(), "scale_share", "shape_share")
    assert r["n"] == 81 and r["clusters"] == 9
    assert r["wins"] == 81 and r["cluster_wins"] == 9
    assert r["cluster_p"] > r["naive_p"]
    assert r["cluster_p"] == pytest.approx(0.5 ** 9, rel=1e-6)


def test_more_seeds_do_not_buy_cluster_level_evidence():
    """Nine datasets are nine datasets, whether measured three times or thirty."""
    a = stats.paired_by_cluster(frame(n_seeds=3), "scale_share", "shape_share")
    b = stats.paired_by_cluster(frame(n_seeds=30), "scale_share", "shape_share")
    assert b["n"] > a["n"]
    assert b["naive_p"] < a["naive_p"]           # the misleading part
    assert b["cluster_p"] == a["cluster_p"]      # the honest part


def test_no_effect_is_not_significant_at_the_cluster_level():
    r = stats.paired_by_cluster(frame(effect=0.0, seed=1), "scale_share", "shape_share")
    assert r["cluster_p"] > 0.05


def test_cluster_can_be_a_list_of_columns():
    r = stats.paired_by_cluster(frame(), "scale_share", "shape_share",
                                cluster=["dataset", "model"])
    assert r["clusters"] == 27


def test_ties_are_dropped_from_the_sign_test():
    d = frame(n_datasets=4, n_models=1, n_seeds=1, effect=0.0, seed=2)
    d.loc[:, "shape_share"] = d.scale_share          # every pair an exact tie
    r = stats.paired_by_cluster(d, "scale_share", "shape_share")
    assert r["clusters"] == 0 and np.isnan(r["cluster_p"])


def test_empty_input_does_not_raise():
    d = frame().iloc[0:0]
    assert stats.paired_by_cluster(d, "scale_share", "shape_share")["n"] == 0


def test_tables_paired_rank_test_reports_both_levels():
    from common import tables
    r = tables.paired_rank_test(frame())
    assert set(["p", "cluster_p", "clusters", "a_closer"]) <= set(r)
