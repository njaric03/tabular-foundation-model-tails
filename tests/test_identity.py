# -*- coding: utf-8 -*-
"""Two identities the result files depend on: the dataset and the model.

Both went wrong the same way, by treating a label as the thing it names:
`218_house_8L` and `house_16H` are one target vector (rule 6), and `XGBoost` and `XGB`
are one model (rule 7).
"""
import glob
import os

import numpy as np
import pandas as pd
import pytest

from common import datasets, models, stats, tables

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = sorted(glob.glob(os.path.join(ROOT, "results", "*", "*.csv")))


# --------------------------------------------------------------- the dataset

def test_the_known_duplicate_target_is_still_detected():
    """If this pair ever stops being detected, the sign tests silently regain a
    free unit they are not entitled to."""
    dups = datasets.duplicate_targets()
    assert dups, "no duplicate targets found; the fingerprint file lost its hashes"
    assert ["218_house_8L", "house_16H"] in dups.values()


def test_a_dataset_with_no_fingerprint_is_its_own_group():
    assert datasets.target_group("a name that was never loaded") == \
        "a name that was never loaded"


def test_target_clustering_removes_the_duplicated_unit():
    d = pd.DataFrame([dict(dataset=n, scale_share=1.0, shape_share=0.5)
                      for n in ["218_house_8L", "house_16H", "diamonds"]])
    by_name = stats.paired_by_cluster(d, "scale_share", "shape_share")
    by_target = stats.paired_by_target(d, "scale_share", "shape_share")
    assert by_name["clusters"] == 3
    assert by_target["clusters"] == 2


def test_the_real_dissociation_loses_exactly_one_unit():
    """The number the thesis quotes: 9 of 9 datasets is 8 of 8 target vectors."""
    path = os.path.join(ROOT, "results", "h1_shape_vs_scale", "dissociation_real.csv")
    d = pd.read_csv(path)
    by_name = stats.paired_by_cluster(d, "scale_share", "shape_share_residual")
    by_target = stats.paired_by_target(d, "scale_share", "shape_share_residual")
    assert (by_name["cluster_wins"], by_name["clusters"]) == (9, 9)
    assert (by_target["cluster_wins"], by_target["clusters"]) == (8, 8)
    assert by_target["cluster_p"] == pytest.approx(0.5 ** 8, rel=1e-6)


# ----------------------------------------------------------------- the model

def test_parse_list_canonicalises_every_spelling():
    assert models.parse_list("GBM, XGBoost ,TabICL") == ["GBM", "XGB", "TabICLv2"]


def test_parse_list_refuses_an_unknown_name():
    with pytest.raises(ValueError):
        models.parse_list("GBM,NoSuchModel")


def test_tabdpt_quantiles_refuse_an_ensemble_they_would_not_measure():
    """The bin head is read from one member; n_est=4 would be recorded but not run."""
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        models.quantiles("TabDPT", X, np.ones(10), X, seed=0, levels=[0.5], n_est=4)


def test_a_finetuned_tabicl_arm_is_a_model_name_and_refuses_a_missing_checkpoint():
    """A fine-tuned arm must never fall back to downloading the base checkpoint."""
    assert models.parse_list("TabICLv2-FT-A1-s1") == ["TabICLv2-FT-A1-s1"]
    pytest.importorskip("tabicl")
    missing = "TabICLv2-FT-A0-s2"
    path = os.path.join(ROOT, models.TABICL_FT_DIR, f"{missing}.ckpt")
    if os.path.exists(path):
        pytest.skip("the checkpoint exists here")
    with pytest.raises(FileNotFoundError):
        models.tabicl_regressor(0, 1, missing)


def test_the_rank_test_clusters_by_target_vector():
    d = pd.DataFrame([dict(dataset=n, scale_share=1.0, shape_share=0.5)
                      for n in ["218_house_8L", "house_16H", "diamonds"]])
    assert tables.paired_rank_test(d)["clusters"] == 2


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: os.path.basename(p))
def test_result_model_names_are_canonical_after_repair(path):
    """Every `model` value must be a supported name once the synonym map has
    been applied. A name that survives the map is one no script can reproduce."""
    d = pd.read_csv(path, low_memory=False)
    if "model" not in d.columns:
        pytest.skip("no model column")
    got = set(tables.canonical_models(d)["model"].dropna().unique())
    unknown = got - set(models.SUPPORTED) - KNOWN_UNRESOLVED
    assert not unknown, f"unresolvable model names: {sorted(unknown)}"


# `TabPFN-V2` appears in `mean_correction.csv`, written before provenance was recorded.
# Package 8.4.0 ships v2.5 and v2.6 and the rows do not say which was loaded, so the name
# stays unresolved until the run is repeated.
KNOWN_UNRESOLVED = {"TabPFN-V2"}
