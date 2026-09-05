# -*- coding: utf-8 -*-
"""Two identities the result files depend on: the dataset and the model.

Both were wrong in the repository at the same time, and both are the same
mistake: a label was treated as the thing it names. `218_house_8L` and
`house_16H` are two OpenML names for one target vector, so a sign test over
names counted nine free units where there are eight. `XGBoost` and `XGB` are two
spellings of one model, so a `groupby("model")` split one control in two. Rule 6
covers the first at the seed level; these tests carry it to the level above.
"""
import glob
import os

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


# `TabPFN-V2` appears once, in `mean_correction.csv`, written before
# `results/provenance.csv` existed. Package 8.4.0 ships v2.5 and v2.6 and the
# row does not say which was loaded, so mapping it to either would assert a
# checkpoint nobody can verify. It stays listed here, and unresolved, until the
# run is repeated. See `findings/NALAZI.md` section 9.
KNOWN_UNRESOLVED = {"TabPFN-V2"}
