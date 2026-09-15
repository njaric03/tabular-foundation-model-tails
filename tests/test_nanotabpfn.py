# -*- coding: utf-8 -*-
"""The pre-training arms and the nanoTabPFN adapter.

The prior treatments are plain numpy and always run. The adapter test needs torch, pfns
and tfmplayground, which live in `venv-tfmp` only, and the released checkpoint under
`.cache/tfmp/released/`; it is skipped elsewhere.
"""
import numpy as np
import pytest

from common import metrics, prior_arms, transforms
from common.adapters import nanotabpfn


def test_contamination_hits_the_target_shift_and_leaves_the_test_rows():
    rng = np.random.default_rng(0)
    y = rng.lognormal(size=1024)
    split = 700
    for target in (4.0, 20.0, 50.0):
        others = y[:split - 1]
        c = prior_arms.value_for_shift(others, target)
        assert prior_arms.shift_of_row(np.append(others, c), split - 1) == \
            pytest.approx(target, rel=1e-9)
    new, i, shift = prior_arms.contaminate(y, split, rng)
    assert i < split
    assert prior_arms.SHIFT_LO - 1e-9 <= shift <= prior_arms.SHIFT_HI + 1e-9
    np.testing.assert_array_equal(new[split:], y[split:])
    assert (np.delete(new, i) == np.delete(y, i)).all()


def test_the_contaminated_row_is_the_largest_and_matches_metrics_sd_shift():
    """Placed above the mean at a shift of 4 or more, the row is the maximum, so the
    repo's own measure, which drops the largest value, reads the same shift."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=600)
    new, i, shift = prior_arms.contaminate(y, 600, rng)
    assert new[i] == new.max()
    assert metrics.sd_shift(new) == pytest.approx(shift, rel=1e-9)


def test_heavy_tail_reads_each_rank_through_its_own_gpd():
    """A row's value is the GPD quantile at its within-table rank level, with the shape of
    its own feature value. With a shape that varies by row the order is not kept, so the
    test checks the map itself."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1024, 4))
    y = X[:, 0] + rng.normal(size=1024)
    new, j, xi = prior_arms.heavy_tail(X, y, 4, rng)
    u = (np.argsort(np.argsort(y, kind="stable"), kind="stable") + 0.5) / len(y)
    np.testing.assert_allclose(new, metrics.gpd_quantile(u, xi), rtol=1e-12)
    z = (X[:, j] - X[:, j].mean()) / X[:, j].std()
    assert abs(np.corrcoef(xi, z)[0, 1]) > 0.8
    assert 0 <= j < 4
    assert prior_arms.XI_LO <= xi.min() and xi.max() <= prior_arms.XI_HI
    assert np.isfinite(new).all() and (new > 0).all()


def test_robust_encoding_is_the_repo_robust_sd():
    y = np.random.default_rng(3).lognormal(size=500)
    centre, scale = nanotabpfn.encode_stats(y, "robust")
    assert centre == pytest.approx(np.median(y))
    assert scale == pytest.approx(transforms.robust_sd(y) + nanotabpfn.EPS)
    centre, scale = nanotabpfn.encode_stats(y, "meansd")
    assert scale == pytest.approx(y.std(ddof=1) + nanotabpfn.EPS)


def test_the_adapter_refuses_an_ensemble_and_an_unknown_arm():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        nanotabpfn.quantiles("nanoTabPFN-A0B0-s1", X, np.ones(10), X, seed=0,
                             levels=[0.5], n_est=4)
    with pytest.raises(ValueError):
        nanotabpfn.checkpoint_path("nanoTabPFN-A2B0-s1")


def test_the_adapter_returns_monotone_quantiles():
    pytest.importorskip("torch")
    pytest.importorskip("pfns")
    pytest.importorskip("tfmplayground")
    path = nanotabpfn.paths.ROOT / ".cache" / "tfmp" / "released" / "nanotabpfn_regressor.pth"
    if not path.exists():
        pytest.skip("released checkpoint not downloaded")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3))
    y = 2.0 * X[:, 0] + rng.normal(size=300)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    q, borders = nanotabpfn.predict(path, X[:250], y[:250], X[250:], levels, dev="cpu")
    assert q.shape == (50, len(levels))
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= -1e-6).all()
    assert (np.diff(borders) >= 0).all()
    # The median follows the signal.
    assert np.corrcoef(q[:, 2], 2.0 * X[250:, 0])[0, 1] > 0.5
