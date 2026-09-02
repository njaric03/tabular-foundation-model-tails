# -*- coding: utf-8 -*-
"""Property tests for the estimators in `common/metrics.py`.

Why these exist. Three numerical bugs are on record in this repository, all in
this module or its 19 former copies, and all found by reading rather than by
running:

  * the Hill estimator used the smallest INCLUDED order statistic as its
    threshold, biasing xi low by about xi/k;
  * `gpd_quantile` divided by zero at xi = 0 in half the copies, and the
    RuntimeWarning went into the repo-wide `filterwarnings("ignore")`;
  * `int(0.999 * 100) == 99`, so a column for 0.99 was overwritten by 0.999.

Every one of them is caught by a test that takes a second to run.
"""
import numpy as np
import pytest
from scipy.stats import genpareto

from common import metrics


# ------------------------------------------------------------ GPD quantiles

@pytest.mark.parametrize("xi", [-0.4, 0.0, 0.15, 0.5, 0.9, 1.5])
def test_gpd_quantile_matches_scipy(xi):
    a = np.array([0.01, 0.5, 0.9, 0.99, 0.999])
    got = metrics.gpd_quantile(a, np.full_like(a, xi))
    want = genpareto.ppf(a, c=xi)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_gpd_quantile_is_the_gumbel_limit_at_zero():
    a = np.array([0.1, 0.5, 0.99])
    assert np.allclose(metrics.gpd_quantile(a, np.zeros_like(a)), -np.log(1 - a))


def test_gpd_quantile_raises_no_numeric_warning_at_zero():
    """The exact bug the blanket warning filter used to hide."""
    with np.errstate(all="raise"), pytest.warns(None) if False else _no_warnings():
        metrics.gpd_quantile(np.array([0.5, 0.99]), np.array([0.0, 0.0]))


class _no_warnings:
    def __enter__(self):
        import warnings
        self.ctx = warnings.catch_warnings()
        self.ctx.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *a):
        return self.ctx.__exit__(*a)


# ------------------------------------------------------------- the inversions

@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.3, 0.7, 1.2, 3.0])
def test_xi_from_ratio_round_trips(xi):
    r = float(metrics.gpd_quantile(0.99, xi) / metrics.gpd_quantile(0.9, xi))
    assert metrics.xi_from_ratio(r) == pytest.approx(xi, abs=1e-6)


@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.3, 0.7, 1.2, 3.0])
def test_xi_from_residual_ratio_round_trips(xi):
    q = metrics.gpd_quantile(np.array([0.99, 0.9, 0.5]), np.full(3, xi))
    r = float((q[0] - q[2]) / (q[1] - q[2]))
    assert metrics.xi_from_residual_ratio(r) == pytest.approx(xi, abs=1e-6)


def test_residual_ratio_is_location_invariant_and_the_raw_one_is_not():
    """The reason the location-invariant estimator replaced the raw ratio."""
    xi, loc = 0.7, 5.0
    q = metrics.gpd_quantile(np.array([0.99, 0.9, 0.5]), np.full(3, xi)) + loc
    raw = metrics.xi_from_ratio(float(q[0] / q[1]))
    res = metrics.xi_from_residual_ratio(float((q[0] - q[2]) / (q[1] - q[2])))
    assert res == pytest.approx(xi, abs=1e-6)
    assert raw < xi - 0.2          # biased low, which is the documented failure


def test_xi_from_ratio_returns_nan_outside_the_bracket():
    assert np.isnan(metrics.xi_from_ratio(1e9))


# ------------------------------------------------------------------ Hill

def test_hill_recovers_a_known_pareto_index():
    rng = np.random.default_rng(0)
    xi = 0.5
    y = (1 - rng.random(200_000)) ** (-xi)      # Pareto with tail index xi
    assert metrics.hill(y, k_share=0.02) == pytest.approx(xi, abs=0.02)


def test_hill_threshold_is_the_k_plus_first_order_statistic():
    """The biased form uses y[-k] and drops about xi/k on average.

    One draw of that gap is a single order-statistic spacing and therefore
    noisy, so the size is checked over 400 samples and the direction on each.
    """
    rng = np.random.default_rng(1)
    xi, k = 0.9, 30
    gaps = []
    for _ in range(400):
        y = np.sort((1 - rng.random(300)) ** (-xi))
        correct = float(np.mean(np.log(y[-k:]) - np.log(y[-k - 1])))
        biased = float(np.mean(np.log(y[-k:]) - np.log(y[-k])))
        assert metrics.hill(y, k_share=0.0, k_min=k) == pytest.approx(correct)
        assert correct > biased
        gaps.append(correct - biased)
    assert float(np.mean(gaps)) == pytest.approx(xi / k, rel=0.25)


def test_hill_is_nan_when_the_sample_is_shorter_than_k():
    assert np.isnan(metrics.hill(np.array([1.0, 2.0, 3.0])))


# --------------------------------------------------------- truncated mean

@pytest.mark.parametrize("xi", [0.2, 0.5, 0.9])
def test_truncated_mean_matches_numeric_integration(xi):
    t = 8.0
    grid = np.linspace(0, t, 400_001)
    pdf = genpareto.pdf(grid, c=xi)
    # numpy 2 renamed trapz to trapezoid; the venvs run different majors.
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    want = trapz(grid * pdf, grid) / trapz(pdf, grid)
    assert float(metrics.truncated_mean_std(t, xi)) == pytest.approx(want, rel=1e-3)


# --------------------------------------------------------------- sd shift

def test_sd_shift_is_one_without_an_outlier_and_large_with_one():
    x = np.arange(100.0)
    assert metrics.sd_shift(x) == pytest.approx(np.std(x) / np.std(x[:-1]), rel=1e-12)
    assert metrics.sd_shift(np.append(x, 1e6)) > 50


# --------------------------------------------------------- scoring rules

def test_pinball_matches_the_definition():
    q = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2.5])
    got = metrics.pinball_all(q, y, [0.5, 0.9, 0.99])
    # y is below the 0.99 quantile, so that term is (alpha - 1) * (y - q) > 0.
    want = np.array([[0.5 * 1.5, 0.9 * 0.5, (0.99 - 1) * (2.5 - 3.0)]])
    assert np.allclose(got, want)


def test_pinball_at_selects_by_level_not_by_a_rounded_key():
    """int(0.999 * 100) == 99 is how a 0.99 column was overwritten once."""
    q = np.tile(np.array([[1.0, 2.0, 3.0, 4.0]]), (5, 1))
    y = np.full(5, 2.5)
    levels = [0.5, 0.9, 0.99, 0.999]
    a = metrics.pinball_at(q, y, 0.99, levels)
    b = metrics.pinball_at(q, y, 0.999, levels)
    assert a != b


def test_twcrps_ignores_everything_below_the_threshold():
    q = np.array([[1.0, 2.0, 3.0]])
    y = np.array([2.0])
    assert metrics.twcrps(q, y, threshold=10.0) == pytest.approx(0.0)
    assert metrics.twcrps(q, y, threshold=0.0) == pytest.approx(metrics.crps(q, y))


def test_mean_from_quantiles_recovers_the_mean_of_a_uniform():
    levels = list(np.linspace(0.01, 0.99, 99))
    q = np.array([levels])                      # quantile function of U(0, 1)
    assert float(metrics.mean_from_quantiles(q, levels)[0]) == pytest.approx(0.5, abs=0.01)


# -------------------------------------------------------------- guards

def test_share_tracked_refuses_a_reference_gradient_that_is_too_small():
    assert np.isnan(metrics.share_tracked(0.0, 1.0, 0.0, 0.01))
    assert metrics.share_tracked(0.0, 0.5, 0.0, 1.0) == pytest.approx(0.5)


def test_rel_change_skips_zero_denominators_instead_of_returning_inf():
    before = np.array([0.0, 1.0, 2.0])
    after = np.array([5.0, 2.0, 4.0])
    assert metrics.rel_change(before, after) == pytest.approx(1.0)
