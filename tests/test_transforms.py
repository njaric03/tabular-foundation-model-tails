# -*- coding: utf-8 -*-
"""The context transforms of `common/transforms.py`: what each one changes, and that the
ones meant to be inverted come back exactly."""
import numpy as np
import pytest

from common import metrics, transforms

RANK_ARMS = ["rank_emp", "rank_gpd", "rank_gpd_trim", "rank_exp"]


def _pareto(n, xi, seed):
    return np.random.default_rng(seed).pareto(1 / xi, n) + 1.0


def test_raw_changes_nothing():
    y = _pareto(2000, 0.7, 0)
    z, back, n = transforms.transform("raw", 0.0, y)
    assert n == 0 and np.array_equal(z, y) and np.array_equal(back(y), y)


def test_clip_caps_at_median_plus_c_robust_sd():
    y = _pareto(2000, 0.7, 1)
    cap = np.median(y) + 50 * transforms.robust_sd(y)
    z, _, n = transforms.transform("clip", 50.0, y)
    assert z.max() <= cap + 1e-12 and n == int((y > cap).sum())
    assert np.array_equal(z[y <= cap], y[y <= cap])


def test_log_and_tail_log_invert_exactly():
    y = _pareto(2000, 0.9, 2)
    for arm, c in (("log", 0.0), ("tail_log", 50.0)):
        z, back, _ = transforms.transform(arm, c, y)
        assert np.allclose(back(z), y, rtol=1e-9)


def test_tail_log_is_identity_below_the_cap_and_monotone():
    y = _pareto(2000, 0.9, 3)
    cap = np.median(y) + 50 * transforms.robust_sd(y)
    z, _, _ = transforms.transform("tail_log", 50.0, y)
    assert np.array_equal(z[y <= cap], y[y <= cap])
    assert np.all(np.diff(z[np.argsort(y)]) >= 0)


def test_evt_trim_rarely_touches_a_clean_pareto_sample():
    touched = sum(transforms.evt_trim(_pareto(2000, 0.7, s))[1] > 0 for s in range(200))
    assert touched <= 10          # alpha = 0.01 by construction; 5% is generous


def test_evt_trim_pulls_a_gross_error_down_to_the_next_value():
    y = _pareto(2000, 0.5, 4)
    second = np.sort(y)[-1]
    z, n = transforms.evt_trim(np.append(y, second * 1e4))
    assert n == 1 and z.max() == second


def test_sd_cap_leaves_a_context_already_below_tau_untouched():
    """Where there is nothing to repair the rule costs nothing, by construction."""
    y = _pareto(2000, 0.2, 10)
    assert metrics.sd_shift(y) <= transforms.TAU
    z, _, n = transforms.transform("sd_cap", transforms.TAU, y)
    assert n == 0 and np.array_equal(z, y)


def test_sd_cap_brings_the_shift_down_to_tau():
    for seed in range(5):
        y = _pareto(2000, 0.9, 100 + seed)
        cap, n = transforms.sd_cap(y)
        assert metrics.sd_shift(np.minimum(y, cap)) <= transforms.TAU + 1e-9
        assert n == int((y > cap).sum())


def test_sd_cap_does_not_swallow_a_tau_of_zero():
    """`c or TAU` once turned a caller's 0.0 into 1.2 and recorded clip_c = 0.0."""
    y = _pareto(2000, 0.9, 11)
    strict, _, n_strict = transforms.transform("sd_cap", 0.0, y)
    loose, _, n_loose = transforms.transform("sd_cap", transforms.TAU, y)
    assert n_strict > n_loose and not np.array_equal(strict, loose)


def test_rank_arms_report_every_row_as_changed():
    y = _pareto(2000, 0.7, 12)
    for arm in RANK_ARMS:
        assert transforms.transform(arm, 0.0, y)[2] == len(y), arm


def test_rank_emp_inverts_exactly():
    y = _pareto(2000, 0.9, 13)
    z, back, _ = transforms.transform("rank_emp", 0.0, y)
    assert np.allclose(back(z), y, rtol=1e-9)


def test_rank_arms_invert_exactly_below_the_gpd_threshold():
    """Below u the map is the empirical inverse; above it the fitted tail is imposed."""
    y = _pareto(2000, 0.9, 14)
    below = y <= np.quantile(y, transforms.RANK_U_SHARE)
    for arm in RANK_ARMS:
        z, back, _ = transforms.transform(arm, 0.0, y)
        with np.errstate(over="ignore", invalid="ignore"):
            rt = back(z)
        assert np.allclose(rt[below], y[below], rtol=1e-9), arm


def test_rank_arms_are_monotone_across_the_splice():
    """The seam once stepped down by 1.5 times the local order-statistic gap."""
    for xi in (0.7, 0.9):
        for seed in (15, 16, 17):
            y = _pareto(2000, xi, seed)
            for arm in RANK_ARMS:
                z, back, _ = transforms.transform(arm, 0.0, y)
                grid = np.linspace(z.min() - 2.0, z.max() + 3.0, 100_000)
                with np.errstate(over="ignore", invalid="ignore"):
                    b = back(grid)
                assert np.all(np.diff(b) >= -1e-12), f"{arm}, xi={xi}, seed={seed}"


def test_the_rank_transform_removes_the_leverage_it_exists_to_remove():
    y = _pareto(2000, 0.9, 19)
    y = np.append(y, y.max() * 1e6)
    for arm in RANK_ARMS:
        assert metrics.sd_shift(transforms.transform(arm, 0.0, y)[0]) < 1.05, arm


def test_an_unknown_score_scale_or_arm_is_refused():
    y = _pareto(200, 0.7, 20)
    with pytest.raises(ValueError):
        transforms.rank_map(y, scores="student")
    with pytest.raises(ValueError):
        transforms.transform("no_such_arm", 0.0, y)
