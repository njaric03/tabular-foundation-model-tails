# -*- coding: utf-8 -*-
"""The context transforms of `clip_context.py`: what each one changes, and that
the ones meant to be inverted come back exactly."""
import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "experiments" / "h3_repair" / "clip_context.py"
_spec = importlib.util.spec_from_file_location("clip_context", _p)
CC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CC)


def _pareto(n, xi, seed):
    return np.random.default_rng(seed).pareto(1 / xi, n) + 1.0


def test_raw_changes_nothing():
    y = _pareto(2000, 0.7, 0)
    z, back, n = CC.transform("raw", 0.0, y)
    assert n == 0 and np.array_equal(z, y) and np.array_equal(back(y), y)


def test_clip_caps_at_median_plus_c_robust_sd():
    y = _pareto(2000, 0.7, 1)
    cap = np.median(y) + 50 * CC.robust_sd(y)
    z, _, n = CC.transform("clip", 50.0, y)
    assert z.max() <= cap + 1e-12 and n == int((y > cap).sum())
    assert np.array_equal(z[y <= cap], y[y <= cap])


def test_log_and_tail_log_invert_exactly():
    y = _pareto(2000, 0.9, 2)
    for arm, c in (("log", 0.0), ("tail_log", 50.0)):
        z, back, _ = CC.transform(arm, c, y)
        assert np.allclose(back(z), y, rtol=1e-9)


def test_tail_log_is_identity_below_the_cap_and_monotone():
    y = _pareto(2000, 0.9, 3)
    cap = np.median(y) + 50 * CC.robust_sd(y)
    z, _, _ = CC.transform("tail_log", 50.0, y)
    assert np.array_equal(z[y <= cap], y[y <= cap])
    o = np.argsort(y)
    assert np.all(np.diff(z[o]) >= 0)


def test_evt_trim_rarely_touches_a_clean_pareto_sample():
    touched = sum(CC.evt_trim(_pareto(2000, 0.7, s))[1] > 0 for s in range(200))
    assert touched <= 10          # alpha = 0.01 by construction; 5% is generous


def test_evt_trim_pulls_a_gross_error_down_to_the_next_value():
    y = _pareto(2000, 0.5, 4)
    second = np.sort(y)[-1]
    y = np.append(y, second * 1e4)
    z, n = CC.evt_trim(y)
    assert n == 1 and z.max() == second


# ------------------------------------------------ the derived cap, and the ranks
#
# Everything below covers the arms added in the third, fourth and fifth runs.
# They arrived with no tests, and the two properties this file already asserts for
# tail_log -- exact inversion and monotonicity -- are exactly the two the rank map
# claimed in its docstring and did not have.

RANK_ARMS = ["rank_emp", "rank_gpd", "rank_gpd_trim", "rank_exp"]


def test_sd_cap_leaves_a_context_already_below_tau_untouched():
    """The claim P9 rests on: where there is nothing to repair it costs nothing,
    by construction rather than by measurement."""
    y = _pareto(2000, 0.2, 10)
    assert CC.metrics.sd_shift(y) <= CC.TAU        # the premise of the test
    z, _, n = CC.transform("sd_cap", CC.TAU, y)
    assert n == 0 and np.array_equal(z, y)


def test_sd_cap_brings_the_shift_down_to_tau():
    for seed in range(5):
        y = _pareto(2000, 0.9, 100 + seed)
        cap, n = CC.sd_cap(y)
        capped = np.minimum(y, cap)
        assert CC.metrics.sd_shift(capped) <= CC.TAU + 1e-9
        assert n == int((y > cap).sum())


def test_sd_cap_does_not_swallow_a_tau_of_zero():
    """`c or TAU` turned a caller's 0.0 into 1.2 and recorded clip_c = 0.0."""
    y = _pareto(2000, 0.9, 11)
    strict, _, n_strict = CC.transform("sd_cap", 0.0, y)
    loose, _, n_loose = CC.transform("sd_cap", CC.TAU, y)
    assert n_strict > n_loose and not np.array_equal(strict, loose)


def test_rank_arms_report_every_row_as_changed():
    """`transform` promises how many context values the arm changed; the rank map
    rewrites all of them, and reporting 0 made it indistinguishable from raw."""
    y = _pareto(2000, 0.7, 12)
    for arm in RANK_ARMS:
        _, _, n = CC.transform(arm, 0.0, y)
        assert n == len(y), arm


def test_rank_emp_inverts_exactly():
    y = _pareto(2000, 0.9, 13)
    z, back, _ = CC.transform("rank_emp", 0.0, y)
    assert np.allclose(back(z), y, rtol=1e-9)


def test_rank_arms_invert_exactly_below_the_gpd_threshold():
    """Below u the map is the empirical inverse and must be exact. Above it the
    fitted tail is imposed on purpose, so only the body is asserted here."""
    y = _pareto(2000, 0.9, 14)
    u = np.quantile(y, CC.RANK_U_SHARE)
    for arm in RANK_ARMS:
        z, back, _ = CC.transform(arm, 0.0, y)
        with np.errstate(over="ignore", invalid="ignore"):
            rt = back(z)
        below = y <= u
        assert np.allclose(rt[below], y[below], rtol=1e-9), arm


def test_rank_arms_are_monotone_across_the_splice():
    """The seam between the empirical inverse and the Pareto tail stepped DOWN by
    1.5x the local order-statistic gap, because the two branches were handed
    different exceedance rates."""
    for xi in (0.7, 0.9):
        for seed in (15, 16, 17):
            y = _pareto(2000, xi, seed)
            for arm in RANK_ARMS:
                z, back, _ = CC.transform(arm, 0.0, y)
                grid = np.linspace(z.min() - 2.0, z.max() + 3.0, 100_000)
                with np.errstate(over="ignore", invalid="ignore"):
                    b = back(grid)
                assert np.all(np.diff(b) >= -1e-12), f"{arm}, xi={xi}, seed={seed}"


def test_back_transformed_borders_stay_sorted_for_searchsorted():
    """`one()` reads same_bar_90_99 off np.searchsorted(borders, ...), which is
    undefined unless back() preserves the order of the bar-distribution borders."""
    y = _pareto(2000, 0.9, 18)
    for arm in RANK_ARMS:
        z, back, _ = CC.transform(arm, 0.0, y)
        borders = np.linspace(z.min() - 2.0, z.max() + 2.0, 5000)
        with np.errstate(over="ignore", invalid="ignore"):
            b = back(borders)
        assert np.all(np.diff(b) >= -1e-12), arm


def test_the_rank_transform_removes_the_leverage_it_exists_to_remove():
    """R3: a rank cannot inflate an sd, whatever the injected row was."""
    y = _pareto(2000, 0.9, 19)
    y = np.append(y, y.max() * 1e6)
    for arm in RANK_ARMS:
        z, _, _ = CC.transform(arm, 0.0, y)
        assert CC.metrics.sd_shift(z) < 1.05, arm


def test_an_unknown_score_scale_is_refused():
    y = _pareto(200, 0.7, 20)
    try:
        CC.rank_gpd(y, scores="student")
    except ValueError:
        return
    raise AssertionError("an unknown score scale must not fall through silently")
