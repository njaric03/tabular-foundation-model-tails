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
