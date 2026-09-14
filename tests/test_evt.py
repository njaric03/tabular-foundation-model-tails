# -*- coding: utf-8 -*-
"""The EVT graft in `common/evt.py`."""
import numpy as np
import pytest

from common import evt

LEVELS = np.asarray(evt.LEVELS)


def test_scores_read_the_full_grid_and_the_right_columns():
    """With the 3-level default grid the scores raised on a broadcast error, and
    `pinball_at(q, y, 0.999)` would have scored the column of Q(0.03)."""
    q = np.tile(LEVELS, (200, 1))                 # quantile function of U(0, 1)
    y = np.random.default_rng(0).random(200)
    m = evt.scores(q, y, 0.9)
    assert all(np.isfinite(m[k]) for k in ("crps", "twcrps", "pinball99", "pinball999"))
    d = y - 0.999
    assert m["pinball999"] == pytest.approx(float(np.mean(np.where(d >= 0, 0.999 * d, -0.001 * d))))
    assert abs(m["mean"] - 0.5) < 0.01


def test_splice_keeps_the_body_and_stays_monotone():
    q = np.tile(LEVELS, (10, 1))
    s = evt.splice(q, np.full(10, 0.5), xi_hat=0.3, sigma_hat=0.2)
    body = LEVELS <= evt.ALPHA0
    assert np.array_equal(s[:, body], q[:, body])
    assert np.all(np.diff(s, axis=1) >= 0)


def test_splice_without_a_fit_returns_the_model():
    q = np.tile(LEVELS, (10, 1))
    assert np.array_equal(evt.splice(q, np.full(10, 0.5), np.nan, np.nan), q)


def test_residuals_drop_degenerate_medians_rather_than_floor_them():
    rng = np.random.default_rng(1)
    y = rng.pareto(2.0, 1000) + 1
    med = np.full(1000, 1.0)
    med[:5] = -0.5
    z = evt.residuals(y, med)
    assert len(z) == 995 and z.max() < 1e6
