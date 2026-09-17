# -*- coding: utf-8 -*-
"""The generator, and its parity with the formula every published run used.

The oracle control of H1 is valid only while it draws from exactly the generator the
models are given. The reference below is the historical inline formula; if
`common/generator.py` stops reproducing it, every published number changes.
"""
import numpy as np
import pytest
from scipy.stats import norm

from common import generator


def reference(n, seed, d=5):
    """The inline implementation the scripts used to carry, verbatim."""
    rng = np.random.default_rng(seed)
    W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])
    X = rng.normal(size=(n, d))
    s = np.exp(0.6 * (X @ W) / np.linalg.norm(W))
    xi = 0.15 + (0.90 - 0.15) * norm.cdf(X[:, 4])
    U = rng.random(n)
    y = s * ((1 - U) ** (-xi) - 1) / xi
    return X, y, s, xi


@pytest.mark.parametrize("n", [500, 2000])
def test_generator_reproduces_the_historical_inline_formula(n):
    X, y, s, xi = reference(n, seed=7000)
    p = generator.gpd(n, np.random.default_rng(7000), xi=generator.XI_OF_X)
    assert np.array_equal(p.X, X)
    # The shared version guards its denominator, which reorders one division and
    # moves the last bit: agreement is to 1e-15 relative, not bit for bit.
    assert np.allclose(p.y, y, rtol=1e-15, atol=0)
    assert np.array_equal(p.s, s)
    assert np.array_equal(p.xi, xi)


def test_the_shape_covariate_carries_no_weight_in_the_scale():
    """x4 drives xi and nothing else. That separation is the whole design."""
    assert generator.W[4] == 0.0
    assert generator.W[3] == 0.0
    p = generator.gpd(20_000, np.random.default_rng(0), xi=generator.XI_OF_X)
    assert abs(np.corrcoef(p.X[:, 4], np.log(p.s))[0, 1]) < 0.03


def test_xi_of_x_spans_the_declared_range():
    p = generator.gpd(20_000, np.random.default_rng(0), xi=generator.XI_OF_X)
    assert generator.XI_LO <= p.xi.min() < p.xi.max() <= generator.XI_HI
    assert p.xi.mean() == pytest.approx((generator.XI_LO + generator.XI_HI) / 2, abs=0.02)


def test_constant_xi_is_broadcast_and_clipping_only_caps_the_top():
    a = generator.gpd(1000, np.random.default_rng(3), xi=0.7, clip=False)
    b = generator.gpd(1000, np.random.default_rng(3), xi=0.7, clip=True)
    assert np.all(a.xi == 0.7)
    assert b.y.max() <= a.y.max()
    assert np.allclose(np.median(a.y), np.median(b.y))


def test_true_mean_is_infinite_at_xi_one_and_finite_below():
    p = generator.gpd(10, np.random.default_rng(0), xi=1.0)
    assert np.all(np.isinf(generator.true_mean(p)))
    p = generator.gpd(10, np.random.default_rng(0), xi=0.5)
    assert np.allclose(generator.true_mean(p), p.s / 0.5)


def test_the_families_share_x_and_u_and_match_their_quantiles():
    """With one rng every family draws the same X and U; only the map from U to y differs,
    so y / s has the same order in every family whose map is monotone (all but the folded
    t, |T(U)|), and each follows its own quantile function."""
    draws = {f: generator.family(f, 20_000, np.random.default_rng(9), xi=0.7)
             for f in generator.FAMILIES}
    ref = draws["gpd"]
    for f, p in draws.items():
        assert np.array_equal(p.X, ref.X) and np.array_equal(p.s, ref.s)
        if f != "studentt":
            assert np.array_equal(np.argsort(p.y / p.s), np.argsort(ref.y / ref.s))
        for a in (0.5, 0.9):
            assert np.quantile(p.y / p.s, a) == pytest.approx(
                generator.family_quantile(f, a, 0.7, 1.0), rel=0.03)
