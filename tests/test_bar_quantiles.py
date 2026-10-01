# -*- coding: utf-8 -*-
"""The inverse CDF that reads quantiles out of LimiX-2's bar head.

The package decodes only a mean, so `common/adapters/limix.py` inverts the distribution
itself. These cases pin that inversion to the law the head defines: linear inside a
bucket, half-normal in the outer two with the median at the bucket width. The example
is the one PriorLabs/TabPFN#1215 uses for the same distribution.
"""
import numpy as np
import pytest

from common.adapters.limix import bar_quantiles

BORDERS = np.array([-2.0, -1.0, 1.0, 2.0])
LOGP = np.log(np.array([[0.25, 0.5, 0.25]]))


def test_interior_levels_are_linear_inside_the_bucket():
    q = bar_quantiles(LOGP, BORDERS, [0.25, 0.5, 0.625])[0]
    assert q == pytest.approx([-1.0, 0.0, 0.5])


def test_outer_buckets_put_their_median_at_one_width():
    q = bar_quantiles(LOGP, BORDERS, [0.125, 0.875])[0]
    assert q == pytest.approx([-2.0, 2.0])


def test_the_tail_is_not_stopped_at_the_last_border():
    q = bar_quantiles(LOGP, BORDERS, [0.999])[0]
    assert q[0] > BORDERS[-1]


def test_the_mean_of_the_quantiles_is_the_mean_of_the_head():
    """The package's own `mean` places the outer buckets at the half-normal mean."""
    rng = np.random.default_rng(0)
    borders = np.concatenate([[-30.0], np.linspace(-5, 5, 41), [30.0]])
    p = rng.dirichlet(np.ones(len(borders) - 1), size=3)
    w = np.diff(borders)
    mid = borders[:-1] + w / 2
    s0, s1 = w[0] / 0.6744897501960817, w[-1] / 0.6744897501960817
    mid[0], mid[-1] = borders[1] - s0 * np.sqrt(2 / np.pi), borders[-2] + s1 * np.sqrt(2 / np.pi)
    taus = (np.arange(1, 200001) - 0.5) / 200000
    q = bar_quantiles(np.log(p), borders, taus)
    assert q.mean(axis=1) == pytest.approx(p @ mid, rel=2e-3)
