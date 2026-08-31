# -*- coding: utf-8 -*-
"""One narrow warning filter, replacing `filterwarnings("ignore")` in 31 files.

The blanket ignore was in every script, and it is on record as having hidden a
real bug for months: `gpd_quantile` divided by zero at xi = 0 in half the copies
and the RuntimeWarning that was the only sign of it went to /dev/null
(`common/metrics.py`, `common/generator.py`).

The distinction that matters is who raised the warning:

  * UserWarning, FutureWarning, DeprecationWarning come from the model packages
    and from sklearn, thousands of times per run, and say nothing about the
    measurement. Silenced.
  * RuntimeWarning is numeric: overflow, divide-by-zero, invalid value in a
    reduction. That is the class that hides a wrong number. Shown once per
    location, so it stays visible without flooding the log.

    from common import quiet
    quiet.silence()

With TFM_STRICT=1 a RuntimeWarning becomes an exception, which is how a script
should be run once after any change to an estimator.
"""
from __future__ import annotations

import os
import warnings

# Everything here is package noise, not a statement about the measurement.
NOISE = (UserWarning, FutureWarning, DeprecationWarning,
         PendingDeprecationWarning, ImportWarning, ResourceWarning)


def silence(strict: bool | None = None) -> None:
    """Silence package noise, keep numeric warnings visible."""
    if strict is None:
        strict = os.environ.get("TFM_STRICT", "") not in ("", "0", "false")
    for category in NOISE:
        warnings.filterwarnings("ignore", category=category)
    # The numeric class. "once" per location: visible, not a flood.
    warnings.filterwarnings("error" if strict else "once", category=RuntimeWarning)
