# -*- coding: utf-8 -*-
"""A narrow warning filter for every script.

Package warnings (UserWarning, FutureWarning, DeprecationWarning and the like) are
silenced. RuntimeWarning, the class that signals a wrong number, stays visible once per
location and becomes an error under TFM_STRICT=1. A blanket `filterwarnings("ignore")`
once hid a divide-by-zero in the tail-index inversion for months.
"""
from __future__ import annotations

import warnings

from common import env

NOISE = (UserWarning, FutureWarning, DeprecationWarning,
         PendingDeprecationWarning, ImportWarning, ResourceWarning)


def silence(strict: bool | None = None) -> None:
    """Silence package noise and keep numeric warnings visible."""
    if strict is None:
        strict = env.flag("TFM_STRICT")
    for category in NOISE:
        warnings.filterwarnings("ignore", category=category)
    warnings.filterwarnings("error" if strict else "once", category=RuntimeWarning)
