# -*- coding: utf-8 -*-
"""Environment knobs, typed, with a record of which ones the process read.

    from common import env

    SEEDS = env.seeds(env.integer("SEEDS", 20))
    XI = env.floats("XI", [0.3, 0.7])
    MODELS = env.models("GBM,TabICLv2,TabPFN-V3")

Every name read lands in `READ`. `tests/test_scripts_import.py` checks that set against
`provenance.ENV_KNOBS`, so a knob cannot reach a script without reaching provenance too.
"""
from __future__ import annotations

import os

READ: set[str] = set()

# Every comparison shares these seeds: 7000, 8000, 9000, ...
SEED_BASE, SEED_STEP = 7000, 1000


def text(name: str, default: str) -> str:
    """The raw value, or `default` when the variable is unset or empty."""
    READ.add(name)
    value = os.environ.get(name, "")
    return value if value != "" else default


def integer(name: str, default: int) -> int:
    return int(text(name, str(default)))


def number(name: str, default: float) -> float:
    return float(text(name, repr(default)))


def names(name: str, default) -> list[str]:
    """A comma-separated list; `default` is a list or a comma-separated string."""
    if not isinstance(default, str):
        default = ",".join(str(v) for v in default)
    return [v.strip() for v in text(name, default).split(",") if v.strip()]


def floats(name: str, default) -> list[float]:
    return [float(v) for v in names(name, default)]


def integers(name: str, default) -> list[int]:
    return [int(v) for v in names(name, default)]


def models(default: str, name: str = "MODELS") -> list[str]:
    """Model names under their canonical spelling (rule 7)."""
    from common import models as _models
    return _models.parse_list(text(name, default))


def flag(name: str) -> bool:
    """True unless the variable is unset, empty, 0 or false."""
    READ.add(name)
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false")


def seeds(n: int) -> list[int]:
    return [SEED_BASE + SEED_STEP * i for i in range(n)]
