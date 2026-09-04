# -*- coding: utf-8 -*-
"""What produced a row: package versions, commit, interpreter, env knobs.

The gap this closes. `findings/NALAZI.md` section 2.1 is a claim about specific
package versions -- TabPFN 8.4.0, TabICL 2.1.1, TabDPT 1.2.0, TabFM 1.0.1,
EXAONE-Tabular 1.0.0 -- and the whole of H2 is a measured consequence of what
those versions do to the target. Until now no result row said which version, or
even which of the three virtualenvs, produced it. Six months later, at the
defence, that is not reconstructible.

Why a sidecar file and not columns. Adding provenance columns to every result
CSV would change every header, and `append.write` refuses to append to a file
whose header moved -- correctly, that guard exists because shifted columns cost
measurements once already. So provenance goes to one shared log,
`results/provenance.csv`, joined to a result by its file name and the time the
rows were written.

Nothing calls this by hand: `common/append.py` records once per process the
first time it writes to an output. A knob that is not recorded is a knob that
cannot be reconstructed (rule 2 of RULES.md), and a package version is a knob
the environment sets rather than the script.
"""
from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Model packages, plus the numerical stack whose major version differs between
# the three virtualenvs (numpy 1 and 2 are both in use; see
# metrics.mean_from_quantiles).
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "tabpfn",
            "tabicl", "tabdpt", "tabfm", "exaone-tabular", "xgboost", "catboost"]

# Knobs read from the environment by the experiment scripts. Every one of them
# changes a measurement, and several are not columns in every output.
ENV_KNOBS = ["MODELS", "MODEL", "SEEDS", "N_EST", "XI", "DOSES", "DOSE_MODE",
             "SD_SHIFTS", "DATASETS", "POSITION", "VARIANTS", "LEVELS",
             "MEMBERS", "N_GROUPS", "N_TRAIN", "OUTPUT", "TFM_STRICT"]

COLUMNS = (["utc", "output", "script", "git_sha", "git_dirty", "python",
            "platform", "venv"]
           + [f"v_{p.replace('-', '_')}" for p in PACKAGES]
           + [f"env_{k}" for k in ENV_KNOBS])

_recorded: set[str] = set()


def _version(pkg: str) -> str:
    """Installed version without importing the package."""
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return ""


def _git() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return sha.stdout.strip(), "yes" if dirty.stdout.strip() else "no"
    except Exception:
        return "", ""


def row(output: str) -> dict:
    """The provenance record for the current process, as a dict."""
    sha, dirty = _git()
    r = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             output=str(output),
             script=Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "",
             git_sha=sha, git_dirty=dirty,
             python=platform.python_version(),
             platform=f"{platform.system()}-{platform.machine()}",
             venv=Path(sys.prefix).name)
    for p in PACKAGES:
        r[f"v_{p.replace('-', '_')}"] = _version(p)
    for k in ENV_KNOBS:
        r[f"env_{k}"] = os.environ.get(k, "")
    return r


def record(output: str) -> None:
    """Append one provenance row, at most once per output per process."""
    key = str(output)
    if key in _recorded:
        return
    _recorded.add(key)
    from common import paths
    path = paths.result("provenance.csv")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row(output))
    except Exception as e:
        # Provenance must never take a measurement down with it.
        print(f"[provenance] not recorded: {type(e).__name__}: {e}", flush=True)
