# -*- coding: utf-8 -*-
"""What produced a result row: package versions, commit, interpreter and env knobs.

One row per (process, output) goes to `results/provenance.csv`, joined to a result by
file name and time. `append.write` calls `record` on its first write, so no script has
to remember it. A sidecar file keeps every result header unchanged.
"""
from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from common import files, paths

# Model packages plus the numerical stack, whose major versions differ between venvs.
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "tabpfn",
            "tabicl", "tabdpt", "tabfm", "exaone-tabular", "xgboost", "catboost"]

# Distribution names that differ from the PACKAGES spelling. PACKAGES also names the
# column (`v_exaone_tabular`), and renaming a column would break the header, so the
# lookup is redirected here instead.
DISTRIBUTIONS = {"exaone-tabular": "exaonetabular"}

# Every environment variable that changes a measurement. The order is the column order
# of the file on disk: append new names at the end.
ENV_KNOBS = ["MODELS", "MODEL", "SEEDS", "N_EST", "XI", "DOSES", "DOSE_MODE",
             "SD_SHIFTS", "DATASETS", "POSITION", "VARIANTS", "LEVELS",
             "MEMBERS", "N_GROUPS", "N_TRAIN", "OUTPUT", "TFM_STRICT",
             "ARM", "D", "PARTS", "DOSE", "FAMILIES", "GENERATOR", "K",
             "N_PER_BIN", "K_SHARE", "LOG_SCALE", "MAX_ATTEMPTS", "MASS",
             "N_BOOT", "N_FIT", "N_GRID", "N_PERM", "N_PER_GROUP", "N_TEST",
             "OUTPUT1", "OUTPUT2", "N_REPEATS", "TRANSFORMS", "N_SUBSAMPLES",
             # read in common/ rather than in a script
             "DTYPE", "PER_LEVEL", "SELECTED", "CATEGORICAL",
             # TabPFN's own setting, which picks the checkpoint when no model_path is given
             "TABPFN_MODEL_VERSION",
             "CLIP_C", "U_SHARES", "REPEATS", "N_CALIB",
             # TabPFN's CPU output depends on the thread count bit for bit: the same
             # generator cells give an implied xi of 0.6516 at 6 threads and 0.7003 at 3.
             # An unset value means the library default, which depends on the machine.
             "OMP_NUM_THREADS", "MKL_NUM_THREADS"]

COLUMNS = (["utc", "output", "script", "git_sha", "git_dirty", "python",
            "platform", "venv"]
           + [f"v_{p.replace('-', '_')}" for p in PACKAGES]
           + [f"env_{k}" for k in ENV_KNOBS])

_recorded: set[str] = set()


def _version(pkg: str) -> str:
    """Installed version without importing the package, or '' when it is absent."""
    try:
        return version(DISTRIBUTIONS.get(pkg, pkg))
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
    """The provenance record of the current process."""
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


def _widen(path: Path) -> bool:
    """Bring the header on disk up to COLUMNS. Returns False when that is impossible.

    Only added columns can be reconciled, and old rows get blanks there. A column that
    disappeared from COLUMNS is refused, because appending anyway would shift every
    value after it.
    """
    old = files.csv_header(path)
    if old is None or old == COLUMNS:
        return True
    dropped = [c for c in old if c not in COLUMNS]
    if dropped:
        print(f"[provenance] header lost columns {dropped}, so this row is NOT recorded. "
              f"Restore the names in COLUMNS, or move the file aside.", flush=True)
        return False
    n = files.rewrite_csv(path, COLUMNS)
    added = [c for c in COLUMNS if c not in old]
    print(f"[provenance] header now matches COLUMNS (added {added}), {n} rows kept",
          flush=True)
    return True


def record(output: str) -> None:
    """Append one provenance row, at most once per output per process. Never raises."""
    if str(output) in _recorded:
        return
    _recorded.add(str(output))
    path = paths.result("provenance.csv")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        r = row(output)
        # Every running measurement shares this file, so the header check and the
        # append happen under one lock.
        with files.FileLock(path):
            if not _widen(path):
                return
            new = files.csv_header(path) is None
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
                if new:
                    w.writeheader()
                w.writerow(r)
    except Exception as e:
        # Provenance must never take a measurement down with it.
        print(f"[provenance] not recorded: {type(e).__name__}: {e}", flush=True)
