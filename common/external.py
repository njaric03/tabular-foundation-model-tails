# -*- coding: utf-8 -*-
"""Heavy-tailed tables that are not on OpenML, fetched once and cached.

The OpenML pool supplies one table with both leverage and a heavy tail, freMTPL2sev:
benchmark tables are curated. Heavy tails with covariates live in insurance severity,
health expenditure and online popularity, so the candidates come from there. Each entry
fixes the target and drops the columns that are outcomes of the same event (claim counts,
claim indicators, totals), all named before any table was loaded.
`experiments/h2_leverage/external_selection.py` decides which enter the pool.

`dataCar` is left out because it is the same portfolio as `ausprivauto0405`, and
`AutoBi` because 1340 rows do not fit a 2000 + 1000 split.

Converting an `.rda` needs `pyreadr`, which the model environments do not carry:

    venv-data/Scripts/python -m common.external      # fetch and convert, once
    datasets.load("beMTPL97")                        # any venv, reads the csv.gz

Raw files go to `.cache/external/raw/` and converted tables to `.cache/external/`, both
outside git. The sha256 of every raw file is recorded in `data/external_sources.json`
on the first fetch and checked on every later one.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile

import pandas as pd

from common import files, paths

CAS = "https://github.com/dutangc/CASdatasets/raw/master/data/"

MEPS_KEEP = ["AGE16X", "SEX", "RACEV1X", "REGION16", "MARRY16X", "EDUCYR", "HIDEG",
             "POVCAT16", "INSCOV16", "FAMSZE16", "RTHLTH53", "MNHLTH53", "HIBPDX",
             "CHDDX", "DIABDX", "CANCERDX", "ASTHDX", "ARTHDX", "STRKDX", "EMPHDX",
             "ADLHLP53", "EMPST53", "BMINDX53"]

REGISTRY = {
    "beMTPL97": dict(
        url=CAS + "beMTPL97.rda", kind="rda", target="average",
        drop=["id", "claim", "nclaims", "amount"],
        domain="motor third-party liability, Belgium 1997, average claim amount"),
    "ausprivauto0405": dict(
        url=CAS + "ausprivauto0405.rda", kind="rda", target="ClaimAmount",
        drop=["ClaimOcc", "ClaimNb"],
        domain="private motor, Australia 2004-05, claim amount"),
    "freMPL1": dict(
        url=CAS + "freMPL1.rda", kind="rda", target="ClaimAmount",
        drop=["ClaimInd", "RecordBeg", "RecordEnd"],
        domain="motor personal line, France, claim amount"),
    "norauto": dict(
        url=CAS + "norauto.rda", kind="rda", target="ClaimAmount",
        drop=["NbClaim"],
        domain="motor, Norway, claim amount"),
    "swmotorcycle": dict(
        url=CAS + "swmotorcycle.rda", kind="rda", target="ClaimAmount",
        drop=["ClaimNb"],
        domain="motorcycle, Sweden 1994-98, claim amount"),
    "AutoClaims": dict(
        url="https://cran.r-project.org/src/contrib/insuranceData_1.0.tar.gz",
        kind="rda-in-tar", member="insuranceData/data/AutoClaims.rda", target="PAID",
        drop=[],
        domain="automobile claims paid, US"),
    "MEPS_2016_TOTEXP": dict(
        url="https://meps.ahrq.gov/mepsweb/data_files/pufs/h192ssp.zip",
        kind="xport-in-zip", member="h192.ssp", target="TOTEXP16", keep=MEPS_KEEP,
        domain="total health expenditure per person, US 2016 (MEPS HC-192)"),
    "BlogFeedback": dict(
        url="https://archive.ics.uci.edu/static/public/304/blogfeedback.zip",
        kind="csv-in-zip", member="blogData_train.csv", target=-1, drop=[],
        domain="comments a blog post receives in the next 24 hours (UCI)"),
}

SOURCES = "external_sources.json"


def _root():
    return paths.ROOT / ".cache" / "external"


def _raw_path(name):
    return _root() / "raw" / REGISTRY[name]["url"].rsplit("/", 1)[-1]


def table_path(name):
    return _root() / f"{name}.csv.gz"


def _check_sha(name, blob: bytes) -> None:
    sha = hashlib.sha256(blob).hexdigest()
    path = paths.data(SOURCES)
    url = REGISTRY[name]["url"]
    with files.FileLock(path):
        known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if url not in known:
            known[url] = sha
            path.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
            print(f"  [external] sha256 recorded for {url.rsplit('/', 1)[-1]}: {sha[:16]}")
            return
    if known[url] != sha:
        raise RuntimeError(f"{url}: the file is not the one recorded before "
                           f"({known[url][:16]} then, {sha[:16]} now)")


def fetch(name) -> bytes:
    """The raw file, downloaded on first use and checked against its recorded hash."""
    raw = _raw_path(name)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(REGISTRY[name]["url"], timeout=600) as r:
            raw.write_bytes(r.read())
    blob = raw.read_bytes()
    _check_sha(name, blob)
    return blob


def _read(name, blob) -> pd.DataFrame:
    spec = REGISTRY[name]
    kind = spec["kind"]
    if kind in ("rda", "rda-in-tar"):
        import tempfile

        import pyreadr  # venv-data only
        if kind == "rda-in-tar":
            with tarfile.open(fileobj=io.BytesIO(blob)) as t:
                blob = t.extractfile(spec["member"]).read()
        with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as fh:
            fh.write(blob)
        return next(iter(pyreadr.read_r(fh.name).values()))
    if kind == "xport-in-zip":
        with zipfile.ZipFile(io.BytesIO(blob)) as z, z.open(spec["member"]) as f:
            df = pd.read_sas(io.BytesIO(f.read()), format="xport")
        df = df[spec["keep"] + [spec["target"]]]
        # pandas reads a SAS XPORT zero as 5.4e-79. Before 1.10.2026 these stayed in, so
        # 6933 persons with no expenditure passed `positive_only` as positive amounts.
        return df.mask(df.abs() < 1e-70, 0.0)
    if kind == "csv-in-zip":
        with zipfile.ZipFile(io.BytesIO(blob)) as z, z.open(spec["member"]) as f:
            df = pd.read_csv(f, header=None)
        df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] + ["target"]
        return df
    raise ValueError(kind)


def convert(name) -> None:
    """Raw file to `<name>.csv.gz`, with the target in the column `__target__`."""
    spec = REGISTRY[name]
    df = _read(name, fetch(name))
    target = df.columns[spec["target"]] if isinstance(spec["target"], int) else spec["target"]
    y = pd.to_numeric(df.pop(target), errors="coerce")
    df = df.drop(columns=[c for c in spec.get("drop", []) if c in df.columns])
    # A date would be ordinal-coded into nonsense by `datasets.prepare`; fail instead.
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            raise RuntimeError(f"{name}: column {c} is a date; add it to `drop`")
    df["__target__"] = y
    out = table_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"  [external] {name}: {df.shape[0]} rows, {df.shape[1] - 1} covariates -> "
          f"{out.relative_to(paths.ROOT)}")


def load(name):
    """Covariates and target of a converted table."""
    path = table_path(name)
    if not path.exists():
        raise RuntimeError(f"{name} is not converted yet; run "
                           f"`venv-data/Scripts/python -m common.external`")
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    y = pd.to_numeric(df.pop("__target__"), errors="coerce").to_numpy(dtype=float)
    return df, y


if __name__ == "__main__":
    for n in REGISTRY:
        convert(n)
