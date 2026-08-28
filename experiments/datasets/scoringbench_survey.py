# -*- coding: utf-8 -*-
"""
Indeks repa za svih 102 skupa iz ScoringBench-a.

ZASTO
-----
ScoringBench (arXiv:2603.29928) daje 51 model x 102 skupa x 5 foldova javnih rezultata,
ukljucujuci `wcrps_right`. Ali tezina repa nije osa tog benchmarka, pa se ne zna da li
razlike izmedju modela zavise od nje. Ovde se ta osa dodaje: isti estimatori i isti
kriterijum ulaska kao u `dataset_selection.py`, pusteni preko njihovog skupa skupova.

Rezultat se spaja sa `sb_all.parquet` (njihovi rezultati) i daje stratifikaciju po repu
BEZ ijednog pokretanja modela.

KRITERIJUM (isti kao dataset_selection.py)
  1. n >= 3000 pozitivnih vrednosti
  2. max(procene) >= 0.15
  3. raspon tri procene <= 0.25
  4. GPD-MLE na 5/10/20% u rasponu <= 0.25

NASTAVLJIVOST
-------------
Pise red po red u `scoringbench_survey.csv` i preskace vec uradjene. Jedan proces,
jedan fajl - bez trke pri upisu.

POKRETANJE
----------
    python -u scoringbench_survey.py
"""
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml

from common import append, datasets, metrics, paths

warnings.filterwarnings("ignore")

MAX_ROWS = 30000
MIN_N = 3000        # prag za PROLAZAK kriterijuma (isti kao dataset_selection.py)
MIN_N_MERI = 500    # ispod ovoga se procena ne racuna; iznad se racuna i belezi
MIN_XI = 0.15
MAX_RASPON = 0.25
FRACS = [0.05, 0.10, 0.20]
OUT = os.environ.get("OUTPUT", "scoringbench_survey.csv")
IMENA = "sb_datasets.json"
IDS = "sb_openml_ids.json"
# Imena koja se na OpenML-u zovu drugacije nego u ScoringBench-u.
ALIJASI = {"Ele2": 42362}
KOLONE = ["dataset", "n", "d", "shifted", "gpd05", "gpd10", "gpd20", "hill", "quantile",
          "estimate_range", "range_threshold", "tied_share", "reason", "passes"]
KEY = ["dataset"]


# --------------------------------------------------------- estimatori (identicni)


# --------------------------------------------------------- ucitavanje
PMLB_URL = ("https://github.com/EpistasisLab/pmlb/raw/master/datasets/"
            "{ime}/{ime}.tsv.gz")


def ucitaj_pmlb(ime):
    """PMLB skupovi (imena tipa `215_2dplanes`) nisu na OpenML-u pod tim imenom."""
    df = pd.read_csv(PMLB_URL.format(ime=ime), sep="\t", compression="gzip")
    y = pd.to_numeric(df.pop("target"), errors="coerce").to_numpy(dtype=float)
    return df, y


def ucitaj(ime, ids):
    """Probaj po imenu na OpenML-u, pa po ID-ju iz mape ScoringBench-a, pa PMLB."""
    if re.match(r"^\d+_", ime):
        X, y = ucitaj_pmlb(ime)
        return pripremi(X, y)
    last = None
    kandidati = [{"name": ime}]
    if ime in ALIJASI:
        kandidati.insert(0, {"data_id": ALIJASI[ime]})
    if ime in ids:
        kandidati.append({"data_id": ids[ime]})
    kandidati.append({"name": ime, "as_frame": False})   # retki ARFF
    for kw in kandidati:
        try:
            af = kw.pop("as_frame", True)
            d = fetch_openml(as_frame=af, parser="auto", **kw)
            if not af:
                d.data = pd.DataFrame(np.asarray(d.data.todense() if hasattr(d.data, "todense")
                                                 else d.data))
            break
        except Exception as e:
            last = e
            d = None
    if d is None:
        # poslednja sansa: mozda ipak stoji u PMLB-u
        try:
            return pripremi(*ucitaj_pmlb(ime))
        except Exception:
            raise RuntimeError(f"ne mogu da ucitam ({type(last).__name__}: {last})")

    X = d.data.copy()
    y = pd.to_numeric(pd.Series(np.asarray(d.target).ravel()),
                      errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).any():
        # Podrazumevani target nije numericki (npr. datum ili kategorija).
        # Uzmi poslednju numericku kolonu kao target -- ScoringBench radi regresiju,
        # pa numericki target mora da postoji negde u okviru.
        num = X.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise RuntimeError("nema nijedne numericke kolone")
        kol = num[-1]
        y = X.pop(kol).to_numpy(dtype=float)
    return pripremi(X, y)


def pripremi(X, y):
    """Kategorije u kodove, popuna nedostajucih, izbacivanje neispravnog targeta."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        raise RuntimeError("target nije numericki")
    X = X.copy()
    for c in X.select_dtypes(exclude="number").columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    ok = np.isfinite(y)
    return X.to_numpy(dtype=float)[ok], y[ok]


def obradi(ime, ids):
    X, y = ucitaj(ime, ids)
    if len(y) > MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]

    n_poz = int((y > 0).sum())
    # Podrazumevano se, kao u `dataset_selection.py`, nepozitivne vrednosti prosto
    # ispuste. Pomeranje se radi SAMO kao spas za skupove kojima bi inace ostalo
    # premalo tacaka -- inace bi se brojke razisle sa vec objavljenim
    # `dataset_selection.csv` (npr. Buzz ima mnogo nula).
    shifted = False
    if n_poz < MIN_N_MERI <= len(y):
        y = y - y.min() + 1e-6 * (np.ptp(y) or 1.0)
        n_poz = int((y > 0).sum())
        shifted = True
    if n_poz < MIN_N_MERI:
        return dict(dataset=ime, n=n_poz, reason="premalo za procenu", passes=False)

    z = metrics.residuals(X, y)
    po_pragu = [metrics.gpd_mle(z, f) for f in FRACS]
    g10, h10 = po_pragu[1], metrics.hill(z)
    q50, q90, q99 = np.quantile(z, [0.5, 0.9, 0.99])
    kv = metrics.xi_from_ratio((q99 - q50) / (q90 - q50)) if (q90 - q50) > 1e-12 else np.nan

    tri = np.array([g10, h10, kv], dtype=float)
    raspon_tri = float(np.nanmax(tri) - np.nanmin(tri))
    pp = np.array(po_pragu, dtype=float)
    range_threshold = float(np.nanmax(pp) - np.nanmin(pp))
    _, cnt = np.unique(y[y > 0], return_counts=True)
    tied_share = float(cnt.max() / n_poz)

    passes = bool(n_poz >= MIN_N and np.nanmax(tri) >= MIN_XI
                   and raspon_tri <= MAX_RASPON and range_threshold <= MAX_RASPON)
    return dict(dataset=ime, n=n_poz, d=X.shape[1], shifted=shifted,
                gpd05=po_pragu[0], gpd10=g10, gpd20=po_pragu[2],
                hill=h10, quantile=kv,
                estimate_range=raspon_tri, range_threshold=range_threshold,
                tied_share=tied_share, reason="", passes=passes)


def main():
    imena = json.load(open(IMENA, encoding="utf-8"))
    ids = json.load(open(IDS, encoding="utf-8"))
    gotovi = set()
    if paths.result(OUT).exists():
        gotovi = set(pd.read_csv(paths.result(OUT)).dataset)
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)

    t0 = time.time()
    for i, ime in enumerate(imena, 1):
        if append.key(dict(dataset=ime), KEY) in gotovi:
            continue
        try:
            r = obradi(ime, ids)
        except Exception as e:
            r = dict(dataset=ime, reason=f"{type(e).__name__}: {e}"[:120], passes=False)
        # fiksan redosled kolona: redovi sa greskom imaju manje kljuceva, pa bi
        # bez ovoga header i vrednosti otisli u razmak
        append.write(OUT, r, KOLONE)
        if r.get("reason"):
            print(f"  [{i:3d}/{len(imena)}] {ime:34s} -- {r['reason']}", flush=True)
        else:
            print(f"  [{i:3d}/{len(imena)}] {ime:34s} "
                  f"GPD={r['gpd10']:+.2f} Hill={r['hill']:+.2f} kv={r['quantile']:+.2f} "
                  f"| raspon {r['estimate_range']:.2f}/{r['range_threshold']:.2f} "
                  f"| {'PROLAZI' if r['passes'] else '-'}  [{time.time()-t0:.0f}s]",
                  flush=True)

    df = pd.read_csv(paths.result(OUT))
    print(f"\ngotovo za {time.time()-t0:.0f}s | ucitano {df.gpd10.notna().sum()} / {len(df)}"
          f" | passes {int(df.passes.sum())}")


if __name__ == "__main__":
    main()
