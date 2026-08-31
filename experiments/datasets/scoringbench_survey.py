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
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import genpareto

from common import append, datasets, metrics, paths, quiet

quiet.silence()

MAX_ROWS = 30000
MIN_N = 3000        # prag za PROLAZAK kriterijuma (isti kao dataset_selection.py)
MIN_N_MERI = 500    # ispod ovoga se procena ne racuna; iznad se racuna i belezi
MIN_XI = 0.15
MAX_RASPON = 0.25
FRACS = [0.05, 0.10, 0.20]
OUT = os.environ.get("OUTPUT", "scoringbench_survey.csv")
IMENA = "sb_datasets.json"
IDS = "sb_openml_ids.json"
KOLONE = ["dataset", "n", "d", "shifted", "gpd05", "gpd10", "gpd20", "hill", "quantile",
          "estimate_range", "range_threshold", "tied_share",
          # Odakle je meta dosla i koliko je kolona bilo kategorijsko. Prvo je do
          # sada bila tiha odluka unutar loadera, a iz ovog pregleda izlazi bazna
          # stopa poluge, dakle naslovna brojka drugog dela.
          "source", "target_source", "n_categorical",
          "reason", "passes"]
KEY = ["dataset"]


# --------------------------------------------------------- estimatori (identicni)


# --------------------------------------------------------- ucitavanje
# Ucitavanje je `common/datasets.py`. Ovde je do sada stajala peta kopija istog
# koda -- iste kandidature na OpenML-u, isti PMLB fallback, ista `pripremi` --
# iako `common/datasets.py` u svom zaglavlju pise da je bas ovaj loader
# kanonski. Kopija je bila i mesto gde se cutke bira meta: kada podrazumevani
# target nije numericki, uzima se poslednja numericka kolona. Sada to bira ista
# funkcija za sve skripte i upisuje se u `datasets.LAST["target_source"]`, pa
# ulazi u CSV.


def obradi(ime, ids):
    X, y = datasets.load(ime, ids)
    info = dict(datasets.LAST)
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
                tied_share=tied_share, reason="", passes=passes,
                source=info.get("source", ""),
                target_source=info.get("target_source", ""),
                n_categorical=info.get("n_categorical", 0))


def main():
    imena = paths.load_json(IMENA)
    ids = paths.load_json(IDS)
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
