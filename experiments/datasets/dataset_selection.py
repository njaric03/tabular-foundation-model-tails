# -*- coding: utf-8 -*-
"""
Stroga selekcija skupova za evaluaciju post-hoc popravke repa.

Prvi probe (`experiments/datasets/dataset_selection.py`) je pokazao da se kod vecine javnih skupova tri
procenjivaca indeksa repa ne slazu, pa referenca nije pouzdana. Ovde se to formalizuje
u KRITERIJUM ULASKA, umesto da se skupovi biraju po osecaju.

Skup ulazi ako zadovolji sve cetiri stavke:

  1. dovoljno podataka                n >= 3000 pozitivnih vrednosti
  2. rep uopste postoji               max(procene) >= 0.15
  3. procene se slazu                 raspon tri procene <= 0.25
  4. rep je stabilan na prag          GPD-MLE na 5%, 10% i 20% u rasponu <= 0.25

Bez (3) i (4) se ocenjuje model prema referenci koja i sama skace, sto je bio glavni
razlog neujednacenih rezultata na stvarnim podacima.

Uz to se prijavljuje i udeo vezanih vrednosti (mass points), jer skupovi tipa
freMTPL2sev imaju veliki deo mase na par tacaka, sto kvari i procenu i kvantilne mere.

Rezultati u `dataset_selection.csv`; oni koji prodju idu u `selected_datasets.txt`.
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml

from common import datasets, metrics, paths

warnings.filterwarnings("ignore")

MAX_ROWS = 30000
MIN_N = 3000
MIN_XI = 0.15
MAX_RASPON = 0.25
FRACS = [0.05, 0.10, 0.20]

KANDIDATI = [
    ("freMTPL2sev", "claim severity, French motor insurance"),
    ("delays_zurich_transport", "transit delay"),
    ("Airlines_DepDelay_1M", "departure delay"),
    ("Buzzinsocialmedia_Twitter", "interaction count"),
    ("medical_charges", "medical cost"),
    ("nyc-taxi-green-dec-2016", "fare amount"),
    ("black_friday", "purchase amount"),
    ("diamonds", "price"),
    ("SGEMM_GPU_kernel_performance", "runtime"),
    ("Allstate_Claims_Severity", "claim severity, Allstate"),
    ("video_transcoding", "transcoding time"),
    ("Bike_Sharing_Demand", "rental count"),
    ("house_sales", "cena nekretnine"),
    ("Yolanda", "sinteticki regresioni"),
    ("wine_quality", "wine score, light-tailed control"),
]


def ucitaj(ime):
    if ime == "freMTPL2sev":
        sev = fetch_openml("freMTPL2sev", as_frame=True, parser="auto").data
        freq = fetch_openml("freMTPL2freq", as_frame=True, parser="auto").data
        sev = sev.groupby("IDpol", as_index=False).ClaimAmount.sum()
        df = sev.merge(freq.drop(columns=["ClaimNb"]), on="IDpol", how="inner")
        y = df.pop("ClaimAmount").to_numpy(dtype=float)
        X = df.drop(columns=["IDpol"])
    else:
        d = fetch_openml(ime, as_frame=True, parser="auto")
        X = d.data.copy()
        y = pd.to_numeric(pd.Series(np.asarray(d.target).ravel()),
                          errors="coerce").to_numpy(dtype=float)
    for c in X.select_dtypes(exclude="number").columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    ok = np.isfinite(y)
    return X.to_numpy(dtype=float)[ok], y[ok]


def main():
    rows, t0 = [], time.time()
    for ime, opis in KANDIDATI:
        try:
            X, y = ucitaj(ime)
        except Exception as e:
            print(f"  {ime:30s} preskacem ({type(e).__name__})", flush=True)
            continue
        if len(y) > MAX_ROWS:
            idx = np.random.default_rng(0).choice(len(y), MAX_ROWS, replace=False)
            X, y = X[idx], y[idx]
        n_poz = int((y > 0).sum())
        if n_poz < MIN_N:
            print(f"  {ime:30s} premalo pozitivnih ({n_poz})", flush=True)
            continue

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

        passes = (n_poz >= MIN_N and np.nanmax(tri) >= MIN_XI
                   and raspon_tri <= MAX_RASPON and range_threshold <= MAX_RASPON)
        rows.append(dict(dataset=ime, description=opis, n=n_poz, d=X.shape[1],
                         gpd10=g10, hill=h10, quantile=kv,
                         estimate_range=raspon_tri, range_threshold=range_threshold,
                         tied_share=tied_share, passes=passes))
        znak = "PROLAZI" if passes else "-"
        print(f"  {ime:30s} GPD={g10:+.2f} Hill={h10:+.2f} kv={kv:+.2f} "
              f"| raspon {raspon_tri:.2f} / prag {range_threshold:.2f} "
              f"| vezanih {tied_share:.1%}  {znak}  [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows).sort_values(["passes", "gpd10"], ascending=[False, False])
    df.to_csv(paths.result("dataset_selection.csv"), index=False)
    pd.set_option("display.width", 240)
    print("\n=== SVI KANDIDATI ===")
    print(df.round(3).to_string(index=False))
    izabrani = list(df[df.passes].dataset)
    with open("selected_datasets.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(izabrani))
    print(f"\nPROLAZE ({len(izabrani)}): {izabrani}")
    print(f"ukupno {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
