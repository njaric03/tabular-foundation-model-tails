# -*- coding: utf-8 -*-
"""
Evaluacija post-hoc EVT popravke: sirov model naspram popravljenog.

Skupovi su izabrani `dataset_selection.py`-jem, po kriterijumu da im je indeks repa
uopste identifikovan (tri procenjivaca se slazu, procena stabilna na izbor praga).
Raspon ide od teskog (freMTPL2sev) do lakog (Allstate), sto je bas ono sto treba za
ocenu metoda: mora da pomogne gde je rep tezak i da NE STETI gde nije.

Uz stvarne skupove ide i sinteticki deo, gde se pravo xi zna, pa se vidi koliko
popravka priblizi model istini.

Metrike su u `posthoc-evt.py`; namerno ih je vise od implicirano-xi, jer to nije
standardna mera. Glavna je twCRPS, koja jedina zapravo tezi rep.

    python evt_graft_experiment.py            # sve
    DEO=sint python evt_graft_experiment.py   # samo sinteticki
    DEO=real python evt_graft_experiment.py   # samo stvarni

Rezultati u `evt_graft.csv`.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from common import datasets, generator, models, paths, quiet

quiet.silence()

# `posthoc_evt_mod` je staro ime modula `evt_graft.py`, iz istog direktorijuma.
# Uvoz je posle preimenovanja 7.9.2026 ostao neispravljen.
from evt_graft import (NIVOI, I50, I90, ALPHA0, fit_evt_rep,
                       fit_evt_rep_reg, spoji, sve_metrike)

# W je u `common/generator.py`; ovde je stajala kopija formule.
XI_SINT = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_TR_SINT, N_TE_SINT = 2000, 800
N_TR_REAL, N_TE_REAL = 3000, 1000
N_EST = 4
MODELI = ["TabICLv2", "TabPFN-V3"]


# ------------------------------------------------------------------ modeli
def kvantili(model, Xtr, ytr, Xte, seed):
    return models.quantiles(model, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


# ------------------------------------------------------------------ podaci
def sint(xi, n, rng):
    """Isti generator koji dobijaju modeli; xi = 0 je Gumbelova granica u njemu."""
    p = generator.gpd(n, rng, xi=xi)
    return p.X, p.y


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
    ok = np.isfinite(y) & (y > 0)
    return X.to_numpy(dtype=float)[ok], y[ok]


def gpd_ref_xi(X, y):
    """Referentno xi iz podataka, isto kao u probe-u."""
    D = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(D, np.log(y), rcond=None)
    z = y / np.exp(D @ b)
    u = np.quantile(z, 0.90)
    return float(genpareto.fit(z[z > u] - u, floc=0)[0])


# ------------------------------------------------------------------- petlja
OUT = os.environ.get("OUTPUT", "evt_graft.csv")


VARIJANTE = ("raw", "EVT simple", "EVT regularised")


def vec_uradjeno():
    """Sta je vec kompletno u CSV-u.

    Jedinica se smatra gotovom samo ako ima SVE tri varijante. Bez te provere je
    jednom vec proslo da su redovi iz starije verzije skripte (dve grane, pre
    ispravke degenerisane medijane) bili prepoznati kao gotovi, pa su nove jedinice
    preskakane i CSV je ostao mesavina dva formata.
    """
    if not os.path.exists(paths.result(OUT)):
        return set()
    d = pd.read_csv(paths.result(OUT))
    if "variant" not in d.columns:
        return set()
    d = d[d.variant.isin(VARIJANTE)]
    br = d.groupby(["dataset", "model", "seed"]).variant.nunique()
    return set(br[br == len(VARIJANTE)].index)


def upisi(rows):
    """Dopisi u CSV odmah. Bez ovoga jedan segfault pojede ceo sat racunanja."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(paths.result(OUT)):
        df = pd.concat([pd.read_csv(paths.result(OUT)), df], ignore_index=True)
        df = df.drop_duplicates(subset=["dataset", "model", "seed", "variant"], keep="last")
    df.to_csv(paths.result(OUT), index=False)


def jedan(oznaka, Xtr, ytr, Xte, yte, model, seed, xi_pravo, t0, rows):
    q_tr, q_te = kvantili(model, Xtr, ytr, Xte, seed)
    med_tr, med_te = q_tr[:, I50], q_te[:, I50]
    prag = float(np.quantile(yte, 0.90))          # prag za twCRPS

    xi_n, sg_n, _ = fit_evt_rep(ytr, med_tr)              # prosta procena
    xi_r, sg_r, _ = fit_evt_rep_reg(ytr, med_tr)          # po vise pragova + granica
    q_n = spoji(q_te, med_te, xi_n, sg_n)
    q_r = spoji(q_te, med_te, xi_r, sg_r)

    for varijanta, q in [("raw", q_te), ("EVT simple", q_n), ("EVT regularised", q_r)]:
        m = sve_metrike(q, yte, prag)
        m.update(dataset=oznaka, model=model, seed=seed, variant=varijanta,
                 xi_true=xi_pravo, xi_evt_simple=xi_n, xi_evt_reg=xi_r)
        rows.append(m)
    a, b, c = rows[-3], rows[-2], rows[-1]
    dn = 100 * (b["twcrps"] - a["twcrps"]) / a["twcrps"] if a["twcrps"] else np.nan
    dr = 100 * (c["twcrps"] - a["twcrps"]) / a["twcrps"] if a["twcrps"] else np.nan
    print(f"  {oznaka:24s} {model:10s} s={seed}  twCRPS {a['twcrps']:.4g} | "
          f"prosta {dn:+.1f}% (xi {xi_n:.2f}) | regular {dr:+.1f}% (xi {xi_r:.2f})"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    upisi(rows[-3:])                       # odmah na disk


def main():
    deo = os.environ.get("DEO", "sve")
    rows, t0 = [], time.time()
    gotovo = vec_uradjeno()
    if gotovo:
        print(f"nastavljam; vec uradjeno {len(gotovo)} jedinica", flush=True)

    if deo in ("sve", "sint"):
        print("=== SINTETICKI (pravo xi poznato) ===", flush=True)
        for xi in XI_SINT:
            for seed in SEEDS:
                rng = np.random.default_rng(7000 + seed * 1000 + int(xi * 10))
                Xtr, ytr = sint(xi, N_TR_SINT, rng)
                Xte, yte = sint(xi, N_TE_SINT, rng)
                for model in MODELI:
                    if (f"sint xi={xi}", model, seed) in gotovo:
                        continue
                    jedan(f"sint xi={xi}", Xtr, ytr, Xte, yte, model, seed, xi, t0, rows)

    if deo in ("sve", "real"):
        print("\n=== STVARNI (izabrani po dataset_selection.py) ===", flush=True)
        izabrani = [l.strip() for l in open(paths.data("selected_datasets.txt"), encoding="utf-8")
                    if l.strip()]
        for ime in izabrani:
            try:
                X, y = ucitaj(ime)
            except Exception as e:
                print(f"  {ime}: preskacem ({type(e).__name__})", flush=True)
                continue
            for seed in SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, train_size=min(N_TR_REAL, int(0.6 * len(y))),
                    test_size=min(N_TE_REAL, int(0.3 * len(y))), random_state=seed)
                ref = gpd_ref_xi(Xtr, ytr)
                for model in MODELI:
                    if (ime, model, seed) in gotovo:
                        continue
                    try:
                        jedan(ime, Xtr, ytr, Xte, yte, model, seed, ref, t0, rows)
                    except Exception as e:
                        print(f"  {ime} {model} s={seed}: pao ({type(e).__name__}), dalje",
                              flush=True)

    df = pd.read_csv(paths.result(OUT))

    pd.set_option("display.width", 250)
    print("\n=== twCRPS: sirov naspram popravljenog (nize je bolje) ===")
    p = df.pivot_table(index=["dataset", "model"], columns="variant", values="twcrps")
    for v in ["EVT simple", "EVT regularised"]:
        if v in p.columns:
            p[v + " %"] = 100 * (p[v] - p["raw"]) / p["raw"]
    print(p.round(3).to_string())
    print("\n=== pinball na 0,999 (rep direktno) ===")
    p2 = df.pivot_table(index=["dataset", "model"], columns="variant", values="pinball999")
    for v in ["EVT simple", "EVT regularised"]:
        if v in p2.columns:
            p2[v + " %"] = 100 * (p2[v] - p2["raw"]) / p2["raw"]
    print(p2.round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
