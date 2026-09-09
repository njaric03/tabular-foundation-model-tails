# -*- coding: utf-8 -*-
"""
Faze 1-3 iz `thesis/notes/evt_graft_plan.md`, u jednom prolazu.

Odgovara na tri pitanja odjednom, jer sva tri koriste iste fitove:

  FAZA 1  Koliki je POSTEN orakl?
          Naivni orakl bira bolju opciju na test skupu, pa deo razlike hvata sum i
          precenjuje sta se moze dobiti. Zato se test deli na dve polovine: odluka na
          prvoj, merenje na drugoj. Razlika naivnog i ugnezdjenog orakla je pristrasnost.

  FAZA 2  Radi li VALIDACIONA kapija?
          Umesto praga na xi koji se postavlja rukom, odluka se donosi na izdvojenom
          `val` delu koji model nije video. Ako je kalem tamo bolji, primenjuje se na
          `test`. Nema pretpostavke o obliku repa i nema praga koji se brani recima.

  FAZA 3  Sta je zapravo signal?
          Iz `fit` dela se racunaju obelezja koja bi mogla da predvide isplati li se
          kalem. Kljucni par koji treba razdvojiti: Buzz (xi ~ 0,63, kalem pogorsa za
          44%) naspram freMTPL2sev (xi ~ 0,62, kalem popravi za 30%). Ako ih neko
          obelezje razdvoji, to je odgovor na pitanje iz naslova rada.

PODELA
------
    fit   model + EVT procena repa
    val   odluka kapije
    test  evaluacija; deli se na dve polovine za posten orakl

Rezultati u `graft_gate.csv`, jedan red po (skup, model, seed).
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import genpareto
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from common import quiet

quiet.silence()

# `posthoc_evt_mod` je ime koje je taj modul imao pre preimenovanja 7.9.2026; sada
# je `evt_graft.py`, u istom direktorijumu. Uvoz je ostao na starom imenu, pa se
# skripta nije mogla ni pokrenuti. `tests/test_scripts_import.py` to sada hvata.
from evt_graft import NIVOI, I50, ALPHA0, fit_evt_rep_reg, spoji, sve_metrike

N_TRAIN, N_TEST = 2000, 900

# W je u `common/generator.py`; ovde je stajala kopija formule.
XI_SINT = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
# DEKONTAMINACIJA (vidi `findings/h1/vincentization.md` i `findings/side/coherence.md`):
#   N_EST=1     jer agregacija spusta implicirano xi za 0.167 pri xi=0.9
#   LOG_SCALE=1 jer je fit na log skali bolji 5/5 i smanjuje neslaganje modela
# Podrazumevano ostaje staro ponasanje, da se ranije brojke reprodukuju.
import os as _os

from common import datasets, generator, metrics, paths, quiet
N_EST = int(_os.environ.get("N_EST", "4"))
LOG_SKALA = _os.environ.get("LOG_SCALE", "0") == "1"
N_FIT, N_VAL, N_TEST = 2000, 800, 1000
MODELI = ["TabICLv2", "TabPFN-V3"]
OUT = _os.environ.get("OUTPUT", "graft_gate.csv")
PRAGOVI_XI = (0.05, 0.10, 0.20)


# ------------------------------------------------------------------ modeli
def kvantili_fn(model, Xfit, yfit, seed):
    """Uz LOG_SCALE=1 model se fituje na log(y), a kvantili se vracaju eksponenciranjem.
    Kvantili su ekvivarijantni pod monotonom transformacijom, pa je to legitimno."""
    if LOG_SKALA:
        unutra = _kvantili_fn_sirovo(model, Xfit, np.log(np.maximum(yfit, 1e-12)), seed)
        return lambda X: np.exp(unutra(X))
    return _kvantili_fn_sirovo(model, Xfit, yfit, seed)


def _kvantili_fn_sirovo(model, Xfit, yfit, seed):
    if model == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=N_EST, device="cpu", random_state=seed)
        m.fit(Xfit, yfit)
        return lambda X: np.asarray(m.predict(X, output_type="quantiles", alphas=NIVOI))
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=N_EST, device="cpu", random_state=seed,
                        ignore_pretraining_limits=True)
    m.fit(Xfit, yfit)
    return lambda X: np.stack([np.asarray(a) for a in
                               m.predict(X, output_type="quantiles",
                                         quantiles=NIVOI)], axis=1)


# ------------------------------------------------------------- obelezja (faza 3)


def _implied(r, hi=0.99, lo=0.9):
    try:
        return brentq(lambda x: ((metrics.gpd_quantile(hi, x) - metrics.gpd_quantile(0.5, x)) /
                                 (metrics.gpd_quantile(lo, x) - metrics.gpd_quantile(0.5, x))) - r, -0.9, 6.0)
    except Exception:
        return np.nan


def ad_statistika(ex, xi, sigma):
    """Anderson-Darling nad prekoracenjima, posle prelaska u uniformnu skalu.

    Parametri su procenjeni iz istih podataka, pa nulta raspodela nije standardna i
    statistika se ne koristi kao formalan test, nego kao OBELEZJE: veca vrednost znaci
    da GPD losije pristaje repu, sto je kandidat za signal kapije.
    """
    u = np.sort(genpareto.cdf(ex, xi, loc=0, scale=sigma))
    u = np.clip(u, 1e-12, 1 - 1e-12)
    n = len(u)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))))


def obelezja(Xfit, yfit, med_fit, q_fit):
    """Sve sto se moze izracunati u trenutku odluke, dakle samo iz `fit` dela."""
    o = {}
    # 1) referentno xi: log-linearna skala pa GPD MLE, isto kao u probe-skupova2
    D = np.c_[np.ones(len(yfit)), Xfit]
    b, *_ = np.linalg.lstsq(D, np.log(yfit), rcond=None)
    z = yfit / np.exp(D @ b)
    xis = []
    for p in PRAGOVI_XI:
        u = np.quantile(z, 1 - p)
        ex = z[z > u] - u
        if len(ex) >= 50:
            xis.append(float(genpareto.fit(ex, floc=0)[0]))
    o["xi_ref"] = float(np.median(xis)) if xis else np.nan
    o["xi_ref_range"] = float(np.max(xis) - np.min(xis)) if len(xis) > 1 else np.nan

    # 2) Hill i kvantilna procena, pa slaganje sva tri
    zs = np.sort(z)
    k = max(int(0.10 * len(zs)), 30)
    o["xi_hill"] = float(np.mean(np.log(zs[-k:])) - np.log(zs[-k]))
    q50, q90, q99 = np.quantile(z, [0.5, 0.9, 0.99])
    o["xi_quantile"] = _implied((q99 - q50) / (q90 - q50)) if (q90 - q50) > 1e-12 else np.nan
    tri = np.array([o["xi_ref"], o["xi_hill"], o["xi_quantile"]], dtype=float)
    o["agreement_range"] = float(np.nanmax(tri) - np.nanmin(tri))

    # 3) koliko GPD pristaje repu reziduala (glavni kandidat za signal)
    u = np.quantile(z, 0.90)
    ex = z[z > u] - u
    o["n_exceedances"] = int(len(ex))
    if len(ex) >= 50:
        xi_h, _, sg_h = genpareto.fit(ex, floc=0)
        o["ad_gpd"] = ad_statistika(ex, xi_h, sg_h)
    else:
        o["ad_gpd"] = np.nan

    # 4) diskretnost targeta, hipoteza za Buzz
    _, cnt = np.unique(yfit, return_counts=True)
    o["tied_share"] = float(cnt.max() / len(yfit))
    o["unique_share"] = float(len(cnt) / len(yfit))

    # 5) sta model sam tvrdi o repu, i koliko je medijana degenerisana
    den = q_fit[:, NIVOI.index(ALPHA0)] - q_fit[:, I50]
    ok = den > 1e-9
    if ok.any():
        r = float(np.median((q_fit[ok, NIVOI.index(0.99)] - q_fit[ok, I50]) / den[ok]))
        o["xi_model"] = _implied(r)
    else:
        o["xi_model"] = np.nan
    o["median_unknown_share"] = float(np.mean(med_fit <= 0))
    return o


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


# ------------------------------------------------------------------- jedinica
def jedan(oznaka, delovi, model, seed, t0, rows):
    (Xf, yf), (Xv, yv), (Xt, yt) = delovi
    pred = kvantili_fn(model, Xf, yf, seed)
    q_f, q_v, q_t = pred(Xf), pred(Xv), pred(Xt)

    xi_e, sg_e, _ = fit_evt_rep_reg(yf, q_f[:, I50])
    kal_v = spoji(q_v, q_v[:, I50], xi_e, sg_e)
    kal_t = spoji(q_t, q_t[:, I50], xi_e, sg_e)

    r = dict(dataset=oznaka, model=model, seed=seed, xi_evt=xi_e)
    r.update(obelezja(Xf, yf, q_f[:, I50], q_f))

    # --- val: odluka kapije
    pv = float(np.quantile(yv, 0.90))
    mv_s, mv_k = sve_metrike(q_v, yv, pv), sve_metrike(kal_v, yv, pv)
    for m in ["pinball999", "twcrps"]:
        r[f"val_sirov_{m}"], r[f"val_kalem_{m}"] = mv_s[m], mv_k[m]
    r["val_says_graft"] = bool(mv_k["pinball999"] < mv_s["pinball999"])

    # --- test: evaluacija, cela i po polovinama (za posten orakl)
    pt = float(np.quantile(yt, 0.90))
    mt_s, mt_k = sve_metrike(q_t, yt, pt), sve_metrike(kal_t, yt, pt)
    for m in ["pinball999", "pinball99", "twcrps", "gamma_dev", "cov999"]:
        r[f"test_sirov_{m}"], r[f"test_kalem_{m}"] = mt_s[m], mt_k[m]

    pola = len(yt) // 2
    for ime, sl in [("A", slice(0, pola)), ("B", slice(pola, None))]:
        yy, pp = yt[sl], float(np.quantile(yt[sl], 0.90))
        r[f"{ime}_sirov"] = sve_metrike(q_t[sl], yy, pp)["pinball999"]
        r[f"{ime}_kalem"] = sve_metrike(kal_t[sl], yy, pp)["pinball999"]

    rows.append(r)
    d = 100 * (mt_k["pinball999"] - mt_s["pinball999"]) / mt_s["pinball999"]
    print(f"  {oznaka:24s} {model:10s} s={seed}  test {d:+6.1f}%  "
          f"val kaze {'KALEMI' if r['val_says_graft'] else 'ne diraj'}  "
          f"xi_ref={r['xi_ref']:.2f} AD={r['ad_gpd']:.1f}  [{time.time()-t0:.0f}s]", flush=True)
    upisi([r])


def upisi(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(paths.result(OUT)):
        df = pd.concat([pd.read_csv(paths.result(OUT)), df], ignore_index=True)
        df = df.drop_duplicates(subset=["dataset", "model", "seed"], keep="last")
    df.to_csv(paths.result(OUT), index=False)


def gotovo_set():
    if not os.path.exists(paths.result(OUT)):
        return set()
    d = pd.read_csv(paths.result(OUT))
    return set(zip(d.dataset, d.model, d.seed))


def main():
    rows, t0 = [], time.time()
    gotovo = gotovo_set()
    if gotovo:
        print(f"nastavljam, vec {len(gotovo)} jedinica", flush=True)

    print("=== SINTETICKI ===", flush=True)
    for xi in XI_SINT:
        for seed in SEEDS:
            rng = np.random.default_rng(5000 + seed * 100 + int(xi * 10))
            delovi = (sint(xi, N_FIT, rng), sint(xi, N_VAL, rng), sint(xi, N_TEST, rng))
            for model in MODELI:
                if (f"sint xi={xi}", model, seed) in gotovo:
                    continue
                jedan(f"sint xi={xi}", delovi, model, seed, t0, rows)

    print("\n=== STVARNI ===", flush=True)
    for ime in [l.strip() for l in open(paths.data("selected_datasets.txt"), encoding="utf-8") if l.strip()]:
        try:
            X, y = ucitaj(ime)
        except Exception as e:
            print(f"  {ime}: preskacem ({type(e).__name__})", flush=True)
            continue
        for seed in SEEDS:
            n = len(y)
            Xr, Xt, yr, yt = train_test_split(
                X, y, test_size=min(N_TEST, int(0.25 * n)), random_state=seed)
            Xf, Xv, yf, yv = train_test_split(
                Xr, yr, train_size=min(N_FIT, int(0.6 * len(yr))),
                test_size=min(N_VAL, int(0.25 * len(yr))), random_state=seed)
            delovi = ((Xf, yf), (Xv, yv), (Xt, yt))
            for model in MODELI:
                if (ime, model, seed) in gotovo:
                    continue
                try:
                    jedan(ime, delovi, model, seed, t0, rows)
                except Exception as e:
                    print(f"  {ime} {model} s={seed}: pao ({type(e).__name__})", flush=True)

    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
