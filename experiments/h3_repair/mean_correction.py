# -*- coding: utf-8 -*-
"""
Post-hoc korekcija uslovne sredine, preko cele familije modela.

ZASTO OVO A NE KALEMLJENJE REPA
-------------------------------
Kalem repa je ciljao kvantil 0,999, koji se na test skupovima reda 10^3 ne moze
pouzdano izmeriti, izmereno je da se efekat ne reprodukuje (r = 0,04 izmedju dva
nezavisna eksperimenta), dok se iste mere na nivou 0,99 reprodukuju sa r = 0,80.

Deficit sredine je suprotan slucaj: velik, stabilan i merljiv. TabPFN-V3 pri xi = 0,9
vraca 0,304 prave uslovne sredine sa rasipanjem od 0,012 po pet seedova, uz granicu
dostiznosti od 0,549. Gama devijansa, metrika koja to hvata, reprodukuje se sa r = 0,91.

Uz to, metod trazi SAMO tackovnu prognozu, pa vazi i za modele bez uslovne raspodele
(TabFM, Mitra, LimiX), za razliku od kalemljenja repa.

METOD
-----
    y_ispravljeno = c * y_predvidjeno

`c` se bira na izdvojenom `val` delu tako da minimizuje gama devijansu. Izvod po c daje
zatvoreno resenje:

    d/dc [ 2*mean( y/(c*yhat) - 1 - log(y/(c*yhat)) ) ] = 0   =>   c* = mean(y/yhat)

Dakle jedan broj, bez optimizacije i bez pretpostavke o obliku repa.

Uz to se meri i bogatija varijanta: izotona regresija yhat -> y naucena na `val`, za
slucaj da deficit nije isti po celom opsegu predikcije.

POKRETANJE
----------
    MODEL=TabICLv2   python mean_correction.py
    MODEL=TabPFN-V3  python mean_correction.py        # bira se i TABPFN_MODEL_VERSION
    MODEL=GBM        python mean_correction.py
    MODEL=TabDPT     venv-tabdpt/Scripts/python mean_correction.py
    MODEL=TabFM      venv-tabfm/Scripts/python mean_correction.py

Rezultati se dopisuju u `mean_correction.csv`.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from common import datasets, paths

warnings.filterwarnings("ignore")

N_TRAIN, N_TEST = 2000, 900

MODEL = os.environ.get("MODEL", "TabICLv2")
W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])
XI_SINT = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_FIT, N_VAL, N_TEST = 2000, 800, 1500
OUT = os.environ.get("OUTPUT", "mean_correction.csv")

# TabFM je ~1,6 mlrd parametara na CPU-u, oko dva i po minuta po fitu; smanjen profil.
if MODEL == "TabFM":
    N_FIT, N_VAL, N_TEST = 1000, 500, 600


# ------------------------------------------------------------------ modeli
def napravi_prediktor(Xf, yf, seed):
    """Vrati funkciju X -> predvidjena uslovna sredina."""
    if MODEL == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=4, device="cpu", random_state=seed)
        m.fit(Xf, yf)
        return lambda X: np.asarray(m.predict(X, output_type="mean"), dtype=float)
    if MODEL.startswith("TabPFN"):
        from tabpfn import TabPFNRegressor
        m = TabPFNRegressor(n_estimators=4, device="cpu", random_state=seed,
                            ignore_pretraining_limits=True)
        m.fit(Xf, yf)
        return lambda X: np.asarray(m.predict(X, output_type="mean"), dtype=float)
    if MODEL == "TabDPT":
        from tabdpt import TabDPTRegressor
        m = TabDPTRegressor(device="cpu", use_flash=False)
        m.fit(Xf, yf)
        return lambda X: np.asarray(m.predict(X, n_ensembles=4, seed=seed), dtype=float)
    if MODEL == "TabFM":
        import tabfm
        from tabfm import TabFMRegressor
        ck = os.path.expanduser("~/tabfm-regression")
        import torch
        mod = tabfm.tabfm_v1_0_0_pytorch.load(model_type="regression", checkpoint_path=ck,
                                              device="cpu", dtype=torch.bfloat16)
        r = TabFMRegressor(mod, n_estimators=1, random_state=seed)
        r.fit(Xf, yf)
        return lambda X: np.asarray(r.predict(X), dtype=float)
    from sklearn.ensemble import GradientBoostingRegressor
    g = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=seed)
    g.fit(Xf, yf)
    return lambda X: np.asarray(g.predict(X), dtype=float)


# ------------------------------------------------------------------ metrike
def gama_dev(mu, y):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    return float(2.0 * np.mean((y - mu) / mu - np.log(y / mu)))


def rmse(mu, y):
    return float(np.sqrt(np.mean((np.asarray(mu) - np.asarray(y)) ** 2)))


# ------------------------------------------------------------------ podaci
def sint(xi, n, rng):
    X = rng.normal(size=(n, 5))
    s = np.exp(0.6 * (X @ W) / np.linalg.norm(W))
    U = rng.random(n)
    y = s * (-np.log(1 - U)) if xi == 0 else s * ((1 - U) ** (-xi) - 1) / xi
    return X, y, s / (1 - xi) if xi < 1 else np.full(n, np.nan)


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
def _pod(mu, yf):
    """Podigni predikciju na smislen pozitivan pod.

    ZASTO: odsecanje na 1e-9 je vec jednom razvalilo merenje. Modeli daju nepozitivne
    predikcije i na strogo pozitivnom targetu; deljenje takvom vrednoscu pravi odnos
    y/yhat reda 1e9, koji sam odredi faktor `c`. Pod se vezuje za skalu targeta, ne za
    masinsku epsilon vrednost.
    """
    return np.maximum(np.asarray(mu, dtype=float), 1e-3 * float(np.median(yf)))


def jedan(oznaka, delovi, seed, prava_sredina, t0, rows):
    (Xf, yf), (Xv, yv), (Xt, yt) = delovi
    pred = napravi_prediktor(Xf, yf, seed)
    sirovo_v, sirovo_t = pred(Xv), pred(Xt)
    mv, mt = _pod(sirovo_v, yf), _pod(sirovo_t, yf)
    unknown_share = float(np.mean(sirovo_t <= 0))

    # --- korekcija naucena SAMO na val delu; degenerisane tacke se izbacuju
    ok = sirovo_v > 0
    sel0 = ok if ok.sum() >= 50 else np.ones(len(mv), dtype=bool)
    # Dva kandidata za faktor. `c_dev` minimizuje gama devijansu u zatvorenom obliku,
    # ali tezi tacke sa MALOM predikcijom (deli se sa yhat), pa ga sitne predikcije
    # razvale, GBM ih daje do 8,6%. `c_ratio` tezi po velicini i na to je otporan.
    # Koji je bolji nije ocigledno, pa se mere oba.
    c = float(np.mean(yv[sel0] / mv[sel0]))                       # c_dev
    c_ratio = float(np.mean(yv[sel0]) / np.mean(mv[sel0]))
    # sel0 vazi i za izotonu
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(mv[sel0], yv[sel0])

    var = {"raw": mt, "konstanta": c * mt, "odnos_proseka": c_ratio * mt,
           "izotona": np.maximum(iso.predict(mt), 1e-9)}

    r = dict(dataset=oznaka, model=MODEL, seed=seed, c_val=c,
             c_test=float(np.mean(yt / mt)),          # koliko bi bio idealan faktor
             c_ratio=c_ratio, unknown_share=unknown_share)
    for ime, mu in var.items():
        r[f"gdev_{ime}"] = gama_dev(mu, yt)
        r[f"rmse_{ime}"] = rmse(mu, yt)
        if prava_sredina is not None and np.all(np.isfinite(prava_sredina)):
            r[f"odnos_{ime}"] = float(mu.mean() / prava_sredina.mean())
    rows.append(r)
    d1 = 100 * (r["gdev_constant"] - r["gdev_raw"]) / r["gdev_raw"]
    d3 = 100 * (r["gdev_mean_ratio"] - r["gdev_raw"]) / r["gdev_raw"]
    d2 = 100 * (r["gdev_isotonic"] - r["gdev_raw"]) / r["gdev_raw"]
    up = f" nepoz={unknown_share:.1%}" if unknown_share > 0 else ""
    print(f"  {oznaka:22s} s={seed}  c={c:5.2f} (idealno {r['c_test']:5.2f}){up}  "
          f"gdev: c_dev {d1:+6.1f}%  c_ratio {d3:+6.1f}%  izo {d2:+6.1f}%  "
          f"[{time.time()-t0:.0f}s]",
          flush=True)
    upisi([r])


def upisi(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(paths.result(OUT)):
        df = pd.concat([pd.read_csv(paths.result(OUT)), df], ignore_index=True)
        df = df.drop_duplicates(subset=["dataset", "model", "seed"], keep="last")
    df.to_csv(paths.result(OUT), index=False)


def gotovo():
    if not os.path.exists(paths.result(OUT)):
        return set()
    d = pd.read_csv(paths.result(OUT))
    return set(zip(d.dataset, d.model, d.seed))


def main():
    print(f"MODEL={MODEL}", flush=True)
    rows, t0, g = [], time.time(), gotovo()

    print("=== SINTETICKI ===", flush=True)
    for xi in XI_SINT:
        for seed in SEEDS:
            if (f"sint xi={xi}", MODEL, seed) in g:
                continue
            rng = np.random.default_rng(3000 + seed * 100 + int(xi * 10))
            Xf, yf, _ = sint(xi, N_FIT, rng)
            Xv, yv, _ = sint(xi, N_VAL, rng)
            Xt, yt, st = sint(xi, N_TEST, rng)
            jedan(f"sint xi={xi}", ((Xf, yf), (Xv, yv), (Xt, yt)), seed, st, t0, rows)

    print("\n=== STVARNI ===", flush=True)
    for ime in [l.strip() for l in open(paths.data("selected_datasets.txt"), encoding="utf-8") if l.strip()]:
        try:
            X, y = ucitaj(ime)
        except Exception as e:
            print(f"  {ime}: preskacem ({type(e).__name__})", flush=True)
            continue
        for seed in SEEDS:
            if (ime, MODEL, seed) in g:
                continue
            n = len(y)
            Xr, Xt, yr, yt = train_test_split(X, y, test_size=min(N_TEST, int(0.25 * n)),
                                              random_state=seed)
            Xf, Xv, yf, yv = train_test_split(
                Xr, yr, train_size=min(N_FIT, int(0.6 * len(yr))),
                test_size=min(N_VAL, int(0.25 * len(yr))), random_state=seed)
            try:
                jedan(ime, ((Xf, yf), (Xv, yv), (Xt, yt)), seed, None, t0, rows)
            except Exception as e:
                print(f"  {ime} s={seed}: pao ({type(e).__name__})", flush=True)

    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
