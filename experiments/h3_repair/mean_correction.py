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

MODELI, PODACI I UPIS IDU KROZ `common/`
---------------------------------------
Ovaj fajl je do sada gradio modele sam, ucitavao skupove sam i pisao CSV sam.
Bio je jedini takav: `common/models.py` je nastao 31.8., tri dana posle njega, i
tada nije prevucen.

Prelazak nije bio mehanicki, jer `models.mean` za TabICL i TabPFN vraca MEDIJANU,
uz obrazlozenje da je to najblizi tackovni sazetak bez dodatnih pretpostavki. Tu
je predmet merenja SREDINA, pa je u `common/models.py` dodat `predictive_mean`,
koji od modela trazi `output_type="mean"`. `mean` ostaje netaknut, jer ga citaju
`truncated_support`, `repair_mean` i `credibility`.

Uz to: `datasets.load(..., positive_only=True)` umesto lokalne kopije ucitavanja,
pa svaki skup prolazi kroz proveru otiska; i `append.write` umesto rucnog
prepisivanja celog CSV-a, pa se dobija zakljucavanje, provenance red i zastita od
pomeranja kolona. `n_est` je sada knob, kolona i deo kljuca, kako trazi pravilo 2.

POKRETANJE
----------
    MODEL=TabICLv2   python mean_correction.py
    MODEL=TabPFN-V3  python mean_correction.py
    MODEL=TabPFN-v2.5 python mean_correction.py       # generacija ide kroz TABPFN_PATHS
    MODEL=GBM        python mean_correction.py
    MODEL=TabDPT     venv-tabdpt/Scripts/python mean_correction.py
    MODEL=TabFM      venv-tabfm/Scripts/python mean_correction.py

Rezultati se dopisuju u `mean_correction.csv`.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from common import append, datasets, generator, models, paths, quiet

quiet.silence()

N_TRAIN, N_TEST = 2000, 900

MODEL = models.parse_list(os.environ.get("MODEL", "TabICLv2"))[0]
N_EST = int(os.environ.get("N_EST", "4"))
# W je u `common/generator.py`; ovde je stajala kopija formule.
XI_SINT = [0.0, 0.3, 0.5, 0.7, 0.9]
SEEDS = [0, 1, 2]
N_FIT, N_VAL, N_TEST = 2000, 800, 1500
OUT = os.environ.get("OUTPUT", "mean_correction.csv")

VARIANTS = ["raw", "constant", "mean_ratio", "isotonic"]
KOLONE = (["dataset", "model", "seed", "n_est", "c_val", "c_test",
           "c_ratio", "unknown_share"]
          + [f"{p}_{v}" for p in ("gdev", "rmse", "ratio") for v in VARIANTS]
          + ["seconds", "reason"])
KLJUC = ["dataset", "model", "seed", "n_est"]

# TabFM je ~1,6 mlrd parametara na CPU-u, oko dva i po minuta po fitu; smanjen profil.
if MODEL == "TabFM":
    N_FIT, N_VAL, N_TEST = 1000, 500, 600


# ------------------------------------------------------------------ modeli
def napravi_prediktor(Xf, yf, seed):
    """Vrati funkciju X -> predvidjena uslovna sredina.

    Jedno mesto, `models.predictive_mean`. Model se fituje po pozivu, pa se
    predikcija racuna odjednom za sve delove koji je traze.
    """
    def pred(X):
        return models.predictive_mean(MODEL, Xf, yf, X, seed=seed, n_est=N_EST)
    return pred


# ------------------------------------------------------------------ metrike
def gama_dev(mu, y):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    y = np.maximum(np.asarray(y, dtype=float), 1e-9)
    return float(2.0 * np.mean((y - mu) / mu - np.log(y / mu)))


def rmse(mu, y):
    return float(np.sqrt(np.mean((np.asarray(mu) - np.asarray(y)) ** 2)))


# ------------------------------------------------------------------ podaci
def sint(xi, n, rng):
    """Isti generator koji dobijaju modeli, plus prava uslovna sredina s/(1-xi)."""
    p = generator.gpd(n, rng, xi=xi)
    return p.X, p.y, generator.true_mean(p)


def ucitaj(ime):
    """Kroz `common.datasets`, pa svaki skup prolazi i kroz proveru otiska."""
    return datasets.load(ime, positive_only=True)


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
    t1 = time.time()
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

    # Kljucevi su engleski jer od njih nastaju imena kolona: `gdev_constant`,
    # `rmse_mean_ratio`, `ratio_isotonic`. Migracija je prevela CSV i citanja
    # ispod, a ovaj recnik nije, pa je `r["gdev_constant"]` dizao KeyError na
    # prvom redu i skripta nije mogla da izmeri nista.
    var = {"raw": mt, "constant": c * mt, "mean_ratio": c_ratio * mt,
           "isotonic": np.maximum(iso.predict(mt), 1e-9)}

    r = dict(dataset=oznaka, model=MODEL, seed=seed, n_est=N_EST, c_val=c,
             c_test=float(np.mean(yt / mt)),          # koliko bi bio idealan faktor
             c_ratio=c_ratio, unknown_share=unknown_share, reason="")
    for ime, mu in var.items():
        r[f"gdev_{ime}"] = gama_dev(mu, yt)
        r[f"rmse_{ime}"] = rmse(mu, yt)
        if prava_sredina is not None and np.all(np.isfinite(prava_sredina)):
            r[f"ratio_{ime}"] = float(mu.mean() / prava_sredina.mean())
    r["seconds"] = round(time.time() - t1, 1)
    rows.append(r)
    d1 = 100 * (r["gdev_constant"] - r["gdev_raw"]) / r["gdev_raw"]
    d3 = 100 * (r["gdev_mean_ratio"] - r["gdev_raw"]) / r["gdev_raw"]
    d2 = 100 * (r["gdev_isotonic"] - r["gdev_raw"]) / r["gdev_raw"]
    up = f" nepoz={unknown_share:.1%}" if unknown_share > 0 else ""
    print(f"  {oznaka:22s} s={seed}  c={c:5.2f} (idealno {r['c_test']:5.2f}){up}  "
          f"gdev: c_dev {d1:+6.1f}%  c_ratio {d3:+6.1f}%  izo {d2:+6.1f}%  "
          f"[{r['seconds']}s]",
          flush=True)
    append.write(OUT, r, KOLONE)


def gotovo():
    return append.done(OUT, KLJUC)


def main():
    print(f"MODEL={MODEL} N_EST={N_EST}", flush=True)
    rows, t0, g = [], time.time(), gotovo()

    print("=== SINTETICKI ===", flush=True)
    for xi in XI_SINT:
        for seed in SEEDS:
            k = dict(dataset=f"sint xi={xi}", model=MODEL, seed=seed,
                     n_est=N_EST)
            if append.key(k, KLJUC) in g:
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
            k = dict(dataset=ime, model=MODEL, seed=seed, n_est=N_EST)
            if append.key(k, KLJUC) in g:
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
