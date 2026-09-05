# -*- coding: utf-8 -*-
"""
Disocijacija na stvarnim podacima bez izbora obelezja i bez deljenja slabim brojem.

STA NE VALJA U POSTOJECOJ MERI
------------------------------
`dissociation_real.py` bira JEDNO obelezje, ono uz koje je |xi(t3) - xi(t1)| na fit
delu najvece, pa udeo racuna kao kolicnik gradijenta modela i referentnog
gradijenta uz to obelezje. Dve posledice, obe izmerene:

1. IZBOR JE MAKSIMUM PREKO KOLONA, pa i kad zavisnosti nema, izabere se kolona sa
   najvecom greskom procene. `reference_instrument.csv` meri nultu raspodelu tog
   izbora: na vecini skupova referentni gradijent NE prelazi 95. percentil onoga
   sto izbor proizvodi iz suma.
2. KOLICNIK SA MALIM IMENIOCEM EKSPLODIRA, otud raspon udela -2,80 do 6,16 i
   odustajanje od procenta u korist ranga. Ali rang je tu delom mehanicki: kad je
   referentni gradijent oblika sum, udeo oblika je daleko od 1 po konstrukciji,
   dok je gradijent skale uz isto obelezje stvaran, pa je udeo skale blizu 1.
   Poredjenje je time nagnuto u korist nalaza.

STA SE OVDE MERI UMESTO TOGA
----------------------------
Model se fituje JEDNOM po (skup, model, seed), a onda se ista predikcija gleda
kroz tercile SVAKOG upotrebljivog obelezja. Za svako obelezje se dobija par:

    ref_grad_j   referentni gradijent repa kroz tercile obelezja j
    mod_grad_j   isti gradijent iz predikcije modela

Umesto kolicnika za jedno j, racuna se NAGIB regresije mod_grad na ref_grad kroz
sva j, kroz koordinatni pocetak. Nema izbora, pa nema ni njegove nulte raspodele,
i nema imenioca koji moze biti nula.

ATENUACIJA, I ZASTO SE MORA ISPRAVITI
-------------------------------------
Referentni gradijent je i sam procena sa greskom, a greska u regresoru vuce nagib
ka nuli. To nije sitnica nego bas ono sto bi moglo da proizvede ceo nalaz: oblik
se procenjuje mnogo losije od skale, pa bi jaci pad nagiba kod oblika mogao da
bude posledica losijeg instrumenta, a ne slepila modela.

Zato se, uz sirovi nagib, racuna i nagib ispravljen za poznatu varijansu greske
(regresija sa greskom u promenljivoj):

    nagib = sum(ref * mod) / (sum(ref^2) - sum(var_ref))

gde var_ref dolazi iz bootstrap-a nad test redovima, po obelezju. Isti postupak i
za skalu, pa se dve polovine porede posle iste ispravke.

KAKO CITATI
-----------
  slope_scale ~ 1, slope_shape mali        disocijacija stoji i bez izbora
  oba nagiba padnu posle ispravke podjednako   razlika nije instrument
  slope_shape poraste do slope_scale posle
  ispravke                                  nalaz JE bio atenuacija, i to je
                                            deveti pad na sopstvenoj proveri

Rezultati u `dissociation_real_slope.csv`.

POKRETANJE
----------
    python -u experiments/h1_shape_vs_scale/dissociation_real_slope.py
    MODELS=TabICLv2 DATASETS=houses SEEDS=1 python -u dissociation_real_slope.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
HI, LO, MID = 0.99, 0.9, 0.5
N_FIT, N_TEST = 3000, 6000
N_TERCILA = 3
MIN_PO_TERCILU = 400
N_BOOT = int(os.environ.get("N_BOOT", "100"))
N_EST = int(os.environ.get("N_EST", "4"))
SEEDOVA = int(os.environ.get("SEEDS", "3"))
PODRAZUMEVANI = ["OnlineNewsPopularity", "diamonds", "particulate-matter-ukair-2017",
                 "Buzzinsocialmedia_Twitter", "CPS1988", "218_house_8L",
                 "superconduct", "houses", "Allstate_Claims_Severity", "house_16H"]
SKUPOVI = os.environ.get("DATASETS", ",".join(PODRAZUMEVANI)).split(",")
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM"))
OUT = os.environ.get("OUTPUT", "dissociation_real_slope.csv")

KOLONE = ["dataset", "model", "seed", "n_est", "n_features",
          "slope_shape", "slope_shape_corrected", "r_shape",
          "slope_scale", "slope_scale_corrected", "r_scale",
          "var_share_shape", "var_share_scale", "seconds", "reason"]
KEY = ["dataset", "model", "seed", "n_est"]


def skala_regresijom(Xf, yf):
    ok = yf > 0
    D = np.c_[np.ones(ok.sum()), Xf[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(yf[ok]), rcond=None)
    return lambda X: np.clip(np.exp(np.c_[np.ones(len(X)), X] @ b), 1e-9, None)


def tercili(v, ivice):
    return np.clip(np.digitize(v, ivice[1:-1]), 0, N_TERCILA - 1)


def grad_ref(y, s_hat, terc):
    """Referentni gradijent repa: Hill nad rezidualima, t3 minus t1."""
    xs = []
    for t in (0, N_TERCILA - 1):
        z = y[terc == t] / s_hat[terc == t]
        z = z[np.isfinite(z) & (z > 0)]
        xs.append(metrics.hill(z) if len(z) >= MIN_PO_TERCILU else np.nan)
    return xs[1] - xs[0]


def grad_model(q, terc):
    """Gradijent repa iz predikcije, lokacijski invarijantnim odnosom."""
    xs = []
    for t in (0, N_TERCILA - 1):
        sel = terc == t
        if sel.sum() < MIN_PO_TERCILU:
            return np.nan
        hi, lo, mid = (q[sel, QS.index(HI)], q[sel, QS.index(LO)],
                       q[sel, QS.index(MID)])
        ok = (lo - mid) > 1e-12
        if not ok.any():
            return np.nan
        xs.append(metrics.xi_from_residual_ratio(
            float(np.median((hi[ok] - mid[ok]) / (lo[ok] - mid[ok])))))
    return xs[1] - xs[0]


def grad_skale(v_t1, v_t3):
    """Gradijent skale: log-odnos medijana izmedju spoljnih tercila."""
    if v_t1 <= 0 or v_t3 <= 0:
        return np.nan
    return float(np.log(v_t3 / v_t1))


def nagib(x, y, var_x=None):
    """Nagib kroz koordinatni pocetak, opciono ispravljen za varijansu greske u x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    den = float(np.sum(x ** 2))
    raw = float(np.sum(x * y) / den) if den > 0 else np.nan
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
    corrected, var_share = np.nan, np.nan
    if var_x is not None:
        v = np.asarray(var_x, float)[ok]
        v = float(np.nansum(v))
        var_share = v / den if den > 0 else np.nan
        # Imenilac je varijansa PRAVOG regresora: izmerena minus greska merenja.
        if den - v > 0.05 * den:
            corrected = float(np.sum(x * y) / (den - v))
    return raw, corrected, r, var_share


def jedan(skup, ime, seed, kes):
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"], positive_only=True)
    X, y = kes[skup]
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]

    s_hat = skala_regresijom(Xf, yf)
    s_te = s_hat(Xte)
    # Model se fituje jednom; tercili su naknadno grupisanje ISTIH predikcija,
    # pa cena ne raste sa brojem obelezja.
    q = models.quantiles(ime, Xf, yf, Xte, seed=seed, levels=QS, n_est=N_EST)

    ref_sh, mod_sh, var_sh = [], [], []
    ref_sc, mod_sc, var_sc = [], [], []
    for j in range(Xf.shape[1]):
        v = Xf[:, j]
        if len(np.unique(v)) < N_TERCILA * 3:
            continue
        iv = np.quantile(v, np.linspace(0, 1, N_TERCILA + 1))
        if len(np.unique(iv)) < N_TERCILA + 1:
            continue
        terc = tercili(Xte[:, j], iv)
        if min((terc == t).sum() for t in range(N_TERCILA)) < MIN_PO_TERCILU:
            continue

        g_ref, g_mod = grad_ref(yte, s_te, terc), grad_model(q, terc)
        if not (np.isfinite(g_ref) and np.isfinite(g_mod)):
            continue

        # Varijansa greske referentnog gradijenta, bootstrap nad test redovima.
        b = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(yte), len(yte))
            b.append(grad_ref(yte[k], s_te[k], terc[k]))
        b = np.array([x for x in b if np.isfinite(x)])
        if len(b) < N_BOOT // 2:
            continue
        ref_sh.append(g_ref)
        mod_sh.append(g_mod)
        var_sh.append(float(np.var(b)))

        # Skala: ista tri tercila, medijane umesto repa.
        e1, e3 = (float(np.median(yte[terc == 0])),
                  float(np.median(yte[terc == N_TERCILA - 1])))
        m1, m3 = (float(np.median(q[terc == 0, QS.index(MID)])),
                  float(np.median(q[terc == N_TERCILA - 1, QS.index(MID)])))
        gs_ref, gs_mod = grad_skale(e1, e3), grad_skale(m1, m3)
        bs = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(yte), len(yte))
            t = terc[k]
            if min((t == u).sum() for u in (0, N_TERCILA - 1)) < 2:
                continue
            bs.append(grad_skale(float(np.median(yte[k][t == 0])),
                                 float(np.median(yte[k][t == N_TERCILA - 1]))))
        bs = np.array([x for x in bs if np.isfinite(x)])
        if np.isfinite(gs_ref) and np.isfinite(gs_mod) and len(bs) > 2:
            ref_sc.append(gs_ref)
            mod_sc.append(gs_mod)
            var_sc.append(float(np.var(bs)))

    if len(ref_sh) < 3:
        raise RuntimeError(f"samo {len(ref_sh)} upotrebljivih obelezja")

    a_raw, a_cor, a_r, a_v = nagib(ref_sh, mod_sh, var_sh)
    b_raw, b_cor, b_r, b_v = nagib(ref_sc, mod_sc, var_sc)
    return dict(n_features=len(ref_sh),
                slope_shape=a_raw, slope_shape_corrected=a_cor, r_shape=a_r,
                slope_scale=b_raw, slope_scale_corrected=b_cor, r_scale=b_r,
                var_share_shape=a_v, var_share_scale=b_v, reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                k = dict(dataset=skup, model=ime, seed=seed, n_est=N_EST)
                if append.key(k, KEY) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(skup, ime, seed, kes)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                r.update(k, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                poruka = r.get("reason") or (
                    f"oblik {r['slope_shape']:+.2f} (isp. {r['slope_shape_corrected']:+.2f}) | "
                    f"skala {r['slope_scale']:+.2f} (isp. {r['slope_scale_corrected']:+.2f}) | "
                    f"{r['n_features']} obelezja")
                print(f"  {skup:30s} {ime:10s} s={seed}  {poruka}  [{r['seconds']}s]",
                      flush=True)
    report(t0)


def report(t0=None):
    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        return
    print("\n=== nagib po modelu (medijana preko skupova i seedova) ===")
    print(d.groupby("model")[["slope_scale", "slope_scale_corrected",
                              "slope_shape", "slope_shape_corrected"]]
          .median().round(3).to_string())
    print("\n=== udeo varijanse referentnog gradijenta koji je greska merenja ===")
    print(d.groupby("model")[["var_share_scale", "var_share_shape"]]
          .median().round(3).to_string())
    print("\nAko je var_share kod oblika mnogo veci nego kod skale, sirovi nagibi se")
    print("ne smeju porediti bez ispravke; upravo zato ispravka i postoji.")
    from common import stats
    r = stats.paired_by_cluster(d, "slope_scale_corrected", "slope_shape_corrected",
                                cluster="dataset")
    if r["clusters"]:
        print(f"\nskala blize 1 od oblika: {r['wins']}/{r['n']} redova, "
              f"{r['cluster_wins']}/{r['clusters']} skupova, p = {r['cluster_p']:.3g}")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
