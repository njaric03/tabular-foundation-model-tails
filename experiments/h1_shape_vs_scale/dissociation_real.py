# -*- coding: utf-8 -*-
"""
Disocijacija nivo/oblik na STVARNIM podacima.

ZASTO
-----
`findings/h1/dissociation.md` je u celini na sintetickom generatoru, gde se pravo xi(x) zna.
Na stvarnim podacima se istina ne zna, pa se poredi sa EMPIRIJSKOM referencom iz
izdvojenog test skupa: sta bi savrsen model morao da vidi, vidi se iz samih y.

Bez ovoga ceo glavni nalaz stoji samo na jednom generatoru, i to je prvo sto ce se
pitati na odbrani.

DIZAJN
------
    fit   3000 redova: model + log-linearni model skale
    test  6000 redova: i referenca i merenje

  1. IZBOR OBELEZJA -- iskljucivo na `fit` delu, da nema selekcije na test skupu.
     Za svako obelezje: tercili, pa procena xi po tercilu iz reziduala. Bira se ono sa
     najvecim rasponom |xi(T3) - xi(T1)|. To je obelezje uz koje rep NAJVISE varira.

  2. MERENJE na `test`, po tercilima sa granicama iz `fit`:

     OBLIK   referenca = xi po tercilu iz test reziduala (Hill)
             model     = implicirano xi iz medijane Q(0.99)/Q(0.9)
             udeo      = (model T3 - model T1) / (ref T3 - ref T1)

     SKALA   referenca = log-odnos empirijskih medijana T3/T1
             model     = log-odnos predvidjenih medijana T3/T1
             udeo      = model / referenca

  3. KONTROLA: isto merenje na NASUMICNOM obelezju. Ako i tamo ispadne isto, mera
     ne meri nista specificno.

Rezultati u `dissociation_real.csv`.

POKRETANJE
----------
    python -u dissociation_real.py
    SKUPOVI=diamonds,houses SEEDOVA=1 MODELI=TabICLv2 python -u dissociation_real.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
HI, LO = 0.99, 0.9
N_EST = int(os.environ.get("N_EST", "4"))


PODRAZUMEVANI = ["OnlineNewsPopularity", "diamonds", "particulate-matter-ukair-2017",
                 "Buzzinsocialmedia_Twitter", "CPS1988", "218_house_8L",
                 "superconduct", "houses", "Allstate_Claims_Severity", "house_16H"]
SKUPOVI = os.environ.get("DATASETS", ",".join(PODRAZUMEVANI)).split(",")
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "3"))
N_FIT, N_TEST = 3000, 6000
N_TERCILA = 3
MIN_PO_TERCILU = 400
OUT = os.environ.get("OUTPUT", "dissociation_real.csv")
KLJUC = ["dataset", "model", "seed", "n_est"]
KOLONE = ["dataset", "model", "seed", "n_est", "feature", "shape_share", "scale_share",
          "ref_xi_t1", "ref_xi_t3", "mod_xi_t1", "mod_xi_t3",
          "shape_share_control", "ref_range", "seconds", "reason",
          # dodato 24.8.2026: lokacijski invarijantan estimator (findings/h1/xi_residual.md).
          # Sirovi odnos Q99/Q90 se pod lokacijskom komponentom skuplja ka 1, pa
          # potcenjuje xi modela. Na stvarnim podacima lokacija po pravilu postoji,
          # a referenca (Hill na rezidualima) tu pristrasnost nema -- pa je udeo
          # oblika sistematski potcenjen. Ovde se meri i jedno i drugo.
          "shape_share_residual", "mod_xi_r_t1", "mod_xi_r_t3"]


def _xi_rezid(q_hi, q_lo, q_med):
    """xi iz odnosa (Q99-Q50)/(Q90-Q50). Lokacija se skrati."""
    im = q_lo - q_med
    ok = im > 1e-12
    if not np.any(ok):
        return np.nan
    r = float(np.median((q_hi[ok] - q_med[ok]) / im[ok]))
    if not np.isfinite(r):
        return np.nan
    from scipy.optimize import brentq
    def f(x):
        c = metrics.gpd_quantile(np.array([HI, LO, 0.5]), x)
        return float((c[0] - c[2]) / (c[1] - c[2])) - r
    try:
        return brentq(f, -0.9, 6.0)
    except Exception:
        return np.nan


def skala_model(Xf, yf):
    """Log-linearna skala naucena na `fit`, primenjiva na bilo koji X."""
    ok = yf > 0
    D = np.c_[np.ones(ok.sum()), Xf[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(yf[ok]), rcond=None)
    return lambda X: np.clip(np.exp(np.c_[np.ones(len(X)), X] @ b), 1e-9, None)


def xi_po_tercilu(y, s_hat, terc):
    """Hillov indeks repa po tercilu, na reziduale y/s_hat."""
    out = []
    for t in range(N_TERCILA):
        z = y[terc == t] / s_hat[terc == t]
        z = z[np.isfinite(z) & (z > 0)]
        out.append(metrics.hill(z) if len(z) >= MIN_PO_TERCILU else np.nan)
    return out


def tercili(v, ivice):
    return np.clip(np.digitize(v, ivice[1:-1]), 0, N_TERCILA - 1)


def izaberi_obelezje(Xf, yf, s_hat_f):
    """Obelezje uz koje rep najvise varira -- biran ISKLJUCIVO na fit delu."""
    najbolje, najveci, ivice_naj = None, -np.inf, None
    for j in range(Xf.shape[1]):
        v = Xf[:, j]
        if len(np.unique(v)) < N_TERCILA * 3:
            continue
        iv = np.quantile(v, np.linspace(0, 1, N_TERCILA + 1))
        if len(np.unique(iv)) < N_TERCILA + 1:
            continue
        xs = xi_po_tercilu(yf, s_hat_f, tercili(v, iv))
        if any(not np.isfinite(x) for x in xs):
            continue
        r = abs(xs[2] - xs[0])
        if r > najveci:
            najbolje, najveci, ivice_naj = j, r, iv
    return najbolje, ivice_naj


def meri(j, ivice, Xte, yte, s_hat_te, q):
    terc = tercili(Xte[:, j], ivice)
    if min((terc == t).sum() for t in range(N_TERCILA)) < MIN_PO_TERCILU:
        return None
    ref = xi_po_tercilu(yte, s_hat_te, terc)
    mod, mod_r = [], []
    for t in range(N_TERCILA):
        sel = terc == t
        num, den = q[sel, QS.index(HI)], q[sel, QS.index(LO)]
        med = q[sel, QS.index(0.5)]
        ok = den > 1e-9
        mod.append(metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))) if ok.any() else np.nan)
        mod_r.append(_xi_rezid(num, den, med))
    raspon = ref[2] - ref[0]
    udeo_ob = (mod[2] - mod[0]) / raspon if abs(raspon) > 0.05 else np.nan
    udeo_ob_r = (mod_r[2] - mod_r[0]) / raspon if abs(raspon) > 0.05 else np.nan

    med_ref = [float(np.median(yte[terc == t])) for t in range(N_TERCILA)]
    med_mod = [float(np.median(q[terc == t, QS.index(0.5)])) for t in range(N_TERCILA)]
    udeo_sk = np.nan
    if med_ref[0] > 0 and med_ref[2] > 0 and med_mod[0] > 0 and med_mod[2] > 0:
        d = np.log(med_ref[2] / med_ref[0])
        if abs(d) > 0.05:
            udeo_sk = np.log(med_mod[2] / med_mod[0]) / d
    return dict(shape_share=udeo_ob, scale_share=udeo_sk, ref_range=raspon,
                ref_xi_t1=ref[0], ref_xi_t3=ref[2], mod_xi_t1=mod[0], mod_xi_t3=mod[2],
                shape_share_residual=udeo_ob_r, mod_xi_r_t1=mod_r[0], mod_xi_r_t3=mod_r[2])


def jedan(skup, ime, seed, kes):
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"])
    X, y = kes[skup]
    rng = np.random.default_rng(seed)
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]
    if (yf > 0).sum() < N_FIT * 0.5:
        pom = min(yf.min(), yte.min())
        yf, yte = yf - pom + 1e-6, yte - pom + 1e-6

    sm = skala_model(Xf, yf)
    j, ivice = izaberi_obelezje(Xf, yf, sm(Xf))
    if j is None:
        raise RuntimeError("nijedno obelezje nije proslo izbor")

    q = models.quantiles(ime, Xf, yf, Xte, seed=seed, levels=QS, n_est=N_EST)
    r = meri(j, ivice, Xte, yte, sm(Xte), q)
    if r is None:
        raise RuntimeError("tercili pretanki na test skupu")

    # kontrola: nasumicno obelezje, isti postupak
    jj = int(rng.integers(X.shape[1]))
    iv2 = np.quantile(Xf[:, jj], np.linspace(0, 1, N_TERCILA + 1))
    rk = meri(jj, iv2, Xte, yte, sm(Xte), q) if len(np.unique(iv2)) == N_TERCILA + 1 else None
    r["shape_share_control"] = rk["shape_share"] if rk else np.nan
    r["feature"] = j
    r["reason"] = ""
    return r


def main():
    gotovi = append.done(OUT, KLJUC)
    if gotovi:
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                k = dict(dataset=skup, model=ime, seed=seed, n_est=N_EST)
                if append.key(k, KLJUC) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(skup, ime, seed, kes)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                r.update(k, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                if r.get("reason"):
                    print(f"  {skup[:26]:27s} {ime:10s} s={seed} -- {r['reason']}", flush=True)
                else:
                    print(f"  {skup[:26]:27s} {ime:10s} s={seed}  "
                          f"oblik {r['shape_share']:>7.0%}  skala {r['scale_share']:>7.0%}  "
                          f"(ref xi {r['ref_xi_t1']:.2f}->{r['ref_xi_t3']:.2f})  "
                          f"[{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""]
    print(f"\n=== UDEO PRAVE PROMENE, stvarni podaci (n={len(d)}) ===")
    print(d.pivot_table(index="dataset", columns="model",
                        values=["shape_share", "scale_share"], aggfunc="mean")
          .map(lambda v: f"{v:.0%}").to_string())
    print("\n=== zbirno ===")
    for m, g in d.groupby("model"):
        print(f"  {m:<12} skala {g.scale_share.median():>5.0%}  oblik {g.shape_share.median():>5.0%}"
              f"  (kontrola {g.shape_share_control.median():>5.0%})")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
