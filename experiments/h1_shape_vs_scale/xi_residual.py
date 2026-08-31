# -*- coding: utf-8 -*-
"""
Slabost 2 iz TEMA-POTVRDA-v2: merni instrument potcenjuje uz lokacijsku komponentu.

PROBLEM
-------
Ceo prvi deo rada meri implicirano xi iz odnosa Q(0.99)/Q(0.9). Taj odnos
pretpostavlja SKALNU familiju. Cim postoji lokacija mu, odnos se skuplja ka 1 i
implicirano xi ispadne manje: nominalno 0.7 vratilo je referencu 0.43
(findings/NALAZI.md 3.5). To je centralna alatka poglavlja 1 i nije provucena kroz nalaze.

ISPRAVKA
--------
Umesto odnosa nivoa, uzeti odnos RAZLIKA u odnosu na medijanu:

    r_res = (Q(0.99) - Q(0.5)) / (Q(0.9) - Q(0.5))

Dodavanje bilo kakvog mu se skrati i u brojiocu i u imeniocu, pa je mera
invarijantna na lokaciju po konstrukciji. xi se resava iz

    (core_q(0.99, xi) - core_q(0.5, xi)) / (core_q(0.9, xi) - core_q(0.5, xi)) = r_res

STA SE MERI
-----------
DEO 1 (bez modela, sekunde): na uzorcima iz GPD-a sa poznatim xi i dodatom
lokacijom mu, koliko svaki estimator promasi. Ocekivanje: sirovi pada sa mu,
rezidualni ne.

DEO 2 (sa modelima): ponoviti merenje udela pracenog gradijenta xi(x) -- glavni
broj rada, 42-45% -- sa OBA estimatora. Pitanje nije da li se broj menja nego
da li se menja ZAKLJUCAK (skala >> oblik).

PREDVIDJANJE, ZAPISANO PRE MERENJA
----------------------------------
  Q1. Rezidualni estimator vraca xi bez pristrasnosti po mu; sirovi pada
      monotono sa mu.
  Q2. Udeo pracenog gradijenta ce se promeniti za manje od 10 procentnih poena
      i zakljucak (skala ~94%, oblik znatno manje) ostaje.
  Ako Q2 padne -- dakle ako se pod ispravnim estimatorom oblik prati jednako
  dobro kao skala -- glavni rezultat rada pada i to se tako pise.

POKRETANJE
----------
    python -u xi_residual.py
    DEO=1 python -u xi_residual.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
HI, LO = 0.99, 0.9
N_EST = int(os.environ.get("N_EST", "4"))


DEO = os.environ.get("DEO", "12")
SEEDOVA = int(os.environ.get("SEEDS", "10"))
N_FIT = int(os.environ.get("N_FIT", "2000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
MODELI = os.environ.get("MODELS", "GBM,TabICLv2,TabPFN-V3").split(",")
OUT1 = os.environ.get("OUTPUT1", "xi_residual_part1.csv")
OUT2 = os.environ.get("OUTPUT2", "xi_residual_part2.csv")
COLUMNS2 = ["model", "seed", "d_true", "seconds", "reason",
            "xi_raw_t0", "xi_residual_t0", "xi_raw_t1", "xi_residual_t1",
            "xi_raw_t2", "xi_residual_t2", "xi_true_t0", "xi_true_t2"]
KEY2 = ["model", "seed"]

HI, LO, MED = 0.99, 0.9, 0.5


def xi_sirovi(q_hi, q_lo, q_med=None):
    """Postojeca mera: odnos nivoa. Pretpostavlja skalnu familiju."""
    if q_lo <= 0 or not np.isfinite(q_hi / q_lo):
        return np.nan
    return metrics.xi_from_ratio(q_hi / q_lo)


def _xi_iz_rezid_odnosa(r):
    """Inverzija rezidualnog odnosa (Q99-Q50)/(Q90-Q50) -> xi."""
    def f(x):
        c = metrics.gpd_quantile(np.array([HI, LO, MED]), x)
        return float((c[0] - c[2]) / (c[1] - c[2])) - r
    try:
        return brentq(f, -0.9, 6.0)
    except Exception:
        return np.nan


def xi_rezidualni(q_hi, q_lo, q_med):
    """Ispravka: odnos razlika u odnosu na medijanu. Invarijantno na lokaciju."""
    im = q_lo - q_med
    if im <= 1e-12:
        return np.nan
    r = (q_hi - q_med) / im
    def f(x):
        c = metrics.gpd_quantile(np.array([HI, LO, MED]), x)
        return float((c[0] - c[2]) / (c[1] - c[2])) - r
    try:
        return brentq(f, -0.9, 6.0)
    except Exception:
        return np.nan


# --------------------------------------------------------------- DEO 1
def deo1():
    print("=== DEO 1: estimatori na uzorcima sa poznatim xi i dodatom lokacijom ===",
          flush=True)
    redovi = []
    for xi in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for mu_odnos in [0.0, 0.5, 1.0, 2.0, 5.0]:
            for seed in range(SEEDOVA):
                rng = np.random.default_rng(9000 + seed)
                n = 200000
                U = rng.random(n)
                y0 = ((1 - U) ** (-xi) - 1) / xi          # GPD, skala 1
                mu = mu_odnos * np.median(y0)
                y = y0 + mu
                q = np.quantile(y, [HI, LO, MED])
                redovi.append(dict(xi=xi, mu_ratio=mu_odnos, seed=seed,
                                   xi_raw=xi_sirovi(q[0], q[1]),
                                   xi_residual=xi_rezidualni(q[0], q[1], q[2])))
    d = pd.DataFrame(redovi)
    d["gr_sirovi"] = d.xi_raw - d.xi
    d["gr_rezid"] = d.xi_residual - d.xi
    d.to_csv(paths.result(OUT1), index=False)

    print("\n  greska SIROVOG estimatora (xi_procenjeno - xi_pravo):")
    print(d.pivot_table(index="xi", columns="mu_ratio", values="gr_sirovi").round(3).to_string())
    print("\n  greska REZIDUALNOG estimatora:")
    print(d.pivot_table(index="xi", columns="mu_ratio", values="gr_rezid").round(3).to_string())
    print(f"\n  medijana |greska|: sirovi {d.gr_sirovi.abs().median():.3f} | "
          f"rezidualni {d.gr_rezid.abs().median():.3f}")
    return d


# --------------------------------------------------------------- DEO 2
def deo2():
    print("\n=== DEO 2: udeo pracenog gradijenta xi(x), oba estimatora ===", flush=True)
    gotovi = set()
    if paths.result(OUT2).exists():
        gotovi = append.done(OUT2, KEY2)
        if gotovi:
            print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    for seed in range(SEEDOVA):
        rng = np.random.default_rng(seed)
        Xtr, ytr, _, _ = generator.gpd(N_FIT, rng, xi=generator.XI_OF_X)
        Xte, _, _, xi_te = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
        # referentni gradijent: pravo xi po tercilima x4
        terc = np.quantile(Xte[:, 4], [1 / 3, 2 / 3])
        grupa = np.digitize(Xte[:, 4], terc)
        pravo = np.array([xi_te[grupa == g].mean() for g in range(3)])
        d_true = pravo[2] - pravo[0]

        for model in MODELI:
            if append.key(dict(model=model, seed=seed), KEY2) in gotovi:
                continue
            t1 = time.time()
            try:
                Q = models.quantiles(model, Xtr, ytr, Xte, seed=seed, levels=QS, n_est=N_EST)
                reason = ""
            except Exception as e:
                Q, reason = None, f"{type(e).__name__}: {e}"[:110]
            r = dict(model=model, seed=seed, d_true=d_true,
                     seconds=round(time.time() - t1, 1), reason=reason)
            if Q is not None:
                qs = list(np.round(QS, 4))
                ih, il = qs.index(round(HI, 4)), qs.index(round(LO, 4))
                im = qs.index(round(MED, 4))
                # AGREGACIJA PO TACKAMA: medijana odnosa preko test tacaka, isto kao
                # shape_of_x.py. Ranija verzija je uzimala odnos proseka --
                # drugi estimator, pa brojevi nisu bili uporedivi sa objavljenima.
                for g in range(3):
                    m = grupa == g
                    qh, ql, qm = Q[m, ih], Q[m, il], Q[m, im]
                    ok = ql > 1e-9
                    rs = float(np.median(qh[ok] / ql[ok])) if ok.any() else np.nan
                    r[f"xi_raw_t{g}"] = metrics.xi_from_ratio(rs) if np.isfinite(rs) else np.nan
                    im_ = ql - qm
                    ok2 = im_ > 1e-12
                    rr = float(np.median((qh[ok2] - qm[ok2]) / im_[ok2])) if ok2.any() else np.nan
                    r[f"xi_residual_t{g}"] = _xi_iz_rezid_odnosa(rr) if np.isfinite(rr) else np.nan
                r["xi_true_t0"], r["xi_true_t2"] = float(pravo[0]), float(pravo[2])
            append.write(OUT2, r, COLUMNS2)
            print(f"  {model:10s} s={seed} "
                  f"sirovi t0/t2={r.get('xi_raw_t0', float('nan')):.2f}/"
                  f"{r.get('xi_raw_t2', float('nan')):.2f} "
                  f"rezid t0/t2={r.get('xi_residual_t0', float('nan')):.2f}/"
                  f"{r.get('xi_residual_t2', float('nan')):.2f} "
                  f"{r['reason']} [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT2))
    # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""]
    if not len(d):
        return
    prav = float(d.xi_true_t2.mean() - d.xi_true_t0.mean())
    print(f"\npravi nagib xi(x) po tercilima: {prav:.3f}")
    print("\n=== agregacija KAO U RADU: prosek preko seedova, pa nagib ===")
    g = d.groupby("model").agg(
        sirovi_t0=("xi_sirovi_t0", "mean"), sirovi_t2=("xi_sirovi_t2", "mean"),
        rezid_t0=("xi_rezid_t0", "mean"), rezid_t2=("xi_rezid_t2", "mean"))
    g["nagib_sirovi"] = g.sirovi_t2 - g.sirovi_t0
    g["nagib_rezid"] = g.rezid_t2 - g.rezid_t0
    g["udeo_sirovi"] = g.nagib_sirovi / prav
    g["udeo_rezid"] = g.nagib_rezid / prav
    print(g[["nagib_sirovi", "nagib_rezid", "udeo_sirovi", "udeo_rezid"]].round(3).to_string())

    print("\n=== rasipanje PO SEEDU (nagib po seedu, pa raspodela) ===")
    d = d.copy()
    d["nagib_s"] = d.xi_sirovi_t2 - d.xi_sirovi_t0
    d["nagib_r"] = d.xi_rezid_t2 - d.xi_rezid_t0
    print(d.groupby("model")[["nagib_s", "nagib_r"]]
          .agg(["median", "min", "max", "std"]).round(3).to_string())

    print("\n=== bootstrap 95% interval za udeo (sirovi), preko seedova ===")
    rng = np.random.default_rng(0)
    for mdl, pod in d.groupby("model"):
        v = pod.nagib_s.to_numpy()
        bs = [rng.choice(v, len(v), replace=True).mean() / prav for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {mdl:10s} udeo {v.mean()/prav:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")
    print("\nQ2 PROLAZI ako se zakljucak ne menja (oblik i dalje znatno ispod skale ~0.94).")


if __name__ == "__main__":
    t0 = time.time()
    if "1" in DEO:
        deo1()
    if "2" in DEO:
        deo2()
    print(f"\nukupno {time.time()-t0:.0f}s")
