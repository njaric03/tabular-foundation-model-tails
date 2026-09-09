# -*- coding: utf-8 -*-
"""
Uklanja li ROBUSNA obrada targeta katastrofalan odgovor na jedan outlier?

ZASTO
-----
`findings/h2/preprocessing_asymmetry.md`: sva cetiri fundaciona modela standardizuju target sa
mean/std, dok atribute stite od outliera. Posledica je da jedna tacka pomeri skalu na
koju je vezan ceo izlazni sloj: TabICL -31% na Q(0.99), TabDPT -307% na sredini,
TabFM +1572%.

Popravka je ocigledna -- robusna procena skale targeta -- ali dok se ne izmeri, to je
samo tvrdnja.

ZAMKA KOJU TREBA IZBECI
-----------------------
Winsorizacija po definiciji uklanja ubacenu tacku, pa ce naravno smanjiti odgovor.
To samo po sebi ne dokazuje nista. Zato se mere DVE stvari, i popravka vredi samo ako
prodje obe:

  KORIST   koliko se smanji odgovor na ubacenu tacku (kontaminirani podaci)
  CENA     koliko se pogorsa pinball gubitak na CISTIM podacima

Popravka koja ubije odgovor a pokvari cist slucaj nije popravka nego brisanje repa.

VARIJANTE
---------
  sirovo   bez ikakve obrade (polazna tacka)
  log      fituj na log(y), vrati eksponenciranjem
  winsor   odsecanje gornjih 0.5% pre fitovanja, u originalnim jedinicama
  rang     rang-normalna transformacija, inverzija preko empirijskih kvantila

Kvantili su ekvivarijantni pod monotonom transformacijom, pa su sve cetiri legitimne.

POKRETANJE
----------
    python -u repair.py
    XI=0.9 SEEDS=3 MODELS=TabICLv2 python -u repair.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import append, generator, models, paths, quiet

quiet.silence()


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.7,0.9").split(",")]
VARIJANTE = os.environ.get("VARIANTS", "sirovo,log,winsor,rang,asinh").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3"))
N_EST = int(os.environ.get("N_EST", "1"))
N_TRAIN, N_TEST = 2000, 900
NIVOI = [0.5, 0.9, 0.99, 0.999]
WINSOR_UDEO = 0.005
OUT = os.environ.get("OUTPUT", "repair.csv")
KOLONE = ["n_est", "xi", "model", "variant", "seed", "response_q99", "response_q999",
          "pb50", "pb90", "pb99", "pb999", "twcrps",
          "seconds", "reason"]
KEY = ["xi", "model", "variant", "seed", "n_est"]


def sirovi_kvantili(ime, Xtr, ytr, Xte, seed):
    if ime == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=N_EST, device="cpu", random_state=seed)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=NIVOI))
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=N_EST, device="cpu", random_state=seed,
                        ignore_pretraining_limits=True)
    m.fit(Xtr, ytr)
    return np.stack([np.asarray(a) for a in
                     m.predict(Xte, output_type="quantiles", quantiles=NIVOI)], axis=1)


def kvantili(varijanta, ime, Xtr, ytr, Xte, seed):
    """Fituj pod datom obradom targeta, vrati kvantile u ORIGINALNIM jedinicama."""
    y = np.maximum(np.asarray(ytr, dtype=float), 1e-12)

    if varijanta == "raw":
        return sirovi_kvantili(ime, Xtr, y, Xte, seed)

    if varijanta == "log":
        return np.exp(sirovi_kvantili(ime, Xtr, np.log(y), Xte, seed))

    if varijanta == "winsor":
        # odseci samo gornjih WINSOR_UDEO; ostaje u originalnim jedinicama pa
        # nikakva inverzija nije potrebna
        kapa = np.quantile(y, 1 - WINSOR_UDEO)
        return sirovi_kvantili(ime, Xtr, np.minimum(y, kapa), Xte, seed)

    if varijanta == "asinh":
        # Meko odsecanje: monotono i invertibilno, pa su kvantili ekvivarijantni.
        # Za vrednosti u telu je prakticno identiteta, a polugu ekstrema na sd
        # ogranicava logaritamski umesto da ih brise (za razliku od winsorizacije).
        med = np.median(y); iqr = np.subtract(*np.percentile(y, [75, 25])) or 1.0
        z = med + iqr * np.arcsinh((y - med) / iqr)
        q = sirovi_kvantili(ime, Xtr, z, Xte, seed)
        return med + iqr * np.sinh((q - med) / iqr)

    if varijanta == "rank":
        # rang -> normalni skorovi; inverzija linearnom interpolacijom empirijskih kvantila
        n = len(y)
        r = np.argsort(np.argsort(y))
        z = norm.ppf((r + 0.5) / n)
        q_z = sirovi_kvantili(ime, Xtr, z, Xte, seed)
        ys = np.sort(y)
        zs = norm.ppf((np.arange(n) + 0.5) / n)
        return np.interp(q_z, zs, ys)

    raise ValueError(varijanta)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def jedan(xi, ime, varijanta, seed):
    rng = np.random.default_rng(seed)
    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = _p.X, _p.y
    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, yte = _p.X, _p.y

    # CENA: kvalitet na cistim podacima
    q0 = kvantili(varijanta, ime, Xtr, ytr, Xte, seed)
    # NE koristiti int(a*100) kao kljuc: int(0.999*100) == 99, pa bi 0.999 prepisao 0.99.
    IME_NIVOA = {0.5: "pb50", 0.9: "pb90", 0.99: "pb99", 0.999: "pb999"}
    pb = {IME_NIVOA[a]: pinball(yte, q0[:, i], a) for i, a in enumerate(NIVOI)}

    # KORIST: odgovor na jednu ubacenu tacku
    x0 = np.zeros((1, Xtr.shape[1]))
    q1 = kvantili(varijanta, ime, np.vstack([Xtr, x0]),
                  np.concatenate([ytr, [100 * ytr.max()]]), Xte, seed)
    def odg(i):
        a, b = q0[:, i], q1[:, i]
        ok = a > 1e-9
        return float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan

    # twCRPS sa pragom na 0.9 kvantilu test uzorka: mera koja tezi bas gornji rep
    prag = float(np.quantile(yte, 0.90))
    tw = float(np.mean([pinball(np.maximum(yte, prag), np.maximum(q0[:, i], prag), a)
                        for i, a in enumerate(NIVOI)]))

    return dict(response_q99=odg(NIVOI.index(0.99)),
                response_q999=odg(NIVOI.index(0.999)),
                twcrps=tw, reason="", **pb)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for varijanta in VARIJANTE:
                for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                    if append.key(dict(xi=xi, model=ime, variant=varijanta, seed=seed,
                                       n_est=N_EST), KEY) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        r = jedan(xi, ime, varijanta, seed)
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                    r.update(n_est=N_EST, xi=xi, model=ime, variant=varijanta, seed=seed,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    poruka = r.get("reason") or (
                        f"odgovor {r['response_q99']:+7.1%}  pb99 {r['pb99']:.4f}")
                    print(f"  xi={xi} {ime:10s} {varijanta:7s} s={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.response_q99.notna()]
    print("\n=== KORIST: odgovor na jednu ubacenu tacku (medijana) ===")
    print(d.pivot_table(index=["xi", "model"], columns="variant",
                        values="response_q99", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== CENA: pinball 0.99 na CISTIM podacima, relativno prema `sirovo` ===")
    for (xi, ime), g in d.groupby(["xi", "model"]):
        baza = g[g.variant == "raw"].pb99.median()
        red = "  ".join(f"{v}: {100*(g[g.variant==v].pb99.median()-baza)/baza:+6.1f}%"
                        for v in VARIJANTE if v != "raw")
        print(f"  xi={xi} {ime:<12} {red}")
    print("\n=== CENA na medijani (pinball 0.5) ===")
    for (xi, ime), g in d.groupby(["xi", "model"]):
        baza = g[g.variant == "raw"].pb50.median()
        red = "  ".join(f"{v}: {100*(g[g.variant==v].pb50.median()-baza)/baza:+6.1f}%"
                        for v in VARIJANTE if v != "raw")
        print(f"  xi={xi} {ime:<12} {red}")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
