# -*- coding: utf-8 -*-
"""
Koliko tabelarni fundacioni modeli veruju kontekstu? Poredjenje sa Buhlmannovim optimumom.

ZASTO
-----
Aktuarska teorija kredibiliteta (Buhlmann 1967) resava tacno jedno pitanje: koliko tezine
dati POJEDINACNOM iskustvu a koliko KOLEKTIVU. Za model

    theta_i ~ (mu, tau^2)          # grupni efekat
    y_ij | theta_i ~ (theta_i, sigma^2)

najbolji linearni prediktor za grupu i je

    P_i = Z * prosek_grupe_i + (1 - Z) * mu,     Z = n / (n + k),   k = sigma^2 / tau^2

Z je poznat u zatvorenoj formi. A PFN radi istu stvar -- mesa kontekst sa priorom -- samo
sto mu je Z implicitan i nigde zapisan.

Ovde se taj implicitni Z MERI i poredi sa optimumom.

KAKO SE MERI Z
--------------
Iz predikcija po grupi:  pred_i - mu_hat = Z * (prosek_i - mu_hat) + sum
pa je Z nagib regresije kroz koordinatni pocetak:

    Z_hat = sum_i (pred_i - mu_hat)(prosek_i - mu_hat) / sum_i (prosek_i - mu_hat)^2

To je stabilnije od medijane kolicnika, koja eksplodira kad je prosek_i ~ mu_hat.

KAKO CITATI
-----------
  Z_hat ~ Z_opt        -> model je naucio klasicno optimalno skupljanje
  Z_hat < Z_opt        -> POTKREDITUJE: premalo veruje kontekstu, previse prioru
  Z_hat > Z_opt        -> PREKREDITUJE: previse veruje kontekstu

Veza sa ostatkom rada: "previse skuplja ka prioru" je ista tvrdnja kao "kompresuje rep",
samo izrazena u jeziku koji aktuari koriste sezdeset godina.

POKRETANJE
----------
    python -u credibility.py
    N_PO_GRUPI=5,20 K=4 PONAVLJANJA=5 MODELI=TabICLv2 python -u credibility.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, models, paths, quiet

quiet.silence()

N_PO_GRUPI = [int(v) for v in os.environ.get("N_PO_GRUPI", "2,5,10,20,50").split(",")]
K_LISTA = [float(v) for v in os.environ.get("K", "1,4,25").split(",")]
PONAVLJANJA = int(os.environ.get("PONAVLJANJA", "20"))
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM").split(",")
G = int(os.environ.get("N_GROUPS", "40"))
MU, TAU2 = 100.0, 25.0
OUT = os.environ.get("OUTPUT", "credibility.csv")
KOLONE = ["n_groups", "n_per_group", "k", "model", "repeat", "Z_opt", "Z_hat",
          "difference", "r2", "seconds", "reason"]
KEY = ["n_per_group", "k", "model", "repeat", "n_groups"]


def sredina(ime, Xtr, ytr, Xte, seed):
    return models.mean(ime, Xtr, ytr, Xte, seed=seed)


def jedan(n, k, ime, rep):
    r = np.random.default_rng(10_000 * rep + 100 * n + int(k))
    sig2 = TAU2 * k
    theta = r.normal(MU, np.sqrt(TAU2), G)
    gid = np.repeat(np.arange(G), n)
    y = r.normal(theta[gid], np.sqrt(sig2))

    # ID grupe je JEDINI informativan atribut; drugi je cist sum, da model ne bi
    # morao da radi sa jednom kolonom (neki omotaci to ne vole)
    X = np.c_[gid.astype(float), r.normal(size=len(gid))]
    Xq = np.c_[np.arange(G, dtype=float), np.zeros(G)]

    pred = sredina(ime, X, y, Xq, rep)
    prosek = np.array([y[gid == i].mean() for i in range(G)])
    mu_hat = float(y.mean())

    a = prosek - mu_hat
    b = pred - mu_hat
    den = float(np.sum(a * a))
    if den < 1e-12:
        raise RuntimeError("nema varijacije medju grupama")
    Z_hat = float(np.sum(a * b) / den)
    # koliko dobro linearni oblik uopste opisuje predikcije
    ss_res = float(np.sum((b - Z_hat * a) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    Z_opt = n / (n + k)
    return dict(Z_opt=Z_opt, Z_hat=Z_hat, difference=Z_hat - Z_opt, r2=r2, reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for n in N_PO_GRUPI:
        for k in K_LISTA:
            for ime in MODELI:
                for rep in range(PONAVLJANJA):
                    if append.key(dict(n_per_group=n, k=k, model=ime, repeat=rep), KEY[:4]) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        r = jedan(n, k, ime, rep)
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                    r.update(n_groups=G, n_per_group=n, k=k, model=ime, repeat=rep,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    if rep == 0:
                        poruka = r.get("reason") or (
                            f"Z_opt {r['Z_opt']:.3f}  Z_hat {r['Z_hat']:.3f}  "
                            f"({r['difference']:+.3f})  r2={r['r2']:.2f}")
                        print(f"  n={n:>3} k={k:>5.0f} {ime:10s} {poruka}"
                              f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.Z_hat.notna()]
    print(f"\n=== IMPLICITNI KREDIBILITET, {len(d)} pokretanja ===")
    for k in K_LISTA:
        print(f"\n  sigma^2/tau^2 = {k:.0f}")
        print(f"{'n':>6}{'Z_opt':>9}" + "".join(f"{m:>22}" for m in MODELI))
        for n in N_PO_GRUPI:
            g = d[(d.n_per_group == n) & (d.k == k)]
            if not len(g):
                continue
            red = f"{n:>6}{n/(n+k):>9.3f}"
            for m in MODELI:
                gg = g[g.model == m]
                if len(gg):
                    red += f"{gg.Z_hat.median():>13.3f} ({gg.difference.median():+.3f})"
                else:
                    red += f"{'-':>22}"
            print(red)
    print("\n=== koliko linearni (kredibilitetski) oblik uopste opisuje predikcije: R^2 ===")
    print(d.pivot_table(index="model", columns="k", values="r2", aggfunc="median")
          .round(3).to_string())
    print("\n=== prosecno odstupanje od optimuma, po modelu ===")
    print(d.groupby("model").difference.median().round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
