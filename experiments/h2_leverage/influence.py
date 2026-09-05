# -*- coding: utf-8 -*-
"""
Funkcija uticaja u kontekstu: sta model uradi kad mu se u kontekst ubaci JEDNA
ekstremna tacka?

ZASTO
-----
Sve dosadasnje mere gledaju model kao crnu kutiju nad celim skupom. Ova gleda njegovu
IMPLICITNU POLITIKU prema outlierima, i to je pitanje na koje se odgovor ne moze procitati
iz dokumentacije:

  - pravi Bayesovac pod tesko-repnim priorom bi PODIGAO rep: jedna ekstremna tacka je
    evidencija o tome da je rep tezak;
  - robustan estimator bi je IGNORISAO;
  - model koji klipuje outliere u pretprocesiranju bi imao SATURIRAN odgovor.

Kriva odgovora na dozu razdvaja ta tri ponasanja. Radi i za modele koji ne izlazu uslovnu
raspodelu, jer se moze meriti i sama sredina.

DIZAJN
------
Fituje se na istom skupu dvaput: bez i sa jednim dodatnim redom (x0, y0), gde je
x0 = 0 (centar prostora atributa), a y0 = m * max(y_train), m iz {1, 3, 10, 100}.

Mere se dve stvari:

  LOKALNO   promena predikcije na 10% test tacaka najblizih x0
  GLOBALNO  promena na svim test tackama

i to na Q(0.5), Q(0.9), Q(0.99) plus implicirano xi.

KAKO CITATI
-----------
  odgovor raste sa m, i na repu jace nego na medijani  -> Bayesovsko ponasanje
  odgovor ~ 0                                          -> robustno / ignorise
  odgovor saturira pa stane                            -> klipovanje u pretprocesiranju
  odgovor jak i GLOBALNO a ne samo lokalno             -> jedna tacka pomera ceo fit

Rezultati u `influence.csv`.

POKRETANJE
----------
    python -u influence.py
    XI=0.7 SEEDS=3 MODELS=TabICLv2 python -u influence.py

    # doza kao ciljani pomeraj sd: realan opseg iz `prevalence_data.csv`
    DOSE_MODE=sd DOSES=2,4,10,20 OUTPUT=influence_sd.csv python -u influence.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.3,0.7").split(",")]
DOZE = [float(v) for v in os.environ.get("DOSES", "1,3,10,100").split(",")]
# Sta doza znaci. "max": y0 = doza * max(y_train), kako je i mereno do sada.
# "sd": doza JE ciljani pomeraj sd, a y0 se resava iz njega
# (`metrics.y0_for_sd_shift`).
#
# Zasto postoji drugi rezim. max(y_train) je i sam tesko-repna slucajna
# velicina, pa ista nominalna doza nije ista intervencija po seedovima: pri
# xi = 0.9 i dozi 100 stvarni pomeraj sd ide od 50.3 do 97.1 preko pet seedova.
# Deo rasipanja koji je devet puta obarao nalaze (`findings/NALAZI.md` §6) je
# zato rasipanje TRETMANA, ne modela. Uz to, prevalencija poluge je merena bas
# u pomeraju sd (najgori od 99 skupova: 19.9), pa se tek u ovom rezimu sintetika
# i stvarni podaci crtaju na istoj osi.
DOSE_MODE = os.environ.get("DOSE_MODE", "max")
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM"))
N_EST = int(os.environ.get("N_EST", "1"))
N_TRAIN, N_TEST = 2000, 900
NIVOI = [0.5, 0.9, 0.99]
LEVEL_NAME = {0.5: "q50", 0.9: "q90", 0.99: "q99", 0.999: "q999"}
UDEO_BLIZU = 0.10
# Eksperiment 4: gde u prostoru atributa outlier stoji. Eksperiment 5: velicina konteksta.
POZICIJA = os.environ.get("POSITION", "centre")   # centar | rub | slucajna
N_TRAIN = int(os.environ.get("N_TRAIN", str(N_TRAIN)))
OUT = os.environ.get("OUTPUT", "influence.csv")
KOLONE = ["xi", "model", "dose", "dose_mode", "seed", "position", "n_train", "n_est",
          # sd_shift je IZMERENA jacina intervencije, a `dose` je samo nominalna.
          # Bez nje se sintetika ne moze uporediti sa prevalencijom poluge, koja
          # je merena u pomeraju sd (`prevalence_data.csv`).
          "sd_shift", "y0_over_max",
          "local_q50", "local_q90", "local_q99", "global_q50", "global_q90", "global_q99",
          "xi_without", "xi_with", "seconds", "reason"]
# Svaki podesivi parametar mora u kljuc, inace pokretanje sa drugom vrednoscu
# nadje "vec uradjeno" i tiho ne uradi nista. `dose_mode` je u kljucu jer doza 10
# znaci dve razlicite intervencije u dva rezima.
KLJUC = ["xi", "model", "dose", "dose_mode", "seed", "position", "n_train", "n_est"]


def kvantili(ime, Xtr, ytr, Xte, seed):
    return models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


def xi_iz(q):
    num, den = q[:, NIVOI.index(0.99)], q[:, NIVOI.index(0.9)]
    ok = den > 1e-9
    return metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))) if ok.any() else np.nan


def jedan(xi, ime, doza, seed, kes):
    kljuc = (xi, ime, seed)
    rng = np.random.default_rng(seed)
    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = _p.X, _p.y
    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, _ = _p.X, _p.y

    if kljuc not in kes:
        kes[kljuc] = (kvantili(ime, Xtr, ytr, Xte, seed), Xtr, ytr, Xte)
    q_bez, Xtr, ytr, Xte = kes[kljuc]

    if POZICIJA == "edge":
        x0 = 2.0 * np.ones((1, Xtr.shape[1]))       # ~2 sd od centra, retko naseljeno
    elif POZICIJA == "slucajna":
        x0 = Xtr[[int(np.random.default_rng(seed).integers(len(Xtr)))]].copy()
    else:
        x0 = np.zeros((1, Xtr.shape[1]))
    if DOSE_MODE == "sd":
        v = metrics.y0_for_sd_shift(ytr, doza)
        if not np.isfinite(v):
            return dict(reason=f"sd shift {doza} unreachable: a duplicate of the "
                               f"maximum already gives more")
        y0 = np.array([v])
    else:
        y0 = np.array([doza * ytr.max()])
    q_sa = kvantili(ime, np.vstack([Xtr, x0]), np.concatenate([ytr, y0]), Xte, seed)

    d = np.linalg.norm(Xte - x0, axis=1)
    blizu = d <= np.quantile(d, UDEO_BLIZU)

    def promena(sel, i):
        a, b = q_bez[sel, i], q_sa[sel, i]
        ok = a > 1e-9
        return float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan

    r = {}
    for i, a in enumerate(NIVOI):
        # Not int(a*100) as the key: int(0.999*100) == 99, so 0.999 would
        # overwrite 0.99. Same guard as in h3_repair/repair.py.
        naziv = LEVEL_NAME[a]
        r[f"local_{naziv}"] = promena(blizu, i)
        r[f"global_{naziv}"] = promena(np.ones(len(Xte), bool), i)
    r["xi_without"], r["xi_with"] = xi_iz(q_bez), xi_iz(q_sa)
    r["sd_shift"] = metrics.sd_shift(np.append(ytr, y0))
    r["y0_over_max"] = float(y0[0] / ytr.max())
    r["reason"] = ""
    return r


def main():
    gotovi = append.done(OUT, KLJUC)
    if gotovi:
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)
    kes, t0 = {}, time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                for doza in DOZE:
                    k = dict(xi=xi, model=ime, dose=doza, dose_mode=DOSE_MODE,
                             seed=seed, position=POZICIJA, n_train=N_TRAIN,
                             n_est=N_EST)
                    if append.key(k, KLJUC) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        r = jedan(xi, ime, doza, seed, kes)
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                    r.update(k, seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    poruka = r.get("reason") or (
                        f"lok q50 {r['local_q50']:+.1%} q99 {r['local_q99']:+.1%} | "
                        f"glob q99 {r['global_q99']:+.1%} | xi {r['xi_without']:.2f}->{r['xi_with']:.2f}")
                    print(f"  xi={xi} {ime:10s} dose={doza:>5} s={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)
                kes.clear()

    # Filtrira se po USPEHU MERENJA, ne po `xi_bez.notna()`. `implied_xi` vraca nan kad
    # brentq ne uhvati koren u [-0.9, 6.0], a to se desava bas na najtezim repovima --
    # dakle na redovima koji najvise znace. Stari filter je takav red izbacivao ceo,
    # zajedno sa savrseno validnim `lok_q*` i `glob_q*`, pa je tabela odgovora bila
    # pristrasna ka lakim slucajevima. Sada svaka tabela sama preskoci svoje nedostajuce
    # vrednosti (pivot_table to radi), a koliko ih je -- pise.
    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    bez_xi = int(d.xi_without.isna().sum())
    if bez_xi:
        print(f"\nnapomena: {bez_xi} od {len(d)} redova nema implicirano xi "
              f"(odnos Q99/Q90 van opsega brentq-a). Ti redovi UCESTVUJU u tabelama "
              f"odgovora, a izostaju samo iz tabele promene xi.")
    print("\n=== LOKALNI ODGOVOR (10% test tacaka najblizih ubacenoj) ===")
    for kol in ["local_q50", "local_q90", "local_q99"]:
        print(f"\n  {kol}")
        print("    " + d.pivot_table(index="dose", columns="model", values=kol,
                                     aggfunc="median")
              .map(lambda v: f"{v:+.1%}").to_string().replace("\n", "\n    "))
    print("\n=== GLOBALNI ODGOVOR na Q(0.99) ===")
    print(d.pivot_table(index="dose", columns="model", values="global_q99", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== promena impliciranog xi ===")
    d["dxi"] = d.xi_with - d.xi_without
    print(d.pivot_table(index="dose", columns="model", values="dxi", aggfunc="median")
          .round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
