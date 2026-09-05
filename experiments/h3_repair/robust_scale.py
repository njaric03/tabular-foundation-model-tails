# -*- coding: utf-8 -*-
"""
Prava popravka: zameniti PROCENITELJ skale, ne dirati podatke.

RAZLIKA U ODNOSU NA `repair.py`
-----------------------------------------
Tamo su testirane transformacije targeta (log, winsor, rang, asinh). Sve one menjaju
PODATKE, pa deformisu rep -- i zato placaju na alfa = 0.999.

Ovde se podaci ne diraju. Menja se samo BROJ kojim model deli target: umesto empirijske
standardne devijacije koristi se robusna procena iste velicine (IQR / 1.349, sto je
konzistentna procena sd za normalnu raspodelu).

Zasto se to ne moze uraditi spolja: standardizacija je invarijantna na afino skaliranje.
Ako se preda y' = a*y + b, model izracuna std(y') = a*std(y) i podeli sa njim, pa se `a`
skrati. Jedini nacin je zahvat u sam procenitelj, dakle monkeypatch.

STA SE OCEKUJE
--------------
Robusna skala je MANJA od naduvane sd, pa standardizovane vrednosti postaju VECE. Kod
modela sa fiksnom mrezom korpi (TabPFN: granice u z-prostoru; TabDPT: +-10) to znaci da
ekstremi padaju dalje u mrezu i mogu biti odseceni. Dakle ocekuje se druga vrsta cene
nego kod transformacija -- ne gubitak informacije o repu, nego moguce odsecanje.

MERI SE
-------
  KORIST   odgovor na jednu ubacenu tacku (y = 100*max)
  CENA     pinball na 0.5 / 0.9 / 0.99 / 0.999 na CISTIM podacima

POKRETANJE
----------
    python -u robust_scale.py
    XI=0.9 SEEDOVA=3 MODELI=TabICLv2 python -u robust_scale.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, models, paths, quiet

quiet.silence()


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.7,0.9").split(",")]
VARIJANTE = os.environ.get("VARIANTS", "sirovo,robusna").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3"))
N_TRAIN, N_TEST = 2000, 900
NIVOI = [0.5, 0.9, 0.99, 0.999]
IME_NIVOA = {0.5: "pb50", 0.9: "pb90", 0.99: "pb99", 0.999: "pb999"}
OUT = os.environ.get("OUTPUT", "robust_scale.csv")
# The wrappers below pin n_estimators=1; recorded so the run is reconstructable.
N_EST = 1

KOLONE = ["n_est", "xi", "model", "variant", "seed", "response_q99", "response_q999",
          "pb50", "pb90", "pb99", "pb999", "scale_ratio", "seconds", "reason"]
KEY = ["xi", "model", "variant", "seed", "n_est"]


def robusna_sd(y):
    """IQR / 1.349 -- konzistentna procena sd za normalnu, otporna na outliere."""
    q75, q25 = np.percentile(y, [75, 25])
    return float(max((q75 - q25) / 1.349, 1e-12))


def kvantili(varijanta, ime, Xtr, ytr, Xte, seed, info):
    if ime == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=1, device="cpu", random_state=seed)
        m.fit(Xtr, ytr)
        if varijanta == "robust":
            # TabICL standardizuje y StandardScaler-om; zameni `scale_` robusnom procenom
            sc = getattr(m, "y_scaler_", None) or getattr(m, "scaler_", None)
            if sc is None or not hasattr(sc, "scale_"):
                raise RuntimeError("ne nalazim skaler za y kod TabICL-a")
            staro = float(np.ravel(sc.scale_)[0])
            novo = robusna_sd(ytr)
            sc.scale_ = np.full_like(np.asarray(sc.scale_, dtype=float), novo)
            info["scale_ratio"] = novo / staro
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=NIVOI))

    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=1, device="cpu", random_state=seed,
                        ignore_pretraining_limits=True)
    m.fit(Xtr, ytr)
    if varijanta == "robust":
        staro = float(m.y_train_std_)
        novo = robusna_sd(ytr)
        m.y_train_std_ = float(novo)
        m._rebuild_raw_space_bardist()      # granice zavise od sd, moraju se prezidati
        info["scale_ratio"] = novo / staro
    return np.stack([np.asarray(a) for a in
                     m.predict(Xte, output_type="quantiles", quantiles=NIVOI)], axis=1)


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def jedan(xi, ime, varijanta, seed):
    rng = np.random.default_rng(seed)
    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = _p.X, _p.y
    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, yte = _p.X, _p.y
    info = {"scale_ratio": np.nan}

    q0 = kvantili(varijanta, ime, Xtr, ytr, Xte, seed, info)
    pb = {IME_NIVOA[a]: pinball(yte, q0[:, i], a) for i, a in enumerate(NIVOI)}

    x0 = np.zeros((1, Xtr.shape[1]))
    q1 = kvantili(varijanta, ime, np.vstack([Xtr, x0]),
                  np.concatenate([ytr, [100 * ytr.max()]]), Xte, seed, {})

    def odg(i):
        a, b = q0[:, i], q1[:, i]
        ok = a > 1e-9
        return float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan

    return dict(response_q99=odg(NIVOI.index(0.99)),
                response_q999=odg(NIVOI.index(0.999)),
                scale_ratio=info["scale_ratio"], reason="", **pb)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for varijanta in VARIJANTE:
                for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                    if append.key(dict(xi=xi, model=ime, variant=varijanta, seed=seed), KEY[:4]) in gotovi:
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
                        f"odgovor99 {r['response_q99']:+7.1%} odgovor999 {r['response_q999']:+7.1%} "
                        f"| pb99 {r['pb99']:.3f} pb999 {r['pb999']:.3f} "
                        f"| skala x{r['scale_ratio']:.3f}" if np.isfinite(r.get("scale_ratio", np.nan))
                        else f"odgovor99 {r['response_q99']:+7.1%} | pb99 {r['pb99']:.3f}")
                    print(f"  xi={xi} {ime:10s} {varijanta:8s} s={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.response_q99.notna()]
    print("\n=== KORIST: odgovor na jednu ubacenu tacku ===")
    for kol in ["response_q99", "response_q999"]:
        print(f"\n  {kol}")
        print("    " + d.pivot_table(index=["xi", "model"], columns="variant", values=kol,
                                     aggfunc="median")
              .map(lambda v: f"{v:+.1%}").to_string().replace("\n", "\n    "))
    print("\n=== CENA na CISTIM podacima, relativno prema `sirovo` ===")
    for kol in ["pb50", "pb90", "pb99", "pb999"]:
        print(f"\n  {kol}")
        for (xi, ime), g in d.groupby(["xi", "model"]):
            baza = g[g.variant == "raw"][kol].median()
            red = "  ".join(f"{v}: {100*(g[g.variant==v][kol].median()-baza)/baza:+7.1f}%"
                            for v in VARIJANTE if v != "raw")
            print(f"    xi={xi} {ime:<12} {red}")
    print("\n=== koliko je robusna skala manja od sd ===")
    print(d[d.variant == "robust"].groupby(["xi", "model"]).scale_ratio
          .median().round(4).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
