# -*- coding: utf-8 -*-
"""
Izvlacenje KVANTILA iz TabDPT-a, pa disocijacija nivo/oblik za cetvrti model.

PROBLEM
-------
TabDPT javno vraca samo sredinu. Ali interno ima punu raspodelu preko 2048 korpi:

    edges = linspace(regression_bin_min=-10, regression_bin_max=+10, 2049)
    weights = softmax(reg_logits)
    E[y] = sum(weights * bin_centres)          <- jedino sto izlazi napolje

Za rep nam treba cela raspodela, ne sredina.

RESENJE BEZ DIRANJA PAKETA
--------------------------
`_expectation_from_regression_logits` se presretne i logiti se odloze sa strane. Logiti
su u NORMALIZOVANOM prostoru, a paket vraca `y_hat * std_y + mean_y`. Umesto da se kopa
za `std_y`/`mean_y`, ta afina veza se rekonstruise regresijom vracenih predikcija na
sopstveno izracunatu normalizovanu sredinu:

    raw = a * y_hat_norm + b   =>   a = std_y,  b = mean_y

Egzaktno je (afina veza, bez suma) i ne zavisi od internih imena.

Kvantili se onda racune iz kumulativne sume tezina po korpama, pa se linearno
interpoliraju unutar korpe i skaliraju sa (a, b).

STA SE MERI
-----------
Isto sto i za ostale modele u `findings/h1/dissociation.md`:
  udeo prave promene SKALE  s(x)  koji model uhvati
  udeo prave promene OBLIKA xi(x) koji model uhvati

POKRETANJE
----------
    venv-tabdpt/Scripts/python.exe dissociation_tabdpt.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.stats import norm

from common import append, generator, metrics, paths

warnings.filterwarnings("ignore")

W = np.array([1.0, -0.7, 0.5, 0.0, 0.0])
XI_LO, XI_HI = 0.15, 0.90
N_TRAIN, N_TEST = 2000, 900
N_TERCILA = 3
HI, LO = 0.99, 0.90
SEEDOVA = int(os.environ.get("SEEDS", "10"))
OUT = os.environ.get("OUTPUT", "dissociation_tabdpt.csv")
KOLONE = ["seed", "shape_share", "scale_share", "xi_t1", "xi_t3",
          "xi_true_t1", "xi_true_t3", "seconds", "reason"]
KEY = ["seed"]


def kvantili_tabdpt(Xtr, ytr, Xte, seed, nivoi):
    """Quantiles from the internal bin head; see common/adapters/tabdpt.py."""
    from common.adapters import tabdpt
    return tabdpt.quantiles(Xtr, ytr, Xte, seed, nivoi)


def jedan(seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr, _, _ = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
    Xte, _, s_te, xi_te = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
    nivoi = [0.5, LO, HI]
    q = kvantili_tabdpt(Xtr, ytr, Xte, seed, nivoi)

    # OBLIK: tercili po x4 (tezina u skali je nula)
    iv = np.quantile(Xte[:, 4], np.linspace(0, 1, N_TERCILA + 1))
    t_ob = np.clip(np.digitize(Xte[:, 4], iv[1:-1]), 0, N_TERCILA - 1)
    xi_hat = []
    for t in range(N_TERCILA):
        sel = t_ob == t
        num, den = q[sel, 2], q[sel, 1]
        ok = den > 1e-9
        xi_hat.append(metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))) if ok.any() else np.nan)
    xi_pr = [float(xi_te[t_ob == t].mean()) for t in range(N_TERCILA)]
    udeo_ob = (xi_hat[2] - xi_hat[0]) / (xi_pr[2] - xi_pr[0])

    # SKALA: tercili po linearnom prediktoru
    lin = Xte @ W / np.linalg.norm(W)
    iv = np.quantile(lin, np.linspace(0, 1, N_TERCILA + 1))
    t_sk = np.clip(np.digitize(lin, iv[1:-1]), 0, N_TERCILA - 1)
    med_pr = s_te * metrics.gpd_quantile(0.5, xi_te)
    med_mo = q[:, 0]
    a_m, b_m = np.median(med_mo[t_sk == 0]), np.median(med_mo[t_sk == N_TERCILA - 1])
    a_p, b_p = np.median(med_pr[t_sk == 0]), np.median(med_pr[t_sk == N_TERCILA - 1])
    udeo_sk = np.log(b_m / a_m) / np.log(b_p / a_p) if a_m > 0 and b_m > 0 else np.nan

    return dict(shape_share=udeo_ob, scale_share=udeo_sk,
                xi_t1=xi_hat[0], xi_t3=xi_hat[2],
                xi_true_t1=xi_pr[0], xi_true_t3=xi_pr[2], reason="")


def main():
    gotovi = set()
    if paths.result(OUT).exists():
        gotovi = set(pd.read_csv(paths.result(OUT)).seed)
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)
    t0 = time.time()
    for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
        if append.key(dict(seed=seed), KEY) in gotovi:
            continue
        t1 = time.time()
        try:
            r = jedan(seed)
        except Exception as e:
            r = dict(reason=f"{type(e).__name__}: {e}"[:110])
        r.update(seed=seed, seconds=round(time.time() - t1, 1))
        append.write(OUT, r, KOLONE)
        poruka = r.get("reason") or (
            f"oblik {r['shape_share']:>6.0%}  skala {r['scale_share']:>6.0%}  "
            f"xi {r['xi_t1']:.2f}->{r['xi_t3']:.2f} (pravo {r['xi_true_t1']:.2f}->"
            f"{r['xi_true_t3']:.2f})")
        print(f"  seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""] if "reason" in d.columns else d[d.shape_share.notna()]
    print(f"\n=== TabDPT, {len(d)} seedova ===")
    print(f"  SKALA  {d.scale_share.mean():.0%}  (sd {d.scale_share.std():.2f})")
    print(f"  OBLIK  {d.shape_share.mean():.0%}  (sd {d.shape_share.std():.2f})")
    print("\nporedjenje (n_est=1, 20 seedova, iz `findings/h1/dissociation.md`):")
    print("  GBM 88% / 45%   TabICLv2 99% / 44%   TabPFN-V3 95% / 42%")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
