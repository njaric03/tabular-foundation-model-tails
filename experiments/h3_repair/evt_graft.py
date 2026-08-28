# -*- coding: utf-8 -*-
"""
Pravac 2: post-hoc kalemljenje repa iz teorije ekstremnih vrednosti.

METOD
-----
Model daje celu uslovnu raspodelu. Telo pogadja, rep ne. Ideja je zadrzati telo, a rep
zameniti GPD-om procenjenim IZ PODATAKA, skaliranim modelovim sopstvenim uslovnim nivoom.

  1. Fituj model. Uzmi uslovnu medijanu m(x) na trening skupu.
  2. Standardizuj trening reziduale multiplikativno:  z_i = y_i / m(x_i).
     Multiplikativno, jer je target pozitivan i teskorepan, pa je skala uslovna.
  3. Prag u = kvantil(z, ALPHA0). GPD MLE nad prekoracenjima (z - u)+  ->  xi_hat, sigma_hat.
  4. Za test tacku, spoji:
         alpha <= ALPHA0 :  Q(alpha|x) = Q_model(alpha|x)              (telo od modela)
         alpha >  ALPHA0 :  Q(alpha|x) = Q_model(ALPHA0|x)
                            + m(x) * sigma_hat * (((1-alpha)/(1-ALPHA0))^(-xi_hat) - 1)/xi_hat

Rep tako nasledjuje OBLIK iz podataka, a NIVO iz modela, pa se heteroskedasticnost koju
je model nasao zadrzava. Spoj je neprekidan po konstrukciji.

METRIKE, namerno vise od jedne, jer implicirano xi nije standardna mera
------------------------------------------------------------------------
  crps        priblizno, preko pinball gubitka na mrezi kvantila (2 * srednji pinball)
  twcrps      threshold-weighted CRPS, Gneiting & Ranjan (2011), preko chaining funkcije
              v(y) = max(y, t) sa t = empirijski kvantil 0,90 test targeta.
              Ovo je metrika koja zapravo meri rep; obican CRPS ga jedva vidi.
  pinball99   pinball na nivou 0,99   \\ direktna tacnost ekstremnih kvantila
  pinball999  pinball na nivou 0,999  /
  cov99/999   stvarna pokrivenost, nominalno 0,99 i 0,999
  gamma_dev   gama devijansa nad sredinom raspodele, mera iz rada o osiguranju
  xi_implied  zadrzano radi uporedivosti sa ranijim merenjima

Nize je bolje za crps, twcrps, pinball i gamma_dev.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import genpareto
from common import metrics

ALPHA0 = 0.90                      # prag spajanja
GRID = np.round(np.arange(1, 100) / 100.0, 4).tolist()
REP_NIVOI = [0.995, 0.999, 0.9995]
NIVOI = sorted(set(GRID + REP_NIVOI + [0.5, ALPHA0, 0.99]))
I50, I90, I99, I999 = (NIVOI.index(0.5), NIVOI.index(ALPHA0),
                       NIVOI.index(0.99), NIVOI.index(0.999))


def _reziduali(y_tr, med_tr):
    """Multiplikativni reziduali y/m(x), uz zastitu od nepozitivne medijane.

    ZASTO POSTOJI: modeli umeju da predvide NEGATIVNU uslovnu medijanu i kad je
    target strogo pozitivan (izmereno: TabICLv2 na GPD podacima daje med < 0 za 11
    od 2000 tacaka, min -0,65). Deljenje takvom medijanom, ili njenim odsecanjem na
    1e-12, pravi rezidual reda 1e13, koji sam dominira GPD fitom i daje xi > 3.
    Popravka koja se onda nakalemi je stotinu puta gora od nikakve.

    Zato se tacke sa nepozitivnom ili apsurdno malom medijanom IZBACUJU, umesto da se
    odsecaju. Prag je 1% medijane svih pozitivnih predikcija, dovoljno nisko da ne
    dira zdrave tacke, dovoljno visoko da ukloni degenerisane.
    """
    y = np.asarray(y_tr, dtype=float)
    m = np.asarray(med_tr, dtype=float)
    poz = np.isfinite(m) & (m > 0) & np.isfinite(y) & (y > 0)
    if poz.sum() < 200:
        return None
    prag_m = 0.01 * float(np.median(m[poz]))
    ok = poz & (m > prag_m)
    if ok.sum() < 200:
        return None
    z = y[ok] / m[ok]
    return z[np.isfinite(z) & (z > 0)]


# --------------------------------------------------------------------- metod
def fit_evt_rep(y_tr, med_tr, alpha0=ALPHA0):
    """Proceni oblik repa iz trening reziduala. Vrati (xi_hat, sigma_hat, u)."""
    z = _reziduali(y_tr, med_tr)
    if z is None:
        return np.nan, np.nan, np.nan
    u = float(np.quantile(z, alpha0))
    ex = z[z > u] - u
    if len(ex) < 50:
        return np.nan, np.nan, np.nan
    xi_hat, _, sigma_hat = genpareto.fit(ex, floc=0)
    return float(xi_hat), float(sigma_hat), u


def spoji(q, med_te, xi_hat, sigma_hat, nivoi=NIVOI, alpha0=ALPHA0):
    """Zameni rep iznad alpha0 GPD-om; telo ostaje modelovo."""
    if not np.isfinite(xi_hat) or not np.isfinite(sigma_hat):
        return q.copy()
    q = np.asarray(q, dtype=float).copy()
    nivoi = np.asarray(nivoi, dtype=float)
    i0 = int(np.argmin(np.abs(nivoi - alpha0)))
    baza = q[:, i0]
    med = np.maximum(np.asarray(med_te, dtype=float), 1e-12)

    gornji = nivoi > alpha0
    p = (1.0 - nivoi[gornji]) / (1.0 - alpha0)
    if abs(xi_hat) < 1e-8:
        prirast = -np.log(p) * sigma_hat
    else:
        prirast = (p ** (-xi_hat) - 1.0) / xi_hat * sigma_hat
    q[:, gornji] = baza[:, None] + med[:, None] * prirast[None, :]
    return np.maximum.accumulate(q, axis=1)      # monotonost


# ------------------------------------------------------------------ metrike


def implicirano_xi(q, nivoi=NIVOI, hi=0.99, lo=ALPHA0):
    ih, il, im = (int(np.argmin(np.abs(np.asarray(nivoi) - v))) for v in (hi, lo, 0.5))
    den = q[:, il] - q[:, im]
    ok = den > 1e-9
    if not ok.any():
        return np.nan
    r = float(np.median((q[ok, ih] - q[ok, im]) / den[ok]))
    try:
        return brentq(lambda x: ((metrics.gpd_quantile(hi, x) - metrics.gpd_quantile(0.5, x)) /
                                 (metrics.gpd_quantile(lo, x) - metrics.gpd_quantile(0.5, x))) - r, -0.9, 6.0)
    except Exception:
        return np.nan


def sve_metrike(q, y_te, prag):
    mu = metrics.mean_from_quantiles(q)
    return dict(
        crps=metrics.crps(q, y_te),
        twcrps=metrics.twcrps(q, y_te, prag),
        pinball99=metrics.pinball_at(q, y_te, 0.99),
        pinball999=metrics.pinball_at(q, y_te, 0.999),
        cov99=float(np.mean(y_te <= q[:, I99])),
        cov999=float(np.mean(y_te <= q[:, I999])),
        gamma_dev=metrics.gamma_deviance(mu, y_te),
        xi_implied=implicirano_xi(q),
        mean=float(mu.mean()),
    )


# ---------------------------------------------------- regularizovana procena
# Naivna verzija (`fit_evt_rep`) se na 2000 redova pokazala nestabilnom: od 30
# procena, cetiri su presle 1 (dakle beskonacna sredina), jedna je bila 3,32, i
# tamo je "popravka" bila 200 puta gora od nista. Uzrok je dvostruk: GPD MLE je
# na malom broju prekoracenja sumovit, a ovde se jos i skala nosi modelovom
# medijanom, sto dodaje sum. Tri standardne mere to resavaju.

XI_MAX = 0.95          # sredina mora da postoji; iznad 1 GPD je nema
XI_MIN = 0.0           # popravlja se samo tezak rep, ne pravi se laksi
PRAGOVI = (0.05, 0.10, 0.20)


def fit_evt_rep_reg(y_tr, med_tr, pragovi=PRAGOVI, xi_min=XI_MIN, xi_max=XI_MAX):
    """Robusnija procena repa: medijana po vise pragova, pa ogranicenje.

    1. xi i sigma se procene na svakom od `pragovi`, pa se uzme MEDIJANA, time
       nestaje osetljivost na izbor praga, koja je najveci izvor rasipanja.
    2. xi se ogranici na [xi_min, xi_max]. Gornja granica je ispod 1, jer iznad 1
       fitovani rep nema sredinu i ekstremni kvantili eksplodiraju.
    """
    z = _reziduali(y_tr, med_tr)
    if z is None:
        return np.nan, np.nan, np.nan

    xis, sigmas = [], []
    for p in pragovi:
        u_p = float(np.quantile(z, 1 - p))
        ex = z[z > u_p] - u_p
        if len(ex) < 50:
            continue
        xi_p, _, sg_p = genpareto.fit(ex, floc=0)
        # svedi skalu na zajednicki prag ALPHA0 radi uporedivosti
        u0 = float(np.quantile(z, ALPHA0))
        if abs(xi_p) > 1e-8:
            sg_p = sg_p + xi_p * (u0 - u_p)
        xis.append(float(xi_p))
        sigmas.append(float(max(sg_p, 1e-9)))

    if not xis:
        return np.nan, np.nan, np.nan
    xi_hat = float(np.clip(np.median(xis), xi_min, xi_max))
    sigma_hat = float(np.median(sigmas))
    return xi_hat, sigma_hat, float(np.quantile(z, ALPHA0))
