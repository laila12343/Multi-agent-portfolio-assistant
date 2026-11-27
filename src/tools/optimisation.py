"""Étape 6 : Ledoit-Wolf + Black-Litterman + frontière efficiente (PyPortfolioOpt / cvxpy).

Si PyPortfolioOpt n'est pas installé (tests hors-ligne), un repli numpy/scipy applique
exactement les mêmes formules (Ledoit-Wolf scikit-learn, Black-Litterman avec Ω d'Idzorek,
max Sharpe / min volatilité par SLSQP). Le moteur utilisé est indiqué dans le résultat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.observabilite import trace

try:
    from pypfopt import EfficientFrontier, objective_functions, risk_models
    from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns

    HAS_PYPFOPT = True
except ImportError:  # pragma: no cover - dépend de l'environnement
    HAS_PYPFOPT = False


# ------------------------------------------------------------------ covariance
@trace("ledoit_wolf")
def covariance_ledoit_wolf(prices: pd.DataFrame) -> pd.DataFrame:
    """Covariance annualisée rétrécie (Ledoit-Wolf)."""
    if HAS_PYPFOPT:
        return risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()
    from sklearn.covariance import LedoitWolf

    r = prices.pct_change().dropna(how="all").fillna(0.0)
    return pd.DataFrame(LedoitWolf().fit(r.values).covariance_ * 252, index=prices.columns, columns=prices.columns)


# ------------------------------------------------------------------ Black-Litterman
@trace("black_litterman")
def black_litterman(cov: pd.DataFrame, mcaps: pd.Series, vues: dict, cfg: dict, rf: float) -> dict:
    """a priori de marché + vues du juge -> rendements a posteriori.

    vues : {ticker: {"vue": -2..2, "confiance": faible|moyenne|elevee}}
    Q_i = pi_i + kappa * vue_i ; confiance -> Ω (Idzorek)
    """
    tick = list(cov.index)
    delta, tau, kappa = cfg["delta"], cfg["tau"], cfg["kappa"]
    w_mkt = (mcaps.reindex(tick).fillna(mcaps.median()) / mcaps.reindex(tick).fillna(mcaps.median()).sum())
    if HAS_PYPFOPT:
        pi = market_implied_prior_returns(w_mkt, delta, cov, risk_free_rate=rf)
    else:
        pi = delta * cov.dot(w_mkt) + rf
    Q = pd.Series({t: pi[t] + kappa * vues[t]["vue"] for t in tick})
    conf = np.array([cfg["confiance"][vues[t]["confiance"]] for t in tick])
    if HAS_PYPFOPT:
        bl = BlackLittermanModel(cov, pi=pi, absolute_views=Q.to_dict(), omega="idzorek",
                                 view_confidences=conf, tau=tau)
        post = bl.bl_returns()
    else:
        S, P = cov.values, np.eye(len(tick))
        omega = np.diag([tau * (1 - c) / c * S[i, i] for i, c in enumerate(conf)])
        ts_inv = np.linalg.inv(tau * S)
        om_inv = np.linalg.inv(omega)
        post = pd.Series(np.linalg.solve(ts_inv + P.T @ om_inv @ P, ts_inv @ pi.values + P.T @ om_inv @ Q.values), index=tick)
    return {"prior": pi, "vues": Q, "posterior": post}


# ------------------------------------------------------------------ frontière efficiente
@trace("efficient_frontier")
def optimiser_poids(mu: pd.Series, cov: pd.DataFrame, contraintes: dict, rf: float,
                    objectif: str = "max_sharpe", gamma_l2: float = 0.0, delta: float = 2.5,
                    w_prev: pd.Series | None = None, penalite_rotation: float = 0.0) -> dict:
    """Poids sous contraintes (bornes par titre, plafonds sectoriels). Renvoie poids + attendus.

    objectif : max_sharpe | max_quadratic_utility (w'mu - delta/2 w'Sw) | min_volatility
    penalite_rotation : coût k * somme|w - w_prev| (freine la rotation vs le portefeuille actuel)
    """
    w_prev = (w_prev if w_prev is not None else pd.Series(0.0, index=mu.index)).reindex(mu.index).fillna(0.0)
    if penalite_rotation > 0 and objectif == "max_sharpe":
        objectif = "max_quadratic_utility"      # max_sharpe n'accepte pas bien les objectifs additionnels
    if HAS_PYPFOPT:
        try:
            return _pypfopt(mu, cov, contraintes, rf, objectif, gamma_l2, delta, w_prev, penalite_rotation)
        except Exception as exc:
            print(f"[optimisation] PyPortfolioOpt : {exc} -> repli scipy")
    return _scipy(mu, cov, contraintes, rf, objectif, gamma_l2, delta, w_prev, penalite_rotation)


def _perf(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> dict:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov.loc[w.index, w.index] @ w))
    return {"rendement": ret, "volatilite": vol, "sharpe": (ret - rf) / vol if vol > 0 else float("nan")}


def _clean(w: pd.Series) -> pd.Series:
    w = w.clip(lower=0)
    w[w < 1e-4] = 0.0
    w = (w / w.sum()).round(4)
    return w[w > 0].sort_values(ascending=False)


def _pypfopt(mu, cov, c, rf, objectif, gamma, delta, w_prev, pen):
    ef = EfficientFrontier(mu, cov, weight_bounds=c["bornes"])
    ef.add_sector_constraints(c["secteurs"], c["secteur_min"], c["secteur_max"])
    if gamma > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=gamma)
    if pen > 0:
        ef.add_objective(objective_functions.transaction_cost, w_prev=w_prev.values, k=pen)
    if objectif == "max_sharpe" and (mu > rf).any():
        ef.max_sharpe(risk_free_rate=rf)
    elif objectif == "max_quadratic_utility":
        ef.max_quadratic_utility(risk_aversion=delta)
    else:
        ef.min_volatility()
    w = _clean(pd.Series(ef.clean_weights()))
    return {"poids": w.to_dict(), "attendus": _perf(w, mu, cov, rf), "moteur": "PyPortfolioOpt"}


def _scipy(mu, cov, c, rf, objectif, gamma, delta, w_prev, pen):
    tick = list(mu.index)
    m, S, wp = mu.values, cov.loc[tick, tick].values, w_prev.values
    n = len(tick)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    for sec, cap in c["secteur_max"].items():
        idx = np.array([c["secteurs"][t] == sec for t in tick])
        cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - w[idx].sum()})

    def cout(w):  # L2 + rotation (valeur absolue lissée pour SLSQP)
        return gamma * (w @ w) + pen * np.sqrt((w - wp) ** 2 + 1e-8).sum()

    def neg_sharpe(w):
        return -(w @ m - rf) / np.sqrt(w @ S @ w + 1e-12) + cout(w)

    def neg_utility(w):
        return -(w @ m - 0.5 * delta * w @ S @ w) + cout(w)

    def variance(w):
        return w @ S @ w + cout(w)

    f = (neg_sharpe if objectif == "max_sharpe" and (m > rf).any()
         else neg_utility if objectif == "max_quadratic_utility" else variance)
    lo, hi = c["bornes"]
    x0 = np.full(n, 1 / n)
    res = minimize(f, x0, bounds=[(lo, hi)] * n, constraints=cons, method="SLSQP",
                   options={"maxiter": 1000, "ftol": 1e-12})
    w = _clean(pd.Series(res.x if res.success else np.minimum(x0, hi), index=tick))
    return {"poids": w.to_dict(), "attendus": _perf(w, mu, cov, rf), "moteur": "scipy (repli)"}
