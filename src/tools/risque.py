"""Étape 7 : verifier_conformite, calculer_var, stress_test, concentration, rotation (code pur)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.tools.metrics import TRADING_DAYS, max_drawdown, portfolio_returns
from src.observabilite import trace

TOL = 1e-3


def verifier_conformite(poids: dict, mandat: dict, secteurs: dict, univers: list[str]) -> list[str]:
    """Contraintes du mandat (H5). Renvoie la liste des manquements (vide = conforme)."""
    c, out = mandat["contraintes"], []
    w = pd.Series({t: v for t, v in poids.items() if t != "CASH"}, dtype=float)
    if abs(sum(poids.values()) - 1) > 5 * TOL:
        out.append(f"somme des poids = {sum(poids.values()):.3f} (≠ 1)")
    if c.get("long_only", True) and (w < -TOL).any():
        out.append("position courte interdite (long-only)")
    for t, v in w[w > c["poids_max_titre"] + TOL].items():
        out.append(f"{t} : {v:.1%} > {c['poids_max_titre']:.0%} par titre")
    sec = w.groupby(pd.Series(secteurs).reindex(w.index).fillna("Autre")).sum()
    for s, v in sec[sec > c["poids_max_secteur"] + TOL].items():
        out.append(f"secteur {s} : {v:.1%} > {c['poids_max_secteur']:.0%}")
    if (w > 0).sum() < c.get("lignes_min", 1):
        out.append(f"{(w > 0).sum()} lignes < minimum {c['lignes_min']}")
    hors = [t for t in w.index if t not in univers]
    if hors:
        out.append(f"titres hors univers éligible : {hors}")
    return out


def calculer_var(port_returns: pd.Series, niveau: float = 0.95) -> dict:
    r = port_returns.dropna()
    q = np.quantile(r, 1 - niveau)
    return {
        "var_historique_1j": float(-q),
        "cvar_historique_1j": float(-r[r <= q].mean()),
        "var_parametrique_1j": float(-(r.mean() + norm.ppf(1 - niveau) * r.std(ddof=1))),
        "volatilite_annuelle": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "drawdown_max_historique": max_drawdown(r),
    }


def concentration(poids: dict) -> dict:
    w = pd.Series({t: v for t, v in poids.items() if t != "CASH"}, dtype=float).sort_values(ascending=False)
    return {"top5": float(w.head(5).sum()), "hhi": float((w ** 2).sum()), "nb_effectif": float(1 / (w ** 2).sum())}


def rotation(poids_cible: dict, poids_actuels: dict) -> float:
    keys = set(poids_cible) | set(poids_actuels)
    return float(0.5 * sum(abs(poids_cible.get(k, 0) - poids_actuels.get(k, 0)) for k in keys))


@trace("stress_test")
def stress_test(poids: dict, metriques: dict, returns: pd.DataFrame, scenarios: dict) -> dict:
    """Perte estimée du portefeuille par scénario (positive = perte)."""
    w = pd.Series({t: v for t, v in poids.items() if t != "CASH"}, dtype=float)
    out = {}
    for nom, sc in scenarios.items():
        if sc["type"] == "beta":
            b = pd.Series({t: metriques[t].get("beta") or 1.0 for t in w.index})
            out[nom] = float(-(w * b * sc["choc_marche"]).sum())
        elif sc["type"] == "secteur":
            hit = pd.Series({t: sc["choc"] if metriques[t]["secteur"] == sc["secteur"] else 0.0 for t in w.index})
            out[nom] = float(-(w * hit).sum())
        elif sc["type"] == "historique":
            pr = portfolio_returns(w.to_dict(), returns[w.index])
            roll = (1 + pr).rolling(sc["fenetre_jours"]).apply(np.prod, raw=True) - 1
            out[nom] = float(-roll.min())
    return out


@trace("controle_risque_calculs")
def evaluer_risque(poids: dict, poids_actuels: dict, mandat: dict, metriques: dict,
                   returns: pd.DataFrame, univers: list[str], cfg: dict) -> dict:
    """Rapport de risque complet + verdict de conformité aux limites du mandat."""
    lim = mandat["limites_risque"]
    secteurs = {t: m["secteur"] for t, m in metriques.items()}
    r = returns.copy()
    r["CASH"] = 0.0
    pr = portfolio_returns(poids, r)
    var = calculer_var(pr, cfg["var_niveau"])
    conc = concentration(poids)
    rot = rotation(poids, poids_actuels)
    stress = stress_test(poids, metriques, returns, cfg["stress_tests"])
    manquements = verifier_conformite(poids, mandat, secteurs, univers)
    checks = [
        ("volatilité annuelle", var["volatilite_annuelle"], lim["volatilite_annuelle_max"]),
        ("VaR 95 % 1 jour", var["var_historique_1j"], lim["var95_1j_max"]),
        ("CVaR 95 % 1 jour", var["cvar_historique_1j"], lim["cvar95_1j_max"]),
        ("rotation", rot, lim["rotation_max"]),
        ("concentration top 5", conc["top5"], lim["concentration_top5_max"]),
        ("pire stress test", max(stress.values()), lim["perte_stress_max"]),
    ]
    depassements = [f"{n} = {v:.2%} > limite {l:.2%}" for n, v, l in checks if v > l + 1e-9]
    return {
        "conforme": not manquements and not depassements,
        "manquements_mandat": manquements,
        "depassements_limites": depassements,
        "mesures": {**var, **conc, "rotation": rot},
        "stress": stress,
        "limites": lim,
        "explication": "",
    }
