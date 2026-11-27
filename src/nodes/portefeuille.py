"""Étapes 6 et 7 : optimiser (code) et controle_risque (code + LLM), avec la boucle H9.

    optimiser -> controle_risque -> conforme                -> validation_gerant
                                 -> non conforme, essai < 3 -> optimiser (stratégie plus prudente)
                                 -> non conforme, essai = 3 -> validation_gerant avec ALERTE
"""
from __future__ import annotations

import json

import pandas as pd

from src.agents import prompts
from src.llm import text_call
from src.nodes.preparation import prix_frame
from src.state import EtatPortefeuille
from src.tools.mandat import contraintes_mandat
from src.tools.optimisation import black_litterman, covariance_ledoit_wolf, optimiser_poids
from src.tools.reporting import lire_portefeuille_actuel, valoriser
from src.tools.risque import evaluer_risque


def _rf(ctx) -> float:
    return ctx.settings["risque"].get("taux_sans_risque", 0.04)


def optimiser(state: EtatPortefeuille, ctx) -> dict:
    essai = state.get("iterations_risque", 0)
    tent = ctx.settings["optimisation"]["tentatives"]
    strat = tent[min(essai, len(tent) - 1)]
    sel, met = state["selection"], state["metriques"]
    prices = prix_frame(state["donnees"])[sel]
    cov = covariance_ledoit_wolf(prices)
    mcaps = pd.Series({t: met[t].get("capitalisation") for t in sel}, dtype=float)
    bl = black_litterman(cov, mcaps, state["vues"], ctx.settings["optimisation"], _rf(ctx))
    c = contraintes_mandat(state["mandat"], sel, {t: met[t]["secteur"] for t in sel})
    poids_actuels, _, _ = contexte_risque(state, ctx)
    res = optimiser_poids(bl["posterior"], cov, c, _rf(ctx), strat["objectif"], strat.get("gamma_l2", 0.0),
                          ctx.settings["optimisation"]["delta"], pd.Series(poids_actuels, dtype=float),
                          strat.get("penalite_rotation", 0.0))
    a = res["attendus"]
    attendus = {**a, "moteur": res["moteur"], "strategie": strat, "essai": essai + 1,
                "rendements": {t: {"a_priori": round(float(bl["prior"][t]), 4), "vue": round(float(bl["vues"][t]), 4),
                                   "a_posteriori": round(float(bl["posterior"][t]), 4)} for t in sel}}
    return {"poids": res["poids"], "attendus": attendus, "iterations_risque": essai + 1,
            "journal": [f"[optimiser] essai {essai + 1} ({strat['objectif']}, L2={strat.get('gamma_l2', 0)}, "
                        f"rotation x{strat.get('penalite_rotation', 0)}, {res['moteur']}) : "
                        f"{len(res['poids'])} lignes, rendement attendu {a['rendement']:.1%}, vol {a['volatilite']:.1%}, "
                        f"Sharpe {a['sharpe']:.2f}"]}


def contexte_risque(state: EtatPortefeuille, ctx) -> tuple[dict, pd.DataFrame, dict]:
    prices = prix_frame(state["donnees"])
    dernier = prices.iloc[-1].to_dict()
    positions = lire_portefeuille_actuel(ctx.settings)
    poids_actuels, _ = valoriser(positions, dernier)
    elig = [t for t in state["univers"]["eligibles"] if t in prices.columns]
    return poids_actuels, prices[elig].pct_change().dropna(how="all"), dernier


def controle_risque(state: EtatPortefeuille, ctx) -> dict:
    poids_actuels, rets, _ = contexte_risque(state, ctx)
    rr = evaluer_risque(state["poids"], poids_actuels, state["mandat"], state["metriques"], rets,
                        list(rets.columns), ctx.settings["risque"])
    essai = state.get("iterations_risque", 0)
    rr["essai"] = essai
    rr["alerte"] = (not rr["conforme"]) and essai >= ctx.settings["risque"]["max_tentatives"]
    m = rr["mesures"]
    explication = ""
    if ctx.llm_fort is not None:
        faits = {k: rr[k] for k in ("conforme", "manquements_mandat", "depassements_limites", "mesures", "stress", "limites")}
        explication = text_call(ctx.llm_fort, prompts.RISQUE, json.dumps(faits, ensure_ascii=False, default=float))
    if not explication:
        explication = (f"Essai {essai} : volatilité {m['volatilite_annuelle']:.1%}, VaR 95 % {m['var_historique_1j']:.2%}/jour, "
                       f"CVaR {m['cvar_historique_1j']:.2%}, top 5 = {m['top5']:.0%}, rotation {m['rotation']:.0%}, "
                       f"pire stress {max(rr['stress'].values()):.1%}. "
                       + ("Portefeuille conforme au mandat." if rr["conforme"] else
                          "Non conforme : " + "; ".join(rr["manquements_mandat"] + rr["depassements_limites"]) + "."))
    rr["explication"] = explication
    statut = "CONFORME" if rr["conforme"] else ("ALERTE — escalade au gérant" if rr["alerte"] else "non conforme -> nouvel essai")
    return {"rapport_risque": rr, "journal": [f"[controle_risque] essai {essai} : {statut}"
                                              + ("" if rr["conforme"] else f" ({'; '.join(rr['depassements_limites'] + rr['manquements_mandat'])})")]}


def router_risque(state: EtatPortefeuille, ctx) -> str:
    rr = state["rapport_risque"]
    if rr["conforme"] or state.get("iterations_risque", 0) >= ctx.settings["risque"]["max_tentatives"]:
        return "validation_gerant"
    return "optimiser"
