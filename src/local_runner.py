"""Exécution SÉQUENTIELLE du même workflow, sans LangGraph (tests, débogage pas à pas).

Mêmes nœuds, même ordre, même boucle de risque ; Send est remplacé par une boucle et
interrupt() par une liste de décisions fournies à l'avance.
"""
from __future__ import annotations

from src.agents.analystes import analyse_titre_sans_graphe
from src.agents.debat import executer_debat
from src.nodes import portefeuille, preparation, rapport, validation


def _apply(state: dict, upd: dict):
    for k, v in upd.items():
        state[k] = state.get(k, []) + v if k in ("notes", "journal") else v


def executer_sans_langgraph(ctx, date_analyse: str | None = None, decisions=("approuve",), verbose: bool = False) -> dict:
    state: dict = {"date_analyse": date_analyse, "notes": [], "journal": []}
    for node in (preparation.charger_mandat, preparation.construire_univers,
                 preparation.collecter_donnees, preparation.calculer_metriques):
        _apply(state, node(state, ctx))
    for payload in preparation.payloads_analyse(state, ctx):          # Send x N
        _apply(state, analyse_titre_sans_graphe(payload, ctx))
    _apply(state, executer_debat(state, ctx, avec_graphe=False))
    decisions = list(decisions)
    etape = "optimiser"
    while etape != "__end__":
        if etape == "optimiser":
            _apply(state, portefeuille.optimiser(state, ctx))
            etape = "controle_risque"
        elif etape == "controle_risque":
            _apply(state, portefeuille.controle_risque(state, ctx))
            etape = portefeuille.router_risque(state, ctx)
        elif etape == "validation_gerant":
            state["demande_validation"] = validation.demande_validation(state)   # ce qu'afficherait interrupt()
            _apply(state, validation.appliquer_decision(state, ctx, decisions.pop(0) if decisions else "approuve"))
            etape = validation.router_validation(state)
        elif etape == "rapport":
            _apply(state, rapport.rapport(state, ctx))
            etape = "__end__"
    if verbose:
        print("\n".join(state["journal"]))
    return state
