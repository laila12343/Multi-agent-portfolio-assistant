"""Étape 8 — validation_gerant : point d'arrêt humain (H1).

interrupt() met le graphe en pause et sauvegarde l'état (checkpointer). Le gérant répond avec
Command(resume=...) :
    {"decision": "approuve"}
    {"decision": "modifie", "poids": {"AAPL": 0.08, ...}, "motif": "..."}
    {"decision": "rejete", "motif": "..."}

Nœud idempotent AVANT interrupt() : à la reprise, LangGraph ré-exécute le nœud depuis le début ;
on n'écrit donc rien (ni journal, ni fichier) avant l'appel à interrupt().
"""
from __future__ import annotations

from src.state import EtatPortefeuille
from src.tools.reporting import journaliser

DECISIONS = {"approuve", "modifie", "rejete"}


def demande_validation(state: EtatPortefeuille) -> dict:
    """Ce que voit le gérant (pur, sans effet de bord)."""
    rr = state["rapport_risque"]
    return {
        "date_analyse": state["date_analyse"],
        "poids": state["poids"],
        "attendus": {k: state["attendus"][k] for k in ("rendement", "volatilite", "sharpe")},
        "conforme": rr["conforme"],
        "alerte": rr.get("alerte", False),
        "problemes": rr["manquements_mandat"] + rr["depassements_limites"],
        "explication_risque": rr["explication"],
        "vues": state["vues"],
        "consigne": "Répondre {'decision': 'approuve'|'modifie'|'rejete', 'poids': {...} si modifié, 'motif': '...'}",
    }


def normaliser_decision(rep) -> dict:
    if isinstance(rep, str):
        rep = {"decision": rep}
    d = str(rep.get("decision", "")).lower().replace("é", "e")
    if d not in DECISIONS:
        raise ValueError(f"décision inconnue : {rep!r} (attendu : {sorted(DECISIONS)})")
    out = {"decision": d, "motif": rep.get("motif", "")}
    if d == "modifie":
        w = {t.upper(): float(v) for t, v in (rep.get("poids") or {}).items() if float(v) > 0}
        if not w:
            raise ValueError("décision « modifie » sans poids")
        tot = sum(w.values())
        out["poids"] = {t: round(v / tot, 4) for t, v in w.items()}   # renormalisation à 100 %
    return out


def appliquer_decision(state: EtatPortefeuille, ctx, reponse) -> dict:
    dec = normaliser_decision(reponse)
    journaliser(ctx.store, state["date_analyse"], f"decision_{dec['decision']}",
                {"decision": dec, "poids_proposes": state["poids"], "alerte": state["rapport_risque"].get("alerte", False)})
    upd = {"decision_gerant": dec,
           "journal": [f"[validation_gerant] {dec['decision'].upper()}" + (f" — {dec['motif']}" if dec["motif"] else "")]}
    if dec["decision"] == "modifie":
        upd["poids"] = dec["poids"]
    return upd


def validation_gerant(state: EtatPortefeuille, ctx) -> dict:
    from langgraph.types import interrupt

    reponse = interrupt(demande_validation(state))   # pause ; reprise avec Command(resume=...)
    return appliquer_decision(state, ctx, reponse)


def router_validation(state: EtatPortefeuille) -> str:
    return {"approuve": "rapport", "modifie": "controle_risque", "rejete": "__end__"}[state["decision_gerant"]["decision"]]
