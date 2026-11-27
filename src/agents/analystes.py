"""Étape 4 — sous-graphe analyse_titre : 3 agents LLM en parallèle pour UN titre.

    START ─┬─ agent_fondamental  ─┬─ END        chaque agent renvoie une NoteAnalyste (Pydantic) ;
           ├─ agent_sentiment    ─┤            le reducer operator.add fusionne les notes
           └─ agent_valorisation ─┘

Trois modes selon le backend LLM :
  outils   (ollama, openai_compatible) : create_agent (LangChain) + outils + response_format=NoteAnalyste
  contexte (hf)  : le code appelle les outils et fournit leurs résultats ; sortie structurée parsée
  regles   (mock) : note déterministe dérivée des scores standardisés (tests, démo hors-ligne)
"""
from __future__ import annotations

import json
import operator
import re
from typing import Annotated, TypedDict

from src.agents import prompts
from src.llm import structured_call, supports_tools
from src.state import NoteAnalyste
from src.tools.news import recherche_web


class AnalyseTitreState(TypedDict, total=False):
    ticker: str
    nom: str
    secteur: str
    date_analyse: str
    metriques_titre: dict
    pairs: dict
    actualites: list[dict]
    notes: Annotated[list[dict], operator.add]


# ------------------------------------------------------------------ outils (fermetures sur le titre)
def _outils(s: AnalyseTitreState, ctx) -> dict:
    """Outils de l'inventaire, liés au titre analysé. Renvoient du JSON (lecture du state)."""
    from langchain_core.tools import tool

    @tool
    def get_metriques_titre() -> str:
        """Métriques calculées par le code pour le titre (rendements, volatilité, ratios, percentiles)."""
        return json.dumps(s["metriques_titre"], ensure_ascii=False, default=str)

    @tool
    def rechercher_rapport(question: str) -> str:
        """Recherche dans les rapports annuels du titre. Argument : une question précise en anglais."""
        res = _rag(s, ctx, question)
        return json.dumps(res, ensure_ascii=False) if res else "aucun rapport disponible pour ce titre"

    @tool
    def get_actualites() -> str:
        """Actualités récentes du titre (titre, date, source), publiées avant la date d'analyse."""
        return json.dumps(_actus(s), ensure_ascii=False)

    @tool
    def recherche_web(requete: str) -> str:
        """Recherche web d'actualités complémentaires. Argument : requête courte."""
        res = _web(s, ctx, requete)
        return json.dumps(res, ensure_ascii=False) if res else "recherche web indisponible ou sans résultat"

    @tool
    def comparer_pairs() -> str:
        """Valorisation du titre vs médiane des pairs de son secteur (PER, EV/EBITDA, P/B…)."""
        return json.dumps(s["pairs"], ensure_ascii=False)

    return {"fondamental": [get_metriques_titre, rechercher_rapport],
            "sentiment": [get_actualites, recherche_web],
            "valorisation": [comparer_pairs, get_metriques_titre]}


def _rag(s, ctx, question):
    from src.tools import rag

    return rag.rechercher_rapport(s["ticker"], question, ctx.settings, s["date_analyse"])


def _actus(s):
    return [{"titre": a["title"], "date": a["published"][:10], "source": a.get("source", "")} for a in s["actualites"]]


def _web(s, ctx, requete):
    return recherche_web(requete, ctx.settings)


def _contexte(agent: str, s: AnalyseTitreState, ctx) -> str:
    """Mode « contexte » : le code exécute les outils et donne leurs résultats au LLM."""
    if agent == "fondamental":
        extraits = _rag(s, ctx, f"{s['nom']} risks outlook profitability debt growth")
        return json.dumps({"metriques": s["metriques_titre"], "rapports": extraits}, ensure_ascii=False, default=str)
    if agent == "sentiment":
        return json.dumps({"actualites": _actus(s)}, ensure_ascii=False)
    return json.dumps({"pairs": s["pairs"], "metriques": s["metriques_titre"]}, ensure_ascii=False, default=str)


# ------------------------------------------------------------------ règles (mode mock)
POS = {"beat", "beats", "record", "upgrade", "upgrades", "rally", "surge", "growth", "raises", "strong", "robust",
       "buyback", "profit", "gain", "gains", "wins", "expands"}
NEG = {"miss", "misses", "downgrade", "downgrades", "fall", "falls", "drop", "plunge", "lawsuit", "investigation",
       "cuts", "weak", "weakens", "loss", "recall", "slowing", "fraud", "decline", "disruptions"}


def _to_score(x: float, k: float = 1.5) -> int:
    return max(-2, min(2, int(round(k * x))))


def note_regles(agent: str, s: AnalyseTitreState) -> NoteAnalyste:
    m, t = s["metriques_titre"], s["ticker"]
    if agent == "fondamental":
        x = ((m.get("z_qualite") or 0) + (m.get("z_croissance") or 0)) / 2
        these = (f"ROE au percentile {m.get('percentile_roe', 'n.d.')}, marge nette au percentile "
                 f"{m.get('percentile_marge_nette', 'n.d.')}, croissance du CA au percentile {m.get('percentile_croissance_ca', 'n.d.')}.")
        risques = ["endettement élevé"] if (m.get("dette_fp") or 0) > 1.5 else []
        return NoteAnalyste(ticker=t, agent=agent, score=_to_score(x), these=these, risques=risques,
                            sources=["get_metriques_titre"])
    if agent == "valorisation":
        e = s["pairs"]["ecart_relatif_vs_mediane"]
        x = m.get("z_valorisation") or 0
        these = f"PER prévisionnel {_pct(e.get('per_prevu'))} vs médiane du secteur, EV/EBITDA {_pct(e.get('ev_ebitda'))}."
        return NoteAnalyste(ticker=t, agent=agent, score=_to_score(x), these=these,
                            risques=["prime de valorisation"] if x < -0.5 else [], sources=["comparer_pairs"])
    vals = []
    for a in s["actualites"]:
        w = re.findall(r"[a-z]+", a["title"].lower())
        p, n = sum(x in POS for x in w), sum(x in NEG for x in w)
        if p + n:
            vals.append((p - n) / (p + n))
    x = sum(vals) / len(s["actualites"]) if s["actualites"] else 0.0
    return NoteAnalyste(ticker=t, agent=agent, score=_to_score(x, 2.5),
                        these=f"{len(s['actualites'])} actualités : {sum(v > 0 for v in vals)} positives, {sum(v < 0 for v in vals)} négatives.",
                        risques=[a["title"] for a in s["actualites"] if any(k in a["title"].lower() for k in NEG)][:2],
                        sources=[a["title"] for a in s["actualites"][:3]])


def _pct(v):
    return "n.d." if v is None else f"{v:+.0%}"


# ------------------------------------------------------------------ un agent
PROMPTS = {"fondamental": prompts.FONDAMENTAL, "sentiment": prompts.SENTIMENT, "valorisation": prompts.VALORISATION}


def analyser(agent: str, s: AnalyseTitreState, ctx) -> dict:
    note = None
    demande = f"Analyse {s['nom']} ({s['ticker']}, secteur {s['secteur']}) au {s['date_analyse']}."
    if ctx.llm_rapide is not None and supports_tools(ctx.settings):
        try:
            from langchain.agents import create_agent

            ag = create_agent(model=ctx.llm_rapide, tools=_outils(s, ctx)[agent],
                              system_prompt=PROMPTS[agent], response_format=NoteAnalyste)
            note = ag.invoke({"messages": [{"role": "user", "content": demande}]})["structured_response"]
        except Exception as exc:
            print(f"[{agent}] {s['ticker']} : agent à outils en échec ({type(exc).__name__}: {str(exc)[:200]}) -> mode contexte")
    if note is None and ctx.llm_rapide is not None:
        note = structured_call(ctx.llm_rapide, NoteAnalyste, PROMPTS[agent],
                               f"{demande}\nDonnées des outils :\n{_contexte(agent, s, ctx)}")
    if isinstance(note, dict):                     # sortie brute : on la valide
        try:
            note = NoteAnalyste.model_validate({**note, "ticker": s["ticker"], "agent": agent})
        except Exception:
            note = None
    if not isinstance(note, NoteAnalyste):
        note = note_regles(agent, s)
    note = note.model_copy(update={"ticker": s["ticker"], "agent": agent})   # garde-fou : identité imposée
    return {"notes": [note.model_dump()]}


def build_analyse_titre(ctx):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AnalyseTitreState)
    for agent in ("fondamental", "sentiment", "valorisation"):
        g.add_node(f"agent_{agent}", lambda s, a=agent: analyser(a, s, ctx))
        g.add_edge(START, f"agent_{agent}")
        g.add_edge(f"agent_{agent}", END)
    return g.compile()


def analyse_titre_sans_graphe(payload: dict, ctx) -> dict:
    """Équivalent séquentiel du sous-graphe (tests sans LangGraph, débogage d'un titre)."""
    notes = []
    for agent in ("fondamental", "sentiment", "valorisation"):
        notes += analyser(agent, payload, ctx)["notes"]
    return {"notes": notes}
