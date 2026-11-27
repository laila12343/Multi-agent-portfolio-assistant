"""Assemblage LangGraph (section 3 de l'architecture).

START -> charger_mandat -> construire_univers -> collecter_donnees [RetryPolicy] -> calculer_metriques
      -> Send x N -> analyse_titre (sous-graphe : 3 agents en parallèle, notes fusionnées par reducer)
      -> debat (sous-graphe : haussier <-> baissier, 2 tours, puis juge_selection)
      -> optimiser -> controle_risque --non conforme (<= 3)--> optimiser
                                      --conforme (ou 3 essais)--> validation_gerant [interrupt()]
      validation_gerant --approuvé--> rapport -> END
                        --modifié---> controle_risque
                        --rejeté----> END
Checkpointer (Memory / SQLite / PostgresSaver), Store (journal d'audit), recursion_limit.
"""
from __future__ import annotations

import sqlite3
import uuid

from src.agents.analystes import build_analyse_titre
from src.agents.debat import executer_debat
from src.config import AppContext, resolve_path
from src.nodes import portefeuille, preparation, rapport, validation


def get_checkpointer(settings: dict):
    kind = settings["graphe"].get("checkpointer", "memory")
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = resolve_path(settings["graphe"]["sqlite_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))
    if kind == "postgres":
        import os

        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection

        conn = Connection.connect(os.environ["POSTGRES_URI"], autocommit=True, prepare_threshold=0)
        saver = PostgresSaver(conn)
        saver.setup()
        return saver
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def build_graph(ctx: AppContext, checkpointer=None):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import RetryPolicy

    from src.state import EtatPortefeuille

    sous_graphe_analyse = build_analyse_titre(ctx)

    def analyse_titre(payload: dict) -> dict:
        # on ne remonte que les notes : seule clé partagée avec l'état principal
        return {"notes": sous_graphe_analyse.invoke(payload)["notes"]}

    g = StateGraph(EtatPortefeuille)
    g.add_node("charger_mandat", lambda s: preparation.charger_mandat(s, ctx))
    g.add_node("construire_univers", lambda s: preparation.construire_univers(s, ctx))
    g.add_node("collecter_donnees", lambda s: preparation.collecter_donnees(s, ctx),
               retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0, backoff_factor=2.0))
    g.add_node("calculer_metriques", lambda s: preparation.calculer_metriques(s, ctx))
    g.add_node("analyse_titre", analyse_titre)
    g.add_node("debat", lambda s: executer_debat(s, ctx))
    g.add_node("optimiser", lambda s: portefeuille.optimiser(s, ctx))
    g.add_node("controle_risque", lambda s: portefeuille.controle_risque(s, ctx))
    g.add_node("validation_gerant", lambda s: validation.validation_gerant(s, ctx))
    g.add_node("rapport", lambda s: rapport.rapport(s, ctx))

    g.add_edge(START, "charger_mandat")
    g.add_edge("charger_mandat", "construire_univers")
    g.add_edge("construire_univers", "collecter_donnees")
    g.add_edge("collecter_donnees", "calculer_metriques")
    g.add_conditional_edges("calculer_metriques", lambda s: preparation.router_analyses(s, ctx), ["analyse_titre"])
    g.add_edge("analyse_titre", "debat")          # attend la fin de TOUTES les analyses (super-step)
    g.add_edge("debat", "optimiser")
    g.add_edge("optimiser", "controle_risque")
    g.add_conditional_edges("controle_risque", lambda s: portefeuille.router_risque(s, ctx),
                            ["optimiser", "validation_gerant"])
    g.add_conditional_edges("validation_gerant", validation.router_validation,
                            {"rapport": "rapport", "controle_risque": "controle_risque", "__end__": END})
    g.add_edge("rapport", END)
    return g.compile(checkpointer=checkpointer or get_checkpointer(ctx.settings), store=ctx.store)


# ------------------------------------------------------------------ exécution d'un cycle
def config_cycle(ctx: AppContext, thread_id: str | None = None) -> dict:
    return {"configurable": {"thread_id": thread_id or f"cycle-{uuid.uuid4().hex[:8]}"},
            "recursion_limit": ctx.settings["graphe"].get("recursion_limit", 100),
            "max_concurrency": ctx.settings["analyse"].get("max_concurrency", 4),
            "_settings": ctx.settings}


def _avec_trace(config: dict, date_analyse: str | None, etape: str) -> dict:
    """Ajoute run_id / nom / tags / métadonnées LangSmith ; garde la trace des run_id du cycle."""
    from src.observabilite import config_trace

    cfg = {k: v for k, v in config.items() if k != "_settings"}
    tr = config_trace(config["_settings"], config["configurable"]["thread_id"], date_analyse, etape)
    config.setdefault("_runs", []).append({"etape": etape, "run_id": str(tr["run_id"])})
    return {**cfg, **tr}


def interruption_en_cours(app, config) -> dict | None:
    """Contenu de l'interrupt() en attente (demande de validation au gérant), sinon None."""
    snap = app.get_state(_propre(config))
    for task in snap.tasks:
        for it in getattr(task, "interrupts", ()) or ():
            return it.value
    return None


def lancer_cycle(app, config, date_analyse: str | None = None, verbose: bool = True) -> dict | None:
    """Exécute jusqu'à la validation du gérant. Renvoie la demande de validation (ou None si terminé)."""
    _stream(app, {"date_analyse": date_analyse} if date_analyse else {},
            _avec_trace(config, date_analyse, "analyse"), verbose)
    return interruption_en_cours(app, config)


def reprendre(app, config, decision, verbose: bool = True) -> dict | None:
    """Reprend après la décision du gérant (Command(resume=...))."""
    from langgraph.types import Command

    etape = decision.get("decision", "decision") if isinstance(decision, dict) else str(decision)
    _stream(app, Command(resume=decision), _avec_trace(config, None, f"validation_{etape}"), verbose)
    return interruption_en_cours(app, config)


def executer_flux(app, config: dict, entree, date_analyse: str | None = None, etape: str = "analyse"):
    """Générateur (nœud, mise à jour) pour suivre un cycle en direct (interface Streamlit)."""
    for chunk in app.stream(entree, _propre(_avec_trace(config, date_analyse, etape)), stream_mode="updates"):
        for node, upd in chunk.items():
            yield node, upd


def liens_traces(config: dict) -> list[dict]:
    """[{etape, run_id, url}] pour les runs du cycle (url None si LangSmith inactif)."""
    from src.observabilite import url_run

    return [{**r, "url": url_run(r["run_id"])} for r in config.get("_runs", [])]


def _propre(config: dict) -> dict:
    """Config sans les clés internes (préfixées « _ »)."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def etat_cycle(app, config) -> dict:
    return app.get_state(_propre(config)).values


def _stream(app, inp, config, verbose):
    vus = 0
    for chunk in app.stream(inp, _propre(config), stream_mode="updates"):
        for node, upd in chunk.items():
            if not verbose or not isinstance(upd, dict):
                continue
            if node == "analyse_titre":
                vus += 1
                if vus % 10 == 0:
                    print(f"  analyse_titre : {vus} titres analysés")
            for line in upd.get("journal", []):
                print(line)
