"""Observabilité LangSmith : traces de chaque nœud, appel LLM, outil, et liens vers les traces.

Activation : variable LANGSMITH_API_KEY (offre « Developer » gratuite, 5 000 traces / mois).
Sans clé, tout fonctionne à l'identique, simplement sans traces.

Ce qui est tracé :
  - automatiquement : chaque nœud LangGraph, chaque sous-graphe, chaque appel LLM (ChatOllama…)
  - explicitement (@trace) : les outils « code » (Black-Litterman, optimisation, risque, rapport…)
  - métadonnées par cycle : thread_id (vue « Threads »), date d'analyse, version du mandat, backend LLM
"""
from __future__ import annotations

import functools
import os
import uuid

DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def configurer_langsmith(settings: dict) -> bool:
    """Active le traçage si une clé est disponible. Renvoie True si actif."""
    cfg = settings.get("observabilite", {}).get("langsmith", {})
    mode = str(cfg.get("actif", "auto")).lower()
    cle = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    actif = (mode == "auto" and bool(cle)) or (mode in ("true", "oui", "1") and bool(cle))
    os.environ["LANGSMITH_TRACING"] = "true" if actif else "false"
    if actif:
        os.environ.setdefault("LANGSMITH_PROJECT", cfg.get("projet", "portfolio-agents"))
        os.environ.setdefault("LANGSMITH_ENDPOINT", cfg.get("endpoint", DEFAULT_ENDPOINT))
    return actif


def langsmith_actif() -> bool:
    return os.getenv("LANGSMITH_TRACING", "false").lower() == "true"


def projet() -> str:
    return os.getenv("LANGSMITH_PROJECT", "portfolio-agents")


def _resume(v):
    """Allège les entrées/sorties envoyées à LangSmith (pas de DataFrame de 50 x 750 dans les traces)."""
    try:
        import pandas as pd

        if isinstance(v, pd.DataFrame):
            return f"DataFrame {v.shape[0]}x{v.shape[1]} : {list(v.columns)[:12]}"
        if isinstance(v, pd.Series):
            return v.round(4).to_dict() if len(v) <= 60 else f"Series de {len(v)} valeurs"
    except ImportError:
        pass
    if isinstance(v, dict) and len(str(v)) > 4000:
        return {k: _resume(x) if not isinstance(x, (int, float, str, bool)) else x for k, x in list(v.items())[:60]}
    return v


def _process(d: dict) -> dict:
    return {k: _resume(v) for k, v in d.items()}


def trace(name: str | None = None, run_type: str = "tool"):
    """Décorateur : @traceable de LangSmith s'il est installé, sinon fonction inchangée."""
    def deco(fn):
        try:
            from langsmith import traceable

            try:
                return traceable(name=name or fn.__name__, run_type=run_type,
                                 process_inputs=_process, process_outputs=lambda o: _resume(o))(fn)
            except TypeError:   # anciennes versions de langsmith
                return traceable(name=name or fn.__name__, run_type=run_type)(fn)
        except ImportError:
            @functools.wraps(fn)
            def inner(*a, **k):
                return fn(*a, **k)
            return inner
    return deco


def config_trace(settings: dict, thread_id: str, date_analyse: str | None, etape: str) -> dict:
    """Champs de config LangGraph/LangChain pour nommer et retrouver un run dans LangSmith."""
    return {
        "run_id": uuid.uuid4(),
        "run_name": f"cycle {date_analyse or 'auto'} / {etape}",
        "tags": ["portfolio-agents", etape, settings["llm"]["backend"]],
        "metadata": {"thread_id": thread_id, "date_analyse": date_analyse, "etape": etape,
                     "backend_llm": settings["llm"]["backend"], "fournisseur": settings["donnees"]["fournisseur"]},
    }


def url_run(run_id) -> str | None:
    """Lien direct vers la trace d'un run (None si LangSmith inactif ou indisponible)."""
    if not langsmith_actif() or run_id is None:
        return None
    try:
        from langsmith import Client

        client = Client()
        return client.get_run_url(run=client.read_run(str(run_id)), project_name=projet())
    except Exception:
        return None


def url_projet() -> str | None:
    if not langsmith_actif():
        return None
    try:
        from langsmith import Client

        return Client().read_project(project_name=projet()).url
    except Exception:
        return "https://smith.langchain.com"
