"""Chargement des réglages et contexte partagé (hors état) des nœuds."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS = ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    with open(Path(path) if path else DEFAULT_SETTINGS, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if os.getenv("LLM_BACKEND"):
        settings["llm"]["backend"] = os.environ["LLM_BACKEND"]
    if os.getenv("DATA_SOURCE"):
        settings["donnees"]["fournisseur"] = os.environ["DATA_SOURCE"]
    if overrides:
        _deep_update(settings, overrides)
    return settings


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@dataclass
class AppContext:
    """Dépendances injectées dans les nœuds par fermeture (jamais dans l'état) :
    réglages, modèles LLM, Store du journal d'audit, caches en mémoire."""

    settings: dict
    llm_rapide: Any = None            # BaseChatModel LangChain, ou None en mode mock
    llm_fort: Any = None
    store: Any = None                 # langgraph BaseStore (journal d'audit, H10)
    cache: dict = field(default_factory=dict)

    @property
    def backend(self) -> str:
        return self.settings["llm"]["backend"]

    @property
    def mock(self) -> bool:
        return self.backend == "mock"

    @classmethod
    def build(cls, settings: dict | None = None, backend: str | None = None, store: Any = None) -> "AppContext":
        from src.llm import get_chat_model
        from src.observabilite import configurer_langsmith

        settings = settings or load_settings()
        if backend:
            settings["llm"]["backend"] = backend
        configurer_langsmith(settings)      # avant la création des modèles : leurs appels seront tracés
        if store is None:
            store = _default_store()
        return cls(settings=settings, llm_rapide=get_chat_model("rapide", settings),
                   llm_fort=get_chat_model("fort", settings), store=store)


def _default_store():
    try:
        from langgraph.store.memory import InMemoryStore

        return InMemoryStore()
    except ImportError:          # tests hors LangGraph
        from src.tools.reporting import DictStore

        return DictStore()
