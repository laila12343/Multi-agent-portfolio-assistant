"""État partagé du graphe (EtatPortefeuille) et sorties structurées des LLM.

Règle d'équipe : chaque nœud LIT l'état et RENVOIE uniquement les clés qu'il possède.

    charger_mandat       -> date_analyse, mandat
    construire_univers   -> univers
    collecter_donnees    -> donnees
    calculer_metriques   -> metriques
    analyse_titre (xN)   -> notes            (fusionnées par le reducer operator.add)
    debat                -> debat, selection, vues
    optimiser            -> poids, iterations_risque
    controle_risque      -> rapport_risque
    validation_gerant    -> decision_gerant (et poids si « modifié »)
    rapport              -> rapport_final

Tout est JSON-sérialisable (le checkpointer sauvegarde l'état à chaque étape).
Les modèles Pydantic sont convertis en dict (model_dump) avant d'entrer dans l'état.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

Agent = Literal["fondamental", "sentiment", "valorisation"]
Confiance = Literal["faible", "moyenne", "elevee"]


# ------------------------------------------------------------------ sorties structurées (Pydantic)
class NoteAnalyste(BaseModel):
    """Sortie d'un agent analyste pour UN titre (étape 4)."""

    ticker: str
    agent: Agent
    score: int = Field(description="Avis de -2 (très négatif) à +2 (très positif)", ge=-2, le=2)
    these: str = Field(description="Thèse en 2-3 phrases, sans inventer de chiffre")
    risques: list[str] = Field(default_factory=list, description="Principaux risques")
    sources: list[str] = Field(default_factory=list, description="Outils ou documents utilisés")

    @field_validator("score", mode="before")
    @classmethod
    def _borne(cls, v):
        return max(-2, min(2, int(round(float(v)))))


class VueTitre(BaseModel):
    ticker: str
    vue: int = Field(description="-2 très baissier … +2 très haussier", ge=-2, le=2)
    confiance: Confiance
    justification: str

    @field_validator("vue", mode="before")
    @classmethod
    def _borne(cls, v):
        return max(-2, min(2, int(round(float(v)))))


class Selection(BaseModel):
    """Sortie du juge (étape 5) : liste restreinte, vues et confiance."""

    titres: list[VueTitre]
    synthese: str = ""


# ------------------------------------------------------------------ état LangGraph
class EtatPortefeuille(TypedDict, total=False):
    # contexte
    date_analyse: str                     # AAAA-MM-JJ (point-in-time, H8)
    mandat: dict
    # étapes 1-3 (code)
    univers: dict                         # {"eligibles": [...], "exclus": [{"ticker", "motif"}]}
    donnees: dict                         # prix, volumes, fondamentaux, actualités, version, sources
    metriques: dict                       # ticker -> métriques + scores standardisés
    # étape 4 (LLM, en parallèle)
    notes: Annotated[list[dict], operator.add]
    # étape 5 (LLM)
    debat: list[dict]                     # [{"role": "haussier"|"baissier", "tour": n, "texte": ...}]
    synthese_debat: str                   # arbitrage rédigé par le juge
    selection: list[str]
    vues: dict                            # ticker -> {"vue", "confiance", "justification"}
    # étapes 6-7 (code + LLM)
    poids: dict                           # ticker -> poids cible
    attendus: dict                        # rendement / volatilité / Sharpe attendus
    rapport_risque: dict
    iterations_risque: int
    # étapes 8-9
    decision_gerant: dict                 # {"decision": approuve|modifie|rejete, "motif", ...}
    rapport_final: str                    # version Markdown
    rapport_fichier: str                  # chemin du rapport HTML autonome
    ordres: list[dict]                    # ordres à passer manuellement
    journal: Annotated[list[str], operator.add]   # trace lisible de chaque étape
