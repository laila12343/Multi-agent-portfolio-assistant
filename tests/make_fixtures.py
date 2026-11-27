"""Régénère tests/fixtures/*.json (données synthétiques, LLM mock) :

    python -m tests.make_fixtures

Une fixture par frontière d'étape : chacun teste SON nœud sans faire tourner l'amont.
  etat_metriques.json   état après calculer_metriques (étapes 1-3)  -> pour les analystes
  notes.json            notes des 3 analystes (étape 4)            -> pour le débat
  etat_selection.json   état après le débat (étape 5)              -> pour optimiser / controle_risque
"""
import json
from pathlib import Path

from src.config import AppContext, load_settings
from src.local_runner import _apply
from src.agents.analystes import analyse_titre_sans_graphe
from src.agents.debat import executer_debat
from src.nodes import preparation

FIX = Path(__file__).parent / "fixtures"
DATE = "2026-11-20"


def offline_settings():
    return load_settings(overrides={"llm": {"backend": "mock"},
                                    "donnees": {"fournisseur": "synthetic", "historique_annees": 2}})


def main():
    ctx = AppContext.build(offline_settings())
    st = {"date_analyse": DATE, "notes": [], "journal": []}
    for node in (preparation.charger_mandat, preparation.construire_univers,
                 preparation.collecter_donnees, preparation.calculer_metriques):
        _apply(st, node(st, ctx))
    FIX.mkdir(exist_ok=True)
    (FIX / "etat_metriques.json").write_text(json.dumps(st, ensure_ascii=False))
    for p in preparation.payloads_analyse(st, ctx):
        _apply(st, analyse_titre_sans_graphe(p, ctx))
    (FIX / "notes.json").write_text(json.dumps(st["notes"], ensure_ascii=False, indent=1))
    _apply(st, executer_debat(st, ctx, avec_graphe=False))
    (FIX / "etat_selection.json").write_text(json.dumps(st, ensure_ascii=False))
    for f in sorted(FIX.glob("*.json")):
        print("écrit", f, f"{f.stat().st_size / 1024:.0f} Ko")


if __name__ == "__main__":
    main()
