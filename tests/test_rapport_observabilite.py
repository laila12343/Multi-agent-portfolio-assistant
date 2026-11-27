"""Rapport HTML, tableaux de synthèse et observabilité LangSmith."""
import os

import pytest

from src.local_runner import executer_sans_langgraph
from src.observabilite import config_trace, configurer_langsmith, langsmith_actif
from src.tools import syntheses as sy
from src.tools.rapport_html import num, pct


@pytest.fixture
def cycle(ctx, date_analyse):
    return executer_sans_langgraph(ctx, date_analyse, decisions=[{"decision": "approuve", "motif": "test"}])


def test_formats_francais():
    assert pct(0.1234) == "12,3 %" and pct(-0.05, 0) == "−5 %" and num(1234567) == "1 234 567"


def test_rapport_complet(cycle):
    doc = open(cycle["rapport_fichier"], encoding="utf-8").read()
    for titre in ("Synthèse", "Les convictions du comité", "Allocation", "Des vues aux poids", "Risque",
                  "Le débat du comité", "Ordres à passer", "Méthode et traçabilité"):
        assert titre in doc
    assert doc.count("<svg") >= 4                       # graphiques intégrés
    assert "nan" not in doc.lower().replace("financ", "")  # aucune valeur manquante affichée brute
    assert "Approuvé par le gérant" in doc


def test_syntheses_coherentes(cycle):
    assert abs(sy.exposition_sectorielle(cycle)["portefeuille"].sum() - 1) < 1e-3
    tr = sy.tableau_risque(cycle)
    assert set(tr["statut"]) <= {"respectée", "dépassée"} and len(tr) == 6
    perf = sy.performance(cycle)
    assert perf["dd_portefeuille"].max() <= 0


def test_langsmith_inactif_sans_cle(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert configurer_langsmith({"observabilite": {"langsmith": {"actif": "auto"}}}) is False
    assert not langsmith_actif()


def test_langsmith_actif_avec_cle(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert configurer_langsmith({"observabilite": {"langsmith": {"actif": "auto", "projet": "pa-test"}}}) is True
    assert os.environ["LANGSMITH_TRACING"] == "true" and os.environ["LANGSMITH_PROJECT"] == "pa-test"
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_metadonnees_de_trace(ctx):
    tr = config_trace(ctx.settings, "cycle-x", "2026-11-20", "analyse")
    assert tr["metadata"]["thread_id"] == "cycle-x" and "analyse" in tr["tags"] and tr["run_id"]
