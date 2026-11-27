"""Étapes 8-9 : décision du gérant, rapport, journal d'audit — et le workflow complet sans LangGraph."""
import pytest

from src.local_runner import executer_sans_langgraph
from src.nodes.validation import normaliser_decision, router_validation
from src.tools.reporting import calculer_mouvements, lire_journal


def test_normaliser_decision():
    assert normaliser_decision("approuvé")["decision"] == "approuve"
    d = normaliser_decision({"decision": "modifie", "poids": {"aapl": 2, "msft": 2}})
    assert d["poids"] == {"AAPL": 0.5, "MSFT": 0.5}
    with pytest.raises(ValueError):
        normaliser_decision({"decision": "peut-être"})


def test_router_validation():
    assert router_validation({"decision_gerant": {"decision": "modifie"}}) == "controle_risque"
    assert router_validation({"decision_gerant": {"decision": "rejete"}}) == "__end__"


def test_mouvements():
    df = calculer_mouvements({"A": 0.5, "B": 0.5}, {"A": 10, "C": 5, "CASH": 0}, {"A": 10, "B": 20, "C": 1}, 1000)
    sens = dict(zip(df["ticker"], df["sens"]))
    assert sens == {"A": "ACHAT", "B": "ACHAT", "C": "VENTE"}


def test_cycle_approuve(ctx, date_analyse):
    st = executer_sans_langgraph(ctx, date_analyse, decisions=["approuve"])
    assert "Ordres à passer manuellement" in st["rapport_final"]
    evts = [e["evenement"] for e in lire_journal(ctx.store, date_analyse)]
    assert evts[0] == "selection" and "decision_approuve" in evts and evts[-1] == "rapport"
    rec = lire_journal(ctx.store, date_analyse)[-1]
    assert rec["donnees"]["version"] and rec["mandat"]["empreinte"]          # H10


def test_cycle_rejete(ctx, date_analyse):
    st = executer_sans_langgraph(ctx, date_analyse, decisions=[{"decision": "rejete", "motif": "marché trop volatil"}])
    assert "rapport_final" not in st
    rej = [e for e in lire_journal(ctx.store, date_analyse) if e["evenement"] == "decision_rejete"]
    assert rej[0]["decision"]["motif"] == "marché trop volatil"
