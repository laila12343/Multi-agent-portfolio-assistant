"""Graphe LangGraph complet : Send, sous-graphes, boucle de risque, interrupt() / Command(resume)."""
import pytest

pytest.importorskip("langgraph")

from src.graph import build_graph, config_cycle, lancer_cycle, reprendre  # noqa: E402


def test_cycle_avec_validation_humaine(ctx, date_analyse):
    ctx.settings["analyse"]["max_titres"] = 20
    app = build_graph(ctx)
    cfg = config_cycle(ctx, "test-approuve")
    demande = lancer_cycle(app, cfg, date_analyse, verbose=False)
    assert demande is not None and demande["poids"]                       # en pause sur interrupt()
    etat = app.get_state(cfg).values
    assert len(etat["notes"]) == 3 * 20                                   # 3 notes x 20 Send
    assert reprendre(app, cfg, {"decision": "approuve"}, verbose=False) is None
    final = app.get_state(cfg).values
    assert final["rapport_final"] and final["decision_gerant"]["decision"] == "approuve"


def test_rejet(ctx, date_analyse):
    ctx.settings["analyse"]["max_titres"] = 15
    app = build_graph(ctx)
    cfg = config_cycle(ctx, "test-rejet")
    lancer_cycle(app, cfg, date_analyse, verbose=False)
    reprendre(app, cfg, {"decision": "rejete", "motif": "test"}, verbose=False)
    assert "rapport_final" not in app.get_state(cfg).values
