"""Étape 4 : sous-graphe analyse_titre (mode règles hors-ligne)."""
from src.agents.analystes import analyse_titre_sans_graphe, note_regles
from src.nodes.preparation import payloads_analyse
from src.state import NoteAnalyste


def test_payload_contient_le_necessaire(etat_metriques, ctx):
    p = payloads_analyse(etat_metriques, ctx)
    assert len(p) == min(ctx.settings["analyse"]["max_titres"], len(etat_metriques["metriques"]))
    assert {"ticker", "metriques_titre", "pairs", "actualites", "date_analyse"} <= set(p[0])


def test_trois_notes_par_titre(etat_metriques, ctx):
    p = payloads_analyse(etat_metriques, ctx)[0]
    notes = analyse_titre_sans_graphe(p, ctx)["notes"]
    assert sorted(n["agent"] for n in notes) == ["fondamental", "sentiment", "valorisation"]
    assert all(n["ticker"] == p["ticker"] and -2 <= n["score"] <= 2 for n in notes)


def test_note_bornee():
    assert NoteAnalyste(ticker="X", agent="sentiment", score=7, these="t").score == 2


def test_regles_valorisation(etat_metriques, ctx):
    p = payloads_analyse(etat_metriques, ctx)[0]
    n = note_regles("valorisation", p)
    assert "médiane" in n.these and n.sources == ["comparer_pairs"]
