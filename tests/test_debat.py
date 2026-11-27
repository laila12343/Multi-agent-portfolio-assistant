"""Étape 5 : débat + juge + garde-fous du code."""
from src.agents.debat import MAX_PAR_SECTEUR, agreger_notes, executer_debat, valider_selection
from src.state import Selection, VueTitre


def _agg(etat, notes, ctx):
    sect = {t: m["secteur"] for t, m in etat["metriques"].items()}
    return agreger_notes(notes, ctx.settings["analyse"]["poids_notes"], sect)


def test_garde_fous_juge(etat_metriques, notes, ctx):
    agg = _agg(etat_metriques, notes, ctx)
    sel = Selection(titres=[VueTitre(ticker="INVENTE", vue=2, confiance="elevee", justification="x")] +
                    [VueTitre(ticker=t, vue=1, confiance="moyenne", justification="x") for t in agg.index[:3]])
    out = valider_selection(sel, agg, ctx.settings["debat"], etat_metriques["mandat"])
    tickers = [v["ticker"] for v in out]
    assert "INVENTE" not in tickers                                   # ticker hors candidats rejeté
    assert ctx.settings["debat"]["taille_min"] <= len(out) <= ctx.settings["debat"]["taille_max"]
    assert tickers[:3] == list(agg.index[:3])                         # choix du juge conservés
    secteurs = [agg.loc[t, "secteur"] for t in tickers]
    assert max(secteurs.count(s) for s in set(secteurs)) <= MAX_PAR_SECTEUR


def test_debat_complet(etat_metriques, notes, ctx):
    etat = {**etat_metriques, "notes": notes}
    out = executer_debat(etat, ctx, avec_graphe=False)
    assert [m["role"] for m in out["debat"]] == ["haussier", "baissier"] * ctx.settings["debat"]["tours"]
    assert set(out["vues"]) == set(out["selection"])
    assert all(v["confiance"] in {"faible", "moyenne", "elevee"} for v in out["vues"].values())
