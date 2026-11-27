"""Étapes 1-3 : mandat, univers, données, métriques (code déterministe)."""
import pandas as pd
import pytest

from src.tools.mandat import contraintes_mandat, lire_mandat
from src.tools.market_data import filtrer_univers
from src.tools.metrics import comparer_pairs, get_metriques_titre


def test_lire_mandat():
    m = lire_mandat("config/mandat.yaml")
    assert m["contraintes"]["poids_max_titre"] == 0.10 and len(m["empreinte"]) == 12


def test_mandat_incoherent(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("version: x\nvaleur_portefeuille: 1\nindice_reference: {}\nlimites_risque: {}\n"
                 "contraintes: {poids_max_titre: 0.5, poids_max_secteur: 0.3}\n")
    with pytest.raises(ValueError):
        lire_mandat(p)


def test_filtrer_univers_motifs():
    c = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "sector": ["Energy", "IT", "IT", "IT"],
                      "liquidite_usd_jour": [1e9, 1e6, 1e9, 1e9], "capitalisation": [1e12, 1e12, 1e12, 1e9]})
    mandat = {"contraintes": {"liquidite_min_usd_jour": 5e7, "capitalisation_min_usd": 2e10},
              "exclusions": {"secteurs": ["Energy"], "tickers": []}}
    elig, exclus = filtrer_univers(c, mandat)
    assert elig == ["C"]
    motifs = {e["ticker"]: e["motif"] for e in exclus}
    assert "secteur" in motifs["A"] and "liquidité" in motifs["B"] and "capitalisation" in motifs["D"]


def test_contraintes_long_only():
    m = lire_mandat("config/mandat.yaml")
    c = contraintes_mandat(m, ["A", "B"], {"A": "x", "B": "y"})
    assert c["bornes"] == (0.01, 0.10) and c["secteur_max"] == {"x": 0.30, "y": 0.30}


def test_etat_metriques(etat_metriques):
    met = etat_metriques["metriques"]
    assert len(met) == len(etat_metriques["univers"]["eligibles"])
    assert all(-3 <= v["z_valorisation"] <= 3 for v in met.values())
    assert etat_metriques["donnees"]["version"]


def test_point_in_time(etat_metriques):
    assert etat_metriques["donnees"]["prix"]["dates"][-1] <= etat_metriques["date_analyse"]


def test_lectures_agents(etat_metriques):
    met = etat_metriques["metriques"]
    t = next(iter(met))
    m = get_metriques_titre(met, t)
    assert 0 <= m["percentile_roe"] <= 100
    p = comparer_pairs(met, t)
    assert t not in p["pairs"] and p["secteur"] == met[t]["secteur"]
