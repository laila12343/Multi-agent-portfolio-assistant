"""Étapes 6-7 : Black-Litterman, optimisation sous contraintes, contrôle des risques, boucle H9."""
import numpy as np
import pandas as pd

from src.nodes import portefeuille
from src.tools.optimisation import black_litterman, optimiser_poids
from src.tools.risque import calculer_var, rotation, verifier_conformite


def _cov():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    return pd.DataFrame(a @ a.T / 20 + np.eye(4) * 0.02, index=list("ABCD"), columns=list("ABCD"))


def test_black_litterman_suit_les_vues():
    cov = _cov()
    cfg = {"delta": 2.5, "tau": 0.05, "kappa": 0.03, "confiance": {"faible": .25, "moyenne": .5, "elevee": .75}}
    vues = {"A": {"vue": 2, "confiance": "elevee"}, "B": {"vue": -2, "confiance": "elevee"},
            "C": {"vue": 0, "confiance": "faible"}, "D": {"vue": 0, "confiance": "faible"}}
    bl = black_litterman(cov, pd.Series({"A": 1, "B": 1, "C": 1, "D": 1.0}), vues, cfg, 0.04)
    assert bl["posterior"]["A"] > bl["prior"]["A"] and bl["posterior"]["B"] < bl["prior"]["B"]


def test_optimisation_respecte_les_contraintes():
    cov = _cov()
    mu = pd.Series([0.12, 0.10, 0.08, 0.15], index=cov.index)
    c = {"bornes": (0.05, 0.4), "secteurs": {"A": "x", "B": "x", "C": "y", "D": "z"},
         "secteur_max": {"x": 0.5, "y": 0.5, "z": 0.5}, "secteur_min": {"x": 0, "y": 0, "z": 0}}
    w = pd.Series(optimiser_poids(mu, cov, c, 0.04)["poids"])
    assert abs(w.sum() - 1) < 1e-3 and w.max() <= 0.4 + 1e-3 and w.min() >= 0.05 - 1e-3
    assert w[["A", "B"]].sum() <= 0.5 + 1e-3


def test_conformite_detecte_les_ecarts(etat_selection):
    m = etat_selection["mandat"]
    sect = {t: v["secteur"] for t, v in etat_selection["metriques"].items()}
    out = verifier_conformite({"AAPL": 0.6, "XXX": 0.4}, m, sect, list(sect))
    assert any("par titre" in x for x in out) and any("hors univers" in x for x in out) and any("lignes" in x for x in out)


def test_var_et_rotation():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 2000))
    v = calculer_var(r)
    assert v["cvar_historique_1j"] > v["var_historique_1j"] > 0
    assert rotation({"A": 1.0}, {"B": 1.0}) == 1.0 and rotation({"A": .5, "B": .5}, {"A": .5, "B": .5}) == 0


def test_boucle_de_risque_et_escalade(etat_selection, ctx):
    """Limites impossibles : 3 essais puis escalade au gérant avec alerte (H9)."""
    st = dict(etat_selection)
    st["mandat"] = {**st["mandat"], "limites_risque": {**st["mandat"]["limites_risque"], "var95_1j_max": 0.001}}
    st["iterations_risque"] = 0
    etape, essais = "optimiser", 0
    while etape == "optimiser":
        st.update(portefeuille.optimiser(st, ctx))
        st.update(portefeuille.controle_risque(st, ctx))
        etape = portefeuille.router_risque(st, ctx)
        essais += 1
    assert essais == ctx.settings["risque"]["max_tentatives"] and etape == "validation_gerant"
    assert st["rapport_risque"]["alerte"] is True


def test_premier_essai_respecte_le_mandat(etat_selection, ctx):
    st = {**etat_selection, "iterations_risque": 0}
    st.update(portefeuille.optimiser(st, ctx))
    sect = {t: v["secteur"] for t, v in st["metriques"].items()}
    assert verifier_conformite(st["poids"], st["mandat"], sect, st["univers"]["eligibles"]) == []
