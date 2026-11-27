"""Aperçu du marché pour la page d'accueil : indice + indices sectoriels équipondérés (base 100, un an)."""
from __future__ import annotations

import pandas as pd

from src.tools import market_data as md
from src.tools.rapport_html import secteur_fr


def apercu_marche(settings: dict, date_analyse: str, n_secteurs: int = 4) -> pd.DataFrame:
    univers = md.get_constituants(settings)
    secteurs = dict(zip(univers["ticker"], univers["sector"]))
    bench = "SPY"
    close, _ = md.get_prix(list(secteurs) + [bench], settings, date_analyse, secteurs)
    close = close.ffill().dropna(axis=1, how="all").tail(252)
    base = close / close.iloc[0] * 100
    principaux = univers.groupby("sector").size().sort_values(ascending=False).head(n_secteurs).index
    out = {"Indice (SPY)": base[bench]} if bench in base else {}
    for s in principaux:
        cols = [t for t, sec in secteurs.items() if sec == s and t in base]
        if cols:
            out[secteur_fr(s)] = base[cols].mean(axis=1)
    return pd.DataFrame(out)
