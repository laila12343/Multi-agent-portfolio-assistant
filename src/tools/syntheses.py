"""Tableaux de synthèse calculés depuis l'état (H7 : tous les chiffres viennent du code).

Partagés par le rapport HTML et l'interface Streamlit, pour qu'ils affichent exactement les mêmes chiffres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.nodes.preparation import prix_frame
from src.tools.metrics import portfolio_returns

AGENTS = ["fondamental", "valorisation", "sentiment"]
LIBELLES_RISQUE = {
    "volatilite_annuelle": ("Volatilité annuelle", "volatilite_annuelle_max"),
    "var_historique_1j": ("VaR 95 % sur 1 jour", "var95_1j_max"),
    "cvar_historique_1j": ("CVaR 95 % sur 1 jour", "cvar95_1j_max"),
    "top5": ("Poids des 5 premières lignes", "concentration_top5_max"),
    "rotation": ("Rotation du portefeuille", "rotation_max"),
}
LIBELLES_STRESS = {"krach_marche": "Krach de marché (−20 %, via le bêta)",
                   "choc_tech": "Choc technologie (−30 %)",
                   "pire_mois_historique": "Pire mois de l'historique"}


def _info(state) -> dict:
    return state["univers"]["info"]


def poids_series(state) -> pd.Series:
    return pd.Series(state["poids"], dtype=float).sort_values(ascending=False)


def notes_pivot(state, tickers: list[str] | None = None) -> pd.DataFrame:
    if not state.get("notes"):
        return pd.DataFrame(columns=AGENTS)
    df = pd.DataFrame(state["notes"])
    piv = df.pivot_table(index="ticker", columns="agent", values="score", aggfunc="last").reindex(columns=AGENTS)
    return piv.reindex(tickers) if tickers is not None else piv


def these(state, ticker: str, agent: str) -> str:
    for n in state.get("notes", []):
        if n["ticker"] == ticker and n["agent"] == agent:
            return n["these"]
    return ""


def tableau_allocation(state, poids_actuels: dict | None = None) -> pd.DataFrame:
    w = poids_series(state)
    info, vues = _info(state), state.get("vues", {})
    piv = notes_pivot(state, list(w.index))
    actu = poids_actuels or {}
    rows = []
    for t, p in w.items():
        if t == "CASH":
            rows.append({"ticker": "CASH", "nom": "Liquidités", "secteur": "Liquidités", "poids": p,
                         "actuel": actu.get("CASH", 0.0)})
            continue
        v = vues.get(t, {})
        rows.append({"ticker": t, "nom": info.get(t, {}).get("name", t), "secteur": info.get(t, {}).get("sector", ""),
                     "poids": p, "actuel": actu.get(t, 0.0), "vue": v.get("vue"), "confiance": v.get("confiance"),
                     "justification": v.get("justification", ""),
                     **{a[:4]: (None if pd.isna(piv.loc[t, a]) else int(piv.loc[t, a])) for a in AGENTS if t in piv.index}})
    df = pd.DataFrame(rows)
    df["variation"] = df["poids"] - df["actuel"]
    return df


def exposition_sectorielle(state, max_secteurs: int = 8) -> pd.DataFrame:
    """Poids par secteur : portefeuille vs indice (univers pondéré par la capitalisation)."""
    info, met = _info(state), state["metriques"]
    w = poids_series(state)
    port = w.groupby(lambda t: info.get(t, {}).get("sector", "Liquidités") if t != "CASH" else "Liquidités").sum()
    caps = pd.Series({t: (m.get("capitalisation") or 0) for t, m in met.items()}, dtype=float)
    idx = (caps / caps.sum()).groupby(lambda t: met[t]["secteur"]).sum()
    df = pd.DataFrame({"portefeuille": port, "indice": idx}).fillna(0.0)
    df = df.sort_values("portefeuille", ascending=False)
    if len(df) > max_secteurs:   # au-delà de 8 couleurs : regrouper en « Autres secteurs »
        top, rest = df.iloc[: max_secteurs - 1], df.iloc[max_secteurs - 1:]
        df = pd.concat([top, pd.DataFrame([rest.sum()], index=["Autres secteurs"])])
    df["plafond"] = state["mandat"]["contraintes"]["poids_max_secteur"]
    return df


def performance(state) -> pd.DataFrame:
    """Valeur de 1 investi (dans l'échantillon) : portefeuille proposé vs indice, et drawdowns."""
    px = prix_frame(state["donnees"])
    r = px.pct_change().dropna(how="all")
    r["CASH"] = 0.0
    port = (1 + portfolio_returns(state["poids"], r)).cumprod()
    bench = (1 + r[state["donnees"]["benchmark"]].fillna(0)).cumprod()
    out = pd.DataFrame({"portefeuille": port, "indice": bench})
    out["dd_portefeuille"] = out["portefeuille"] / out["portefeuille"].cummax() - 1
    out["dd_indice"] = out["indice"] / out["indice"].cummax() - 1
    return out


def tableau_risque(state) -> pd.DataFrame:
    rr, lim = state["rapport_risque"], state["mandat"]["limites_risque"]
    rows = []
    for k, (lib, kl) in LIBELLES_RISQUE.items():
        v, l = rr["mesures"][k], lim[kl]
        rows.append({"indicateur": lib, "valeur": v, "limite": l, "utilisation": v / l if l else np.nan,
                     "statut": "respectée" if v <= l + 1e-9 else "dépassée"})
    pire = max(rr["stress"].values())
    rows.append({"indicateur": "Pire scénario de stress", "valeur": pire, "limite": lim["perte_stress_max"],
                 "utilisation": pire / lim["perte_stress_max"], "statut": "respectée" if pire <= lim["perte_stress_max"] + 1e-9 else "dépassée"})
    return pd.DataFrame(rows)


def tableau_stress(state) -> pd.DataFrame:
    return pd.DataFrame([{"scenario": LIBELLES_STRESS.get(k, k), "perte": v} for k, v in state["rapport_risque"]["stress"].items()])


def tableau_bl(state) -> pd.DataFrame:
    """Des vues aux rendements : a priori de marché, vue du comité, a posteriori Black-Litterman."""
    rend = state.get("attendus", {}).get("rendements", {})
    df = pd.DataFrame(rend).T
    if df.empty:
        return df
    df["poids"] = pd.Series(state["poids"])
    df["vue_comite"] = pd.Series({t: v["vue"] for t, v in state.get("vues", {}).items()})
    df["confiance"] = pd.Series({t: v["confiance"] for t, v in state.get("vues", {}).items()})
    return df.sort_values("poids", ascending=False)


def resume_debat(state) -> dict:
    d = state.get("debat", [])
    return {"haussier": [m["texte"] for m in d if m["role"] == "haussier"],
            "baissier": [m["texte"] for m in d if m["role"] == "baissier"], "tours": max([m["tour"] for m in d], default=0)}


def chiffres_cles(state) -> dict:
    a, rr = state["attendus"], state["rapport_risque"]
    w = poids_series(state)
    return {"rendement": a["rendement"], "volatilite": a["volatilite"], "sharpe": a["sharpe"],
            "lignes": int((w.index != "CASH").sum()), "var": rr["mesures"]["var_historique_1j"],
            "rotation": rr["mesures"]["rotation"], "conforme": rr["conforme"], "alerte": rr.get("alerte", False)}
