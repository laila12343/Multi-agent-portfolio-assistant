"""Nœuds déterministes des étapes 1 à 3 + routage Send vers analyse_titre.

    charger_mandat -> construire_univers -> collecter_donnees (RetryPolicy) -> calculer_metriques
    -> Send("analyse_titre", payload) x N titres
"""
from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd

from src.state import EtatPortefeuille
from src.tools import market_data as md
from src.tools.mandat import lire_mandat
from src.tools.metrics import calculer_facteurs, comparer_pairs, get_metriques_titre, to_records
from src.tools.news import get_actualites


def derniere_seance(d: date | None = None) -> str:
    return (pd.Timestamp(d or date.today()) - pd.offsets.BDay(0)).date().isoformat()


def prix_frame(donnees: dict) -> pd.DataFrame:
    p = donnees["prix"]
    return pd.DataFrame(p["data"], index=pd.to_datetime(p["dates"]))


# ------------------------------------------------------------------ nœuds
def charger_mandat(state: EtatPortefeuille, ctx) -> dict:
    mandat = lire_mandat(ctx.settings["project"]["mandat_file"])
    d = state.get("date_analyse") or derniere_seance()
    return {"mandat": mandat, "date_analyse": d, "iterations_risque": 0,
            "journal": [f"[charger_mandat] mandat {mandat['version']} (empreinte {mandat['empreinte']}), date {d}"]}


def construire_univers(state: EtatPortefeuille, ctx) -> dict:
    constituants = md.get_constituants(ctx.settings)
    eligibles, exclus = md.filtrer_univers(constituants, state["mandat"])
    info = constituants.set_index("ticker")[["name", "sector", "industry"]].to_dict(orient="index")
    return {"univers": {"eligibles": eligibles, "exclus": exclus, "info": info},
            "journal": [f"[construire_univers] {len(eligibles)} éligibles, {len(exclus)} exclus"
                        + (f" ({', '.join(e['ticker'] + ' : ' + e['motif'] for e in exclus[:5])})" if exclus else "")]}


def collecter_donnees(state: EtatPortefeuille, ctx) -> dict:
    """Lève une exception si le fournisseur échoue : RetryPolicy relance le nœud (pannes réseau)."""
    s, d = ctx.settings, state["date_analyse"]
    uni = state["univers"]
    tickers = uni["eligibles"]
    bench = state["mandat"]["indice_reference"]["benchmark"]
    secteurs = {t: uni["info"][t]["sector"] for t in tickers}
    try:
        close, _vol = md.get_prix(tickers + [bench], s, d, secteurs)
        fond, meta = md.get_fondamentaux(tickers, s)
    except Exception:
        if not s["donnees"].get("repli_synthetique"):
            raise
        print("[collecter_donnees] fournisseur indisponible -> données synthétiques (repli)")
        s = {**s, "donnees": {**s["donnees"], "fournisseur": "synthetic"}}
        close, _vol = md.get_prix(tickers + [bench], s, d, secteurs)
        fond, meta = md.get_fondamentaux(tickers, s)
    close = close.dropna(axis=1, thresh=int(0.8 * len(close))).ffill().bfill()
    tickers = [t for t in tickers if t in close.columns]
    actus = {t: get_actualites(t, s, d, uni["info"][t]["name"]) for t in tickers}
    version = hashlib.sha256(pd.util.hash_pandas_object(close.round(4)).values.tobytes()).hexdigest()[:12]
    donnees = {
        "prix": {"dates": [x.date().isoformat() for x in close.index],
                 "data": {c: close[c].round(4).tolist() for c in close.columns}},
        "fondamentaux": to_records(fond.loc[tickers]),
        "actualites": actus,
        "benchmark": bench,
        "version": version,
        "sources": {"prix": s["donnees"]["fournisseur"], "fondamentaux": meta,
                    "actualites": "yfinance / Google News RSS" if s["donnees"]["fournisseur"] != "synthetic" else "synthetic"},
        "recupere_le": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    warn = "" if meta.get("point_in_time", True) else " — fondamentaux NON point-in-time"
    return {"donnees": donnees,
            "journal": [f"[collecter_donnees] {len(tickers)} titres, {len(close)} séances jusqu'au {close.index[-1].date()}, "
                        f"{sum(len(v) for v in actus.values())} actualités, version {version}{warn}"]}


def calculer_metriques(state: EtatPortefeuille, ctx) -> dict:
    don = state["donnees"]
    close = prix_frame(don)
    bench = don["benchmark"]
    tickers = [t for t in state["univers"]["eligibles"] if t in close.columns]
    fond = pd.DataFrame(don["fondamentaux"]).T.apply(pd.to_numeric, errors="coerce")
    secteurs = {t: state["univers"]["info"][t]["sector"] for t in tickers}
    df = calculer_facteurs(close[tickers], close[bench], fond, secteurs, ctx.settings.get("risque", {}).get("taux_sans_risque", 0.04))
    return {"metriques": to_records(df),
            "journal": [f"[calculer_metriques] {len(df.columns)} indicateurs x {len(df)} titres"]}


def router_analyses(state: EtatPortefeuille, ctx) -> list:
    """Arête conditionnelle : un Send par titre vers le sous-graphe analyse_titre."""
    from langgraph.types import Send

    return [Send("analyse_titre", p) for p in payloads_analyse(state, ctx)]


def payloads_analyse(state: EtatPortefeuille, ctx) -> list[dict]:
    """Contenu de chaque Send : uniquement ce dont les 3 analystes ont besoin pour CE titre."""
    met = state["metriques"]
    n = ctx.settings["analyse"].get("max_titres", len(met))
    ordre = sorted(met, key=lambda t: -(met[t].get("score_quant") or 0))[:n]
    info = state["univers"]["info"]
    return [{"ticker": t, "nom": info[t]["name"], "secteur": info[t]["sector"], "date_analyse": state["date_analyse"],
             "metriques_titre": get_metriques_titre(met, t), "pairs": comparer_pairs(met, t),
             "actualites": state["donnees"]["actualites"].get(t, []), "notes": []} for t in ordre]
