"""Métriques (étape 3) et lectures pour les agents : calculer_facteurs, get_metriques_titre, comparer_pairs.

H7 : TOUS les chiffres viennent d'ici, jamais d'un LLM.
Convention : VaR / CVaR / drawdown exprimés en PERTES positives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.observabilite import trace

TRADING_DAYS = 252


# ------------------------------------------------------------------ fonctions de base
def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1) if len(r) else float("nan")


def annualized_vol(r: pd.Series) -> float:
    return float(r.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(r: pd.Series, rf: float = 0.0) -> float:
    v = annualized_vol(r)
    return float((annualized_return(r) - rf) / v) if v > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.fillna(0)).cumprod()
    return float((1 - w / w.cummax()).max())


def momentum(p: pd.Series, lookback: int, skip: int = 0) -> float:
    p = p.dropna()
    return float(p.iloc[-1 - skip] / p.iloc[-1 - lookback] - 1) if len(p) > lookback else float("nan")


def beta(r: pd.Series, rm: pd.Series) -> float:
    df = pd.concat([r, rm], axis=1).dropna()
    return float(df.cov().iloc[0, 1] / df.iloc[:, 1].var()) if len(df) > 30 else float("nan")


def zscore(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-clip, clip).fillna(0.0)  # NaN = neutre


def zscore_sectoriel(s: pd.Series, secteurs: pd.Series) -> pd.Series:
    """z-score à l'intérieur de chaque secteur (un PER de banque ≠ un PER de logiciel)."""
    return s.groupby(secteurs).transform(lambda x: zscore(x) if x.notna().sum() >= 3 else pd.Series(0.0, index=x.index))


# ------------------------------------------------------------------ étape 3
@trace("calculer_facteurs")
def calculer_facteurs(close: pd.DataFrame, benchmark: pd.Series, fondamentaux: pd.DataFrame,
                      secteurs: dict[str, str], rf: float) -> pd.DataFrame:
    """Tableau de métriques par titre + scores standardisés (z-scores bornés à ±3)."""
    rets = close.pct_change().dropna(how="all")
    rm = benchmark.pct_change().dropna()
    rows = {}
    for t in close.columns:
        p, r = close[t].dropna(), rets[t].dropna()
        rows[t] = {
            "rendement_1a": momentum(p, TRADING_DAYS),
            "rendement_annualise": annualized_return(r),
            "volatilite": annualized_vol(r),
            "sharpe": sharpe_ratio(r, rf),
            "drawdown_max": max_drawdown(r),
            "momentum_12_1": momentum(p, TRADING_DAYS, 21),
            "momentum_3m": momentum(p, 63),
            "beta": beta(r, rm),
        }
    df = pd.DataFrame(rows).T.join(fondamentaux, how="left")
    df.insert(0, "secteur", pd.Series(secteurs))
    sec = df["secteur"]
    # scores standardisés : + = favorable
    df["z_momentum"] = (zscore(df["momentum_12_1"]) + zscore(df["momentum_3m"])) / 2
    df["z_risque"] = -(zscore(df["volatilite"]) + zscore(df["drawdown_max"])) / 2
    df["z_qualite"] = (zscore(df["roe"]) + zscore(df["marge_nette"]) - zscore(df["dette_fp"])) / 3
    df["z_croissance"] = zscore(df["croissance_ca"])
    df["z_valorisation"] = -(zscore_sectoriel(df["per_prevu"].fillna(df["per"]), sec)
                             + zscore_sectoriel(df["ev_ebitda"], sec)) / 2   # bon marché vs pairs = +
    df["score_quant"] = df[["z_momentum", "z_risque", "z_qualite", "z_croissance", "z_valorisation"]].mean(axis=1)
    return df


def to_records(df: pd.DataFrame) -> dict:
    """DataFrame -> dict JSON-sérialisable (NaN -> None) pour l'état."""
    out = {}
    for t, row in df.iterrows():
        out[t] = {k: (None if isinstance(v, float) and not np.isfinite(v) else (round(v, 6) if isinstance(v, float) else v))
                  for k, v in row.items()}
    return out


# ------------------------------------------------------------------ lectures pour les agents
def get_metriques_titre(metriques: dict, ticker: str) -> dict:
    """Métriques d'un titre + son rang percentile dans l'univers (lecture du state)."""
    m = dict(metriques[ticker])
    df = pd.DataFrame(metriques).T
    for k in ["rendement_1a", "volatilite", "sharpe", "roe", "marge_nette", "croissance_ca", "per", "score_quant"]:
        col = pd.to_numeric(df[k], errors="coerce")
        if pd.notna(m.get(k)):
            m[f"percentile_{k}"] = round(float((col < m[k]).mean() * 100), 0)
    return m


def comparer_pairs(metriques: dict, ticker: str) -> dict:
    """Valorisation du titre face aux pairs de son secteur (médianes et écarts relatifs)."""
    df = pd.DataFrame(metriques).T
    sec = metriques[ticker]["secteur"]
    pairs = df[df["secteur"] == sec]
    cols = ["per", "per_prevu", "ev_ebitda", "price_to_book", "rendement_dividende"]
    tab = pairs[cols].apply(pd.to_numeric, errors="coerce")
    med = tab.median()
    titre = tab.loc[ticker]
    ecarts = {c: (None if pd.isna(titre[c]) or pd.isna(med[c]) or med[c] == 0 else round(float(titre[c] / med[c] - 1), 3))
              for c in cols}
    return {"secteur": sec, "nb_pairs": int(len(pairs) - 1),
            "pairs": [t for t in pairs.index if t != ticker],
            "titre": {c: (None if pd.isna(titre[c]) else round(float(titre[c]), 3)) for c in cols},
            "mediane_secteur": {c: (None if pd.isna(med[c]) else round(float(med[c]), 3)) for c in cols},
            "ecart_relatif_vs_mediane": ecarts}


# ------------------------------------------------------------------ portefeuille
def portfolio_returns(weights: dict[str, float], returns: pd.DataFrame) -> pd.Series:
    w = pd.Series(weights, dtype=float).reindex(returns.columns).fillna(0.0)
    return returns.fillna(0.0) @ w
