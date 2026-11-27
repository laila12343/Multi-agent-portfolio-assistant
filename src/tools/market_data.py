"""Outils de données de marché : get_constituants, filtrer_univers, get_prix, get_fondamentaux.

Fournisseur par défaut : yfinance (gratuit, adapté au PROTOTYPE ; l'API Yahoo est réservée à un
usage personnel). Pour la production : EODHD, FMP, ou Bourse de Casablanca / AMMC (voir docs).
`fournisseur: synthetic` produit des données réalistes hors-ligne (tests, démo sans réseau).

Point-in-time (H8) : les prix sont coupés à date_analyse. Les fondamentaux yfinance (Ticker.info)
ne sont PAS point-in-time : c'est signalé dans donnees["sources"] et dans le rapport.
"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.observabilite import trace

FUNDAMENTAL_FIELDS = {
    "per": "trailingPE", "per_prevu": "forwardPE", "price_to_book": "priceToBook",
    "ev_ebitda": "enterpriseToEbitda", "roe": "returnOnEquity", "marge_nette": "profitMargins",
    "dette_fp": "debtToEquity", "croissance_ca": "revenueGrowth", "rendement_dividende": "dividendYield",
    "capitalisation": "marketCap",
}


def _cache_dir(settings: dict) -> Path:
    d = resolve_path(settings["project"]["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fresh(path: Path, ttl_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def _synthetic(settings: dict) -> bool:
    return settings["donnees"].get("fournisseur") == "synthetic"


def _rng(ticker: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(seed + zlib.crc32(ticker.encode()) % 100_000)


# ------------------------------------------------------------------ étape 1
def get_constituants(settings: dict) -> pd.DataFrame:
    """Composition de l'indice + capitalisation et liquidité (volume moyen en $/jour)."""
    df = pd.read_csv(resolve_path(settings["project"]["univers_file"]))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    tickers = df["ticker"].tolist()
    if _synthetic(settings):
        seed = settings["project"]["seed"]
        df["capitalisation"] = [float(_rng(t, seed).uniform(8e10, 3.5e12)) for t in tickers]
        df["liquidite_usd_jour"] = [float(_rng(t, seed + 1).uniform(3e8, 2e10)) for t in tickers]
        return df
    cache = _cache_dir(settings) / "constituants.csv"
    if _fresh(cache, settings["donnees"]["cache_ttl_heures"]):
        return pd.read_csv(cache)
    try:
        import yfinance as yf

        w = settings["donnees"].get("liquidite_fenetre_jours", 60)
        raw = yf.download(tickers, period="6mo", auto_adjust=True, progress=False)
        adv = (raw["Close"] * raw["Volume"]).tail(w).mean()
        caps = {}
        for t in tickers:
            try:
                caps[t] = float(yf.Ticker(t).fast_info["market_cap"])
            except Exception:
                caps[t] = np.nan
        df["capitalisation"] = df["ticker"].map(caps)
        df["liquidite_usd_jour"] = df["ticker"].map(adv.to_dict())
        df.to_csv(cache, index=False)
        return df
    except Exception as exc:
        if not settings["donnees"].get("repli_synthetique", False):
            raise
        print(f"[market_data] constituants : fournisseur indisponible ({exc}) -> synthétique")
        return get_constituants({**settings, "donnees": {**settings["donnees"], "fournisseur": "synthetic"}})


@trace("filtrer_univers")
def filtrer_univers(constituants: pd.DataFrame, mandat: dict) -> tuple[list[str], list[dict]]:
    """Applique liquidité, capitalisation et exclusions du mandat. Renvoie (éligibles, exclus+motif)."""
    c, ex = mandat["contraintes"], mandat.get("exclusions", {})
    eligibles, exclus = [], []
    for _, r in constituants.iterrows():
        motif = None
        if r["ticker"] in (ex.get("tickers") or []):
            motif = "exclu nommément par le mandat"
        elif r["sector"] in (ex.get("secteurs") or []):
            motif = f"secteur exclu ({r['sector']})"
        elif pd.notna(r.get("liquidite_usd_jour")) and r["liquidite_usd_jour"] < c["liquidite_min_usd_jour"]:
            motif = f"liquidité insuffisante ({r['liquidite_usd_jour'] / 1e6:,.0f} M$/j)"
        elif pd.notna(r.get("capitalisation")) and r["capitalisation"] < c["capitalisation_min_usd"]:
            motif = f"capitalisation insuffisante ({r['capitalisation'] / 1e9:,.0f} Md$)"
        (exclus.append({"ticker": r["ticker"], "motif": motif}) if motif else eligibles.append(r["ticker"]))
    return eligibles, exclus


# ------------------------------------------------------------------ étape 2
def get_prix(tickers: list[str], settings: dict, date_analyse: str,
             secteurs: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cours ajustés, volumes), index = dates <= date_analyse (point-in-time)."""
    years = settings["donnees"].get("historique_annees", 3)
    end = pd.Timestamp(date_analyse)
    if _synthetic(settings):
        return synthetic_prices(tickers, years, settings["project"]["seed"], secteurs, end)
    cache = _cache_dir(settings) / f"prix_{date_analyse}_{years}a.pkl"
    if cache.exists():
        close, vol = pd.read_pickle(cache)
        if set(tickers) <= set(close.columns):
            return close[tickers], vol[tickers]
    import yfinance as yf  # l'exception éventuelle est gérée par RetryPolicy puis par l'appelant

    raw = yf.download(tickers, start=end - pd.DateOffset(years=years), end=end + pd.Timedelta(days=1),
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("le fournisseur a renvoyé un tableau vide")
    close, vol = raw["Close"].ffill(), raw["Volume"].fillna(0)
    close, vol = close.loc[close.index <= end], vol.loc[vol.index <= end]
    pd.to_pickle((close, vol), cache)
    return close, vol


def synthetic_prices(tickers, years=3, seed=42, secteurs=None, end=None):
    """Modèle à facteurs : r = alpha + beta*marché + secteur + idiosyncratique."""
    rng = np.random.default_rng(seed)
    n = int(252 * years)
    dates = pd.bdate_range(end=(end or pd.Timestamp.today()).normalize(), periods=n)
    market = rng.normal(0.0004, 0.010, n)
    sector_f, close, vol = {}, {}, {}
    for t in tickers:
        r_t = _rng(t, seed)
        sec = (secteurs or {}).get(t, "Index" if t in ("SPY", "QQQ") else "Other")
        if sec not in sector_f:
            sector_f[sec] = rng.normal(0, 0.006, n)
        if sec == "Index":
            r = market
        else:
            beta, alpha = r_t.uniform(0.6, 1.5), r_t.uniform(-3e-4, 4e-4)
            r = alpha + beta * market + 0.8 * sector_f[sec] + r_t.normal(0, r_t.uniform(0.007, 0.016), n)
        close[t] = 100 * np.exp(np.cumsum(r))
        vol[t] = r_t.uniform(2e6, 6e7) * np.exp(r_t.normal(0, 0.3, n))
    return pd.DataFrame(close, index=dates), pd.DataFrame(vol, index=dates)


def get_fondamentaux(tickers: list[str], settings: dict) -> tuple[pd.DataFrame, dict]:
    """Ratios par ticker + métadonnées de source (dont le statut point-in-time)."""
    if _synthetic(settings):
        return synthetic_fundamentals(tickers, settings["project"]["seed"]), {"fournisseur": "synthetic", "point_in_time": True}
    cache = _cache_dir(settings) / "fondamentaux.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if not _fresh(cache, settings["donnees"]["cache_ttl_heures"] * 7):
        data = {}
    todo = [t for t in tickers if t not in data]
    if todo:
        import yfinance as yf

        for t in todo:
            try:
                info = yf.Ticker(t).info or {}
                data[t] = {k: info.get(v) for k, v in FUNDAMENTAL_FIELDS.items()}
            except Exception as exc:
                print(f"[market_data] fondamentaux {t} indisponibles : {exc}")
        cache.write_text(json.dumps(data, indent=1))
    df = pd.DataFrame({t: data.get(t, {}) for t in tickers}).T.reindex(columns=list(FUNDAMENTAL_FIELDS))
    df = df.apply(pd.to_numeric, errors="coerce")
    df["dette_fp"] = df["dette_fp"] / 100            # yfinance : 150 = 1,5x
    if df["rendement_dividende"].median() > 0.2:      # certaines versions renvoient des %
        df["rendement_dividende"] = df["rendement_dividende"] / 100
    meta = {"fournisseur": "yfinance (Ticker.info)", "point_in_time": False,
            "avertissement": "ratios courants, non point-in-time : ne pas utiliser pour un backtest (H8)"}
    return df, meta


def synthetic_fundamentals(tickers: list[str], seed: int = 42) -> pd.DataFrame:
    rows = {}
    for t in tickers:
        r = _rng(t, seed + 7)
        rows[t] = {"per": r.uniform(9, 60), "per_prevu": r.uniform(8, 45), "price_to_book": r.uniform(1, 40),
                   "ev_ebitda": r.uniform(6, 40), "roe": r.uniform(-0.05, 0.6), "marge_nette": r.uniform(-0.05, 0.45),
                   "dette_fp": r.uniform(0, 3), "croissance_ca": r.uniform(-0.1, 0.4),
                   "rendement_dividende": r.uniform(0, 0.04), "capitalisation": r.uniform(1e11, 3e12)}
    return pd.DataFrame(rows).T.astype(float)
