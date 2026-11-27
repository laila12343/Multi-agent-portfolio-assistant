"""get_actualites() et recherche_web() — gratuits.

get_actualites : yfinance (Ticker.news) puis Google News RSS (feedparser), filtrées
point-in-time (publiées <= date_analyse) ; titres synthétiques si hors-ligne.
recherche_web  : Tavily si TAVILY_API_KEY (offre gratuite limitée), sinon DuckDuckGo (paquet `ddgs`, sans clé).
"""
from __future__ import annotations

import json
import os
import time
import zlib
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import numpy as np

from src.tools.market_data import _cache_dir, _fresh
from src.observabilite import trace


def _parse_dt(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def get_actualites(ticker: str, settings: dict, date_analyse: str, nom: str | None = None) -> list[dict]:
    """[{title, summary, published (ISO), source, link}], plus récentes d'abord, <= date_analyse."""
    cfg = settings["donnees"]
    n, max_age = cfg.get("actualites_par_titre", 8), cfg.get("actualites_age_max_jours", 30)
    fin = datetime.fromisoformat(date_analyse).replace(tzinfo=timezone.utc) + timedelta(days=1)
    if cfg.get("fournisseur") == "synthetic":
        return synthetic_news(ticker, n, settings["project"]["seed"], fin)

    cache = _cache_dir(settings) / f"news_{ticker}_{date_analyse}.json"
    if _fresh(cache, cfg.get("cache_ttl_heures", 24)):
        return json.loads(cache.read_text())[:n]
    errors: list[str] = []
    items = _from_yfinance(ticker, errors) or _from_google_rss(ticker, nom, errors)
    if not items and errors:
        print(f"[news] {ticker} : sources indisponibles ({errors[0]}) -> titres synthétiques")
        return synthetic_news(ticker, n, settings["project"]["seed"], fin)
    debut = fin - timedelta(days=max_age)
    items = sorted([i for i in items if debut <= _parse_dt(i["published"]) < fin],
                   key=lambda i: i["published"], reverse=True)[:n]
    cache.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    return items


def _from_yfinance(ticker: str, errors: list[str]) -> list[dict]:
    try:
        import yfinance as yf

        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
        return []
    out = []
    for it in raw:
        c = it.get("content", it)           # format 2025+ : champs sous "content"
        if not c.get("title"):
            continue
        if "pubDate" in c:
            published = c["pubDate"]
        elif "providerPublishTime" in c:
            published = datetime.fromtimestamp(c["providerPublishTime"], tz=timezone.utc).isoformat()
        else:
            continue
        prov = c.get("provider") or {}
        url = c.get("canonicalUrl") or {}
        out.append({"title": c["title"], "summary": c.get("summary") or "", "published": published,
                    "source": prov.get("displayName", "") if isinstance(prov, dict) else str(prov),
                    "link": url.get("url", "") if isinstance(url, dict) else c.get("link", "")})
    return out


def _from_google_rss(ticker: str, nom: str | None, errors: list[str]) -> list[dict]:
    try:
        import feedparser
    except ImportError as exc:
        errors.append(f"feedparser: {exc}")
        return []
    q = quote_plus(f'"{nom}" OR {ticker} stock' if nom else f"{ticker} stock")
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    if feed.get("bozo") and not feed.entries:
        errors.append(f"Google News RSS: {feed.get('bozo_exception')}")
    out = []
    for e in feed.entries:
        ts = e.get("published_parsed")
        if not ts:
            continue
        out.append({"title": e.get("title", ""), "summary": "",
                    "published": datetime.fromtimestamp(time.mktime(ts), tz=timezone.utc).isoformat(),
                    "source": (e.get("source") or {}).get("title", "Google News"), "link": e.get("link", "")})
    return out


@trace("recherche_web")
def recherche_web(query: str, settings: dict, max_results: int = 5) -> list[dict]:
    """[{title, url, content}] — Tavily (clé) ou DuckDuckGo (sans clé). Liste vide si indisponible."""
    mode = settings["analyse"].get("recherche_web", "auto")
    if mode == "off" or settings["donnees"].get("fournisseur") == "synthetic":
        return []
    if mode in ("auto", "tavily") and os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            res = TavilySearch(max_results=max_results).invoke({"query": query})
            return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")[:500]}
                    for r in res.get("results", [])]
        except Exception as exc:
            print(f"[news] Tavily indisponible : {exc}")
    if mode in ("auto", "duckduckgo"):
        try:
            from ddgs import DDGS

            return [{"title": r.get("title", ""), "url": r.get("url", r.get("href", "")),
                     "content": r.get("body", "")[:500]} for r in DDGS().news(query, max_results=max_results)]
        except Exception as exc:
            print(f"[news] DuckDuckGo indisponible : {exc}")
    return []


# ------------------------------------------------------------------ hors-ligne
_POS = ["{t} beats earnings expectations, raises full-year guidance",
        "{t} shares rally after strong quarterly revenue growth",
        "Analysts upgrade {t} citing robust demand and margin expansion",
        "{t} announces record buyback and dividend increase"]
_NEG = ["{t} misses revenue estimates as demand weakens",
        "{t} shares fall after regulators open investigation",
        "Analysts downgrade {t} on slowing growth and rising costs",
        "{t} cuts outlook amid supply chain disruptions"]
_NEU = ["{t} to present at upcoming industry conference",
        "{t} announces date for quarterly earnings call",
        "What to watch for {t} this week"]


def synthetic_news(ticker: str, n: int, seed: int, fin: datetime) -> list[dict]:
    r = np.random.default_rng(seed + zlib.crc32(ticker.encode()) % 10_000)
    tilt = r.uniform(-0.5, 0.5)
    out = []
    for _ in range(n):
        u = r.random()
        pool = _POS if u < 0.35 + tilt * 0.4 else (_NEG if u > 0.65 + tilt * 0.4 else _NEU)
        out.append({"title": pool[r.integers(len(pool))].format(t=ticker), "summary": "",
                    "published": (fin - timedelta(days=float(r.uniform(1, 20)))).isoformat(),
                    "source": "synthetic", "link": ""})
    return sorted(out, key=lambda i: i["published"], reverse=True)
