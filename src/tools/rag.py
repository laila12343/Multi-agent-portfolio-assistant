"""rechercher_rapport() : RAG sur les rapports annuels (data/rapports/<TICKER>/).

Base vectorielle open source et légère :
  tfidf  scikit-learn (défaut, aucun téléchargement)
  hf     sentence-transformers (all-MiniLM-L6-v2), meilleure recherche sémantique
Point-in-time (H8) : un fichier nommé « …_AAAA-MM-JJ.* » n'est utilisé que si sa date <= date_analyse.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import resolve_path
from src.observabilite import trace

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        except Exception as exc:
            print(f"[rag] PDF illisible {path.name} : {exc}")
            return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def _chunks(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [text[i: i + size] for i in range(0, max(len(text) - overlap, 1), size - overlap)]


@lru_cache(maxsize=64)
def _index(folder: str, date_analyse: str, moteur: str, modele: str):
    docs = []
    for f in sorted(Path(folder).glob("*")):
        if f.suffix.lower() not in {".txt", ".md", ".html", ".htm", ".pdf"}:
            continue
        m = DATE_RE.search(f.name)
        if m and m.group(1) > date_analyse:      # publié après la date d'analyse : interdit
            continue
        docs += [(f.name, c) for c in _chunks(_read(f)) if len(c) > 100]
    if not docs:
        return None
    texts = [c for _, c in docs]
    if moteur == "hf":
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(modele)
        return docs, ("hf", enc), enc.encode(texts, normalize_embeddings=True)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    return docs, ("tfidf", vec), vec.fit_transform(texts)


@trace("rechercher_rapport")
def rechercher_rapport(ticker: str, question: str, settings: dict, date_analyse: str, k: int = 4) -> list[dict]:
    """Les k extraits de rapports les plus pertinents : [{"source", "extrait"}]."""
    folder = resolve_path(settings["project"]["rapports_dir"]) / ticker
    if not folder.exists():
        return []
    a = settings["analyse"]
    idx = _index(str(folder), date_analyse, a.get("rag_moteur", "tfidf"), a.get("rag_modele", ""))
    if idx is None:
        return []
    docs, (kind, enc), mat = idx
    if kind == "hf":
        sims = mat @ enc.encode([question], normalize_embeddings=True)[0]
    else:
        sims = (mat @ enc.transform([question]).T).toarray().ravel()
    best = np.argsort(-sims)[:k]
    return [{"source": docs[i][0], "extrait": docs[i][1][:700]} for i in best if sims[i] > 0]


def telecharger_rapport_sec(ticker: str, email: str, settings: dict | None = None, n: int = 1) -> list[Path]:
    """Télécharge les n derniers 10-K (SEC EDGAR, gratuit) dans data/rapports/<TICKER>/10-K_<date>.txt."""
    import requests
    from bs4 import BeautifulSoup

    headers = {"User-Agent": f"portfolio-agents projet-ecole {email}"}
    base = resolve_path((settings or {}).get("project", {}).get("rapports_dir", "data/rapports"))
    tick = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30).json()
    cik = next((v["cik_str"] for v in tick.values() if v["ticker"].upper() == ticker.replace("-", ".").upper()), None)
    if cik is None:
        raise ValueError(f"{ticker} introuvable sur EDGAR")
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=headers, timeout=30).json()
    rec = sub["filings"]["recent"]
    out = []
    for form, acc, doc, date in zip(rec["form"], rec["accessionNumber"], rec["primaryDocument"], rec["filingDate"]):
        if form != "10-K":
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        html = requests.get(url, headers=headers, timeout=60).text
        dest = base / ticker / f"10-K_{date}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(BeautifulSoup(html, "html.parser").get_text(" "), encoding="utf-8")
        out.append(dest)
        if len(out) >= n:
            break
    _index.cache_clear()
    return out
