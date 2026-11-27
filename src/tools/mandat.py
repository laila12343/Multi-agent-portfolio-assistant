"""lire_mandat() et contraintes_mandat() — le mandat est la source de vérité des contraintes (H5)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from src.config import resolve_path
from src.observabilite import trace

REQUIRED = ["version", "valeur_portefeuille", "indice_reference", "contraintes", "limites_risque"]


@trace("lire_mandat")
def lire_mandat(path: str | Path) -> dict:
    """Lit le mandat YAML (ou JSON), vérifie les champs obligatoires, ajoute son empreinte."""
    p = resolve_path(path)
    raw = p.read_text(encoding="utf-8")
    mandat = yaml.safe_load(raw)  # YAML est un sur-ensemble de JSON
    missing = [k for k in REQUIRED if k not in mandat]
    if missing:
        raise ValueError(f"mandat incomplet, champs manquants : {missing}")
    c = mandat["contraintes"]
    if not (0 < c["poids_max_titre"] <= c["poids_max_secteur"] <= 1):
        raise ValueError("contraintes incohérentes : 0 < poids_max_titre <= poids_max_secteur <= 1")
    mandat.setdefault("exclusions", {"secteurs": [], "tickers": []})
    mandat["empreinte"] = hashlib.sha256(raw.encode()).hexdigest()[:12]  # traçabilité (H10)
    return mandat


def contraintes_mandat(mandat: dict, tickers: list[str], secteurs: dict[str, str]) -> dict:
    """Traduit le mandat en contraintes d'optimisation."""
    c = mandat["contraintes"]
    lower = -c["poids_max_titre"] if not c.get("long_only", True) else c.get("poids_min_ligne", 0.0)
    return {
        "bornes": (lower, c["poids_max_titre"]),
        "secteurs": {t: secteurs[t] for t in tickers},
        "secteur_max": {s: c["poids_max_secteur"] for s in set(secteurs[t] for t in tickers)},
        "secteur_min": {s: 0.0 for s in set(secteurs[t] for t in tickers)},
    }
