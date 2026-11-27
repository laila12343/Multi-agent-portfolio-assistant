"""Étape 9 : calculer_mouvements, generer_rapport, journal d'audit (Store)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import resolve_path
from src.observabilite import trace


def lire_portefeuille_actuel(settings: dict) -> dict:
    """{ticker: quantite} (CASH en devise)."""
    p = resolve_path(settings["project"]["portefeuille_actuel_file"])
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {str(t).upper(): float(q) for t, q in zip(df["ticker"], df["quantite"])}


def valoriser(positions: dict, prix: dict) -> tuple[dict, float]:
    """(poids actuels, valeur totale). Les titres sans prix sont ignorés (signalés)."""
    vals = {}
    for t, q in positions.items():
        if t == "CASH":
            vals[t] = q
        elif t in prix:
            vals[t] = q * prix[t]
        else:
            print(f"[reporting] pas de prix pour {t} (hors univers) : ignoré dans la valorisation")
    total = sum(vals.values())
    return ({t: v / total for t, v in vals.items()} if total else {}), total


@trace("calculer_mouvements")
def calculer_mouvements(poids_cible: dict, positions: dict, prix: dict, valeur: float) -> pd.DataFrame:
    """Ordres à passer MANUELLEMENT (H1) : quantités entières, sens, montant."""
    rows = []
    for t in sorted(set(poids_cible) | {k for k in positions if k != "CASH"}):
        if t == "CASH" or t not in prix:
            continue
        cible_q = int(poids_cible.get(t, 0) * valeur // prix[t])
        delta = cible_q - int(positions.get(t, 0))
        if delta:
            rows.append({"ticker": t, "sens": "ACHAT" if delta > 0 else "VENTE", "quantite": abs(delta),
                         "prix_ref": round(prix[t], 2), "montant": round(abs(delta) * prix[t], 2),
                         "poids_cible": poids_cible.get(t, 0.0)})
    return pd.DataFrame(rows, columns=["ticker", "sens", "quantite", "prix_ref", "montant", "poids_cible"])


def table_md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    if df.empty:
        return "_(aucune ligne)_"
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ journal d'audit (H10)
class DictStore:
    """Repli minimal de langgraph BaseStore (put/get/search) pour les tests sans LangGraph."""

    def __init__(self):
        self.data: dict[tuple, dict] = {}

    def put(self, namespace: tuple, key: str, value: dict):
        self.data[(tuple(namespace), key)] = value

    def get(self, namespace: tuple, key: str):
        v = self.data.get((tuple(namespace), key))
        return type("Item", (), {"value": v, "key": key})() if v is not None else None

    def search(self, namespace: tuple, limit: int = 100):
        return [type("Item", (), {"value": v, "key": k})() for (ns, k), v in self.data.items()
                if ns[: len(namespace)] == tuple(namespace)][:limit]


def journaliser(store, date_analyse: str, evenement: str, contenu: dict) -> str:
    """Écrit une entrée d'audit : namespace ("audit", date_analyse), clé horodatée."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    key = f"{ts}_{evenement}"
    store.put(("audit", date_analyse), key, {"evenement": evenement, "horodatage": ts,
                                             **json.loads(json.dumps(contenu, default=str))})
    return key


def lire_journal(store, date_analyse: str) -> list[dict]:
    return sorted((i.value for i in store.search(("audit", date_analyse), limit=1000)), key=lambda v: v["horodatage"])
