"""Étape 5 — sous-graphe debat (2 tours max) puis juge_selection.

    chercheur_haussier <-> chercheur_baissier   (aucun outil : ils lisent les notes)
              └──── après N tours ────> juge_selection  (sortie structurée Selection)

Garde-fous en code après le juge : tickers valides uniquement, taille 15-30,
au plus `max_par_secteur` titres par secteur, liste faisable pour le mandat.
"""
from __future__ import annotations

import json
import math
import operator
from typing import Annotated, TypedDict

import pandas as pd

from src.agents import prompts
from src.llm import structured_call, text_call
from src.state import Selection, VueTitre

MAX_PAR_SECTEUR = 6


# ------------------------------------------------------------------ agrégation des notes (code)
def agreger_notes(notes: list[dict], poids: dict, secteurs: dict) -> pd.DataFrame:
    df = pd.DataFrame(notes)
    piv = df.pivot_table(index="ticker", columns="agent", values="score", aggfunc="last").reindex(
        columns=["fondamental", "sentiment", "valorisation"]).fillna(0)
    piv["moyenne"] = sum(piv[a] * w for a, w in poids.items()) / sum(poids.values())
    piv["dispersion"] = piv[["fondamental", "sentiment", "valorisation"]].max(axis=1) - piv[["fondamental", "sentiment", "valorisation"]].min(axis=1)
    piv["secteur"] = pd.Series(secteurs).reindex(piv.index)
    theses = df.groupby("ticker").apply(lambda g: " | ".join(f"{a[:4]}: {t}" for a, t in zip(g["agent"], g["these"])),
                                        include_groups=False)
    piv["theses"] = theses
    return piv.sort_values("moyenne", ascending=False)


def digest(agg: pd.DataFrame, max_chars: int = 220) -> str:
    """Résumé compact des notes, lisible par un petit LLM (1 ligne par titre)."""
    lines = ["ticker | secteur | fond | sent | valo | moyenne | thèses"]
    for t, r in agg.iterrows():
        lines.append(f"{t} | {r['secteur']} | {int(r['fondamental']):+d} | {int(r['sentiment']):+d} | "
                     f"{int(r['valorisation']):+d} | {r['moyenne']:+.2f} | {str(r['theses'])[:max_chars]}")
    return "\n".join(lines)


def _confiance(dispersion: float) -> str:
    return "elevee" if dispersion <= 1 else "moyenne" if dispersion <= 2 else "faible"


def valider_selection(sel: Selection | None, agg: pd.DataFrame, cfg: dict, mandat: dict) -> list[dict]:
    """Le code garantit une liste exploitable, quoi qu'ait répondu le juge."""
    tmin, tmax = cfg["taille_min"], cfg["taille_max"]
    c = mandat["contraintes"]
    vues, compte = [], {}

    def ajout(v: VueTitre) -> bool:
        sec = agg.loc[v.ticker, "secteur"]
        if v.ticker in {x.ticker for x in vues} or compte.get(sec, 0) >= MAX_PAR_SECTEUR or len(vues) >= tmax:
            return False
        vues.append(v)
        compte[sec] = compte.get(sec, 0) + 1
        return True

    for v in (sel.titres if sel else []):
        if v.ticker in agg.index:
            ajout(v)

    def capacite() -> float:
        return sum(min(c["poids_max_secteur"], n * c["poids_max_titre"]) for n in compte.values())

    # compléter (taille minimale, faisabilité) avec les meilleurs titres restants
    for t, r in agg.iterrows():
        if len(vues) >= tmin and capacite() >= 1.0:
            break
        ajout(VueTitre(ticker=t, vue=max(-2, min(2, round(r["moyenne"]))), confiance=_confiance(r["dispersion"]),
                       justification="ajouté par le code (taille minimale ou faisabilité du mandat)"))
    return [v.model_dump() for v in vues]


def selection_regles(agg: pd.DataFrame, cfg: dict) -> Selection:
    n = min(max((agg["moyenne"] > 0).sum(), cfg["taille_min"]), cfg["taille_max"])
    titres = [VueTitre(ticker=t, vue=max(-2, min(2, round(r["moyenne"] * 1.5))), confiance=_confiance(r["dispersion"]),
                       justification=f"notes F/S/V = {int(r['fondamental']):+d}/{int(r['sentiment']):+d}/{int(r['valorisation']):+d}")
              for t, r in agg.head(n).iterrows()]
    return Selection(titres=titres, synthese="Sélection par règles (mode sans LLM) : meilleures moyennes pondérées des notes.")


# ------------------------------------------------------------------ sous-graphe
class DebatState(TypedDict, total=False):
    digest: str
    candidats: list[str]
    transcript: Annotated[list[dict], operator.add]
    tour: int
    selection: dict


def _historique(s: DebatState) -> str:
    return "\n\n".join(f"[{m['role']} — tour {m['tour']}]\n{m['texte']}" for m in s.get("transcript", []))


def chercheur(role: str, s: DebatState, ctx, agg: pd.DataFrame) -> dict:
    tour = s.get("tour", 0) + 1
    if ctx.llm_fort is not None:
        texte = text_call(ctx.llm_fort, prompts.HAUSSIER if role == "haussier" else prompts.BAISSIER,
                          f"Notes des analystes :\n{s['digest']}\n\nDébat jusqu'ici :\n{_historique(s) or '(début)'}")
    else:
        texte = ""
    if not texte:
        if role == "haussier":
            texte = "\n".join(f"- {t} : notes {int(r['fondamental']):+d}/{int(r['sentiment']):+d}/{int(r['valorisation']):+d}, "
                              f"{str(r['theses'])[:90]}" for t, r in agg.head(10).iterrows())
        else:
            divis = agg.sort_values("dispersion", ascending=False).head(5)
            texte = "\n".join(f"- {t} : avis divergents (écart {int(r['dispersion'])}), {str(r['theses'])[:90]}"
                              for t, r in divis.iterrows())
    out = {"transcript": [{"role": role, "tour": tour, "texte": texte}]}
    if role == "baissier":          # un tour = haussier puis baissier
        out["tour"] = tour
    return out


def juge(s: DebatState, ctx, agg: pd.DataFrame, mandat: dict) -> dict:
    cfg = ctx.settings["debat"]
    sel = None
    if ctx.llm_fort is not None:
        sys = prompts.JUGE.format(taille_min=cfg["taille_min"], taille_max=cfg["taille_max"], max_par_secteur=MAX_PAR_SECTEUR)
        sel = structured_call(ctx.llm_fort, Selection, sys,
                              f"Candidats : {', '.join(s['candidats'])}\n\nNotes :\n{s['digest']}\n\nDébat :\n{_historique(s)}")
    if sel is None:
        sel = selection_regles(agg, cfg)
    titres = valider_selection(sel, agg, cfg, mandat)
    return {"selection": {"titres": titres, "synthese": sel.synthese}}


def build_debat(ctx, agg: pd.DataFrame, mandat: dict):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(DebatState)
    g.add_node("chercheur_haussier", lambda s: chercheur("haussier", s, ctx, agg))
    g.add_node("chercheur_baissier", lambda s: chercheur("baissier", s, ctx, agg))
    g.add_node("juge_selection", lambda s: juge(s, ctx, agg, mandat))
    g.add_edge(START, "chercheur_haussier")
    g.add_edge("chercheur_haussier", "chercheur_baissier")
    g.add_conditional_edges("chercheur_baissier",
                            lambda s: "chercheur_haussier" if s["tour"] < ctx.settings["debat"]["tours"] else "juge_selection",
                            ["chercheur_haussier", "juge_selection"])
    g.add_edge("juge_selection", END)
    return g.compile()


def executer_debat(state, ctx, avec_graphe: bool = True) -> dict:
    """Nœud « debat » du graphe principal : prépare les notes, lance le sous-graphe, applique les garde-fous."""
    from src.tools.reporting import journaliser

    secteurs = {t: m["secteur"] for t, m in state["metriques"].items()}
    agg = agreger_notes(state["notes"], ctx.settings["analyse"]["poids_notes"], secteurs)
    init = {"digest": digest(agg), "candidats": list(agg.index), "transcript": [], "tour": 0}
    if avec_graphe:
        out = build_debat(ctx, agg, state["mandat"]).invoke(init)
    else:  # équivalent séquentiel (tests sans LangGraph)
        out = dict(init)
        while True:
            for role in ("haussier", "baissier"):
                upd = chercheur(role, out, ctx, agg)
                out["transcript"] = out["transcript"] + upd["transcript"]
                out["tour"] = upd.get("tour", out["tour"])
            if out["tour"] >= ctx.settings["debat"]["tours"]:
                break
        out.update(juge(out, ctx, agg, state["mandat"]))
    titres = out["selection"]["titres"]
    vues = {v["ticker"]: {k: v[k] for k in ("vue", "confiance", "justification")} for v in titres}
    journaliser(ctx.store, state["date_analyse"], "selection",
                {"selection": list(vues), "vues": vues, "synthese": out["selection"]["synthese"]})
    return {"debat": out["transcript"], "selection": list(vues), "vues": vues,
            "synthese_debat": out["selection"]["synthese"],
            "journal": [f"[debat] {len(out['transcript'])} interventions, {len(vues)} titres retenus : "
                        + ", ".join(f"{t}({v['vue']:+d},{v['confiance']})" for t, v in list(vues.items())[:8]) + "…"]}
