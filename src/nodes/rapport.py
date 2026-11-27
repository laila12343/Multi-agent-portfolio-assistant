"""Étape 9 — rapport : calculer_mouvements (code), generer_rapport (LLM + tableaux du code),
journal d'audit dans le Store (H10 : thèse, sources, version des données)."""
from __future__ import annotations

import json

import pandas as pd

from src.agents import prompts
from src.llm import text_call
from src.nodes.portefeuille import contexte_risque
from src.state import EtatPortefeuille
from src.config import resolve_path
from src.tools.rapport_html import generer_rapport_html
from src.tools.reporting import calculer_mouvements, journaliser, lire_journal, lire_portefeuille_actuel, table_md, valoriser


def rapport(state: EtatPortefeuille, ctx) -> dict:
    poids, mandat, met = state["poids"], state["mandat"], state["metriques"]
    poids_actuels, _, prix = contexte_risque(state, ctx)
    positions = lire_portefeuille_actuel(ctx.settings)
    _, valeur_actuelle = valoriser(positions, prix)
    valeur = valeur_actuelle or mandat["valeur_portefeuille"]
    mvts = calculer_mouvements(poids, positions, prix, valeur)
    rr, vues = state["rapport_risque"], state["vues"]

    alloc = pd.DataFrame([{"ticker": t, "secteur": met.get(t, {}).get("secteur", ""), "poids": w,
                           "actuel": poids_actuels.get(t, 0.0), "vue": vues.get(t, {}).get("vue", 0),
                           "confiance": vues.get(t, {}).get("confiance", "—")} for t, w in poids.items()])
    m = rr["mesures"]
    risque = pd.DataFrame([
        ("Volatilité annuelle", m["volatilite_annuelle"], mandat["limites_risque"]["volatilite_annuelle_max"]),
        ("VaR 95 % 1 jour", m["var_historique_1j"], mandat["limites_risque"]["var95_1j_max"]),
        ("CVaR 95 % 1 jour", m["cvar_historique_1j"], mandat["limites_risque"]["cvar95_1j_max"]),
        ("Concentration top 5", m["top5"], mandat["limites_risque"]["concentration_top5_max"]),
        ("Rotation", m["rotation"], mandat["limites_risque"]["rotation_max"]),
        ("Pire stress test", max(rr["stress"].values()), mandat["limites_risque"]["perte_stress_max"]),
    ], columns=["indicateur", "portefeuille", "limite"])

    narration = ""
    if ctx.llm_fort is not None:
        faits = {"vues": vues, "poids": poids, "synthese_debat": state.get("debat", [])[-1:] , "risque": rr["explication"],
                 "decision_gerant": state.get("decision_gerant")}
        narration = text_call(ctx.llm_fort, prompts.RAPPORT, json.dumps(faits, ensure_ascii=False, default=str))
    if not narration:
        top = sorted(poids, key=poids.get, reverse=True)[:3]
        narration = ("Allocation construite par Black-Litterman à partir des vues du comité, sous les contraintes du mandat. "
                     f"Principales positions : {', '.join(f'{t} ({vues.get(t, {}).get('justification', '')})' for t in top)}. "
                     + rr["explication"])

    a = state["attendus"]
    src = state["donnees"]["sources"]
    texte = "\n\n".join([
        f"# Note d'investissement — {mandat['nom']}",
        f"Date d'analyse : **{state['date_analyse']}** · mandat {mandat['version']} (empreinte {mandat['empreinte']}) · "
        f"données version {state['donnees']['version']} · décision du gérant : **{state['decision_gerant']['decision']}**",
        "## Synthèse", narration,
        f"## Allocation cible ({len(poids)} lignes) — attendu : rendement {a['rendement']:.1%}, volatilité {a['volatilite']:.1%}, Sharpe {a['sharpe']:.2f}",
        table_md(alloc, {"poids": "{:.1%}", "actuel": "{:.1%}", "vue": "{:+d}"}),
        "## Risque", table_md(risque, {"portefeuille": "{:.2%}", "limite": "{:.2%}"}),
        f"## Ordres à passer manuellement (valeur du portefeuille {valeur:,.0f} {mandat['devise']})",
        table_md(mvts, {"quantite": "{:,}", "prix_ref": "{:,.2f}", "montant": "{:,.0f}", "poids_cible": "{:.1%}"}),
        "## Sources", f"Prix : {src['prix']} · Fondamentaux : {src['fondamentaux'].get('fournisseur')}"
        + ("" if src["fondamentaux"].get("point_in_time", True) else f" (⚠ {src['fondamentaux']['avertissement']})")
        + f" · Actualités : {src['actualites']}",
        "_Aide à la décision : aucun ordre n'est transmis automatiquement (H1)._",
    ])
    journaliser(ctx.store, state["date_analyse"], "rapport", {
        "mandat": {"version": mandat["version"], "empreinte": mandat["empreinte"]},
        "donnees": {"version": state["donnees"]["version"], "sources": src},
        "selection": state["selection"], "vues": vues, "poids": poids, "attendus": {k: a[k] for k in ("rendement", "volatilite", "sharpe")},
        "risque": {"conforme": rr["conforme"], "alerte": rr.get("alerte"), "mesures": m},
        "decision_gerant": state["decision_gerant"], "ordres": mvts.to_dict(orient="records"),
    })
    ordres = mvts.to_dict(orient="records")
    html_doc = generer_rapport_html({**state, "ordres": ordres}, poids_actuels, narration,
                                    journal=lire_journal(ctx.store, state["date_analyse"]))
    dossier = resolve_path(ctx.settings.get("rapport", {}).get("dossier", "data/rapports_generes"))
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = dossier / f"note_investissement_{state['date_analyse']}.html"
    fichier.write_text(html_doc, encoding="utf-8")
    return {"rapport_final": texte, "ordres": ordres, "rapport_fichier": str(fichier),
            "journal": [f"[rapport] {len(mvts)} ordres, rotation {m['rotation']:.0%} ; note HTML : {fichier.name} ; journal d'audit écrit"]}
