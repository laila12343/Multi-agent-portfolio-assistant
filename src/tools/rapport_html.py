"""generer_rapport_html() : la note d'investissement mensuelle, en un fichier HTML autonome.

Forme : note de recherche de banque privée. Spectral (titres) + Public Sans (texte, chiffres tabulaires),
encre bleu-pétrole, un seul accent laiton réservé à la décision du comité. Graphiques en SVG intégrés
(nets à l'impression), mise en page A4 : « Imprimer → Enregistrer en PDF » depuis le navigateur.

Contenu : tous les chiffres viennent de src/tools/syntheses.py (code) ; le LLM n'écrit que la narration (H7).
"""
from __future__ import annotations

import html
import io
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # polices web absentes : rendu CSS
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment  # noqa: E402

from src.observabilite import trace  # noqa: E402
from src.tools import syntheses as sy  # noqa: E402

# ------------------------------------------------------------------ identité visuelle
ENCRE, PAPIER, LAITON, ARDOISE, FILET = "#0E3B43", "#FAFAF8", "#B0802A", "#5B6670", "#DADDD8"
SERIE_1, SERIE_2, NEUTRE = "#2a78d6", "#eb6834", "#B9BDB8"
# couleur fixe par secteur (la couleur suit le secteur, jamais son rang)
COULEURS_SECTEUR = {
    "Information Technology": "#2a78d6", "Health Care": "#1baf7a", "Financials": "#6250d6",
    "Consumer Staples": "#eda100", "Communication Services": "#e87ba4", "Consumer Discretionary": "#eb6834",
    "Industrials": "#008300", "Energy": "#e34948", "Liquidités": "#E4E6E1",
}
SECTEURS_FR = {
    "Information Technology": "Technologie", "Health Care": "Santé", "Financials": "Finance",
    "Consumer Staples": "Consommation de base", "Communication Services": "Communication",
    "Consumer Discretionary": "Consommation discrétionnaire", "Industrials": "Industrie", "Energy": "Énergie",
    "Materials": "Matériaux", "Utilities": "Services publics", "Real Estate": "Immobilier",
    "Liquidités": "Liquidités", "Autres secteurs": "Autres secteurs",
}
# diverging bleu <-> rouge, milieu gris (notes -2..+2)
NOTE_COULEUR = {-2: ("#c7403f", "#fff"), -1: ("#f1a09f", "#3a1414"), 0: ("#f0efec", "#3b3b38"),
                1: ("#9ec5f4", "#0d2a4f"), 2: ("#2a78d6", "#fff")}
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
        "novembre", "décembre"]


def couleur_secteur(s: str) -> str:
    return COULEURS_SECTEUR.get(s, "#9a9a94")


# ------------------------------------------------------------------ formats français
NNBSP = " "


def pct(v, d: int = 1, signe: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = f"{v * 100:+.{d}f}" if signe else f"{v * 100:.{d}f}"
    return s.replace(".", ",").replace("-", "−") + NNBSP + "%"


def num(v, d: int = 0) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    s = f"{v:,.{d}f}".replace(",", NNBSP).replace(".", ",")
    return s.replace("-", "−")


def date_fr(d: str) -> str:
    dt = datetime.fromisoformat(d)
    return f"{dt.day} {MOIS[dt.month - 1]} {dt.year}"


def secteur_fr(s: str) -> str:
    return SECTEURS_FR.get(s, s)


# ------------------------------------------------------------------ graphiques SVG
def _style():
    plt.rcParams.update({
        "font.family": ["Public Sans", "DejaVu Sans"], "font.size": 9, "svg.fonttype": "none",
        "axes.edgecolor": FILET, "axes.labelcolor": ARDOISE, "xtick.color": ARDOISE, "ytick.color": ENCRE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ECEDEA",
        "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "none", "axes.facecolor": "none",
    })


def _svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    s = buf.getvalue()
    return s[s.find("<svg"):]


def graphique_poids(alloc: pd.DataFrame, plafond: float) -> str:
    _style()
    d = alloc.sort_values("poids")
    fig, ax = plt.subplots(figsize=(6.6, 0.24 * len(d) + 0.6))
    ax.barh(d["ticker"], d["poids"], color=[couleur_secteur(s) for s in d["secteur"]], height=0.62)
    for y, (v, a) in enumerate(zip(d["poids"], d["actuel"])):
        ax.text(v + 0.0015, y, pct(v), va="center", fontsize=7.5, color=ENCRE)
        if a > 0:
            ax.plot([a, a], [y - 0.34, y + 0.34], color=ENCRE, lw=1.4)
    ax.axvline(plafond, color=LAITON, lw=1, ls=(0, (3, 2)))
    ax.text(plafond, len(d) - 0.2, " plafond par titre", color=LAITON, fontsize=7.5, va="bottom")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pct(x, 0)))
    ax.set_xlim(0, max(plafond, d["poids"].max()) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return _svg(fig)


def graphique_secteurs(expo: pd.DataFrame) -> str:
    _style()
    d = expo.iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(d) + 0.7))
    ax.barh(y + 0.19, d["portefeuille"], height=0.36, color=ENCRE, label="Portefeuille")
    ax.barh(y - 0.19, d["indice"], height=0.36, color=NEUTRE, label="Indice (pondéré par la capitalisation)")
    for yy, v in zip(y, d["portefeuille"]):
        ax.text(v + 0.003, yy + 0.19, pct(v), va="center", fontsize=7.5, color=ENCRE)
    ax.axvline(d["plafond"].iloc[0], color=LAITON, lw=1, ls=(0, (3, 2)))
    ax.set_yticks(y, [secteur_fr(s) for s in d.index])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pct(x, 0)))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _svg(fig)


def graphique_bl(bl: pd.DataFrame, n: int = 15) -> str:
    _style()
    d = bl.head(n).iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(d) + 0.8))
    for yy, (a, p) in zip(y, zip(d["a_priori"], d["a_posteriori"])):
        ax.plot([a, p], [yy, yy], color=FILET, lw=2.2, zorder=1)
    ax.scatter(d["a_priori"], y, s=42, color=NEUTRE, zorder=2, label="A priori du marché")
    ax.scatter(d["a_posteriori"], y, s=42, color=SERIE_1, zorder=3, label="Après les vues du comité")
    ax.set_yticks(y, d.index)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pct(x, 0)))
    ax.set_xlabel("Rendement annuel attendu")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    return _svg(fig)


def graphique_performance(perf: pd.DataFrame) -> str:
    _style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.8, 3.9), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.08})
    ax.plot(perf.index, perf["portefeuille"], color=SERIE_1, lw=1.8, label="Portefeuille proposé")
    ax.plot(perf.index, perf["indice"], color=SERIE_2, lw=1.8, label="Indice")
    for col, c, lab in (("portefeuille", SERIE_1, "Portefeuille"), ("indice", SERIE_2, "Indice")):
        ax.annotate(f"{lab} {num(perf[col].iloc[-1], 2)}", (perf.index[-1], perf[col].iloc[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=ENCRE)
    ax.set_ylabel("Valeur de 1 investi")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: num(x, 2)))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    ax2.fill_between(perf.index, perf["dd_portefeuille"], 0, color=SERIE_1, alpha=0.25, lw=0)
    ax2.plot(perf.index, perf["dd_portefeuille"], color=SERIE_1, lw=1)
    ax2.plot(perf.index, perf["dd_indice"], color=SERIE_2, lw=1)
    ax2.set_ylabel("Repli depuis\nle plus haut")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pct(x, 0)))
    ax2.grid(axis="x", visible=False)
    return _svg(fig)


# ------------------------------------------------------------------ petits blocs HTML
def ruban(expo: pd.DataFrame) -> list[dict]:
    """Bande d'allocation par secteur (largeurs proportionnelles aux poids)."""
    d = expo[expo["portefeuille"] > 0.0005]
    return [{"secteur": secteur_fr(s), "poids": float(v), "largeur": f"{v * 100:.3f}%", "couleur": couleur_secteur(s),
             "libelle": pct(v, 0)} for s, v in d["portefeuille"].items()]


def puce_note(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="note vide">·</span>'
    v = int(v)
    bg, fg = NOTE_COULEUR[v]
    return f'<span class="note" style="background:{bg};color:{fg}">{v:+d}</span>'.replace("+0", "0")


def confiance_fr(c) -> str:
    return {"elevee": "élevée", "moyenne": "moyenne", "faible": "faible"}.get(c, c or "—")


def evenement_fr(e: str) -> str:
    return {"selection": "Sélection du comité enregistrée", "decision_approuve": "Décision du gérant : approuvé",
            "decision_modifie": "Décision du gérant : poids modifiés", "decision_rejete": "Décision du gérant : rejeté",
            "rapport": "Note d'investissement émise"}.get(e, e.replace("_", " "))


def md(texte: str) -> str:
    """Markdown -> HTML (le LLM écrit en Markdown) ; repli : paragraphes échappés."""
    texte = (texte or "").strip()
    try:
        import markdown

        return markdown.markdown(texte, extensions=["sane_lists"])
    except ImportError:
        return "".join(f"<p>{html.escape(p)}</p>" for p in texte.split("\n\n") if p.strip())


HYPOTHESES = [
    ("H1", "Aide à la décision", "Le graphe s'arrête avant toute conclusion ; les ordres ci-dessus sont à passer manuellement."),
    ("H2", "Actions long-only", "Univers de l'indice de référence, aucune position courte."),
    ("H3", "Rééquilibrage mensuel", "Un cycle par mois ; chaque cycle a son identifiant de suivi."),
    ("H4", "Sharpe face à l'indice", "Optimisation du ratio de Sharpe ; performance comparée à l'indice."),
    ("H5", "Contraintes du mandat", "Plafonds par titre et par secteur, liquidité minimale, vérifiés par le code."),
    ("H6", "Données", "Prix quotidiens sur trois ans, ratios financiers, actualités."),
    ("H7", "Règle d'or", "Les agents interprètent ; métriques, rendements attendus et poids sont calculés par le code."),
    ("H8", "Point-in-time", "Prix et actualités coupés à la date d'analyse."),
    ("H9", "Boucle de risque", "Trois tentatives d'ajustement au maximum, puis escalade au gérant."),
    ("H10", "Traçabilité", "Thèses, sources, versions des données et du mandat consignées dans le journal d'audit."),
]


# ------------------------------------------------------------------ assemblage
@trace("generer_rapport_html")
def generer_rapport_html(state: dict, poids_actuels: dict | None = None, narration: str = "",
                         liens_traces: list[dict] | None = None, journal: list[dict] | None = None) -> str:
    mandat, rr = state["mandat"], state["rapport_risque"]
    alloc = sy.tableau_allocation(state, poids_actuels)
    alloc_titres = alloc[alloc["ticker"] != "CASH"]
    expo = sy.exposition_sectorielle(state)
    bl = sy.tableau_bl(state)
    perf = sy.performance(state)
    risque = sy.tableau_risque(state)
    k = sy.chiffres_cles(state)
    dec = state.get("decision_gerant") or {}
    piv = sy.notes_pivot(state, list(alloc_titres["ticker"]))
    deb = sy.resume_debat(state)
    ordres = pd.DataFrame(state.get("ordres", []))
    src = state["donnees"]["sources"]
    fond = src["fondamentaux"]

    top = alloc_titres.head(5)
    convictions = [{
        "ticker": r["ticker"], "nom": r["nom"], "secteur": secteur_fr(r["secteur"]), "poids": pct(r["poids"]),
        "vue": r.get("vue"), "confiance": r.get("confiance") or "—", "justification": r.get("justification") or "",
        "theses": [(a.capitalize(), sy.these(state, r["ticker"], a)) for a in sy.AGENTS],
    } for _, r in top.iterrows()]

    perf_port = perf["portefeuille"].iloc[-1] - 1
    perf_ind = perf["indice"].iloc[-1] - 1
    statut = {"approuve": ("Approuvé par le gérant", "approuve"), "modifie": ("Modifié par le gérant", "modifie"),
              "rejete": ("Rejeté par le gérant", "rejete")}.get(dec.get("decision"), ("En attente de validation", "attente"))

    ctx = dict(
        css=CSS, titre=mandat["nom"], date=date_fr(state["date_analyse"]), statut=statut, motif=dec.get("motif", ""),
        k=k, f_pct=pct, f_num=num, ruban=ruban(expo), narration=md(narration),
        alerte=rr.get("alerte", False), problemes=rr["manquements_mandat"] + rr["depassements_limites"],
        explication_risque=md(rr.get("explication", "")),
        convictions=convictions, puce=puce_note,
        alloc=[{**r, "secteur_fr": secteur_fr(r["secteur"]), "couleur": couleur_secteur(r["secteur"])} for r in alloc.to_dict("records")],
        svg_poids=graphique_poids(alloc_titres, mandat["contraintes"]["poids_max_titre"]),
        svg_secteurs=graphique_secteurs(expo), svg_bl=graphique_bl(bl) if not bl.empty else "",
        svg_perf=graphique_performance(perf),
        bl=bl.head(15).reset_index(names="ticker").to_dict("records") if not bl.empty else [],
        kappa=state.get("attendus", {}), opt=state.get("attendus", {}),
        risque=risque.to_dict("records"), stress=sy.tableau_stress(state).to_dict("records"),
        perf_port=perf_port, perf_ind=perf_ind,
        notes=[{"ticker": t, "f": piv.loc[t, "fondamental"], "v": piv.loc[t, "valorisation"], "s": piv.loc[t, "sentiment"]}
               for t in piv.index],
        debat_h=md(deb["haussier"][-1]) if deb["haussier"] else "", debat_b=md(deb["baissier"][-1]) if deb["baissier"] else "",
        tours=deb["tours"], synthese=md(state.get("synthese_debat", "")),
        ordres=ordres.to_dict("records") if not ordres.empty else [],
        total_achats=float(ordres.loc[ordres["sens"] == "ACHAT", "montant"].sum()) if not ordres.empty else 0.0,
        total_ventes=float(ordres.loc[ordres["sens"] == "VENTE", "montant"].sum()) if not ordres.empty else 0.0,
        devise=mandat.get("devise", ""), hypotheses=HYPOTHESES,
        version_mandat=mandat["version"], empreinte=mandat["empreinte"], version_donnees=state["donnees"]["version"],
        sources=src, fond_pit=fond.get("point_in_time", True), fond_avert=fond.get("avertissement", ""),
        nb_eligibles=len(state["univers"]["eligibles"]), nb_exclus=len(state["univers"]["exclus"]),
        nb_notes=len(state.get("notes", [])), nb_selection=len(state.get("selection", [])),
        iterations=state.get("iterations_risque", 0), moteur=state.get("attendus", {}).get("moteur", ""),
        liens=liens_traces or [], journal=journal or [], genere=datetime.now().strftime("%d/%m/%Y %H:%M"),
        contraintes=mandat["contraintes"], conf=confiance_fr, evenement=evenement_fr,
    )
    return Environment(autoescape=False).from_string(TEMPLATE).render(**ctx)


# ------------------------------------------------------------------ style et gabarit
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=Public+Sans:wght@400;500;600&display=swap');
:root { --encre:#0E3B43; --papier:#FAFAF8; --laiton:#B0802A; --ardoise:#5B6670; --filet:#DADDD8; --texte:#1E2A30; }
* { box-sizing: border-box; }
html { background: #ECEEEA; }
body { margin: 0; color: var(--texte); font: 400 10.5pt/1.55 'Public Sans', system-ui, sans-serif;
       font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased; }
.page { max-width: 210mm; margin: 24px auto; background: var(--papier); padding: 22mm 20mm 18mm;
        box-shadow: 0 1px 0 var(--filet), 0 18px 40px -24px rgba(14,59,67,.35); }
h1, h2, h3 { font-family: 'Spectral', Georgia, serif; color: var(--encre); font-weight: 500; }
h1 { font-size: 34pt; line-height: 1.05; margin: 0 0 6px; letter-spacing: -.01em; }
h2 { font-size: 17pt; margin: 0 0 4px; display: flex; gap: 12px; align-items: baseline; }
h2 .num { font-size: 11pt; color: var(--laiton); font-weight: 600; min-width: 1.6em; }
h3 { font-size: 12.5pt; margin: 18px 0 6px; }
p { margin: 0 0 9px; max-width: 72ch; }
.chapeau { color: var(--ardoise); max-width: 70ch; margin: 2px 0 16px; }
section { padding-top: 26px; margin-top: 26px; border-top: 1px solid var(--filet); }
/* couverture */
.entete { display: flex; justify-content: space-between; color: var(--ardoise); font-size: 9pt; padding-bottom: 28px; }
.entete strong { color: var(--encre); font-weight: 600; }
.sous-titre { font-family: 'Spectral', Georgia, serif; font-size: 14pt; color: var(--ardoise); margin: 0 0 26px; }
.verdict { display: grid; grid-template-columns: auto 1fr; gap: 18px; align-items: center; margin: 0 0 26px;
           border-left: 3px solid var(--laiton); padding: 6px 0 6px 16px; }
.verdict .etat { font-family: 'Spectral', Georgia, serif; font-size: 15pt; color: var(--laiton); font-weight: 600; }
.verdict .detail { color: var(--ardoise); font-size: 9.5pt; }
.verdict.rejete { border-color: #c7403f; } .verdict.rejete .etat { color: #c7403f; }
.verdict.attente { border-color: var(--ardoise); } .verdict.attente .etat { color: var(--ardoise); }
.ruban { display: flex; height: 46px; border-radius: 3px; overflow: hidden; gap: 2px; background: var(--papier); }
.ruban div { min-width: 2px; }
.ruban-legende { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 9px 0 0; font-size: 8.5pt; color: var(--ardoise); }
.ruban-legende span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px;
                              margin-right: 6px; background: var(--c); vertical-align: 0; }
.ruban-legende b { color: var(--encre); font-weight: 600; margin-left: 4px; }
.chiffres { display: grid; grid-template-columns: repeat(5, 1fr); margin: 26px 0 4px; border-top: 1px solid var(--encre); }
.chiffres div { padding: 10px 12px 4px 0; }
.chiffres .v { font-family: 'Spectral', Georgia, serif; font-size: 20pt; color: var(--encre); line-height: 1.1; }
.chiffres .l { font-size: 8.5pt; color: var(--ardoise); }
/* tableaux */
table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 8px 0 4px; }
th { text-align: left; font-weight: 600; color: var(--ardoise); font-size: 8pt; padding: 6px 8px 6px 0;
     border-bottom: 1px solid var(--encre); }
td { padding: 5px 8px 5px 0; border-bottom: 1px solid #ECEDEA; vertical-align: top; }
td.n, th.n { text-align: right; white-space: nowrap; }
td.t { font-weight: 600; color: var(--encre); white-space: nowrap; }
tr.cash td { color: var(--ardoise); font-style: italic; }
.pastille { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
.pos { color: #1c5cab; } .neg { color: #b03a39; }
.note { display: inline-block; min-width: 26px; text-align: center; border-radius: 3px; padding: 1px 4px;
        font-weight: 600; font-size: 8.5pt; }
.note.vide { color: var(--filet); }
.figure { margin: 12px 0 4px; } .figure svg { width: 100%; height: auto; }
.legende { font-size: 8.5pt; color: var(--ardoise); margin: 2px 0 0; }
.deux { display: grid; grid-template-columns: 1fr 1fr; gap: 26px; }
/* convictions */
.conviction { display: grid; grid-template-columns: 112px 1fr; gap: 16px; padding: 12px 0; border-bottom: 1px solid #ECEDEA; }
.conviction .tk { font-family: 'Spectral', Georgia, serif; font-size: 16pt; color: var(--encre); line-height: 1; }
.conviction .meta { font-size: 8.5pt; color: var(--ardoise); margin-top: 4px; }
.conviction .juge { margin: 0 0 6px; }
.conviction dl { margin: 0; display: grid; grid-template-columns: 92px 1fr; gap: 3px 10px; font-size: 9pt; }
.conviction dt { color: var(--ardoise); } .conviction dd { margin: 0; }
/* risque */
.jauge { position: relative; height: 8px; background: #ECEDEA; border-radius: 4px; min-width: 120px; }
.jauge i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; background: var(--encre); }
.jauge i.over { background: #c7403f; }
.jauge b { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--laiton); left: calc(100% / 1.25); }
.statut { font-size: 8.5pt; white-space: nowrap; } .statut.ok { color: #0a7a0a; } .statut.ko { color: #b03a39; font-weight: 600; }
.statut.ok::before { content: "✓ "; } .statut.ko::before { content: "✕ "; }
.encadre { background: #F1F3EF; padding: 12px 16px; border-radius: 3px; margin: 12px 0; }
.encadre.alerte { background: #FBEDEC; border-left: 3px solid #c7403f; }
/* débat */
.debat h3 { margin-top: 4px; } .debat ul { padding-left: 18px; margin: 0; } .debat li { margin-bottom: 4px; font-size: 9pt; }
/* méthode */
.hyp { display: grid; grid-template-columns: 38px 170px 1fr; gap: 4px 10px; font-size: 9pt; }
.hyp .h { color: var(--laiton); font-weight: 600; } .hyp .t { color: var(--encre); font-weight: 600; }
.pied { margin-top: 30px; padding-top: 12px; border-top: 1px solid var(--filet); font-size: 8pt; color: var(--ardoise); }
a { color: var(--encre); }
@media print {
  html { background: none; } .page { box-shadow: none; margin: 0; padding: 0; max-width: none; background: none; }
  h2, h3, .chapeau { break-after: avoid; } .bloc { break-inside: avoid; }
  @page { size: A4; margin: 16mm 15mm; }
  section { break-inside: auto; } .saut { break-before: page; } .conviction, tr, .figure { break-inside: avoid; }
}
@media (max-width: 760px) { .page { padding: 18px; } .chiffres { grid-template-columns: repeat(2, 1fr); }
  .deux { grid-template-columns: 1fr; } h1 { font-size: 26pt; } .conviction { grid-template-columns: 1fr; } }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Note d'investissement, {{ date }}</title><style>{{ css }}</style></head>
<body><main class="page">

<div class="entete"><span><strong>Comité d'investissement</strong>, note mensuelle</span><span>{{ date }}</span></div>
<h1>Note d'investissement</h1>
<p class="sous-titre">{{ titre }}</p>

<div class="verdict {{ statut[1] }}">
  <div class="etat">{{ statut[0] }}</div>
  <div class="detail">{% if motif %}« {{ motif }} »<br>{% endif %}
  {% if k.conforme %}Allocation conforme au mandat et à ses limites de risque.{% elif alerte %}Limites de risque non tenues après {{ iterations }} tentatives : décision prise en connaissance de cause.{% else %}Allocation hors limites : voir la section risque.{% endif %}</div>
</div>

<div class="ruban" role="img" aria-label="Allocation par secteur">{% for r in ruban %}<div style="flex: 0 0 {{ r.largeur }}; background: {{ r.couleur }}" title="{{ r.secteur }} {{ r.libelle }}"></div>{% endfor %}</div>
<div class="ruban-legende">{% for r in ruban %}<span style="--c: {{ r.couleur }}">{{ r.secteur }}<b>{{ r.libelle }}</b></span>{% endfor %}</div>

<div class="chiffres">
  <div><div class="v">{{ f_pct(k.rendement) }}</div><div class="l">rendement annuel attendu</div></div>
  <div><div class="v">{{ f_pct(k.volatilite) }}</div><div class="l">volatilité attendue</div></div>
  <div><div class="v">{{ f_num(k.sharpe, 2) }}</div><div class="l">ratio de Sharpe attendu</div></div>
  <div><div class="v">{{ k.lignes }}</div><div class="l">lignes en portefeuille</div></div>
  <div><div class="v">{{ f_pct(k.var, 2) }}</div><div class="l">perte journalière à 95 %</div></div>
</div>

<section>
  <h2><span class="num">1</span>Synthèse</h2>
  {{ narration }}
  {% if problemes %}<div class="encadre {% if alerte %}alerte{% endif %}"><strong>Points d'attention :</strong> {{ problemes | join('; ') }}.</div>{% endif %}
  <p class="legende">Parcours de ce cycle : {{ nb_eligibles }} titres éligibles ({{ nb_exclus }} exclus par le mandat), {{ nb_notes }} notes d'analystes,
  {{ nb_selection }} titres retenus par le comité, {{ iterations }} passage{{ 's' if iterations > 1 }} par le contrôle des risques.</p>
</section>

<section>
  <h2><span class="num">2</span>Les convictions du comité</h2>
  <p class="chapeau">Les cinq premières positions, avec l'avis de chacun des trois analystes et l'arbitrage du juge.</p>
  {% for c in convictions %}
  <div class="conviction">
    <div><div class="tk">{{ c.ticker }}</div><div class="meta">{{ c.nom }}<br>{{ c.secteur }}<br><strong>{{ c.poids }}</strong> du portefeuille</div></div>
    <div>
      <p class="juge">{{ puce(c.vue) }} &nbsp;Vue du comité, confiance {{ conf(c.confiance) }}. {{ c.justification }}</p>
      <dl>{% for a, t in c.theses %}<dt>{{ a }}</dt><dd>{{ t }}</dd>{% endfor %}</dl>
    </div>
  </div>
  {% endfor %}
</section>

<section class="saut">
  <h2><span class="num">3</span>Allocation</h2>
  <p class="chapeau">Poids cible par titre, coloré par secteur. Le trait vertical sur une barre marque le poids actuel ; les pointillés, le plafond de {{ f_pct(contraintes.poids_max_titre, 0) }} par titre fixé par le mandat.</p>
  <div class="figure">{{ svg_poids }}</div>
  <div class="bloc"><h3>Exposition sectorielle</h3>
  <p class="chapeau">Face à l'indice pondéré par la capitalisation ; plafond de {{ f_pct(contraintes.poids_max_secteur, 0) }} par secteur.</p>
  <div class="figure">{{ svg_secteurs }}</div></div>
  <h3>Détail des lignes</h3>
  <table><thead><tr><th>Titre</th><th>Secteur</th><th class="n">Cible</th><th class="n">Actuel</th><th class="n">Écart</th><th class="n">Fond.</th><th class="n">Valo.</th><th class="n">Sent.</th><th class="n">Vue</th></tr></thead><tbody>
  {% for r in alloc %}<tr class="{{ 'cash' if r.ticker == 'CASH' }}"><td class="t"><span class="pastille" style="background: {{ r.couleur }}"></span>{{ r.ticker }}</td><td>{{ r.secteur_fr }}</td>
  <td class="n">{{ f_pct(r.poids) }}</td><td class="n">{{ f_pct(r.actuel) }}</td>
  <td class="n {{ 'pos' if r.variation > 0 else 'neg' }}">{{ f_pct(r.variation, 1, True) }}</td>
  {% if r.ticker == 'CASH' %}<td></td><td></td><td></td><td></td>{% else %}<td class="n">{{ puce(r.get('fond')) }}</td><td class="n">{{ puce(r.get('valo')) }}</td><td class="n">{{ puce(r.get('sent')) }}</td><td class="n">{{ puce(r.get('vue')) }}</td>{% endif %}</tr>{% endfor %}
  </tbody></table>
  <p class="legende">Notes de −2 (très défavorable) à +2 (très favorable) : fondamental, valorisation, sentiment de presse, puis vue finale du comité.</p>
</section>

{% if bl %}
<section>
  <h2><span class="num">4</span>Des vues aux poids</h2>
  <p class="chapeau">Black-Litterman part des rendements implicites du marché, puis les déplace vers les vues du comité, d'autant plus que la confiance est élevée. Le comité donne une vue qualitative ; le code la chiffre.</p>
  <div class="figure">{{ svg_bl }}</div>
  <table><thead><tr><th>Titre</th><th class="n">Vue</th><th>Confiance</th><th class="n">A priori</th><th class="n">Vue chiffrée</th><th class="n">A posteriori</th><th class="n">Poids</th></tr></thead><tbody>
  {% for r in bl %}<tr><td class="t">{{ r.ticker }}</td><td class="n">{{ puce(r.vue_comite) }}</td><td>{{ conf(r.confiance) }}</td><td class="n">{{ f_pct(r.a_priori) }}</td><td class="n">{{ f_pct(r.vue) }}</td><td class="n">{{ f_pct(r.a_posteriori) }}</td><td class="n">{{ f_pct(r.poids) }}</td></tr>{% endfor %}
  </tbody></table>
  <p class="legende">Moteur d'optimisation : {{ moteur }}. Covariance Ledoit-Wolf, optimisation sous les contraintes du mandat.</p>
</section>
{% endif %}

<section class="saut">
  <h2><span class="num">5</span>Risque</h2>
  <p class="chapeau">Chaque mesure comparée à la limite du mandat ; le repère doré marque 100 % de la limite.</p>
  <table><thead><tr><th>Indicateur</th><th class="n">Mesure</th><th class="n">Limite</th><th style="width: 32%">Utilisation de la limite</th><th>Statut</th></tr></thead><tbody>
  {% for r in risque %}<tr><td>{{ r.indicateur }}</td><td class="n">{{ f_pct(r.valeur, 2) }}</td><td class="n">{{ f_pct(r.limite, 2) }}</td>
  <td><div class="jauge"><i class="{{ 'over' if r.utilisation > 1 }}" style="width: {{ [r.utilisation, 1.25] | min / 1.25 * 100 }}%"></i><b></b></div></td>
  <td><span class="statut {{ 'ok' if r.statut == 'respectée' else 'ko' }}">{{ r.statut }}</span></td></tr>{% endfor %}
  </tbody></table>
  <div class="encadre">{{ explication_risque }}</div>
  <h3>Scénarios de stress</h3>
  <table><thead><tr><th>Scénario</th><th class="n">Perte estimée du portefeuille</th></tr></thead><tbody>
  {% for s in stress %}<tr><td>{{ s.scenario }}</td><td class="n">{{ f_pct(s.perte) }}</td></tr>{% endfor %}</tbody></table>
  <div class="bloc"><h3>Comportement passé de l'allocation proposée</h3>
  <div class="figure">{{ svg_perf }}</div>
  <p class="legende">Calcul dans l'échantillon : {{ f_pct(perf_port, 1, True) }} pour le portefeuille contre {{ f_pct(perf_ind, 1, True) }} pour l'indice sur la période. Les poids sont choisis avec ces mêmes données : c'est une illustration, pas un backtest (H8).</p></div>
</section>

<section>
  <h2><span class="num">6</span>Le débat du comité</h2>
  <p class="chapeau">{{ tours }} tour{{ 's' if tours > 1 }} d'échanges entre un chercheur haussier et un chercheur baissier, puis l'arbitrage du juge.</p>
  <div class="deux debat"><div><h3>Arguments haussiers</h3>{{ debat_h }}</div><div><h3>Objections baissières</h3>{{ debat_b }}</div></div>
  {% if synthese %}<h3>Arbitrage du juge</h3>{{ synthese }}{% endif %}
</section>

<section class="saut">
  <h2><span class="num">7</span>Ordres à passer</h2>
  <p class="chapeau">Quantités entières au cours de référence. Aucun ordre n'est transmis automatiquement : le gérant les passe lui-même (H1).</p>
  {% if ordres %}
  <table><thead><tr><th>Titre</th><th>Sens</th><th class="n">Quantité</th><th class="n">Cours de référence</th><th class="n">Montant ({{ devise }})</th><th class="n">Poids cible</th></tr></thead><tbody>
  {% for o in ordres %}<tr><td class="t">{{ o.ticker }}</td><td class="{{ 'pos' if o.sens == 'ACHAT' else 'neg' }}">{{ 'Achat' if o.sens == 'ACHAT' else 'Vente' }}</td><td class="n">{{ f_num(o.quantite) }}</td><td class="n">{{ f_num(o.prix_ref, 2) }}</td><td class="n">{{ f_num(o.montant) }}</td><td class="n">{{ f_pct(o.poids_cible) }}</td></tr>{% endfor %}
  </tbody></table>
  <p class="legende">Total des achats : {{ f_num(total_achats) }} {{ devise }}. Total des ventes : {{ f_num(total_ventes) }} {{ devise }}.</p>
  {% else %}<p>Aucun ordre : le portefeuille est déjà à sa cible.</p>{% endif %}
</section>

<section>
  <h2><span class="num">8</span>Méthode et traçabilité</h2>
  <div class="hyp">{% for h, t, d in hypotheses %}<div class="h">{{ h }}</div><div class="t">{{ t }}</div><div>{{ d }}</div>{% endfor %}</div>
  <h3>Versions et sources</h3>
  <table><tbody>
  <tr><td>Mandat</td><td>version {{ version_mandat }}, empreinte {{ empreinte }}</td></tr>
  <tr><td>Données</td><td>version {{ version_donnees }}</td></tr>
  <tr><td>Prix</td><td>{{ sources.prix }}</td></tr>
  <tr><td>Ratios financiers</td><td>{{ sources.fondamentaux.fournisseur }}{% if not fond_pit %} (attention : {{ fond_avert }}){% endif %}</td></tr>
  <tr><td>Actualités</td><td>{{ sources.actualites }}</td></tr>
  {% for l in liens %}<tr><td>Trace LangSmith</td><td>{% if l.url %}<a href="{{ l.url }}">{{ l.etape }}</a>{% else %}{{ l.etape }} ({{ l.run_id[:8] }}){% endif %}</td></tr>{% endfor %}
  </tbody></table>
  {% if journal %}<h3>Journal d'audit</h3><table><tbody>{% for e in journal %}<tr><td style="width: 34%">{{ e.horodatage[6:8] }}/{{ e.horodatage[4:6] }}/{{ e.horodatage[:4] }} à {{ e.horodatage[9:11] }}:{{ e.horodatage[11:13] }}:{{ e.horodatage[13:15] }} UTC</td><td>{{ evenement(e.evenement) }}</td></tr>{% endfor %}</tbody></table>{% endif %}
  <div class="pied">Document généré le {{ genere }} par le système multi-agents du comité. Outil pédagogique d'aide à la décision : ce document n'est pas un conseil en investissement.</div>
</section>

</main></body></html>"""
