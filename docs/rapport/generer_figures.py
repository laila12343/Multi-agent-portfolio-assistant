"""Exécute un cycle complet et exporte figures, tableaux et chiffres clés pour le rapport LaTeX et les slides.

    python docs/rapport/generer_figures.py                          # démonstration hors-ligne (mock + synthétique)
    python docs/rapport/generer_figures.py --backend ollama --source yfinance   # vrai cycle (Colab)

Sorties (dans docs/rapport/) :
  figures/*.pdf (LaTeX) et figures/*.png (PowerPoint)
  chiffres.tex   : macros \\RendementAttendu, \\Volatilite… (le rapport les lit avec \\input)
  tab_*.tex      : tableaux prêts à inclure
Recompiler rapport.tex après chaque nouvel export : les chiffres du rapport se mettent à jour.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import AppContext, load_settings  # noqa: E402
from src.local_runner import executer_sans_langgraph  # noqa: E402
from src.nodes.preparation import prix_frame  # noqa: E402
from src.tools import rapport_html as rh  # noqa: E402
from src.tools import syntheses as sy  # noqa: E402
from src.tools.reporting import lire_portefeuille_actuel, valoriser  # noqa: E402

FIG = ICI / "figures"


def tex_pct(v, d=1):
    return f"{v * 100:.{d}f}".replace(".", "{,}").replace("-", "$-$") + r"\,\%"


def tex_esc(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def capturer(nom: str, fonction, *args):
    """Réutilise les graphiques du rapport HTML, mais les enregistre en PDF + PNG."""
    original = rh._svg

    def enregistrer(fig):
        fig.savefig(FIG / f"{nom}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{nom}.png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return ""

    rh._svg = enregistrer
    try:
        fonction(*args)
    finally:
        rh._svg = original


def fig_notes(piv):
    rh._style()
    d = piv.reindex(columns=sy.AGENTS)
    fig, ax = plt.subplots(figsize=(4.2, 0.26 * len(d) + 0.8))
    cmap = matplotlib.colors.ListedColormap([rh.NOTE_COULEUR[k][0] for k in range(-2, 3)])
    ax.imshow(d.values.astype(float), cmap=cmap, vmin=-2.5, vmax=2.5, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v):+d}".replace("+0", "0"), ha="center", va="center", fontsize=7.5,
                        color=rh.NOTE_COULEUR[int(v)][1])
    ax.set_xticks(range(3), ["Fondamental", "Valorisation", "Sentiment"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(d)), d.index)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return rh._svg(fig)


def fig_risque(tr):
    rh._style()
    d = tr.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 0.6))
    u = d["utilisation"].clip(upper=1.3)
    ax.barh(d["indicateur"], [1.3] * len(d), color="#ECEDEA", height=0.5)
    ax.barh(d["indicateur"], u, color=[rh.ENCRE if x <= 1 else "#c7403f" for x in d["utilisation"]], height=0.5)
    ax.axvline(1, color=rh.LAITON, lw=1.5)
    for y, (x, v, l) in enumerate(zip(u, d["valeur"], d["limite"])):
        ax.text(1.32, y, f"{rh.pct(v, 2)} / {rh.pct(l, 2)}", va="center", fontsize=7.5, color=rh.ENCRE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: rh.pct(x, 0)))
    ax.set_xlim(0, 1.3)
    ax.set_xlabel("Utilisation de la limite du mandat")
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    return rh._svg(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--date", default=None)
    ap.add_argument("--titres", type=int, default=50)
    a = ap.parse_args()
    s = load_settings(overrides={"llm": {"backend": a.backend}, "analyse": {"max_titres": a.titres},
                                 "donnees": {"fournisseur": a.source, "repli_synthetique": True}})
    ctx = AppContext.build(s)
    st = executer_sans_langgraph(ctx, a.date, decisions=[{"decision": "approuve", "motif": "validé en comité"}])
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    prix = prix_frame(st["donnees"]).iloc[-1].to_dict()
    actuels, _ = valoriser(lire_portefeuille_actuel(s), prix)
    alloc = sy.tableau_allocation(st, actuels)
    titres = alloc[alloc["ticker"] != "CASH"]
    expo, bl, perf, tr = sy.exposition_sectorielle(st), sy.tableau_bl(st), sy.performance(st), sy.tableau_risque(st)
    piv = sy.notes_pivot(st, list(titres["ticker"]))

    capturer("poids", rh.graphique_poids, titres, st["mandat"]["contraintes"]["poids_max_titre"])
    capturer("secteurs", rh.graphique_secteurs, expo)
    capturer("black_litterman", rh.graphique_bl, bl)
    capturer("performance", rh.graphique_performance, perf)
    capturer("notes", fig_notes, piv)
    capturer("risque", fig_risque, tr)

    k = sy.chiffres_cles(st)
    rr = st["rapport_risque"]
    macros = {
        "DateAnalyse": rh.date_fr(st["date_analyse"]), "RendementAttendu": tex_pct(k["rendement"]),
        "Volatilite": tex_pct(k["volatilite"]), "Sharpe": f"{k['sharpe']:.2f}".replace(".", "{,}"),
        "NbLignes": str(k["lignes"]), "VaR": tex_pct(k["var"], 2), "CVaR": tex_pct(rr["mesures"]["cvar_historique_1j"], 2),
        "Rotation": tex_pct(k["rotation"], 0), "NbEligibles": str(len(st["univers"]["eligibles"])),
        "NbNotes": str(len(st["notes"])), "NbSelection": str(len(st["selection"])), "NbIterations": str(st["iterations_risque"]),
        "Conforme": "conforme" if rr["conforme"] else "non conforme", "Moteur": tex_esc(st["attendus"]["moteur"]),
        "BackendLLM": tex_esc(a.backend), "SourceDonnees": tex_esc(a.source),
        "PerfPortefeuille": tex_pct(perf["portefeuille"].iloc[-1] - 1), "PerfIndice": tex_pct(perf["indice"].iloc[-1] - 1),
        "PireStress": tex_pct(max(rr["stress"].values())), "VersionDonnees": st["donnees"]["version"],
    }
    (ICI / "chiffres.tex").write_text("% généré par generer_figures.py\n" + "".join(
        f"\\newcommand{{\\{n}}}{{{v}}}\n" for n, v in macros.items()), encoding="utf-8")

    lignes = [f"{t} & {tex_esc(rh.secteur_fr(sec))} & {tex_pct(p)} & {int(v):+d} & {tex_esc(rh.confiance_fr(c))} \\\\"
              for t, sec, p, v, c in titres.head(12)[["ticker", "secteur", "poids", "vue", "confiance"]].itertuples(index=False)]
    (ICI / "tab_allocation.tex").write_text("\n".join(lignes), encoding="utf-8")
    lignes = [f"{tex_esc(r.indicateur)} & {tex_pct(r.valeur, 2)} & {tex_pct(r.limite, 2)} & {r.statut} \\\\" for r in tr.itertuples()]
    (ICI / "tab_risque.tex").write_text("\n".join(lignes), encoding="utf-8")
    lignes = [f"{t} & {int(r.vue_comite):+d} & {tex_esc(rh.confiance_fr(r.confiance))} & {tex_pct(r.a_priori)} & {tex_pct(r.a_posteriori)} & {tex_pct(r.poids)} \\\\"
              for t, r in bl.head(8).iterrows()]
    (ICI / "tab_bl.tex").write_text("\n".join(lignes), encoding="utf-8")
    lignes = [f"{tex_esc(r.scenario)} & {tex_pct(r.perte)} \\\\" for r in sy.tableau_stress(st).itertuples()]
    (ICI / "tab_stress.tex").write_text("\n".join(lignes), encoding="utf-8")
    print("Figures et tableaux exportés dans", ICI)


if __name__ == "__main__":
    main()
