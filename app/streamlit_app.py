"""Interface du comité d'investissement : lancer un cycle, suivre les agents, décider, lire la note.

    streamlit run app/streamlit_app.py
Le graphe s'arrête sur interrupt() à l'étape 8 (validation) ; les boutons reprennent avec Command(resume=...).
"""
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import AppContext, load_settings  # noqa: E402
from src.graph import build_graph, config_cycle, etat_cycle, executer_flux, interruption_en_cours, liens_traces  # noqa: E402
from src.observabilite import langsmith_actif, url_projet  # noqa: E402
from src.tools import syntheses as sy  # noqa: E402
from src.tools.mandat import lire_mandat  # noqa: E402
from src.tools.rapport_html import confiance_fr, evenement_fr, pct, puce_note, ruban, secteur_fr  # noqa: E402
from src.tools.reporting import lire_journal, lire_portefeuille_actuel, valoriser  # noqa: E402
from src.ui import charts  # noqa: E402
from src.ui.marche import apercu_marche  # noqa: E402

st.set_page_config(page_title="Assistant multi-agent de portefeuille", page_icon="📈", layout="wide")

# ================================================================== style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], button, input, textarea { font-family: 'Public Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Spectral', Georgia, serif !important; color: #0E3B43; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.2rem; max-width: 1360px; }

/* bandeau */
.hero { background: linear-gradient(120deg, #0E3B43 0%, #14555E 55%, #1D7A6E 100%); border-radius: 18px;
        padding: 26px 30px; color: #fff; display: flex; gap: 22px; align-items: center; margin-bottom: 20px;
        box-shadow: 0 18px 40px -24px rgba(14,59,67,.7); }
.hero .logo { flex: 0 0 74px; height: 74px; border-radius: 20px; background: rgba(255,255,255,.12);
              display: flex; align-items: center; justify-content: center; font-size: 40px; }
.hero h1 { color: #fff !important; font-size: 2.15rem; margin: 0; padding: 0; line-height: 1.1; }
.hero p { margin: 6px 0 10px; color: #CFE3E0; font-size: 1.02rem; }
.hero .puces span { display: inline-block; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.22);
                    border-radius: 999px; padding: 3px 11px; margin: 0 6px 4px 0; font-size: .8rem; color: #fff; }
.hero .etat { margin-left: auto; align-self: flex-start; padding: 8px 14px; border-radius: 999px; font-weight: 700;
              font-size: .85rem; white-space: nowrap; }
.etat.vide { background: rgba(255,255,255,.14); color: #fff; }
.etat.attente { background: #F5C451; color: #3D2A00; }
.etat.fini { background: #59C98A; color: #06351C; }
.etat.rejet { background: #F08A88; color: #4A0F0E; }

/* suivi des 9 étapes */
.etapes { display: grid; grid-template-columns: repeat(9, 1fr); gap: 6px; margin: 2px 0 22px; position: relative; }
.etape { text-align: center; font-size: .8rem; color: #8A939A; }
.etape .rond { width: 46px; height: 46px; margin: 0 auto 6px; border-radius: 50%; display: flex; align-items: center;
               justify-content: center; font-size: 21px; background: #fff; border: 2px solid #DADDD8; position: relative; }
.etape.fait .rond { background: var(--c); border-color: var(--c); }
.etape.fait { color: #0E3B43; font-weight: 600; }
.etape.fait .rond::after { content: "✓"; position: absolute; right: -4px; bottom: -4px; width: 18px; height: 18px;
                           border-radius: 50%; background: #1BAF7A; color: #fff; font-size: 11px; line-height: 18px; }
.etape.actif .rond { border-color: var(--c); box-shadow: 0 0 0 5px color-mix(in srgb, var(--c) 22%, transparent); }
.etape.actif { color: var(--c); font-weight: 700; }
@media (prefers-reduced-motion: no-preference) { .etape.actif .rond { animation: pouls 1.6s ease-in-out infinite; } }
@keyframes pouls { 50% { box-shadow: 0 0 0 9px color-mix(in srgb, var(--c) 10%, transparent); } }

/* cartes */
.cartes { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 6px 0 18px; }
.carte { background: #fff; border-radius: 14px; padding: 18px; border: 1px solid #E6E9E4;
         box-shadow: 0 10px 24px -20px rgba(14,59,67,.55); }
.carte .ic { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
             font-size: 22px; background: var(--f); margin-bottom: 10px; }
.carte h4 { margin: 0 0 4px; font-size: 1.08rem; }
.carte p { margin: 0; color: #4E5A63; font-size: .9rem; line-height: 1.45; }
.kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 0 0 14px; }
.kpi { background: #fff; border-radius: 14px; padding: 14px 16px; border: 1px solid #E6E9E4;
       box-shadow: 0 10px 24px -20px rgba(14,59,67,.55); display: flex; gap: 12px; align-items: center; }
.kpi .ic { flex: 0 0 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
           font-size: 20px; background: var(--f); }
.kpi .v { font-family: 'Spectral', Georgia, serif; font-size: 1.65rem; font-weight: 700; color: var(--c); line-height: 1.05; }
.kpi .l { font-size: .78rem; color: #5B6670; }
.ruban { display: flex; height: 14px; gap: 2px; border-radius: 999px; overflow: hidden; margin: 4px 0 6px; }
.legende { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: .8rem; color: #5B6670; margin-bottom: 14px; }
.legende span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; background: var(--c); }
.mandat span { display: inline-block; background: #EEF3F2; color: #0E3B43; border-radius: 999px; padding: 5px 12px;
               margin: 0 6px 8px 0; font-size: .85rem; font-weight: 600; }
.agent { background: #fff; border: 1px solid #E6E9E4; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; }
.agent .tete { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.agent .tete b { font-family: 'Spectral', Georgia, serif; font-size: 1.1rem; color: #0E3B43; }
.agent .ic { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; background: var(--f); font-size: 17px; }
.note { display: inline-block; min-width: 30px; text-align: center; border-radius: 6px; padding: 2px 6px; font-weight: 700; font-size: .85rem; }
.discret { color: #5B6670; font-size: .86rem; }
.juge { background: linear-gradient(135deg, #FBF4E6, #FFFFFF); border: 1px solid #EAD9B5; border-radius: 14px; padding: 16px 18px; }
.trace { border-left: 2px solid #DADDD8; margin-left: 8px; padding-left: 16px; }
.trace div { position: relative; padding: 4px 0 10px; font-size: .9rem; }
.trace div::before { content: ""; position: absolute; left: -23px; top: 9px; width: 12px; height: 12px; border-radius: 50%; background: #1D7A6E; }

/* boutons et onglets */
.stButton > button, .stDownloadButton > button, .stLinkButton > a { border-radius: 10px !important; font-weight: 600 !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(120deg, #0E3B43, #1D7A6E) !important; border: 0 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: #EEF3F2; border-radius: 10px 10px 0 0; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: #0E3B43 !important; color: #fff !important; }
.stTabs [aria-selected="true"] p { color: #fff !important; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #EAF2F0, #F7FAF9); }
[data-testid="stSidebar"] h3 { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

CODE, LLM, HUMAIN = "#2a78d6", "#6250d6", "#e07b2e"
ETAPES = [("univers", "Univers", "construire_univers", "🌍", CODE), ("donnees", "Données", "collecter_donnees", "🗄️", CODE),
          ("metriques", "Métriques", "calculer_metriques", "📐", CODE), ("notes", "Analystes", "analyse_titre", "🤖", LLM),
          ("selection", "Comité", "debat", "⚖️", LLM), ("poids", "Optimisation", "optimiser", "🎯", CODE),
          ("rapport_risque", "Risque", "controle_risque", "🛡️", CODE), ("decision_gerant", "Validation", "validation_gerant", "✍️", HUMAIN),
          ("rapport_fichier", "Note", "rapport", "📄", CODE)]
AGENTS = {"fondamental": ("🏭", "#E3EEFB"), "sentiment": ("📰", "#FCE8EF"), "valorisation": ("🏷️", "#FFF3D6")}


def suivi_html(etat: dict, actif: str | None = None) -> str:
    cases = []
    for cle, lib, noeud, ico, c in ETAPES:
        cls = "fait" if etat.get(cle) else ("actif" if noeud == actif else "")
        cases.append(f'<div class="etape {cls}" style="--c:{c}"><div class="rond">{ico}</div>{lib}</div>')
    return f'<div class="etapes">{"".join(cases)}</div>'


def hero(sous_titre: str, libelle: str, cls: str) -> None:
    st.markdown(f"""<div class="hero"><div class="logo">📈</div><div>
<h1>Assistant multi-agent de sélection de portefeuille</h1><p>{sous_titre}</p>
<div class="puces"><span>🧠 LangGraph</span><span>🤖 Qwen2.5 · Ollama</span><span>⚖️ Black-Litterman</span><span>🛡️ VaR · CVaR</span><span>🔍 LangSmith</span></div>
</div><div class="etat {cls}">{libelle}</div></div>""", unsafe_allow_html=True)


def cartes_html(items) -> str:
    return '<div class="cartes">' + "".join(
        f'<div class="carte"><div class="ic" style="--f:{f}">{i}</div><h4>{t}</h4><p>{d}</p></div>' for i, f, t, d in items) + "</div>"


def kpis_html(items) -> str:
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="ic" style="--f:{f}">{i}</div><div><div class="v" style="--c:{c}">{v}</div><div class="l">{l}</div></div></div>'
        for i, f, c, v, l in items) + "</div>"


def graphe(fig, key: str):
    """Chaque graphique a une clé unique : le même graphique peut ainsi apparaître dans deux onglets."""
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


# ================================================================== barre latérale
with st.sidebar:
    st.markdown("### 🚀 Nouveau cycle")
    moteurs = ["mock", "ollama", "openai_compatible", "hf"]
    defaut = os.getenv("LLM_BACKEND", "mock")
    backend = st.selectbox("🤖 Moteur des agents", moteurs, index=moteurs.index(defaut) if defaut in moteurs else 0,
                           help="« mock » : règles fixes, sans modèle (démonstration instantanée). « ollama » : Qwen2.5 en local.")
    source = st.selectbox("📊 Données de marché", ["yfinance", "synthetic"], help="« synthetic » fonctionne sans internet.")
    max_titres = st.slider("🔎 Titres confiés aux analystes", 5, 50, 15 if backend != "mock" else 30,
                           help="Chaque titre mobilise 3 agents : avec un vrai modèle, comptez environ 1 minute par titre.")
    d = st.date_input("📅 Date d'analyse", value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
    lancer = st.button("Lancer l'analyse", type="primary", use_container_width=True, icon=":material/rocket_launch:")
    st.divider()
    st.markdown("### 🔍 Observabilité")


@st.cache_resource(show_spinner="Préparation des agents…")
def preparer(backend: str, source: str, max_titres: int):
    s = load_settings(overrides={"llm": {"backend": backend}, "analyse": {"max_titres": max_titres},
                                 "donnees": {"fournisseur": source, "repli_synthetique": True}})
    ctx = AppContext.build(s)
    return build_graph(ctx), ctx


@st.cache_data(ttl=3600, show_spinner="Chargement de l'aperçu du marché…")
def marche(source: str, jour: str):
    s = load_settings(overrides={"donnees": {"fournisseur": source, "repli_synthetique": True}})
    try:
        return apercu_marche(s, jour)
    except Exception:
        return apercu_marche(load_settings(overrides={"donnees": {"fournisseur": "synthetic"}}), jour)


app, ctx = preparer(backend, source, max_titres)

with st.sidebar:
    if langsmith_actif():
        st.success(f"Traces actives : projet {os.getenv('LANGSMITH_PROJECT', 'portfolio-agents')}", icon="✅")
        lien = url_projet()
        if lien:
            st.link_button("Ouvrir LangSmith", lien, use_container_width=True, icon=":material/open_in_new:")
    else:
        st.info("Ajoutez LANGSMITH_API_KEY dans le fichier .env pour suivre chaque agent dans LangSmith.", icon="💡")


# ================================================================== exécution
def executer(entree, etape: str, date_analyse: str | None = None):
    zone = st.empty()
    etat = dict(etat_cycle(app, st.session_state.cfg)) if etape != "analyse" else {}
    vus = 0
    with st.status("Les agents travaillent…", expanded=True) as status:
        try:
            for noeud, upd in executer_flux(app, st.session_state.cfg, entree, date_analyse, etape):
                if not isinstance(upd, dict):
                    continue
                etat.update({k: True for k in upd})
                if noeud == "analyse_titre":
                    vus += 1
                    status.update(label=f"🤖 Analyse des titres : {vus} terminés")
                    continue
                zone.markdown(suivi_html(etat, noeud), unsafe_allow_html=True)
                for ligne in upd.get("journal", []):
                    st.write("✅ " + ligne.split("] ", 1)[-1])
            status.update(label="Étape terminée", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Le cycle s'est arrêté sur une erreur", state="error")
            st.error(f"{type(exc).__name__} : {exc}. Vérifiez le moteur choisi (Ollama lancé ?) ou passez en « mock ».")
    zone.empty()


if lancer:
    st.session_state.cfg = config_cycle(ctx)
    executer({"date_analyse": d.isoformat()}, "analyse", d.isoformat())

cfg = st.session_state.get("cfg")
etat = etat_cycle(app, cfg) if cfg else {}
demande = interruption_en_cours(app, cfg) if cfg else None
decision = (etat.get("decision_gerant") or {}).get("decision")

if not cfg:
    libelle, cls = "Aucun cycle en cours", "vide"
elif demande:
    libelle, cls = "⏳ En attente de votre décision", "attente"
elif decision == "rejete":
    libelle, cls = "⛔ Proposition rejetée", "rejet"
elif etat.get("rapport_fichier"):
    libelle, cls = "✅ Note d'investissement émise", "fini"
else:
    libelle, cls = "Cycle interrompu", "vide"

mandat = etat.get("mandat") or {}
sous = "Le LLM analyse, le code calcule, le gérant décide."
if etat.get("date_analyse"):
    sous = f"{mandat.get('nom', '')} — analyse du {pd.Timestamp(etat['date_analyse']).strftime('%d/%m/%Y')}"
hero(sous, libelle, cls)

# ================================================================== accueil
if not cfg:
    st.markdown(suivi_html({}), unsafe_allow_html=True)
    st.markdown(cartes_html([
        ("🌍", "#E3EEFB", "Univers et données", "50 grandes capitalisations filtrées par le mandat, prix sur 3 ans, ratios et actualités, sans regarder le futur."),
        ("🤖", "#ECE9FB", "Trois analystes IA par titre", "Fondamental, sentiment et valorisation travaillent en parallèle et notent chaque titre de −2 à +2."),
        ("⚖️", "#FFF3D6", "Débat et juge", "Un chercheur haussier et un baissier s'affrontent ; un juge retient 15 à 30 titres avec une vue et une confiance."),
        ("🛡️", "#E3F2EA", "Optimisation et risque", "Black-Litterman sous contraintes, VaR, CVaR et stress tests, puis votre validation avant toute conclusion."),
    ]), unsafe_allow_html=True)
    g, dr = st.columns([1.7, 1], gap="large")
    with g:
        st.markdown("#### 📈 Aperçu du marché sur un an")
        apercu = marche(source, d.isoformat())
        if apercu is not None and not apercu.empty:
            graphe(charts.fig_marche(apercu), key="fig_marche_1")
            st.caption("Indice et indices sectoriels équipondérés de l'univers, base 100. "
                       + ("Données simulées." if source == "synthetic" else "Données yfinance."))
    with dr:
        st.markdown("#### 📋 Le mandat en bref")
        m = lire_mandat(ctx.settings["project"]["mandat_file"])
        c, lim = m["contraintes"], m["limites_risque"]
        st.markdown(f"""<div class="carte"><p style="margin-bottom:10px">{m['nom']}</p><div class="mandat">
<span>🎯 {pct(c['poids_max_titre'], 0)} max par titre</span><span>🏢 {pct(c['poids_max_secteur'], 0)} max par secteur</span>
<span>📉 VaR ≤ {pct(lim['var95_1j_max'], 1)} / jour</span><span>🌊 Volatilité ≤ {pct(lim['volatilite_annuelle_max'], 0)}</span>
<span>🔄 Rotation ≤ {pct(lim['rotation_max'], 0)}</span><span>🚫 Pas de vente à découvert</span></div>
<p style="margin-top:8px">Valeur du portefeuille : <b>{f"{m['valeur_portefeuille']:,.0f}".replace(",", " ")} {m['devise']}</b></p></div>""",
                    unsafe_allow_html=True)
        st.info("Choisissez le moteur et la date à gauche, puis cliquez sur **Lancer l'analyse**.", icon="👈")
    st.stop()

st.markdown(suivi_html(etat, "validation_gerant" if demande else None), unsafe_allow_html=True)
if not etat.get("poids"):
    st.warning("Le cycle n'a pas atteint l'optimisation. Relancez l'analyse depuis le panneau de gauche.", icon="⚠️")
    st.stop()

# ================================================================== tableau de bord du cycle
prix = sy.prix_frame(etat["donnees"]).iloc[-1].to_dict()
poids_actuels, _ = valoriser(lire_portefeuille_actuel(ctx.settings), prix)
alloc = sy.tableau_allocation(etat, poids_actuels)
expo = sy.exposition_sectorielle(etat)
k = sy.chiffres_cles(etat)
tr = sy.tableau_risque(etat)

st.markdown(kpis_html([
    ("📈", "#E3F2EA", "#0F8A4F", pct(k["rendement"]), "rendement annuel attendu"),
    ("🌊", "#E3EEFB", "#1C5CAB", pct(k["volatilite"]), "volatilité attendue"),
    ("⭐", "#FFF3D6", "#9A6B12", f"{k['sharpe']:.2f}".replace(".", ","), "ratio de Sharpe attendu"),
    ("🧩", "#ECE9FB", "#4B3BB8", str(k["lignes"]), "lignes en portefeuille"),
    ("🛡️", "#FCE8EF" if not k["conforme"] else "#E3F2EA", "#B03A39" if not k["conforme"] else "#0F8A4F",
     "Conforme" if k["conforme"] else "À revoir", f"VaR 95 % : {pct(k['var'], 2)} / jour"),
]), unsafe_allow_html=True)
rb = ruban(expo)
st.markdown('<div class="ruban">' + "".join(f'<div style="flex:0 0 {r["largeur"]};background:{r["couleur"]}" title="{r["secteur"]} {r["libelle"]}"></div>' for r in rb)
            + '</div><div class="legende">' + "".join(f'<span style="--c:{r["couleur"]}">{r["secteur"]} <b>{r["libelle"]}</b></span>' for r in rb)
            + "</div>", unsafe_allow_html=True)

onglets = st.tabs([":material/how_to_reg: Décision", ":material/pie_chart: Allocation", ":material/psychology: Analystes",
                   ":material/forum: Comité", ":material/shield: Risque", ":material/description: Note d'investissement",
                   ":material/travel_explore: Traçabilité"])

# ------------------------------------------------------------------ décision
with onglets[0]:
    rr = etat["rapport_risque"]
    if demande is None:
        if decision == "rejete":
            st.error(f"Proposition rejetée. Motif consigné : {etat['decision_gerant'].get('motif') or '—'}", icon="⛔")
        else:
            st.success("Décision enregistrée : la note d'investissement est disponible dans son onglet.", icon="✅")
    else:
        if demande["alerte"]:
            st.error("Limites de risque non tenues après trois tentatives : " + "; ".join(demande["problemes"])
                     + ". Vous pouvez approuver en connaissance de cause, ajuster ou rejeter.", icon="🚨")
        elif demande["conforme"]:
            st.success("La proposition respecte le mandat et toutes ses limites de risque. Elle attend votre décision.", icon="✅")
        g, dr = st.columns([1.35, 1], gap="large")
        with g:
            st.markdown("#### ✍️ Poids proposés")
            st.caption("Modifiez la colonne « Poids (%) » pour ajuster ; le total est remis à 100 % à l'enregistrement.")
            ed = alloc[["ticker", "nom", "secteur", "poids", "actuel", "vue", "confiance"]].copy()
            ed["poids"] = (ed["poids"] * 100).round(2)
            ed["actuel"] = (ed["actuel"] * 100).round(2)
            ed["secteur"] = ed["secteur"].map(secteur_fr)
            ed["confiance"] = ed["confiance"].map(confiance_fr)
            modif = st.data_editor(
                ed, hide_index=True, use_container_width=True, key="editeur", height=420,
                disabled=["ticker", "nom", "secteur", "actuel", "vue", "confiance"],
                column_config={"ticker": "Titre", "nom": "Société", "secteur": "Secteur",
                               "poids": st.column_config.NumberColumn("Poids (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f"),
                               "actuel": st.column_config.NumberColumn("Actuel (%)", format="%.2f"),
                               "vue": st.column_config.NumberColumn("Vue", format="%+d"), "confiance": "Confiance"})
            total = modif["poids"].sum()
            st.caption(f"Total saisi : {total:.2f} %".replace(".", ","))
        with dr:
            st.markdown("#### 🧭 Répartition sectorielle")
            graphe(charts.fig_donut(expo), key="fig_donut_1")
            st.markdown("#### 🛡️ Lecture du risque")
            st.write(rr["explication"])
            motif = st.text_area("💬 Votre commentaire", placeholder="Obligatoire pour ajuster ou rejeter ; consigné au journal d'audit.")
            change = not modif["poids"].round(2).equals(ed["poids"].round(2))
            if st.button("Approuver la proposition", type="primary", use_container_width=True, icon=":material/check_circle:"):
                from langgraph.types import Command

                executer(Command(resume={"decision": "approuve", "motif": motif}), "validation_approuve")
                st.rerun()
            b2, b3 = st.columns(2)
            if b2.button("Enregistrer les ajustements", use_container_width=True, icon=":material/edit:",
                         disabled=not (motif and change), help="Modifiez au moins un poids et ajoutez un commentaire."):
                from langgraph.types import Command

                nouveaux = {t: p / 100 for t, p in zip(modif["ticker"], modif["poids"]) if p > 0}
                executer(Command(resume={"decision": "modifie", "poids": nouveaux, "motif": motif}), "validation_modifie")
                st.rerun()
            if b3.button("Rejeter", use_container_width=True, icon=":material/cancel:", disabled=not motif):
                from langgraph.types import Command

                executer(Command(resume={"decision": "rejete", "motif": motif}), "validation_rejete")
                st.rerun()
    st.markdown("#### 🎛️ Limites du mandat")
    graphe(charts.fig_jauges(tr), key="fig_jauges_1")

# ------------------------------------------------------------------ allocation
with onglets[1]:
    g, dr = st.columns([1.1, 1], gap="large")
    with g:
        st.markdown("#### 🎯 Poids par titre")
        graphe(charts.fig_poids(alloc, mandat["contraintes"]["poids_max_titre"]), key="fig_poids_1")
    with dr:
        st.markdown("#### 🧭 Répartition sectorielle")
        graphe(charts.fig_donut(expo), key="fig_donut_2")
        st.markdown("#### 🏢 Face à l'indice")
        graphe(charts.fig_secteurs(expo), key="fig_secteurs_1")
    bl = sy.tableau_bl(etat)
    if not bl.empty:
        st.markdown("#### ⚖️ Des vues du comité aux rendements attendus (Black-Litterman)")
        st.caption("Gris : a priori du marché. Bleu : après les vues du comité, d'autant plus déplacé que la confiance est forte.")
        graphe(charts.fig_bl(bl), key="fig_bl_1")

# ------------------------------------------------------------------ analystes
with onglets[2]:
    piv = sy.notes_pivot(etat)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    g, dr = st.columns([1, 1.25], gap="large")
    with g:
        st.markdown("#### 🗺️ Les trois avis sur chaque titre")
        st.caption("Bleu : favorable. Rouge : défavorable. De −2 à +2.")
        graphe(charts.fig_notes(piv), key="fig_notes_1")
    with dr:
        t = st.selectbox("🔎 Choisir un titre", list(piv.index),
                         format_func=lambda x: f"{x} — {etat['univers']['info'].get(x, {}).get('name', '')}")
        info = etat["univers"]["info"].get(t, {})
        retenu = t in etat.get("selection", [])
        st.markdown(f"<p class='discret'>{secteur_fr(info.get('sector', ''))}, "
                    f"{'✅ retenu par le comité' if retenu else '❌ non retenu par le comité'}</p>", unsafe_allow_html=True)
        for n in [x for x in etat["notes"] if x["ticker"] == t]:
            ico, fond = AGENTS.get(n["agent"], ("🤖", "#EEF3F2"))
            risques = "".join(f"<li>{r}</li>" for r in n["risques"]) or "<li>aucun relevé</li>"
            st.markdown(f"""<div class="agent"><div class="tete"><div class="ic" style="--f:{fond}">{ico}</div>
<b>{n['agent'].capitalize()}</b>{puce_note(n['score'])}</div><p>{n['these']}</p>
<p class="discret">⚠️ Risques :</p><ul class="discret">{risques}</ul>
<p class="discret">📚 Sources : {', '.join(n['sources'][:4]) or '—'}</p></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ comité
with onglets[3]:
    g, dr = st.columns([1.3, 1], gap="large")
    with g:
        st.markdown("#### 💬 Le débat")
        for m_ in etat.get("debat", []):
            haussier = m_["role"] == "haussier"
            with st.chat_message("assistant" if haussier else "user", avatar="🐂" if haussier else "🐻"):
                st.markdown(f"**Chercheur {'haussier' if haussier else 'baissier'}**, tour {m_['tour']}")
                st.markdown(m_["texte"])
    with dr:
        st.markdown("#### ⚖️ L'arbitrage du juge")
        st.markdown(f"<div class='juge'>{etat.get('synthese_debat') or '—'}</div>", unsafe_allow_html=True)
        vues = pd.DataFrame(etat.get("vues", {})).T.reset_index(names="ticker")
        if not vues.empty:
            vues["confiance"] = vues["confiance"].map(confiance_fr)
            st.dataframe(vues[["ticker", "vue", "confiance", "justification"]], hide_index=True, use_container_width=True,
                         column_config={"ticker": "Titre", "vue": st.column_config.NumberColumn("Vue", format="%+d"),
                                        "confiance": "Confiance", "justification": "Justification"})

# ------------------------------------------------------------------ risque
with onglets[4]:
    st.markdown("#### 🎛️ Utilisation des limites du mandat")
    st.caption("Vert : marge confortable. Ambre : proche de la limite. Rouge : dépassement. Trait doré : 100 % de la limite.")
    graphe(charts.fig_jauges(tr), key="fig_jauges_2")
    g, dr = st.columns([1, 1.4], gap="large")
    with g:
        st.markdown("#### 🌪️ Scénarios de stress")
        icones = ["📉", "💻", "🌩️"]
        st.markdown(cartes_html([(icones[i % 3], "#FCE8EF", f"Perte estimée : {pct(r.perte)}", r.scenario)
                                 for i, r in enumerate(sy.tableau_stress(etat).itertuples())]).replace("repeat(4, 1fr)", "1fr"),
                    unsafe_allow_html=True)
    with dr:
        st.markdown("#### 📈 Comportement passé de l'allocation")
        graphe(charts.fig_performance(sy.performance(etat)), key="fig_performance_1")
        st.caption("Calcul dans l'échantillon : les poids sont choisis avec ces mêmes données. Illustration, pas un backtest.")

# ------------------------------------------------------------------ note
with onglets[5]:
    fichier = etat.get("rapport_fichier")
    if fichier and Path(fichier).exists():
        doc = Path(fichier).read_text(encoding="utf-8")
        c1, c2 = st.columns([1, 3])
        c1.download_button("Télécharger la note", doc, file_name=Path(fichier).name, mime="text/html",
                           type="primary", use_container_width=True, icon=":material/download:")
        c2.caption("Ouvrez le fichier dans votre navigateur puis « Imprimer → Enregistrer en PDF » : la mise en page est prévue pour l'A4.")
        components.html(doc, height=1500, scrolling=True)
    elif decision == "rejete":
        st.info("Aucune note : la proposition a été rejetée.", icon="⛔")
    else:
        st.info("La note d'investissement est rédigée dès que vous approuvez la proposition (onglet Décision).", icon="📄")

# ------------------------------------------------------------------ traçabilité
with onglets[6]:
    g, dr = st.columns(2, gap="large")
    with g:
        st.markdown("#### 🔍 Traces LangSmith")
        if not langsmith_actif():
            st.info("Traces désactivées : ajoutez LANGSMITH_API_KEY (offre gratuite) pour voir chaque agent, outil et appel de modèle.", icon="💡")
        for i, l in enumerate(liens_traces(cfg), 1):
            if l["url"]:   # numéro dans le libellé : deux boutons identiques provoqueraient la même erreur d'ID
                st.link_button(f"Trace {i} : {l['etape']}", l["url"], icon=":material/open_in_new:")
            else:
                st.markdown(f"- {l['etape']} : `{l['run_id'][:8]}`")
        st.markdown("#### 🧾 Journal d'audit")
        lignes = []
        for e in lire_journal(ctx.store, etat["date_analyse"]):
            h = e["horodatage"]
            lignes.append(f"<div><b>{h[9:11]}:{h[11:13]}:{h[13:15]}</b> — {evenement_fr(e['evenement'])}</div>")
        st.markdown(f"<div class='trace'>{''.join(lignes) or '<div>Aucun événement</div>'}</div>", unsafe_allow_html=True)
    with dr:
        st.markdown("#### 🧭 Déroulé du cycle")
        st.code("\n".join(etat.get("journal", [])), language=None)
        st.caption(f"Mandat {mandat.get('version')} (empreinte {mandat.get('empreinte')}), données {etat['donnees']['version']}, "
                   f"cycle {cfg['configurable']['thread_id']}.")