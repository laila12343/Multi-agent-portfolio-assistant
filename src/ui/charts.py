"""Graphiques Plotly de l'interface Streamlit (mêmes couleurs et mêmes chiffres que le rapport)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tools.rapport_html import (ARDOISE, ENCRE, FILET, LAITON, NEUTRE, NOTE_COULEUR, SERIE_1, SERIE_2,
                                    couleur_secteur, secteur_fr)

FONT = "Public Sans, system-ui, sans-serif"


def _layout(fig: go.Figure, hauteur: int, **kw) -> go.Figure:
    base = dict(
        height=hauteur, margin=dict(l=8, r=24, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=ENCRE), separators=", ", hoverlabel=dict(font_family=FONT, bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=ARDOISE)))
    base.update(kw)                     # un graphique peut remplacer n'importe quel réglage (ex. la légende)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="#ECEDEA", zeroline=False, linecolor=FILET, tickfont=dict(color=ARDOISE))
    fig.update_yaxes(gridcolor="#ECEDEA", zeroline=False, linecolor=FILET)
    return fig


def fig_poids(alloc: pd.DataFrame, plafond: float) -> go.Figure:
    d = alloc[alloc["ticker"] != "CASH"].sort_values("poids")
    fig = go.Figure(go.Bar(
        x=d["poids"], y=d["ticker"], orientation="h", marker_color=[couleur_secteur(s) for s in d["secteur"]],
        text=[f"{v:.1%}".replace(".", ",") for v in d["poids"]], textposition="outside", cliponaxis=False,
        customdata=list(zip(d["nom"], [secteur_fr(s) for s in d["secteur"]], d["actuel"])),
        hovertemplate="<b>%{y}</b> %{customdata[0]}<br>%{customdata[1]}<br>Cible %{x:.1%}<br>Actuel %{customdata[2]:.1%}<extra></extra>"))
    fig.add_vline(x=plafond, line=dict(color=LAITON, dash="dot", width=1.5),
                  annotation_text="plafond par titre", annotation_font_color=LAITON, annotation_position="top")
    fig.update_xaxes(tickformat=".0%", range=[0, max(plafond, d["poids"].max()) * 1.2])
    return _layout(fig, 26 * len(d) + 70, showlegend=False, bargap=0.32)


def fig_secteurs(expo: pd.DataFrame) -> go.Figure:
    d = expo.iloc[::-1]
    noms = [secteur_fr(s) for s in d.index]
    fig = go.Figure()
    fig.add_bar(y=noms, x=d["indice"], orientation="h", name="Indice", marker_color=NEUTRE,
                hovertemplate="%{y}<br>Indice %{x:.1%}<extra></extra>")
    fig.add_bar(y=noms, x=d["portefeuille"], orientation="h", name="Portefeuille", marker_color=ENCRE,
                hovertemplate="%{y}<br>Portefeuille %{x:.1%}<extra></extra>")
    fig.add_vline(x=float(d["plafond"].iloc[0]), line=dict(color=LAITON, dash="dot", width=1.5),
                  annotation_text="plafond sectoriel", annotation_font_color=LAITON, annotation_position="top")
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, 44 * len(d) + 90, barmode="group", bargap=0.28, bargroupgap=0.08)


def fig_bl(bl: pd.DataFrame, n: int = 15) -> go.Figure:
    d = bl.head(n).iloc[::-1]
    xs, ys = [], []
    for t, r in d.iterrows():
        xs += [r["a_priori"], r["a_posteriori"], None]
        ys += [t, t, None]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=FILET, width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=d["a_priori"], y=d.index, mode="markers", name="A priori du marché",
                    marker=dict(color=NEUTRE, size=11), hovertemplate="%{y}<br>A priori %{x:.1%}<extra></extra>")
    fig.add_scatter(x=d["a_posteriori"], y=d.index, mode="markers", name="Après les vues du comité",
                    marker=dict(color=SERIE_1, size=11), hovertemplate="%{y}<br>A posteriori %{x:.1%}<extra></extra>")
    fig.update_xaxes(tickformat=".0%", title_text="Rendement annuel attendu", title_font=dict(color=ARDOISE, size=11))
    return _layout(fig, 30 * len(d) + 90)


def fig_notes(piv: pd.DataFrame) -> go.Figure:
    """Carte des notes (-2..+2) : palette divergente bleu / rouge, valeur écrite dans chaque case."""
    cols = ["fondamental", "valorisation", "sentiment"]
    d = piv.reindex(columns=cols).iloc[::-1]
    z = d.values.astype(float)
    texte = [[("" if pd.isna(v) else f"{int(v):+d}".replace("+0", "0")) for v in row] for row in z]
    echelle = []
    for i, k in enumerate(range(-2, 3)):
        echelle += [[i / 5, NOTE_COULEUR[k][0]], [(i + 1) / 5, NOTE_COULEUR[k][0]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Fondamental", "Valorisation", "Sentiment"], y=list(d.index), zmin=-2.5, zmax=2.5,
                               colorscale=echelle, showscale=False, text=texte, texttemplate="%{text}", xgap=3, ygap=3,
                               hovertemplate="%{y} / %{x} : %{text}<extra></extra>"))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 24 * len(d) + 80)


def fig_performance(perf: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    for col, c, nom in (("portefeuille", SERIE_1, "Portefeuille proposé"), ("indice", SERIE_2, "Indice")):
        fig.add_scatter(x=perf.index, y=perf[col], name=nom, line=dict(color=c, width=2), row=1, col=1,
                        hovertemplate=f"{nom}<br>%{{x|%d/%m/%Y}} : %{{y:.2f}}<extra></extra>")
        fig.add_scatter(x=perf.index, y=perf[f"dd_{col}"], showlegend=False, line=dict(color=c, width=1),
                        fill="tozeroy" if col == "portefeuille" else None, fillcolor="rgba(42,120,214,.15)",
                        row=2, col=1, hovertemplate=f"{nom}<br>repli %{{y:.1%}}<extra></extra>")
    fig.update_yaxes(title_text="Valeur de 1 investi", row=1, col=1, title_font=dict(size=11, color=ARDOISE))
    fig.update_yaxes(title_text="Repli", tickformat=".0%", row=2, col=1, title_font=dict(size=11, color=ARDOISE))
    return _layout(fig, 420, hovermode="x unified")


# ------------------------------------------------------------------ graphiques « vitrine »
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def fig_marche(df: pd.DataFrame) -> go.Figure:
    """Aperçu du marché : indice et indices sectoriels équipondérés, base 100."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        epais = 3.2 if i == 0 else 1.8
        fig.add_scatter(x=df.index, y=df[col], name=col, mode="lines",
                        line=dict(color=ENCRE if i == 0 else PALETTE[(i - 1) % len(PALETTE)], width=epais),
                        fill="tozeroy" if i == 0 else None, fillcolor="rgba(14,59,67,0.06)",
                        hovertemplate=f"{col}<br>%{{x|%d/%m/%Y}} : %{{y:.1f}}<extra></extra>")
    lo, hi = float(df.min().min()), float(df.max().max())
    fig.update_yaxes(range=[lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.05], title_text="base 100", title_font=dict(size=11, color=ARDOISE))
    return _layout(fig, 360, hovermode="x unified")


def fig_donut(expo: pd.DataFrame) -> go.Figure:
    d = expo[expo["portefeuille"] > 0.0005]
    fig = go.Figure(go.Pie(labels=[secteur_fr(s) for s in d.index], values=d["portefeuille"], hole=0.62, sort=False,
                           marker=dict(colors=[couleur_secteur(s) for s in d.index], line=dict(color="white", width=2)),
                           textinfo="percent", textfont=dict(size=12, color="white"),
                           hovertemplate="%{label}<br>%{value:.1%}<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(d)}</b><br>secteurs", showarrow=False, font=dict(size=16, color=ENCRE))
    return _layout(fig, 320, showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=11, color=ARDOISE)))


def fig_jauges(tr: pd.DataFrame) -> go.Figure:
    """Une jauge par limite du mandat : vert sous 80 %, ambre jusqu'à 100 %, rouge au-delà."""
    n = len(tr)
    fig = make_subplots(rows=2, cols=(n + 1) // 2, specs=[[{"type": "indicator"}] * ((n + 1) // 2)] * 2, vertical_spacing=0.18)
    for i, r in enumerate(tr.itertuples()):
        u = float(r.utilisation) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=u, number=dict(suffix=" %", font=dict(size=22, color=ENCRE)),
            title=dict(text=r.indicateur, font=dict(size=12, color=ARDOISE)),
            gauge=dict(axis=dict(range=[0, 130], tickwidth=0, tickfont=dict(size=9, color=ARDOISE)),
                       bar=dict(color=ENCRE if u <= 100 else "#c7403f", thickness=0.28),
                       steps=[dict(range=[0, 80], color="#E3F2EA"), dict(range=[80, 100], color="#FBF0D9"),
                              dict(range=[100, 130], color="#F8DEDD")],
                       threshold=dict(line=dict(color=LAITON, width=3), thickness=0.8, value=100))),
            row=i // ((n + 1) // 2) + 1, col=i % ((n + 1) // 2) + 1)
    return _layout(fig, 430)
