# Architecture — Multi-agent : sélection d'actifs et optimisation de portefeuille

> **Le LLM analyse, le code calcule, le gérant décide.**

Ce document décrit comment le code implémente le schéma d'architecture de l'équipe
(hypothèses H1-H10, workflow en 9 étapes, graphe LangGraph, inventaire des outils).

## 1. Hypothèses de travail (à valider avec les gérants)

| | Hypothèse | Où c'est appliqué |
|---|---|---|
| H1 | Aide à la décision : le gérant valide, aucun ordre automatique | `nodes/validation.py` (`interrupt()`), ordres « à passer manuellement » dans le rapport |
| H2 | Actions cotées long-only, univers de 50 à 300 titres d'un indice | `data/universe_sp50.csv`, `mandat.contraintes.long_only` |
| H3 | Rééquilibrage mensuel, traitement par lot | `mandat.frequence`, un `thread_id` par cycle |
| H4 | Maximiser le Sharpe par rapport à l'indice | `optimiser` (max Sharpe), benchmark `SPY` dans `calculer_metriques` (bêta) |
| H5 | Max 10 %/titre, 30 %/secteur, liquidité minimale, pas de vente à découvert | `config/mandat.yaml`, `filtrer_univers`, `contraintes_mandat`, `verifier_conformite` |
| H6 | Prix quotidiens ≥ 3 ans, états financiers, actualités | `settings.donnees`, `collecter_donnees` |
| H7 | Le LLM ne produit aucun chiffre | métriques et poids calculés dans `src/tools/` ; prompts (`agents/prompts.py`) ; tableaux du rapport générés par le code |
| H8 | Backtest point-in-time | prix coupés à `date_analyse`, actualités filtrées, RAG filtré par date de fichier ; **fondamentaux yfinance non PIT (signalé)** |
| H9 | 3 tentatives d'ajustement du risque, puis escalade | `router_risque`, `settings.optimisation.tentatives`, drapeau `alerte` |
| H10 | Chaque décision journalisée : thèse, sources, version des données | `Store` (`journaliser`), empreinte du mandat, version des données |

## 2. Workflow → nœuds

| Étape | Type | Nœud(s) | Fichier |
|---|---|---|---|
| 1 Construire l'univers | Code | `charger_mandat`, `construire_univers` | `src/nodes/preparation.py` |
| 2 Collecter les données | Code | `collecter_donnees` (RetryPolicy) | `src/nodes/preparation.py` |
| 3 Calculer les métriques | Code | `calculer_metriques` → `Send` × N | `src/nodes/preparation.py` |
| 4 Analyser chaque titre | LLM ×3 en parallèle | sous-graphe `analyse_titre` | `src/agents/analystes.py` |
| 5 Débattre et sélectionner | LLM | sous-graphe `debat` (haussier ↔ baissier, juge) | `src/agents/debat.py` |
| 6 Optimiser | Code | `optimiser` | `src/nodes/portefeuille.py` |
| 7 Contrôler les risques | Code + LLM | `controle_risque` | `src/nodes/portefeuille.py` |
| 8 Valider | Humain | `validation_gerant` (`interrupt`) | `src/nodes/validation.py` |
| 9 Rapport et journal | Code + LLM | `rapport` | `src/nodes/rapport.py` |

## 3. Graphe LangGraph (`src/graph.py`)

```
START → charger_mandat → construire_univers → collecter_donnees → calculer_metriques
      ─Send × N titres→ analyse_titre [agent_fondamental ‖ agent_sentiment ‖ agent_valorisation]
      → debat [chercheur_haussier ↔ chercheur_baissier (2 tours) → juge_selection]
      → optimiser → controle_risque ─non conforme (≤ 3)→ optimiser
                                     ─conforme (ou 3 essais)→ validation_gerant
      validation_gerant ─approuvé→ rapport → END
                        ─modifié→ controle_risque
                        ─rejeté→ END
```

**État partagé `EtatPortefeuille`** (`src/state.py`) : `date_analyse`, `mandat`, `univers`, `donnees`,
`metriques`, `notes` (reducer `operator.add`), `selection`, `vues`, `poids`, `rapport_risque`,
`iterations_risque`, `decision_gerant`, `rapport_final` — plus `debat`, `attendus` et `journal` (trace lisible).

**Sorties structurées Pydantic** : `NoteAnalyste` (score −2..+2, thèse, risques, sources) et
`Selection` (liste de `VueTitre` : vue −2..+2, confiance faible/moyenne/elevee, justification).

## 4. Inventaire des outils — statut

| Outil | Nœud | Implémentation retenue |
|---|---|---|
| `lire_mandat()` | charger_mandat | YAML (`config/mandat.yaml`), empreinte SHA-256 |
| `get_constituants()` | construire_univers | CSV + capitalisation/liquidité yfinance |
| `filtrer_univers()` | construire_univers | pandas, motif d'exclusion par titre |
| `get_prix()`, `get_fondamentaux()` | collecter_donnees | yfinance (prototype) / synthétique |
| `get_actualites()` | collecter_donnees, agent_sentiment | yfinance news + Google News RSS |
| `RetryPolicy` | collecter_donnees | LangGraph (3 essais, back-off) |
| `calculer_facteurs()` | calculer_metriques | pandas/numpy, z-scores (valorisation : intra-secteur) |
| `Send()`, reducer | calculer_metriques | LangGraph |
| `create_agent` | 3 analystes | LangChain 1.0, `response_format=NoteAnalyste` |
| `get_metriques_titre()` | fondamental, valorisation | lecture du state (+ percentiles) |
| `rechercher_rapport()` | agent_fondamental | RAG TF-IDF ou sentence-transformers ; 10-K SEC EDGAR |
| recherche web | agent_sentiment | **DuckDuckGo (`ddgs`, sans clé)** ; Tavily si `TAVILY_API_KEY` |
| `comparer_pairs()` | agent_valorisation | pandas (médianes sectorielles) |
| `NoteAnalyste`, `Selection` | agents, juge | Pydantic |
| Ledoit-Wolf, Black-Litterman | optimiser | PyPortfolioOpt (repli scikit-learn/numpy) |
| EfficientFrontier, solveur | optimiser | PyPortfolioOpt + cvxpy (repli scipy SLSQP) |
| `contraintes_mandat()` | optimiser | Python |
| `verifier_conformite()` | controle_risque | Python |
| `calculer_var()`, `stress_test()` | controle_risque | numpy (historique, paramétrique, bêta, sectoriel, pire mois) |
| `interrupt()`, `Command` | validation_gerant | LangGraph |
| `calculer_mouvements()` | rapport | pandas (quantités entières) |
| `generer_rapport()` | rapport | note HTML autonome A4 (`tools/rapport_html.py`, graphiques SVG) + version Markdown |
| `Store` | rapport, debat, validation | `InMemoryStore` (namespace `("audit", date)`) |
| Checkpointer | tout le graphe | **InMemorySaver** par défaut ; SQLite ; `PostgresSaver` (production) |
| LangSmith | tout le graphe | `src/observabilite.py` : actif si `LANGSMITH_API_KEY` ; `@trace` sur les outils « code », métadonnées par cycle, liens vers les traces |
| tâche planifiée | déclenchement | cron mensuel → `python -m ...` ou demande du gérant (Streamlit) |

## 5. Paramètres κ et δ (Black-Litterman)

- **δ (delta)** = aversion au risque implicite du marché : a priori `π = δ·Σ·w_marché + r_f`.
- **κ (kappa)** = rendement excédentaire annuel associé à un point de vue du juge : `Q_i = π_i + κ·vue_i`.
- Confiance du juge → méthode d'Idzorek (faible 25 %, moyenne 50 %, élevée 75 %).

Ainsi le LLM ne fixe jamais un rendement (H7) : il donne une vue qualitative, le code la chiffre.

## 6. Écarts assumés et points à trancher

1. **Données** : prototype sur le S&P 500 (50 titres) avec yfinance, faute de licence Bourse de Casablanca.
   L'interface des outils ne change pas pour EODHD / FMP / AMMC.
2. **Gratuit d'abord** : DuckDuckGo plutôt que Tavily, InMemory/SQLite plutôt que Postgres, LangSmith optionnel.
3. **Sous-graphes** appelés depuis un nœud « enveloppe » (seules les `notes` remontent dans l'état principal).
4. **Garde-fous du juge** en code : tickers valides, 15-30 titres, ≤ 6 par secteur, liste faisable.
5. **Stratégie des 3 essais** (H9) : max Sharpe → utilité quadratique + L2 + coût de rotation → min volatilité.
6. **Modifié → controle_risque** : si les poids du gérant sont non conformes, le schéma renvoie vers
   `optimiser`, qui remplace alors ses poids. À discuter : renvoyer plutôt au gérant avec alerte ?
7. **Rotation max 75 %** dans le mandat de démonstration (1er cycle, 25 % de liquidités à investir).
