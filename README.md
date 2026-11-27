# portfolio-agents

**Multi-agent : sélection d'actifs et optimisation de portefeuille** — LangGraph, 100 % open source.

> Le LLM analyse, le code calcule, le gérant décide.

9 étapes : univers → données → métriques → **3 analystes LLM en parallèle par titre** (fondamental,
sentiment, valorisation) → **débat haussier/baissier + juge** → **Black-Litterman** sous contraintes du
mandat → **contrôle des risques** (3 essais max) → **validation humaine** (`interrupt()`) → rapport et
journal d'audit. Détails et correspondance avec le schéma : [`docs/architecture.md`](docs/architecture.md).

> Projet pédagogique : ce n'est pas un conseil en investissement.

## Arborescence

```
portfolio-agents/
├── README.md · requirements.txt · .env.example
├── config/
│   ├── settings.yaml          # réglages techniques : LLM, poids des notes, kappa, delta, tentatives…
│   └── mandat.yaml            # mandat client (H5) : 10 %/titre, 30 %/secteur, limites de risque
├── data/
│   ├── universe_sp50.csv      # 50 tickers + secteur + industrie
│   ├── portefeuille_actuel.csv
│   ├── rapports/              # documents du RAG (un dossier par ticker)
│   └── cache/                 # prix, fondamentaux, news, checkpoints (gitignore)
├── src/
│   ├── state.py               # EtatPortefeuille + NoteAnalyste / Selection (Pydantic)
│   ├── config.py · llm.py     # réglages, AppContext ; modèles rapide / fort
│   ├── graph.py               # assemblage LangGraph, lancer_cycle / reprendre
│   ├── local_runner.py        # même workflow sans LangGraph (tests, débogage)
│   ├── nodes/                 # nœuds « code » : preparation, portefeuille, validation, rapport
│   ├── agents/                # nœuds « LLM » : analystes (sous-graphe), debat (sous-graphe), prompts
│   └── tools/                 # inventaire des outils : mandat, market_data, news, rag, metrics,
│                              #   optimisation, risque, reporting
├── app/streamlit_app.py       # interface du gérant (approuver / modifier / rejeter)
├── notebooks/01_pipeline_langgraph.ipynb
├── tests/                     # fixtures/*.json + un test par étape + graphe complet
└── docs/architecture.md
```

## Démarrage

```bash
pip install -r requirements.txt

# 1) hors-ligne, sans modèle (quelques secondes)
LLM_BACKEND=mock DATA_SOURCE=synthetic pytest -q

# 2) avec LLM open source (Ollama)
ollama pull qwen2.5:3b && ollama pull qwen2.5:7b
streamlit run app/streamlit_app.py
```

Sur **Google Colab** : ouvrir `notebooks/01_pipeline_langgraph.ipynb` (GPU T4), qui installe Ollama et
déroule les 9 étapes une par une, puis le graphe complet avec la pause de validation.

## Interface du comité (Streamlit)

`streamlit run app/streamlit_app.py`, ou depuis Colab avec la dernière cellule du notebook (lien public Cloudflare).

- suivi des 9 étapes en direct pendant que les agents travaillent ;
- **Décision** : chiffres clés, lecture du risque, poids modifiables, boutons *Approuver*, *Enregistrer les ajustements*, *Rejeter* ;
- **Allocation**, **Analystes** (carte des 3 notes par titre), **Comité** (débat haussier/baissier, arbitrage du juge),
  **Risque** (limites du mandat, stress tests, comportement passé), **Note d'investissement**, **Traçabilité**.

## Note d'investissement

Après approbation, le nœud `rapport` produit `data/rapports_generes/note_investissement_<date>.html` :
un fichier autonome, mis en page pour l'A4 (*Imprimer → Enregistrer en PDF*). Huit parties : synthèse,
convictions du comité, allocation, des vues aux poids (Black-Litterman), risque, débat, ordres à passer,
méthode et traçabilité. Tous les chiffres et graphiques viennent du code ; le LLM écrit la narration (H7).

## Observabilité (LangSmith)

Définis `LANGSMITH_API_KEY` (offre gratuite) : chaque nœud, agent, outil et appel de modèle est tracé dans le
projet `portfolio-agents`, avec le `thread_id` du cycle (vue *Threads*), la date et le moteur LLM.
Les liens vers les traces apparaissent dans le notebook, l'onglet *Traçabilité* et la note d'investissement.
Sans clé, rien ne change, sauf l'absence de traces.

## Rapport LaTeX

`docs/rapport/rapport.tex` (pdflatex, ou Overleaf : importer le dossier `docs/rapport/`).
Les figures, tableaux et chiffres viennent de `python docs/rapport/generer_figures.py`
(`--backend ollama --source yfinance` pour un vrai cycle), puis recompiler.

## En équipe

- Une étape = un fichier = des clés de l'état (`src/state.py`).
- Chacun teste son nœud sur les fixtures (`tests/fixtures/`), sans faire tourner l'amont ;
  `python -m tests.make_fixtures` les régénère.
- Les réglages numériques vont dans `config/`, jamais dans le code.
