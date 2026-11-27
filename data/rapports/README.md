# Documents pour le RAG (`rechercher_rapport()`)

Un dossier par ticker, des fichiers `.txt`, `.md`, `.html` ou `.pdf` :

```
data/rapports/
├── AAPL/
│   └── 10-K_2025.txt
└── MSFT/
    └── 10-K_2025.pdf
```

Téléchargement automatique des rapports annuels américains (SEC EDGAR, gratuit) :

```python
from src.tools.rag import telecharger_rapport_sec
telecharger_rapport_sec("AAPL", email="prenom.nom@ecole.ma")   # la SEC exige un e-mail dans le User-Agent
```

Pour le marché marocain : déposer ici les PDF publiés par l'AMMC ou les émetteurs.
Sans document, l'agent fondamental travaille uniquement à partir des métriques.
