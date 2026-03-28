# 🕸️ Web Data Mining — Knowledge Graph Project

> Construction, embedding et analyse d'un graphe de connaissances à partir de données issues du web.

---

## 📋 Table des matières

- [À propos](#-à-propos)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Pipeline](#-pipeline)
- [Technologies utilisées](#-technologies-utilisées)
- [Auteurs](#-auteurs)

---

## 🧠 À propos

Ce projet s'inscrit dans le cadre d'un cours de **Web Data Mining**. L'objectif est de collecter des données depuis le web, de les structurer sous forme de **graphe de connaissances (Knowledge Graph)**, puis d'appliquer des techniques d'**embedding de graphe (KGE — Knowledge Graph Embedding)** pour en extraire des représentations vectorielles utiles à des tâches telles que la prédiction de liens ou la complétion de graphe.

---

## 📁 Structure du projet

```
Webdatamining-project/
│
├── data/               # Données brutes collectées (scraping, APIs, etc.)
├── kg_artifacts/       # Artefacts du graphe de connaissances (triplets, graphe sérialisé, etc.)
├── kge_data/           # Données formatées pour l'entraînement des modèles d'embedding
├── notebooks/          # Jupyter Notebooks d'exploration, construction et analyse
├── reports/            # Rapports, figures et résultats d'expériences
├── src/                # Code source Python (modules réutilisables)
└── README.md
```

---

## ⚙️ Installation

### Prérequis

- Python **3.8+**
- `pip` ou `conda`

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/thot0x/Webdatamining-project.git
cd Webdatamining-project

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

Les notebooks Jupyter dans le dossier `notebooks/` constituent le point d'entrée principal :

```bash
jupyter notebook notebooks/
```

Les scripts Python réutilisables se trouvent dans `src/` et peuvent être importés dans les notebooks ou exécutés directement.

---

## 🔄 Pipeline

```
Collecte de données       Construction du KG        Embedding (KGE)
  (Web scraping)    →    (Entités & relations)   →   (TransE, RotatE…)
     data/                  kg_artifacts/               kge_data/
                                                           ↓
                                                   Analyse & résultats
                                                       reports/
```

1. **Collecte** — Récupération et nettoyage des données depuis des sources web
2. **Construction du graphe** — Extraction d'entités et de relations → triplets `(sujet, prédicat, objet)`
3. **Embedding** — Entraînement d'un modèle KGE sur les triplets
4. **Évaluation** — Mesure des performances (MRR, Hits@K) et visualisation

---

## 🛠️ Technologies utilisées

| Catégorie | Outils |
|---|---|
| Langage | Python 3 |
| Notebooks | Jupyter |
| Manipulation de données | pandas, NumPy |
| Graphes | NetworkX, RDFLib |
| Knowledge Graph Embedding | PyKEEN / AmpliGraph |
| Visualisation | Matplotlib, Seaborn |
| Web scraping | requests, BeautifulSoup |

---

## 👤 Auteurs

- **Thomas WARTELLE**
- **Marcel YAMMINE**
