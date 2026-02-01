# Modèle de Tarification Assurance Auto (Severity & Frequency)

![Status](https://img.shields.io/badge/Status-En%20Cours-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

Projet de Data Science visant à prédire la **Prime Pure** d'assurance automobile.
Le projet est divisé en deux modules principaux : la Sévérité (Coût moyen des sinistres) et la Fréquence (Nombre de sinistres).

👤 **Auteur :** [@1drien](https://github.com/1drien)

---

## 📂 Architecture du projet

Le code est structuré de manière modulaire :

```text
.
├── data/                 # Fichiers CSV (train.csv, test.csv)
├── severity_model/       # Dossier de sortie (Graphiques, Logs)
├── src/                  # Code Source
│   ├── preprocessing.py  # Nettoyage et préparation des données
│   ├── severity.py       # Modèle XGBoost et Validation Croisée
│   └── evaluation.py     # Graphiques et analyse de surapprentissage
└── main.py               # Point d'entrée principal (Lancer ce fichier)
```
