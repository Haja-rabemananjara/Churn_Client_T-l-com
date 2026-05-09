# 📊 Churn Client Télécom — Analyse & Prédiction

Projet Data Analyst complet : nettoyage, analyse exploratoire, visualisation et Machine Learning  
sur le dataset IBM Telco Customer Churn (7 043 clients).

---

## 🎯 Résultats clés

| Indicateur | Valeur |
|-----------|--------|
| Taux de churn global | **26.5%** |
| Facteur n°1 | Contrat mensuel : **42.7%** de churn vs 2.8% pour contrat 2 ans |
| Période critique | **47%** des départs ont lieu dans la 1ère année |
| Meilleur modèle | Random Forest — AUC-ROC : **~0.845** |
| Top variable | `tenure` (ancienneté) |

---

## 🗂️ Structure du projet

```
Churn_Client_Telecom/
├── data/
│   ├── raw/                    ← Dataset brut Kaggle (ne jamais modifier)
│   └── processed/              ← Données nettoyées (générées par 02_data_cleaning)
│
├── notebooks/
│   ├── 01_import.ipynb         ← Chargement et diagnostics initiaux
│   ├── 02_data_cleaning.ipynb  ← Nettoyage, encodage, sauvegarde
│   ├── 03_eda.ipynb            ← Analyse exploratoire et insights
│   ├── 04_visualisation.ipynb  ← Graphiques métier + export Power BI/Tableau
│   └── 05_machine_learning.ipynb ← Modèles ML, évaluation, feature importance
│
├── src/
│   ├── __init__.py
│   ├── cleaning.py             ← Fonctions de nettoyage réutilisables
│   ├── features.py             ← Feature engineering + préparation ML
│   └── viz.py                  ← Fonctions de visualisation
│
├── outputs/
│   ├── figures/                ← Graphiques exportés (PNG 150 dpi)
│   └── models/                 ← Modèles sauvegardés (.pkl)
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Démarrage rapide

### 1. Cloner et installer
```bash
git clone https://github.com/TON_USER/Churn_Client_Telecom.git
cd Churn_Client_Telecom
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Télécharger le dataset
Télécharger `WA_Fn-UseC_-Telco-Customer-Churn.csv` depuis  
[Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
et le placer dans `data/raw/`.

### 3. Exécuter les notebooks dans l'ordre
```bash
jupyter notebook
```
Exécuter dans cet ordre :
1. `notebooks/01_import.ipynb`
2. `notebooks/02_data_cleaning.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_visualisation.ipynb`
5. `notebooks/05_machine_learning.ipynb`

---

## 🔍 Insights principaux

### Par type de contrat
| Contrat | Taux de churn |
|---------|--------------|
| Month-to-month | 42.7% 🔴 |
| One year | 11.3% 🟡 |
| Two year | 2.8% 🟢 |

### Par ancienneté
| Tranche | Taux de churn |
|---------|--------------|
| 0-1 an | ~47% 🔴 |
| 1-2 ans | ~35% 🟡 |
| 2-4 ans | ~20% 🟡 |
| 4-6 ans | ~6% 🟢 |

### Recommandations business
1. **Contrat mensuel → annuel** : offre de passage avec remise de 10-15%
2. **Onboarding renforcé** : programme dédié sur les 6 premiers mois
3. **Fibre sans support** : offrir TechSupport gratuit 3 mois
4. **Alerte CRM** : déclencher une action dès que `proba_churn > 50%`

---

## 🛠️ Stack technique

| Domaine | Outils |
|---------|--------|
| Data Cleaning | `pandas`, `numpy` |
| EDA | `pandas`, `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Export BI | Excel multi-onglets (Power BI / Tableau) |
| Environnement | Jupyter Notebook, Git |

---

## 🔬 Dataset

**IBM Telco Customer Churn**  
7 043 clients × 21 variables  
Source : [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Variables : ancienneté, contrat, services souscrits, facturation, mode de paiement, **Churn** (cible)

---

## 🚀 Évolutions prévues

- [ ] **XGBoost / LightGBM** → améliorer l'AUC-ROC
- [ ] **SHAP values** → expliquer chaque prédiction individuelle
- [ ] **API FastAPI** → exposer le modèle en production
- [ ] **Streamlit** → dashboard interactif
- [ ] **PySpark** → passage à l'échelle Big Data
