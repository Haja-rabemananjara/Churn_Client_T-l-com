"""
Fonctions de visualisation réutilisables pour le projet Churn.

Principe: chaque fonction produit UN graphique, répond à UNE question,
et peut sauvegarder automatique dans outputs/figures/.

Palette officielle du projet :
    BLUE_MAIN  #1A56DB  → non-churners, bonne performance
    RED_CHURN  #EF4444  → churners, risque
    GREEN_OK   #10B981  → positif, validation
    AMBER      #F59E0B  → intermédiaire, attention
    NAVY       #0F2B5B  → titres, fond

Usage dans les notebooks:
    fomr src.viz import {fonctions}
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

# Palette & style
BLUE_MAIN = "#1A56DB"
RED_CHURN = "#EF4444"
GREEN_OK  = "#10B981"
AMBER     = "#F59E0B"
NAVY      = "#0F2B5B"

PALETTE_CHURN = {"No": BLUE_MAIN, "Yes": RED_CHURN}

def set_project_style():
    """Configuration du style matplotlib global pour le projet"""
    
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _save(fig: plt.Figure, filename: str, output_dir: str = "../outputs/figures"):
    """Sauvegarder un graphique si filename est fourni"""

    if filename:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi= 150, bbox_inches="tight")
        print(f"Sauvegarder: {path}")

