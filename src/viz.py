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


# 1. Distribution du churn
def plot_churn_distribution(
        df: pd.DataFrame,
        col: str = "Churn",
        save_as: str = "01_churn_distribution.png",
) -> plt.Figure:
    """
    Affiche la distribution de la variable cible (camembert + barres).

    Args:
        df      : DataFrame brut (Churn en Yes/No) ou nettoyé (Churn en 0/1)
        col     : nom de la colonne cible
        save_as : nom du fichier de sortie (None pour ne pas sauvegarder)

    Returns:
        Figure matplolib
    """
    set_project_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df[col].value_counts()
    total = counts.sum()

    # Camembert
    axes[0].pie(
        counts.values,
        labels=[f"{idx}\n({value/total*100:.1f}%)" for idx, value in counts.items()],
        colors=[BLUE_MAIN, RED_CHURN],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 3},
        textprops={"fontsize": 11},
    )
    axes[0].set_title("Distribution du Churn", pad=15)

    # Barres annotées
    bar_colors=[BLUE_MAIN, RED_CHURN]
    bars = axes[1].bar(
        counts.index.astype(str), counts.values,
        color=bar_colors, edgecolor="white", linewidth=2, width=0.5,
    )
    for bar, count in zip(bars, counts.values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30,
            f"{count:,}\nClients",
            ha="center", fontsize=10, fontweight="bold",
        )
    axes[1].set_title("Nombre de clients par groupe")
    axes[1].set_ylabel("Nombre de clients")
    axes[1].set_ylim(0, counts.max() *1.2)

    fig.suptitle(
        f"Vue d'ensemble : Churn Cient Télécom (n={total:,})",
        fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, save_as)
    return fig
