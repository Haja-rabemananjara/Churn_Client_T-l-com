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
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

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

# 2. Churn par variable catégorielle
def plot_churn_by_category(
    df: pd.DataFrame,
    category_col: str,
    churn_col: str = "Churn_num",
    order: list= None,
    bar_colors: list = None,
    avg_line: bool= True,
    save_as: str= None,
) -> plt.Figure:
    """
    Taux de churn par modalité d'une variable catégorielle.

    Exemple: plot_churn_by_category(df_eda, 'Contract', order=['Month-to-month, 'One year', 'Two year'])

    Args:
        df          : DataFrame avec la variable catégorielle et Churn_num
        category_col: colonne de regroupement
        churn_col   : colonne numérique 0/1 du churn
        order       : ordre des barres (None = tri décroissant)
        bar_colors  : liste de couleurs (None = dégradé rouge->vert)
        avg_line    : si Truen trace la ligne de la moyenne globale
        save_as     : nom du fichier de sortie

    Returns:
        Figure matplotlib
    """

    set_project_style()
    rate = df.groupby(category_col)[churn_col].mean() * 100

    if order:
        rate = rate.reindex(order)
    else:
        rate = rate.sort_values(ascending=False)
    
    nb_cat = len(rate)
    if bar_colors is None:
        bar_colors = [RED_CHURN, AMBER, GREEN_OK, BLUE_MAIN][:nb_cat]
        if len(bar_colors) < nb_cat:
            bar_colors = [RED_CHURN] * nb_cat

    fig, ax = plt.subplots(figsize=(max(8, nb_cat * 2), 5))
    bars = ax.bar(
        rate.index.astype(str), rate.values,
        color = bar_colors, edgecolor="white", linewidth=2, width=0.5,
    )

    for bar, val in zip(bars, rate.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() +0.8,
            f"{val:.1f}%",
            ha="center", fontsize=12, fontweight="bold",
        )

    if avg_line:
        avg = df[churn_col].mean()*100
        ax.axhline(avg, color="grey", linestyle="--", linewidth =1.5, alpha=0.7)
        ax.text(
            len(rate) - 0.5, avg + 0.5,
            f"Moyenne globale: {avg:.1f}%",
            color="grey", fontsize=9,
        )

    ax.set_title(f"Taux de Churn par {category_col}", pad=15)
    ax.set_xlabel(category_col)
    ax.set_ylabel("Taux de churn (%)")
    ax.set_ylim(0, rate.max() * 1.3)
    plt.tight_layout()
    _save(fig, save_as)
    return fig


# 3. Churn vs Ancienneté

def plot_churn_vs_tenure(
        df: pd.DataFrame,
        tenure_col: str= 'tenure',
        churn_col: str='Churn',
        tenure_group_col: str= 'tenure_group',
        save_as: str='03_churn_tenure.png',
    ) -> plt.Figure:
    """
    Double graphique: histogramme empilé + taux de churn par tranche.

    Args:
        df              : DataFrame brut (avec Churn en Yes/No et tenure_group)
        tenure_col      : colonne d'ancienneté brute
        churn_col       : colonne Churn (Yes/No ou 0/1)
        tenure_group_col: Colonne de tranches d'ancienneté (créée par add_tenure_group)
        save_as         : nom du fichier de sortie

    Returns:
        Figure matplotlib
    """

    set_project_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogrammes empilé
    sns.histplot(
        data=df,
        x=tenure_col,
        hue=churn_col,
        multiple="stack",
        bins=30,
        palette=PALETTE_CHURN,
        ax=axes[0],
    )
    axes[0].set_title("Distribution de l'Ancienneté selon le Churn")
    axes[0].set_xlabel("Ancienneté (mois)")
    axes[0].set_ylabel("Nombre de clients")
    axes[0].annotate(
        "Pic de churn\n(mois 1-6)",
        xy=(3, 200), xytext=(15, 320),
        arrowprops=dict(arrowstyle="->", color=RED_CHURN),
        color=RED_CHURN, fontsize=9, fontweight="bold",
    )

    # Taux par tranche
    if tenure_group_col in df.columns:
        _order      = ["0-1 an", "1-2 ans", "2-4 ans", "4-6 ans"]
        _raw        =   np.asarray(df[churn_col])
        _churn_arr  = (_raw == "Yes").astype(np.int8) \
                     if _raw.dtype.kind in ("U", "O", "S") \
                     else (_raw != 0).astype(np.int8)
        _tag_arr    = df[tenure_group_col].astype(str).to_numpy()
        _rate_dict  = {
            lbl: _churn_arr[_tag_arr == lbl].mean() * 100
            for lbl in _order
            if (_tag_arr == lbl).sum() > 0
        }
        rate = pd.Series(_rate_dict)
        t_colors = [RED_CHURN, AMBER, GREEN_OK, GREEN_OK]
        bars = axes[1].bar(
            rate.index.astype(str), rate.values,
            color=t_colors[:len(rate)], edgecolor="white", linewidth=2, width=0.5,
        )
        for bar, val in zip(bars, rate.values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", fontsize=12, fontweight="bold",
            )
        axes[1].set_title("Taux de Churn par tranche d'Ancienneté")
        axes[1].set_xlabel("Ancienneté")
        axes[1].set_ylabel("Taux de churn (%)")
        axes[1].set_ylim(0, rate.max() * 1.3)

    plt.tight_layout()
    _save(fig, save_as)
    return fig

# 4. Heatmap de corrélation

def plot_correlation_heatmp(
        df: pd.DataFrame,
        cols: list = None,
        rename_map: dict = None,
        save_as: str='04_Corrélation_heatmap.png',
) -> plt.Figure:
    """
    Heatmap de corrélation (triangle inférieur uniquement)

    df doit contenur uniquement des colonnes numériques.

    Args:
        df          : DataFrame avce colonnes numériques uniquement
        cols        : sous-ensemble de colonnes (None = toutes)
        rename_map  : dict pour renommer les colonnes sur le graphique
        save_as     : nom du fichier de sortie

    Returns:
        Figure matplotlib
    """

    set_project_style()
    data = df[cols].copy() if cols else df.copy()

    if rename_map:
        data = data.rename(columns=rename_map)

    corr = data.corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=2,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink":0.8}
    )
    ax.set_title("Matrice de Corrélation - Variables clés", pad=15)
    plt.tight_layout()
    _save(fig, save_as)
    return fig

# 5. Comparaison des modèles ML

def plot_roc_curves(
    y_test,
    models_probs: dict,
    save_as: str = "05_roc_curves.png"
) -> plt.Figure:
    """
    Courbes ROC pour comparer plusieurs modèles.

    Args:
        y_test      : labels réels (0/1)
        models_probs: dict {nom_modèle: probabilité_prédites}
                        ex: {'Logistic Regression': y_prob_lr, 'Random Forest': y_prob_rf}
        save_as     : nom du fichier de sortie

    Returns:
        Figure matplotlib
    """
    set_project_style()
    colors = [AMBER, BLUE_MAIN, GREEN_OK, RED_CHURN]
    fig, ax = plt.subplots(figsize=(8, 6))

    for (name, y_prob), color in zip(models_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aléatoire (AUC=0.5)")
    ax.set_title("Courbes ROC — Comparaison des modèles", pad=15)
    ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)")
    ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save(fig, save_as)
    return fig

def plot_confusion_matrices(
    y_test,
    models_preds: dict,
    save_as: str="06_confusion_matrices.png"
) -> plt.Figure:
    """
    Matrices de confusion pour plusieurs modèles, côte à côte.

    Args:
        y_test      : labels réels
        models_preds: dict {nom_modèle: prédictions}
        save_as     : nom du fichier

    Returns:
        Figure matplotlib
    """
    set_project_style()
    n = len(models_preds)
    cmaps = ["Blues", "Reds", "Greens"]
    fig, axes = plt.subplots(1, n, figsize=(6 *n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (name, y_pred), cmap in zip(axes, models_preds.items(), cmaps):
        cm  = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=["Prédit Non-Churn", "Prédit Churn"],
            yticklabels=["Réel Non-Churn", "Réel Churn"],
        )
        ax.set_title(f"Confusion Matrix\n{name}")
    
    plt.tight_layout()
    _save(fig, save_as)
    return fig