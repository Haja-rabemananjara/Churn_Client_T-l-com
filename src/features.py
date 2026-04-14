"""
src/features.py

Feature engineering : création et sélection de variables pour le ML.

Ces fonctions enrichissent le DataFrame nettoyé avec des variables dérivées qui peuvent améliorer les performances des modèles.

Usage dans les notebooks:
    from src.features import {functions}
"""

import pandas as pd
import numpy as np


# 1. Variables dérivées métier
"""
Créer une variable catégorielle groupant l'ancienneté en 4 tranches métier.

Tranches choisies selon l'analyse EDA:
    - 0-1 an    : période critique (taux de churn)
    - 1-2 ans   : période de transition
    - 2-4 ans   : clients stabilisés
    - 4-6 ans   : clients très fidèles

Args:
    df: DataFrame (brut ou nettoyé)
    col: colonne d'ancienneté (défaut: 'tenure')

Returns:
    DataFrame avec une colonne 'tenure_group' ajoutée
"""

def add_tenure_group(df: pd.DataFrame, col: str='tenure') -> pd.DataFrame:

    df = df.copy()
    df['tenure_group'] = pd.cut(
        df[col],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1 an', '1-2 ans', '2-4 ans', '4-6 ans'],
        include_lowest=True
    )
    return df

