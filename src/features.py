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

"""
Créer le ratio charges totales / charges mensuelles.

Interpréation métier:
    - Valeur attendue = tenure (nombre de mois)
    - Un ratio anormalement bas peut indiquer des remises ou erreurs de facturation
    - Utile pour détercter des patterns de départ prématuré

Args:
    df: DataFrame contenant 'TotalCharges' et 'MonthlyCharges' en float

Returns:
    DataFrame avec une colonne 'charge_ratio' ajoutée
"""

def add_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['charge_ratio'] = np.where(
        df['MonthlyCharges'] > 0,
        df['TotalCharges'] /df['MonthlyCharges'],
        0,
    )
    return df

"""
Flag binaire: client récent (tenure <= threshold mois)

Justification EDA : les 6 premiers mois sont la période la plus à risque.
Cette variable booléenne est plus directement interprétable pour un modèleL.

Args:
    df          : DataFrame contenant 'tenure'
    threshold   : seuil en mois (défaut: 6)

Returns:
    DataFrame avec colonne 'is_new_customer' (1 si nouveau, 0 sinon)
"""

def add_is_new_customer(df: pd.DataFrame, thereshold: int=6) -> pd.DataFrame:

    df = df.copy()
    df['is_new_customer'] = (df['tenure'] <= thereshold).astype(int)

    return df

"""
Compte le nombre de services additionnels souscrits par le client.

Services comptés: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Hypothèse: un client multi-services est plus engagé et moins susceptible de churner (à confimer avec l'EDA)

Args: df: DataFrame après encode_bin_columns (colonnes en 0/1)

Returns:
    DataFrame avec colonne 'nb_services' (entier 0-6)
"""

def add_nb_services(df: pd.DataFrame) -> pd.DataFrame:
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    existing = [column for column in service_cols if column in df.columns]
    df = df.copy()
    df['nb_services'] = df[existing].sum(axis=1)

    return df