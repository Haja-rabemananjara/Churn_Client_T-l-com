"""
src/features.py

Feature engineering : création et sélection de variables pour le ML.

Ces fonctions enrichissent le DataFrame nettoyé avec des variables dérivées qui peuvent améliorer les performances des modèles.

Usage dans les notebooks:
    from src.features import {functions}
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Variables dérivées métier

def add_tenure_group(df: pd.DataFrame, col: str='tenure') -> pd.DataFrame:
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

    df = df.copy()
    df['tenure_group'] = pd.cut(
        df[col],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1 an', '1-2 ans', '2-4 ans', '4-6 ans'],
        include_lowest=True
    )
    return df



def add_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df.copy()

    # Conversion défensive: TotalCharges peut être string dans df_eda (données brutes)
    # Les espaces ' ' (clients tenure=0) deviennent NaN puis 0
    total = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce").fillna(0)
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    
    df["charge_ratio"] = np.where(monthly > 0, total / monthly, 0)
    return df



def add_is_new_customer(df: pd.DataFrame, thereshold: int=6) -> pd.DataFrame:
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

    df = df.copy()
    df['is_new_customer'] = (df['tenure'] <= thereshold).astype(int)

    return df



def add_nb_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compte le nombre de services additionnels souscrits par le client.

    Services comptés: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

    Hypothèse: un client multi-services est plus engagé et moins susceptible de churner (à confimer avec l'EDA)

    Args: df: DataFrame après encode_bin_columns (colonnes en 0/1)

    Returns:
        DataFrame avec colonne 'nb_services' (entier 0-6)
    """

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    existing = [column for column in service_cols if column in df.columns]
    df = df.copy()
    df['nb_services'] = df[existing].sum(axis=1)

    return df

## 2. Préparation pour le Machine Learning

# Colonnes numériques à normaliser
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def prepare_ml_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> dict:
    """
    Préparation des données pour l'entraînement ML.

    Etapes :
        1. Séparation features (X) / cible (y)
        2. Train/test split stratifié (préserve le ratio churners/non-churners)
        3. Normalisation StandardScaler sur les features numériques
            IMPORTANT : fit sur train uniquement -> évite le data leakage

    Args:
        df          : DataFrame nettoyé (sorti de run_cleaning_pipeline)
        target      : nom de la colonne cible (défaut : 'Churn')
        test_size   : proportion du jeu de test (défaut : 0.2 = 20%)
        random_state: graine aléatoire pour la reproductibilité
        scale       : si True, applique StandardScaler sur les colonnes numériques

    Returns:
        dict avec clés: X_train, X_test, y_train, y_test, scaler, feature_names
        Le scaler est None si scale=False
    """

    # Séparation x / y
    X = df.drop(target, axis=1)
    y = df[target]

    # Colonnes numériques présentes dans X
    num_cols_present = [col for col in NUMERICAL_COLS if col in X.columns]

    # Split stratifié
    # stratify = y : garantit la même proportion de churners (26.5%) dansz train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = None
    if scale and num_cols_present:
        scaler = StandardScaler()
        # fit_transform sur train, transform uniquement sur test
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[num_cols_present] = scaler.fit_transform(X_train[num_cols_present])
        X_test[num_cols_present] = scaler.transform(X_test[num_cols_present])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }

# 3. Prédiction sur un nouveau client

def predict_new_cient(
    client_dict: dict,
    model,
    scaler,
    feature_names: list,
    numerical_cols: list,
    threshold: float = 0.5,
) -> dict:
    """
    Prédit le risque de churn pour un nouveau client non vu.

    Args:
        client_dict     : dict des caractéristiques du client (valeurs brut)
        model           : modèle sklearn entraîné (avec .predict_proba)
        scaler          : StandardScaler ajusté sur le train
        feature_names   : liste des features dans l'ordre du modèle
        numerical_cols  : colonnes à normaliser (défaut : NUMERICAL_COLS)
        threshold       : seuil de décision (défaut : 0.5)

    Returns:
        dict avec 'proba', 'prediction', 'risk_label'
    """
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS
    
    # Construire le DataFrame dans le bon ordre de colonnes
    client_df = pd.DataFrame([client_dict])
    client_df = client_df.reindex(columns=feature_names, fill_value=0)

    # Normaliser les colonnes numériques présentes
    num_present = [col for col in numerical_cols if col in client_df.columns]
    if scaler is not None and num_present:
        client_df[num_present] = scaler.transform(client_df[num_present])

    proba = model.predict_proba(client_df)[0][1]

    return {
        "proba": round(float(proba), 4),
        "prediction": int(proba >= threshold),
        "risk_label": "RISQUE DE CHURN" if proba >= threshold else "Client STABLE",
    }