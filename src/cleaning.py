"""
cleaning.py

Il s'agit des fonctions de nettoyage et de prépration des données.

Principe : chaque fonction est pure (prend un DataFrame, retourne un DataFrame)
et ne modifie jamais les données brutes en place.

Usage dans les notebooks:
    from src.cleaning import {nom des fonctions}
"""

import pandas as pd
import numpy as np

DATA_PATH = '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Prélude. Chargement de la data
def loadData(path = DATA_PATH):
    df_raw = pd.read_csv(path)

    # Premier aperçut de la forme et vérification des colonnes
    print(f'Shape : {df_raw.shape} ({df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes)')
    print(f'\nColonne ({len(df_raw.columns)}) :')
    for i, col in enumerate(df_raw.columns, 1):
        print(f"  {i:2d}. {col} ({df_raw[col].dtype})")

    return df_raw

# 1. Correction de TotalCharges


def fix_TotalCharges(df: pd.DataFrame) -> pd.DataFrame:
    """
    On va corriger la colonne TotalCharges stockée en string dans le dataset brut.

    Problème : les clients avec tenure=0 ont un espace ' ' au lieu d'une valeur
    numérique → pandas charge la colonne entière comme object.

    Correction:
        1. Remplace les espaces par NaN
        2. Convertir en float
        3. Impute les NaN avec la médiane

    Args:
        df : DataFrame contenant la colonne 'TotalCharges'

    Returns:
        DataFrame avec TotalCharges en float64, sans valeur manquante
    """

    df = df.copy()

    # Etape 1: espace en NaN
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

    # Etape 2: conversion numérique (errors='coerce' en NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Etape 3: imputation par la médiane
    # On va utiliser la médiane et non la moyenne car MonthlyCharges est légèrement skewed à droite; la médiane est plus robuste aux valeurs extrêmes.
    mediane = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(mediane)

    return df

# 2. Encodage de la variable cible

def encode_target(df: pd.DataFrame, col: str = "Churn") -> pd.DataFrame:
    """
    Encoder la variable cible binaire Yes/No en 1/0

    Args:
        df: DataFrame source
        col: nom de la colonne cible (défaut : 'Churn')

    Returns:
        DataFrame avec la colonne cible en int (1 churner, 0 = fidèle)
    """
    
    df = df.copy()
    df[col] = (df[col] == 'Yes').astype(int)

    return df

# 3. encodage des colonnes binaires

# Colonnes avec uniquement Yes/No
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

# Colonnes avec Yes / No / "No phone service" / "No internet service"
BINARY_SERVICES = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

BINARY_MAP = {
    'Yes': 1,
    'No': 0,
    'No internet service': 0,
    'No phone service':0,
}

def encode_bin_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode toutes les colonnes binaires Yes/No en 0/1.

    - BINARY_YES_NO : colonnes à 2 modalités strictes
    - BINARY_SERVICE : colonnes pouvant contenir 'No phone/internet service'
    - gender : Male→1, Female→0

    Args:
        df : DataFrame après fix_TotalCharges et encode_target

    Returns:
        DataFrame avec colonnes binaires en int
    """
    
    df = df.copy()
    
    for col in BINARY_YES_NO:
        df[col] = (df[col] == 'Yes').astype(int)

    for col in BINARY_SERVICES:
        df[col] = df[col].map(BINARY_MAP)

    df['gender'] = (df['gender'] == 'Male').astype(int)

    return df

# 4. One-Hot Encoding des colonnes nominales
NOMINAL_COLS = ["InternetService", "Contract", "PaymentMethod"]

def encode_nominal_cols(
        df: pd.DataFrame,
        cols: list = None,
        drop_first: bool = False,
    ) -> pd.DataFrame:
    """
    Applique le One-Hot Encoding aux colonnes nominales (plusieurs modalités).

    drop_first = False par défaut :
        - Meilleure interprétabilité (on voit toutes les modalités)
        - Nécessaire pour Random Forest (les arbres gèrent la muticolinéarité)
        - A passer à True si on utilise uniquement la Régression Logistique

    Args:
        df          : DataFrame après encode_binary_cols
        cols        : liste des colonnes à encoder (défaut: NOMINAL_COLS)
        drop_first  : si True, supprime la première modalité (évite multicolinéarité)

    Returns:
        DataFrame avec colonnes dummies en int
    """

    if cols is None:
        cols = NOMINAL_COLS

    df = df.copy()
    df = pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)
    return df

# 5. Pipeline complet
def run_cleaning_pipeline(
        df_raw: pd.DataFrame,
        drop_id: bool = True,
) -> pd.DataFrame:
    """
    Exécute le pipeline de nettoyage complet dans le bon ordre.

    Ordre des étapes:
        1. fix_totalCharge          -> correction type + imputation
        2. encode_target            -> Churn Yes/No en 1/0
        3. encode_bin_columns       -> conversion de toutes les colonnes binaires
        4. encode_nominal_cols      -> OHE sur les colonnes nominales
        5. drop customerID          -> identifiant non informatif pour le ML

    Args:
        df_raw  : DataFrame brut chargé depuis data/raw/
        drop_id : si True (défaut), supprime la colonne customerID

    Returns:
        DataFrame nettoyé, prêt pour EDA et ML
    """
    
    df = fix_TotalCharges(df_raw)
    df = encode_target(df)
    df = encode_bin_columns(df)
    df = encode_nominal_cols(df)

    if drop_id and "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    
    return df

# 6. Validation post-cleaning
def validate_clean_df(df:pd.DataFrame) -> bool:
    """
    Vérifier que le DataFrame nettoyé soit prêt pour le ML.

    Contrôles effectués:
        - Aucune valeur manquante
        - Aucune colonne de type object restante
        - Colonne 'Churn' présente avec uniquement 0 et 1

    Args:
        df: DataFrame après run_cleaning_pipeline

    Returns:
        True si toutes les vérifications passent, lève une AssertionError sinon
    """

    nan_count = df.isnull().sum().sum()
    assert nan_count == 0, f"{nan_count} valeurs manquantes détectées."

    object_cols = df.select_dtypes(include='object').columns.to_list()
    assert len(object_cols) == 0, f"Colonnes objet restantes : {object_cols}"

    assert 'Churn' in df.columns, f"Colonne 'Churn' manquante"
    assert set(df['Churn'].unique()).issubset({0, 1}), f"Churn contient des valeurs != 0/1"

    print(f"Validation OK - Shape : {df.shape} | Churn rate : {df['Churn'].mean()*100:.1f}%")

    return True