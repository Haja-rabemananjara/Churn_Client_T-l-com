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

# 1. Chargement de la data
def loadData(path = DATA_PATH):
    df_raw = pd.read_csv(path)

    # Premier aperçut de la forme et vérification des colonnes
    print(f'Shape : {df_raw.shape} ({df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes)')
    print(f'\nColonne ({len(df_raw.columns)}) :')
    for i, col in enumerate(df_raw.columns, 1):
        print(f"  {i:2d}. {col} ({df_raw[col].dtype})")

    return df_raw

# 2. Correction de TotalCharges
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

def fix_TotalCharges(df: pd.DataFrame) -> pd.DataFrame:

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

# 3. Encodage de la variable cible
"""
Encoder la variable cible binaire Yes/No en 1/0

Args:
    df: DataFrame source
    col: nom de la colonne cible (défaut : 'Churn')

Returns:
    DataFrame avec la colonne cible en int (1 churner, 0 = fidèle)
"""
def encode_target(df: pd.DataFrame, col: str = "Churn") -> pd.DataFrame:
    df = df.copy()
    df[col] = (df[col] == 'Yes').astype(int)

    return df

