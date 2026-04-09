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

def loadData(path = DATA_PATH):
    df_raw = pd.read_csv(path)

    # Premier aperçut de la forme et vérification des colonnes
    print(f'Shape : {df_raw.shape} ({df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes)')
    print(f'\nColonne ({len(df_raw.columns)}) :')
    for i, col in enumerate(df_raw.columns, 1):
        print(f"  {i:2d}. {col} ({df_raw[col].dtype})")

    return df_raw

