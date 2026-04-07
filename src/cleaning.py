import pandas as pd

DATA_PATH = '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

def loadData(path = DATA_PATH):
    df_raw = pd.read_csv(path)

    # Premier aperçut de la forme et vérification des colonnes
    print(f'Shape : {df_raw.shape}')
    print(f'\nColonne ({len(df_raw.columns)}) :')
    print(df_raw.columns.to_list())

    return df_raw

if __name__ == "__main__":
    df_raw = loadData()