import pandas as pd

# On lit juste la première ligne pour voir les noms
df_test = pd.read_csv("data/test.csv", nrows=5)
print("--- NOMS DES COLONNES ---")
print(df_test.columns.tolist())

print("\n--- 5 PREMIERS IDs ---")
# Regarde la première colonne, souvent c'est celle-là
print(df_test.iloc[:, 0].head())