import pandas as pd
import numpy as np

# Option pour éviter le warning futur de Pandas (Downcasting)
pd.set_option('future.no_silent_downcasting', True)

def load_and_clean_common_data(filepath):
    """
    ÉTAPE 1 : Nettoyage COMMUN (Sévérité & Fréquence).
    Style "Homemade" : Transformations ligne par ligne explicites.
    """
    print(f"--- Chargement des données : {filepath} ---")
    df = pd.read_csv(filepath)

    # 1. Gestion des valeurs manquantes (NaN)
    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].fillna('Inconnu')

    # 2. Encodage Manuel (Mapping) - Ligne par ligne comme tu préfères
    # J'utilise "df[col] = ..." au lieu de "inplace=True" car c'est la nouvelle norme Pandas,
    # mais c'est exactement la même logique que ton code original.

    if 'sex_conducteur1' in df.columns:
        df['sex_conducteur1'] = df['sex_conducteur1'].replace(['M', 'F'], [0, 1]).infer_objects(copy=False)

    if 'type_vehicule' in df.columns:
        df['type_vehicule'] = df['type_vehicule'].replace(['Tourism', 'Commercial'], [0, 1]).infer_objects(copy=False)

    if 'utilisation' in df.columns:
        df['utilisation'] = df['utilisation'].replace(['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4]).infer_objects(copy=False)

    if 'freq_paiement' in df.columns:
        df['freq_paiement'] = df['freq_paiement'].replace(['Monthly', 'Quarterly', 'Biannual', 'Yearly'], [0, 1, 2, 3]).infer_objects(copy=False)

    # 3. One-Hot Encoding (Pour les variables sans ordre logique : Marque, Modèle...)
    cols_to_encode = [
        'marque_vehicule', 'modele_vehicule', 'sex_conducteur2', 
        'type_contrat', 'paiement', 'conducteur2', 'essence_vehicule'
    ]
    # On vérifie que les colonnes existent avant d'encoder pour éviter les bugs
    exist_cols = [c for c in cols_to_encode if c in df.columns]
    
    # dtype=int permet d'avoir des 0 et 1 au lieu de False/True
    df = pd.get_dummies(df, columns=exist_cols, dtype=int)
    
    print(f"Dataframe nettoyé (Taille : {df.shape})")
    return df


def prepare_for_severity(df):
    """
    ÉTAPE 2 (SPÉCIFIQUE SÉVÉRITÉ) :
    - On ne garde que les sinistres > 0.
    - On transforme le montant en Log.
    - On retire les colonnes inutiles (ID, Index...).
    """
    print("--- Préparation pour le modèle de Sévérité ---")
    
    # 1. FILTRE : On supprime ceux qui n'ont pas eu d'accident
    # Ton code original : df = df.drop(df[df['nombre_sinistres'] == 0].index)
    # Ma version (plus rapide) : on garde ceux > 0
    df_sev = df[df['nombre_sinistres'] > 0].copy()
    
    # 2. CIBLE (y) : Log transformation
    y = np.log1p(df_sev['montant_sinistre']).values
    
    # 3. FEATURES (X) : Suppression des colonnes inutiles
    # C'est ici qu'on retire l'index et les IDs pour éviter le surapprentissage
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    
    # errors='ignore' permet de ne pas planter si une colonne n'existe pas
    X_df = df_sev.drop(columns=cols_to_drop, errors='ignore')
    
    # Sauvegarde des noms
    feature_names = X_df.columns
    
    # Conversion en Numpy
    X = X_df.values
    
    print(f"Données prêtes pour Sévérité : {X.shape[0]} lignes.")
    return X, y, feature_names


def prepare_for_frequency(df):
    """
    ÉTAPE 2 BIS (SPÉCIFIQUE FRÉQUENCE) - Pour plus tard.
    """
    pass