import pandas as pd
import numpy as np

# Option pour éviter le warning futur de Pandas
pd.set_option('future.no_silent_downcasting', True)

def load_and_clean_common_data(filepath):
    """
    ÉTAPE 1 : Nettoyage & Feature Engineering (Commun).
    """
    print(f"--- Chargement des données : {filepath} ---")
    df = pd.read_csv(filepath)

    # --- A. FEATURE ENGINEERING (L'apport de ton collègue) ---
    # On le fait AVANT le nettoyage pour avoir les colonnes brutes
    
    # 1. Ratio Puissance/Poids
    if 'din_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_puissance_poids'] = df['din_vehicule'] / (df['poids_vehicule'] + 1)
        
    # 2. Intensité d'usage (Vitesse / Poids)
    if 'vitesse_vehicule' in df.columns and 'poids_vehicule' in df.columns:
        df['ratio_vitesse_poids'] = df['vitesse_vehicule'] / (df['poids_vehicule'] + 1)
        
    # 3. Expérience relative
    if 'age_conducteur1' in df.columns and 'anciennete_permis1' in df.columns:
        df['age_obtention_permis'] = df['age_conducteur1'] - df['anciennete_permis1']
        
    # 4. Score "Jeune Sportif"
    if 'age_conducteur1' in df.columns and 'din_vehicule' in df.columns:
        df['risque_jeune_sportif'] = (1 / (df['age_conducteur1'] + 1)) * df['din_vehicule']

    # 5. Log Prix (utile pour les modèles linéaires/boostés)
    if 'prix_vehicule' in df.columns:
        df['log_prix_vehicule'] = np.log1p(df['prix_vehicule'])


    # --- B. NETTOYAGE CLASSIQUE ---

    if 'sex_conducteur2' in df.columns:
        df['sex_conducteur2'] = df['sex_conducteur2'].fillna('NoDriver')

    # Encodage Manuel
    if 'sex_conducteur1' in df.columns:
        df['sex_conducteur1'] = df['sex_conducteur1'].replace(['M', 'F'], [0, 1]).infer_objects(copy=False)
    if 'type_vehicule' in df.columns:
        df['type_vehicule'] = df['type_vehicule'].replace(['Tourism', 'Commercial'], [0, 1]).infer_objects(copy=False)
    if 'utilisation' in df.columns:
        df['utilisation'] = df['utilisation'].replace(['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4]).infer_objects(copy=False)
    if 'freq_paiement' in df.columns:
        df['freq_paiement'] = df['freq_paiement'].replace(['Monthly', 'Quarterly', 'Biannual', 'Yearly'], [0, 1, 2, 3]).infer_objects(copy=False)

    # One-Hot Encoding
    cols_to_encode = [
        'marque_vehicule', 'modele_vehicule', 'sex_conducteur2', 
        'type_contrat', 'paiement', 'conducteur2', 'essence_vehicule'
    ]
    exist_cols = [c for c in cols_to_encode if c in df.columns]
    df = pd.get_dummies(df, columns=exist_cols, dtype=int)
    
    print(f"Dataframe nettoyé (Taille : {df.shape})")
    return df


def prepare_for_severity(df):
    """ ÉTAPE 2 : SÉVÉRITÉ (Coût > 0) """
    print("--- Préparation pour Sévérité ---")
    df_sev = df[df['nombre_sinistres'] > 0].copy()
    y = np.log1p(df_sev['montant_sinistre']).values
    
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', 
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    X_df = df_sev.drop(columns=cols_to_drop, errors='ignore')
    return X_df.values, y, X_df.columns


def prepare_for_frequency(df):
    """
    ÉTAPE 2 BIS : FRÉQUENCE (Probabilité d'avoir au moins 1 accident).
    """
    print("--- Préparation pour Fréquence ---")
    
    # 1. CIBLE : 0 si rien, 1 si au moins un sinistre
    # On garde TOUT LE MONDE (contrairement à la sévérité)
    y = (df['nombre_sinistres'] > 0).astype(int).values
    
    # 2. FEATURES : On enlève les infos de cible et d'ID
    cols_to_drop = [
        'montant_sinistre', 'nombre_sinistres', # On ne doit pas connaitre le résultat !
        'id_contrat', 'id_client', 'id_vehicule', 'code_postal', 'index'
    ]
    
    X_df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # On sauvegarde les noms
    feature_names = X_df.columns
    
    return X_df.values, y, feature_names

def prepare_for_inference(df, model_features):
    print("--- Préparation des données de TEST (Inférence) ---")
    
    # --- CORRECTION ICI ---
    # Ajoute le nom que tu as trouvé dans la liste (ex: 'index' ou 'id')
    if 'id_contrat' in df.columns:
        ids = df['id_contrat'].values
    elif 'id_client' in df.columns:
        ids = df['id_client'].values
    elif 'index' in df.columns:  # <--- Ajoute ça si la colonne s'appelle 'index'
        ids = df['index'].values
    elif 'id' in df.columns:     # <--- Ajoute ça si elle s'appelle 'id'
        ids = df['id'].values
    else:
        # En dernier recours, on prend la 1ère colonne du fichier
        # (Souvent l'ID est la toute première colonne)
        ids = df.iloc[:, 0].values

    # ... la suite de la fonction reste pareille ...
    df_aligned = df.reindex(columns=model_features, fill_value=0)
    return df_aligned.values, ids