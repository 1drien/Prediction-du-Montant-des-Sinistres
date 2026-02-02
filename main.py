import os
import pandas as pd
import numpy as np
from src import preprocessing, severity, frequency, evaluation

def main():
    # --- CHEMINS ---
    DATA_PATH = os.path.join("data", "train.csv")
    TEST_PATH = os.path.join("data", "test.csv") # Assure-toi que test.csv est bien là
    OUTPUT_DIR = "severity_model"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_final.csv")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("=== DÉMARRAGE DU PROJET ACTUARIAT ===")

    # ---------------------------------------------------------
    # 1. ENTRAÎNEMENT DES MODÈLES
    # ---------------------------------------------------------
    df_train = preprocessing.load_and_clean_common_data(DATA_PATH)

    # --- SÉVÉRITÉ ---
    print("\n--- PARTIE A : SÉVÉRITÉ ---")
    X_sev, y_sev, feats_sev = preprocessing.prepare_for_severity(df_train)
    mae_sev, rmse_sev = severity.run_kfold_validation(X_sev, y_sev)
    model_sev = severity.train_final_model(X_sev, y_sev)
    
    # --- FRÉQUENCE ---
    print("\n--- PARTIE B : FRÉQUENCE ---")
    X_freq, y_freq, feats_freq = preprocessing.prepare_for_frequency(df_train)
    # On entraîne le modèle final (soit l'ensemble, soit le XGB calibré)
    # Pour la soumission, utilisons le modèle final simple défini dans frequency.py
    model_freq = frequency.train_final_model(X_freq, y_freq)

    # ... (le début du main reste pareil) ...

    # ---------------------------------------------------------
    # 2. PRÉDICTION SUR LE TEST
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(" PARTIE C : GÉNÉRATION DE LA SOUMISSION ")
    print("="*40)

    # a. Chargement et Nettoyage du Test
    df_test = preprocessing.load_and_clean_common_data(TEST_PATH)
    
    # b. Préparation X (Inférence) AVEC ALIGNEMENT
    # CORRECTION ICI : On passe 'feats_freq' pour forcer les mêmes colonnes que l'entraînement
    # (feats_freq et feats_sev sont identiques normalement, donc on prend l'un des deux)
    X_test, ids_test = preprocessing.prepare_for_inference(df_test, feats_freq)
    
    # c. Prédiction FRÉQUENCE
    print("Calcul des probabilités de sinistre...")
    probas_freq = model_freq.predict_proba(X_test)[:, 1]
    
    # d. Prédiction SÉVÉRITÉ
    print("Calcul des coûts estimés...")
    # Attention : model_sev attend aussi exactement les mêmes colonnes
    # Comme X_test est maintenant aligné sur feats_freq (qui est == feats_sev), ça marche.
    log_couts = model_sev.predict(X_test)
    couts_sev = np.expm1(log_couts)
    
    # ... (la fin reste pareille) ...
    
    # e. CALCUL DE LA PRIME PURE
    # Formule : Probabilité * Coût
    prime_pure = probas_freq * couts_sev
    
    # f. Création du fichier CSV
    # CORRECTION : On renomme les colonnes pour respecter le format imposé
    df_submission = pd.DataFrame({
        'index': ids_test,       # La plateforme VEUT que ça s'appelle 'index'
        'pred': prime_pure       # La plateforme VEUT que ça s'appelle 'pred'
    })
    
    # Sauvegarde
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nFichier généré avec succès : {SUBMISSION_PATH}")
    print(df_submission.head())
    
    # Sauvegarde
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n Fichier généré avec succès : {SUBMISSION_PATH}")
    print(df_submission.head())

if __name__ == "__main__":
    main()