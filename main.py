import os
# Plus besoin de sys.path.append car on est à la racine !
from src import preprocessing, severity, evaluation

def main():
    # --- 1. CONFIGURATION DES CHEMINS ---
    # Les chemins sont maintenant relatifs à la racine du projet
    DATA_PATH = os.path.join("data", "train.csv")
    
    # On va sauvegarder les résultats dans le dossier 'severity_model'
    OUTPUT_DIR = "severity_model"
    IMG_NAME = "feature_importance_final.png"
    IMG_PATH = os.path.join(OUTPUT_DIR, IMG_NAME)

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== DÉMARRAGE DU PIPELINE SÉVÉRITÉ ===")

    # --- 2. PRÉPARATION DES DONNÉES (Cuisine) ---
    # Appel au module src/preprocessing.py
    df_clean = preprocessing.load_and_clean_common_data(DATA_PATH)
    X, y, feature_names = preprocessing.prepare_for_severity(df_clean)

    # --- 3. VALIDATION CROISÉE (Moteur) ---
    # Appel au module src/severity.py
    avg_mae, avg_rmse = severity.run_kfold_validation(X, y)
    
    print("\n" + "-"*40)
    print(f" RÉSULTATS K-FOLD (ESTIMATION) ")
    print("="*40)
    print(f"MAE Moyenne  : {avg_mae:.2f} €")
    print(f"RMSE Moyen   : {avg_rmse:.2f} €")

    # --- 4. MODÈLE FINAL (Moteur) ---
    full_model = severity.train_final_model(X, y)

    # --- 5. ÉVALUATION & GRAPHIQUES (Tableau de bord) ---
    # Appel au module src/evaluation.py
    evaluation.plot_feature_importance(full_model, feature_names, IMG_PATH)
    evaluation.check_overfitting(full_model, X, y, avg_rmse)

    print(f"\nTerminé avec succès. Graphique sauvegardé dans {IMG_PATH}")

if __name__ == "__main__":
    main()