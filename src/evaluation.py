import matplotlib.pyplot as plt

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')

def plot_feature_importance(model, feature_names, save_path):
    """
    Génère et sauvegarde le graphique d'importance des variables.
    """
    # On réinjecte les noms des colonnes dans le modèle
    model.get_booster().feature_names = list(feature_names)
    
    plt.figure(figsize=(10, 8))
    # importance_type='weight' : combien de fois la variable coupe l'arbre
    xgb.plot_importance(model, max_num_features=15, height=0.5, 
                        importance_type='weight', title='Importance des variables (XGBoost)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graphique sauvegardé sous : {save_path}")
    # plt.show() # Décommenter si tu n'es pas en mode script pur

def check_overfitting(model, X, y, test_rmse_avg):
    """
    Compare l'erreur Train (Mémoire) vs l'erreur Test (Réalité K-Fold).
    """
    print("\n--- VERDICT SURAPPRENTISSAGE ---")
    
    # 1. Prédiction sur le Train (Ce que le modèle connait déjà)
    pred_log = model.predict(X)
    pred_euros = np.expm1(pred_log)
    y_true_euros = np.expm1(y)
    
    rmse_train = np.sqrt(mean_squared_error(y_true_euros, pred_euros))
    
    # 2. Affichage
    print(f"Erreur sur le Train (Mémoire)   : {rmse_train:.2f} €")
    print(f"Erreur sur le Test (Est. KFold) : {test_rmse_avg:.2f} €")
    
    ecart = test_rmse_avg - rmse_train
    print(f"Écart                           : {ecart:.2f} €")
    
    # 3. Diagnostic
    if ecart > 300:
        print("RISQUE : Le modèle apprend trop par cœur (Overfitting).")
    elif ecart < 0:
        print("Situation rare (Sous-apprentissage ou chance).")
    else:
        print("Le modèle généralise bien (Robuste).")