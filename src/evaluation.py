import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

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

def plot_frequency_metrics(y_true, y_prob, save_path):
    """
    Affiche la courbe de Calibration et la courbe ROC.
    
    """
    plt.figure(figsize=(12, 5))
    
    # 1. Calibration Curve (Fiabilité)
    plt.subplot(1, 2, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Modèle', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Idéal')
    plt.xlabel('Probabilité Prédite')
    plt.ylabel('Fréquence Réelle')
    plt.title('Qualité de la Calibration')
    plt.legend()
    plt.grid(True)
    
    # 2. ROC Curve (Puissance de classement)
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrais Positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graphiques Fréquence sauvegardés sous : {save_path}")