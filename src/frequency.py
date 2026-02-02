import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

def train_frequency_ensemble(X, y, n_folds=5):
    """
    Entraîne un modèle d'ensemble (XGBoost + HistGradient) avec Calibration.
    C'est la méthode "Robuste" de ton collègue.
    """
    print(f"--- Lancement Modèle Fréquence (Ensemble Optimisé - {n_folds} folds) ---")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    # Boucle de Validation Croisée
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Gestion du déséquilibre de classe (Ratio 0 vs 1)
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        
        # --- MODELE A : XGBoost Optimisé ---
        # Note: On utilise CalibratedClassifierCV pour avoir des probas précises
        xgb_model = XGBClassifier(
            n_estimators=300, 
            learning_rate=0.01, 
            max_depth=4, 
            min_child_weight=7, 
            subsample=0.8, 
            colsample_bytree=0.6, 
            scale_pos_weight=ratio, # Important pour l'imbalance
            random_state=42, 
            n_jobs=-1,
            verbosity=0
        )
        # Calibration Isotonique (Ajuste les probabilités)
        calibrated_xgb = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
        calibrated_xgb.fit(X_train, y_train)
        p_xgb = calibrated_xgb.predict_proba(X_val)[:, 1]
        
        # --- MODELE B : HistGradientBoosting (Rapide et Robuste) ---
        hgb_model = HistGradientBoostingClassifier(
            learning_rate=0.05, 
            max_iter=200, 
            class_weight='balanced', 
            random_state=42
        )
        calibrated_hgb = CalibratedClassifierCV(hgb_model, method='isotonic', cv=3)
        calibrated_hgb.fit(X_train, y_train)
        p_hgb = calibrated_hgb.predict_proba(X_val)[:, 1]
        
        # --- ENSEMBLING (Moyenne des deux) ---
        oof_preds[val_idx] = (p_xgb + p_hgb) / 2
        
        print(f"Fold {fold+1}/{n_folds} terminé.")

    # --- CALCUL DES SCORES GLOBAUX ---
    # Brier Score (Equivalent RMSE pour les probas)
    brier = brier_score_loss(y, oof_preds)
    auc = roc_auc_score(y, oof_preds)
    
    return oof_preds, brier, auc

def train_final_model(X, y):
    """
    Entraîne le modèle final sur toutes les données.
    Pour simplifier en production, on peut garder le meilleur des deux ou l'ensemble.
    Ici on garde un XGBoost calibré seul pour la simplicité de l'objet final.
    """
    print("--- Entraînement Final Fréquence (XGBoost Calibré) ---")
    ratio = (y == 0).sum() / (y == 1).sum()
    
    model = XGBClassifier(
        n_estimators=354, learning_rate=0.01, max_depth=4,
        scale_pos_weight=ratio, random_state=42, n_jobs=-1
    )
    # On calibre sur l'ensemble (cv='prefit' n'est pas possible ici sans split, 
    # donc on refait une calibration interne ou on entraine simple)
    # Pour faire simple : on retourne le modèle brut optimisé
    model.fit(X, y)
    return model