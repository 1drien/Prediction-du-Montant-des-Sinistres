import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Tes paramètres stabilisés (Anti-Surapprentissage)
PARAMS = {
    'objective': 'reg:squarederror',
    'booster': 'gbtree',
    'verbosity': 0,
    'learning_rate': 0.05,   
    'n_estimators': 300,     
    'max_depth': 3,          
    'min_child_weight': 5,   
    'subsample': 0.7,        
    'colsample_bytree': 0.7, 
    'reg_alpha': 0.1,        
    'reg_lambda': 1.0        
}

def run_kfold_validation(X, y, n_splits=5):
    """
    Lance la validation croisée (K-Fold).
    Retourne : MAE moyen, RMSE moyen (sur le Test).
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=None)    
    mae_scores = []
    rmse_scores = []

    print(f"--- Lancement du K-Fold ({n_splits} splits) ---")

    for train_index, val_index in kf.split(X):
        # 1. Séparation (Numpy arrays)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 2. Entraînement
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(X_train, y_train)
        
        # 3. Prédiction (Log)
        pred_log = model.predict(X_val)

        # 4. Conversion (Log -> Euros)
        # np.expm1 est l'inverse exact de np.log1p (équivalent à exp(x) - 1)
        pred_euros = np.expm1(pred_log)
        y_true_euros = np.expm1(y_val)

        # 5. Scores
        mae_scores.append(mean_absolute_error(y_true_euros, pred_euros))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true_euros, pred_euros)))

    return np.mean(mae_scores), np.mean(rmse_scores)


def train_final_model(X, y):
    """
    Entraîne le modèle final sur 100% des données.
    Retourne : Le modèle entraîné.
    """
    print("--- Entraînement du modèle final (Production) ---")
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X, y)
    return model