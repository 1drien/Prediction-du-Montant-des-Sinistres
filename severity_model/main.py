import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')   # Backend for non-GUI environments (saves files instead of showing them)
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
import optuna


df = pd.read_csv('train.csv')
print(df.head())

#df = df.drop(columns=['nombre_sinistres'])
df = df.drop(df[df['nombre_sinistres'] == 0].index)

print(df.head())
print(df['nombre_sinistres'] == 0)

print(df.info()) #object : Ce sont tes variables catégorielles

df['sex_conducteur2'] = df['sex_conducteur2'].fillna('Inconnu')

print(df.info())

#ici on remplace les valeurs catégorielles par des valeurs numériques
df['sex_conducteur1'].replace(['M', 'F'], [0, 1], inplace=True)
df['type_vehicule'].replace(['Tourism', 'Commercial'], [0, 1], inplace=True)
df['utilisation'].replace(['Retired', 'WorkPrivate', 'Professional', 'AllTrips'], [0, 1, 2, 4], inplace=True)
df['freq_paiement'].replace(['Monthly', 'Quarterly','Biannual', 'Yearly'], [0, 1, 2, 3], inplace=True)


print(df.head())

#one hot encoding pour les autres variables catégorielles 
df = pd.get_dummies(df, columns=['marque_vehicule', 
                                 'modele_vehicule', 
                                 'sex_conducteur2', 
                                 'type_contrat', 
                                 'paiement', 
                                 'conducteur2', 
                                 'essence_vehicule'], dtype=int) 

#on supprime les autres colonnes types id poour eviter le surapprentissage
df = df.drop(columns=['id_contrat', 'id_client', 'id_vehicule', 'id_contrat', 'code_postal'])

y = df["montant_sinistre"]
X = df.drop(columns=["montant_sinistre"])

# Split initial
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Création du modèle
model = LinearRegression()

# 2. Entraînement sur les 80% de données (Train)
model.fit(X_train, y_train)

# 3. Prédiction sur les 20% restants (Test)
predictions = model.predict(X_test)

# 4. Calcul de l'erreur (La vérité - La prédiction)
erreur = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {erreur}")

# On crée un tableau rempli avec la moyenne (1770) pour faire semblant d'être le modèle "naïf"
y_pred_baseline = np.full((len(y_test),), y_train.mean())

mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f"RMSE between Linear Regression and Baseline: {rmse}")

print(f"Erreur du modèle 'Naïf' (Moyenne partout) : {mae_baseline:.2f} €")
print(f"Erreur du modèle (LinearRegression) : {erreur:.2f} €")

# Visualisation des valeurs 
plt.hist(df['montant_sinistre'], bins=50, alpha=0.7, label='Montant Sinistre')
plt.savefig("montant_sinistre_distribution.png")
print("Saved histogram plot to 'montant_sinistre_distribution.png'")
plt.show()

# nous allons maintenant log les données de montant_sinistre pour voir si cela améliore les performances du modèle
df_log = df.copy()
log_montant_sinistre = np.log(df_log['montant_sinistre'] + 1)  # Ajout de 1 pour éviter le log(0)
df_log['montant_sinistre'] = log_montant_sinistre

X_log = df_log.drop(columns=["montant_sinistre"])
y_log = df_log["montant_sinistre"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)

predictions_log = model_log.predict(X_test_log)
true_prediction = np.exp(predictions_log) -1
y_test_original = np.exp(y_test_log) -1

erreur_log = mean_absolute_error(y_test_original, true_prediction)

print(f"error after log transformation: {erreur_log}")

# rmse 
rmse_log = np.sqrt(np.mean((y_test_original - true_prediction)**2))
print(f"RMSE after log transformation :  {rmse_log}")

# nous allons maintenant faire un xgboost pour voir si cela améliore les performances du modèle
X = df_log.drop(columns=["montant_sinistre"])
y = df_log["montant_sinistre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#optimisation of hyperparameters with optuna
final_params = {
    'objective': 'reg:squarederror',
    'verbosity': 0,
    'booster': 'gblinear',
    'lambda': 1.049e-05,
    'alpha': 0.000823,
    'subsample': 0.897,
    'colsample_bytree': 0.204
}

# 2. On crée et entraîne le modèle unique avec ces paramètres
final_model = XGBRegressor(**final_params)
final_model.fit(X_train, y_train)

# 3. Prédictions finales sur le Test Set
y_pred_log = final_model.predict(X_test)

# 4. Conversion inverse (Log -> Euros)
y_pred_euros = np.exp(y_pred_log) - 1
y_test_euros = np.exp(y_test) - 1

# 5. Calcul des scores finaux
mae_final = mean_absolute_error(y_test_euros, y_pred_euros)
rmse_final = np.sqrt(np.mean((y_test_euros - y_pred_euros)**2))

print("------------------------------------------------")
print(f" MAE Finale : {mae_final:.2f} €")
print(f" RMSE Final : {rmse_final:.2f} €")
print("------------------------------------------------")

# --- IMPORTANCE DES FEATURES ---
# Note : Avec 'gblinear', l'importance correspond aux 'poids' (coefficients) de la régression.

# On récupère les poids
weights = pd.Series(final_model.coef_, index=X_train.columns)

# On prend les 10 plus impactants (en valeur absolue, car un poids négatif est aussi important)
top_features = weights.abs().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='purple')
plt.title("Top 10 des variables les plus importantes (Modèle gblinear)")
plt.xlabel("Poids (Impact sur le prix)")
plt.gca().invert_yaxis() # Pour avoir le plus gros en haut
plt.savefig("feature_importance_final.png")
print("Graphique sauvegardé sous 'feature_importance_final.png'")
plt.show()


#ici le code utilisé avec optuna pour optimiser les hyperparamètres de xgboost
# def objective(trial):
#     param = {
#         "verbosity": 0,
#         "objective": "reg:squarederror",
#         # use exact for small dataset.
#         "tree_method": "exact",
#         # defines booster, gblinear for linear functions.
#         "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
#         # L2 regularization weight.
#         "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
#         # L1 regularization weight.
#         "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
#         # sampling ratio for training data.
#         "subsample": trial.suggest_float("subsample", 0.2, 1.0),
#         # sampling according to each tree.
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
#     }

#     if param["booster"] in ["gbtree", "dart"]:
#         # maximum depth of the tree, signifies complexity of the tree.
#         param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
#         # minimum child weight, larger the term more conservative the tree.
#         param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
#         param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
#         # defines how selective algorithm is.
#         param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
#         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

#     if param["booster"] == "dart":
#         param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
#         param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
#         param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
#         param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
#     xgb_reg = XGBRegressor(**param)
#     xgb_reg.fit(X_train, y_train)
# #prediction sur le test set 
#     y_pred_log = xgb_reg.predict(X_test)
# #on converti nos valeurs en euros 
#     y_pred_euros = np.exp(y_pred_log) -1
#     y_test_euros = np.exp(y_test) - 1
#     mae_xgb = mean_absolute_error(y_pred_euros, y_test_euros)
#     print(f"XGBoost MAE on training set: {mae_xgb}")
#     rmse_xgb_train = np.sqrt(np.mean((y_pred_euros - y_test_euros)**2))
#     return rmse_xgb_train


# if __name__ == "__main__":
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=100, timeout=600)

#     print("Number of finished trials: ", len(study.trials))
#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: {}".format(trial.value))
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

# --- VERDICT SURAPPRENTISSAGE ---

# 1. Prédiction sur le jeu d'ENTRAÎNEMENT (ce qu'il connait déjà)
y_pred_train_log = final_model.predict(X_train)
y_pred_train_euros = np.exp(y_pred_train_log) - 1
y_train_euros = np.exp(y_train) - 1

# 2. Calcul du RMSE Train
rmse_train = np.sqrt(np.mean((y_train_euros - y_pred_train_euros)**2))

print("\n--- VERDICT SURAPPRENTISSAGE ---")
print(f"Erreur sur le Train (Mémoire) : {rmse_train:.2f} €")
print(f"Erreur sur le Test (Réalité)  : {rmse_final:.2f} €")

ecart = rmse_final - rmse_train
print(f"Écart : {ecart:.2f} €")

if ecart > 200: # Seuil arbitraire, dépend du métier
    print("Risque de surapprentissage")
elif ecart < 0:
    print("Le modèle est meilleur sur l'inconnu")
else:
    print("Le modèle généralise bien ")