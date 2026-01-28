import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')   # Backend for non-GUI environments (saves files instead of showing them)
import matplotlib.pyplot as plt
import xgboost as xgb

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
print(f"Erreur de TON modèle (LinearRegression) : {erreur:.2f} €")

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

