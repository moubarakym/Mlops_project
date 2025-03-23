import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


# Initialisation de MLflow
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

all_experiments = client.search_experiments()


# Description de l'expérience MLflow
experiment_description = "This experience is for our mlops project purpose."

experiment_tags = {
    "project_name": "mlops PROJECT",
    "store_dept": "produce",
    "team": "Hajer-GAM, Eya-SAIDI, Rahaf-ATRI, Moubarak-YAHAYA_MOUSSA",
    "mlflow.note.content": experiment_description,
}

produce_apples_experiment = client.create_experiment(
    name="Loan prediction", tags=experiment_tags
)

apples_experiment = client.search_experiments(
    filter_string="tags.project_name = 'mlops PROJECT'"
)

mlflow.set_tracking_uri("http://127.0.0.1:8080")
loan_experiment = mlflow.set_experiment("Loan prediction")

# Chargement des données
file_path = "Loan_Data.csv"
df = pd.read_csv(file_path)


df_cleaned = df.drop(columns=["customer_id"])

# Séparation des features et de la cible
df_cleaned = df.drop(columns=["customer_id"])
X = df_cleaned.drop(columns=["default"])
y = df_cleaned["default"]

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Liste des modèles à entraîner
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
}

params = {"test_size": 0.2, "random_state": 42}

best_model = None
best_score = -float('inf')

for model_name, model in models.items():
    # Entraînement
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    # Ajout des paramètres spécifiques du modèle
    if model_name == "Decision Tree":
        model_params = {
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "random_state": model.random_state
        }
    elif model_name == "Random Forest":
        model_params = {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "random_state": model.random_state
        }
    else:
        model_params = {"test_size":0.2, "random_state":42}

    # Suivi avec MLflow
    with mlflow.start_run(run_name=f"{model_name}_test") as run:
        # Log des paramètres généraux
        mlflow.log_params(params)
        # Log des paramètres spécifiques au modèle
        mlflow.log_params(model_params)
        # Log des métriques
        mlflow.log_metrics(metrics)
        # Log du modèle
        mlflow.sklearn.log_model(
            sk_model=model, input_example=X_train_scaled, artifact_path=f"{model_name}_model"
        )

        # Comparaison des modèles
        if r2 > best_score:
            best_score = r2
            best_model = model

# Sauvegarde du modèle Linear Regression localement
if best_model:
    with open('model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("Meilleur modèle sauvegardé sous 'model.pkl' !")
else:
    print("Aucun modèle n'a été trouvé pour être sauvegardé.")

