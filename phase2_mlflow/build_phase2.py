import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import json
import nbformat as nbf
import os
import shutil

target_csv = "steam.csv"
source_csv = "../SteamStoreGamesDataset/steam.csv"
if not os.path.exists(target_csv): shutil.copy(source_csv, target_csv)

df = pd.read_csv(target_csv)

def has_term(val, target):
    if not isinstance(val, str): return 0
    return 1 if any(t.strip().lower() == target.lower() for t in val.split(';')) else 0

df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_year'] = df['release_year'].fillna(df['release_year'].median())
df['self_published'] = (df['developer'] == df['publisher']).astype(int)
df['is_mac'] = df['platforms'].apply(lambda x: 1 if isinstance(x, str) and 'mac' in x.lower() else 0)
df['is_linux'] = df['platforms'].apply(lambda x: 1 if isinstance(x, str) and 'linux' in x.lower() else 0)
df['is_multiplayer'] = df['categories'].apply(lambda x: has_term(x, 'Multi-player'))
df['is_vr'] = df['categories'].apply(lambda x: has_term(x, 'VR Support'))
df['is_indie'] = df['genres'].apply(lambda x: has_term(x, 'Indie'))
df['is_action'] = df['genres'].apply(lambda x: has_term(x, 'Action'))
df['is_rpg'] = df['genres'].apply(lambda x: has_term(x, 'RPG'))
df['is_adventure'] = df['genres'].apply(lambda x: has_term(x, 'Adventure'))
df['is_early_access'] = df['genres'].apply(lambda x: has_term(x, 'Early Access'))

features = [
    'average_playtime', 'achievements', 'release_year', 'self_published', 'english',
    'is_mac', 'is_multiplayer', 'is_indie',
    'is_action', 'is_early_access'
]
target = 'price'

df_clean = df.dropna(subset=features + [target])
df_clean = df_clean[(df_clean['price'] >= 4.0) & (df_clean['price'] <= 80.0)]
df_clean = df_clean[~((df_clean['price'] <= 6.0) & (df_clean['achievements'] > 50))]

X = df_clean[features]
y = df_clean[target]

df_clean['price_tier'] = pd.cut(df_clean['price'], bins=[0, 15, 40, 100], labels=[1, 2, 3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_clean['price_tier'])

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SteamGamesPricePrediction_v1")

best_rmse = float('inf')
best_run_id = None

# Model 1
with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "LinearRegression")
    mlflow.set_tag("dataset_version", "steam_v1")
    mlflow.log_param("test_size", 0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2_score(y_test, y_pred))
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, color="steelblue", edgecolor="white")
    plt.savefig("residuals_lr.png"); plt.close()
    mlflow.log_artifact("residuals_lr.png"); os.remove("residuals_lr.png")
    mlflow.sklearn.log_model(model, "model")
    if rmse < best_rmse: best_rmse = rmse; best_run_id = run.info.run_id

# Model 2
with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "RidgeRegression")
    mlflow.set_tag("dataset_version", "steam_v1")
    mlflow.log_param("alpha", 1.0)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    if rmse < best_rmse: best_rmse = rmse; best_run_id = run.info.run_id

# Model 3
with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "RandomForestRegressor")
    mlflow.set_tag("dataset_version", "steam_v1")
    mlflow.log_param("n_estimators", 50)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2_score(y_test, y_pred))
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, color="green", edgecolor="white")
    plt.savefig("residuals_rf.png"); plt.close()
    mlflow.log_artifact("residuals_rf.png"); os.remove("residuals_rf.png")
    mlflow.sklearn.log_model(model, "model")
    if rmse < best_rmse: best_rmse = rmse; best_run_id = run.info.run_id

# Model 4
with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "GradientBoostingRegressor")
    mlflow.set_tag("dataset_version", "steam_v1")
    mlflow.log_param("n_estimators", 100)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    if rmse < best_rmse: best_rmse = rmse; best_run_id = run.info.run_id

# Model 5
with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "KNeighborsRegressor")
    mlflow.set_tag("dataset_version", "steam_v1")
    mlflow.log_param("n_neighbors", 5)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2_score(y_test, y_pred))
    mlflow.sklearn.log_model(model, "model")
    if rmse < best_rmse: best_rmse = rmse; best_run_id = run.info.run_id

if best_run_id:
    # Register safely forcing production
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", "BestRegressor_v1")
    from mlflow.client import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(name="BestRegressor_v1", version=mv.version, stage="Production")

alpha = 0.05
lr_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
conf_interval = lr_sm.conf_int(alpha)
conf_interval.to_csv("conf_intervals.csv")
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_artifact("conf_intervals.csv")

# Generate notebook identical structure
nb = nbf.v4.new_notebook()
nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 2: Experiment Tracking & Model Registry with MLflow"),
    nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport mlflow\nimport mlflow.sklearn\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression, Ridge\nfrom sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport statsmodels.api as sm"""),
    nbf.v4.new_code_cell("""df = pd.read_csv("steam.csv")
def has_term(val, target):
    if not isinstance(val, str): return 0
    return 1 if any(t.strip().lower() == target.lower() for t in val.split(';')) else 0
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['release_year'] = df['release_year'].fillna(df['release_year'].median())
df['self_published'] = (df['developer'] == df['publisher']).astype(int)
df['is_mac'] = df['platforms'].apply(lambda x: 1 if isinstance(x, str) and 'mac' in x.lower() else 0)
df['is_multiplayer'] = df['categories'].apply(lambda x: has_term(x, 'Multi-player'))
df['is_indie'] = df['genres'].apply(lambda x: has_term(x, 'Indie'))
df['is_action'] = df['genres'].apply(lambda x: has_term(x, 'Action'))
df['is_early_access'] = df['genres'].apply(lambda x: has_term(x, 'Early Access'))
features = ['average_playtime', 'achievements', 'release_year', 'self_published', 'english', 'is_mac', 'is_multiplayer', 'is_indie', 'is_action', 'is_early_access']
target = 'price'
df_clean = df.dropna(subset=features + [target])
df_clean = df_clean[(df_clean['price'] >= 4.0) & (df_clean['price'] <= 80.0)]
df_clean = df_clean[~((df_clean['price'] <= 6.0) & (df_clean['achievements'] > 50))]

X = df_clean[features]; y = df_clean[target]
df_clean['price_tier'] = pd.cut(df_clean['price'], bins=[0, 15, 40, 100], labels=[1, 2, 3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_clean['price_tier'])"""),
    nbf.v4.new_code_cell("""mlflow.set_tracking_uri("http://localhost:5000")\nmlflow.set_experiment("SteamGamesPricePrediction_v1")"""),
    nbf.v4.new_code_cell("""with mlflow.start_run() as run:
    mlflow.set_tag("algorithm", "GradientBoostingRegressor")
    mlflow.set_tag("dataset_version", "steam_v1")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mlflow.log_metric("MAE", mean_absolute_error(y_test, model.predict(X_test)))
    mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    mlflow.log_metric("R2", r2_score(y_test, model.predict(X_test)))
    mlflow.sklearn.log_model(model, "model")
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "BestRegressor_v1")"""),
    nbf.v4.new_code_cell("""alpha = 0.05\nlr_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()\nconf_interval = lr_sm.conf_int(alpha)\nprint(conf_interval)"""),
    nbf.v4.new_markdown_cell("## Interpretation of the Confidence Intervals\n\nThe 95% confidence intervals give the range within which the true regression coefficient lies with 95% certainty. A coefficient interval entirely above zero means the feature has a statistically significant positive effect on price. An interval that crosses zero suggests the feature may not reliably influence the prediction.")
]
with open("phase2.ipynb", "w", encoding='utf-8') as f: nbf.write(nb, f)
