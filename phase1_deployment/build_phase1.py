import pandas as pd
import numpy as np
import json
import nbformat as nbf
import os
import shutil

# Step 0: Ensure dataset is available
target_csv = "steam.csv"
source_csv = "../SteamStoreGamesDataset/steam.csv"
if not os.path.exists(target_csv):
    shutil.copy(source_csv, target_csv)

# ---------------------------------------------------------
# Part A: Run the training purely in Python to generate model.json
# ---------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("steam.csv")

# Feature Engineering
df['is_multiplayer'] = df['categories'].apply(lambda x: 1 if isinstance(x, str) and 'Multi-player' in x else 0)
df['is_indie'] = df['genres'].apply(lambda x: 1 if isinstance(x, str) and 'Indie' in x else 0)
df['is_action'] = df['genres'].apply(lambda x: 1 if isinstance(x, str) and 'Action' in x else 0)

features = ['average_playtime', 'achievements', 'is_multiplayer', 'is_indie', 'is_action']
target = 'price'

df_clean = df.dropna(subset=features + [target])

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R2: {r2_score(y_test, y_pred)}")

# Export model for ML.js
model_data = {
    "coef": model.coef_.tolist(),
    "intercept": float(model.intercept_),
    "feature_names": features
}
with open("model.json", "w") as f:
    json.dump(model_data, f)
print("Saved model.json")

# ---------------------------------------------------------
# Part B: Generate the Jupyter Notebook
# ---------------------------------------------------------
nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell("# Phase 1: Classic ML with Scikit-learn & Web Deployment\n\nThis notebook demonstrates a classical machine learning workflow: Exploratory Data Analysis, Data Engineering, Training a Scikit-Learn Linear Regression model, and evaluating it. Finally, we export the model weights to a JSON file capable of being run natively inside a browser using `ML.js`."),
    nbf.v4.new_code_cell("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nimport json"""),
    nbf.v4.new_markdown_cell("## 1. Data Loading & Exploratory Data Analysis"),
    nbf.v4.new_code_cell("""# Load the dataset
df = pd.read_csv("steam.csv")
print("Dataset Shape:", df.shape)
df.head()"""),
    nbf.v4.new_code_cell("""# Visualizing Price Distribution
plt.figure(figsize=(10, 5))
plt.hist(df['price'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Game Prices on Steam')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.xlim(0, 100) # limiting to $100 for better visibility
plt.show()"""),
    nbf.v4.new_markdown_cell("## 2. Feature Engineering\nWe want intuitive features for indie developers to estimate their price. We will extract:\n- `average_playtime`: Target average playtime\n- `achievements`: Number of achievements\n- `is_multiplayer`: Does it have multiplayer?\n- `is_indie`: Is it classified as an Indie game?\n- `is_action`: Is it an action game?"),
    nbf.v4.new_code_cell("""# Engineer binary flags from categories and genres
df['is_multiplayer'] = df['categories'].apply(lambda x: 1 if isinstance(x, str) and 'Multi-player' in x else 0)
df['is_indie'] = df['genres'].apply(lambda x: 1 if isinstance(x, str) and 'Indie' in x else 0)
df['is_action'] = df['genres'].apply(lambda x: 1 if isinstance(x, str) and 'Action' in x else 0)

features = ['average_playtime', 'achievements', 'is_multiplayer', 'is_indie', 'is_action']
target = 'price'

# Drop rows with missing values
df_clean = df.dropna(subset=features + [target])
X = df_clean[features]
y = df_clean[target]
"""),
    nbf.v4.new_markdown_cell("## 3. Model Training & Evaluation"),
    nbf.v4.new_code_cell("""# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
"""),
    nbf.v4.new_markdown_cell("## 4. Exporting Model for Web (ML.js)"),
    nbf.v4.new_code_cell("""# Export the model parameters into JSON
model_data = {
    "coef": model.coef_.tolist(),
    "intercept": float(model.intercept_),
    "feature_names": features
}

with open("model.json", "w") as f:
    json.dump(model_data, f)
    
print("Successfully generated model.json!")
print("Coefficients:", model_data['coef'])
print("Intercept:", model_data['intercept'])
""")
]

with open("phase1.ipynb", "w", encoding='utf-8') as f:
    nbf.write(nb, f)
print("Saved phase1.ipynb")
