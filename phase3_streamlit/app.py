import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import statsmodels.api as sm

st.set_page_config(page_title="Steam Price Predictor", layout="wide", page_icon="🎮")
st.title("📊 Steam Game Price Predictor Dashboard (10 Features)")
st.markdown("Use this dashboard to evaluate the expected market price of your next indie game.")

@st.cache_resource
def load_mlflow_model_and_artifact(_cache_buster="v11_Optimized"):
    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        model = mlflow.sklearn.load_model(model_uri="models:/BestRegressor_v1/Production")
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("BestRegressor_v1", stages=["Production"])
        ci_df = None
        if versions:
            run_id = versions[0].run_id
            local_path = client.download_artifacts(run_id, "conf_intervals.csv")
            ci_df = pd.read_csv(local_path, index_col=0)
        return model, ci_df
    except Exception as e:
        st.error(f"Failed to load BestRegressor_v1 from MLflow. Ensure server is running. Error: {e}")
        return None, None

@st.cache_resource
def load_ols_for_ci(_cache_buster="v11_Optimized"):
    try:
        df = pd.read_csv("../SteamStoreGamesDataset/steam.csv")
        def has_term(val, target):
            if not isinstance(val, str): return 0
            return 1 if any(t.strip().lower() == target.lower() for t in val.split(';')) else 0

        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
        df['self_published'] = (df['developer'] == df['publisher']).astype(int)
        df['english'] = df['english']
        df['is_mac'] = df['platforms'].apply(lambda x: 1 if isinstance(x, str) and 'mac' in x.lower() else 0)
        df['is_multiplayer'] = df['categories'].apply(lambda x: has_term(x, 'Multi-player'))
        df['is_indie'] = df['genres'].apply(lambda x: has_term(x, 'Indie'))
        df['is_action'] = df['genres'].apply(lambda x: has_term(x, 'Action'))
        df['is_early_access'] = df['genres'].apply(lambda x: has_term(x, 'Early Access'))

        features = [
            'average_playtime', 'achievements', 'release_year', 'self_published', 'english',
            'is_mac', 'is_multiplayer', 'is_indie',
            'is_action', 'is_early_access'
        ]
        
        df_clean = df.dropna(subset=features + ['price'])
        df_clean = df_clean[(df_clean['price'] >= 4.0) & (df_clean['price'] <= 80.0)]
        df_clean = df_clean[~((df_clean['price'] <= 6.0) & (df_clean['achievements'] > 50))]
        
        X_train = df_clean[features]
        y_train = df_clean['price']
        
        lr_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        return lr_sm, features
    except Exception as e:
        st.error(f"Failed to load OLS Baseline for Confidence Interval calc: {e}")
        return None, None

model, artefact_ci_df = load_mlflow_model_and_artifact()
lr_sm, feature_names = load_ols_for_ci()

# --- Sidebar Inputs ---
st.sidebar.header("🛠️ Feature Engineering (10 Features)")

playtime_input = st.sidebar.slider("Target Average Playtime (Minutes)", min_value=0, max_value=20000, value=300, step=30)
achievements_input = st.sidebar.slider("Planned Achievements", min_value=0, max_value=1000, value=25, step=1)
release_year = st.sidebar.slider("Target Release Year", min_value=2000, max_value=2026, value=2024, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Network OS & Publisher Target**")
is_windows = st.sidebar.checkbox("Windows Support", True, disabled=True)
is_mac = st.sidebar.checkbox("Mac OS Support", False)
self_published = st.sidebar.checkbox("Self-Published (Anti-AAA)", True)
english = st.sidebar.checkbox("English Translation Available", True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Store Tags / Genres**")
is_multiplayer = st.sidebar.checkbox("Multiplayer Core", False)
is_indie = st.sidebar.checkbox("Indie Class", True)
is_action = st.sidebar.checkbox("Action Combat", False)
is_early_access = st.sidebar.checkbox("Early Access Build", False)

input_data = pd.DataFrame([{
    "average_playtime": playtime_input,
    "achievements": achievements_input,
    "release_year": release_year,
    "self_published": 1 if self_published else 0,
    "english": 1 if english else 0,
    "is_mac": 1 if is_mac else 0,
    "is_multiplayer": 1 if is_multiplayer else 0,
    "is_indie": 1 if is_indie else 0,
    "is_action": 1 if is_action else 0,
    "is_early_access": 1 if is_early_access else 0
}])

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔮 Prediction")
    if st.button("Calculate Optimal Price"):
        if model is not None:
            pred = model.predict(input_data)[0]
            pred = max(0, pred)
            
            if lr_sm is not None:
                sm_input = sm.add_constant(input_data, has_constant='add')
                for ft in ["const"] + feature_names:
                    if ft not in sm_input:
                        sm_input[ft] = 1.0 if ft == "const" else input_data[ft].iloc[0]
                sm_input = sm_input[["const"] + feature_names]

                pred_sm_details = lr_sm.get_prediction(sm_input).summary_frame(alpha=0.05)
                ci_low = max(0, pred_sm_details["mean_ci_lower"].values[0])
                ci_high = max(0, pred_sm_details["mean_ci_upper"].values[0])
                
                st.metric(label="Predicted Price by Production Model", value=f"€ {pred:,.2f}")
                st.info(f"**Statistical Range [95% CI]**: € {ci_low:,.2f} – € {ci_high:,.2f}")
                
                if artefact_ci_df is not None:
                    with st.expander("Show Base Linear Variable Bounds (from MLflow Artifact)"):
                        st.dataframe(artefact_ci_df)
            else:
                st.metric(label="Predicted Price", value=f"€ {pred:,.2f}")
        else:
            st.warning("Model isn't loaded correctly.")

with col2:
    st.header("💡 Key Insights")
    st.markdown("Understanding what dictates the MLflow Production Model decisions.")
    
    if model is not None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(importances)), importances[indices], color='#8b5cf6', align='center')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Relative Importance')
        ax.set_title(f'Feature Importance ({model.__class__.__name__})')
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        
        st.pyplot(fig)
    else:
        st.info("Feature importance is not available for this model type.")
