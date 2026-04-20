# Phase 2: Experiment Tracking & Model Registry Report

## 1. Local MLflow Setup and Execution

In this phase, all experiments were logged using MLflow on our local tracking server (`http://localhost:5000`).
Five different machine learning models were developed and compared to predict the `price` of Steam games based on 10 engineered features: `average_playtime`, `achievements`, `release_year`, `self_published`, `english`, `is_mac`, `is_multiplayer`, `is_indie`, `is_action`, and `is_early_access`.

The parameters, metrics, and generated artifacts were logged into the tracking server.

---

## 2. Screenshot: Experiment Overview

The screenshot below shows the MLflow UI with all runs from the `SteamGamesPricePrediction_v1` experiment. Runs are compared side-by-side across MAE, R² and RMSE metrics using the Chart View.

![MLflow Experiment Overview – All Runs with MAE, R², RMSE metrics](ScreenshotPhase2_2_1.png)

---

## 3. Comparison of Models

During the experiment, five configurations were assessed and logged as separate MLflow runs:

1. **LinearRegression** (Baseline)
2. **RidgeRegression** (Alpha = 1.0)
3. **RandomForestRegressor** (n_estimators = 50, max_depth = 10)
4. **GradientBoostingRegressor** (n_estimators = 100) ← **Best Model**
5. **KNeighborsRegressor** (n_neighbors = 5)

### Best Model Selection
The best performing model was the **GradientBoostingRegressor**. This was determined automatically by comparing the **Root Mean Squared Error (RMSE)** across all five runs – the run with the lowest RMSE was automatically registered to Production in the MLflow Model Registry.

The metric **RMSE** was chosen as the primary decision factor because it penalizes large prediction errors disproportionately. For a price prediction task, wildly wrong predictions (e.g., predicting €5 for a €60 game) are far more damaging than small consistent errors, making RMSE the most appropriate optimization target.

As visible in the MLflow chart above, `chill-stork-205` (GradientBoostingRegressor) achieved the lowest RMSE (~6.32) compared to linear models (~6.6+) and KNN.

### Metric Discussion
- **MAE (Mean Absolute Error)**: On average, predictions are off by ~€4.00–4.50. This is a reasonable margin for a game whose price may range from €4 to €80.
- **RMSE (Root Mean Squared Error)**: Our best model scores ~6.32. The higher RMSE relative to MAE reveals that some games (particularly mid-tier AAA titles at €40-60) are harder to predict precisely – the model underestimates their price due to limited AAA data in the dataset.
- **R² Score**: Best result ~0.26, meaning roughly 26% of price variance is explained by the 10 engineered features. The remaining variance is driven by brand awareness, marketing budget, and publisher reputation – factors not present in the dataset.

### Dataset Curation Decisions
To improve model quality, strict data hygiene rules were applied before training:
- Games priced **below €4.00** and **above €80** were excluded to remove free-to-play titles and outlier bundles.
- Games priced under €6 with more than 50 achievements were removed, as these are typically achievement-farming products with artificially low prices.
- The 80/20 train-test split was stratified across three price tiers (€4–15, €15–40, €40–80) to ensure all price segments are represented in both the training and test sets.

---

## 4. Screenshot: Registered Model

The screenshot below confirms that `BestRegressor_v1` – Version 11 (GradientBoostingRegressor) – is successfully deployed to the **Production** stage in the MLflow Model Registry.

![BestRegressor_v1 – Version 11 in Production stage](ScreenshotPhase2_4_1.png)

---

## 5. Bonus: Confidence Interval Interpretation (Statsmodels)

A Bonus Step was conducted using `statsmodels` (OLS) to calculate 95% confidence intervals for each regression coefficient. The results were exported as `conf_intervals.csv` and logged as an MLflow artifact on the best run.

**What do these results represent?**
The 95% confidence intervals express the range within which the true coefficient value lies with 95% certainty. For example:
- `average_playtime`: interval is entirely above zero, meaning longer playtimes are associated with higher prices.
- `is_indie`: interval is entirely below zero, meaning indie-tagged games are statistically priced lower.
- `release_year`: positive interval, meaning more recent releases tend to be priced higher.

These intervals are also loaded directly into the Streamlit dashboard (Phase 3) and displayed as the `[95% CI: € X – € Y]` range beneath each prediction.
