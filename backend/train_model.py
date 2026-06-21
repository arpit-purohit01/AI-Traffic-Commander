import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import json

print("--- Data Engineering ---")
csv_file = '../Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(csv_file)
df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
df = df.dropna(subset=['start_datetime', 'closed_datetime']).copy()

# Target
df['actual_delay_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
valid_df = df[(df['actual_delay_mins'] > 5) & (df['actual_delay_mins'] < 480)].copy()

# Temporal Features
valid_df['hour_of_day'] = valid_df['start_datetime'].dt.hour
valid_df['day_of_week'] = valid_df['start_datetime'].dt.dayofweek
valid_df['month'] = valid_df['start_datetime'].dt.month
valid_df['quarter'] = valid_df['start_datetime'].dt.quarter
valid_df['is_weekend'] = valid_df['day_of_week'].isin([5, 6]).astype(int)

# Categorical Fills
valid_df['event_cause'] = valid_df['event_cause'].fillna('unknown')
valid_df['priority'] = valid_df['priority'].fillna('Low')
valid_df['zone'] = valid_df['zone'].fillna('Central Zone 2')
valid_df['requires_road_closure'] = valid_df['requires_road_closure'].fillna(False).astype(int)

# Traffic/Location/Environmental (Simulated from existing cols or hardcoded for inference match)
# The user wants exact shared feature engineering.
# We will train on exact features:
features = ['event_cause', 'priority', 'requires_road_closure', 'zone', 'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'quarter']
categorical_features = ['event_cause', 'priority', 'zone']
numeric_features = ['requires_road_closure', 'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'quarter']

X = valid_df[features].copy()
y = valid_df['actual_delay_mins']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Building Preprocessor ---")
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

print("--- Training Models ---")
models = {
    "cat": CatBoostRegressor(iterations=100, depth=6, verbose=0, random_state=42),
    "xgb": XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
    "et": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "lin": LinearRegression()
}

fitted_models = {}
for name, model in models.items():
    model.fit(X_train_trans, y_train)
    fitted_models[name] = model

print("--- Ensemble Strategy Evaluation ---")
preds_cat = fitted_models['cat'].predict(X_test_trans)
preds_xgb = fitted_models['xgb'].predict(X_test_trans)
preds_et = fitted_models['et'].predict(X_test_trans)
preds_lin = fitted_models['lin'].predict(X_test_trans)

# 40% Cat, 30% XGB, 20% ET, 10% Lin
ensemble_preds = 0.40 * preds_cat + 0.30 * preds_xgb + 0.20 * preds_et + 0.10 * preds_lin

mae = mean_absolute_error(y_test, ensemble_preds)
rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
r2 = r2_score(y_test, ensemble_preds)

print(f"Ensemble MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

print("--- Conformal Prediction (95% Coverage Guarantee) ---")
residuals = np.abs(y_test - ensemble_preds)
conformal_score = np.percentile(residuals, 95)
print(f"Conformal 95% Interval: +/- {conformal_score:.2f} mins")

print("--- KNN Similar Event Engine ---")
from sklearn.neighbors import NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(X_train_trans)
historical_events = valid_df[['event_cause', 'zone', 'start_datetime', 'actual_delay_mins']].loc[X_train.index].reset_index(drop=True)

print("--- Saving Artifacts Exactly As Requested ---")
os.makedirs('models', exist_ok=True)
with open('models/preprocessor.pkl', 'wb') as f: pickle.dump(preprocessor, f)
with open('models/nn_model.pkl', 'wb') as f: pickle.dump(nn_model, f)
with open('models/historical_events.pkl', 'wb') as f: pickle.dump(historical_events, f)

for name, model in fitted_models.items():
    with open(f'models/production_model_{name}.pkl', 'wb') as f: pickle.dump(model, f)

feature_config = {
    "features": features,
    "categorical": categorical_features,
    "numeric": numeric_features,
    "conformal_score": float(conformal_score)
}
with open('models/feature_config.json', 'w') as f: json.dump(feature_config, f)

metrics = {
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 2)
}
with open('models/metrics.json', 'w') as f: json.dump(metrics, f)

# Save historical data for Drift PSI baseline
valid_df[['actual_delay_mins']].to_csv('models/baseline_delays.csv', index=False)

print("Training Complete. Models and configs saved.")
