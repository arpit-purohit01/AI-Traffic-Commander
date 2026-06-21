import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import create_engine
import shap

print("--- V10 Automated Retraining Pipeline ---")

# 1. Load Original Dataset
csv_file = '../Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
base_df = pd.read_csv(csv_file)

# Parse times
base_df['start_datetime'] = pd.to_datetime(base_df['start_datetime'], errors='coerce')
base_df['closed_datetime'] = pd.to_datetime(base_df['closed_datetime'], errors='coerce')
base_df = base_df.dropna(subset=['start_datetime', 'closed_datetime']).copy()

# Calculate actual delay
base_df['actual_delay_mins'] = (base_df['closed_datetime'] - base_df['start_datetime']).dt.total_seconds() / 60.0

# 2. Extract specific features required
base_df['hour_of_day'] = base_df['start_datetime'].dt.hour
base_df['day_of_week'] = base_df['start_datetime'].dt.dayofweek
base_df['month'] = base_df['start_datetime'].dt.month
base_df['is_weekend'] = base_df['day_of_week'].isin([5, 6]).astype(int)

base_df['event_type'] = base_df['event_type'].fillna('unknown')
base_df['event_cause'] = base_df['event_cause'].fillna('unknown')
base_df['priority'] = base_df['priority'].fillna('Low')
base_df['zone'] = base_df['zone'].fillna('Central Zone 2')
base_df['requires_road_closure'] = base_df['requires_road_closure'].fillna(False).astype(int)

# 3. Load Event Logs (Simulated "Live" Data Feedback)
try:
    engine = create_engine("sqlite:///./events.db")
    logs_df = pd.read_sql("SELECT * FROM event_logs WHERE actual_delay IS NOT NULL", engine)
    print(f"Found {len(logs_df)} live validated logs to merge.")
    # In a real system, we would map the logs_df back to the features, but for this hackathon
    # demo, we will just use the base_df to simulate the retraining mechanic.
except Exception as e:
    print("No live event logs database found or empty.")

# 4. Clean Data (Strict V10 Bounds)
valid_df = base_df[(base_df['actual_delay_mins'] > 5) & (base_df['actual_delay_mins'] < 480)].copy()

features = ['event_type', 'event_cause', 'priority', 'requires_road_closure', 'zone', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']
categorical_features = ['event_type', 'event_cause', 'priority', 'zone']
numeric_features = ['requires_road_closure', 'hour_of_day', 'day_of_week', 'month', 'is_weekend']

X = valid_df[features].copy()
y = valid_df['actual_delay_mins']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# V10 Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42))
])

print("Retraining Pipeline...")
pipeline.fit(X_train, y_train)

# Metrics
preds = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print(f"New Test Metrics - MAE: {mae:.2f}m, RMSE: {rmse:.2f}m, R^2: {r2:.2f}")
metrics = {"mae": round(mae, 1), "rmse": round(rmse, 1), "r2": round(r2, 2)}

# Feature Names & Importances
cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

model_importances = pipeline.named_steps['model'].feature_importances_
feature_importances = [{"feature": all_feature_names[i], "importance": round(float(model_importances[i]), 4)} for i in range(len(all_feature_names))]
feature_importances = sorted(feature_importances, key=lambda x: x['importance'], reverse=True)

# Rebuild KNN Engine
X_transformed = pipeline.named_steps['preprocessor'].transform(X)
nn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn_model.fit(X_transformed)

historical_events = valid_df[['event_cause', 'zone', 'start_datetime', 'actual_delay_mins']].reset_index(drop=True)

# Overwrite artifacts
os.makedirs('models', exist_ok=True)
with open('models/pipeline.pkl', 'wb') as f: pickle.dump(pipeline, f)
with open('models/metrics.pkl', 'wb') as f: pickle.dump(metrics, f)
with open('models/nn_model.pkl', 'wb') as f: pickle.dump(nn_model, f)
with open('models/historical_events.pkl', 'wb') as f: pickle.dump(historical_events, f)
with open('models/feature_names.pkl', 'wb') as f: pickle.dump(all_feature_names, f)
with open('models/feature_importances.json', 'w') as f: json.dump(feature_importances, f)

print("Retraining successful! Models overwritten.")
