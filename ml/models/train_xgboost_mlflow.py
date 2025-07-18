import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn

# 1. Load data
DATA_PATH = r'E:\renewable-energy-optimization-ai\data\Active Power Load - 33_11KV - Godishala Substation .xlsx'
df = pd.read_excel(DATA_PATH)

# 2. Preprocessing
df['DATE'] = df['DATE'].astype(str)
df['TIME'] = df['TIME'].astype(str).str.replace('-', ':').str.zfill(5)
dt_strings = df['DATE'] + ' ' + df['TIME']
df['DATETIME'] = pd.to_datetime(dt_strings, errors='coerce')
df = df.dropna(subset=['DATETIME']).sort_values('DATETIME')

drop_cols = ['Substation Shutdown', 'F1', 'F2', 'F3', 'F4', 'Jul-Oct-', 'Rainy', '0']
df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 3. Feature engineering
df_xgb = df_clean.copy().sort_values('DATETIME')
df_xgb['hour'] = df_xgb['DATETIME'].dt.hour
df_xgb['dayofweek'] = df_xgb['DATETIME'].dt.dayofweek
df_xgb['month'] = df_xgb['DATETIME'].dt.month
df_xgb['lag1'] = df_xgb['POWER (KW)'].shift(1)
df_xgb['rolling_mean_3'] = df_xgb['POWER (KW)'].rolling(window=3).mean()
df_xgb = df_xgb.dropna(subset=['lag1', 'rolling_mean_3']).reset_index(drop=True)

# 4. Train/test split
split_idx = int(0.9 * len(df_xgb))
train = df_xgb.iloc[:split_idx]
test = df_xgb.iloc[split_idx:]

feature_cols = [
    'VOLTAGE', 'CURRENT', 'PF', 'Temp (F)', 'Humidity (%)',
    '"WEEKEND/WEEKDAY"', 'SEASON',
    'lag1', 'rolling_mean_3'
]
target_col = 'POWER (KW)'
X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

# 5. Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
tscv = TimeSeriesSplit(n_splits=5)
xgb_reg = xgb.XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 6. Evaluate and log
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mape_best = np.mean(np.abs((y_test - y_pred_best) / y_test)) * 100
r2_best = r2_score(y_test, y_pred_best)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"MAE (best): {mae_best:.2f}")
print(f"RMSE (best): {rmse_best:.2f}")
print(f"MAPE (best): {mape_best:.2f}%")
print(f"R² Score (best): {r2_best:.3f} ({r2_best*100:.1f}%)")

# 7. Save model
MODEL_PATH = r'E:\renewable-energy-optimization-ai\ml\models\xgboost_best_model.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(best_model, f)
print(f"Tuned model saved as {MODEL_PATH}")

# 8. MLflow logging
mlflow.set_experiment("energy-forecast-xgboost")
with mlflow.start_run(run_name="xgboost-tuned"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mae", mae_best)
    mlflow.log_metric("rmse", rmse_best)
    mlflow.log_metric("mape", mape_best)
    mlflow.log_metric("r2", r2_best)
    mlflow.sklearn.log_model(best_model, "model")
    print("Logged run to MLflow")
