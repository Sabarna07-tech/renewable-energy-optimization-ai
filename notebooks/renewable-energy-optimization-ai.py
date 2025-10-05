#!/usr/bin/env python
# coding: utf-8
# NOTE: This analysis supports load/demand forecasting; older labels referenced renewables but the model here predicts electric demand (kW).

# In[2]:


get_ipython().system("pip install scikit-learn pandas matplotlib seaborn prophet")


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_excel(
    r"E:\renewable-energy-optimization-ai\data\Active Power Load - 33_11KV - Godishala Substation .xlsx"
)  # Change path/filename as needed
# df['DATE'] = pd.to_datetime(df['DATE'])
# if 'TIME' in df.columns:
#     df['DATETIME'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
# else:
#     df['DATETIME'] = df['DATE']
# df = df.sort_values('DATETIME')
df.head()


# In[6]:


print(df[["DATE", "TIME"]].head(10))
print(df["TIME"].unique()[:10])  # See first 10 unique TIME values


# In[7]:


# Make sure DATE is string
df["DATE"] = df["DATE"].astype(str)
# Clean TIME, replace '-' with ':', pad if needed
df["TIME"] = df["TIME"].astype(str).str.replace("-", ":")
df["TIME"] = df["TIME"].str.zfill(5)  # Ensures "2:00" -> "02:00"

# Combine
dt_strings = df["DATE"] + " " + df["TIME"]

# Now parse to datetime, errors='coerce' will turn bad values into NaT (can drop them)
df["DATETIME"] = pd.to_datetime(dt_strings, errors="coerce")
df = df.dropna(subset=["DATETIME"])
df = df.sort_values("DATETIME")
df.head()


# In[8]:


print(df.info())
print(df.describe())
print(df.isnull().sum())


# In[11]:


drop_cols = ["Substation Shutdown", "F1", "F2", "F3", "F4", "Jul-Oct-", "Rainy", "0"]
df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])


# In[12]:


plt.figure(figsize=(15, 4))
plt.plot(df_clean["DATETIME"], df_clean["POWER (KW)"])
plt.title("Power (KW) Over Time")
plt.xlabel("Datetime")
plt.ylabel("Power (KW)")
plt.tight_layout()
plt.show()


# In[13]:


get_ipython().system("pip install prophet --quiet")
from prophet import Prophet

prophet_df = df_clean[["DATETIME", "POWER (KW)"]].rename(columns={"DATETIME": "ds", "POWER (KW)": "y"}).dropna()
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model.fit(prophet_df)
future = model.make_future_dataframe(periods=30, freq="D")  # Forecast next 30 days
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title("Power (KW) Forecast")
plt.show()


# In[14]:


from sklearn.ensemble import IsolationForest

numeric_cols = ["POWER (KW)", "VOLTAGE", "CURRENT", "PF", "Temp (F)", "Humidity (%)"]
features = df_clean[numeric_cols].fillna(0)
iso = IsolationForest(contamination=0.02, random_state=42)
df_clean["anomaly"] = iso.fit_predict(features)

# Plot anomalies
plt.figure(figsize=(15, 4))
plt.plot(df_clean["DATETIME"], df_clean["POWER (KW)"], label="Power (KW)")
plt.scatter(
    df_clean[df_clean["anomaly"] == -1]["DATETIME"],
    df_clean[df_clean["anomaly"] == -1]["POWER (KW)"],
    color="red",
    label="Anomaly",
    s=20,
)
plt.legend()
plt.title("Anomaly Detection in Power Data")
plt.show()


# In[15]:


# Prepare your dataframe for Prophet
prophet_df = df_clean[["DATETIME", "POWER (KW)"]].rename(columns={"DATETIME": "ds", "POWER (KW)": "y"}).dropna()

# Initialize and train the model (this is the training step!)
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model.fit(prophet_df)


# In[16]:


future = model.make_future_dataframe(periods=30, freq="D")
forecast = model.predict(future)


# In[17]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predict only for the time range of your actual data
forecast_in_sample = model.predict(prophet_df[["ds"]])

# Compare predicted vs. actual
y_true = prophet_df["y"].values
y_pred = forecast_in_sample["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot Actual vs. Predicted
plt.figure(figsize=(15, 4))
plt.plot(prophet_df["ds"], y_true, label="Actual")
plt.plot(prophet_df["ds"], y_pred, label="Prophet Predicted")
plt.legend()
plt.title("Actual vs. Prophet Predicted Power (KW)")
plt.tight_layout()
plt.show()


# In[18]:


df_xgb = df_clean.copy()
df_xgb = df_xgb.sort_values("DATETIME")

# Date parts
df_xgb["hour"] = df_xgb["DATETIME"].dt.hour
df_xgb["dayofweek"] = df_xgb["DATETIME"].dt.dayofweek
df_xgb["month"] = df_xgb["DATETIME"].dt.month

# Lag features (just lag1)
df_xgb["lag1"] = df_xgb["POWER (KW)"].shift(1)
# Rolling mean features (just window=3)
df_xgb["rolling_mean_3"] = df_xgb["POWER (KW)"].rolling(window=3).mean()

# Drop only NA from lag1 and rolling_mean_3
df_xgb = df_xgb.dropna(subset=["lag1", "rolling_mean_3"]).reset_index(drop=True)
df_xgb.head()


# In[19]:


print("Data shape after feature engineering:", df_xgb.shape)

split_idx = int(0.9 * len(df_xgb))  # try 90% train
train = df_xgb.iloc[:split_idx]
test = df_xgb.iloc[split_idx:]

print("Train size:", train.shape)
print("Test size:", test.shape)


# In[20]:


feature_cols = [
    "VOLTAGE",
    "CURRENT",
    "PF",
    "Temp (F)",
    "Humidity (%)",
    '"WEEKEND/WEEKDAY"',
    "SEASON",
    "lag1",
    "rolling_mean_3",
]
target_col = "POWER (KW)"

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]


# In[21]:


get_ipython().system("pip install xgboost --quiet")
import xgboost as xgb

model_xgb = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model_xgb.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

y_pred_xgb = model_xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}")

plt.figure(figsize=(15, 4))
plt.plot(test["DATETIME"], y_test, label="Actual")
plt.plot(test["DATETIME"], y_pred_xgb, label="XGBoost Predicted")
plt.legend()
plt.title("Actual vs. XGBoost Predicted Power (KW)")
plt.tight_layout()
plt.show()


# In[29]:


mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")


# In[30]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred_xgb)
print(f"R² Score: {r2:.3f} ({r2*100:.1f}%)")


# In[31]:


# MAPE
mape = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# R² Score
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred_xgb)
print(f"R² Score: {r2:.3f} ({r2*100:.1f}%)")


# In[32]:


import matplotlib.pyplot as plt
import xgboost as xgb

# Plot built-in feature importance
xgb.plot_importance(model_xgb, importance_type="gain", max_num_features=10)
plt.title("Top 10 XGBoost Feature Importances")
plt.show()

# If you want a pretty seaborn plot:
importances = model_xgb.feature_importances_
feat_names = X_train.columns
feat_imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.show()


# In[34]:


import pickle
import os

os.makedirs(r"E:\renewable-energy-optimization-ai\ml\models", exist_ok=True)
with open(r"E:\renewable-energy-optimization-ai\ml\models\xgboost_model.pkl", "wb") as f:
    pickle.dump(model_xgb, f)


# In[35]:


from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

# Define parameter grid to search
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


# In[36]:


tscv = TimeSeriesSplit(n_splits=5)  # Keeps order, good for time series


# In[37]:


xgb_reg = xgb.XGBRegressor(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    verbose=1,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)


# In[38]:


print("Best Hyperparameters:", grid_search.best_params_)
print("Best CV Score (MAE):", -grid_search.best_score_)


# In[39]:


best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mape_best = np.mean(np.abs((y_test - y_pred_best) / y_test)) * 100
r2_best = r2_score(y_test, y_pred_best)

print(f"MAE (best): {mae_best:.2f}")
print(f"RMSE (best): {rmse_best:.2f}")
print(f"MAPE (best): {mape_best:.2f}%")
print(f"R² Score (best): {r2_best:.3f} ({r2_best*100:.1f}%)")


# In[41]:


import pickle

with open(r"E:\renewable-energy-optimization-ai\ml\models\xgboost_best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Tuned model saved as xgboost_best_model.pkl")


# In[42]:


get_ipython().system("pip install mlflow --quiet")


# In[43]:


# Inside your XGBoost training/tuning notebook
import mlflow
import mlflow.sklearn

mlflow.set_experiment("energy-forecast-xgboost")

with mlflow.start_run(run_name="xgboost-tuned"):
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("mae", mae_best)
    mlflow.log_metric("rmse", rmse_best)
    mlflow.log_metric("mape", mape_best)
    mlflow.log_metric("r2", r2_best)
    mlflow.sklearn.log_model(best_model, "model")
    print("Logged run to MLflow")


# In[ ]:
