from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

# 1. Load the model at startup
model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'xgboost_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 2. Define the features (order must match training!)
feature_cols = [
    'VOLTAGE', 'CURRENT', 'PF', 'Temp (F)', 'Humidity (%)',
    '"WEEKEND/WEEKDAY"', 'SEASON',
    'lag1', 'rolling_mean_3'
]

# 3. Set up FastAPI and the input schema
app = FastAPI(title="Energy Forecasting API", description="Predict power (KW) with XGBoost")

class EnergyInput(BaseModel):
    VOLTAGE: float
    CURRENT: float
    PF: float
    Temp_F: float
    Humidity: float
    WEEKEND_WEEKDAY: int
    SEASON: int
    lag1: float
    rolling_mean_3: float

    def as_feature_list(self):
        return [
            self.VOLTAGE, self.CURRENT, self.PF, self.Temp_F, self.Humidity,
            self.WEEKEND_WEEKDAY, self.SEASON, self.lag1, self.rolling_mean_3
        ]

@app.post("/predict")
def predict_power(input: EnergyInput):
    X = np.array(input.as_feature_list()).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {"predicted_power_kw": float(prediction)}
