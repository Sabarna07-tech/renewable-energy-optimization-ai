import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Renewable Energy Optimization AI", version="1.0.0")

FEATURE_ORDER = [
    "VOLTAGE",
    "CURRENT",
    "PF",
    "Temp_F",
    "Humidity",
    "WEEKEND_WEEKDAY",
    "SEASON",
    "lag1",
    "rolling_mean_3",
]

COLUMN_RENAMES = {
    "Temp_F": "Temp (F)",
    "Humidity": "Humidity (%)",
    "WEEKEND_WEEKDAY": '"WEEKEND/WEEKDAY"',
}
MODEL_FEATURE_ORDER = [COLUMN_RENAMES.get(name, name) for name in FEATURE_ORDER]


class Features(BaseModel):
    VOLTAGE: float
    CURRENT: float
    PF: float = Field(..., ge=0.0, le=1.0)
    Temp_F: float
    Humidity: float
    WEEKEND_WEEKDAY: int = Field(..., ge=0, le=1)
    SEASON: int = Field(..., ge=0, le=3)
    lag1: float
    rolling_mean_3: float


class Prediction(BaseModel):
    predicted_power_kw: float


model = joblib.load("ml/models/xgboost_model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(payload: Features):
    try:
        df = pd.DataFrame(
            [[getattr(payload, key) for key in FEATURE_ORDER]],
            columns=FEATURE_ORDER,
        ).rename(columns=COLUMN_RENAMES)
        df = df[MODEL_FEATURE_ORDER]
        y = float(model.predict(df)[0])
        if not np.isfinite(y):
            raise ValueError("non-finite prediction")
        return {"predicted_power_kw": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
