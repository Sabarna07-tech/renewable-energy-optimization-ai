from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import numpy as np
import os
from prometheus_client import Counter, Histogram, make_asgi_app

# Load the XGBoost model at startup
model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'xgboost_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI(
    title="Energy Forecasting API",
    description="Predict power (KW) with XGBoost"
)

# Expose Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    'api_request_count_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'HTTP request latency in seconds',
    ['endpoint']
)

# Middleware to record metrics for each request
@app.middleware("http")
async def record_metrics(request: Request, call_next):
    endpoint = request.url.path
    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code
    ).inc()
    return response

# Input schema
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
            self.VOLTAGE,
            self.CURRENT,
            self.PF,
            self.Temp_F,
            self.Humidity,
            self.WEEKEND_WEEKDAY,
            self.SEASON,
            self.lag1,
            self.rolling_mean_3
        ]

# Prediction endpoint
@app.post("/predict")
def predict_power(input: EnergyInput):
    X = np.array(input.as_feature_list()).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {"predicted_power_kw": float(prediction)}
