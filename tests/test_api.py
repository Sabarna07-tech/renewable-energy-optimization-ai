from fastapi.testclient import TestClient

from serving.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    body = {
        "VOLTAGE": 230.0,
        "CURRENT": 10.5,
        "PF": 0.96,
        "Temp_F": 82.0,
        "Humidity": 55.0,
        "WEEKEND_WEEKDAY": 0,
        "SEASON": 3,
        "lag1": 70.8,
        "rolling_mean_3": 72.1,
    }
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "predicted_power_kw" in j
    assert isinstance(j["predicted_power_kw"], float)
