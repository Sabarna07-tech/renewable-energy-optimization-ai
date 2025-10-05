# Energy Load Forecasting (Power Demand)

A production-ready reference implementation for predicting near-term electric load (kW) using an XGBoost regression model. The project combines streaming ingestion, scheduled retraining, and observability components so demand-planning teams can deploy, monitor, and extend power forecasts quickly.

## What it predicts & inputs
- **Prediction**: near-term electric load / demand in kilowatts exposed as `predicted_power_kw`.
- **Required inputs** (JSON fields and order enforced internally): `VOLTAGE`, `CURRENT`, `PF`, `Temp_F`, `Humidity`, `WEEKEND_WEEKDAY`, `SEASON`, `lag1`, `rolling_mean_3`.
- **Signals explained**:
  - `VOLTAGE`, `CURRENT`, `PF`: electrical measurements reflecting instantaneous load.
  - `Temp_F`, `Humidity`: weather context influencing HVAC and equipment usage.
  - `WEEKEND_WEEKDAY`, `SEASON`: categorical demand patterns.
  - `lag1`, `rolling_mean_3`: recent load history for temporal continuity.
- **Model**: gradient boosted trees stored at `ml/models/xgboost_model.pkl`.

## Target audience
- Utilities and grid operators forecasting facility or feeder load.
- Microgrid and campus energy managers planning demand response programs.
- Facility operations and scheduling teams balancing asset usage.
- Energy analytics and MLOps engineers packaging load forecasters for production.

## Quickstart
```bash
docker compose -f docker-compose.min.yml up --build -d
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"VOLTAGE":230,"CURRENT":10.5,"PF":0.96,"Temp_F":82,"Humidity":55,"WEEKEND_WEEKDAY":0,"SEASON":3,"lag1":70.8,"rolling_mean_3":72.1}'
```
After the stack starts, browse interactive docs at: http://localhost:8000/docs

Bring the stack down with:
```bash
docker compose -f docker-compose.min.yml down
```

> **Note:** Out of scope for this release: solar/wind production forecasting. This project targets demand/load. A future module can combine on-site renewables to produce net-load estimates.

## API endpoints
- `GET /health` -> `{"status": "ok"}` for readiness probes.
- `POST /predict` -> returns `{ "predicted_power_kw": <float> }`.

Example response:
```json
{
  "predicted_power_kw": 71.42
}
```

## Project structure highlights
- `serving/app.py` - FastAPI app loading the XGBoost model and exposing REST endpoints.
- `pipelines/dags/ml_pipeline.py` - Airflow DAG orchestrating retraining (optional).
- `iot_simulator/producer.py` - Kafka simulator for synthetic telemetry.
- `monitoring/` - Prometheus and Grafana configuration for metrics and dashboards.
- `docker-compose.min.yml` - minimal serving stack with the API container only.

A future `docker-compose.full.yml` will orchestrate Kafka, MLflow, Airflow, Prometheus, and Grafana for full-stack demos.

## Development
1. Install dependencies: `pip install -r serving/requirements.txt`.
2. Run checks: `pytest -q`.
3. Format and lint with Ruff/Black via `pre-commit run --all-files`.
4. Build the container locally using the provided Dockerfile.

## Continuous integration
GitHub Actions runs linting (Ruff, Black), tests (pytest), and on pushes to `main` builds/pushes `ghcr.io/<owner>/<repo>/energy-backend:latest`.

## Contributing and security
- Follow the guidelines in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Report vulnerabilities per [`SECURITY.md`](SECURITY.md).
- Community expectations are defined in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## License
Distributed under the [Apache License 2.0](LICENSE).
