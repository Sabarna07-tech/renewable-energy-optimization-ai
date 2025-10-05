# Renewable Energy Optimization AI

A production-ready reference implementation for forecasting renewable energy outputs using an XGBoost regression model. The project combines streaming ingestion, scheduled retraining, and observability components so that data teams can deploy, monitor, and extend power predictions quickly.

## What it predicts & inputs
- **Prediction**: real-time estimate of generated power in kilowatts.
- **Required inputs** (JSON fields and order enforced internally): `VOLTAGE`, `CURRENT`, `PF`, `Temp_F`, `Humidity`, `WEEKEND_WEEKDAY`, `SEASON`, `lag1`, `rolling_mean_3`.
- **Model**: gradient boosted trees stored at `ml/models/xgboost_model.pkl`.

## Target audience
- Energy operations teams needing a lightweight forecasting API for on-prem or edge deployments.
- Data scientists prototyping energy models with simulated IoT streams before scaling to full production stacks.
- Platform engineers evaluating an end-to-end MLOps reference that still fits in a single repository.

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
- `serving/app.py` — FastAPI app loading the XGBoost model and exposing REST endpoints.
- `pipelines/dags/ml_pipeline.py` — Airflow DAG orchestrating retraining (optional).
- `iot_simulator/producer.py` — Kafka simulator for synthetic telemetry.
- `monitoring/` — Prometheus and Grafana configuration for metrics and dashboards.
- `docker-compose.min.yml` — minimal serving stack with the API container only.

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
