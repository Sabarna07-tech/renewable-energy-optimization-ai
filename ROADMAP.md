# Roadmap

## Short-term goals
- Harden Kafka simulator and consumer integration with load tests and schema validation.
- Containerize and orchestrate the Airflow retraining pipeline alongside Postgres metadata storage.
- Expand monitoring with Prometheus scrape configs and Grafana dashboards tuned for energy metrics.
- Document an optional `docker-compose.full.yml` that runs the end-to-end stack (Kafka, MLflow, Airflow, Prometheus, Grafana).

## Looking ahead
- Publish public dataset samples and benchmarking notebooks.
- Add alerting hooks for anomaly detections and forecast drift.
- Explore hardware acceleration paths for on-prem deployments.
