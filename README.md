# Renewable Energy Optimization AI

An end-to-end machine learning pipeline for forecasting, anomaly detection, and automated retraining in renewable energy smart grids. Leverages simulated IoT streams → Apache Kafka → FastAPI consumer → CSV storage → Jupyter data exploration → XGBoost forecasting + anomaly detection → MLflow experiment tracking → Airflow–orchestrated monthly retraining → FastAPI serving → Grafana dashboards.

---

## 🔍 Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Prerequisites](#prerequisites)
6. [Quickstart Installation](#quickstart-installation)
7. [Data Ingestion](#data-ingestion)
   * [IoT Simulator](#iot-simulator)
   * [Kafka Setup & Consumer](#kafka-setup--consumer)
8. [Exploratory Data Analysis](#exploratory-data-analysis)
9. [Model Training & Tuning](#model-training--tuning)
   * [Training Script (`train_xgboost_mlflow.py`)](#training-script-train_xgboost_mlflowpy)
   * [MLflow Tracking](#mlflow-tracking)
10. [Model Serving (FastAPI)](#model-serving-fastapi)
11. [Automated Retraining (Airflow)](#automated-retraining-airflow)
12. [Monitoring & Dashboards](#monitoring--dashboards)
13. [Docker Support](#docker-support)
14. [Contributing](#contributing)
15. [License](#license)

---

## ✨ Features

* **Streaming Ingestion**: Simulate IoT device data → Kafka topics
* **CSV Storage**: Lightweight time-series storage for prototyping
* **Forecasting**: XGBoost model with lag & rolling features
* **Anomaly Detection**: Isolation Forest on power/voltage/time features
* **Experiment Tracking**: MLflow logging of parameters, metrics, and model artifacts
* **Model Serving**: RESTful FastAPI endpoint for real-time predictions
* **Automated Retraining**: Airflow DAG to retrain & log monthly to MLflow
* **Dashboards**: Grafana integration for live metrics & forecasts

---

## 🏗 Architecture Overview

```
┌──────────────┐      ┌────────┐      ┌───────────────┐      ┌────────┐      ┌─────────────┐
│ IoT Simulator│ ──▶  │ Kafka  │ ──▶  │ FastAPI       │ ──▶  │ CSV    │ ──▶  │  Jupyter    │
│ (producer.py)│      │(topics)│      │ Consumer      │      │ storage│      │  Notebook   │
└──────────────┘      └────────┘      │ (kafka_consumer│      └────────┘      │ (EDA & ML)  │
                                        └───────────────┘                      └─────────────┘
                                                                              │
                                                                              ▼
                                                                      ┌───────────────┐
                                                                      │ train_xgboost │
                                                                      │ _mlflow.py    │
                                                                      └───────────────┘
                                                                              │
                                                                              ▼
                                                   ┌──────────┐          ┌───────────────┐
                                                   │ MLflow   │◀─────────┤ Airflow DAG   │
                                                   │ Tracking │          │ (ml_pipeline) │
                                                   └──────────┘          └───────────────┘
                                                                              │
                                                                              ▼
                                                                       ┌───────────────┐
                                                                       │ FastAPI       │
                                                                       │ Serving App   │
                                                                       └───────────────┘
                                                                              │
                                                                              ▼
                                                                       ┌───────────────┐
                                                                       │ Grafana       │
                                                                       │ Dashboards    │
                                                                       └───────────────┘
```

---

## 🛠 Tech Stack

* **Language & Data**: Python, Pandas, NumPy
* **Streaming**: Apache Kafka (producer & consumer)
* **ML Models**: XGBoost, Prophet, Isolation Forest
* **ML Workflow**: scikit-learn, GridSearchCV, Airflow, MLflow
* **API Serving**: FastAPI, Uvicorn
* **Visualization**: Matplotlib, Seaborn, Grafana
* **Orchestration**: Airflow (Docker Compose + Postgres)
* **Containerization**: Docker, Docker Compose

---

## 📂 Repository Structure

```
.
├── data/                         # Raw and processed data
│   └── energy_data.csv
├── docker/                       # Dockerfiles for each service
│   ├── airflow.Dockerfile
│   ├── fastapi.Dockerfile
│   ├── grafana.Dockerfile
│   └── kafka.Dockerfile
├── iot_simulator/                # Simulated IoT producer
│   ├── producer.py
│   └── requirements.txt
├── kafka_consumer/               # FastAPI consumer & CSV storage
│   └── main.py
├── ml/                           # ML code
│   ├── anomaly/
│   │   └── isolation_forest.py
│   ├── forecasting/
│   │   └── prophet_train.py
│   ├── models/
│   │   └── xgboost_model.pkl
│   └── utils.py
├── mlruns/                       # Auto-generated MLflow runs
├── notebooks/                    # Exploratory notebooks
│   └── renewable-energy-optimization-ai.ipynb
├── pipelines/
│   └── dags/
│       ├── ml_pipeline.py        # Airflow DAG
│       └── (ignore airflow.cfg)  # remove this
├── serving/                      # FastAPI serving app
│   ├── app.py
│   └── requirements.txt
├── docker-compose-airflow.yml    # Airflow & Postgres stack
├── docker-compose.yml            # Full stack (Kafka, FastAPI, Grafana, etc.)
├── requirements.txt              # Python dependencies
├── README.md
└── LICENSE
```

---

## 🖥 Prerequisites

* Docker & Docker Compose
* Python 3.8–3.11
* Kafka & Zookeeper (if running outside Docker)
* (Optional) Grafana & Postgres for dashboards

---

## 🚀 Quickstart Installation

1. **Clone repository**

   ```bash
   git clone https://github.com/Sabarna07-tech/renewable-energy-optimization-ai.git
   cd renewable-energy-optimization-ai
   ```

2. **Python environment**

   ```bash
   python -m venv env
   source env/bin/activate      # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```


3. **Simulate IoT → Kafka**

   * Start Kafka & Zookeeper:

     ```bash
     docker compose up -d zookeeper kafka  # or `docker-compose`
     ```

   * In another terminal, run:

     ```bash
     cd iot_simulator
     pip install -r requirements.txt
     python producer.py
     ```

   * If `NoBrokersAvailable` appears, confirm Kafka is running and reachable at `localhost:9092`.
   * Topic `energy-data` will be created automatically and populated with JSON messages.


4. **Run Kafka Consumer → CSV**

   ```bash
   cd kafka_consumer
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8001
   ```

   Incoming messages will be appended to `data/energy_data.csv`.
   *Note*: this service requires `email-validator>=2`, which conflicts with Airflow's dependencies. Run it in a separate virtual environment from Airflow.

---

## 📊 Exploratory Data Analysis

Open `notebooks/renewable-energy-optimization-ai.ipynb` to explore:

* Time-series plots
* Feature distributions
* Missing-value handling
* Basic Prophet forecasting
* Anomaly detection visualization

---

## 🤖 Model Training & Tuning

### Training Script

`ml/models/train_xgboost_mlflow.py` handles:

1. Data loading & cleaning
2. Feature engineering (lags, rolling means, date parts)
3. Train/test split
4. GridSearchCV hyperparameter tuning
5. Metrics evaluation (MAE, RMSE, MAPE, R²)
6. Model serialization (`xgboost_best_model.pkl`)
7. MLflow logging (params, metrics, artifact)

**Run locally**:

```bash
python ml/models/train_xgboost_mlflow.py
```

### MLflow Tracking

* **Start MLflow UI**:

  ```bash
  mlflow ui
  ```
* **Browse** at [http://localhost:5000](http://localhost:5000)
* Compare runs, parameters, and models

---

## 🌐 Model Serving (FastAPI)

The serving app in `serving/app.py`:

* Loads `xgboost_best_model.pkl`
* Defines `POST /predict` endpoint accepting JSON feature input
* Returns real-time `predicted_power_kw`

**Run**:

```bash
cd serving
pip install -r requirements.txt
uvicorn app:app --reload --port 8002
```

---

## ⏰ Automated Retraining (Airflow)

We use **Docker Compose** with Postgres for stable metadata storage.

1. **Initialize & start**:

   ```bash
   docker-compose -f docker-compose-airflow.yml up -d
   docker-compose -f docker-compose-airflow.yml run --rm airflow-webserver airflow db init
   docker-compose -f docker-compose-airflow.yml up -d
   ```

2. **Login** at [http://localhost:8080](http://localhost:8080)

   * Username: `admin` / Password: `admin`

3. **DAG**: `retrain_xgboost_energy_forecast` runs **monthly**, calls:

   ```bash
   python /opt/airflow/dags/train_xgboost_mlflow.py
   ```

4. **Results**: new MLflow run per schedule

---


## 📈 Monitoring & Dashboards

Prometheus and Grafana are part of the default Docker stack.

1. Start the monitoring services:

   ```bash
   docker compose up -d prometheus grafana
   ```

2. Visit **http://localhost:3001** and log in with `admin` / `admin`.

Grafana is pre-provisioned with a Prometheus data source and a sample
"System Overview" dashboard located in `monitoring/grafana/dashboards/`.


---

## 🐳 Docker Support

* **Root `docker-compose.yml`** can orchestrate:

  * Kafka & Zookeeper
  * IoT simulator
  * FastAPI consumer
  * FastAPI serving
  * Grafana
* **Airflow Compose** in `docker-compose-airflow.yml`

---

## 🤝 Contributing

1. Fork & clone
2. Create a branch: `git checkout -b feature/awesome`
3. Commit your changes & push
4. Open a Pull Request

Please follow the [Apache 2.0](LICENSE) license and Python code style.

---

## 📄 License

This project is licensed under the **Apache 2.0 License** – see the [LICENSE](LICENSE) file for details.
