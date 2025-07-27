from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='retrain_xgboost_energy_forecast',
    default_args=default_args,
    description='Monthly retraining of the XGBoost energy forecast model via MLflow',
    schedule_interval='@monthly',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    train_xgb = BashOperator(
        task_id='train_xgboost_mlflow',
        bash_command=(
            'python /opt/airflow/dags/ml/models/train_xgboost_mlflow.py '
            '--data-path /opt/airflow/data/processed/energy_features.csv '
            '--output-model-path /opt/airflow/ml/models/xgboost_model.pkl '
            '--mlflow-tracking-uri http://mlflow:5000 '
            '--experiment-name energy_forecast'
        )
    )

    train_xgb
