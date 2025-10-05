import argparse
import os

import joblib
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb

from ml.utils import feature_engineering, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model with MLflow tracking")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/energy_features.csv",
        help="Path to processed feature CSV",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="ml/models/xgboost_model.pkl",
        help="Path to save the trained model artifact",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="energy_forecast",
        help="MLflow experiment name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Load and prepare data
        df = load_data(args.data_path)
        df = feature_engineering(df)
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        }

        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Log best parameters
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Evaluate on test set
        preds = grid_search.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = mse**0.5
        mlflow.log_metric("rmse", rmse)

        # Log model to MLflow
        mlflow.xgboost.log_model(grid_search.best_estimator_, "model")

        # Save model artifact locally
        joblib.dump(grid_search.best_estimator_, args.output_model_path)
        mlflow.log_artifact(args.output_model_path)


if __name__ == "__main__":
    main()
