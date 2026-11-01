"""Prefect flow for training and tracking the baseline churn model."""

import os

import mlflow
import mlflow.sklearn
from prefect import flow, get_run_logger

from churn_ml.data import FEATURE_COLUMNS, generate_employee_data
from churn_ml.model import train_model


@flow(name="employee-churn-baseline")
def run_training(n_samples: int = 1_000, random_state: int = 42) -> dict[str, float]:
    """Generate example data, train a model, and log its run to MLflow."""
    logger = get_run_logger()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("employee-churn-baseline")

    data = generate_employee_data(n_samples=n_samples, random_state=random_state)
    model, metrics = train_model(data)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "model": "logistic_regression",
                "n_samples": n_samples,
                "random_state": random_state,
                "features": ",".join(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="employee-churn-model",
            input_example=data[FEATURE_COLUMNS].head(5),
        )
        logger.info("MLflow run %s completed with metrics %s", run.info.run_id, metrics)

    return metrics


if __name__ == "__main__":
    print(run_training())
