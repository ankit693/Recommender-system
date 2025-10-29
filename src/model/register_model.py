# register_model.py

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
import warnings

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ✅ Load DagsHub token from environment
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# ✅ Set MLflow authentication and tracking URI
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit693"
repo_name = "Recommender-system"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# -----------------------------
# Utility functions
# -----------------------------

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model in the MLflow Model Registry with alias support."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        client = mlflow.tracking.MlflowClient()

        # ✅ Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logging.debug(f"Model {model_name} registered with version {model_version.version}")

        # ✅ Assign alias (instead of deprecated 'stage')
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",  # replaces old Staging stage
            version=model_version.version
        )

        logging.debug(f"Alias 'staging' assigned to {model_name} version {model_version.version}")
        print(f"✅ Model '{model_name}' version {model_version.version} registered and tagged as 'staging'.")

    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)

    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
