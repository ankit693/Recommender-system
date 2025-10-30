# scripts/promote_model.py
import os
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# -------------------------------------------------------------------------
# ✅ Setup DagsHub MLflow Authentication FIRST
# -------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("❌ CAPSTONE_TEST environment variable is not set.")

# Set DagsHub credentials for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define your DagsHub repo MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "ankit693"
repo_name = "Recommender-System"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
print("🔍 Using MLflow Tracking URI:", mlflow.get_tracking_uri())


def promote_model():
    """
    Promote the latest 'staging' model to 'production' in MLflow Model Registry
    using model aliases (requires MLflow >= 2.9).
    """
    client = MlflowClient()
    model_name = "my_model"

    # ---------------------------------------------------------------------
    # ✅ Step 1: Get version currently tagged as 'staging'
    # ---------------------------------------------------------------------
    try:
        staging_version = client.get_model_version_by_alias(model_name, "staging")
        staging_version_num = staging_version.version
        print(f"✅ Found model version {staging_version_num} tagged as 'staging'.")
    except MlflowException as e:
        raise ValueError(f"❌ No 'staging' model alias found for '{model_name}': {e}")
    except Exception as e:
        raise ValueError(f"❌ Unexpected error while fetching staging alias: {e}")

    # ---------------------------------------------------------------------
    # ✅ Step 2: Check for current 'production' model (if any)
    # ---------------------------------------------------------------------
    current_prod_version = None
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        current_prod_version = prod_version.version
        print(f"ℹ️ Current 'production' model version: {current_prod_version}")
    except MlflowException:
        print("⚠️ No current 'production' model found. This will be the first promotion.")
    except Exception as e:
        print(f"⚠️ Unable to fetch 'production' alias: {e}")

    # ---------------------------------------------------------------------
    # ✅ Step 3: Promote staging → production
    # ---------------------------------------------------------------------
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version_num
    )
    print(f"🚀 Model version {staging_version_num} promoted to 'production'.")

    # ---------------------------------------------------------------------
    # ✅ Step 4: (Optional) Cleanup old alias or logging
    # ---------------------------------------------------------------------
    if current_prod_version and current_prod_version != staging_version_num:
        try:
            client.delete_registered_model_alias(
                name=model_name,
                alias="staging"  # optional cleanup — remove old staging alias
            )
            print(f"📦 Old staging alias removed. Previous production version: {current_prod_version}.")
        except Exception as e:
            print(f"⚠️ Could not remove old staging alias: {e}")

    print("✅ Promotion completed successfully.")


if __name__ == "__main__":
    promote_model()
