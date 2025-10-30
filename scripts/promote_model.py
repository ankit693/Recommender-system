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
    Promote the latest 'staging' model to 'production' in MLflow Model Registry.
    Uses manual stage search instead of alias API (for DagsHub compatibility).
    """
    client = MlflowClient()
    model_name = "my_model"

    # ---------------------------------------------------------------------
    # ✅ Step 1: Get version currently tagged as 'staging'
    # ---------------------------------------------------------------------
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        staging_version = None
        for v in versions:
            # Check either aliases list or stage name
            if "staging" in getattr(v, "aliases", []) or v.current_stage.lower() == "staging":
                staging_version = v
                break

        if staging_version is None:
            raise ValueError("No model version tagged or staged as 'staging' found.")

        staging_version_num = staging_version.version
        print(f"✅ Found model version {staging_version_num} tagged as 'staging'.")
    except Exception as e:
        raise ValueError(f"❌ Failed to locate 'staging' model version for '{model_name}': {e}")

    # ---------------------------------------------------------------------
    # ✅ Step 2: Find current 'production' model (if any)
    # ---------------------------------------------------------------------
    current_prod_version = None
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if "production" in getattr(v, "aliases", []) or v.current_stage.lower() == "production":
                current_prod_version = v.version
                print(f"ℹ️ Current 'production' model version: {current_prod_version}")
                break
        if current_prod_version is None:
            print("⚠️ No current 'production' model found. This will be the first promotion.")
    except MlflowException:
        print("⚠️ Could not retrieve existing 'production' version.")
    except Exception as e:
        print(f"⚠️ Unexpected error fetching 'production': {e}")

    # ---------------------------------------------------------------------
    # ✅ Step 3: Promote staging → production
    # ---------------------------------------------------------------------
    try:
        # Update the stage (DagsHub honors this field)
        client.transition_model_version_stage(
            name=model_name,
            version=staging_version_num,
            stage="Production",
            archive_existing_versions=False
        )
        print(f"🚀 Model version {staging_version_num} promoted to 'Production' stage.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to promote model to production: {e}")

    # ---------------------------------------------------------------------
    # ✅ Step 4: Optional cleanup or reporting
    # ---------------------------------------------------------------------
    if current_prod_version and current_prod_version != staging_version_num:
        print(f"📦 Previous production model (v{current_prod_version}) retained. Consider archiving manually.")

    print("✅ Promotion completed successfully.")


if __name__ == "__main__":
    promote_model()
