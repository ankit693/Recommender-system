# promote_model.py

import os
import mlflow

def promote_model():
    """Promote the latest 'staging' model to 'production' using MLflow aliases (>=2.9)."""
    # -------------------------------------------------------------------------
    # ✅ Setup DagsHub MLflow Authentication
    # -------------------------------------------------------------------------
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "ankit693"
    repo_name = "Recommender-System"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

    # -------------------------------------------------------------------------
    # ✅ Initialize MLflow client
    # -------------------------------------------------------------------------
    client = mlflow.MlflowClient()
    model_name = "my_model"

    # -------------------------------------------------------------------------
    # ✅ Step 1: Get version currently tagged as 'staging'
    # -------------------------------------------------------------------------
    try:
        staging_version = client.get_model_version_by_alias(model_name, "staging")
        staging_version_num = staging_version.version
        print(f"✅ Found model version {staging_version_num} tagged as 'staging'.")
    except Exception as e:
        raise ValueError(f"❌ No 'staging' model alias found for '{model_name}': {e}")

    # -------------------------------------------------------------------------
    # ✅ Step 2: Find current 'production' model (if any)
    # -------------------------------------------------------------------------
    current_prod_version = None
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        current_prod_version = prod_version.version
        print(f"ℹ️ Current 'production' model version: {current_prod_version}")
    except Exception:
        print("⚠️ No current 'production' model found. This will be the first promotion.")

    # -------------------------------------------------------------------------
    # ✅ Step 3: Move production alias to new version .
    # -------------------------------------------------------------------------
    # Assign alias 'production' to staging version
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=staging_version_num
    )
    print(f"🚀 Model version {staging_version_num} promoted to 'production'.")

    # -------------------------------------------------------------------------
    # ✅ Step 4: (Optional) Archive old production
    # -------------------------------------------------------------------------
    if current_prod_version and current_prod_version != staging_version_num:
        # You can tag or log this version as "archived" if needed
        client.delete_registered_model_alias(
            name=model_name,
            alias="staging"  # optional cleanup — remove old staging alias
        )
        print(f"📦 Old production model version {current_prod_version} retained (you can manually archive if desired).")

    print("✅ Promotion completed successfully.")


if __name__ == "__main__":
    promote_model()
