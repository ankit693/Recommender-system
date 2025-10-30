# scripts/promote_model.py

import os
import mlflow

# --------------------- CONFIG ---------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit693"
repo_name = "Recommendation-System"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "my_model"

# --------------------- PROMOTION FUNCTION ---------------------
def promote_model():
    client = mlflow.MlflowClient()
    print(f"🔍 Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    # Search for all versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"❌ No versions found for model '{model_name}'")

    # Try to find staging version
    staging_versions = [v for v in versions if v.current_stage.lower() == "staging"]

    if staging_versions:
        # Pick first staging version
        version_to_promote = int(staging_versions[0].version)
        print(f"✅ Found staging version {version_to_promote}")
    else:
        # Fallback to latest version if no staging exists
        version_to_promote = max(int(v.version) for v in versions)
        print(f"⚠️ No staging version found. Using latest version {version_to_promote} instead.")

    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=version_to_promote,
        stage="Production",
        archive_existing_versions=True  # Archive previous production versions
    )

    print(f"🎉 Model '{model_name}' version {version_to_promote} is now in PRODUCTION.")

# --------------------- MAIN ---------------------
if __name__ == "__main__":
    promote_model()
