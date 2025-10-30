# tests/test_model_loading_and_performance.py

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up MLflow connection, load model, vectorizer, and test data."""
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ankit693"
        repo_name = "Recommender-System"

        # ✅ Set MLflow tracking URI for DagsHub
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Load latest model
        cls.new_model_name = "my_model"
        cls.new_model_uri = cls.get_model_uri_by_alias(cls.new_model_name, alias="staging")
        print(f"✅ Fetching model from URI: {cls.new_model_uri}")

        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    # ----------------------------------------------------------------
    # ✅ Updated helper function for MLflow ≥ 2.9
    # ----------------------------------------------------------------
    @staticmethod
    def get_model_uri_by_alias(model_name, alias="staging"):
        """Get model URI using MLflow model alias; fallback to latest version if alias not found."""
        client = mlflow.MlflowClient()
        try:
            version = client.get_model_version_by_alias(model_name, alias)
            if version:
                print(f"✅ Using model alias '{alias}' → version {version.version}")
                return f"models:/{model_name}@{alias}"
        except Exception as e:
            print(f"⚠️ Alias '{alias}' not found: {e}")

        # Fallback to latest version if alias missing
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = max([int(v.version) for v in versions])
            print(f"✅ Falling back to latest model version: {latest_version}")
            return f"models:/{model_name}/{latest_version}"

        raise ValueError(f"No available versions found for model '{model_name}'")

    # ----------------------------------------------------------------
    # ✅ TESTS
    # ----------------------------------------------------------------
    def test_model_loaded_properly(self):
        """Ensure the MLflow model loaded successfully."""
        self.assertIsNotNone(self.new_model, "Model should not be None after loading.")

    def test_model_signature(self):
        """Test model input and output shapes."""
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        prediction = self.new_model.predict(input_df)

        # Verify input feature count matches vectorizer output
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out()),
            "Input feature count should match vectorizer features."
        )

        # Verify output shape — one prediction per input
        self.assertEqual(len(prediction), input_df.shape[0], "Output length mismatch.")
        self.assertEqual(len(prediction.shape), 1, "Prediction should be a 1D array.")

    def test_model_performance(self):
        """Validate model performance on holdout data."""
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)

        print(f"📊 Model performance: accuracy={accuracy_new:.3f}, precision={precision_new:.3f}, recall={recall_new:.3f}, f1={f1_new:.3f}")

        # ✅ Minimum performance thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f"Accuracy should be ≥ {expected_accuracy}")
        self.assertGreaterEqual(precision_new, expected_precision, f"Precision should be ≥ {expected_precision}")
        self.assertGreaterEqual(recall_new, expected_recall, f"Recall should be ≥ {expected_recall}")
        self.assertGreaterEqual(f1_new, expected_f1, f"F1 score should be ≥ {expected_f1}")

# ----------------------------------------------------------------
# MAIN ENTRY POINT for tests
# ----------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
