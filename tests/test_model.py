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

        # -------------------------------------------------------------------------
        # ✅ MLflow Authentication for DagsHub
        # -------------------------------------------------------------------------
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "ankit693"
        repo_name = "Recommender-System"

        # Set MLflow tracking URI for DagsHub
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
        print("🔍 Using MLflow Tracking URI:", mlflow.get_tracking_uri())

        # -------------------------------------------------------------------------
        # ✅ Load model, vectorizer, and test data
        # -------------------------------------------------------------------------
        cls.new_model_name = "my_model"
        cls.new_model_uri = cls.get_model_uri_by_stage(cls.new_model_name, stage="staging")
        print(f"✅ Fetching model from URI: {cls.new_model_uri}")

        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        cls.vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    # -------------------------------------------------------------------------
    # ✅ Helper function: Get model URI by stage (staging/production)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_model_uri_by_stage(model_name, stage="staging"):
        """
        Get MLflow model URI using stage ('staging' or 'production').
        Fallback to latest version if stage not found.
        """
        client = mlflow.MlflowClient()

        # Search all model versions
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None

        # Pick version matching the requested stage
        for v in versions:
            if v.current_stage.lower() == stage.lower():
                target_version = v
                break

        if target_version:
            print(f"✅ Using model stage '{stage}' → version {target_version.version}")
            return f"models:/{model_name}/{target_version.version}"

        # Fallback to latest version
        if versions:
            latest_version = max([int(v.version) for v in versions])
            print(f"⚠️ Stage '{stage}' not found, falling back to latest version {latest_version}")
            return f"models:/{model_name}/{latest_version}"

        raise ValueError(f"No available versions found for model '{model_name}'")

    # -------------------------------------------------------------------------
    # ✅ TEST: Model loads correctly
    # -------------------------------------------------------------------------
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model, "Model should not be None after loading.")

    # -------------------------------------------------------------------------
    # ✅ TEST: Model signature and input/output shapes
    # -------------------------------------------------------------------------
    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        prediction = self.new_model.predict(input_df)

        # Verify input feature count matches vectorizer
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out()),
            "Input feature count should match vectorizer features."
        )

        # Verify output shape
        self.assertEqual(len(prediction), input_df.shape[0], "Output length mismatch.")
        self.assertEqual(len(prediction.shape), 1, "Prediction should be a 1D array.")

    # -------------------------------------------------------------------------
    # ✅ TEST: Model performance on holdout data
    # -------------------------------------------------------------------------
    def test_model_performance(self):
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)

        print(f"📊 Model performance: accuracy={accuracy_new:.3f}, precision={precision_new:.3f}, recall={recall_new:.3f}, f1={f1_new:.3f}")

        # Minimum thresholds
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f"Accuracy should be ≥ {expected_accuracy}")
        self.assertGreaterEqual(precision_new, expected_precision, f"Precision should be ≥ {expected_precision}")
        self.assertGreaterEqual(recall_new, expected_recall, f"Recall should be ≥ {expected_recall}")
        self.assertGreaterEqual(f1_new, expected_f1, f"F1 score should be ≥ {expected_f1}")


# -------------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
