import unittest
from unittest.mock import patch, MagicMock
from flask_app.app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch("flask_app.app.mlflow.pyfunc.load_model")
    @patch("flask_app.app.mlflow.MlflowClient")
    @patch("flask_app.app.pickle.load")
    def test_home(self, mock_pickle_load, mock_mlflow_client, mock_load_model):
        # Mock the vectorizer and model
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value.toarray.return_value = [[0, 1, 0]]
        mock_pickle_load.return_value = mock_vectorizer

        mock_model = MagicMock()
        mock_model.predict.return_value = ["positive"]
        mock_load_model.return_value = mock_model

        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<html", response.data)

    @patch("flask_app.app.mlflow.pyfunc.load_model")
    @patch("flask_app.app.mlflow.MlflowClient")
    @patch("flask_app.app.pickle.load")
    def test_predict(self, mock_pickle_load, mock_mlflow_client, mock_load_model):
        # Mock the vectorizer and model
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value.toarray.return_value = [[0, 1, 0]]
        mock_pickle_load.return_value = mock_vectorizer

        mock_model = MagicMock()
        mock_model.predict.return_value = ["positive"]
        mock_load_model.return_value = mock_model

        response = self.app.post("/predict", data={"text": "Test input"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"positive", response.data)

if __name__ == "__main__":
    unittest.main()
