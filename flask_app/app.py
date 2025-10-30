# app.py

from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import warnings
import numpy as np

# --------------------- SETUP ---------------------
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# --------------------- TEXT PREPROCESSING ---------------------
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# --------------------- DAGSHUB + MLFLOW SETUP ---------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit693"
repo_name = "Recommendation-System"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# --------------------- FLASK APP ---------------------
app = Flask(__name__)

# --------------------- PROMETHEUS METRICS ---------------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# --------------------- MODEL LOADING ---------------------
model_name = "my_model"

def get_model_uri_by_stage(model_name, stage="production"):
    """
    Get MLflow model URI using stage (production/staging).
    Fallback to latest version if stage not found.
    """
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    target_version = None

    # Try to find the version in the requested stage
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


# Load model and vectorizer
model_uri = get_model_uri_by_stage(model_name, stage="production")
print(f"Fetching model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# --------------------- ROUTES ---------------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    text = normalize_text(text)
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    result = model.predict(features_df)
    prediction = result[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# --------------------- MAIN ENTRY POINT ---------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
