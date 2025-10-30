import mlflow

mlflow.set_tracking_uri("https://dagshub.com/ankit693/Recommendation-System.mlflow")

client = mlflow.MlflowClient()

try:
    models = client.search_registered_models()
    for model in models:
        print(model.name)
except Exception as e:
    print("Failed to list registered models:", e)
