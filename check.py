import os
import sys
import mlflow

THRESHOLD = 0.85

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    if not os.path.exists("model_info.txt"):
        print("model_info.txt not found")
        sys.exit(1)

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    if not run_id:
        print("Run ID is empty")
        sys.exit(1)

    run = mlflow.get_run(run_id)
    metrics = run.data.metrics

    if "accuracy" not in metrics:
        print("Accuracy metric not found in MLflow run")
        sys.exit(1)

    accuracy = metrics["accuracy"]
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")

    if accuracy < THRESHOLD:
        print(f"Deployment failed: accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
        sys.exit(1)

    print(f"Deployment passed: accuracy {accuracy:.4f} is above threshold {THRESHOLD}")

if __name__ == "__main__":
    main()
