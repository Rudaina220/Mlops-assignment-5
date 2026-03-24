import os
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics.get("best_accuracy", 0)

print("Best accuracy:", acc)

if acc < 0.85:
    raise ValueError(f"Model accuracy {acc:.4f} is below threshold 0.85")
