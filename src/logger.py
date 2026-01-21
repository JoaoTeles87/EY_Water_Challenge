import json
import datetime
import os
import uuid
import numpy as np

class ExperimentLogger:
    def __init__(self, experiment_name: str, log_dir: str = "../experiments", user: str = "User"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.user = user
        self.run_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.datetime.now().isoformat()
        
        # Create log dir if not exists
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "experiment_log.jsonl") # JSON Lines for appendability

    def log_run(self, params: dict, metrics: dict, notes: str = ""):
        """
        Logs a single experiment run to the JSONL file.
        """
        # Convert numpy types to python native for JSON serialization
        metrics_clean = {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in metrics.items()}
        params_clean = {k: str(v) for k, v in params.items()}

        entry = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "experiment_name": self.experiment_name,
            "user": self.user,
            "parameters": params_clean,
            "metrics": metrics_clean,
            "notes": notes
        }

        # Append to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        
        print(f"✅ Experiment logged with Run ID: {self.run_id}")
        return self.run_id

# Example Usage
if __name__ == "__main__":
    logger = ExperimentLogger("Test_Run")
    logger.log_run({"lr": 0.01}, {"rmse": 0.5}, "Developing logger class")
