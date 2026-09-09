import joblib
import pandas as pd
import json
from pathlib import Path

class RainfallForecaster:
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Resolve path relative to this file to be CWD-independent
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            model_dir = project_root / "models" / "rainfall_forecasting" / "random_forest" / "v2"
        else:
            model_dir = Path(model_dir)

        self.model = joblib.load(model_dir / "model.pkl")
        with open(model_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

    def forecast(self, rainfall, lag_1, rolling_mean_3):
        features = pd.DataFrame([[rainfall, lag_1, rolling_mean_3]], columns=['RAINFALL', 'lag_1', 'rolling_mean_3'])
        prediction = self.model.predict(features)
        return {
            "forecasted_rainfall": float(prediction[0]),
            "model_version": self.metadata["model_version"],
            "data_state": "HISTORICAL"
        }

