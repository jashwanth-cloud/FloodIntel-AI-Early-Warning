import xarray as xr
import glob
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime


class RainfallForecasterTrainerV2:
    def __init__(
        self,
        data_dir="../data/raw/rainfall",
        model_dir="../models/rainfall_forecasting/random_forest/v2",
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_data(self):
        files = sorted(glob.glob(os.path.join(self.data_dir, "*.nc")))
        dfs = []

        for file in files:
            ds = xr.open_dataset(file)

            df = ds["RAINFALL"].to_dataframe().reset_index()

            # Same feature engineering as v1
            df["lag_1"] = (
                df.groupby(["LATITUDE", "LONGITUDE"])["RAINFALL"]
                .shift(1)
            )

            df["rolling_mean_3"] = (
                df.groupby(["LATITUDE", "LONGITUDE"])["RAINFALL"]
                .transform(lambda x: x.rolling(3).mean())
            )

            # Target: rainfall at T+1
            df["target"] = (
                df.groupby(["LATITUDE", "LONGITUDE"])["RAINFALL"]
                .shift(-1)
            )

            df = df.dropna()
            dfs.append(df)

            ds.close()

        return pd.concat(dfs, ignore_index=True)

    def train(self):
        df = self.prepare_data()

        # Same chronological training period as v1
        train_df = df[
            (df["TIME"].dt.year >= 2021)
            & (df["TIME"].dt.year <= 2022)
        ]

        X_train = train_df[
            ["RAINFALL", "lag_1", "rolling_mean_3"]
        ]
        y_train = train_df["target"]

        print(f"Training samples: {len(X_train)}")

        # Deployment-friendly constrained forest
        model = RandomForestRegressor(
            n_estimators=10,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Training-set metrics
        predictions = model.predict(X_train)

        mae = mean_absolute_error(y_train, predictions)
        rmse = np.sqrt(mean_squared_error(y_train, predictions))
        r2 = r2_score(y_train, predictions)

        print("\nV2 Training Metrics")
        print(f"MAE:  {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R2:   {r2:.6f}")

        # Save v2 model only
        model_path = os.path.join(self.model_dir, "model.pkl")
        joblib.dump(model, model_path)

        metadata = {
            "model_version": "v2",
            "train_years": "2021-2022",
            "target": "RAINFALL(T+1)",
            "features": [
                "RAINFALL",
                "lag_1",
                "rolling_mean_3",
            ],
            "n_estimators": 10,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "random_state": 42,
            "training_samples": int(len(X_train)),
            "train_mae": float(mae),
            "train_rmse": float(rmse),
            "train_r2": float(r2),
            "training_timestamp": datetime.now().isoformat(),
        }

        metadata_path = os.path.join(
            self.model_dir,
            "metadata.json",
        )

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print("\nForecasting V2 training complete.")
        print(f"Model saved to: {model_path}")

        return model


if __name__ == "__main__":
    trainer = RainfallForecasterTrainerV2(
        data_dir=r".\data\raw\rainfall",
        model_dir=r".\models\rainfall_forecasting\random_forest\v2",
    )
    trainer.train()