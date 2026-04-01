"""
training/forecast_trainer.py — Train and save the LSTM multi-step forecaster.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

import config
from data.forecast_preprocessor import prepare_forecast
from models.forecaster import LSTMForecaster, FORECAST_STEPS
from utils.logger import get_logger

log = get_logger(__name__)

FORECASTER_FILE = "forecaster.keras"
FORECAST_SCALER_FILE = "forecast_scaler.joblib"
FORECAST_FEATURES_FILE = "forecast_feature_list.json"
FORECAST_Y_SCALE_FILE = "forecast_y_scale.json"


class ForecastTrainer:
    """
    End-to-end training pipeline for the LSTM multi-step forecaster.
    """

    def __init__(self, model_dir: str = config.MODEL_DIR) -> None:
        self.model_dir  = model_dir
        self.forecaster = LSTMForecaster(steps=FORECAST_STEPS)
        self.scaler     = None
        self.y_scale    = 1.0
        self.feature_names: list[str] = []

    def run(self, df: "pd.DataFrame") -> None:
        """Train the forecaster on *df* and save all artefacts."""
        import pandas as pd
        from data.features import build_features

        # Get feature names before scaling
        from data.preprocessor import _filter_market_hours, _remove_outliers
        df_clean  = _remove_outliers(_filter_market_hours(df))
        feat_df   = build_features(df_clean)
        self.feature_names = list(feat_df.dropna().columns)

        # Prepare sequences (y scaled by y_scale for stable training)
        X_seq, y_seq, scaler, y_scale = prepare_forecast(df, fit_scaler=True)
        self.scaler  = scaler
        self.y_scale = y_scale
        print(f"  y_scale = {y_scale:.6f}  (predictions will be divided by this)")

        # Train/val split (80/20, time-ordered)
        n      = len(X_seq)
        cutoff = int(n * config.TRAIN_SPLIT)
        X_train, X_val = X_seq[:cutoff], X_seq[cutoff:]
        y_train, y_val = y_seq[:cutoff], y_seq[cutoff:]

        log.info("Train: %d  Val: %d  Steps: %d", len(X_train), len(X_val), FORECAST_STEPS)

        # Build and train
        self.forecaster.build(input_shape=(X_train.shape[1], X_train.shape[2]))
        self.forecaster.train(X_train, y_train, X_val, y_val)

        # Evaluate on val set (unscale back to original RV units)
        preds   = self.forecaster.predict(X_val) * self.y_scale
        actuals = y_val * self.y_scale
        mse  = float(np.mean((preds - actuals) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(preds - actuals)))
        print(f"\n  Forecaster val metrics (original RV scale):")
        print(f"    RMSE : {rmse:.6f}")
        print(f"    MAE  : {mae:.6f}\n")

        # Save
        self._save()

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)

        self.forecaster.save(os.path.join(self.model_dir, FORECASTER_FILE))
        joblib.dump(self.scaler, os.path.join(self.model_dir, FORECAST_SCALER_FILE))

        with open(os.path.join(self.model_dir, FORECAST_FEATURES_FILE), "w") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(os.path.join(self.model_dir, FORECAST_Y_SCALE_FILE), "w") as f:
            json.dump({"y_scale": self.y_scale}, f)

        log.info("Forecaster artefacts saved → %s", self.model_dir)
        print(f"  Saved to: {self.model_dir}")
