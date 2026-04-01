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
DIRECTION_MODEL_FILE = "direction_model.joblib"


class ForecastTrainer:
    """
    End-to-end training pipeline for the LSTM multi-step forecaster.
    """

    def __init__(self, model_dir: str = config.MODEL_DIR) -> None:
        self.model_dir      = model_dir
        self.forecaster     = LSTMForecaster(steps=FORECAST_STEPS)
        self.scaler         = None
        self.y_scale        = 1.0
        self.direction_model = None
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

        # Direction model (LightGBM classifier: 1=up, 0=down)
        self._train_direction(df_clean, feat_df)

        # Save
        self._save()

    def _train_direction(self, df_clean: "pd.DataFrame", feat_df: "pd.DataFrame") -> None:
        """Train a LightGBM classifier to predict price direction."""
        import lightgbm as lgb

        close = df_clean["Close"].reindex(feat_df.index)
        future_close = close.shift(-FORECAST_STEPS)

        import pandas as pd
        combined = feat_df.join(future_close.rename("future_close"), how="inner").dropna()
        X_dir = combined.drop(columns=["future_close"]).values
        y_dir = (combined["future_close"].values > close.reindex(combined.index).values).astype(int)

        cutoff = int(len(X_dir) * config.TRAIN_SPLIT)
        X_tr, X_vl = X_dir[:cutoff], X_dir[cutoff:]
        y_tr, y_vl = y_dir[:cutoff], y_dir[cutoff:]

        self.direction_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, verbose=-1,
        )
        self.direction_model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )

        acc = float(np.mean(self.direction_model.predict(X_vl) == y_vl))
        print(f"  Direction model val accuracy: {acc:.1%}")

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)

        self.forecaster.save(os.path.join(self.model_dir, FORECASTER_FILE))
        joblib.dump(self.scaler, os.path.join(self.model_dir, FORECAST_SCALER_FILE))

        with open(os.path.join(self.model_dir, FORECAST_FEATURES_FILE), "w") as f:
            json.dump(self.feature_names, f, indent=2)

        with open(os.path.join(self.model_dir, FORECAST_Y_SCALE_FILE), "w") as f:
            json.dump({"y_scale": self.y_scale}, f)

        if self.direction_model is not None:
            joblib.dump(self.direction_model, os.path.join(self.model_dir, DIRECTION_MODEL_FILE))

        log.info("Forecaster artefacts saved → %s", self.model_dir)
        print(f"  Saved to: {self.model_dir}")
