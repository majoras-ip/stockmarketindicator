"""
prediction/predictor.py — Real-time prediction pipeline.

Loads saved models, downloads the latest data, engineers features,
and returns a human-readable volatility forecast for the next 10 minutes.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

import config
from data.collector import download
from data.preprocessor import prepare
from models.robust_model import RobustModel
from utils.logger import get_logger

log = get_logger(__name__)

# Volatility classification thresholds (from config.py, overrideable here).
THRESHOLDS = [
    (config.VOL_LOW,    "Low",     "Markets appear calm. Typical day-to-day movement."),
    (config.VOL_MEDIUM, "Medium",  "Moderate volatility. Normal intraday swings."),
    (config.VOL_HIGH,   "High",    "Elevated volatility. Expect larger price moves."),
    (float("inf"),      "Extreme", "Very high volatility. Exercise caution."),
]


def classify_volatility(vol: float) -> tuple[str, str]:
    """Return (level, description) string pair for a volatility value."""
    for threshold, level, desc in THRESHOLDS:
        if vol < threshold:
            return level, desc
    return "Extreme", THRESHOLDS[-1][2]


class Predictor:
    """
    Loads saved models and predicts next-10-minute volatility for any ticker.
    """

    def __init__(self, model_dir: str = config.MODEL_DIR) -> None:
        self.model_dir  = model_dir
        self.model      = RobustModel()
        self.scaler     = None
        self.feature_names: list[str] = []
        self._loaded    = False
        self._metrics: dict = {}

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all saved artefacts from MODEL_DIR."""
        self._check_model_dir()

        self.model.load_all(self.model_dir)

        scaler_path = os.path.join(self.model_dir, config.SCALER_FILE)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: {scaler_path}. Run training first."
            )
        self.scaler = joblib.load(scaler_path)

        feat_path = os.path.join(self.model_dir, config.FEATURE_LIST_FILE)
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.feature_names = json.load(f)

        metrics_path = os.path.join(self.model_dir, config.METRICS_FILE)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self._metrics = json.load(f)

        self._loaded = True
        log.info("Predictor ready.")
        self._print_model_info()

    def _check_model_dir(self) -> None:
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: '{self.model_dir}'. "
                "Train the model first (option 2 in the menu)."
            )

    def _print_model_info(self) -> None:
        """Display saved training metrics to the user."""
        if not self._metrics:
            return
        rob = self._metrics.get("robust", {}).get("test", {})
        if rob:
            print("\n  Loaded model — test-set performance:")
            print(f"    RMSE : {rob.get('RMSE', 'N/A'):.6f}")
            print(f"    MAE  : {rob.get('MAE',  'N/A'):.6f}")
            print(f"    R²   : {rob.get('R2',   'N/A'):.4f}")
            print()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, ticker: str) -> dict:
        """
        Download latest data for *ticker* and predict next-10-min volatility.

        Returns
        -------
        dict with keys:
          ticker, predicted_volatility, level, description,
          confidence_note, data_points_used
        """
        if not self._loaded:
            self.load()

        log.info("Fetching latest data for %s …", ticker)
        try:
            # Use 5-minute bars for the last 5 days — best resolution for
            # near-real-time inference without the 7-day 1-min API limit.
            df = download(ticker, period="5d", interval="5m", save_csv=False)
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch data for '{ticker}': {exc}"
            ) from exc

        if len(df) < config.SEQUENCE_LENGTH + 50:
            raise ValueError(
                f"Only {len(df)} bars retrieved for '{ticker}'. "
                "Market may be closed or the ticker is invalid."
            )

        # Preprocess (use saved scaler — do NOT refit).
        X, y, _ = prepare(
            df,
            scaler_path=os.path.join(self.model_dir, config.SCALER_FILE),
            fit_scaler=False,
        )

        # Align feature columns with the saved feature list.
        if self.feature_names:
            missing = set(self.feature_names) - set(X.columns)
            extra   = set(X.columns) - set(self.feature_names)
            if missing:
                log.warning("Missing features %s — filling with 0.", missing)
                for col in missing:
                    X[col] = 0.0
            X = X[self.feature_names]

        preds = self.model.predict(X)
        vol   = float(preds[-1])   # most recent prediction

        level, desc = classify_volatility(vol)

        # Rough confidence: how close is vol to the historical test RMSE?
        rmse = self._metrics.get("robust", {}).get("test", {}).get("RMSE", None)
        if rmse:
            cv = rmse / (vol + 1e-10)   # coefficient of variation proxy
            confidence = "High" if cv < 0.5 else "Moderate" if cv < 1.5 else "Low"
        else:
            confidence = "Unknown (no saved metrics)"

        result = {
            "ticker":                ticker,
            "predicted_volatility":  round(vol, 8),
            "level":                 level,
            "description":           desc,
            "confidence":            confidence,
            "data_points_used":      len(df),
        }

        self._display(result)
        return result

    def _display(self, result: dict) -> None:
        """Pretty-print the prediction to the console."""
        width = 52
        print("\n" + "╔" + "═" * width + "╗")
        print("║" + "  Volatility Forecast — Next 10 Minutes".center(width) + "║")
        print("╠" + "═" * width + "╣")
        print(f"║  Ticker           : {result['ticker']:<30}  ║")
        print(f"║  Predicted Vol    : {result['predicted_volatility']:<30.8f}  ║")
        print(f"║  Level            : {result['level']:<30}  ║")
        print(f"║  Confidence       : {result['confidence']:<30}  ║")
        print("╠" + "═" * width + "╣")
        # Word-wrap the description to fit inside the box.
        words = result["description"].split()
        line  = ""
        lines = []
        for word in words:
            if len(line) + len(word) + 1 <= width - 4:
                line = (line + " " + word).lstrip()
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        for ln in lines:
            print(f"║  {ln:<{width - 2}}  ║")
        print("╚" + "═" * width + "╝\n")
