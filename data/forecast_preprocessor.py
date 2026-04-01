"""
data/forecast_preprocessor.py — Preprocessing for multi-step RV forecasting.

Produces sequences of shape (seq_len, n_features) with targets of shape
(FORECAST_STEPS,) — the next N bars of realised volatility.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from data.features import build_features
from data.preprocessor import _filter_market_hours, _remove_outliers, _realised_volatility
from models.forecaster import build_sequences, FORECAST_STEPS
from utils.logger import get_logger

log = get_logger(__name__)

WINDOW = 10   # RV window size (bars)


def prepare_forecast(
    df: pd.DataFrame,
    scaler_path: str | None = None,
    fit_scaler: bool = True,
    steps: int = FORECAST_STEPS,
    y_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, float]:
    """
    Full preprocessing pipeline for multi-step forecasting.

    Returns
    -------
    X_seq   : (n_samples, seq_len, n_features)  float32
    y_seq   : (n_samples, steps)                float32  — scaled by y_scale
    scaler  : fitted MinMaxScaler (for X)
    y_scale : float multiplier applied to y (divide predictions by this)
    """
    df = _filter_market_hours(df)
    df = _remove_outliers(df)

    features = build_features(df)
    target   = _realised_volatility(df["Close"], WINDOW)

    combined = features.join(target.rename("target"), how="inner").dropna()
    X = combined.drop(columns=["target"])
    y = combined["target"]

    log.info("Forecast prep: %d samples, %d features.", len(X), X.shape[1])

    if fit_scaler:
        scaler  = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        # Scale y so values are in ~[0, 1] range for stable training
        y_scale = float(np.percentile(y.values, 95)) or 1.0
    else:
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "fit_scaler=False but no valid scaler_path supplied."
            )
        scaler   = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
        if y_scale is None:
            y_scale = 1.0

    y_scaled = y.values / y_scale

    X_seq, y_seq = build_sequences(
        X_scaled, y_scaled, config.SEQUENCE_LENGTH, steps
    )

    log.info("Sequences: X=%s  y=%s  y_scale=%.6f", X_seq.shape, y_seq.shape, y_scale)
    return X_seq, y_seq, scaler, y_scale
