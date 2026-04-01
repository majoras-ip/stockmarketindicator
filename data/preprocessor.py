"""
data/preprocessor.py — Data cleaning, target creation, normalisation,
and train/test splitting.

Key design decisions
────────────────────
• Time order is always preserved — no random shuffling.
• The scaler is fit on training data only, then applied to test data,
  preventing any information leakage.
• The target variable is 10-minute realised volatility:
      RV = sqrt( sum( log(p_t / p_{t-1})^2 ) )  over a 10-bar window
  This matches the definition used in the paper.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import config
from data.features import build_features
from utils.logger import get_logger

log = get_logger(__name__)

WINDOW = 10   # bars in each realised-volatility window


# ─── public API ───────────────────────────────────────────────────────────────

def prepare(
    df: pd.DataFrame,
    scaler_path: str | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df          : raw OHLCV + Bid/Ask DataFrame from collector
    scaler_path : if provided and fit_scaler=False, load scaler from here
    fit_scaler  : fit a new scaler on the data (True during training)

    Returns
    -------
    X       : scaled feature DataFrame (index aligned with y)
    y       : target series — 10-min realised volatility
    scaler  : fitted MinMaxScaler (for later inverse-transform or saving)
    """
    df = _filter_market_hours(df)
    df = _remove_outliers(df)

    features = build_features(df)
    target   = _realised_volatility(df["Close"], WINDOW).shift(-config.FORECAST_HORIZON)

    # Align features and target (drop rows where either has NaN).
    combined = features.join(target.rename("target"), how="inner").dropna()

    X = combined.drop(columns=["target"])
    y = combined["target"]

    log.info("After cleaning: %d samples, %d features.", len(X), X.shape[1])

    if fit_scaler:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), index=X.index, columns=X.columns
        )
    else:
        if scaler_path is None or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "fit_scaler=False but no valid scaler_path supplied. "
                "Train a model first."
            )
        scaler   = joblib.load(scaler_path)
        X_scaled = pd.DataFrame(
            scaler.transform(X), index=X.index, columns=X.columns
        )

    return X_scaled, y, scaler


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    split: float = config.TRAIN_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based split — earlier rows are training, later rows are test.
    No shuffling, no random state.
    """
    n     = len(X)
    cutoff = int(n * split)

    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]

    log.info(
        "Train/test split at %.0f%%: train=%d rows, test=%d rows.",
        split * 100, len(X_train), len(X_test),
    )
    return X_train, X_test, y_train, y_test


# ─── internal helpers ─────────────────────────────────────────────────────────

def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars within regular trading hours (9:30 – 16:00 ET)."""
    try:
        eastern = df.index.tz_convert("America/New_York")
    except Exception:
        log.warning(
            "Could not convert index to Eastern time. "
            "Market-hours filter skipped."
        )
        return df

    mask = (
        (eastern.time >= pd.Timestamp("09:30").time())
        & (eastern.time < pd.Timestamp("16:00").time())
        # Monday–Friday only
        & (eastern.dayofweek < 5)
    )
    filtered = df[mask]
    dropped  = len(df) - len(filtered)
    if dropped:
        log.info("Market-hours filter removed %d pre/post-market rows.", dropped)
    return filtered


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any numeric column is beyond OUTLIER_STD std-devs."""
    numeric = df.select_dtypes(include=[np.number])
    z_scores = ((numeric - numeric.mean()) / (numeric.std() + 1e-10)).abs()
    mask = (z_scores < config.OUTLIER_STD).all(axis=1)
    removed = (~mask).sum()
    if removed:
        log.info("Outlier removal: dropped %d rows (>%g σ).", removed, config.OUTLIER_STD)
    return df[mask]


def _realised_volatility(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """
    Compute realised volatility over a rolling *window*.

    RV_t = sqrt( sum_{i=1}^{window} r_{t-i+1}^2 )
    where r_i = log(p_i / p_{i-1}).

    This is the target variable the model learns to predict.
    """
    log_returns = np.log(close / close.shift(1))
    rv = log_returns.pow(2).rolling(window).sum().apply(np.sqrt)
    return rv
