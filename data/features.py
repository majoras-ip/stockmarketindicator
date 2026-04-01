"""
data/features.py — Feature engineering for the volatility model.

All features are derived from OHLCV + Bid/Ask data.  The function
`build_features(df)` is the single entry point used by the rest of
the application.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from utils.logger import get_logger

log = get_logger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(window)


# ─── public API ───────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all model input features from raw OHLCV + Bid/Ask data.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume, Bid, Ask]
         and a DatetimeIndex.

    Returns
    -------
    DataFrame of engineered features (no target column, no NaNs dropped here).
    """
    feat = pd.DataFrame(index=df.index)
    close  = df["Close"]
    volume = df["Volume"]

    # ── Price / Return Features ───────────────────────────────────────────────
    log.debug("Computing price / return features …")

    for lag in (1, 5, 10, 30):
        feat[f"log_return_{lag}"] = _log_return(close).rolling(lag).sum()

    feat["roc_10"]       = (close / close.shift(10) - 1) * 100   # rate of change %

    for w in (10, 20, 50):
        ma = close.rolling(w).mean()
        feat[f"dist_ma_{w}"] = (close - ma) / ma   # normalised distance

    # ── Volatility Features ───────────────────────────────────────────────────
    log.debug("Computing volatility features …")

    log_ret = _log_return(close)
    for w in (10, 20, 30):
        feat[f"roll_vol_{w}"] = _rolling_vol(log_ret, w)

    # Historical volatility ratios
    feat["hvr_10_20"] = feat["roll_vol_10"] / (feat["roll_vol_20"] + 1e-10)
    feat["hvr_10_30"] = feat["roll_vol_10"] / (feat["roll_vol_30"] + 1e-10)

    # Volatility of volatility
    feat["vol_of_vol"] = feat["roll_vol_10"].rolling(10).std()

    # ── Volume Features ───────────────────────────────────────────────────────
    log.debug("Computing volume features …")

    feat["vol_momentum"] = volume / (volume.shift(1) + 1)

    # VWAP (reset daily)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tpv = (typical_price * volume).cumsum()
    cum_vol  = volume.cumsum()
    vwap = cum_tpv / (cum_vol + 1e-10)
    feat["vwap_dist"] = (close - vwap) / (vwap + 1e-10)

    avg_vol_20 = volume.rolling(20).mean()
    feat["rel_volume"] = volume / (avg_vol_20 + 1e-10)

    # ── Bid / Ask Features ────────────────────────────────────────────────────
    log.debug("Computing bid/ask spread features …")

    bid = df["Bid"]
    ask = df["Ask"]
    feat["spread_pct"] = (ask - bid) / (close + 1e-10)

    # ── Time Features ─────────────────────────────────────────────────────────
    log.debug("Computing time features …")

    idx = df.index
    # Attempt to convert to Eastern time if tz-aware; otherwise use as-is.
    try:
        eastern = idx.tz_convert("America/New_York")
    except Exception:
        eastern = idx

    feat["hour"]          = eastern.hour
    feat["minute"]        = eastern.minute
    feat["day_of_week"]   = eastern.dayofweek   # 0=Monday … 4=Friday

    # Minutes elapsed since market open (9:30 = 0)
    feat["mins_since_open"] = (eastern.hour - 9) * 60 + (eastern.minute - 30)
    feat["mins_since_open"] = feat["mins_since_open"].clip(lower=0)

    # Is this bar in the first or last 30 minutes of the session?
    feat["is_open_period"]  = (feat["mins_since_open"] <= 30).astype(int)
    feat["is_close_period"] = (feat["mins_since_open"] >= 360).astype(int)  # 6h - 30m = 330m but mark last 30 as well
    # 390 total minutes in session; last 30 = mins_since_open >= 360
    feat["is_close_period"] = (feat["mins_since_open"] >= 360).astype(int)

    # ── Statistical Features ──────────────────────────────────────────────────
    log.debug("Computing rolling statistical features …")

    for w in (10, 20, 30):
        feat[f"roll_mean_{w}"]  = log_ret.rolling(w).mean()
        feat[f"roll_std_{w}"]   = log_ret.rolling(w).std()
        feat[f"roll_min_{w}"]   = log_ret.rolling(w).min()
        feat[f"roll_max_{w}"]   = log_ret.rolling(w).max()

    # Rolling skewness / kurtosis (need at least window size observations)
    feat["roll_skew_20"] = log_ret.rolling(20).apply(
        lambda x: skew(x) if len(x) > 3 else 0, raw=True
    )
    feat["roll_kurt_20"] = log_ret.rolling(20).apply(
        lambda x: kurtosis(x) if len(x) > 3 else 0, raw=True
    )

    log.info("Feature matrix shape: %s", feat.shape)
    return feat
