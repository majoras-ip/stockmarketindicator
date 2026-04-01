"""
data/collector.py — Download stock price data via yfinance.

yfinance data-resolution limits (as of 2024):
  1-minute  : last 7 days only
  5-minute  : last 60 days
  1-hour    : up to 730 days (~2 years)
  1-day     : unlimited

This module fetches the best available resolution for the requested
period and clearly documents which resolution was actually used.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

import config
from utils.logger import get_logger

log = get_logger(__name__)


# Maps a (period, interval) pair to the actual interval we can fetch.
# Priority: finest resolution that yfinance allows for the window.
_INTERVAL_POLICY: list[tuple[int, str]] = [
    (7,   "1m"),   # ≤7 days  → 1-minute bars
    (60,  "5m"),   # ≤60 days → 5-minute bars
    (730, "1h"),   # ≤2 years → 1-hour bars
    (99999, "1d"), # anything longer → daily bars
]


def _period_to_days(period: str) -> int:
    """Convert a yfinance period string ('3y', '60d', '1mo' …) to days."""
    unit = period[-1]
    num  = int(period[:-1])
    return {"d": 1, "w": 7, "m": 30, "y": 365}[unit] * num


def best_interval(period: str) -> str:
    """Return the finest yfinance interval available for *period*."""
    days = _period_to_days(period)
    for threshold, interval in _INTERVAL_POLICY:
        if days <= threshold:
            return interval
    return "1d"


def download(
    ticker: str = config.TICKER,
    period: str = config.PERIOD,
    interval: str | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for *ticker*.

    Parameters
    ----------
    ticker   : stock symbol, e.g. 'AAPL'
    period   : how far back, e.g. '3y', '60d', '7d'
    interval : bar size; if None, chosen automatically based on *period*
    save_csv : write raw data to data/raw/<ticker>_<interval>.csv

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume] and a
    DatetimeIndex in UTC.
    """
    if interval is None:
        interval = best_interval(period)
        log.info(
            "Auto-selected interval '%s' for period '%s' "
            "(yfinance resolution limit).",
            interval, period,
        )
    else:
        log.info("Using user-supplied interval '%s'.", interval)

    log.info("Downloading %s  period=%s  interval=%s …", ticker, period, interval)

    try:
        raw: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Failed to download data for '{ticker}'. "
            "Check your internet connection and ticker symbol."
        ) from exc

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker='{ticker}' "
            f"period='{period}' interval='{interval}'. "
            "The ticker may be invalid or the market may have been closed "
            "for the entire requested window."
        )

    # Flatten multi-level columns that newer yfinance versions may produce.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Ensure index is a proper DatetimeIndex (some versions return strings).
    if not isinstance(raw.index, pd.DatetimeIndex):
        raw.index = pd.to_datetime(raw.index)

    log.info(
        "Downloaded %d rows  (%s → %s).",
        len(raw),
        raw.index[0].strftime("%Y-%m-%d"),
        raw.index[-1].strftime("%Y-%m-%d"),
    )

    # yfinance doesn't expose live bid/ask for historical data; add
    # placeholder columns so downstream code has a consistent schema.
    if "Bid" not in raw.columns:
        raw["Bid"] = raw["Close"] * 0.9995   # synthetic 0.05 % spread
    if "Ask" not in raw.columns:
        raw["Ask"] = raw["Close"] * 1.0005

    if save_csv:
        _save(raw, ticker, interval)

    return raw


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a user-supplied CSV file instead of downloading from yfinance.

    The CSV must have at minimum: Open, High, Low, Close, Volume columns
    and a datetime index (or a column that can be parsed as datetime).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    log.info("Loading data from CSV: %s", filepath)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            "Expected at minimum: Open, High, Low, Close, Volume."
        )

    if "Bid" not in df.columns:
        df["Bid"] = df["Close"] * 0.9995
    if "Ask" not in df.columns:
        df["Ask"] = df["Close"] * 1.0005

    log.info("Loaded %d rows from CSV.", len(df))
    return df


def _save(df: pd.DataFrame, ticker: str, interval: str) -> None:
    """Persist raw data to data/raw/ for reproducibility."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out = os.path.join(config.DATA_DIR, f"{ticker}_{interval}.csv")
    df.to_csv(out)
    log.info("Raw data saved → %s", out)
