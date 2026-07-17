"""
data/pattern_features.py — Chart-pattern feature engineering for the
directional model.

This is deliberately kept *separate* from `data/features.py`: the volatility
model's scaler and `feature_list.json` are frozen at 38 columns, so adding
columns there would break it.  The directional trainer combines
`build_features()` (38 vol/return/time features) with `build_pattern_features()`
(the structural / candlestick / momentum features below).

Every feature here is:
  • derived purely from OHLCV (works on any ticker straight from yfinance), and
  • strictly backward-looking (uses only bars up to and including t) — no peeking
    at future bars, which is the #1 source of fake accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ─── momentum helpers ─────────────────────────────────────────────────────────

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist        = macd_line - signal_line
    # normalise by price so it's comparable across tickers
    return macd_line / close, signal_line / close, hist / close


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of the last `window` points, normalised by price."""
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        return cov / x_var

    slope = series.rolling(window).apply(_slope, raw=True)
    return slope / (series + 1e-10)   # per-bar % drift


# ─── public API ───────────────────────────────────────────────────────────────

def build_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer structural chart-pattern features from OHLCV.

    Parameters
    ----------
    df : DataFrame with columns [Open, High, Low, Close, Volume] and a
         DatetimeIndex.  (Bid/Ask are ignored here.)

    Returns
    -------
    DataFrame of pattern features aligned to `df.index` (NaNs not dropped).
    """
    feat = pd.DataFrame(index=df.index)

    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    rng = (h - l).replace(0, np.nan)          # bar range, guard div-by-zero
    body = (c - o)

    # ── Candlestick geometry ──────────────────────────────────────────────────
    feat["body_pct"]       = body / rng                       # signed body size
    feat["upper_wick_pct"] = (h - np.maximum(o, c)) / rng     # rejection above
    feat["lower_wick_pct"] = (np.minimum(o, c) - l) / rng     # rejection below
    feat["close_pos"]      = (c - l) / rng                    # where close sits in range (0..1)
    feat["gap_pct"]        = (o - c.shift(1)) / (c.shift(1) + 1e-10)

    # Named single/two-bar patterns (as soft flags)
    prev_body = body.shift(1)
    feat["is_doji"]     = (body.abs() / rng < 0.1).astype(float)
    feat["is_hammer"]   = ((feat["lower_wick_pct"] > 0.5) & (feat["body_pct"].abs() < 0.35)).astype(float)
    feat["is_shooter"]  = ((feat["upper_wick_pct"] > 0.5) & (feat["body_pct"].abs() < 0.35)).astype(float)
    feat["bull_engulf"] = ((body > 0) & (prev_body < 0) & (c > o.shift(1)) & (o < c.shift(1))).astype(float)
    feat["bear_engulf"] = ((body < 0) & (prev_body > 0) & (c < o.shift(1)) & (o > c.shift(1))).astype(float)

    # ── Trend / structure ─────────────────────────────────────────────────────
    for w in (10, 20, 50):
        feat[f"slope_{w}"] = _rolling_slope(c, w)

    # Position within the recent range → proximity to support/resistance
    for w in (20, 50):
        roll_max = h.rolling(w).max()
        roll_min = l.rolling(w).min()
        span = (roll_max - roll_min).replace(0, np.nan)
        feat[f"range_pos_{w}"]  = (c - roll_min) / span          # 0=at support, 1=at resistance
        feat[f"dist_high_{w}"]  = (c - roll_max) / (roll_max + 1e-10)
        feat[f"dist_low_{w}"]   = (c - roll_min) / (roll_min + 1e-10)
        # breakout flags: close clears the *prior* window's extreme (shifted to avoid self-reference)
        feat[f"breakout_up_{w}"]   = (c > roll_max.shift(1)).astype(float)
        feat[f"breakout_down_{w}"] = (c < roll_min.shift(1)).astype(float)

    # Higher-highs / higher-lows structure (uptrend fingerprint) over 20 bars
    hh = (h > h.shift(1)).rolling(20).sum() / 20.0
    hl = (l > l.shift(1)).rolling(20).sum() / 20.0
    feat["hh_frac"] = hh
    feat["hl_frac"] = hl
    feat["trend_structure"] = (hh + hl) - 1.0   # +1 strong up, -1 strong down

    # Consecutive up/down streak (signed, capped)
    up = (c > c.shift(1)).astype(int)
    streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    feat["updown_streak"] = np.where(up == 1, streak, -streak).clip(-10, 10)

    # ── Momentum ──────────────────────────────────────────────────────────────
    feat["rsi_14"] = _rsi(c, 14) / 100.0
    macd_line, signal_line, hist = _macd(c)
    feat["macd"]      = macd_line
    feat["macd_sig"]  = signal_line
    feat["macd_hist"] = hist

    # Stochastic %K over 14
    ll = l.rolling(14).min()
    hh14 = h.rolling(14).max()
    feat["stoch_k"] = (c - ll) / ((hh14 - ll).replace(0, np.nan))

    # Volume confirmation of the move
    feat["vol_trend"] = (v / (v.rolling(20).mean() + 1e-10)) * np.sign(body)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    log.info("Pattern feature matrix shape: %s", feat.shape)
    return feat


# convenience: the combined feature matrix used by the directional model
def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the frozen vol features with the new pattern features."""
    from data.features import build_features
    base = build_features(df)
    pat  = build_pattern_features(df)
    return base.join(pat, how="outer")
