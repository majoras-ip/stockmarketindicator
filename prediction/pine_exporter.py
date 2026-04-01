"""
prediction/pine_exporter.py — Export model predictions as a TradingView Pine Script.

Generates a ready-to-paste .pine file with predicted realised volatility
embedded as arrays, colour-coded by volatility regime.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd

import config
from data.collector import download
from data.preprocessor import prepare
from models.lightgbm_model import LightGBMModel
from utils.logger import get_logger

log = get_logger(__name__)


class PineExporter:
    """
    Loads the saved model, runs it on recent data, and writes a Pine Script
    indicator file whose predictions can be pasted directly into TradingView.
    """

    def __init__(self, model_dir: str = config.MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.lgb = LightGBMModel()
        self.scaler = None
        self.feature_names: list[str] = []
        self._loaded = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load saved LightGBM artefacts from MODEL_DIR."""
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: '{self.model_dir}'. "
                "Train the model first (option 2)."
            )
        self.lgb.load(self.model_dir)

        scaler_path = os.path.join(self.model_dir, config.SCALER_FILE)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        feat_path = os.path.join(self.model_dir, config.FEATURE_LIST_FILE)
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.feature_names = json.load(f)

        self._loaded = True
        log.info("PineExporter: LightGBM model loaded.")

    # ── Export ────────────────────────────────────────────────────────────────

    def export(
        self,
        ticker: str,
        period: str = "5d",
        interval: str = "5m",
        output_dir: str = config.CHART_DIR,
    ) -> str:
        """
        Download *period* of *interval* data, run predictions on every bar,
        and write a ready-to-paste Pine Script file.

        Returns the path to the generated .pine file.
        """
        if not self._loaded:
            self.load()

        log.info("Fetching %s %s data for %s …", period, interval, ticker)
        df = download(ticker, period=period, interval=interval, save_csv=False)

        min_bars = config.SEQUENCE_LENGTH + 10
        if len(df) < min_bars:
            raise ValueError(
                f"Only {len(df)} bars retrieved for '{ticker}' ({period}/{interval}). "
                f"Need at least {min_bars}. Market may be closed or the ticker is invalid."
            )

        X, _y, _ = prepare(
            df,
            scaler_path=os.path.join(self.model_dir, config.SCALER_FILE),
            fit_scaler=False,
        )

        if self.feature_names:
            for col in set(self.feature_names) - set(X.columns):
                X[col] = 0.0
            X = X[self.feature_names]

        preds = self.lgb.predict(X)
        pred_series = pd.Series(preds, index=X.index)

        # Pine Script expects bar open times in milliseconds UTC.
        timestamps_ms = [int(ts.timestamp() * 1000) for ts in pred_series.index]
        values = [round(float(v), 8) for v in pred_series.values]

        interval_mins = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(interval, 5)
        pine_code = _build_pine_script(
            ticker=ticker,
            interval=interval,
            interval_mins=interval_mins,
            timestamps_ms=timestamps_ms,
            values=values,
        )

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker.lower()}_predicted_rv_{interval}.pine"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            f.write(pine_code)

        print(f"\n  Pine Script saved : {out_path}")
        print(f"  Bars exported     : {len(values)}")
        print(f"  Data range        : {pred_series.index[0]} → {pred_series.index[-1]}")
        print(f"\n  Steps to view in TradingView:")
        print(f"    1. Open TradingView and load the {ticker} chart")
        print(f"    2. Open Pine Script editor (bottom toolbar)")
        print(f"    3. Select all → paste the contents of {filename}")
        print(f"    4. Click 'Add to chart'")
        print(f"       Works on any timeframe — predictions auto-aggregate.\n")

        return out_path


# ── Pine Script template ───────────────────────────────────────────────────────

def _build_pine_script(
    ticker: str,
    interval: str,
    interval_mins: int,
    timestamps_ms: list[int],
    values: list[float],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = len(timestamps_ms)

    ts_str  = ", ".join(str(t) for t in timestamps_ms)
    val_str = ", ".join(f"{v:.8f}" for v in values)

    import numpy as np
    arr = np.array(values)
    low_thresh    = round(float(np.percentile(arr, 25)), 8)
    medium_thresh = round(float(np.percentile(arr, 50)), 8)
    high_thresh   = round(float(np.percentile(arr, 75)), 8)

    return f"""\
// ================================================================
// Predicted Realised Volatility — {ticker} ({interval.upper()})
// Generated by stock_volatility on {generated_at}
// Model : LightGBM (k-fold ensemble)
// Bars  : {n}
//
// Works on any TradingView timeframe — predictions will be averaged
// across bars on higher timeframes, or held constant on finer ones.
// ================================================================
//@version=5
indicator("Forecast RV (+{config.FORECAST_HORIZON * interval_mins}min) · {ticker} · {interval.upper()}", overlay=false, max_bars_back=500)

// ── Embedded prediction data ─────────────────────────────────────
var int[]   _ts = array.from({ts_str})
var float[] _rv = array.from({val_str})

// ── Match predictions to current bar (works on any timeframe) ────
// Higher timeframes (e.g. 1H chart, 5m data): average all predictions
//   whose timestamps fall within this bar's [open, close) window.
// Finer timeframes (e.g. 1m chart, 5m data): use the nearest prediction
//   at or before this bar's open time.
float rv = na
float _sum = 0.0
int   _n   = 0

for i = 0 to array.size(_ts) - 1
    int t = array.get(_ts, i)
    if t >= time and t < time_close
        _sum += array.get(_rv, i)
        _n   += 1

if _n > 0
    rv := _sum / _n
else
    // No predictions inside this bar — find the most recent one before it.
    for i = array.size(_ts) - 1 to 0
        if array.get(_ts, i) <= time
            rv := array.get(_rv, i)
            break

// ── Volatility regime thresholds ────────────────────────────────
float LOW_THRESH    = {low_thresh}
float MEDIUM_THRESH = {medium_thresh}
float HIGH_THRESH   = {high_thresh}

// ── Colour by regime ─────────────────────────────────────────────
color rv_color =
     na(rv)                  ? color.gray
     : rv < LOW_THRESH       ? color.new(color.green,  0)
     : rv < MEDIUM_THRESH    ? color.new(color.yellow, 0)
     : rv < HIGH_THRESH      ? color.new(color.orange, 0)
     :                         color.new(color.red,    0)

// ── Main plot ────────────────────────────────────────────────────
plot(rv, "Predicted RV", color=rv_color, linewidth=2,
     style=plot.style_stepline)

// ── Regime threshold lines ───────────────────────────────────────
hline(LOW_THRESH,    "Low",     color=color.new(color.green,  40),
      linestyle=hline.style_dashed, linewidth=1)
hline(MEDIUM_THRESH, "Medium",  color=color.new(color.yellow, 40),
      linestyle=hline.style_dashed, linewidth=1)
hline(HIGH_THRESH,   "High",    color=color.new(color.orange, 40),
      linestyle=hline.style_dashed, linewidth=1)

// ── Background shading by regime ─────────────────────────────────
bgcolor(
     na(rv)               ? na
     : rv < LOW_THRESH    ? color.new(color.green,  90)
     : rv < MEDIUM_THRESH ? color.new(color.yellow, 90)
     : rv < HIGH_THRESH   ? color.new(color.orange, 90)
     :                      color.new(color.red,    90),
     title="Volatility Regime")

// ── Label at start of prediction window ──────────────────────────
if not na(rv) and na(rv[1])
    label.new(bar_index, rv, "◀ predictions start",
              style=label.style_label_right, size=size.small,
              color=color.new(color.blue, 70), textcolor=color.white)
"""
