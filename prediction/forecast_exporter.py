"""
prediction/forecast_exporter.py — Export LSTM multi-step forecast as Pine Script.

The generated indicator draws:
  - A step-line of historical predictions (for context)
  - A dotted forecast line extending 30 min into the future from the last bar
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import config
from data.forecast_preprocessor import prepare_forecast, FORECAST_STEPS
from models.forecaster import LSTMForecaster, build_sequences
from training.forecast_trainer import FORECASTER_FILE, FORECAST_SCALER_FILE, FORECAST_FEATURES_FILE, FORECAST_Y_SCALE_FILE
from utils.logger import get_logger

log = get_logger(__name__)


class ForecastExporter:
    """
    Loads the saved LSTM forecaster and exports a Pine Script file with
    a 30-minute forward forecast line.
    """

    def __init__(self, model_dir: str = config.MODEL_DIR) -> None:
        self.model_dir     = model_dir
        self.forecaster    = LSTMForecaster(steps=FORECAST_STEPS)
        self.scaler        = None
        self.y_scale       = 1.0
        self.feature_names: list[str] = []
        self._loaded       = False

    def load(self) -> None:
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Model directory not found: '{self.model_dir}'. Train first."
            )
        forecaster_path = os.path.join(self.model_dir, FORECASTER_FILE)
        if not os.path.exists(forecaster_path):
            raise FileNotFoundError(
                f"Forecaster not found: {forecaster_path}. "
                "Run option 11 (Train Forecaster) first."
            )
        self.forecaster.load(forecaster_path)
        self.scaler = joblib.load(os.path.join(self.model_dir, FORECAST_SCALER_FILE))

        feat_path = os.path.join(self.model_dir, FORECAST_FEATURES_FILE)
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.feature_names = json.load(f)

        y_scale_path = os.path.join(self.model_dir, FORECAST_Y_SCALE_FILE)
        if os.path.exists(y_scale_path):
            with open(y_scale_path) as f:
                self.y_scale = json.load(f).get("y_scale", 1.0)

        self._loaded = True
        log.info("ForecastExporter loaded.")

    def export(
        self,
        df: pd.DataFrame,
        ticker: str,
        interval: str = "5m",
        output_dir: str = config.CHART_DIR,
    ) -> str:
        """
        Run the forecaster on *df* and write a Pine Script file.

        Returns the path to the generated file.
        """
        if not self._loaded:
            self.load()

        from data.preprocessor import _filter_market_hours, _remove_outliers
        from data.features import build_features

        df_clean = _remove_outliers(_filter_market_hours(df))
        features = build_features(df_clean)
        features = features.dropna()

        if self.feature_names:
            for col in set(self.feature_names) - set(features.columns):
                features[col] = 0.0
            features = features[self.feature_names]

        X_scaled = self.scaler.transform(features)

        # Build sequences for all historical bars
        dummy_y = np.zeros(len(X_scaled))
        X_seq, _ = build_sequences(
            X_scaled, dummy_y, config.SEQUENCE_LENGTH, FORECAST_STEPS
        )

        if len(X_seq) == 0:
            raise ValueError("Not enough data to build sequences.")

        # Get all historical predictions (last value of each forecast = 1-step ahead)
        all_preds = self.forecaster.predict(X_seq) * self.y_scale   # (n, FORECAST_STEPS)
        hist_rv   = all_preds[:, 0]                                  # 1-step ahead for each bar

        # Timestamps for historical predictions
        hist_index = features.index[config.SEQUENCE_LENGTH : config.SEQUENCE_LENGTH + len(hist_rv)]
        timestamps_ms = [int(ts.timestamp() * 1000) for ts in hist_index]
        hist_values   = [round(float(v), 8) for v in hist_rv]

        # Future forecast from the last sequence
        future_forecast = self.forecaster.predict_last(X_seq) * self.y_scale  # (FORECAST_STEPS,)
        future_values   = [round(float(v), 8) for v in future_forecast]

        # Infer bar duration in milliseconds
        interval_ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
        bar_ms = interval_ms_map.get(interval, 300_000)

        # Future timestamps: start right after the last historical bar
        last_ts_ms = timestamps_ms[-1] if timestamps_ms else int(datetime.now(timezone.utc).timestamp() * 1000)
        future_timestamps_ms = [last_ts_ms + bar_ms * (i + 1) for i in range(FORECAST_STEPS)]

        pine_code = _build_forecast_pine(
            ticker=ticker,
            interval=interval,
            timestamps_ms=timestamps_ms,
            hist_values=hist_values,
            future_timestamps_ms=future_timestamps_ms,
            future_values=future_values,
        )

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{ticker.lower()}_forecast_{interval}.pine"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            f.write(pine_code)

        print(f"\n  Forecast Pine Script saved : {out_path}")
        print(f"  Historical bars            : {len(hist_values)}")
        print(f"  Forecast horizon           : {FORECAST_STEPS} bars ({FORECAST_STEPS * int(bar_ms/60000)} min)")
        print(f"  Forecast values            : {[round(v,5) for v in future_values]}")
        print(f"\n  Steps:")
        print(f"    1. Open TradingView → {ticker} → {interval.upper()} chart")
        print(f"    2. Pine Script editor → paste → Add to chart\n")

        return out_path


def _build_forecast_pine(
    ticker: str,
    interval: str,
    timestamps_ms: list[int],
    hist_values: list[float],
    future_timestamps_ms: list[int],
    future_values: list[float],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    steps = len(future_values)

    arr = np.array(hist_values)
    low_thresh    = round(float(np.percentile(arr, 25)), 8)
    medium_thresh = round(float(np.percentile(arr, 50)), 8)
    high_thresh   = round(float(np.percentile(arr, 75)), 8)

    ts_str   = ", ".join(str(t) for t in timestamps_ms)
    hist_str = ", ".join(f"{v:.8f}" for v in hist_values)

    fts_str  = ", ".join(str(t) for t in future_timestamps_ms)
    fval_str = ", ".join(f"{v:.8f}" for v in future_values)

    return f"""\
// ================================================================
// Volatility Forecast — {ticker} ({interval.upper()})
// Generated by stock_volatility on {generated_at}
// Model : LSTM Multi-Step Forecaster
// Horizon: +{steps} bars ({steps * (5 if interval == '5m' else 1)} min)
//
// BLUE dotted line = 30-min forecast extending into the future
// STEP line = historical 1-bar-ahead predictions (for context)
// ================================================================
//@version=5
indicator("Volatility Forecast · {ticker} · {interval.upper()}", overlay=false, max_bars_back=500)

// ── Historical predictions ────────────────────────────────────────
var int[]   _ts   = array.from({ts_str})
var float[] _hist = array.from({hist_str})

// ── Future forecast (next {steps} bars) ──────────────────────────
var int[]   _fts  = array.from({fts_str})
var float[] _fval = array.from({fval_str})

// ── Thresholds (auto-calibrated to this stock) ───────────────────
float LOW_THRESH    = {low_thresh}
float MEDIUM_THRESH = {medium_thresh}
float HIGH_THRESH   = {high_thresh}

// ── Match historical bar ─────────────────────────────────────────
float rv = na
for i = 0 to array.size(_ts) - 1
    int t = array.get(_ts, i)
    if t >= time and t < time_close
        rv := array.get(_hist, i)
        break
if na(rv)
    for i = array.size(_ts) - 1 to 0
        if array.get(_ts, i) <= time
            rv := array.get(_hist, i)
            break

// ── Colour by regime ─────────────────────────────────────────────
color rv_color =
     na(rv)               ? color.gray
     : rv < LOW_THRESH    ? color.new(color.green,  0)
     : rv < MEDIUM_THRESH ? color.new(color.yellow, 0)
     : rv < HIGH_THRESH   ? color.new(color.orange, 0)
     :                      color.new(color.red,    0)

// ── Historical plot ───────────────────────────────────────────────
plot(rv, "Historical RV", color=rv_color, linewidth=2,
     style=plot.style_stepline)

// ── Regime threshold lines ────────────────────────────────────────
hline(LOW_THRESH,    "Low",    color=color.new(color.green,  40), linestyle=hline.style_dashed)
hline(MEDIUM_THRESH, "Medium", color=color.new(color.yellow, 40), linestyle=hline.style_dashed)
hline(HIGH_THRESH,   "High",   color=color.new(color.orange, 40), linestyle=hline.style_dashed)

// ── Background shading ────────────────────────────────────────────
bgcolor(
     na(rv)               ? na
     : rv < LOW_THRESH    ? color.new(color.green,  92)
     : rv < MEDIUM_THRESH ? color.new(color.yellow, 92)
     : rv < HIGH_THRESH   ? color.new(color.orange, 92)
     :                      color.new(color.red,    92),
     title="Volatility Regime")

// ── Future forecast line (drawn on the last bar) ──────────────────
if barstate.islast
    // Draw dotted lines connecting forecast points
    float prev_val = rv
    int   prev_idx = bar_index

    for i = 0 to array.size(_fval) - 1
        float fv = array.get(_fval, i)
        int   fi = bar_index + i + 1

        color fc =
             fv < LOW_THRESH    ? color.new(color.green,  30)
             : fv < MEDIUM_THRESH ? color.new(color.yellow, 30)
             : fv < HIGH_THRESH   ? color.new(color.orange, 30)
             :                      color.new(color.red,    30)

        line.new(prev_idx, prev_val, fi, fv,
                 color=fc, width=2,
                 style=line.style_dotted,
                 extend=extend.none)

        label.new(fi, fv,
                  "+" + str.tostring((i + 1) * 5) + "m\\n" + str.tostring(fv, "#.#####"),
                  style=label.style_label_down,
                  size=size.tiny,
                  color=color.new(fc, 60),
                  textcolor=color.white)

        prev_val := fv
        prev_idx := fi

    // Mark start of forecast
    label.new(bar_index, rv, "▶ forecast",
              style=label.style_label_left,
              size=size.small,
              color=color.new(color.blue, 50),
              textcolor=color.white)
"""
