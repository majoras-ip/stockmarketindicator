"""
dashboard.py — Web dashboard for the LSTM volatility forecaster.

Run with:
    python3 dashboard.py

Then open: http://localhost:5000
"""

from __future__ import annotations

import json
import os
import sys

from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import get_logger

log = get_logger(__name__)
app = Flask(__name__)
CORS(app)


# ── Forecast API ──────────────────────────────────────────────────────────────

@app.route("/api/forecast")
def api_forecast():
    ticker   = request.args.get("ticker", "SPY").upper()
    interval = request.args.get("interval", "5m")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"))
        from live.live_feed import fetch_bars
        interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        df = fetch_bars(ticker, interval_map.get(interval, "5Min"))
    except Exception as exc:
        log.warning("Alpaca failed (%s), falling back to yfinance", exc)
        from data.collector import download
        df = download(ticker, period="5d", interval=interval, save_csv=False)

    from prediction.forecast_exporter import ForecastExporter
    from data.forecast_preprocessor import FORECAST_STEPS
    from models.forecaster import build_sequences
    from data.preprocessor import _filter_market_hours, _remove_outliers
    from data.features import build_features
    import numpy as np

    exporter = ForecastExporter()
    exporter.load()

    df_clean = _remove_outliers(_filter_market_hours(df))
    features = build_features(df_clean).dropna()

    if exporter.feature_names:
        for col in set(exporter.feature_names) - set(features.columns):
            features[col] = 0.0
        features = features[exporter.feature_names]

    X_scaled = exporter.scaler.transform(features)
    dummy_y  = np.zeros(len(X_scaled))
    X_seq, _ = build_sequences(X_scaled, dummy_y, config.SEQUENCE_LENGTH, FORECAST_STEPS)

    if len(X_seq) == 0:
        return jsonify({"error": "Not enough data"}), 400

    all_preds = exporter.forecaster.predict(X_seq) * exporter.y_scale
    hist_rv   = all_preds[:, 0]
    hist_index = features.index[config.SEQUENCE_LENGTH : config.SEQUENCE_LENGTH + len(hist_rv)]

    mean, lower, upper = exporter.forecaster.predict_with_uncertainty(X_seq)
    mean  = mean  * exporter.y_scale
    lower = lower * exporter.y_scale
    upper = upper * exporter.y_scale

    interval_ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
    bar_ms   = interval_ms_map.get(interval, 300_000)
    last_ts  = int(hist_index[-1].timestamp() * 1000)
    future_ts = [last_ts + bar_ms * (i + 1) for i in range(FORECAST_STEPS)]

    direction_up   = None
    direction_conf = None
    if exporter.direction_model is not None:
        last_feat = features.iloc[[-1]].values
        proba = exporter.direction_model.predict_proba(last_feat)[0]
        direction_up   = bool(proba[1] >= 0.5)
        direction_conf = float(max(proba))

    arr = hist_rv
    return jsonify({
        "ticker":   ticker,
        "interval": interval,
        "hist_ts":  [int(ts.timestamp() * 1000) for ts in hist_index],
        "hist_rv":  [round(float(v), 8) for v in hist_rv],
        "future_ts": future_ts,
        "future_mean":  [round(float(v), 8) for v in mean],
        "future_lower": [round(float(v), 8) for v in lower],
        "future_upper": [round(float(v), 8) for v in upper],
        "low_thresh":    round(float(np.percentile(arr, 25)), 8),
        "medium_thresh": round(float(np.percentile(arr, 50)), 8),
        "high_thresh":   round(float(np.percentile(arr, 75)), 8),
        "direction_up":   direction_up,
        "direction_conf": direction_conf,
    })


# ── Pine Script JSON endpoint ─────────────────────────────────────────────────

@app.route("/api/pine")
def api_pine():
    """
    Lightweight endpoint for Pine Script v6 request.json().
    Returns flat JSON with forecast steps as f1..f6 + direction info.
    """
    ticker   = request.args.get("ticker", "SPY").upper()
    interval = request.args.get("interval", "5m")

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"))
        from live.live_feed import fetch_bars
        interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        df = fetch_bars(ticker, interval_map.get(interval, "5Min"))
    except Exception as exc:
        log.warning("Alpaca failed (%s), falling back to yfinance", exc)
        from data.collector import download
        df = download(ticker, period="5d", interval=interval, save_csv=False)

    from prediction.forecast_exporter import ForecastExporter
    from data.forecast_preprocessor import FORECAST_STEPS
    from models.forecaster import build_sequences
    from data.preprocessor import _filter_market_hours, _remove_outliers
    from data.features import build_features
    import numpy as np

    exporter = ForecastExporter()
    exporter.load()

    df_clean = _remove_outliers(_filter_market_hours(df))
    features = build_features(df_clean).dropna()

    if exporter.feature_names:
        for col in set(exporter.feature_names) - set(features.columns):
            features[col] = 0.0
        features = features[exporter.feature_names]

    X_scaled = exporter.scaler.transform(features)
    dummy_y  = __import__("numpy").zeros(len(X_scaled))
    X_seq, _ = build_sequences(X_scaled, dummy_y, config.SEQUENCE_LENGTH, FORECAST_STEPS)

    if len(X_seq) == 0:
        return jsonify({"error": "Not enough data"}), 400

    mean, lower, upper = exporter.forecaster.predict_with_uncertainty(X_seq)
    mean  = mean  * exporter.y_scale
    lower = lower * exporter.y_scale
    upper = upper * exporter.y_scale

    hist_rv    = exporter.forecaster.predict(X_seq)[:, 0] * exporter.y_scale
    low_thresh    = float(np.percentile(hist_rv, 25))
    medium_thresh = float(np.percentile(hist_rv, 50))
    high_thresh   = float(np.percentile(hist_rv, 75))

    direction_up   = None
    direction_conf = 0.5
    if exporter.direction_model is not None:
        proba = exporter.direction_model.predict_proba(features.iloc[[-1]].values)[0]
        direction_up   = bool(proba[1] >= 0.5)
        direction_conf = float(max(proba))

    result = {
        "low_thresh":    round(low_thresh,    8),
        "medium_thresh": round(medium_thresh, 8),
        "high_thresh":   round(high_thresh,   8),
        "direction_up":  1 if direction_up else 0,
        "confidence":    round(direction_conf, 4),
    }
    for i in range(FORECAST_STEPS):
        result[f"f{i+1}"]     = round(float(mean[i]),  8)
        result[f"f{i+1}_low"] = round(float(lower[i]), 8)
        result[f"f{i+1}_hi"]  = round(float(upper[i]), 8)

    return jsonify(result)


# ── Basic indicator Pine Script templates ────────────────────────────────────

def _pine_volume() -> str:
    return """\
//@version=5
indicator("24h Volume", overlay=false, max_bars_back=500)

// Cumulative volume since session open (resets each day)
is_new_day = dayofweek != dayofweek[1] or na(time[1])
cum_vol    = ta.cum(volume)
day_start  = ta.valuewhen(is_new_day, cum_vol - volume, 0)
vol_24h    = cum_vol - day_start

// Average daily volume (using daily timeframe)
avg_daily = request.security(syminfo.tickerid, "D", ta.sma(volume, 20))

is_high = vol_24h > avg_daily * 0.75

bar_color = is_high ? color.new(color.red, 20) : close >= open ? color.new(color.teal, 30) : color.new(color.gray, 40)

plot(vol_24h,   "24h Volume",    style=plot.style_columns, color=bar_color, linewidth=4)
plot(avg_daily, "Avg Daily Vol", color=color.orange, linewidth=2)

bgcolor(is_high ? color.new(color.red, 92) : na, title="Above Avg Volume")
"""

def _pine_vwap(band1: bool = True, band2: bool = True, band3: bool = False) -> str:
    lines = [
        '//@version=5',
        'indicator("VWAP + Bands", overlay=true, max_bars_back=500)',
        '',
        'vwap_val = ta.vwap(hlc3)',
        'stdev    = ta.stdev(hlc3, 20)',
        '',
        'plot(vwap_val, "VWAP", color=color.new(color.blue, 0), linewidth=2)',
    ]
    if band1:
        lines += [
            'upper1 = vwap_val + stdev',
            'lower1 = vwap_val - stdev',
            'p_u1 = plot(upper1, "+1 StDev", color=color.new(color.green, 40), linewidth=1)',
            'p_l1 = plot(lower1, "-1 StDev", color=color.new(color.green, 40), linewidth=1)',
            'fill(p_u1, p_l1, color=color.new(color.green, 90), title="±1 Band")',
        ]
    if band2:
        lines += [
            'upper2 = vwap_val + 2 * stdev',
            'lower2 = vwap_val - 2 * stdev',
            'p_u2 = plot(upper2, "+2 StDev", color=color.new(color.red, 40), linewidth=1)',
            'p_l2 = plot(lower2, "-2 StDev", color=color.new(color.red, 40), linewidth=1)',
            'fill(p_u2, p_l2, color=color.new(color.red, 92), title="±2 Band")',
        ]
    if band3:
        lines += [
            'upper3 = vwap_val + 3 * stdev',
            'lower3 = vwap_val - 3 * stdev',
            'p_u3 = plot(upper3, "+3 StDev", color=color.new(color.purple, 40), linewidth=1)',
            'p_l3 = plot(lower3, "-3 StDev", color=color.new(color.purple, 40), linewidth=1)',
            'fill(p_u3, p_l3, color=color.new(color.purple, 92), title="±3 Band")',
        ]
    return '\n'.join(lines) + '\n'

def _pine_atr() -> str:
    return """\
//@version=5
indicator("ATR", overlay=false, max_bars_back=500)

atr14   = ta.atr(14)
atr_avg = ta.sma(atr14, 20)
is_high = atr14 > atr_avg * 1.5

atr_color = is_high ? color.new(color.red, 20) : color.new(color.blue, 40)

plot(atr14,   "ATR(14)",    color=atr_color, linewidth=2)
plot(atr_avg, "20-bar Avg", color=color.new(color.orange, 0), linewidth=1)

bgcolor(is_high ? color.new(color.red, 90) : na, title="High ATR")
hline(0, color=color.new(color.gray, 80))
"""

def _pine_relvol() -> str:
    return """\
//@version=5
indicator("Relative Volume", overlay=false, max_bars_back=500)

lookback  = input.int(20,  "Lookback bars",      minval=5)
threshold = input.float(2.0, "High RVOL threshold", minval=1.0, step=0.1)

avg_vol = ta.sma(volume, lookback)
rvol    = avg_vol > 0 ? volume / avg_vol : 1.0

is_high   = rvol >= threshold
bar_color = is_high ? color.new(color.red, 20) : rvol >= 1.0 ? color.new(color.teal, 40) : color.new(color.gray, 50)

plot(rvol, "RVOL", style=plot.style_columns, color=bar_color)
hline(1.0,       "Average",   color=color.new(color.gray, 40), linestyle=hline.style_dashed)
hline(threshold, "High RVOL", color=color.new(color.red,  40), linestyle=hline.style_dashed)

bgcolor(is_high ? color.new(color.red, 90) : na, title="High RVOL")
"""

def _pine_ma_cross() -> str:
    return """\
//@version=5
indicator("MA Cross", overlay=true, max_bars_back=500)

fast = input.int(50,  "Fast MA", minval=1)
slow = input.int(200, "Slow MA", minval=1)

ma_fast = ta.sma(close, fast)
ma_slow = ta.sma(close, slow)

golden = ta.crossover(ma_fast,  ma_slow)
death  = ta.crossunder(ma_fast, ma_slow)

plot(ma_fast, "Fast MA", color=color.new(color.blue,   0), linewidth=2)
plot(ma_slow, "Slow MA", color=color.new(color.orange, 0), linewidth=2)

plotshape(golden, "Golden Cross", style=shape.labelup,   location=location.belowbar,
          color=color.new(color.green, 0), text="Golden", size=size.small)
plotshape(death,  "Death Cross",  style=shape.labeldown, location=location.abovebar,
          color=color.new(color.red,   0), text="Death",  size=size.small)

bgcolor(ma_fast > ma_slow ? color.new(color.green, 95) : color.new(color.red, 95), title="Trend")
"""

INDICATORS = {
    "volume":  ("24h Volume",       _pine_volume,   "Cumulative volume since market open vs 20-day average daily volume. Red = above average day."),
    "vwap":    ("VWAP + Bands",    _pine_vwap,     "Volume Weighted Average Price with configurable ±1, ±2, ±3 standard deviation bands. Overlaid on price."),
    "atr":     ("ATR",             _pine_atr,      "Average True Range (14). Shows how much the asset moves per bar. Red = elevated volatility."),
    "relvol":  ("Relative Volume", _pine_relvol,   "Today's volume vs average. RVOL > 2 = unusually active. Threshold is adjustable."),
    "macross": ("MA Cross",        _pine_ma_cross, "50/200 MA crossover. Labels Golden Cross (bullish) and Death Cross (bearish) on the chart."),
}


@app.route("/indicators")
def indicators_page():
    kind = request.args.get("kind", "")

    pine_code   = None
    description = None
    name        = None

    # VWAP band options
    band1 = request.args.get("band1", "on") == "on"
    band2 = request.args.get("band2", "on") == "on"
    band3 = request.args.get("band3", "off") == "on"

    if kind and kind in INDICATORS:
        name, fn, description = INDICATORS[kind]
        if kind == "vwap":
            pine_code = fn(band1=band1, band2=band2, band3=band3)
        else:
            pine_code = fn()

    return render_template_string(INDICATORS_HTML,
        kind=kind,
        indicators=INDICATORS,
        pine_code=pine_code,
        name=name,
        description=description,
        band1=band1, band2=band2, band3=band3)


# ── Pine Script generator endpoint ───────────────────────────────────────────

@app.route("/generate")
def generate():
    ticker   = request.args.get("ticker", "SPY").upper()
    interval = request.args.get("interval", "5m")

    # If user just landed on the page without submitting, show empty form
    if "ticker" not in request.args:
        return render_template_string(GENERATOR_HTML,
            ticker="SPY", interval="5m",
            pine_code=None, arrow=None, conf=None, error=None)

    try:
        from data.collector import download
        from prediction.forecast_exporter import ForecastExporter, _build_forecast_pine
        from data.forecast_preprocessor import FORECAST_STEPS
        from models.forecaster import build_sequences
        from data.preprocessor import _filter_market_hours, _remove_outliers
        from data.features import build_features
        from datetime import timezone
        import numpy as np

        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(config.BASE_DIR, ".env"))
            from live.live_feed import fetch_bars
            interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
            df = fetch_bars(ticker, interval_map.get(interval, "5Min"))
        except Exception:
            df = download(ticker, period="5d", interval=interval, save_csv=False)

        exporter = ForecastExporter()
        exporter.load()

        df_clean = _remove_outliers(_filter_market_hours(df))
        features = build_features(df_clean).dropna()
        if exporter.feature_names:
            for col in set(exporter.feature_names) - set(features.columns):
                features[col] = 0.0
            features = features[exporter.feature_names]

        X_scaled = exporter.scaler.transform(features)
        dummy_y  = np.zeros(len(X_scaled))
        X_seq, _ = build_sequences(X_scaled, dummy_y, config.SEQUENCE_LENGTH, FORECAST_STEPS)

        all_preds = exporter.forecaster.predict(X_seq) * exporter.y_scale
        hist_rv   = all_preds[:, 0]
        hist_index = features.index[config.SEQUENCE_LENGTH : config.SEQUENCE_LENGTH + len(hist_rv)]

        mean, lower, upper = exporter.forecaster.predict_with_uncertainty(X_seq)
        mean  = mean  * exporter.y_scale
        lower = lower * exporter.y_scale
        upper = upper * exporter.y_scale

        interval_ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
        bar_ms   = interval_ms_map.get(interval, 300_000)
        last_ts  = int(hist_index[-1].timestamp() * 1000)
        future_ts = [last_ts + bar_ms * (i + 1) for i in range(FORECAST_STEPS)]

        direction_up   = None
        direction_conf = None
        if exporter.direction_model is not None:
            proba = exporter.direction_model.predict_proba(features.iloc[[-1]].values)[0]
            direction_up   = bool(proba[1] >= 0.5)
            direction_conf = float(max(proba))

        pine_code = _build_forecast_pine(
            ticker=ticker,
            interval=interval,
            timestamps_ms=[int(ts.timestamp() * 1000) for ts in hist_index],
            hist_values=[round(float(v), 8) for v in hist_rv],
            future_timestamps_ms=future_ts,
            future_values=[round(float(v), 8) for v in mean],
            lower_values=[round(float(v), 8) for v in lower],
            upper_values=[round(float(v), 8) for v in upper],
            direction_up=direction_up,
            direction_conf=direction_conf,
        )

        arrow = ("▲ UP" if direction_up else "▼ DOWN") if direction_up is not None else "—"
        conf_pct = f"{direction_conf:.0%}" if direction_conf else "—"

        return render_template_string(GENERATOR_HTML,
            ticker=ticker, interval=interval,
            pine_code=pine_code,
            arrow=arrow, conf=conf_pct,
            error=None)

    except Exception as exc:
        log.exception("Generate error")
        return render_template_string(GENERATOR_HTML,
            ticker=ticker, interval=interval,
            pine_code=None, arrow=None, conf=None,
            error=str(exc))


# ── Indicators HTML ──────────────────────────────────────────────────────────

INDICATORS_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Free TradingView Indicators — Pine Script Generator</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d1117; color: #e6edf3; min-height: 100vh; }
    nav {
      background: #161b22; border-bottom: 1px solid #30363d;
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: #58a6ff; font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links a { color: #8b949e; text-decoration: none; margin-left: 20px; font-size: 0.9rem; }
    .nav-links a:hover { color: #e6edf3; }

    .hero { text-align: center; padding: 48px 24px 32px; border-bottom: 1px solid #21262d; }
    .hero h1 { font-size: 1.8rem; color: #e6edf3; margin-bottom: 10px; }
    .hero h1 span { color: #58a6ff; }
    .hero p { color: #8b949e; font-size: 0.95rem; max-width: 520px; margin: 0 auto; }

    .container { max-width: 760px; margin: 0 auto; padding: 32px 24px; }

    /* Indicator cards */
    .indicator-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 28px; }
    .ind-card {
      background: #161b22; border: 1px solid #30363d; border-radius: 8px;
      padding: 14px 16px; cursor: pointer; transition: border-color 0.15s;
      text-decoration: none; display: block;
    }
    .ind-card:hover { border-color: #58a6ff; }
    .ind-card.active { border-color: #58a6ff; background: #0d2a4a; }
    .ind-card .ind-name { color: #e6edf3; font-size: 0.9rem; font-weight: bold; margin-bottom: 4px; }
    .ind-card .ind-desc { color: #8b949e; font-size: 0.75rem; line-height: 1.4; }

    /* Form */
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 24px; margin-bottom: 24px; }
    .form-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    label { color: #8b949e; font-size: 0.8rem; }
    input, select {
      background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
      padding: 8px 14px; border-radius: 6px; font-family: monospace; font-size: 0.9rem;
    }
    input:focus, select:focus { outline: none; border-color: #58a6ff; }
    .btn-generate {
      background: #1f6feb; color: #fff; border: none;
      padding: 9px 24px; border-radius: 6px; font-family: monospace;
      font-size: 0.9rem; cursor: pointer; font-weight: bold;
    }
    .btn-generate:hover { background: #388bfd; }

    /* Output */
    .pine-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .pine-label span { color: #8b949e; font-size: 0.8rem; }
    .btn-copy {
      background: #21262d; color: #e6edf3; border: 1px solid #30363d;
      padding: 5px 14px; border-radius: 6px; font-family: monospace; font-size: 0.8rem; cursor: pointer;
    }
    .btn-copy:hover { background: #30363d; }
    .btn-copy.copied { border-color: #3fb950; color: #3fb950; }
    textarea {
      width: 100%; height: 320px; background: #0d1117; color: #c9d1d9;
      border: 1px solid #30363d; border-radius: 6px; padding: 14px;
      font-family: monospace; font-size: 0.78rem; line-height: 1.55; resize: vertical;
    }
    .desc-box { background: #0d2a14; border: 1px solid #3fb950; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; color: #3fb950; font-size: 0.85rem; }
    .error-box { background: #2d1316; border: 1px solid #f85149; border-radius: 6px; padding: 14px; color: #f85149; font-size: 0.85rem; }

    footer { text-align: center; padding: 32px 24px; color: #484f58; font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">
    <a href="/indicators">Indicators</a>
    <a href="/generate">Forecast</a>
    <a href="/dashboard">Live Chart</a>
  </div>
</nav>

<div class="hero">
  <h1>Free <span>TradingView</span> Indicators</h1>
  <p>Pick an indicator, enter a ticker, and get Pine Script to paste directly into TradingView. No paid plan needed.</p>
</div>

<div class="container">
  <div class="indicator-grid">
    {% for key, (iname, _, idesc) in indicators.items() %}
    <a class="ind-card {{ 'active' if key == kind else '' }}"
       href="/indicators?kind={{ key }}&ticker={{ ticker }}">
      <div class="ind-name">{{ iname }}</div>
      <div class="ind-desc">{{ idesc[:60] }}…</div>
    </a>
    {% endfor %}
  </div>

  {% if kind == 'vwap' %}
  <div class="card" style="margin-bottom:0; border-bottom-left-radius:0; border-bottom-right-radius:0; border-bottom:none;">
    <form method="GET" action="/indicators" style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;">
      <input type="hidden" name="kind" value="vwap">
      <span style="color:#8b949e; font-size:0.85rem;">Bands:</span>
      <label style="color:#3fb950; font-size:0.85rem; cursor:pointer;">
        <input type="checkbox" name="band1" value="on" {% if band1 %}checked{% endif %} onchange="this.form.submit()">
        ±1 StDev
      </label>
      <label style="color:#f85149; font-size:0.85rem; cursor:pointer;">
        <input type="checkbox" name="band2" value="on" {% if band2 %}checked{% endif %} onchange="this.form.submit()">
        ±2 StDev
      </label>
      <label style="color:#a371f7; font-size:0.85rem; cursor:pointer;">
        <input type="checkbox" name="band3" value="on" {% if band3 %}checked{% endif %} onchange="this.form.submit()">
        ±3 StDev
      </label>
    </form>
  </div>
  {% endif %}

  {% if pine_code %}
  <div class="card" style="{{ 'border-top-left-radius:0; border-top-right-radius:0;' if kind == 'vwap' else '' }}">
    <div class="desc-box">{{ description }}</div>
    <div class="pine-label">
      <span>Pine Script v5 — works on any chart, no ticker needed</span>
      <button class="btn-copy" onclick="copyPine()">Copy</button>
    </div>
    <textarea id="pine-out" readonly>{{ pine_code }}</textarea>
  </div>
  {% endif %}
</div>

<footer>VolForecast — Free Pine Script indicators · Not financial advice</footer>

<script>
function copyPine() {
  const ta = document.getElementById('pine-out');
  ta.select();
  document.execCommand('copy');
  const btn = document.querySelector('.btn-copy');
  btn.textContent = 'Copied!';
  btn.classList.add('copied');
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
}
</script>
</body>
</html>"""


# ── Generator HTML ───────────────────────────────────────────────────────────

GENERATOR_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Volatility Forecast — Free TradingView Pine Script Generator</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d1117; color: #e6edf3; min-height: 100vh; }

    /* Nav */
    nav {
      background: #161b22; border-bottom: 1px solid #30363d;
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: #58a6ff; font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links a { color: #8b949e; text-decoration: none; margin-left: 20px; font-size: 0.9rem; }
    .nav-links a:hover { color: #e6edf3; }

    /* Hero */
    .hero {
      text-align: center; padding: 56px 24px 40px;
      background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
      border-bottom: 1px solid #21262d;
    }
    .hero h1 { font-size: 2rem; color: #e6edf3; margin-bottom: 12px; line-height: 1.3; }
    .hero h1 span { color: #58a6ff; }
    .hero p { color: #8b949e; font-size: 1rem; max-width: 560px; margin: 0 auto 8px; line-height: 1.6; }
    .badge-free {
      display: inline-block; margin-top: 14px;
      background: #0d3349; color: #58a6ff; border: 1px solid #1f6feb;
      padding: 4px 14px; border-radius: 20px; font-size: 0.8rem;
    }

    /* Card */
    .card {
      background: #161b22; border: 1px solid #30363d; border-radius: 10px;
      padding: 28px; margin: 32px auto; max-width: 720px;
    }
    .card h2 { color: #58a6ff; font-size: 1rem; margin-bottom: 20px; }

    /* Form row */
    .form-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    label { color: #8b949e; font-size: 0.8rem; }
    input, select {
      background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
      padding: 8px 14px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; min-width: 120px;
    }
    input:focus, select:focus { outline: none; border-color: #58a6ff; }

    .btn-generate {
      background: #1f6feb; color: #fff; border: none;
      padding: 9px 24px; border-radius: 6px; font-family: monospace;
      font-size: 0.9rem; cursor: pointer; font-weight: bold;
    }
    .btn-generate:hover { background: #388bfd; }
    .btn-generate:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

    /* Direction badge */
    .dir-row { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
    .dir-badge {
      padding: 5px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: bold;
      border: 1px solid #30363d; background: #21262d;
    }
    .dir-badge.up   { border-color: #3fb950; color: #3fb950; background: #0d2a14; }
    .dir-badge.down { border-color: #f85149; color: #f85149; background: #2d1316; }
    .conf-badge { color: #8b949e; font-size: 0.85rem; }

    /* Pine code area */
    .pine-label {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 8px;
    }
    .pine-label span { color: #8b949e; font-size: 0.8rem; }
    .btn-copy {
      background: #21262d; color: #e6edf3; border: 1px solid #30363d;
      padding: 5px 14px; border-radius: 6px; font-family: monospace;
      font-size: 0.8rem; cursor: pointer;
    }
    .btn-copy:hover { background: #30363d; }
    .btn-copy.copied { border-color: #3fb950; color: #3fb950; }

    textarea {
      width: 100%; height: 380px; background: #0d1117; color: #c9d1d9;
      border: 1px solid #30363d; border-radius: 6px; padding: 14px;
      font-family: monospace; font-size: 0.78rem; line-height: 1.55;
      resize: vertical;
    }
    textarea:focus { outline: none; border-color: #58a6ff; }

    /* Steps */
    .steps { margin-top: 28px; }
    .steps h3 { color: #8b949e; font-size: 0.8rem; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
    .step {
      display: flex; gap: 12px; align-items: flex-start;
      padding: 10px 0; border-bottom: 1px solid #21262d; font-size: 0.88rem;
    }
    .step:last-child { border-bottom: none; }
    .step-num {
      background: #1f6feb; color: #fff; border-radius: 50%;
      width: 22px; height: 22px; min-width: 22px; display: flex;
      align-items: center; justify-content: center; font-size: 0.75rem; font-weight: bold;
    }
    .step-text { color: #8b949e; line-height: 1.5; }
    .step-text strong { color: #e6edf3; }

    /* Error */
    .error-box {
      background: #2d1316; border: 1px solid #f85149; border-radius: 6px;
      padding: 14px; color: #f85149; font-size: 0.85rem; margin-bottom: 16px;
    }

    /* Spinner */
    .spinner {
      display: inline-block; width: 14px; height: 14px;
      border: 2px solid #30363d; border-top-color: #58a6ff;
      border-radius: 50%; animation: spin 0.6s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    footer {
      text-align: center; padding: 32px 24px; color: #484f58; font-size: 0.8rem;
      border-top: 1px solid #21262d; margin-top: 40px;
    }
  </style>
</head>
<body>

<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">
    <a href="/indicators">Indicators</a>
    <a href="/generate">Forecast</a>
    <a href="/dashboard">Live Chart</a>
  </div>
</nav>

<div class="hero">
  <h1>Free TradingView<br><span>Volatility Forecast</span> Indicator</h1>
  <p>Generate a custom Pine Script indicator with an LSTM-powered volatility forecast — no paid TradingView plan required.</p>
  <span class="badge-free">✓ Works on free TradingView accounts</span>
</div>

<div class="card">
  <h2>Generate Your Indicator</h2>
  <form method="GET" action="/generate" id="genForm">
    <div class="form-row">
      <div class="form-group">
        <label for="ticker">Ticker</label>
        <input id="ticker" name="ticker" value="{{ ticker }}" placeholder="SPY" style="width:100px; text-transform:uppercase">
      </div>
      <div class="form-group">
        <label for="interval">Interval</label>
        <select id="interval" name="interval">
          <option value="5m"  {% if interval == '5m'  %}selected{% endif %}>5m</option>
          <option value="1m"  {% if interval == '1m'  %}selected{% endif %}>1m</option>
          <option value="15m" {% if interval == '15m' %}selected{% endif %}>15m</option>
          <option value="1h"  {% if interval == '1h'  %}selected{% endif %}>1h</option>
        </select>
      </div>
      <button class="btn-generate" type="submit" id="genBtn">Generate</button>
    </div>
  </form>

  {% if error %}
  <div class="error-box" style="margin-top:20px">Error: {{ error }}</div>
  {% endif %}

  {% if pine_code %}
  <div style="margin-top:24px">
    <div class="dir-row">
      <span class="dir-badge {{ 'up' if '▲' in arrow else 'down' }}">{{ arrow }}</span>
      <span class="conf-badge">Confidence: {{ conf }}</span>
    </div>

    <div class="pine-label">
      <span>Pine Script v5 — paste into TradingView Pine Editor</span>
      <button class="btn-copy" onclick="copyPine()">Copy</button>
    </div>
    <textarea id="pine-out" readonly>{{ pine_code }}</textarea>
  </div>
  {% endif %}

  <div class="steps" style="margin-top: {{ '28px' if pine_code else '28px' }}">
    <h3>How to use</h3>
    <div class="step">
      <div class="step-num">1</div>
      <div class="step-text">Enter a ticker (e.g. <strong>SPY</strong>, <strong>AAPL</strong>) and interval, then click <strong>Generate</strong>.</div>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <div class="step-text">Click <strong>Copy</strong> to copy the generated Pine Script code.</div>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <div class="step-text">Open <strong>TradingView</strong> → any chart → click <strong>Pine Script Editor</strong> at the bottom.</div>
    </div>
    <div class="step">
      <div class="step-num">4</div>
      <div class="step-text">Paste the code → click <strong>Add to chart</strong>. The forecast appears as a separate panel below the chart.</div>
    </div>
  </div>
</div>

<footer>
  VolForecast — LSTM volatility prediction · Not financial advice
</footer>

<script>
  // Show spinner while generating
  document.getElementById('genForm').addEventListener('submit', function() {
    const btn = document.getElementById('genBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Generating…';
  });

  function copyPine() {
    const ta = document.getElementById('pine-out');
    ta.select();
    document.execCommand('copy');
    const btn = document.querySelector('.btn-copy');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  }
</script>
</body>
</html>"""


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Volatility Forecast Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0d1117; color: #e6edf3; padding: 20px; }
    h1 { color: #58a6ff; margin-bottom: 16px; font-size: 1.4rem; }
    .controls { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
    input, select, button {
      background: #161b22; color: #e6edf3; border: 1px solid #30363d;
      padding: 6px 12px; border-radius: 6px; font-family: monospace; font-size: 0.9rem;
    }
    button { background: #21262d; cursor: pointer; }
    button:hover { background: #30363d; }
    #chart { width: 100%; height: 500px; }
    .info-bar { display: flex; gap: 20px; margin-bottom: 12px; flex-wrap: wrap; }
    .badge {
      padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;
      background: #21262d; border: 1px solid #30363d;
    }
    .badge.up   { border-color: #3fb950; color: #3fb950; }
    .badge.down { border-color: #f85149; color: #f85149; }
    .badge.vol  { border-color: #58a6ff; color: #58a6ff; }
    #status { color: #8b949e; font-size: 0.8rem; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Volatility Forecast Dashboard</h1>

  <div class="controls">
    <input id="ticker" value="SPY" style="width:80px; text-transform:uppercase">
    <select id="interval">
      <option value="5m">5m</option>
      <option value="1m">1m</option>
      <option value="15m">15m</option>
      <option value="1h">1h</option>
    </select>
    <button onclick="refresh()">Refresh</button>
    <span id="auto-label" style="color:#8b949e;font-size:0.85rem">Auto-refresh: ON</span>
    <button onclick="toggleAuto()">Pause</button>
  </div>

  <div class="info-bar">
    <span class="badge vol" id="badge-vol">Loading…</span>
    <span class="badge" id="badge-dir"></span>
    <span class="badge" id="badge-conf"></span>
  </div>

  <div id="chart"></div>
  <div id="status"></div>

<script>
let autoRefresh = true;
let timer = null;

function toggleAuto() {
  autoRefresh = !autoRefresh;
  document.getElementById('auto-label').textContent = 'Auto-refresh: ' + (autoRefresh ? 'ON' : 'OFF');
  document.querySelector('button:last-of-type').textContent = autoRefresh ? 'Pause' : 'Resume';
  if (autoRefresh) scheduleRefresh();
  else clearTimeout(timer);
}

function scheduleRefresh() {
  clearTimeout(timer);
  if (autoRefresh) timer = setTimeout(refresh, 5 * 60 * 1000);
}

async function refresh() {
  const ticker   = document.getElementById('ticker').value.toUpperCase();
  const interval = document.getElementById('interval').value;
  document.getElementById('status').textContent = 'Fetching ' + ticker + ' ' + interval + '…';

  try {
    const res  = await fetch('/api/forecast?ticker=' + ticker + '&interval=' + interval);
    const data = await res.json();
    if (data.error) { document.getElementById('status').textContent = 'Error: ' + data.error; return; }
    drawChart(data);
    updateBadges(data);
    document.getElementById('status').textContent =
      'Last updated: ' + new Date().toLocaleTimeString() + ' — next refresh in 5 min';
    scheduleRefresh();
  } catch(e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
    scheduleRefresh();
  }
}

function updateBadges(d) {
  const lastRv = d.hist_rv[d.hist_rv.length - 1];
  document.getElementById('badge-vol').textContent = 'RV: ' + lastRv.toFixed(5);

  const dirEl  = document.getElementById('badge-dir');
  const confEl = document.getElementById('badge-conf');
  if (d.direction_up !== null) {
    dirEl.textContent  = d.direction_up ? '▲ UP' : '▼ DOWN';
    dirEl.className    = 'badge ' + (d.direction_up ? 'up' : 'down');
    confEl.textContent = 'Confidence: ' + Math.round(d.direction_conf * 100) + '%';
  }
}

function drawChart(d) {
  const histDates   = d.hist_ts.map(t => new Date(t));
  const futureDates = d.future_ts.map(t => new Date(t));

  const traces = [
    // Historical RV
    {
      x: histDates, y: d.hist_rv,
      name: 'Historical RV', type: 'scatter', mode: 'lines',
      line: { color: '#58a6ff', width: 2, shape: 'hv' },
    },
    // Forecast mean
    {
      x: futureDates, y: d.future_mean,
      name: 'Forecast', type: 'scatter', mode: 'lines+markers',
      line: { color: '#f0883e', width: 2, dash: 'dot' },
      marker: { size: 6 },
    },
    // Confidence band (upper)
    {
      x: futureDates, y: d.future_upper,
      name: 'Upper band', type: 'scatter', mode: 'lines',
      line: { width: 0 }, showlegend: false,
    },
    // Confidence band (lower — filled to upper)
    {
      x: futureDates, y: d.future_lower,
      name: 'Confidence band', type: 'scatter', mode: 'lines',
      fill: 'tonexty', fillcolor: 'rgba(240,136,62,0.15)',
      line: { width: 0 },
    },
    // Threshold lines as scatter (hlines workaround)
    {
      x: [histDates[0], futureDates[futureDates.length-1]],
      y: [d.low_thresh, d.low_thresh],
      name: 'Low', mode: 'lines',
      line: { color: '#3fb950', width: 1, dash: 'dash' },
    },
    {
      x: [histDates[0], futureDates[futureDates.length-1]],
      y: [d.medium_thresh, d.medium_thresh],
      name: 'Medium', mode: 'lines',
      line: { color: '#d29922', width: 1, dash: 'dash' },
    },
    {
      x: [histDates[0], futureDates[futureDates.length-1]],
      y: [d.high_thresh, d.high_thresh],
      name: 'High', mode: 'lines',
      line: { color: '#f85149', width: 1, dash: 'dash' },
    },
  ];

  const layout = {
    paper_bgcolor: '#0d1117', plot_bgcolor: '#0d1117',
    font: { color: '#e6edf3', family: 'monospace' },
    xaxis: { gridcolor: '#21262d', tickformat: '%H:%M' },
    yaxis: { gridcolor: '#21262d', title: 'Realised Volatility' },
    legend: { bgcolor: '#161b22', bordercolor: '#30363d', borderwidth: 1 },
    margin: { t: 20, r: 20, b: 40, l: 60 },
    shapes: [{
      type: 'line', xref: 'x',
      x0: new Date(d.hist_ts[d.hist_ts.length-1]),
      x1: new Date(d.hist_ts[d.hist_ts.length-1]),
      y0: 0, y1: 1, yref: 'paper',
      line: { color: '#8b949e', width: 1, dash: 'dot' },
    }],
  };

  Plotly.react('chart', traces, layout, { responsive: true, displayModeBar: false });
}

// Initial load
refresh();
</script>
</body>
</html>"""


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/")
def index():
    from flask import redirect
    return redirect("/generate")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════╗
║     Volatility Forecast Dashboard        ║
╠══════════════════════════════════════════╣
║  Open: http://localhost:{port:<16}║
║  Press Ctrl+C to stop                   ║
╚══════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False)
