"""
dashboard.py — Web dashboard for the LSTM volatility forecaster.

Run with:
    python3 dashboard.py

Then open: http://localhost:5000
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import sys
from functools import wraps

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import get_logger

log = get_logger(__name__)
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
CORS(app)

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                pw_hash  TEXT NOT NULL,
                created  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS favorites (
                user_id       INTEGER NOT NULL,
                indicator_key TEXT NOT NULL,
                PRIMARY KEY (user_id, indicator_key),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                author      TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL,
                votes       INTEGER DEFAULT 0,
                created     TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS request_votes (
                user_id    INTEGER NOT NULL,
                request_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, request_id)
            );
        """)

init_db()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(f"/login?next={request.path}")
        return f(*args, **kwargs)
    return decorated

def current_user():
    return session.get("username")


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

def _pine_atr(show_avg: bool = False) -> str:
    avg_line = 'plot(atr_avg, "20-bar Avg", color=color.new(color.orange, 0), linewidth=1)' if show_avg else ''
    return f"""\
//@version=5
indicator("ATR", overlay=false, max_bars_back=500)

atr14   = ta.atr(14)
atr_avg = ta.sma(atr14, 20)
is_high = atr14 > atr_avg * 1.5

plot(atr14, "ATR(14)", color=color.new(color.red, 0), linewidth=2)
{avg_line}

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

def _pine_feargreed() -> str:
    return """\
//@version=5
indicator("Fear & Greed Index", overlay=false, max_bars_back=500)

// ── Components ────────────────────────────────────────────────────
// 1. RSI momentum (already 0-100)
rsi_raw = ta.rsi(close, 14)

// 2. Price vs 125-day MA (normalize to 0-100)
ma125     = ta.sma(close, 125)
ma_pct    = (close - ma125) / ma125 * 100
ma_score  = math.min(math.max(50 + ma_pct * 2, 0), 100)

// 3. Bollinger Band width — low width = greed, high = fear (invert)
basis = ta.sma(close, 20)
[bb_mid, bb_upper, bb_lower] = ta.bb(close, 20, 2)
bb_width  = (bb_upper - bb_lower) / basis * 100
bb_avg    = ta.sma(bb_width, 50)
bb_score  = math.min(math.max(50 - (bb_width - bb_avg) * 5, 0), 100)

// 4. VIX — fear when high (invert: VIX 10=greed 100, VIX 40=fear 0)
vix       = request.security("CBOE:VIX", timeframe.period, close)
vix_score = math.min(math.max(100 - (vix - 10) * (100 / 30), 0), 100)

// 5. 52-week momentum — rate of change
roc52     = ta.roc(close, 252)
roc_score = math.min(math.max(50 + roc52, 0), 100)

// ── Composite (weighted average) ─────────────────────────────────
fg = (rsi_raw * 0.25 + ma_score * 0.25 + bb_score * 0.15 + vix_score * 0.20 + roc_score * 0.15)

// ── Color by zone ─────────────────────────────────────────────────
fg_color =
     fg < 25 ? color.new(color.red,    0) :
     fg < 45 ? color.new(color.orange, 0) :
     fg < 55 ? color.new(color.yellow, 0) :
     fg < 75 ? color.new(color.teal,   0) :
               color.new(color.green,  0)

// ── Plot ──────────────────────────────────────────────────────────
plot(fg, "Fear & Greed", color=fg_color, linewidth=3, style=plot.style_line)

hline(75, "Extreme Greed", color=color.new(color.green,  40), linestyle=hline.style_dashed)
hline(55, "Greed",         color=color.new(color.teal,   40), linestyle=hline.style_dashed)
hline(45, "Fear",          color=color.new(color.orange, 40), linestyle=hline.style_dashed)
hline(25, "Extreme Fear",  color=color.new(color.red,    40), linestyle=hline.style_dashed)

bgcolor(
     fg < 25 ? color.new(color.red,    92) :
     fg < 45 ? color.new(color.orange, 92) :
     fg < 55 ? color.new(color.yellow, 92) :
     fg < 75 ? color.new(color.teal,   92) :
               color.new(color.green,  92), title="Zone")

// ── Label on last bar ─────────────────────────────────────────────
if barstate.islast
    txt = fg < 25 ? "Extreme Fear" : fg < 45 ? "Fear" : fg < 55 ? "Neutral" : fg < 75 ? "Greed" : "Extreme Greed"
    label.new(bar_index, fg, txt + "  " + str.tostring(math.round(fg)),
              style=label.style_label_left, size=size.small,
              color=fg_color, textcolor=color.white)
"""


# Each entry: (name, fn, short_description, category, how_to_use)
INDICATORS = {
    "volume":    ("24h Volume",      _pine_volume,    "Cumulative volume since market open vs 20-day average daily volume. Red = above average day.",          "volume",     "Add to any chart. Appears as a separate panel below the price. Green bars = normal up-volume, red bars = unusually high volume (1.5× avg). Orange line is the average daily volume benchmark — when the bars are tracking above it, today is an active day."),
    "vwap":      ("VWAP + Bands",    _pine_vwap,      "VWAP with configurable ±1, ±2, ±3 standard deviation bands. Overlaid directly on the price chart.",    "volume",     "Select which bands to show, then copy and paste onto your chart. Price above VWAP = buyers in control. Price near the ±2 red band = overextended, often snaps back. Use on intraday charts (1m–1h) — VWAP resets each day."),
    "vwap_only": ("VWAP Only",       lambda: "//@version=5\nindicator(\"VWAP\", overlay=true, max_bars_back=500)\n\nplot(ta.vwap(hlc3), \"VWAP\", color=color.new(color.blue, 0), linewidth=2)\n", "Just the VWAP line, no bands. Clean and simple, overlaid on the price chart.", "volume", "Paste onto any intraday chart. A single blue line shows the volume-weighted average price for the session. Price above = bullish bias, price below = bearish bias. Resets at market open each day."),
    "atr":       ("ATR",             _pine_atr,       "Average True Range (14). Shows how much the asset moves per bar. Red = elevated volatility.",           "volatility", "Add to any chart as a separate panel. The red line shows raw volatility per bar. Toggle the orange average line to see whether current volatility is above or below normal. High ATR = bigger stops needed. Low ATR = tight, choppy market."),
    "relvol":    ("Relative Volume", _pine_relvol,    "Today's volume vs average. RVOL > 2 = unusually active. Threshold is adjustable in TradingView.",      "volume",     "Add as a separate panel. RVOL of 1.0 = exactly average. Teal bars = above average. Red bars = unusually high (default threshold: 2×). High RVOL on a breakout confirms the move. High RVOL on a reversal signals a strong change. Low RVOL moves are often noise."),
    "macross":   ("MA Cross",        _pine_ma_cross,  "50/200 MA crossover. Labels Golden Cross (bullish) and Death Cross (bearish) directly on the chart.",  "trend",      "Paste onto your price chart. Green background = uptrend (50 above 200). Red background = downtrend. A 'Golden' label appears when the 50 crosses above the 200 — historically a strong bullish signal. 'Death' appears on the cross below. Best used on daily charts."),
    "feargreed": ("Fear & Greed",    _pine_feargreed, "Composite 0–100 index built from RSI, trend strength, volatility, VIX, and 52-week momentum.",         "momentum",   "Add as a separate panel on any chart. Reads 0–100: below 25 = Extreme Fear (often a buying opportunity), above 75 = Extreme Greed (market may be overheated). The label on the last bar shows the current reading and zone. Works on any ticker — uses VIX as one of its inputs."),
}

CATEGORIES = {
    "all":        "All",
    "trend":      "Trend",
    "momentum":   "Momentum",
    "volume":     "Volume",
    "volatility": "Volatility",
}


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                with get_db() as conn:
                    conn.execute("INSERT INTO users (username, pw_hash) VALUES (?, ?)",
                                 (username, generate_password_hash(password)))
                row = get_db().execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
                session["user_id"] = row["id"]
                session["username"] = username
                return redirect("/indicators")
            except sqlite3.IntegrityError:
                error = "Username already taken."
    return render_template_string(AUTH_HTML, mode="register", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = get_db().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if not row or not check_password_hash(row["pw_hash"], password):
            error = "Invalid username or password."
        else:
            session["user_id"] = row["id"]
            session["username"] = row["username"]
            next_url = request.args.get("next", "/indicators")
            return redirect(next_url)
    return render_template_string(AUTH_HTML, mode="login", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ── Favorites API ─────────────────────────────────────────────────────────────

@app.route("/api/favorite/<key>", methods=["POST"])
@login_required
def toggle_favorite(key):
    user_id = session["user_id"]
    with get_db() as conn:
        existing = conn.execute(
            "SELECT 1 FROM favorites WHERE user_id=? AND indicator_key=?", (user_id, key)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM favorites WHERE user_id=? AND indicator_key=?", (user_id, key))
            return jsonify({"favorited": False})
        else:
            conn.execute("INSERT INTO favorites (user_id, indicator_key) VALUES (?, ?)", (user_id, key))
            return jsonify({"favorited": True})


@app.route("/favorites")
@login_required
def favorites_page():
    user_id = session["user_id"]
    rows = get_db().execute(
        "SELECT indicator_key FROM favorites WHERE user_id=?", (user_id,)
    ).fetchall()
    keys = [r["indicator_key"] for r in rows]
    saved = {k: v for k, v in INDICATORS.items() if k in keys}
    return render_template_string(FAVORITES_HTML,
        indicators=saved, current_user=current_user())


# ── Request route ─────────────────────────────────────────────────────────────

@app.route("/request", methods=["GET", "POST"])
def request_page():
    error = None
    success = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        desc = request.form.get("description", "").strip()
        author = session.get("username", "Anonymous")
        if not name or not desc:
            error = "Please fill in both fields."
        else:
            with get_db() as conn:
                conn.execute("INSERT INTO requests (author, name, description) VALUES (?, ?, ?)",
                             (author, name, desc))
            success = "Request submitted! Thanks."
    reqs = get_db().execute(
        "SELECT * FROM requests ORDER BY votes DESC, created ASC"
    ).fetchall()
    user_votes = set()
    if "user_id" in session:
        rows = get_db().execute(
            "SELECT request_id FROM request_votes WHERE user_id=?", (session["user_id"],)
        ).fetchall()
        user_votes = {r["request_id"] for r in rows}
    return render_template_string(REQUEST_HTML,
        reqs=reqs, user_votes=user_votes,
        error=error, success=success, current_user=current_user())


@app.route("/api/request/<int:req_id>/vote", methods=["POST"])
@login_required
def vote_request(req_id):
    user_id = session["user_id"]
    with get_db() as conn:
        existing = conn.execute(
            "SELECT 1 FROM request_votes WHERE user_id=? AND request_id=?", (user_id, req_id)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM request_votes WHERE user_id=? AND request_id=?", (user_id, req_id))
            conn.execute("UPDATE requests SET votes = votes - 1 WHERE id=?", (req_id,))
            voted = False
        else:
            conn.execute("INSERT INTO request_votes (user_id, request_id) VALUES (?, ?)", (user_id, req_id))
            conn.execute("UPDATE requests SET votes = votes + 1 WHERE id=?", (req_id,))
            voted = True
        row = conn.execute("SELECT votes FROM requests WHERE id=?", (req_id,)).fetchone()
    return jsonify({"voted": voted, "votes": row["votes"]})


@app.route("/indicators")
def indicators_page():
    kind     = request.args.get("kind", "")
    search   = request.args.get("q", "").lower()
    category = request.args.get("cat", "all")

    pine_code   = None
    description = None
    how_to      = None
    name        = None

    # VWAP band options
    band1 = request.args.get("band1", "on") == "on"
    band2 = request.args.get("band2", "on") == "on"
    band3 = request.args.get("band3", "off") == "on"

    # ATR options
    atr_avg = request.args.get("atr_avg", "off") == "on"

    if kind and kind in INDICATORS:
        name, fn, description, cat, how_to = INDICATORS[kind]
        if kind == "vwap":
            pine_code = fn(band1=band1, band2=band2, band3=band3)
        elif kind == "atr":
            pine_code = fn(show_avg=atr_avg)
        else:
            pine_code = fn()

    # Filter indicators for display
    def visible(k, v):
        iname, _, idesc, icat, _ = v
        if category != "all" and icat != category:
            return False
        if search and search not in iname.lower() and search not in idesc.lower():
            return False
        return True

    filtered = {k: v for k, v in INDICATORS.items() if visible(k, v)}

    # Favorites
    is_favorited = False
    if kind and "user_id" in session:
        row = get_db().execute(
            "SELECT 1 FROM favorites WHERE user_id=? AND indicator_key=?",
            (session["user_id"], kind)
        ).fetchone()
        is_favorited = bool(row)

    return render_template_string(INDICATORS_HTML,
        kind=kind, search=search, category=category,
        indicators=filtered, all_indicators=INDICATORS,
        categories=CATEGORIES,
        pine_code=pine_code, name=name,
        description=description, how_to=how_to,
        band1=band1, band2=band2, band3=band3,
        atr_avg=atr_avg,
        is_favorited=is_favorited,
        current_user=current_user())


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


# ── Shared nav macro ─────────────────────────────────────────────────────────

_NAV_LINKS = """
    <a href="/indicators">Indicators</a>
    <a href="/generate">Forecast</a>
    <a href="/news">News</a>
    <a href="/request">Request</a>
    {% if current_user %}
    <a href="/favorites">♥ Favorites</a>
    <a href="/logout" style="color:var(--muted);">{{ current_user }} (logout)</a>
    {% else %}
    <a href="/login">Login</a>
    <a href="/register">Register</a>
    {% endif %}
    <button class="theme-toggle" onclick="toggleTheme()">☀ Light</button>
"""

_THEME_JS = """
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.querySelector('.theme-toggle').textContent = isDark ? '☾ Dark' : '☀ Light';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
const saved = localStorage.getItem('theme');
if (saved) {
  document.documentElement.setAttribute('data-theme', saved);
  document.querySelector('.theme-toggle').textContent = saved === 'light' ? '☾ Dark' : '☀ Light';
}
"""

# ── Auth HTML ─────────────────────────────────────────────────────────────────

AUTH_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ 'Register' if mode == 'register' else 'Login' }} — VolForecast</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace; }
    .center { flex: 1; display: flex; align-items: center; justify-content: center; padding: 40px 24px; }
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 32px; width: 100%; max-width: 380px; }
    h2 { font-size: 1.2rem; margin-bottom: 24px; text-align: center; }
    .form-group { margin-bottom: 16px; }
    label { display: block; color: var(--muted); font-size: 0.8rem; margin-bottom: 6px; }
    input[type=text], input[type=password] {
      width: 100%; background: var(--bg); color: var(--text);
      border: 1px solid var(--border); padding: 9px 14px;
      border-radius: 6px; font-family: monospace; font-size: 0.9rem;
    }
    input:focus { outline: none; border-color: var(--accent); }
    .btn-submit { width: 100%; background: var(--accent); color: #fff; border: none; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.95rem; font-weight: bold; cursor: pointer; margin-top: 8px; }
    .btn-submit:hover { opacity: 0.88; }
    .error { background: #2d1316; border: 1px solid var(--red); border-radius: 6px; padding: 10px 14px; color: var(--red); font-size: 0.85rem; margin-bottom: 16px; }
    .switch { text-align: center; margin-top: 20px; color: var(--muted); font-size: 0.85rem; }
    .switch a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">
    <a href="/indicators">Indicators</a>
    <a href="/login">Login</a>
    <a href="/register">Register</a>
    <button class="theme-toggle" onclick="toggleTheme()">☀ Light</button>
  </div>
</nav>
<div class="center">
  <div class="card">
    <h2>{{ 'Create account' if mode == 'register' else 'Sign in' }}</h2>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST">
      <div class="form-group">
        <label>Username</label>
        <input type="text" name="username" required autofocus>
      </div>
      <div class="form-group">
        <label>Password{% if mode == 'register' %} (min 6 chars){% endif %}</label>
        <input type="password" name="password" required>
      </div>
      <button class="btn-submit" type="submit">{{ 'Create account' if mode == 'register' else 'Sign in' }}</button>
    </form>
    <div class="switch">
      {% if mode == 'register' %}
        Already have an account? <a href="/login">Sign in</a>
      {% else %}
        No account? <a href="/register">Create one</a>
      {% endif %}
    </div>
  </div>
</div>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Favorites HTML ────────────────────────────────────────────────────────────

FAVORITES_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Favorites — VolForecast</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace; }
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .container { max-width: 800px; margin: 0 auto; padding: 28px 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
    .ind-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; text-decoration: none; display: block; transition: border-color 0.15s; }
    .ind-card:hover { border-color: var(--accent); }
    .cat-tag { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.05em; }
    .ind-name { color: var(--text); font-size: 0.9rem; font-weight: bold; margin-bottom: 4px; }
    .ind-desc { color: var(--muted); font-size: 0.75rem; line-height: 1.4; }
    .empty { text-align: center; padding: 60px 24px; color: var(--muted); }
    .empty a { color: var(--accent); text-decoration: none; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>♥ <span>Favorites</span></h1>
  <p style="color:var(--muted); font-size:0.9rem;">Your saved indicators</p>
</div>
<div class="container">
  {% if indicators %}
  <div class="grid">
    {% for key, (iname, _, idesc, icat, _) in indicators.items() %}
    <a class="ind-card" href="/indicators?kind={{ key }}">
      <div class="cat-tag">{{ icat }}</div>
      <div class="ind-name">{{ iname }}</div>
      <div class="ind-desc">{{ idesc[:65] }}…</div>
    </a>
    {% endfor %}
  </div>
  {% else %}
  <div class="empty">
    <p>No favorites yet.</p>
    <p style="margin-top:10px;">Browse <a href="/indicators">indicators</a> and click ♥ to save them.</p>
  </div>
  {% endif %}
</div>
<footer>VolForecast — Free Pine Script indicators · Not financial advice</footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Request HTML ──────────────────────────────────────────────────────────────

REQUEST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Request an Indicator — VolForecast</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace; }
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; }
    .container { max-width: 700px; margin: 0 auto; padding: 28px 24px; }
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 24px; margin-bottom: 24px; }
    .card h2 { font-size: 1rem; margin-bottom: 20px; }
    .form-group { margin-bottom: 16px; }
    label { display: block; color: var(--muted); font-size: 0.8rem; margin-bottom: 6px; }
    input[type=text], textarea {
      width: 100%; background: var(--bg); color: var(--text);
      border: 1px solid var(--border); padding: 9px 14px;
      border-radius: 6px; font-family: monospace; font-size: 0.88rem;
    }
    textarea { height: 100px; resize: vertical; }
    input:focus, textarea:focus { outline: none; border-color: var(--accent); }
    .btn-submit { background: var(--accent); color: #fff; border: none; padding: 9px 24px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; font-weight: bold; cursor: pointer; }
    .btn-submit:hover { opacity: 0.88; }
    .error   { background: #2d1316; border: 1px solid var(--red);   border-radius: 6px; padding: 10px 14px; color: var(--red);   font-size: 0.85rem; margin-bottom: 16px; }
    .success { background: #0d2a14; border: 1px solid var(--green); border-radius: 6px; padding: 10px 14px; color: var(--green); font-size: 0.85rem; margin-bottom: 16px; }
    .req-list { display: flex; flex-direction: column; gap: 12px; }
    .req-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px 20px; display: flex; gap: 16px; align-items: flex-start; }
    .vote-col { display: flex; flex-direction: column; align-items: center; gap: 4px; min-width: 44px; }
    .vote-count { font-size: 1.1rem; font-weight: bold; }
    .vote-btn { background: var(--bg3); border: 1px solid var(--border); color: var(--muted); width: 32px; height: 32px; border-radius: 6px; cursor: pointer; font-size: 1rem; font-family: monospace; }
    .vote-btn:hover { border-color: var(--accent); color: var(--accent); }
    .vote-btn.voted { border-color: var(--accent); color: var(--accent); background: #0d2a4a; }
    .req-body { flex: 1; }
    .req-name { font-weight: bold; font-size: 0.95rem; margin-bottom: 4px; }
    .req-desc { color: var(--muted); font-size: 0.82rem; line-height: 1.5; }
    .req-meta { color: var(--muted); font-size: 0.73rem; margin-top: 6px; }
    .empty { color: var(--muted); text-align: center; padding: 30px; font-size: 0.9rem; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Request an <span>Indicator</span></h1>
  <p>Don't see what you need? Submit a request and upvote others.</p>
</div>
<div class="container">
  <div class="card">
    <h2>Submit a request</h2>
    {% if error   %}<div class="error">{{ error }}</div>{% endif %}
    {% if success %}<div class="success">{{ success }}</div>{% endif %}
    <form method="POST">
      <div class="form-group">
        <label>Indicator name</label>
        <input type="text" name="name" placeholder="e.g. Stochastic RSI" required>
      </div>
      <div class="form-group">
        <label>Description — what should it show?</label>
        <textarea name="description" placeholder="Describe what you want the indicator to do…" required></textarea>
      </div>
      <button class="btn-submit" type="submit">Submit request</button>
    </form>
  </div>

  <div class="card">
    <h2>All requests — most voted first</h2>
    {% if reqs %}
    <div class="req-list">
      {% for r in reqs %}
      <div class="req-card">
        <div class="vote-col">
          <div class="vote-count" id="votes-{{ r['id'] }}">{{ r['votes'] }}</div>
          {% if current_user %}
          <button class="vote-btn {{ 'voted' if r['id'] in user_votes else '' }}"
                  id="vbtn-{{ r['id'] }}" onclick="vote({{ r['id'] }})">▲</button>
          {% else %}
          <a href="/login" class="vote-btn" title="Login to vote">▲</a>
          {% endif %}
        </div>
        <div class="req-body">
          <div class="req-name">{{ r['name'] }}</div>
          <div class="req-desc">{{ r['description'] }}</div>
          <div class="req-meta">by {{ r['author'] }} · {{ r['created'][:10] }}</div>
        </div>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div class="empty">No requests yet — be the first!</div>
    {% endif %}
  </div>
</div>
<footer>VolForecast — Free Pine Script indicators · Not financial advice</footer>
<script>
async function vote(id) {
  const res  = await fetch('/api/request/' + id + '/vote', {method: 'POST'});
  const data = await res.json();
  document.getElementById('votes-' + id).textContent = data.votes;
  const btn = document.getElementById('vbtn-' + id);
  btn.classList.toggle('voted', data.voted);
}
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Indicators HTML ──────────────────────────────────────────────────────────

INDICATORS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Free TradingView Indicators — Pine Script Generator</title>
  <style>
    :root[data-theme="dark"] {
      --bg: #0d1117; --bg2: #161b22; --bg3: #21262d;
      --border: #30363d; --text: #e6edf3; --muted: #8b949e;
      --accent: #58a6ff; --green: #3fb950; --red: #f85149;
      --orange: #f0883e; --purple: #a371f7;
    }
    :root[data-theme="light"] {
      --bg: #ffffff; --bg2: #f6f8fa; --bg3: #eaeef2;
      --border: #d0d7de; --text: #1f2328; --muted: #636c76;
      --accent: #0969da; --green: #1a7f37; --red: #cf222e;
      --orange: #bc4c00; --purple: #8250df;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }

    nav {
      background: var(--bg2); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle {
      background: var(--bg3); border: 1px solid var(--border); color: var(--text);
      padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace;
    }

    .hero { text-align: center; padding: 48px 24px 32px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.8rem; color: var(--text); margin-bottom: 10px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.95rem; max-width: 520px; margin: 0 auto; }

    .container { max-width: 800px; margin: 0 auto; padding: 28px 24px; }

    /* Search + filters */
    .toolbar { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }
    .search-box {
      flex: 1; min-width: 180px; background: var(--bg2); color: var(--text);
      border: 1px solid var(--border); padding: 8px 14px; border-radius: 6px;
      font-family: monospace; font-size: 0.9rem;
    }
    .search-box:focus { outline: none; border-color: var(--accent); }
    .cat-btn {
      background: var(--bg2); color: var(--muted); border: 1px solid var(--border);
      padding: 6px 14px; border-radius: 20px; font-family: monospace; font-size: 0.8rem;
      cursor: pointer; text-decoration: none;
    }
    .cat-btn:hover { border-color: var(--accent); color: var(--text); }
    .cat-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

    /* Indicator cards */
    .indicator-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; margin-bottom: 28px; }
    .ind-card {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
      padding: 14px 16px; cursor: pointer; transition: border-color 0.15s;
      text-decoration: none; display: block;
    }
    .ind-card:hover { border-color: var(--accent); }
    .ind-card.active { border-color: var(--accent); background: #0d2a4a; }
    .ind-card .cat-tag { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.05em; }
    .ind-card .ind-name { color: var(--text); font-size: 0.9rem; font-weight: bold; margin-bottom: 4px; }
    .ind-card .ind-desc { color: var(--muted); font-size: 0.75rem; line-height: 1.4; }
    .no-results { color: var(--muted); font-size: 0.9rem; padding: 20px 0; }

    /* Options + output cards */
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 24px; margin-bottom: 24px; }
    .card.options { margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; border-bottom: none; padding: 14px 24px; }
    .card.output  { border-top-left-radius: 0; border-top-right-radius: 0; }
    label { color: var(--muted); font-size: 0.8rem; cursor: pointer; }

    .pine-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .pine-label span { color: var(--muted); font-size: 0.8rem; }
    .btn-copy {
      background: var(--bg3); color: var(--text); border: 1px solid var(--border);
      padding: 5px 14px; border-radius: 6px; font-family: monospace; font-size: 0.8rem; cursor: pointer;
    }
    .btn-copy:hover { background: var(--border); }
    .btn-copy.copied { border-color: var(--green); color: var(--green); }

    textarea {
      width: 100%; height: 280px; background: var(--bg); color: #c9d1d9;
      border: 1px solid var(--border); border-radius: 6px; padding: 14px;
      font-family: monospace; font-size: 0.78rem; line-height: 1.55; resize: vertical;
    }

    /* How to use */
    .how-to { margin-top: 16px; border-top: 1px solid var(--border); padding-top: 14px; }
    .how-to-title {
      color: var(--muted); font-size: 0.75rem; text-transform: uppercase;
      letter-spacing: 0.05em; margin-bottom: 8px; cursor: pointer; user-select: none;
    }
    .how-to-title:hover { color: var(--text); }
    .how-to-body { color: var(--muted); font-size: 0.85rem; line-height: 1.6; display: none; }
    .how-to-body.open { display: block; }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">""" + _NAV_LINKS + """</div>
</nav>

<div class="hero">
  <h1>Free <span>TradingView</span> Indicators</h1>
  <p>Pick an indicator and copy the Pine Script — works on any TradingView chart, no paid plan needed.</p>
</div>

<div class="container">
  <!-- Search + category filter -->
  <div class="toolbar">
    <input class="search-box" id="search" placeholder="Search indicators…"
           value="{{ search }}" oninput="filterCards()">
    {% for cat_key, cat_label in categories.items() %}
    <a class="cat-btn {{ 'active' if category == cat_key else '' }}"
       href="/indicators?cat={{ cat_key }}{{ ('&kind=' + kind) if kind else '' }}">{{ cat_label }}</a>
    {% endfor %}
  </div>

  <!-- Indicator grid -->
  <div class="indicator-grid" id="grid">
    {% for key, (iname, _, idesc, icat, _) in indicators.items() %}
    <a class="ind-card {{ 'active' if key == kind else '' }}"
       href="/indicators?kind={{ key }}&cat={{ category }}{{ ('&q=' + search) if search else '' }}"
       data-name="{{ iname.lower() }}" data-desc="{{ idesc.lower() }}">
      <div class="cat-tag">{{ icat }}</div>
      <div class="ind-name">{{ iname }}</div>
      <div class="ind-desc">{{ idesc[:65] }}…</div>
    </a>
    {% endfor %}
    {% if not indicators %}
    <div class="no-results">No indicators match "{{ search }}".</div>
    {% endif %}
  </div>

  <!-- VWAP options -->
  {% if kind == 'vwap' %}
  <div class="card options">
    <form method="GET" action="/indicators" style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;">
      <input type="hidden" name="kind" value="vwap">
      <input type="hidden" name="cat" value="{{ category }}">
      <span style="color:var(--muted); font-size:0.85rem;">Bands:</span>
      <label style="color:var(--green);">
        <input type="checkbox" name="band1" value="on" {% if band1 %}checked{% endif %} onchange="this.form.submit()"> ±1 StDev
      </label>
      <label style="color:var(--red);">
        <input type="checkbox" name="band2" value="on" {% if band2 %}checked{% endif %} onchange="this.form.submit()"> ±2 StDev
      </label>
      <label style="color:var(--purple);">
        <input type="checkbox" name="band3" value="on" {% if band3 %}checked{% endif %} onchange="this.form.submit()"> ±3 StDev
      </label>
    </form>
  </div>
  {% endif %}

  <!-- ATR options -->
  {% if kind == 'atr' %}
  <div class="card options">
    <form method="GET" action="/indicators" style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;">
      <input type="hidden" name="kind" value="atr">
      <input type="hidden" name="cat" value="{{ category }}">
      <span style="color:var(--muted); font-size:0.85rem;">Options:</span>
      <label style="color:var(--orange);">
        <input type="checkbox" name="atr_avg" value="on" {% if atr_avg %}checked{% endif %} onchange="this.form.submit()"> Show 20-bar avg (orange)
      </label>
    </form>
  </div>
  {% endif %}

  <!-- Shareable link + favorite -->
  {% if kind %}
  <div style="display:flex; gap:10px; margin-bottom:12px; align-items:center;">
    <button class="btn-copy" onclick="copyLink(this)">🔗 Copy link</button>
    {% if current_user %}
    <button class="btn-copy {{ 'copied' if is_favorited else '' }}"
            id="heart-btn" onclick="toggleFavorite('{{ kind }}')">
      {{ '♥ Saved' if is_favorited else '♡ Save' }}
    </button>
    {% else %}
    <a href="/login" class="btn-copy" style="text-decoration:none;">♡ Save (login)</a>
    {% endif %}
  </div>
  {% endif %}

  <!-- Pine Script output -->
  {% if pine_code %}
  <div class="card {{ 'output' if kind in ['vwap', 'atr'] else '' }}">
    <div class="pine-label">
      <span>Pine Script v5 — works on any chart, no ticker needed</span>
      <button class="btn-copy" onclick="copyPine()">Copy</button>
    </div>
    <textarea id="pine-out" readonly>{{ pine_code }}</textarea>

    {% if how_to %}
    <div class="how-to">
      <div class="how-to-title" onclick="toggleHowTo()">▶ How to use</div>
      <div class="how-to-body" id="howto-body">{{ how_to }}</div>
    </div>
    {% endif %}
  </div>
  {% endif %}
</div>

<footer>VolForecast — Free Pine Script indicators · Not financial advice</footer>

<script>
function copyPine() {
  document.getElementById('pine-out').select();
  document.execCommand('copy');
  const btn = document.querySelector('.pine-label .btn-copy');
  btn.textContent = 'Copied!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
}
function copyLink(btn) {
  navigator.clipboard.writeText(window.location.href);
  btn.textContent = '✓ Copied!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = '🔗 Copy link'; btn.classList.remove('copied'); }, 2000);
}
async function toggleFavorite(key) {
  const res  = await fetch('/api/favorite/' + key, {method: 'POST'});
  const data = await res.json();
  const btn  = document.getElementById('heart-btn');
  btn.textContent = data.favorited ? '♥ Saved' : '♡ Save';
  btn.classList.toggle('copied', data.favorited);
}
function toggleHowTo() {
  const body = document.getElementById('howto-body');
  const title = document.querySelector('.how-to-title');
  body.classList.toggle('open');
  title.textContent = body.classList.contains('open') ? '▼ How to use' : '▶ How to use';
}
function filterCards() {
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.ind-card').forEach(card => {
    card.style.display = (card.dataset.name.includes(q) || card.dataset.desc.includes(q)) ? '' : 'none';
  });
}
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.querySelector('.theme-toggle').textContent = isDark ? '☾ Dark' : '☀ Light';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
const saved = localStorage.getItem('theme');
if (saved) {
  document.documentElement.setAttribute('data-theme', saved);
  document.querySelector('.theme-toggle').textContent = saved === 'light' ? '☾ Dark' : '☀ Light';
}
</script>
</body>
</html>"""


# ── Home HTML ────────────────────────────────────────────────────────────────

HOME_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VolForecast — Free TradingView Pine Script Indicators</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }

    nav {
      background: var(--bg2); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace; }

    /* Hero */
    .hero {
      text-align: center; padding: 80px 24px 60px;
      background: linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
      border-bottom: 1px solid var(--border);
    }
    .hero h1 { font-size: 2.4rem; line-height: 1.25; margin-bottom: 16px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 1rem; max-width: 540px; margin: 0 auto 28px; line-height: 1.7; }
    .hero-btns { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .btn-primary {
      background: var(--accent); color: #fff; border: none;
      padding: 11px 28px; border-radius: 8px; font-family: monospace;
      font-size: 0.95rem; font-weight: bold; text-decoration: none; cursor: pointer;
    }
    .btn-primary:hover { opacity: 0.88; }
    .btn-secondary {
      background: var(--bg2); color: var(--text); border: 1px solid var(--border);
      padding: 11px 28px; border-radius: 8px; font-family: monospace;
      font-size: 0.95rem; text-decoration: none;
    }
    .btn-secondary:hover { border-color: var(--accent); }
    .badge { display: inline-block; margin-bottom: 20px; background: var(--bg3); border: 1px solid var(--border); color: var(--muted); padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; }

    /* Features */
    .section { max-width: 860px; margin: 0 auto; padding: 60px 24px; }
    .section h2 { font-size: 1.4rem; margin-bottom: 32px; text-align: center; }
    .section h2 span { color: var(--accent); }
    .features { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px; }
    .feature {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
      padding: 22px; transition: border-color 0.15s;
    }
    .feature:hover { border-color: var(--accent); }
    .feature-icon { font-size: 1.6rem; margin-bottom: 10px; }
    .feature h3 { font-size: 0.95rem; margin-bottom: 6px; }
    .feature p { color: var(--muted); font-size: 0.82rem; line-height: 1.55; }

    /* Indicators preview */
    .ind-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-top: 28px; }
    .ind-pill {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
      padding: 12px 16px; text-decoration: none; transition: border-color 0.15s;
    }
    .ind-pill:hover { border-color: var(--accent); }
    .ind-pill .cat { color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 3px; }
    .ind-pill .iname { color: var(--text); font-size: 0.88rem; font-weight: bold; }

    /* Steps */
    .steps { display: flex; gap: 0; margin-top: 0; flex-wrap: wrap; }
    .step { flex: 1; min-width: 180px; padding: 20px; border-right: 1px solid var(--border); }
    .step:last-child { border-right: none; }
    .step-num { background: var(--accent); color: #fff; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; font-weight: bold; margin-bottom: 10px; }
    .step h4 { font-size: 0.88rem; margin-bottom: 4px; }
    .step p { color: var(--muted); font-size: 0.8rem; line-height: 1.5; }
    .steps-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

    /* Divider */
    .divider { border: none; border-top: 1px solid var(--border); margin: 0; }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">
    <a href="/indicators">Indicators</a>
    <a href="/generate">Forecast</a>
    <a href="/news">News</a>
    <a href="/dashboard">Live Chart</a>
    <button class="theme-toggle" onclick="toggleTheme()">☀ Light</button>
  </div>
</nav>

<!-- Hero -->
<div class="hero">
  <div class="badge">✓ Works on free TradingView accounts</div>
  <h1>Free <span>Pine Script</span><br>Indicators for TradingView</h1>
  <p>Professional-grade indicators — volume, VWAP, ATR, MA crossovers, Fear & Greed, and more. Copy and paste into any TradingView chart in seconds.</p>
  <div class="hero-btns">
    <a class="btn-primary" href="/indicators">Browse Indicators</a>
    <a class="btn-secondary" href="/generate">LSTM Forecast</a>
  </div>
</div>

<!-- Features -->
<div class="section">
  <h2>Why <span>VolForecast</span>?</h2>
  <div class="features">
    <div class="feature">
      <div class="feature-icon">⚡</div>
      <h3>Instant copy & paste</h3>
      <p>Click an indicator, copy the code, paste into TradingView's Pine Script editor — on your chart in under 30 seconds.</p>
    </div>
    <div class="feature">
      <div class="feature-icon">🆓</div>
      <h3>No paid plan needed</h3>
      <p>All indicators use Pine Script v5 and work on TradingView's free tier. No Pro, no Pro+, no subscription required.</p>
    </div>
    <div class="feature">
      <div class="feature-icon">🎯</div>
      <h3>Any chart, any ticker</h3>
      <p>Indicators automatically adapt to whatever symbol and timeframe you're viewing. No configuration needed.</p>
    </div>
    <div class="feature">
      <div class="feature-icon">🤖</div>
      <h3>LSTM volatility forecast</h3>
      <p>Powered by a deep learning model trained on real market data. Get a 30-minute volatility forecast for any ticker.</p>
    </div>
  </div>
</div>

<hr class="divider">

<!-- Indicators list -->
<div class="section">
  <h2>Available <span>Indicators</span></h2>
  <div class="ind-list">
    <a class="ind-pill" href="/indicators?kind=volume">
      <div class="cat">Volume</div><div class="iname">24h Volume</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=vwap">
      <div class="cat">Volume</div><div class="iname">VWAP + Bands</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=vwap_only">
      <div class="cat">Volume</div><div class="iname">VWAP Only</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=atr">
      <div class="cat">Volatility</div><div class="iname">ATR</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=relvol">
      <div class="cat">Volume</div><div class="iname">Relative Volume</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=macross">
      <div class="cat">Trend</div><div class="iname">MA Cross</div>
    </a>
    <a class="ind-pill" href="/indicators?kind=feargreed">
      <div class="cat">Momentum</div><div class="iname">Fear & Greed</div>
    </a>
  </div>
</div>

<hr class="divider">

<!-- How it works -->
<div class="section">
  <h2>How it <span>works</span></h2>
  <div class="steps-card">
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <h4>Pick an indicator</h4>
        <p>Browse by category or search by name on the Indicators page.</p>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <h4>Copy the code</h4>
        <p>Hit the Copy button — the Pine Script code is in your clipboard.</p>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <h4>Open TradingView</h4>
        <p>Go to any chart → Pine Script Editor at the bottom of the screen.</p>
      </div>
      <div class="step">
        <div class="step-num">4</div>
        <h4>Paste & add to chart</h4>
        <p>Paste the code, click Add to chart. The indicator appears instantly.</p>
      </div>
    </div>
  </div>
</div>

<footer>VolForecast — Free Pine Script indicators · Not financial advice</footer>

<script>
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.querySelector('.theme-toggle').textContent = isDark ? '☾ Dark' : '☀ Light';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
const saved = localStorage.getItem('theme');
if (saved) {
  document.documentElement.setAttribute('data-theme', saved);
  document.querySelector('.theme-toggle').textContent = saved === 'light' ? '☾ Dark' : '☀ Light';
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
    <a href="/news">News</a>
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


# ── News ─────────────────────────────────────────────────────────────────────

NEWS_FEEDS = [
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Seeking Alpha",  "https://seekingalpha.com/market_currents.xml"),
]

def _fetch_news(max_per_feed: int = 8) -> list[dict]:
    import feedparser, re, calendar
    articles = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                ts = 0
                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    t = entry.published_parsed
                    ts = calendar.timegm(t)
                    hour = t.tm_hour % 12 or 12
                    ampm = "AM" if t.tm_hour < 12 else "PM"
                    published = f"{t.tm_mon}/{t.tm_mday} {hour}:{t.tm_min:02d} {ampm}"
                summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", "") or "")[:200]
                articles.append({
                    "source":    source,
                    "title":     entry.get("title", ""),
                    "link":      entry.get("link", "#"),
                    "published": published,
                    "summary":   summary,
                    "_ts":       ts,
                })
        except Exception as exc:
            log.warning("Feed %s failed: %s", source, exc)
    # newest first
    articles.sort(key=lambda a: a["_ts"], reverse=True)
    for a in articles:
        del a["_ts"]
    return articles


@app.route("/news")
def news_page():
    articles = _fetch_news()
    return render_template_string(NEWS_HTML, articles=articles)


@app.route("/api/news")
def api_news():
    return jsonify(_fetch_news())


NEWS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Market News — VolForecast</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav {
      background: var(--bg2); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    .nav-links { display: flex; align-items: center; gap: 20px; }
    .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; }
    .nav-links a:hover { color: var(--text); }
    .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-family: monospace; }

    .hero { padding: 40px 24px 28px; border-bottom: 1px solid var(--border); text-align: center; }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; }

    .container { max-width: 860px; margin: 0 auto; padding: 28px 24px; }

    /* Filter bar */
    .filter-bar { display: flex; gap: 10px; margin-bottom: 24px; flex-wrap: wrap; align-items: center; }
    .filter-bar input {
      flex: 1; min-width: 200px; background: var(--bg2); color: var(--text);
      border: 1px solid var(--border); padding: 8px 14px; border-radius: 6px;
      font-family: monospace; font-size: 0.9rem;
    }
    .filter-bar input:focus { outline: none; border-color: var(--accent); }
    .src-btn {
      background: var(--bg2); color: var(--muted); border: 1px solid var(--border);
      padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; cursor: pointer;
      font-family: monospace;
    }
    .src-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .refresh-btn {
      background: var(--bg2); color: var(--text); border: 1px solid var(--border);
      padding: 6px 14px; border-radius: 6px; font-size: 0.8rem; cursor: pointer;
      font-family: monospace; margin-left: auto;
    }
    .refresh-btn:hover { border-color: var(--accent); }

    /* News list */
    .news-list { display: flex; flex-direction: column; gap: 12px; }
    .news-card {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
      padding: 16px 20px; transition: border-color 0.15s;
    }
    .news-card:hover { border-color: var(--accent); }
    .news-meta { display: flex; gap: 10px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
    .source-tag {
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
      background: var(--bg3); border: 1px solid var(--border);
      padding: 2px 8px; border-radius: 4px; color: var(--muted);
    }
    .news-time { color: var(--muted); font-size: 0.75rem; }
    .news-title { font-size: 0.95rem; font-weight: bold; margin-bottom: 6px; }
    .news-title a { color: var(--text); text-decoration: none; }
    .news-title a:hover { color: var(--accent); }
    .news-summary { color: var(--muted); font-size: 0.8rem; line-height: 1.55; }
    .no-news { color: var(--muted); text-align: center; padding: 40px; }
    #status { color: var(--muted); font-size: 0.78rem; margin-top: 16px; text-align: center; }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">VolForecast</a>
  <div class="nav-links">
    <a href="/indicators">Indicators</a>
    <a href="/generate">Forecast</a>
    <a href="/news">News</a>
    <a href="/dashboard">Live Chart</a>
    <button class="theme-toggle" onclick="toggleTheme()">☀ Light</button>
  </div>
</nav>

<div class="hero">
  <h1><span>Market</span> News</h1>
  <p>Live financial news from Yahoo Finance, Reuters, and MarketWatch. Auto-refreshes every 5 minutes.</p>
</div>

<div class="container">
  <div class="filter-bar">
    <input id="search" placeholder="Search news…" oninput="filterNews()">
    <button class="src-btn active" onclick="filterSource(this, 'all')">All</button>
    <button class="src-btn" onclick="filterSource(this, 'Yahoo Finance')">Yahoo Finance</button>
    <button class="src-btn" onclick="filterSource(this, 'Reuters')">Reuters</button>
    <button class="src-btn" onclick="filterSource(this, 'MarketWatch')">MarketWatch</button>
    <button class="refresh-btn" onclick="loadNews()">↻ Refresh</button>
  </div>

  <div class="news-list" id="news-list">
    {% for a in articles %}
    <div class="news-card" data-source="{{ a.source }}" data-title="{{ a.title.lower() }}" data-summary="{{ a.summary.lower() }}">
      <div class="news-meta">
        <span class="source-tag">{{ a.source }}</span>
        <span class="news-time">{{ a.published }}</span>
      </div>
      <div class="news-title"><a href="{{ a.link }}" target="_blank" rel="noopener">{{ a.title }}</a></div>
      {% if a.summary %}<div class="news-summary">{{ a.summary }}…</div>{% endif %}
    </div>
    {% endfor %}
    {% if not articles %}
    <div class="no-news">No news available right now. Try refreshing.</div>
    {% endif %}
  </div>
  <div id="status"></div>
</div>

<footer>VolForecast — News sourced from public RSS feeds · Not financial advice</footer>

<script>
let activeSource = 'all';

function filterSource(btn, source) {
  activeSource = source;
  document.querySelectorAll('.src-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterNews();
}

function filterNews() {
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.news-card').forEach(card => {
    const srcMatch  = activeSource === 'all' || card.dataset.source === activeSource;
    const textMatch = !q || card.dataset.title.includes(q) || card.dataset.summary.includes(q);
    card.style.display = srcMatch && textMatch ? '' : 'none';
  });
}

async function loadNews() {
  document.getElementById('status').textContent = 'Refreshing…';
  try {
    const res  = await fetch('/api/news');
    const data = await res.json();
    const list = document.getElementById('news-list');
    list.innerHTML = data.length === 0
      ? '<div class="no-news">No news available right now.</div>'
      : data.map(a => `
        <div class="news-card" data-source="${a.source}" data-title="${a.title.toLowerCase()}" data-summary="${(a.summary||'').toLowerCase()}">
          <div class="news-meta">
            <span class="source-tag">${a.source}</span>
            <span class="news-time">${a.published}</span>
          </div>
          <div class="news-title"><a href="${a.link}" target="_blank" rel="noopener">${a.title}</a></div>
          ${a.summary ? `<div class="news-summary">${a.summary}…</div>` : ''}
        </div>`).join('');
    document.getElementById('status').textContent = 'Updated ' + new Date().toLocaleTimeString();
    filterNews();
  } catch(e) {
    document.getElementById('status').textContent = 'Error fetching news: ' + e.message;
  }
}

// Auto-refresh every 5 min
setInterval(loadNews, 5 * 60 * 1000);

// Theme
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.querySelector('.theme-toggle').textContent = isDark ? '☾ Dark' : '☀ Light';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
}
const saved = localStorage.getItem('theme');
if (saved) {
  document.documentElement.setAttribute('data-theme', saved);
  document.querySelector('.theme-toggle').textContent = saved === 'light' ? '☾ Dark' : '☀ Light';
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
    return render_template_string(HOME_HTML)


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
