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
from datetime import date
from functools import wraps

import stripe

from authlib.integrations.flask_client import OAuth
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

# ── Stripe ────────────────────────────────────────────────────────────────────
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_BASIC_PRICE  = "price_1TILpSBcm3kIFrAZiBGS9BVp"
STRIPE_PRO_PRICE    = "price_1TILpcBcm3kIFrAZDpTJvGud"
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PLAN_LIMITS  = {"free": 3, "basic": 10, "pro": -1}  # -1 = unlimited
APP_URL = "https://chartedge.up.railway.app"

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

def _migrate_db():
    """Add columns introduced after initial schema."""
    try:
        with get_db() as conn:
            conn.execute("ALTER TABLE users ADD COLUMN google_id TEXT")
    except Exception:
        pass
    try:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    user_id INTEGER NOT NULL,
                    ticker  TEXT NOT NULL,
                    added   TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (user_id, ticker),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
    except Exception:
        pass
    try:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indicator_ratings (
                    user_id       INTEGER NOT NULL,
                    indicator_key TEXT NOT NULL,
                    vote          INTEGER NOT NULL,
                    PRIMARY KEY (user_id, indicator_key),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
    except Exception:
        pass
    for col in ["plan TEXT DEFAULT 'free'", "stripe_customer_id TEXT", "stripe_subscription_id TEXT"]:
        try:
            with get_db() as conn:
                conn.execute(f"ALTER TABLE users ADD COLUMN {col}")
        except Exception:
            pass
    try:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS copy_log (
                    user_id INTEGER NOT NULL,
                    date    TEXT NOT NULL,
                    count   INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, date)
                )
            """)
    except Exception:
        pass
    try:
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS promo_codes (
                    code       TEXT PRIMARY KEY,
                    plan       TEXT NOT NULL,
                    used       INTEGER DEFAULT 0,
                    used_by    INTEGER,
                    created    TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (used_by) REFERENCES users(id)
                )
            """)
    except Exception:
        pass

init_db()
_migrate_db()

# ── Google OAuth ──────────────────────────────────────────────────────────────

oauth = OAuth(app)
google_oauth = oauth.register(
    name="google",
    client_id=os.environ.get("GOOGLE_CLIENT_ID"),
    client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(f"/login?next={request.path}")
        return f(*args, **kwargs)
    return decorated

def current_user():
    return session.get("username")


def _get_user_plan(user_id):
    if not user_id:
        return "free"
    row = get_db().execute("SELECT plan FROM users WHERE id=?", (user_id,)).fetchone()
    return row["plan"] if row else "free"


def _copies_used_today(user_id):
    today = date.today().isoformat()
    row = get_db().execute(
        "SELECT count FROM copy_log WHERE user_id=? AND date=?", (user_id, today)
    ).fetchone()
    return row["count"] if row else 0


def _increment_copy(user_id):
    today = date.today().isoformat()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO copy_log (user_id, date, count) VALUES (?, ?, 1)
            ON CONFLICT (user_id, date) DO UPDATE SET count = count + 1
        """, (user_id, today))


@app.route("/api/copy", methods=["POST"])
def api_copy():
    user_id = session.get("user_id")
    today = date.today().isoformat()

    if not user_id:
        # Anonymous: session-based tracking
        if session.get("copy_date") != today:
            session["copy_date"] = today
            session["copy_count"] = 0
        used = session.get("copy_count", 0)
        if used >= 3:
            return jsonify({"ok": False, "reason": "limit", "plan": "free"})
        session["copy_count"] = used + 1
        return jsonify({"ok": True, "remaining": 3 - used - 1, "plan": "free"})

    plan = _get_user_plan(user_id)
    if plan == "pro":
        return jsonify({"ok": True, "remaining": -1, "plan": "pro"})

    limit = STRIPE_PLAN_LIMITS.get(plan, 3)
    used  = _copies_used_today(user_id)
    if used >= limit:
        return jsonify({"ok": False, "reason": "limit", "plan": plan})

    _increment_copy(user_id)
    return jsonify({"ok": True, "remaining": limit - used - 1, "plan": plan})


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

def _pine_rsi() -> str:
    return """\
//@version=5
indicator("RSI", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
length    = input.int(14,   "RSI Length",        minval=1)
ob        = input.int(70,   "Overbought Level",  minval=50, maxval=100)
os        = input.int(30,   "Oversold Level",    minval=0,  maxval=50)
show_ma   = input.bool(true,"Show Signal MA")
ma_len    = input.int(9,    "Signal MA Length",  minval=1)

// ── RSI ───────────────────────────────────────────────────────────
rsi    = ta.rsi(close, length)
sig    = ta.ema(rsi, ma_len)

// ── Color ─────────────────────────────────────────────────────────
rsi_color =
     rsi >= ob ? color.new(color.red,    0) :
     rsi <= os ? color.new(color.green,  0) :
                 color.new(color.blue,   0)

// ── Plots ─────────────────────────────────────────────────────────
plot(rsi,              "RSI",    color=rsi_color, linewidth=2)
plot(show_ma ? sig : na,"Signal",color=color.new(color.orange, 0), linewidth=1)

hline(ob,  "Overbought", color=color.new(color.red,   40), linestyle=hline.style_dashed)
hline(50,  "Midline",    color=color.new(color.gray,  60), linestyle=hline.style_dotted)
hline(os,  "Oversold",   color=color.new(color.green, 40), linestyle=hline.style_dashed)

bgcolor(
     rsi >= ob ? color.new(color.red,   92) :
     rsi <= os ? color.new(color.green, 92) : na,
     title="Zone")

// ── Divergence detection ──────────────────────────────────────────
ph = ta.pivothigh(rsi,  5, 5)
pl = ta.pivotlow(rsi,   5, 5)

bull_div = pl and close[5] < ta.valuewhen(ta.pivotlow(close, 5, 5), close[5], 1) and rsi[5] > ta.valuewhen(pl, rsi[5], 1)
bear_div = ph and close[5] > ta.valuewhen(ta.pivothigh(close, 5, 5), close[5], 1) and rsi[5] < ta.valuewhen(ph, rsi[5], 1)

plotshape(bull_div, "Bull Div", shape.labelup,   location.belowbar, color.new(color.green, 20), text="D+", textcolor=color.white, size=size.tiny, offset=-5)
plotshape(bear_div, "Bear Div", shape.labeldown, location.abovebar, color.new(color.red,   20), text="D-", textcolor=color.white, size=size.tiny, offset=-5)

// ── Label ─────────────────────────────────────────────────────────
if barstate.islast
    zone = rsi >= ob ? "Overbought" : rsi <= os ? "Oversold" : "Neutral"
    label.new(bar_index, rsi, zone + "  " + str.tostring(math.round(rsi, 1)),
              style=label.style_label_left, size=size.small,
              color=color.new(rsi >= ob ? color.red : rsi <= os ? color.green : color.gray, 30),
              textcolor=color.white)
"""


def _pine_ema() -> str:
    return """\
//@version=5
indicator("EMA Ribbon", overlay=true, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
show_8   = input.bool(true,  "EMA 8")
show_21  = input.bool(true,  "EMA 21")
show_50  = input.bool(true,  "EMA 50")
show_100 = input.bool(false, "EMA 100")
show_200 = input.bool(true,  "EMA 200")

// ── EMAs ──────────────────────────────────────────────────────────
ema8   = ta.ema(close, 8)
ema21  = ta.ema(close, 21)
ema50  = ta.ema(close, 50)
ema100 = ta.ema(close, 100)
ema200 = ta.ema(close, 200)

// ── Plots ─────────────────────────────────────────────────────────
plot(show_8   ? ema8   : na, "EMA 8",   color=color.new(color.yellow, 0),  linewidth=1)
plot(show_21  ? ema21  : na, "EMA 21",  color=color.new(color.orange, 0),  linewidth=1)
plot(show_50  ? ema50  : na, "EMA 50",  color=color.new(color.aqua,   0),  linewidth=2)
plot(show_100 ? ema100 : na, "EMA 100", color=color.new(color.purple, 0),  linewidth=1)
plot(show_200 ? ema200 : na, "EMA 200", color=color.new(color.red,    0),  linewidth=2)

// ── Trend fill between EMA 50 and 200 ────────────────────────────
bull = ema50 > ema200
fill_col = bull ? color.new(color.green, 88) : color.new(color.red, 88)
p50  = plot(show_50  ? ema50  : na, display=display.none)
p200 = plot(show_200 ? ema200 : na, display=display.none)
fill(p50, p200, color=fill_col, title="Trend Fill")

// ── Label on last bar ─────────────────────────────────────────────
if barstate.islast
    trend = bull ? "Bullish" : "Bearish"
    label.new(bar_index, ema50,
              trend + " — 50/200",
              style=label.style_label_left, size=size.small,
              color=color.new(bull ? color.green : color.red, 30),
              textcolor=color.white)
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

def _pine_macd() -> str:
    return """\
//@version=5
indicator("MACD", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
fast   = input.int(12, "Fast Length",   minval=1)
slow   = input.int(26, "Slow Length",   minval=1)
signal = input.int(9,  "Signal Length", minval=1)

// ── MACD ──────────────────────────────────────────────────────────
[macd_line, signal_line, hist] = ta.macd(close, fast, slow, signal)

// ── Colors ────────────────────────────────────────────────────────
hist_color =
     hist >= 0 and hist >= hist[1] ? color.new(color.green,  0)  :
     hist >= 0 and hist <  hist[1] ? color.new(color.green,  40) :
     hist <  0 and hist <= hist[1] ? color.new(color.red,    0)  :
                                     color.new(color.red,    40)

// ── Plots ─────────────────────────────────────────────────────────
plot(macd_line,   "MACD",     color=color.new(color.blue,   0), linewidth=2)
plot(signal_line, "Signal",   color=color.new(color.orange, 0), linewidth=1)
plot(hist,        "Histogram",color=hist_color, style=plot.style_columns)
hline(0, "Zero", color=color.new(color.gray, 50), linestyle=hline.style_dashed)

// ── Cross labels ──────────────────────────────────────────────────
bullCross = ta.crossover(macd_line,  signal_line)
bearCross = ta.crossunder(macd_line, signal_line)

plotshape(bullCross, "Bull Cross", shape.labelup,   location.belowbar, color.new(color.green, 20), text="▲", textcolor=color.white, size=size.tiny)
plotshape(bearCross, "Bear Cross", shape.labeldown, location.abovebar, color.new(color.red,   20), text="▼", textcolor=color.white, size=size.tiny)
"""


def _pine_supertrend() -> str:
    return """\
//@version=5
indicator("Supertrend", overlay=true, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
atr_len    = input.int(10,  "ATR Length",     minval=1)
multiplier = input.float(3.0, "ATR Multiplier", minval=0.1, step=0.1)

// ── Supertrend ────────────────────────────────────────────────────
[supertrend, direction] = ta.supertrend(multiplier, atr_len)

// ── Plots ─────────────────────────────────────────────────────────
upTrend   = direction < 0
downTrend = direction > 0

plot(upTrend   ? supertrend : na, "Uptrend",   color=color.new(color.green, 0), linewidth=2, style=plot.style_linebr)
plot(downTrend ? supertrend : na, "Downtrend", color=color.new(color.red,   0), linewidth=2, style=plot.style_linebr)

bgcolor(upTrend ? color.new(color.green, 94) : color.new(color.red, 94), title="Trend Background")

// ── Buy / Sell signals ────────────────────────────────────────────
buySignal  = direction[1] > 0 and direction < 0
sellSignal = direction[1] < 0 and direction > 0

plotshape(buySignal,  "Buy",  shape.labelup,   location.belowbar, color.new(color.green, 10), text="BUY",  textcolor=color.white, size=size.small)
plotshape(sellSignal, "Sell", shape.labeldown, location.abovebar, color.new(color.red,   10), text="SELL", textcolor=color.white, size=size.small)
"""


def _pine_ichimoku() -> str:
    return """\
//@version=5
indicator("Ichimoku Cloud", overlay=true, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
tenkan_len  = input.int(9,  "Tenkan (Conversion)")
kijun_len   = input.int(26, "Kijun (Base)")
senkou_len  = input.int(52, "Senkou B Length")
displacement = input.int(26, "Cloud Displacement")

// ── Lines ─────────────────────────────────────────────────────────
tenkan  = (ta.highest(high, tenkan_len)  + ta.lowest(low, tenkan_len))  / 2
kijun   = (ta.highest(high, kijun_len)   + ta.lowest(low, kijun_len))   / 2
senkou_a = (tenkan + kijun) / 2
senkou_b = (ta.highest(high, senkou_len) + ta.lowest(low, senkou_len))  / 2
chikou  = close

// ── Plots ─────────────────────────────────────────────────────────
plot(tenkan,  "Tenkan",  color=color.new(color.blue,   0), linewidth=1)
plot(kijun,   "Kijun",   color=color.new(color.red,    0), linewidth=2)
plot(chikou,  "Chikou",  color=color.new(color.purple, 40), linewidth=1, offset=-displacement)

sa = plot(senkou_a, "Senkou A", color=color.new(color.green, 0), offset=displacement, linewidth=1)
sb = plot(senkou_b, "Senkou B", color=color.new(color.red,   0), offset=displacement, linewidth=1)
fill(sa, sb, color=senkou_a > senkou_b ? color.new(color.green, 80) : color.new(color.red, 80), title="Cloud")
"""


def _pine_obv() -> str:
    return """\
//@version=5
indicator("OBV — On-Balance Volume", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
ma_len   = input.int(20, "Signal MA Length", minval=1)
show_ma  = input.bool(true, "Show Signal MA")

// ── OBV ───────────────────────────────────────────────────────────
obv = ta.obv

// ── Signal line ───────────────────────────────────────────────────
sig = ta.ema(obv, ma_len)

// ── Color — bullish if OBV above signal, bearish if below ─────────
bull = obv >= sig
obv_color = bull ? color.new(color.teal, 0) : color.new(color.red, 0)

// ── Plots ─────────────────────────────────────────────────────────
plot(obv,              "OBV",    color=obv_color, linewidth=2)
plot(show_ma ? sig : na,"Signal",color=color.new(color.orange, 0), linewidth=1)

// ── Fill between OBV and signal ───────────────────────────────────
p1 = plot(obv, display=display.none)
p2 = plot(show_ma ? sig : na, display=display.none)
fill(p1, p2, color=bull ? color.new(color.teal, 85) : color.new(color.red, 85))

// ── Divergence: OBV rising but price falling = bullish div ────────
price_falling = close < close[10]
obv_rising    = obv > obv[10]
price_rising  = close > close[10]
obv_falling   = obv < obv[10]

bull_div = barstate.isconfirmed and price_falling and obv_rising
bear_div = barstate.isconfirmed and price_rising  and obv_falling

plotshape(bull_div, "Bull Div", shape.labelup,   location.belowbar, color.new(color.green, 20), text="D+", textcolor=color.white, size=size.tiny)
plotshape(bear_div, "Bear Div", shape.labeldown, location.abovebar, color.new(color.red,   20), text="D-", textcolor=color.white, size=size.tiny)

// ── Label ─────────────────────────────────────────────────────────
if barstate.islast
    trend = bull ? "Bullish flow" : "Bearish flow"
    label.new(bar_index, obv, trend,
              style=label.style_label_left, size=size.small,
              color=color.new(bull ? color.teal : color.red, 30),
              textcolor=color.white)
"""


def _pine_bb_squeeze() -> str:
    return """\
//@version=5
indicator("Bollinger Band Squeeze", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
bb_len   = input.int(20,  "BB Length",         minval=1)
bb_mult  = input.float(2.0,"BB Multiplier",    minval=0.1, step=0.1)
kc_len   = input.int(20,  "Keltner Length",    minval=1)
kc_mult  = input.float(1.5,"Keltner Multiplier",minval=0.1, step=0.1)

// ── Bollinger Bands ───────────────────────────────────────────────
basis    = ta.sma(close, bb_len)
dev      = bb_mult * ta.stdev(close, bb_len)
bb_upper = basis + dev
bb_lower = basis - dev

// ── Keltner Channels ──────────────────────────────────────────────
kc_mid   = ta.ema(close, kc_len)
kc_range = ta.atr(kc_len) * kc_mult
kc_upper = kc_mid + kc_range
kc_lower = kc_mid - kc_range

// ── Squeeze detection ─────────────────────────────────────────────
squeeze  = bb_upper < kc_upper and bb_lower > kc_lower
no_sqz   = bb_upper > kc_upper and bb_lower < kc_lower

// ── Momentum histogram ────────────────────────────────────────────
val = ta.linreg(close - math.avg(math.avg(ta.highest(high, kc_len), ta.lowest(low, kc_len)), ta.sma(close, kc_len)), kc_len, 0)

hist_color =
     val > 0 and val >= val[1] ? color.new(color.green,  0)  :
     val > 0 and val <  val[1] ? color.new(color.green,  40) :
     val < 0 and val <= val[1] ? color.new(color.red,    0)  :
                                  color.new(color.red,    40)

plot(val, "Momentum", style=plot.style_columns, color=hist_color)
plot(0,   "Zero",     color=color.new(color.gray, 60), linewidth=1)

// ── Squeeze dots ──────────────────────────────────────────────────
sqz_color =
     squeeze ? color.new(color.orange, 0) :
     no_sqz  ? color.new(color.blue,   0) :
               color.new(color.gray,   0)

plot(0, "Squeeze", style=plot.style_circles, linewidth=3, color=sqz_color)

// ── Label ─────────────────────────────────────────────────────────
if barstate.islast
    state = squeeze ? "⚡ IN SQUEEZE" : no_sqz ? "↗ FIRED" : "Normal"
    label.new(bar_index, val,
              state,
              style=label.style_label_left, size=size.small,
              color=color.new(squeeze ? color.orange : no_sqz ? color.blue : color.gray, 30),
              textcolor=color.white)
"""


def _pine_unusual_options() -> str:
    return """\
//@version=5
indicator("Unusual Options Volume", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
thresh    = input.float(2.0, "Spike Threshold (× avg)", minval=1.0, step=0.1,
              tooltip="Bars this many times above average are flagged as unusual")
len       = input.int(20, "Lookback (bars)", minval=5)

// ── Volume & moving average ───────────────────────────────────────
vol     = volume
vol_avg = ta.sma(vol, len)
ratio   = vol / vol_avg

// ── Classify each bar ─────────────────────────────────────────────
is_unusual = ratio >= thresh
is_high    = ratio >= thresh * 0.6 and ratio < thresh

bar_color =
     is_unusual ? color.new(color.red,    0)  :
     is_high    ? color.new(color.orange, 20) :
     close >= open ? color.new(color.teal, 30) :
                     color.new(color.gray, 40)

// ── Plots ─────────────────────────────────────────────────────────
plot(ratio, "Vol Ratio", color=bar_color, style=plot.style_columns, linewidth=2)
plot(1.0,   "Average",   color=color.new(color.white,  50), linewidth=1, style=plot.style_line)
plot(thresh,"Threshold", color=color.new(color.red,    40), linewidth=1, style=plot.style_line)

hline(thresh, "Spike Level", color=color.new(color.red, 50), linestyle=hline.style_dashed)
hline(1.0,    "1× Avg",      color=color.new(color.gray,60), linestyle=hline.style_dotted)

// ── Alert label on unusual bars ───────────────────────────────────
if is_unusual
    label.new(bar_index, ratio + 0.1,
              "⚡ " + str.tostring(math.round(ratio, 1)) + "×",
              style=label.style_label_down, size=size.tiny,
              color=color.new(color.red, 20), textcolor=color.white)

// ── Background flash on unusual bars ─────────────────────────────
bgcolor(is_unusual ? color.new(color.red, 88) : na, title="Unusual Spike")

// ── Last bar summary ──────────────────────────────────────────────
if barstate.islast
    zone = is_unusual ? "⚡ UNUSUAL" : is_high ? "↑ Elevated" : "Normal"
    label.new(bar_index, ratio,
              zone + "  " + str.tostring(math.round(ratio, 2)) + "× avg",
              style=label.style_label_left, size=size.small,
              color=color.new(is_unusual ? color.red : is_high ? color.orange : color.gray, 30),
              textcolor=color.white)
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
    "rsi":       ("RSI",               _pine_rsi,       "Relative Strength Index with overbought/oversold zones, signal MA, and automatic bullish/bearish divergence labels.", "momentum", "Add as a separate panel. Above 70 = overbought (red zone), below 30 = oversold (green zone). The orange signal line is a 9-bar EMA of RSI — crossovers can signal entries. D+ labels mark bullish divergence (price falling but RSI rising), D- marks bearish divergence. Best used with a trend indicator to filter signals."),
    "ema":       ("EMA Ribbon",       _pine_ema,       "8, 21, 50, 100, and 200 EMAs overlaid on the price chart. Green/red fill between the 50 and 200 shows trend direction.", "trend", "Paste onto your price chart. Toggle which EMAs you want in TradingView settings. Green fill between the 50 and 200 EMA = bullish trend, red = bearish. The 8/21 EMAs react fast and are good for short-term entries. The 50/200 are slower and better for trend confirmation. A label on the last bar shows the current trend."),
    "relvol":    ("Relative Volume", _pine_relvol,    "Today's volume vs average. RVOL > 2 = unusually active. Threshold is adjustable in TradingView.",      "volume",     "Add as a separate panel. RVOL of 1.0 = exactly average. Teal bars = above average. Red bars = unusually high (default threshold: 2×). High RVOL on a breakout confirms the move. High RVOL on a reversal signals a strong change. Low RVOL moves are often noise."),
    "macross":   ("MA Cross",        _pine_ma_cross,  "50/200 MA crossover. Labels Golden Cross (bullish) and Death Cross (bearish) directly on the chart.",  "trend",      "Paste onto your price chart. Green background = uptrend (50 above 200). Red background = downtrend. A 'Golden' label appears when the 50 crosses above the 200 — historically a strong bullish signal. 'Death' appears on the cross below. Best used on daily charts."),
    "macd":      ("MACD",            _pine_macd,      "MACD line, signal line, and histogram. ▲/▼ labels mark bullish and bearish crossovers.",                  "trend",      "Add as a separate panel. The blue MACD line crossing above the orange signal line = bullish momentum. Crossing below = bearish. The histogram shows the gap between the two — green bars growing = strengthening uptrend, red bars growing = strengthening downtrend."),
    "supertrend":("Supertrend",      _pine_supertrend,"Dynamic support/resistance line that flips direction. BUY/SELL labels on every trend change.",             "trend",      "Paste onto your price chart. Green line below price = uptrend. Red line above price = downtrend. BUY label appears when trend flips bullish, SELL when it flips bearish. Adjust the ATR multiplier in settings — higher = fewer signals, less noise."),
    "ichimoku":  ("Ichimoku Cloud",  _pine_ichimoku,  "Full Ichimoku system: Tenkan, Kijun, cloud (Senkou A/B), and Chikou span overlaid on price.",             "trend",      "Paste onto your price chart. Green cloud = bullish, red cloud = bearish. Price above the cloud = strong uptrend. Price inside the cloud = consolidation. Price below = downtrend. The Tenkan/Kijun cross is a short-term signal. Best used on daily or 4h charts."),
    "feargreed":     ("Fear & Greed",           _pine_feargreed,     "Composite 0–100 index built from RSI, trend strength, volatility, VIX, and 52-week momentum.",         "momentum", "Add as a separate panel on any chart. Reads 0–100: below 25 = Extreme Fear (often a buying opportunity), above 75 = Extreme Greed (market may be overheated). The label on the last bar shows the current reading and zone. Works on any ticker — uses VIX as one of its inputs."),
    "obv":           ("OBV",               _pine_obv,       "On-Balance Volume — tracks whether volume is flowing into or out of a stock. Divergence labels flag when price and OBV disagree.", "volume", "Add as a separate panel. OBV rising = money flowing in (bullish). OBV falling = money flowing out (bearish). When OBV diverges from price — price falling but OBV rising (D+ label) = smart money buying the dip. Orange signal line helps confirm trend direction."),
    "bbsqueeze":     ("Bollinger Squeeze",      _pine_bb_squeeze,      "Detects when Bollinger Bands contract inside Keltner Channels — a coiling signal before a big move. Orange dot = in squeeze, blue = fired.", "volatility", "Add as a separate panel. Orange dots on the zero line = squeeze is active (market coiling). Blue dots = squeeze just fired (potential breakout). Green histogram = bullish momentum building, red = bearish. The bigger the histogram bars after the squeeze fires, the stronger the move."),
    "unusualopts":   ("Unusual Options Volume", _pine_unusual_options, "Flags bars where volume spikes above a multiple of the 20-bar average. Red = unusual, orange = elevated.", "volume",   "Add as a separate panel. Each bar shows today's volume as a multiple of the average (1.0 = normal). Red bars with a ⚡ label = unusual spike (default 2× threshold). Orange = elevated but not extreme. Adjust the threshold in TradingView settings. High spikes often precede big moves — watch for them before earnings or news."),
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



@app.route("/login/google")
def google_login():
    redirect_uri = "https://chartedge.up.railway.app/auth/google/callback"
    return google_oauth.authorize_redirect(redirect_uri)


@app.route("/auth/google/callback")
def google_callback():
    try:
        token    = google_oauth.authorize_access_token()
        userinfo = token.get("userinfo") or {}
        google_id = userinfo.get("sub")
        if not google_id:
            return redirect("/login?error=google")
        email = userinfo.get("email", "")
        name  = userinfo.get("name", email.split("@")[0] if email else "user")

        db  = get_db()
        row = db.execute("SELECT * FROM users WHERE google_id=?", (google_id,)).fetchone()
        if not row:
            # Build a unique username from the Google display name
            base = name.replace(" ", "_").lower()[:20]
            username = base
            i = 1
            while db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                username = f"{base}{i}"
                i += 1
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, pw_hash, google_id) VALUES (?, '', ?)",
                    (username, google_id),
                )
            row = get_db().execute("SELECT * FROM users WHERE google_id=?", (google_id,)).fetchone()

        session["user_id"]  = row["id"]
        session["username"] = row["username"]
        return redirect(request.args.get("next", "/indicators"))
    except Exception:
        log.exception("Google OAuth error")
        return redirect("/login?error=google")


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

    # Ratings
    user_id = session.get("user_id")
    ratings = _get_ratings(list(filtered.keys()), user_id)

    # Copy limits
    user_plan = _get_user_plan(user_id)
    if user_plan == "pro":
        copies_remaining = -1
    elif user_id:
        limit = STRIPE_PLAN_LIMITS.get(user_plan, 3)
        copies_remaining = max(0, limit - _copies_used_today(user_id))
    else:
        today = date.today().isoformat()
        if session.get("copy_date") != today:
            session["copy_date"] = today
            session["copy_count"] = 0
        copies_remaining = max(0, 3 - session.get("copy_count", 0))

    return render_template_string(INDICATORS_HTML,
        kind=kind, search=search, category=category,
        indicators=filtered, all_indicators=INDICATORS,
        categories=CATEGORIES,
        pine_code=pine_code, name=name,
        description=description, how_to=how_to,
        band1=band1, band2=band2, band3=band3,
        atr_avg=atr_avg,
        is_favorited=is_favorited,
        ratings=ratings,
        user_plan=user_plan,
        copies_remaining=copies_remaining,
        current_user=current_user())


# ── Stripe / Pricing ──────────────────────────────────────────────────────────

@app.route("/subscribe/<plan>")
@login_required
def subscribe(plan):
    price_id = STRIPE_BASIC_PRICE if plan == "basic" else STRIPE_PRO_PRICE
    user_id  = session["user_id"]
    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=APP_URL + "/billing?success=1",
            cancel_url=APP_URL + "/pricing",
            client_reference_id=str(user_id),
        )
        return redirect(checkout.url)
    except Exception as e:
        log.error("Stripe checkout error: %s", e)
        return redirect("/pricing?error=1")


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        return "Invalid signature", 400

    if event["type"] == "checkout.session.completed":
        cs = event["data"]["object"]
        user_id   = cs.get("client_reference_id")
        customer  = cs.get("customer")
        sub_id    = cs.get("subscription")
        if user_id:
            # Determine plan from subscription
            try:
                sub   = stripe.Subscription.retrieve(sub_id)
                price = sub["items"]["data"][0]["price"]["id"]
                plan  = "pro" if price == STRIPE_PRO_PRICE else "basic"
            except Exception:
                plan = "basic"
            with get_db() as conn:
                conn.execute(
                    "UPDATE users SET plan=?, stripe_customer_id=?, stripe_subscription_id=? WHERE id=?",
                    (plan, customer, sub_id, int(user_id))
                )

    elif event["type"] == "customer.subscription.deleted":
        customer = event["data"]["object"].get("customer")
        if customer:
            with get_db() as conn:
                conn.execute(
                    "UPDATE users SET plan='free', stripe_subscription_id=NULL WHERE stripe_customer_id=?",
                    (customer,)
                )

    return "ok", 200


@app.route("/billing")
@login_required
def billing():
    user_id = session["user_id"]
    db  = get_db()
    row = db.execute("SELECT plan, stripe_customer_id FROM users WHERE id=?", (user_id,)).fetchone()
    plan        = row["plan"] if row else "free"
    customer_id = row["stripe_customer_id"] if row else None
    portal_url  = None
    if customer_id:
        try:
            portal = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=APP_URL + "/billing",
            )
            portal_url = portal.url
        except Exception:
            pass
    success = request.args.get("success") == "1"
    return render_template_string(BILLING_HTML,
        plan=plan, portal_url=portal_url, success=success,
        current_user=current_user())


# BILLING_HTML defined below after _META/_NAV_LINKS/_THEME_JS


# PRICING_HTML defined below after _META/_NAV_LINKS/_THEME_JS


@app.route("/pricing")
def pricing():
    error = request.args.get("error") == "1"
    return render_template_string(PRICING_HTML, current_user=current_user(), error=error)


ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

@app.route("/studio", methods=["GET", "POST"])
def admin_codes():
    # Simple password check via query param or session
    if request.method == "POST" and request.form.get("password"):
        if request.form["password"] == ADMIN_PASSWORD:
            session["admin"] = True
        else:
            return render_template_string(ADMIN_LOGIN_HTML, error=True)

    if not session.get("admin"):
        return render_template_string(ADMIN_LOGIN_HTML, error=False)

    # Generate a new code
    new_code = None
    new_plan = None
    if request.method == "POST" and request.form.get("action") == "generate":
        new_plan = request.form.get("plan", "pro")
        new_code = secrets.token_urlsafe(8).upper()
        with get_db() as conn:
            conn.execute("INSERT INTO promo_codes (code, plan) VALUES (?, ?)", (new_code, new_plan))

    # Delete a code
    if request.method == "POST" and request.form.get("action") == "delete":
        with get_db() as conn:
            conn.execute("DELETE FROM promo_codes WHERE code=?", (request.form.get("code"),))

    codes = get_db().execute("SELECT * FROM promo_codes ORDER BY created DESC").fetchall()
    return render_template_string(ADMIN_CODES_HTML, codes=codes, new_code=new_code, new_plan=new_plan, current_user=current_user())


@app.route("/redeem", methods=["GET", "POST"])
@login_required
def redeem():
    message = None
    error = None
    if request.method == "POST":
        code = request.form.get("code", "").strip().upper()
        user_id = session["user_id"]
        row = get_db().execute("SELECT * FROM promo_codes WHERE code=?", (code,)).fetchone()
        if not row:
            error = "Invalid code."
        elif row["used"]:
            error = "This code has already been used."
        else:
            plan = row["plan"]
            with get_db() as conn:
                conn.execute("UPDATE promo_codes SET used=1, used_by=? WHERE code=?", (user_id, code))
                conn.execute("UPDATE users SET plan=? WHERE id=?", (plan, user_id))
            message = f"Success! Your account has been upgraded to {plan.upper()}."
    return render_template_string(REDEEM_HTML, message=message, error=error, current_user=current_user())


ADMIN_LOGIN_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head><meta charset="UTF-8"><title>Admin · ChartEdge</title>
<style>
  :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);display:flex;align-items:center;justify-content:center;min-height:100vh;}
  .box{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:32px;width:320px;}
  h2{margin-bottom:20px;} input{width:100%;padding:10px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;margin-bottom:12px;}
  button{width:100%;padding:10px;background:var(--accent);color:#fff;border:none;border-radius:6px;cursor:pointer;}
  .err{color:#f85149;font-size:.85rem;margin-bottom:10px;}
</style></head>
<body><div class="box">
  <h2>Admin Login</h2>
  {% if error %}<div class="err">Wrong password.</div>{% endif %}
  <form method="POST">
    <input type="password" name="password" placeholder="Admin password" autofocus>
    <button type="submit">Login</button>
  </form>
</div></body></html>"""


ADMIN_CODES_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head><meta charset="UTF-8"><title>Promo Codes · ChartEdge</title>
<style>
  :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);padding:40px 24px;}
  h1{margin-bottom:24px;} h1 span{color:var(--accent);}
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:24px;margin-bottom:20px;max-width:700px;}
  .gen-form{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
  select,button{padding:9px 14px;border-radius:6px;font-size:.9rem;cursor:pointer;}
  select{background:var(--bg);border:1px solid var(--border);color:var(--text);}
  .btn-gen{background:var(--accent);color:#fff;border:none;}
  .new-code{background:#1f2d1f;border:1px solid #3fb950;border-radius:8px;padding:14px 18px;margin-bottom:20px;max-width:700px;}
  .new-code strong{font-size:1.3rem;letter-spacing:2px;color:#3fb950;}
  table{width:100%;max-width:700px;border-collapse:collapse;font-size:.88rem;}
  th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--border);}
  th{color:var(--muted);font-weight:500;}
  .badge{padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600;}
  .badge-pro{background:#1f2d1f;color:#3fb950;} .badge-basic{background:#1f3a5f;color:#58a6ff;}
  .used{opacity:.5;} .btn-del{background:none;border:1px solid #f85149;color:#f85149;padding:4px 10px;border-radius:6px;font-size:.78rem;cursor:pointer;}
</style></head>
<body>
<h1>Promo <span>Codes</span></h1>

{% if new_code %}
<div class="new-code">
  New code generated — share this:<br><br>
  <strong>{{ new_code }}</strong> &nbsp;·&nbsp; {{ new_plan|upper }}
</div>
{% endif %}

<div class="card">
  <form method="POST" class="gen-form">
    <input type="hidden" name="action" value="generate">
    <select name="plan">
      <option value="pro">Pro</option>
      <option value="basic">Basic</option>
    </select>
    <button type="submit" class="btn-gen">Generate Code</button>
  </form>
</div>

<table>
  <tr><th>Code</th><th>Plan</th><th>Status</th><th>Created</th><th></th></tr>
  {% for c in codes %}
  <tr class="{{ 'used' if c.used else '' }}">
    <td><code>{{ c.code }}</code></td>
    <td><span class="badge badge-{{ c.plan }}">{{ c.plan|upper }}</span></td>
    <td>{{ 'Used' if c.used else 'Available' }}</td>
    <td>{{ c.created[:10] }}</td>
    <td>
      {% if not c.used %}
      <form method="POST" style="display:inline">
        <input type="hidden" name="action" value="delete">
        <input type="hidden" name="code" value="{{ c.code }}">
        <button type="submit" class="btn-del">Delete</button>
      </form>
      {% endif %}
    </td>
  </tr>
  {% endfor %}
  {% if not codes %}<tr><td colspan="5" style="color:var(--muted)">No codes yet.</td></tr>{% endif %}
</table>
</body></html>"""


# REDEEM_HTML defined below after _META/_NAV_LINKS/_THEME_JS

# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.route("/watchlist")
@login_required
def watchlist_page():
    user_id = session["user_id"]
    rows = get_db().execute(
        "SELECT ticker FROM watchlist WHERE user_id=? ORDER BY added ASC", (user_id,)
    ).fetchall()
    tickers = [r["ticker"] for r in rows]
    return render_template_string(WATCHLIST_HTML, tickers=tickers, current_user=current_user())


@app.route("/api/watchlist/<ticker>", methods=["POST"])
@login_required
def toggle_watchlist(ticker):
    ticker = ticker.upper()[:10]
    user_id = session["user_id"]
    with get_db() as conn:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE user_id=? AND ticker=?", (user_id, ticker)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM watchlist WHERE user_id=? AND ticker=?", (user_id, ticker))
            return jsonify({"watching": False})
        else:
            conn.execute("INSERT INTO watchlist (user_id, ticker) VALUES (?, ?)", (user_id, ticker))
            return jsonify({"watching": True})


@app.route("/api/watchlist/prices")
@login_required
def watchlist_prices():
    import yfinance as yf
    user_id = session["user_id"]
    rows = get_db().execute(
        "SELECT ticker FROM watchlist WHERE user_id=?", (user_id,)
    ).fetchall()
    tickers = [r["ticker"] for r in rows]
    if not tickers:
        return jsonify([])
    results = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = round(float(info.last_price), 2)
            prev  = round(float(info.previous_close), 2)
            chg   = round(price - prev, 2)
            chg_pct = round((chg / prev) * 100, 2) if prev else 0
            results.append({
                "ticker":   ticker,
                "price":    price,
                "change":   chg,
                "change_pct": chg_pct,
            })
        except Exception:
            results.append({"ticker": ticker, "price": None, "change": None, "change_pct": None})
    return jsonify(results)


# ── Ratings API ──────────────────────────────────────────────────────────────

def _get_ratings(keys: list[str], user_id: int | None) -> dict:
    """Return {key: {"ups": N, "downs": N, "user_vote": 0/1/-1}} for each key."""
    db = get_db()
    result = {k: {"ups": 0, "downs": 0, "user_vote": 0} for k in keys}
    rows = db.execute(
        "SELECT indicator_key, vote FROM indicator_ratings WHERE indicator_key IN ({})".format(
            ",".join("?" * len(keys))), keys
    ).fetchall()
    for r in rows:
        if r["vote"] == 1:
            result[r["indicator_key"]]["ups"] += 1
        else:
            result[r["indicator_key"]]["downs"] += 1
    if user_id:
        urows = db.execute(
            "SELECT indicator_key, vote FROM indicator_ratings WHERE user_id=? AND indicator_key IN ({})".format(
                ",".join("?" * len(keys))), [user_id] + keys
        ).fetchall()
        for r in urows:
            result[r["indicator_key"]]["user_vote"] = r["vote"]
    return result


@app.route("/api/rate/<key>/<vote>", methods=["POST"])
@login_required
def rate_indicator(key, vote):
    if key not in INDICATORS or vote not in ("up", "down"):
        return jsonify({"error": "invalid"}), 400
    user_id  = session["user_id"]
    vote_val = 1 if vote == "up" else -1
    with get_db() as conn:
        existing = conn.execute(
            "SELECT vote FROM indicator_ratings WHERE user_id=? AND indicator_key=?",
            (user_id, key)
        ).fetchone()
        if existing and existing["vote"] == vote_val:
            conn.execute("DELETE FROM indicator_ratings WHERE user_id=? AND indicator_key=?", (user_id, key))
            user_vote = 0
        else:
            conn.execute(
                "INSERT OR REPLACE INTO indicator_ratings (user_id, indicator_key, vote) VALUES (?, ?, ?)",
                (user_id, key, vote_val)
            )
            user_vote = vote_val
    ratings = _get_ratings([key], user_id)
    return jsonify({**ratings[key], "user_vote": user_vote})


# ── Earnings Calendar ─────────────────────────────────────────────────────────

EARNINGS_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","NFLX",
    "AMD","INTC","JPM","BAC","GS","MS","V","MA","WMT","COST",
    "SPY","QQQ","DIS","UBER","COIN","PLTR","SNAP","ROKU",
]

def _fetch_earnings() -> list[dict]:
    import yfinance as yf
    from datetime import date, timedelta
    results = []
    today = date.today()
    cutoff = today + timedelta(days=30)
    for ticker in EARNINGS_TICKERS:
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None:
                continue
            # calendar is a dict with 'Earnings Date' key (list of timestamps)
            dates = cal.get("Earnings Date", [])
            if not dates:
                continue
            earn_date = dates[0]
            if hasattr(earn_date, "date"):
                earn_date = earn_date.date()
            if today <= earn_date <= cutoff:
                eps_est = cal.get("EPS Estimate", [None])
                eps_est = eps_est[0] if isinstance(eps_est, list) and eps_est else eps_est
                rev_est = cal.get("Revenue Estimate", [None])
                rev_est = rev_est[0] if isinstance(rev_est, list) and rev_est else rev_est
                results.append({
                    "ticker":   ticker,
                    "date":     earn_date.strftime("%b %d, %Y"),
                    "date_ord": earn_date.toordinal(),
                    "eps_est":  f"${eps_est:.2f}" if eps_est and eps_est == eps_est else "—",
                    "rev_est":  f"${rev_est/1e9:.1f}B" if rev_est and rev_est == rev_est else "—",
                })
        except Exception:
            continue
    results.sort(key=lambda x: x["date_ord"])
    return results


@app.route("/earnings")
def earnings_page():
    return render_template_string(EARNINGS_HTML, current_user=current_user())


@app.route("/api/earnings")
def earnings_api():
    try:
        earnings = _fetch_earnings()
        return jsonify({"earnings": earnings})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── FAQ ───────────────────────────────────────────────────────────────────────

@app.route("/faq")
def faq_page():
    return render_template_string(FAQ_HTML, current_user=current_user())


@app.route("/privacy")
def privacy_page():
    return render_template_string(PRIVACY_HTML, current_user=current_user())


@app.route("/terms")
def terms_page():
    return render_template_string(TERMS_HTML, current_user=current_user())


# ── Pine Script generator endpoint ───────────────────────────────────────────

@app.route("/generate")
def generate():
    ticker   = request.args.get("ticker", "SPY").upper()
    interval = request.args.get("interval", "5m")

    # If user just landed on the page without submitting, show empty form
    if "ticker" not in request.args:
        return render_template_string(GENERATOR_HTML,
            ticker="SPY", interval="5m",
            pine_code=None, arrow=None, conf=None, error=None,
            current_user=current_user())

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
            error=None, current_user=current_user())

    except Exception as exc:
        log.exception("Generate error")
        return render_template_string(GENERATOR_HTML,
            ticker=ticker, interval=interval,
            pine_code=None, arrow=None, conf=None,
            error=str(exc), current_user=current_user())


# ── Shared head meta ─────────────────────────────────────────────────────────

_GA_ID = os.environ.get("GA_ID", "")
_GA_SCRIPT = ("""
  <script async src="https://www.googletagmanager.com/gtag/js?id=""" + _GA_ID + """"></script>
  <script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','""" + _GA_ID + """');</script>""") if _GA_ID else ""

_META = """
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📈</text></svg>">
  <meta name="description" content="Free TradingView Pine Script indicators — VWAP, RSI, MACD, Supertrend, Bollinger Squeeze and more. Works on any free TradingView account.">
  <meta name="keywords" content="TradingView indicators, Pine Script, free indicators, VWAP, RSI, MACD, Bollinger Bands, Supertrend">
  <meta property="og:title" content="ChartEdge — Free TradingView Indicators">
  <meta property="og:description" content="Free Pine Script indicators for TradingView. No paid plan required.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://chartedge.up.railway.app">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="ChartEdge — Free TradingView Indicators">
  <meta name="twitter:description" content="Free Pine Script indicators for TradingView. No paid plan required.">""" + _GA_SCRIPT + """
  <script>
    (function() {
      var t = localStorage.getItem('theme') || 'dark';
      document.documentElement.setAttribute('data-theme', t);
    })();
  </script>
"""

# ── Shared nav macro ─────────────────────────────────────────────────────────

_NAV_CSS = """
  nav { position: relative; z-index: 1000; overflow: visible !important; }
  .nav-links { display: flex; align-items: center; gap: 4px; }
  .nav-links a { color: var(--muted); text-decoration: none; font-size: 0.9rem; padding: 6px 12px; border-radius: 6px; }
  .nav-links a:hover { color: var(--text); background: var(--bg3); }
  .dropdown { position: relative; }
  .dropdown > .drop-btn {
    background: none; border: 1px solid transparent; color: var(--muted);
    padding: 6px 12px; border-radius: 6px; font-family: monospace;
    font-size: 0.9rem; cursor: pointer; display: flex; align-items: center; gap: 4px;
  }
  .dropdown > .drop-btn:hover { background: var(--bg3); border-color: var(--border); color: var(--text); }
  .dropdown > .drop-btn.open { background: var(--bg3); border-color: var(--border); color: var(--text); }
  .drop-menu {
    display: none; position: absolute; top: calc(100% + 6px); right: 0;
    background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
    min-width: 160px; z-index: 9999; overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }
  .drop-menu.open { display: block; }
  .drop-menu a {
    display: block; padding: 9px 16px; color: var(--muted);
    text-decoration: none; font-size: 0.88rem; border-bottom: 1px solid var(--border);
  }
  .drop-menu a:last-child { border-bottom: none; }
  .drop-menu a:hover { background: var(--bg3); color: var(--text); }
  .theme-toggle { background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-family: monospace; margin-left: 4px; }
  .hamburger { display: none; background: none; border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 1.1rem; }
  @media (max-width: 640px) {
    .hamburger { display: block; }
    .nav-links { display: none; flex-direction: column; align-items: flex-start; gap: 0; position: absolute; top: 100%; left: 0; right: 0; background: var(--bg2); border-bottom: 1px solid var(--border); padding: 8px 16px; z-index: 9998; }
    .nav-links.open { display: flex; }
    .dropdown { width: 100%; }
    .dropdown > .drop-btn { width: 100%; justify-content: space-between; padding: 10px 4px; border: none; border-bottom: 1px solid var(--border); border-radius: 0; }
    .drop-menu { position: static; box-shadow: none; border: none; border-radius: 0; background: var(--bg3); }
    .drop-menu a { padding: 8px 16px; font-size: 0.9rem; }
    .theme-toggle { width: 100%; margin: 8px 0 4px; }
  }
"""

_NAV_LINKS = """
    <div class="dropdown">
      <button class="drop-btn" onclick="toggleDrop(this, event)">Tools ▾</button>
      <div class="drop-menu">
        <a href="/indicators">Indicators</a>
        <a href="/generate">Forecast</a>
        <a href="/dashboard">Live Chart</a>
        <a href="/earnings">Earnings Calendar</a>
        <a href="/news">News</a>
      </div>
    </div>
    <div class="dropdown">
      <button class="drop-btn" onclick="toggleDrop(this, event)">Community ▾</button>
      <div class="drop-menu">
        <a href="/request">Request Indicator</a>
        <a href="/faq">FAQ</a>
        {% if current_user %}<a href="/favorites">♥ My Favorites</a>{% endif %}
        {% if current_user %}<a href="/watchlist">📋 Watchlist</a>{% endif %}
      </div>
    </div>
    <div class="dropdown">
      {% if current_user %}
      <button class="drop-btn" onclick="toggleDrop(this, event)">{{ current_user }} ▾</button>
      <div class="drop-menu">
        <a href="/favorites">♥ Favorites</a>
        <a href="/billing">Billing</a>
        <a href="/redeem">Redeem Code</a>
        <a href="/logout">Logout</a>
      </div>
      {% else %}
      <button class="drop-btn" onclick="toggleDrop(this, event)">Account ▾</button>
      <div class="drop-menu">
        <a href="/login">Login</a>
        <a href="/register">Register</a>
      </div>
      {% endif %}
    </div>
    <a href="/pricing" style="font-size:.85rem;color:var(--accent);padding:6px 10px;text-decoration:none;font-weight:600;">Pricing</a>
    <button class="theme-toggle" onclick="toggleTheme()">☀ Light</button>
"""

_THEME_JS = """
function toggleDrop(btn, event) {
  event.stopPropagation();
  const menu = btn.nextElementSibling;
  const isOpen = menu.classList.contains('open');
  document.querySelectorAll('.drop-menu').forEach(m => m.classList.remove('open'));
  document.querySelectorAll('.drop-btn').forEach(b => b.classList.remove('open'));
  if (!isOpen) { menu.classList.add('open'); btn.classList.add('open'); }
}
function toggleMobileNav(event) {
  event.stopPropagation();
  const nav = document.getElementById('mobile-nav');
  nav.classList.toggle('open');
}
document.addEventListener('click', function() {
  document.querySelectorAll('.drop-menu').forEach(m => m.classList.remove('open'));
  document.querySelectorAll('.drop-btn').forEach(b => b.classList.remove('open'));
  const nav = document.getElementById('mobile-nav');
  if (nav) nav.classList.remove('open');
});
function toggleTheme() {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  const next = isDark ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  document.querySelector('.theme-toggle').textContent = next === 'dark' ? '☀ Light' : '☾ Dark';
  localStorage.setItem('theme', next);
}
(function() {
  const saved = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  document.addEventListener('DOMContentLoaded', function() {
    const btn = document.querySelector('.theme-toggle');
    if (btn) btn.textContent = saved === 'dark' ? '☀ Light' : '☾ Dark';
  });
})();
"""

# ── Pricing / Billing / Redeem HTML ──────────────────────────────────────────

REDEEM_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Redeem Code · ChartEdge</title>
""" + _META + """
  <style>
    :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
    [data-theme="light"]{--bg:#fff;--bg2:#f6f8fa;--text:#1f2328;--muted:#636c76;--border:#d0d7de;--accent:#0969da;}
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,sans-serif;background:var(--bg);color:var(--text);}
    nav{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:100;flex-wrap:wrap;gap:8px;}
    .logo{font-size:1.2rem;font-weight:700;text-decoration:none;}
    .nav-links{display:flex;align-items:center;gap:4px;flex-wrap:wrap;}
    .drop-btn{background:none;border:none;color:var(--text);font-size:.85rem;cursor:pointer;padding:6px 10px;border-radius:6px;}
    .drop-btn:hover,.drop-btn.open{background:var(--border);}
    .dropdown{position:relative;}
    .drop-menu{display:none;position:absolute;top:calc(100% + 6px);left:0;background:var(--bg2);border:1px solid var(--border);border-radius:8px;min-width:180px;z-index:200;padding:4px 0;box-shadow:0 8px 24px rgba(0,0,0,.4);}
    .drop-menu.open{display:block;}
    .drop-menu a{display:block;padding:8px 14px;font-size:.85rem;color:var(--text);text-decoration:none;}
    .drop-menu a:hover{background:var(--border);}
    .theme-toggle{background:none;border:1px solid var(--border);color:var(--muted);font-size:.78rem;cursor:pointer;padding:5px 10px;border-radius:6px;}
    .hamburger{display:none;background:none;border:1px solid var(--border);color:var(--text);font-size:1.1rem;cursor:pointer;padding:4px 10px;border-radius:6px;}
    @media(max-width:640px){.hamburger{display:block;}.nav-links{display:none;flex-direction:column;align-items:flex-start;width:100%;padding:8px 0;}.nav-links.open{display:flex;}.dropdown{width:100%;}.drop-btn{width:100%;text-align:left;}.drop-menu{position:static;box-shadow:none;border:none;padding-left:12px;}.drop-menu.open{display:block;}}
    .page{max-width:480px;margin:80px auto;padding:0 24px;}
    .page h1{font-size:1.8rem;margin-bottom:8px;} .page h1 span{color:var(--accent);}
    .page p{color:var(--muted);margin-bottom:28px;font-size:.92rem;}
    .code-input{width:100%;padding:14px;font-size:1.1rem;letter-spacing:3px;text-transform:uppercase;background:var(--bg2);border:1px solid var(--border);color:var(--text);border-radius:8px;margin-bottom:12px;text-align:center;}
    .code-input:focus{outline:none;border-color:var(--accent);}
    .btn-redeem{width:100%;padding:12px;background:var(--accent);color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;}
    .success{background:#1f2d1f;border:1px solid #3fb950;border-radius:8px;padding:14px 18px;color:#3fb950;margin-bottom:20px;font-size:.9rem;}
    .error{background:#2d1f1f;border:1px solid #f85149;border-radius:8px;padding:14px 18px;color:#f85149;margin-bottom:20px;font-size:.9rem;}
    footer{text-align:center;padding:32px 24px;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:60px;}
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Redeem <span>Code</span></h1>
  <p>Enter a promo code to upgrade your account instantly.</p>
  {% if message %}<div class="success">{{ message }}</div>{% endif %}
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <input class="code-input" type="text" name="code" placeholder="XXXXXXXX" maxlength="20" autofocus>
    <button class="btn-redeem" type="submit">Redeem</button>
  </form>
</div>
<footer>© 2026 ChartEdge · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


BILLING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Billing · ChartEdge</title>
""" + _META + """
  <style>
    :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
    [data-theme="light"]{--bg:#fff;--bg2:#f6f8fa;--text:#1f2328;--muted:#636c76;--border:#d0d7de;--accent:#0969da;}
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);}
    nav{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:100;flex-wrap:wrap;gap:8px;}
    .logo{font-size:1.2rem;font-weight:700;text-decoration:none;}
    .nav-links{display:flex;align-items:center;gap:4px;flex-wrap:wrap;}
    .drop-btn{background:none;border:none;color:var(--text);font-size:.85rem;cursor:pointer;padding:6px 10px;border-radius:6px;}
    .drop-btn:hover,.drop-btn.open{background:var(--border);}
    .dropdown{position:relative;}
    .drop-menu{display:none;position:absolute;top:calc(100% + 6px);left:0;background:var(--bg2);border:1px solid var(--border);border-radius:8px;min-width:180px;z-index:200;padding:4px 0;box-shadow:0 8px 24px rgba(0,0,0,.4);}
    .drop-menu.open{display:block;}
    .drop-menu a{display:block;padding:8px 14px;font-size:.85rem;color:var(--text);text-decoration:none;}
    .drop-menu a:hover{background:var(--border);}
    .theme-toggle{background:none;border:1px solid var(--border);color:var(--muted);font-size:.78rem;cursor:pointer;padding:5px 10px;border-radius:6px;}
    .hamburger{display:none;background:none;border:1px solid var(--border);color:var(--text);font-size:1.1rem;cursor:pointer;padding:4px 10px;border-radius:6px;}
    @media(max-width:640px){.hamburger{display:block;}.nav-links{display:none;flex-direction:column;align-items:flex-start;width:100%;padding:8px 0;}.nav-links.open{display:flex;}.dropdown{width:100%;}.drop-btn{width:100%;text-align:left;}.drop-menu{position:static;box-shadow:none;border:none;padding-left:12px;}.drop-menu.open{display:block;}}
    .page{max-width:600px;margin:0 auto;padding:48px 24px 80px;}
    .page h1{font-size:1.8rem;margin-bottom:24px;} .page h1 span{color:var(--accent);}
    .card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:24px;margin-bottom:16px;}
    .plan-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.8rem;font-weight:600;margin-bottom:16px;}
    .badge-free{background:var(--bg);border:1px solid var(--border);color:var(--muted);}
    .badge-basic{background:#1f3a5f;color:#58a6ff;} .badge-pro{background:#1f2d1f;color:#3fb950;}
    .btn{display:inline-block;padding:10px 20px;border-radius:8px;font-size:.9rem;text-decoration:none;cursor:pointer;border:none;}
    .btn-primary{background:var(--accent);color:#fff;} .btn-secondary{background:var(--bg);border:1px solid var(--border);color:var(--text);}
    .success-box{background:#1f2d1f;border:1px solid #3fb950;border-radius:8px;padding:14px 18px;margin-bottom:20px;color:#3fb950;font-size:.9rem;}
    footer{text-align:center;padding:32px 24px;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);}
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Your <span>Billing</span></h1>
  {% if success %}<div class="success-box">✓ Payment successful! Your plan has been upgraded.</div>{% endif %}
  <div class="card">
    <div class="plan-badge badge-{{ plan }}">{{ plan|upper }}</div>
    <h3 style="margin-bottom:8px;">Current plan</h3>
    <p style="color:var(--muted);font-size:.9rem;margin-bottom:20px;">
      {% if plan == 'free' %}3 copies/day · Free forever
      {% elif plan == 'basic' %}10 copies/day · $9.99/month
      {% else %}Unlimited copies · $15.99/month{% endif %}
    </p>
    {% if portal_url %}<a href="{{ portal_url }}" class="btn btn-secondary">Manage / Cancel Subscription →</a>
    {% elif plan == 'free' %}<a href="/pricing" class="btn btn-primary">Upgrade Plan</a>{% endif %}
  </div>
  <a href="/indicators" style="color:var(--muted);font-size:.85rem;">← Back to indicators</a>
</div>
<footer>© 2026 ChartEdge · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


PRICING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pricing · ChartEdge</title>
""" + _META + """
  <style>
    :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
    [data-theme="light"]{--bg:#fff;--bg2:#f6f8fa;--text:#1f2328;--muted:#636c76;--border:#d0d7de;--accent:#0969da;}
    *{box-sizing:border-box;margin:0;padding:0;}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);}
    nav{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:100;flex-wrap:wrap;gap:8px;}
    .logo{font-size:1.2rem;font-weight:700;text-decoration:none;}
    .nav-links{display:flex;align-items:center;gap:4px;flex-wrap:wrap;}
    .drop-btn{background:none;border:none;color:var(--text);font-size:.85rem;cursor:pointer;padding:6px 10px;border-radius:6px;}
    .drop-btn:hover,.drop-btn.open{background:var(--border);}
    .dropdown{position:relative;}
    .drop-menu{display:none;position:absolute;top:calc(100% + 6px);left:0;background:var(--bg2);border:1px solid var(--border);border-radius:8px;min-width:180px;z-index:200;padding:4px 0;box-shadow:0 8px 24px rgba(0,0,0,.4);}
    .drop-menu.open{display:block;}
    .drop-menu a{display:block;padding:8px 14px;font-size:.85rem;color:var(--text);text-decoration:none;}
    .drop-menu a:hover{background:var(--border);}
    .theme-toggle{background:none;border:1px solid var(--border);color:var(--muted);font-size:.78rem;cursor:pointer;padding:5px 10px;border-radius:6px;}
    .hamburger{display:none;background:none;border:1px solid var(--border);color:var(--text);font-size:1.1rem;cursor:pointer;padding:4px 10px;border-radius:6px;}
    @media(max-width:640px){.hamburger{display:block;}.nav-links{display:none;flex-direction:column;align-items:flex-start;width:100%;padding:8px 0;}.nav-links.open{display:flex;}.dropdown{width:100%;}.drop-btn{width:100%;text-align:left;}.drop-menu{position:static;box-shadow:none;border:none;padding-left:12px;}.drop-menu.open{display:block;}}
    .page{max-width:860px;margin:0 auto;padding:48px 24px 80px;}
    .page h1{font-size:2rem;text-align:center;margin-bottom:8px;} .page h1 span{color:var(--accent);}
    .subtitle{text-align:center;color:var(--muted);margin-bottom:40px;}
    .plans{display:flex;gap:20px;justify-content:center;flex-wrap:wrap;}
    .plan{flex:1;min-width:220px;max-width:260px;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:28px 24px;position:relative;}
    .plan.featured{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent);}
    .plan-badge{position:absolute;top:-12px;left:50%;transform:translateX(-50%);background:var(--accent);color:#fff;font-size:.72rem;font-weight:700;padding:3px 12px;border-radius:20px;white-space:nowrap;}
    .plan-name{font-size:1rem;font-weight:700;margin-bottom:4px;}
    .plan-price{font-size:2rem;font-weight:800;margin:12px 0 4px;} .plan-price span{font-size:1rem;font-weight:400;color:var(--muted);}
    .plan-desc{color:var(--muted);font-size:.83rem;margin-bottom:20px;}
    .plan-features{list-style:none;margin-bottom:24px;}
    .plan-features li{font-size:.85rem;padding:6px 0;border-bottom:1px solid var(--border);color:var(--muted);}
    .plan-features li:last-child{border:none;} .plan-features li::before{content:"✓ ";color:#3fb950;}
    .plan-features li.no::before{content:"✗ ";color:var(--muted);} .plan-features li.no{opacity:.6;}
    .btn-plan{display:block;text-align:center;padding:11px;border-radius:8px;font-size:.9rem;font-weight:600;text-decoration:none;}
    .btn-free{background:var(--bg);border:1px solid var(--border);color:var(--text);}
    .btn-basic{background:var(--bg);border:2px solid var(--accent);color:var(--accent);}
    .btn-pro{background:var(--accent);color:#fff;} .btn-plan:hover{opacity:.88;}
    .error-box{background:#2d1f1f;border:1px solid #f85149;border-radius:8px;padding:12px 16px;margin-bottom:20px;color:#f85149;font-size:.88rem;text-align:center;}
    footer{text-align:center;padding:32px 24px;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);}
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Simple <span>Pricing</span></h1>
  <p class="subtitle">Start free. Upgrade when you need more.</p>
  {% if error %}<div class="error-box">Something went wrong with checkout. Please try again.</div>{% endif %}
  <div class="plans">
    <div class="plan">
      <div class="plan-name">Free</div>
      <div class="plan-price">$0<span>/mo</span></div>
      <div class="plan-desc">Get started with no commitment.</div>
      <ul class="plan-features">
        <li>3 copies per day</li><li>All indicators visible</li>
        <li>Watchlist</li><li>Earnings calendar</li><li class="no">LSTM forecast</li>
      </ul>
      <a href="/indicators" class="btn-plan btn-free">Start Free</a>
    </div>
    <div class="plan">
      <div class="plan-name">Basic</div>
      <div class="plan-price">$9.99<span>/mo</span></div>
      <div class="plan-desc">For active traders who copy often.</div>
      <ul class="plan-features">
        <li>10 copies per day</li><li>All indicators visible</li>
        <li>Watchlist</li><li>Earnings calendar</li><li class="no">LSTM forecast</li>
      </ul>
      {% if current_user %}<a href="/subscribe/basic" class="btn-plan btn-basic">Get Basic</a>
      {% else %}<a href="/login?next=/subscribe/basic" class="btn-plan btn-basic">Get Basic</a>{% endif %}
    </div>
    <div class="plan featured">
      <div class="plan-badge">MOST POPULAR</div>
      <div class="plan-name">Pro</div>
      <div class="plan-price">$15.99<span>/mo</span></div>
      <div class="plan-desc">Unlimited access for power users.</div>
      <ul class="plan-features">
        <li>Unlimited copies</li><li>All indicators visible</li>
        <li>Watchlist</li><li>Earnings calendar</li><li>LSTM forecast</li>
      </ul>
      {% if current_user %}<a href="/subscribe/pro" class="btn-plan btn-pro">Get Pro</a>
      {% else %}<a href="/login?next=/subscribe/pro" class="btn-plan btn-pro">Get Pro</a>{% endif %}
    </div>
  </div>
</div>
<footer>© 2026 ChartEdge · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Auth HTML ─────────────────────────────────────────────────────────────────

AUTH_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>{{ 'Register' if mode == 'register' else 'Login' }} — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """
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
    .btn-google {
      width: 100%; display: flex; align-items: center; justify-content: center; gap: 10px;
      background: #fff; color: #3c4043; border: 1px solid #dadce0;
      padding: 10px; border-radius: 6px; font-family: monospace; font-size: 0.95rem;
      font-weight: bold; cursor: pointer; text-decoration: none; margin-bottom: 16px;
    }
    .btn-google:hover { background: #f8f9fa; border-color: #c6c9cc; }
    .btn-google svg { flex-shrink: 0; }
    .divider { display: flex; align-items: center; gap: 10px; margin: 16px 0; color: var(--muted); font-size: 0.8rem; }
    .divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .error { background: #2d1316; border: 1px solid var(--red); border-radius: 6px; padding: 10px 14px; color: var(--red); font-size: 0.85rem; margin-bottom: 16px; }
    .switch { text-align: center; margin-top: 20px; color: var(--muted); font-size: 0.85rem; }
    .switch a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
  </div>
</nav>
<div class="center">
  <div class="card">
    <h2>{{ 'Create account' if mode == 'register' else 'Sign in' }}</h2>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}

    <a class="btn-google" href="/login/google">
      <svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/><path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/><path fill="#FBBC05" d="M3.964 10.707A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z"/><path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 6.293C4.672 4.169 6.656 3.58 9 3.58z"/></svg>
      {{ 'Sign up with Google' if mode == 'register' else 'Sign in with Google' }}
    </a>

    <div class="divider">or</div>

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
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Favorites — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
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
<footer>© 2026 ChartEdge · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Request HTML ──────────────────────────────────────────────────────────────

REQUEST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Request an Indicator — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
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
<footer>© 2026 ChartEdge · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
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
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
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
""" + _NAV_CSS + """

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
    .copies-badge { font-size: 0.75rem; color: var(--muted); padding: 3px 8px; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; }
    .pine-wrap { position: relative; }
    .pine-blur textarea { filter: blur(4px); pointer-events: none; user-select: none; }
    .pine-overlay { position: absolute; inset: 0; background: rgba(13,17,23,0.82); border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; }
    .overlay-icon { font-size: 1.6rem; }
    .overlay-msg { color: var(--text); font-size: 0.9rem; }
    .overlay-upgrade { color: var(--accent); font-size: 0.8rem; text-decoration: none; }
    .overlay-upgrade:hover { text-decoration: underline; }
    .modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 1000; align-items: center; justify-content: center; }
    .modal-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 32px 28px; max-width: 440px; width: 90%; text-align: center; }
    .modal-box h2 { margin-bottom: 8px; }
    .plan-cta { display: inline-block; padding: 14px 20px; border-radius: 10px; text-decoration: none; font-size: 0.9rem; font-weight: 600; line-height: 1.5; }
    .basic-cta { background: var(--bg); border: 2px solid var(--border); color: var(--text); }
    .pro-cta { background: var(--accent); border: 2px solid var(--accent); color: #fff; }
    .plan-cta small { font-weight: 400; font-size: 0.78rem; opacity: 0.8; }

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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
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

  <!-- Shareable link + favorite + ratings -->
  {% if kind %}
  <div style="display:flex; gap:10px; margin-bottom:12px; align-items:center; flex-wrap:wrap;">
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
      {% if user_plan != 'pro' %}
      <span class="copies-badge" id="copies-badge">
        {% if copies_remaining == 0 %}0 copies left{% else %}{{ copies_remaining }} copies left today{% endif %}
      </span>
      {% endif %}
      <button class="btn-copy" id="copy-btn" onclick="copyPine()">Copy</button>
    </div>
    <div class="pine-wrap{% if user_plan == 'free' %} pine-blur{% endif %}" id="pine-wrap">
      <textarea id="pine-out" readonly>{{ pine_code }}</textarea>
      {% if user_plan == 'free' %}
      <div class="pine-overlay" id="pine-overlay">
        <div class="overlay-icon">🔒</div>
        <div class="overlay-msg"><strong id="overlay-count">{{ copies_remaining }}</strong> free copies left today</div>
        <button class="btn-copy" onclick="copyPine()">Reveal & Copy</button>
        <a href="/pricing" class="overlay-upgrade">Upgrade for more →</a>
      </div>
      {% endif %}
    </div>

    {% if how_to %}
    <div class="how-to">
      <div class="how-to-title" onclick="toggleHowTo()">▶ How to use</div>
      <div class="how-to-body" id="howto-body">{{ how_to }}</div>
    </div>
    {% endif %}
  </div>
  {% endif %}
</div>

<!-- Upgrade modal -->
<div class="modal-bg" id="upgrade-modal" onclick="if(event.target===this)this.style.display='none'">
  <div class="modal-box">
    <h2 style="margin-bottom:8px">Daily limit reached</h2>
    <p style="color:var(--muted);margin-bottom:24px">Upgrade to copy more indicators every day.</p>
    <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
      <a href="/pricing" class="plan-cta basic-cta">Basic — $9.99/mo<br><small>10 copies/day</small></a>
      <a href="/pricing" class="plan-cta pro-cta">Pro — $15.99/mo<br><small>Unlimited copies</small></a>
    </div>
    <button onclick="document.getElementById('upgrade-modal').style.display='none'"
            style="margin-top:16px;background:none;border:none;color:var(--muted);cursor:pointer;font-size:0.85rem;">
      Maybe later
    </button>
  </div>
</div>

<footer>© 2026 ChartEdge · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
async function copyPine() {
  const res  = await fetch('/api/copy', {method: 'POST'});
  const data = await res.json();
  if (!data.ok) {
    document.getElementById('upgrade-modal').style.display = 'flex';
    return;
  }
  // Remove blur/overlay for free users
  const wrap = document.getElementById('pine-wrap');
  const overlay = document.getElementById('pine-overlay');
  if (wrap) wrap.classList.remove('pine-blur');
  if (overlay) overlay.style.display = 'none';
  // Copy text
  const ta = document.getElementById('pine-out');
  ta.select();
  document.execCommand('copy');
  const btn = document.getElementById('copy-btn');
  btn.textContent = 'Copied!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  // Update badge
  const badge = document.getElementById('copies-badge');
  const oc = document.getElementById('overlay-count');
  if (data.remaining >= 0) {
    if (badge) badge.textContent = data.remaining + ' copies left today';
    if (oc) oc.textContent = data.remaining;
  } else {
    if (badge) badge.textContent = '∞';
  }
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
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Home HTML ────────────────────────────────────────────────────────────────

HOME_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>ChartEdge — Free TradingView Pine Script Indicators</title>
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
""" + _NAV_CSS + """

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

    /* Hero preview */
    .hero-preview {
      margin: 40px auto 0; max-width: 680px;
      background: #0d1117; border: 1px solid #30363d; border-radius: 10px;
      overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .preview-bar {
      background: #161b22; border-bottom: 1px solid #30363d;
      padding: 8px 14px; display: flex; align-items: center; gap: 8px;
    }
    .preview-dot { width: 10px; height: 10px; border-radius: 50%; }
    .preview-title { color: #8b949e; font-size: 0.75rem; margin-left: 6px; }
    .preview-body { padding: 16px; }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
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

  <!-- Chart preview mockup -->
  <div class="hero-preview">
    <div class="preview-bar">
      <div class="preview-dot" style="background:#ff5f57"></div>
      <div class="preview-dot" style="background:#febc2e"></div>
      <div class="preview-dot" style="background:#28c840"></div>
      <span class="preview-title">TradingView — SPY 5m · VWAP + Bands</span>
    </div>
    <div class="preview-body">
      <svg viewBox="0 0 640 200" width="100%" xmlns="http://www.w3.org/2000/svg" style="display:block">
        <!-- Grid lines -->
        <line x1="0" y1="50"  x2="640" y2="50"  stroke="#21262d" stroke-width="1"/>
        <line x1="0" y1="100" x2="640" y2="100" stroke="#21262d" stroke-width="1"/>
        <line x1="0" y1="150" x2="640" y2="150" stroke="#21262d" stroke-width="1"/>
        <!-- VWAP upper band -->
        <polyline points="0,70 60,65 120,60 180,55 240,52 300,48 360,50 420,53 480,55 540,52 640,48"
          fill="none" stroke="#58a6ff" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>
        <!-- VWAP line -->
        <polyline points="0,100 60,95 120,92 180,88 240,85 300,82 360,84 420,86 480,88 540,84 640,80"
          fill="none" stroke="#58a6ff" stroke-width="2"/>
        <!-- VWAP lower band -->
        <polyline points="0,130 60,125 120,124 180,121 240,118 300,116 360,118 420,119 480,121 540,116 640,112"
          fill="none" stroke="#58a6ff" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>
        <!-- VWAP band fill -->
        <polygon points="0,70 60,65 120,60 180,55 240,52 300,48 360,50 420,53 480,55 540,52 640,48 640,112 540,116 480,121 420,119 360,118 300,116 240,118 180,121 120,124 60,125 0,130"
          fill="#58a6ff" opacity="0.06"/>
        <!-- Candlesticks -->
        <rect x="10"  y="88" width="8" height="20" fill="#3fb950" rx="1"/>
        <line x1="14" y1="85" x2="14" y2="112" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="30"  y="84" width="8" height="16" fill="#3fb950" rx="1"/>
        <line x1="34" y1="80" x2="34" y2="103" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="50"  y="86" width="8" height="18" fill="#f85149" rx="1"/>
        <line x1="54" y1="82" x2="54" y2="108" stroke="#f85149" stroke-width="1.5"/>
        <rect x="70"  y="80" width="8" height="14" fill="#3fb950" rx="1"/>
        <line x1="74" y1="76" x2="74" y2="97"  stroke="#3fb950" stroke-width="1.5"/>
        <rect x="90"  y="75" width="8" height="18" fill="#3fb950" rx="1"/>
        <line x1="94" y1="71" x2="94" y2="96"  stroke="#3fb950" stroke-width="1.5"/>
        <rect x="110" y="78" width="8" height="16" fill="#f85149" rx="1"/>
        <line x1="114" y1="74" x2="114" y2="98" stroke="#f85149" stroke-width="1.5"/>
        <rect x="130" y="72" width="8" height="14" fill="#3fb950" rx="1"/>
        <line x1="134" y1="68" x2="134" y2="89" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="150" y="68" width="8" height="16" fill="#3fb950" rx="1"/>
        <line x1="154" y1="64" x2="154" y2="88" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="170" y="70" width="8" height="18" fill="#f85149" rx="1"/>
        <line x1="174" y1="66" x2="174" y2="92" stroke="#f85149" stroke-width="1.5"/>
        <rect x="190" y="65" width="8" height="14" fill="#3fb950" rx="1"/>
        <line x1="194" y1="61" x2="194" y2="82" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="210" y="62" width="8" height="16" fill="#3fb950" rx="1"/>
        <line x1="214" y1="58" x2="214" y2="82" stroke="#3fb950" stroke-width="1.5"/>
        <rect x="230" y="66" width="8" height="18" fill="#f85149" rx="1"/>
        <line x1="234" y1="62" x2="234" y2="88" stroke="#f85149" stroke-width="1.5"/>
        <!-- Volume bars at bottom -->
        <rect x="10"  y="178" width="8" height="12" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="30"  y="175" width="8" height="15" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="50"  y="180" width="8" height="10" fill="#f85149" opacity="0.5" rx="1"/>
        <rect x="70"  y="174" width="8" height="16" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="90"  y="172" width="8" height="18" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="110" y="177" width="8" height="13" fill="#f85149" opacity="0.5" rx="1"/>
        <rect x="130" y="173" width="8" height="17" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="150" y="170" width="8" height="20" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="170" y="175" width="8" height="15" fill="#f85149" opacity="0.5" rx="1"/>
        <rect x="190" y="171" width="8" height="19" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="210" y="169" width="8" height="21" fill="#3fb950" opacity="0.5" rx="1"/>
        <rect x="230" y="174" width="8" height="16" fill="#f85149" opacity="0.5" rx="1"/>
        <!-- Label -->
        <rect x="530" y="72" width="100" height="22" rx="4" fill="#1f6feb" opacity="0.85"/>
        <text x="580" y="87" text-anchor="middle" fill="white" font-size="10" font-family="monospace">VWAP 524.18</text>
      </svg>
    </div>
  </div>
</div>

<!-- Features -->
<div class="section">
  <h2>Why <span>ChartEdge</span>?</h2>
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

<footer>© 2026 ChartEdge · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Generator HTML ───────────────────────────────────────────────────────────

GENERATOR_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Volatility Forecast — Free TradingView Pine Script Generator</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }

    /* Nav */
    nav {
      background: var(--bg2); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """

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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
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
  © 2026 ChartEdge · LSTM volatility prediction · Not financial advice
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
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Earnings HTML ────────────────────────────────────────────────────────────

EARNINGS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Earnings Calendar — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; }
    .container { max-width: 900px; margin: 0 auto; padding: 28px 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 8px 14px; border-bottom: 1px solid var(--border); text-align: left; }
    td { padding: 12px 14px; border-bottom: 1px solid var(--border); }
    tr:hover td { background: var(--bg2); }
    .ticker { color: var(--accent); font-weight: bold; font-size: 0.95rem; }
    .empty { text-align: center; padding: 60px 24px; color: var(--muted); }
    .loading { text-align: center; padding: 60px 24px; color: var(--muted); font-size: 0.9rem; }
    .spin { width: 36px; height: 36px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .disclaimer { color: var(--muted); font-size: 0.78rem; margin-top: 20px; padding: 12px; background: var(--bg2); border-radius: 6px; border: 1px solid var(--border); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Earnings <span>Calendar</span></h1>
  <p>Upcoming earnings for major tickers — next 30 days</p>
</div>
<div class="container" id="earnings-container">
  <div class="loading" id="spinner">
    <div class="spin"></div>
    <p style="margin-top:16px">Fetching earnings data…</p>
  </div>
  <div id="earnings-content" style="display:none"></div>
</div>
<footer>© 2026 ChartEdge · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>
async function loadEarnings() {
  try {
    const res  = await fetch('/api/earnings');
    const data = await res.json();
    document.getElementById('spinner').style.display = 'none';
    const el = document.getElementById('earnings-content');
    el.style.display = 'block';
    if (data.error || !data.earnings || data.earnings.length === 0) {
      el.innerHTML = '<div class="empty">No upcoming earnings found for tracked tickers in the next 30 days.</div>';
      return;
    }
    let html = '<table><thead><tr><th>Ticker</th><th>Date</th><th>EPS Estimate</th><th>Revenue Estimate</th></tr></thead><tbody>';
    data.earnings.forEach(e => {
      html += '<tr><td><span class="ticker">' + e.ticker + '</span></td><td>' + e.date + '</td><td>' + e.eps_est + '</td><td>' + e.rev_est + '</td></tr>';
    });
    html += '</tbody></table><div class="disclaimer">⚠ Earnings dates and estimates from Yahoo Finance. Dates may shift — always verify before trading.</div>';
    el.innerHTML = html;
  } catch(e) {
    document.getElementById('spinner').innerHTML = '<p style="color:var(--red)">Failed to load earnings data.</p>';
  }
}
loadEarnings();
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── FAQ HTML ──────────────────────────────────────────────────────────────────

FAQ_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>FAQ — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; }
    .container { max-width: 760px; margin: 0 auto; padding: 36px 24px; }
    .faq-item { border-bottom: 1px solid var(--border); }
    .faq-q {
      width: 100%; text-align: left; background: none; border: none;
      color: var(--text); font-family: monospace; font-size: 0.95rem;
      padding: 18px 0; cursor: pointer; display: flex; justify-content: space-between; align-items: center;
    }
    .faq-q:hover { color: var(--accent); }
    .faq-q .arrow { color: var(--muted); font-size: 0.8rem; transition: transform 0.2s; }
    .faq-q.open .arrow { transform: rotate(90deg); }
    .faq-a { color: var(--muted); font-size: 0.88rem; line-height: 1.7; padding-bottom: 18px; display: none; }
    .faq-a.open { display: block; }
    .faq-a a { color: var(--accent); text-decoration: none; }
    .section-title { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 32px 0 8px; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Frequently Asked <span>Questions</span></h1>
  <p>Everything you need to know about ChartEdge and our indicators</p>
</div>
<div class="container">

  <div class="section-title">Getting Started</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is ChartEdge? <span class="arrow">▶</span></button>
    <div class="faq-a">ChartEdge is a free platform that provides Pine Script indicators for TradingView. All indicators work on free TradingView accounts — no paid plan required.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">How do I add an indicator to TradingView? <span class="arrow">▶</span></button>
    <div class="faq-a">1. Go to the Indicators page and pick an indicator.<br>2. Click <strong>Copy</strong> to copy the Pine Script code.<br>3. In TradingView, open the Pine Script editor (bottom of the chart).<br>4. Paste the code and click <strong>Add to chart</strong>.<br><br><strong>OR</strong><br><br>1. Go to the Indicators page and pick an indicator.<br>2. Click <strong>Source code</strong> on the indicator in TradingView.<br>3. Create a working copy and delete the original code.<br>4. Paste the code you want to use and click <strong>Add to chart</strong>.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Do I need a paid TradingView account? <span class="arrow">▶</span></button>
    <div class="faq-a">No. All indicators on ChartEdge are written in Pine Script v5 and work on free TradingView accounts.</div>
  </div>

  <div class="section-title">Indicators</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is VWAP and when should I use it? <span class="arrow">▶</span></button>
    <div class="faq-a">VWAP (Volume Weighted Average Price) is the average price weighted by volume. It resets each day at market open. Traders use it as a benchmark — price above VWAP = buyers in control, price below = sellers. Best used on intraday charts (1m–1h).</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What does the Fear & Greed indicator measure? <span class="arrow">▶</span></button>
    <div class="faq-a">It combines five factors — RSI, price vs 125-day MA, Bollinger Band width, VIX, and 52-week momentum — into a single 0–100 score. Below 25 = Extreme Fear (potential buying opportunity), above 75 = Extreme Greed (market may be overheated).</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is Unusual Options Volume? <span class="arrow">▶</span></button>
    <div class="faq-a">It shows today's volume as a multiple of the recent average. A ⚡ spike (red bar) means volume is 2× or more above normal, which can signal institutional activity or an upcoming move. High spikes before earnings or news are especially significant.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is the Supertrend indicator? <span class="arrow">▶</span></button>
    <div class="faq-a">Supertrend is a trend-following indicator that uses ATR to draw a dynamic support/resistance line. When it flips from red to green, a BUY signal appears. When it flips from green to red, a SELL signal appears. It works on any timeframe but is most reliable on 1h+ charts.</div>
  </div>

  <div class="section-title">Account</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Do I need an account? <span class="arrow">▶</span></button>
    <div class="faq-a">No — all indicators are free without an account. An account lets you save favorites and vote on indicators.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">How do I request a new indicator? <span class="arrow">▶</span></button>
    <div class="faq-a">Go to <a href="/request">Community → Request Indicator</a> and submit your idea. Other users can upvote requests — the most popular ones get built first.</div>
  </div>

  <div class="section-title">Other</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Is this financial advice? <span class="arrow">▶</span></button>
    <div class="faq-a">No. ChartEdge provides tools for analysis only. Nothing on this site should be considered financial advice. Always do your own research before making trading decisions.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Is ChartEdge free? <span class="arrow">▶</span></button>
    <div class="faq-a">Yes, completely free. No subscriptions, no paywalls.</div>
  </div>

</div>
<footer>© 2026 ChartEdge · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>
function toggleFaq(btn) {
  const answer = btn.nextElementSibling;
  const isOpen = answer.classList.contains('open');
  document.querySelectorAll('.faq-a').forEach(a => a.classList.remove('open'));
  document.querySelectorAll('.faq-q').forEach(b => b.classList.remove('open'));
  if (!isOpen) { answer.classList.add('open'); btn.classList.add('open'); }
}
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Legal HTML ───────────────────────────────────────────────────────────────

_LEGAL_CSS = """
    .container { max-width: 760px; margin: 0 auto; padding: 36px 24px; }
    h2 { font-size: 1.1rem; margin: 28px 0 10px; color: var(--text); }
    p, li { color: var(--muted); font-size: 0.88rem; line-height: 1.8; margin-bottom: 10px; }
    ul { padding-left: 20px; margin-bottom: 10px; }
    a { color: var(--accent); text-decoration: none; }
    .updated { color: var(--muted); font-size: 0.78rem; margin-bottom: 28px; }
"""

PRIVACY_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Privacy Policy — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + _LEGAL_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Privacy <span>Policy</span></h1>
</div>
<div class="container">
  <p class="updated">Last updated: April 3, 2026</p>

  <h2>1. What we collect</h2>
  <p>When you create an account or sign in with Google, we collect:</p>
  <ul>
    <li>Your username (or Google display name)</li>
    <li>Your Google account ID (if using Google sign-in)</li>
    <li>Your saved favorites and indicator votes</li>
  </ul>
  <p>We do <strong>not</strong> collect payment information, precise location, or any data beyond what is necessary to run the service.</p>

  <h2>2. How we use your data</h2>
  <ul>
    <li>To save your favorite indicators and ratings across sessions</li>
    <li>To attribute indicator requests to your username</li>
    <li>We do not sell, share, or use your data for advertising</li>
  </ul>

  <h2>3. Google sign-in</h2>
  <p>If you sign in with Google, we receive your name and Google account ID via OAuth 2.0. We do not receive your password or payment details. You can revoke access at any time at <a href="https://myaccount.google.com/permissions" target="_blank">myaccount.google.com/permissions</a>.</p>

  <h2>4. Cookies &amp; storage</h2>
  <p>We use a session cookie to keep you logged in and localStorage to remember your light/dark theme preference. No third-party tracking cookies are used.</p>

  <h2>5. Data retention</h2>
  <p>Your account data is stored for as long as your account exists. You can request deletion by emailing us or using the delete account option if available.</p>

  <h2>6. Contact</h2>
  <p>Questions? Email: <a href="mailto:ayden.j.folkerts@gmail.com">ayden.j.folkerts@gmail.com</a></p>
</div>
<footer>© 2026 ChartEdge · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


TERMS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Terms of Service — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + _LEGAL_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Terms of <span>Service</span></h1>
</div>
<div class="container">
  <p class="updated">Last updated: April 3, 2026</p>

  <h2>1. Acceptance</h2>
  <p>By using ChartEdge you agree to these terms. If you do not agree, please do not use the site.</p>

  <h2>2. Use of the service</h2>
  <ul>
    <li>ChartEdge is provided free of charge for personal, non-commercial use</li>
    <li>You may copy and use any Pine Script code from this site on TradingView</li>
    <li>You may not scrape, reproduce, or redistribute the site's content or code for commercial purposes without permission</li>
  </ul>

  <h2>3. Not financial advice</h2>
  <p>Nothing on ChartEdge constitutes financial, investment, or trading advice. All indicators are tools for analysis only. You are solely responsible for your own trading decisions. Past performance of any indicator does not guarantee future results.</p>

  <h2>4. Accounts</h2>
  <p>You are responsible for keeping your account credentials secure. We reserve the right to suspend accounts that abuse the service.</p>

  <h2>5. Disclaimer of warranties</h2>
  <p>ChartEdge is provided "as is" without warranty of any kind. We do not guarantee uptime, accuracy of data, or fitness for any particular purpose.</p>

  <h2>6. Limitation of liability</h2>
  <p>ChartEdge and its operators are not liable for any trading losses or damages arising from use of this site.</p>

  <h2>7. Changes</h2>
  <p>We may update these terms at any time. Continued use of the site after changes constitutes acceptance.</p>

  <h2>8. Contact</h2>
  <p>Questions? Email: <a href="mailto:ayden.j.folkerts@gmail.com">ayden.j.folkerts@gmail.com</a></p>
</div>
<footer>© 2026 ChartEdge · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Watchlist HTML ───────────────────────────────────────────────────────────

WATCHLIST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Watchlist — ChartEdge</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; }
    .container { max-width: 700px; margin: 0 auto; padding: 28px 24px; }
    .add-row { display: flex; gap: 10px; margin-bottom: 28px; }
    .add-row input {
      flex: 1; background: var(--bg2); color: var(--text); border: 1px solid var(--border);
      padding: 9px 14px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; text-transform: uppercase;
    }
    .add-row input:focus { outline: none; border-color: var(--accent); }
    .btn-add { background: var(--accent); color: #fff; border: none; padding: 9px 20px; border-radius: 6px; font-family: monospace; font-size: 0.9rem; cursor: pointer; font-weight: bold; }
    .btn-add:hover { opacity: 0.88; }
    .ticker-grid { display: flex; flex-direction: column; gap: 10px; }
    .ticker-card {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
      padding: 14px 18px; display: flex; align-items: center; justify-content: space-between;
    }
    .ticker-left { display: flex; flex-direction: column; gap: 4px; }
    .ticker-sym { font-size: 1.1rem; font-weight: bold; color: var(--accent); }
    .ticker-price { font-size: 1.2rem; color: var(--text); }
    .ticker-chg { font-size: 0.85rem; }
    .up   { color: var(--green); }
    .down { color: var(--red); }
    .btn-remove { background: none; border: 1px solid var(--border); color: var(--muted); padding: 5px 12px; border-radius: 6px; font-family: monospace; font-size: 0.8rem; cursor: pointer; }
    .btn-remove:hover { border-color: var(--red); color: var(--red); }
    .empty { text-align: center; padding: 60px 24px; color: var(--muted); }
    .spin { width: 28px; height: 28px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .last-updated { color: var(--muted); font-size: 0.75rem; margin-bottom: 14px; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>My <span>Watchlist</span></h1>
  <p>Track your favorite tickers with live prices</p>
</div>
<div class="container">
  <div class="add-row">
    <input id="ticker-input" placeholder="Add ticker… (e.g. AAPL)" maxlength="10"
           onkeydown="if(event.key==='Enter') addTicker()">
    <button class="btn-add" onclick="addTicker()">+ Add</button>
  </div>
  <div class="last-updated" id="last-updated"></div>
  <div class="ticker-grid" id="ticker-grid">
    {% if tickers %}
    {% for t in tickers %}
    <div class="ticker-card" id="card-{{ t }}">
      <div class="ticker-left">
        <span class="ticker-sym">{{ t }}</span>
        <span class="ticker-price" id="price-{{ t }}"><span class="spin"></span></span>
        <span class="ticker-chg"  id="chg-{{ t }}"></span>
      </div>
      <button class="btn-remove" onclick="removeTicker('{{ t }}')">✕ Remove</button>
    </div>
    {% endfor %}
    {% else %}
    <div class="empty" id="empty-msg">No tickers yet — add one above.</div>
    {% endif %}
  </div>
</div>
<footer>© 2026 ChartEdge · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>
async function loadPrices() {
  try {
    const res  = await fetch('/api/watchlist/prices');
    const data = await res.json();
    data.forEach(item => {
      const priceEl = document.getElementById('price-' + item.ticker);
      const chgEl   = document.getElementById('chg-'   + item.ticker);
      if (!priceEl) return;
      if (item.price === null) {
        priceEl.textContent = '—';
        return;
      }
      priceEl.textContent = '$' + item.price.toFixed(2);
      const sign = item.change >= 0 ? '+' : '';
      chgEl.textContent = sign + item.change.toFixed(2) + ' (' + sign + item.change_pct.toFixed(2) + '%)';
      chgEl.className = 'ticker-chg ' + (item.change >= 0 ? 'up' : 'down');
    });
    document.getElementById('last-updated').textContent =
      'Updated ' + new Date().toLocaleTimeString();
  } catch(e) {}
}

async function addTicker() {
  const input  = document.getElementById('ticker-input');
  const ticker = input.value.trim().toUpperCase();
  if (!ticker) return;
  const res  = await fetch('/api/watchlist/' + ticker, {method: 'POST'});
  const data = await res.json();
  if (data.watching) {
    input.value = '';
    document.getElementById('empty-msg') && document.getElementById('empty-msg').remove();
    const grid = document.getElementById('ticker-grid');
    const card = document.createElement('div');
    card.className = 'ticker-card';
    card.id = 'card-' + ticker;
    card.innerHTML = '<div class="ticker-left"><span class="ticker-sym">' + ticker + '</span><span class="ticker-price" id="price-' + ticker + '"><span class="spin"></span></span><span class="ticker-chg" id="chg-' + ticker + '"></span></div><button class="btn-remove" onclick="removeTicker(\'' + ticker + '\')">✕ Remove</button>';
    grid.appendChild(card);
    loadPrices();
  }
}

async function removeTicker(ticker) {
  await fetch('/api/watchlist/' + ticker, {method: 'POST'});
  const card = document.getElementById('card-' + ticker);
  if (card) card.remove();
  if (!document.querySelector('.ticker-card')) {
    document.getElementById('ticker-grid').innerHTML = '<div class="empty" id="empty-msg">No tickers yet — add one above.</div>';
  }
}

{% if tickers %}loadPrices();{% endif %}
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── News ─────────────────────────────────────────────────────────────────────

NEWS_FEEDS = [
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Seeking Alpha",  "https://seekingalpha.com/market_currents.xml"),
    ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Benzinga",       "https://www.benzinga.com/feed"),
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
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Market News — ChartEdge</title>
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
""" + _NAV_CSS + """

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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
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

<footer>© 2026 ChartEdge · News sourced from public RSS feeds · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

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
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Dashboard HTML ────────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Live Chart — ChartEdge</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .page { padding: 28px 24px; max-width: 1100px; margin: 0 auto; }
    h1 { color: var(--accent); margin-bottom: 16px; font-size: 1.4rem; }
    .controls { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
    input, select, button {
      background: var(--bg2); color: var(--text); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 6px; font-family: monospace; font-size: 0.9rem;
    }
    button { background: var(--bg3); cursor: pointer; }
    button:hover { background: var(--border); }
    #chart { width: 100%; height: 500px; }
    .info-bar { display: flex; gap: 20px; margin-bottom: 12px; flex-wrap: wrap; }
    .badge {
      padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;
      background: var(--bg3); border: 1px solid var(--border);
    }
    .badge.up   { border-color: var(--green); color: var(--green); }
    .badge.down { border-color: var(--red);   color: var(--red); }
    .badge.vol  { border-color: var(--accent); color: var(--accent); }
    #status { color: var(--muted); font-size: 0.8rem; margin-top: 8px; }
  </style>
</head>
<body>

<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
  </div>
</nav>

<div class="page">
  <h1>Live Chart</h1>

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
""" + _THEME_JS + """
</script>
</div>
</body>
</html>"""


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML, current_user=current_user())


@app.route("/")
def index():
    return render_template_string(HOME_HTML, current_user=current_user())


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
