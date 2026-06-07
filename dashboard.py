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
import sys
from datetime import date
from functools import wraps

import psycopg2
import psycopg2.extras

import resend
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
STRIPE_BASIC_PRICE         = "price_1TIXWYBcm3kIFrAZRBWgcmuJ"
STRIPE_PRO_PRICE           = "price_1TIXWRBcm3kIFrAZg8gflpnz"
STRIPE_BASIC_PRICE_YEARLY  = "price_1TIXYWBcm3kIFrAZ2xsrdS1q"
STRIPE_PRO_PRICE_YEARLY    = "price_1TIXZ5Bcm3kIFrAZe9m5pxzV"
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PLAN_LIMITS    = {"free": 3, "basic": 8, "pro": -1}  # -1 = unlimited
APP_URL               = "https://chartedge.trade"

# ── hCaptcha ──────────────────────────────────────────────────────────────────
HCAPTCHA_SITE_KEY   = "b06c575c-981a-4884-bbc0-5aa8c1d8aaa0"
HCAPTCHA_SECRET_KEY = os.environ.get("HCAPTCHA_SECRET", "")

# ── Finnhub ───────────────────────────────────────────────────────────────────
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

# ── Resend email ──────────────────────────────────────────────────────────────
resend.api_key = os.environ.get("RESEND_API_KEY", "")

def send_welcome_email(to_email: str, username: str, referral_code: str) -> None:
    if not resend.api_key or not to_email:
        return
    try:
        resend.Emails.send({
            "from": "ChartEdge.trade <onboarding@resend.dev>",
            "to": to_email,
            "subject": "Welcome to ChartEdge.trade!",
            "html": f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px;background:#0d1117;color:#e6edf3;border-radius:10px;">
              <h1 style="color:#58a6ff;margin-bottom:8px;">Welcome to ChartEdge.trade!</h1>
              <p style="color:#8b949e;margin-bottom:24px;">Hi {username}, your account is ready. Start copying free Pine Script indicators to your TradingView charts in seconds.</p>
              <a href="{APP_URL}/indicators" style="display:inline-block;background:#58a6ff;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:600;margin-bottom:24px;">Browse Indicators →</a>
              <hr style="border:none;border-top:1px solid #30363d;margin:24px 0;">
              <p style="color:#8b949e;font-size:.88rem;">Your referral code: <strong style="color:#e6edf3;letter-spacing:2px;">{referral_code}</strong><br>Share it with friends — they get 7 days of Pro free when they sign up.</p>
              <p style="color:#636c76;font-size:.78rem;margin-top:24px;">© 2026 ChartEdge.trade · Not financial advice</p>
            </div>
            """,
        })
    except Exception as e:
        log.warning("Failed to send welcome email: %s", e)


def send_password_reset_email(to_email: str, username: str, reset_url: str) -> None:
    if not resend.api_key or not to_email:
        return
    try:
        resend.Emails.send({
            "from": "ChartEdge.trade <onboarding@resend.dev>",
            "to": to_email,
            "subject": "Reset your ChartEdge.trade password",
            "html": f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;padding:32px;background:#0d1117;color:#e6edf3;border-radius:10px;">
              <h1 style="color:#58a6ff;margin-bottom:8px;">Reset your password</h1>
              <p style="color:#8b949e;margin-bottom:24px;">Hi {username}, click the link below to set a new password. This link expires in 1 hour and can only be used once.</p>
              <a href="{reset_url}" class="notrack" data-no-track="true" style="display:inline-block;background:#58a6ff;color:#fff;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:.95rem;margin-bottom:24px;">Reset Password</a>
              <p style="color:#8b949e;font-size:.82rem;margin-top:16px;">If you didn't request this, you can safely ignore this email — your password won't change.</p>
              <p style="color:#636c76;font-size:.78rem;margin-top:24px;">© 2026 ChartEdge.trade · Not financial advice</p>
            </div>
            """,
        })
    except Exception as e:
        log.warning("Failed to send password reset email: %s", e)

# ── Database ──────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn

def _q(sql, params=()):
    """Execute and return all rows."""
    sql = sql.replace("?", "%s")
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        result = cur.fetchall()
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _one(sql, params=()):
    """Execute and return one row."""
    rows = _q(sql, params)
    return rows[0] if rows else None

def _scalar(sql, params=()):
    """Execute and return first value of first row (for COUNT etc)."""
    row = _one(sql, params)
    return list(row.values())[0] if row else 0

def _run(sql, params=()):
    """Execute a write statement."""
    sql = sql.replace("?", "%s")
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

from contextlib import contextmanager

@contextmanager
def _tx():
    """Context manager for multi-statement transactions."""
    conn = get_db()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                pw_hash  TEXT NOT NULL,
                created  TIMESTAMP DEFAULT NOW(),
                google_id TEXT,
                plan     TEXT DEFAULT 'free',
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                email    TEXT,
                referral_code TEXT,
                referred_by TEXT,
                plan_expires  TIMESTAMP,
                trial_used    INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                user_id       INTEGER NOT NULL,
                indicator_key TEXT NOT NULL,
                PRIMARY KEY (user_id, indicator_key)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id          SERIAL PRIMARY KEY,
                author      TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL,
                votes       INTEGER DEFAULT 0,
                created     TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS request_votes (
                user_id    INTEGER NOT NULL,
                request_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, request_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS indicator_ratings (
                user_id       INTEGER NOT NULL,
                indicator_key TEXT NOT NULL,
                vote          INTEGER NOT NULL,
                PRIMARY KEY (user_id, indicator_key)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS copy_log (
                user_id INTEGER NOT NULL,
                date    TEXT NOT NULL,
                count   INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS promo_codes (
                code    TEXT PRIMARY KEY,
                plan    TEXT NOT NULL,
                used    INTEGER DEFAULT 0,
                used_by INTEGER,
                created TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS priority_requests (
                id          SERIAL PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                username    TEXT NOT NULL,
                title       TEXT NOT NULL,
                description TEXT NOT NULL,
                status      TEXT DEFAULT 'pending',
                created     TIMESTAMP DEFAULT NOW(),
                updated     TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                user_id INTEGER NOT NULL,
                symbol  TEXT NOT NULL,
                added   TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (user_id, symbol)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS copy_history (
                id        SERIAL PRIMARY KEY,
                user_id   INTEGER NOT NULL,
                indicator TEXT NOT NULL,
                ts        TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used       INTEGER DEFAULT 0,
                created    TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _migrate_pg():
    for col in [
        "plan_expires TIMESTAMP",
        "trial_used INTEGER DEFAULT 0",
        "profile_pic TEXT",
        "theme TEXT DEFAULT 'dark'",
        "default_ticker TEXT DEFAULT 'AAPL'",
        "copy_toast INTEGER DEFAULT 1",
    ]:
        try:
            _run(f"ALTER TABLE users ADD COLUMN {col}")
        except Exception:
            pass

init_db()
_migrate_pg()

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


@app.context_processor
def inject_nav_plan():
    uid = session.get("user_id")
    plan = _get_user_plan(uid) if uid else "free"
    email_missing = False
    if uid:
        row = _one("SELECT email FROM users WHERE id=%s", (uid,))
        if row and not (row["email"] or "").strip():
            email_missing = True
    return {"nav_plan": plan, "show_email_banner": email_missing}


# Routes that don't require login
_PUBLIC_ROUTES = {
    "index", "login", "register", "google_login", "google_callback",
    "pricing", "privacy", "terms", "stripe_webhook", "admin_codes",
    "forgot_password", "reset_password",
}

@app.before_request
def require_login():
    if request.endpoint in _PUBLIC_ROUTES:
        return
    if request.path.startswith("/static"):
        return
    if "user_id" not in session:
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_logged_in"}), 401
        return redirect(f"/login?next={request.path}")


def _get_user_plan(user_id):
    if not user_id:
        return "free"
    from datetime import datetime, timezone
    row = _one("SELECT plan, plan_expires FROM users WHERE id=%s", (user_id,))
    if not row:
        return "free"
    plan = row["plan"] or "free"
    expires = row["plan_expires"]
    if expires and plan != "free":
        now = datetime.now(timezone.utc)
        exp = expires.replace(tzinfo=timezone.utc) if expires.tzinfo is None else expires
        if now > exp:
            _run("UPDATE users SET plan='free', plan_expires=NULL WHERE id=%s", (user_id,))
            return "free"
    return plan


def _copies_used_today(user_id):
    today = date.today().isoformat()
    row = _one("SELECT count FROM copy_log WHERE user_id=%s AND date=%s", (user_id, today))
    return row["count"] if row else 0


def _increment_copy(user_id):
    today = date.today().isoformat()
    _run("""
        INSERT INTO copy_log (user_id, date, count) VALUES (%s, %s, 1)
        ON CONFLICT (user_id, date) DO UPDATE SET count = copy_log.count + 1
    """, (user_id, today))


@app.route("/api/copy", methods=["POST"])
def api_copy():
    user_id   = session.get("user_id")
    today     = date.today().isoformat()
    indicator = request.json.get("indicator", "") if request.is_json else request.form.get("indicator", "")

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
        _increment_copy(user_id)
        if indicator:
            _run("INSERT INTO copy_history (user_id, indicator) VALUES (%s, %s)", (user_id, indicator))
        return jsonify({"ok": True, "remaining": -1, "plan": "pro"})

    limit = STRIPE_PLAN_LIMITS.get(plan, 3)
    used  = _copies_used_today(user_id)
    if used >= limit:
        return jsonify({"ok": False, "reason": "limit", "plan": plan})

    _increment_copy(user_id)
    if indicator:
        _run("INSERT INTO copy_history (user_id, indicator) VALUES (%s, %s)", (user_id, indicator))
    return jsonify({"ok": True, "remaining": limit - used - 1, "plan": plan})


# ── Indicator API ─────────────────────────────────────────────────────────────

@app.route("/api/indicator")
def api_indicator():
    kind = request.args.get("kind", "")
    if not kind or kind not in INDICATORS:
        return jsonify({"ok": False, "error": "unknown"})

    band1   = request.args.get("band1",   "on")  == "on"
    band2   = request.args.get("band2",   "on")  == "on"
    band3   = request.args.get("band3",   "off") == "on"
    atr_avg = request.args.get("atr_avg", "off") == "on"

    name, fn, description, cat, how_to, tag = INDICATORS[kind]
    if kind == "vwap":
        pine_code = fn(band1=band1, band2=band2, band3=band3)
    elif kind == "atr":
        pine_code = fn(show_avg=atr_avg)
    else:
        pine_code = fn()
    pine_code = _inject_expiry(pine_code, session.get("username", "user"))

    user_id   = session.get("user_id")
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

    is_favorited = False
    if user_id and kind:
        row = _one("SELECT 1 FROM favorites WHERE user_id=%s AND indicator_key=%s",
                   (user_id, kind))
        is_favorited = bool(row)

    return jsonify({
        "ok":               True,
        "kind":             kind,
        "name":             name,
        "pine_code":        pine_code,
        "description":      description or "",
        "how_to":           how_to or "",
        "copies_remaining": copies_remaining,
        "user_plan":        user_plan,
        "is_favorited":     is_favorited,
        "has_vwap_options": kind == "vwap",
        "has_atr_options":  kind == "atr",
    })


# ── Forecast API ──────────────────────────────────────────────────────────────

@app.route("/api/forecast")
def api_forecast():
    import numpy as np
    ticker   = request.args.get("ticker", "SPY").upper()
    interval = request.args.get("interval", "5m")

    try:
        # Download price data via yfinance
        import yfinance as yf
        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d"}
        period = period_map.get(interval, "60d")
        raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if raw is None or len(raw) < 30:
            return jsonify({"error": "Not enough data"}), 400

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, __import__('pandas').MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close = raw["Close"].dropna()
        if len(close) < 30:
            return jsonify({"error": "Not enough data"}), 400

        # Compute realized volatility: rolling 20-bar std of log returns
        log_ret = np.log(close / close.shift(1)).dropna()
        rv = log_ret.rolling(window=20).std().dropna()

        hist_index = rv.index
        hist_rv    = rv.values

        # Simple forecast: mean-revert toward recent average over next 6 bars
        FORECAST_STEPS = 6
        recent_mean = float(np.mean(hist_rv[-20:]))
        last_rv     = float(hist_rv[-1])
        mean_fcast  = np.array([last_rv + (recent_mean - last_rv) * (i + 1) / FORECAST_STEPS
                                for i in range(FORECAST_STEPS)])
        std_fcast   = float(np.std(hist_rv[-20:])) * 0.5
        lower = mean_fcast - std_fcast
        upper = mean_fcast + std_fcast

        interval_ms_map = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
        bar_ms    = interval_ms_map.get(interval, 300_000)
        last_ts   = int(hist_index[-1].timestamp() * 1000)
        future_ts = [last_ts + bar_ms * (i + 1) for i in range(FORECAST_STEPS)]

        # Direction: compare last RV to recent mean
        direction_up   = bool(last_rv < recent_mean)
        direction_conf = float(min(0.99, 0.5 + abs(last_rv - recent_mean) / (recent_mean + 1e-10) * 0.5))

        return jsonify({
            "ticker":   ticker,
            "interval": interval,
            "hist_ts":  [int(ts.timestamp() * 1000) for ts in hist_index],
            "hist_rv":  [round(float(v), 8) for v in hist_rv],
            "future_ts":    future_ts,
            "future_mean":  [round(float(v), 8) for v in mean_fcast],
            "future_lower": [round(float(v), 8) for v in lower],
            "future_upper": [round(float(v), 8) for v in upper],
            "low_thresh":    round(float(np.percentile(hist_rv, 25)), 8),
            "medium_thresh": round(float(np.percentile(hist_rv, 50)), 8),
            "high_thresh":   round(float(np.percentile(hist_rv, 75)), 8),
            "direction_up":   direction_up,
            "direction_conf": direction_conf,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


# ── Pine Script expiry injection ─────────────────────────────────────────────

def _inject_expiry(script: str, username: str) -> str:
    """
    Embeds daily expiry checks throughout a Pine Script so it breaks
    after the generation date, forcing users to return to ChartEdge.trade.
    """
    from datetime import date as _date
    today = _date.today()
    d, m, y = today.day, today.month, today.year
    exp_int = y * 10000 + m * 100 + d        # e.g. 20260408
    site    = "chartedge.trade"
    msg     = f"Expired \u00b7 Regenerate at {site}"
    user_lbl = username or "user"

    # Three different expiry expressions — same logic, different Pine syntax
    # A: integer date comparison
    chk_a = (
        f"var int _cedge_d = year * 10000 + month * 100 + dayofmonth\n"
        f"var bool _cedge_ok = _cedge_d <= {exp_int}\n"
        f'if not _cedge_ok\n'
        f'    runtime.error("{msg}")\n'
    )
    # B: individual field comparison (harder to spot as the same check)
    chk_b = (
        f"bool _cvalid = year < {y} or (year == {y} and month < {m}) or "
        f"(year == {y} and month == {m} and dayofmonth <= {d})\n"
        f'if not _cvalid\n'
        f'    runtime.error("{msg}")\n'
    )
    # C: obfuscated — split across expressions
    chk_c = (
        f"int _cyr = year\n"
        f"int _cmo = month\n"
        f"int _cdy = dayofmonth\n"
        f"bool _cexp = (_cyr * 372 + _cmo * 31 + _cdy) > "
        f"{y * 372 + m * 31 + d}\n"
        f'if _cexp\n'
        f'    runtime.error("{msg}")\n'
    )

    lines = script.split("\n")
    out   = []
    indicator_done = False
    chk_b_inserted = False
    chk_c_inserted = False
    n = len(lines)

    for i, line in enumerate(lines):
        out.append(line)

        # After indicator(...) declaration — insert header + check A
        if not indicator_done and line.strip().startswith("indicator("):
            out.append(f'// ChartEdge.trade \u00b7 {user_lbl} \u00b7 {today.strftime("%b %d %Y")} \u00b7 {site}')
            out.append(chk_a)
            indicator_done = True

        # ~33% through — insert check B at next global-scope standalone line
        elif not chk_b_inserted and i >= max(n // 3, 3) and line and not line[0].isspace():
            next_l = lines[i + 1] if i + 1 < n else ""
            if not (next_l and next_l[0].isspace()):
                out.append(f"// \u2014 validity \u2014")
                out.append(chk_b)
                chk_b_inserted = True

        # ~66% through — insert check C at next global-scope standalone line
        elif not chk_c_inserted and i >= max(2 * n // 3, 5) and line and not line[0].isspace():
            next_l = lines[i + 1] if i + 1 < n else ""
            if not (next_l and next_l[0].isspace()):
                out.append(f"// \u2014 runtime guard \u2014")
                out.append(chk_c)
                chk_c_inserted = True

    # Wrap every plot(...) call so the value multiplies by _cedge_ok
    # (plots become na when expired, even if someone deletes the if blocks)
    wrapped = []
    for line in out:
        stripped = line.strip()
        # Match plot/plotshape/hline — add `and _cedge_ok` to condition args
        # Simpler: multiply numeric values — too fragile. Instead append a
        # final guard that resets all plots to na via bgcolor trick.
        wrapped.append(line)

    # Final guard at very end — belt + suspenders
    wrapped.append(f'\n// ChartEdge.trade guard')
    wrapped.append(f'if not _cedge_ok\n    runtime.error("{msg}")')

    return "\n".join(wrapped)


# ── Basic indicator Pine Script templates ────────────────────────────────────

def _pine_volume() -> str:
    return """\
//@version=6
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
        '//@version=6',
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
//@version=6
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
//@version=6
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
    zone      = rsi >= ob ? "Overbought" : rsi <= os ? "Oversold" : "Neutral"
    lbl_color = rsi >= ob ? color.new(color.red, 30) : rsi <= os ? color.new(color.green, 30) : color.new(color.gray, 30)
    label.new(bar_index, rsi, zone + "  " + str.tostring(math.round(rsi, 1)),
              style=label.style_label_left, size=size.small,
              color=lbl_color,
              textcolor=color.white)
"""


def _pine_ema() -> str:
    return """\
//@version=6
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
    trend      = bull ? "Bullish" : "Bearish"
    lbl_color  = bull ? color.new(color.green, 30) : color.new(color.red, 30)
    label.new(bar_index, ema50,
              trend + " - 50/200",
              style=label.style_label_left, size=size.small,
              color=lbl_color,
              textcolor=color.white)
"""


def _pine_relvol() -> str:
    return """\
//@version=6
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
//@version=6
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
//@version=6
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
//@version=6
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
//@version=6
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
cloud_color = senkou_a > senkou_b ? color.new(color.green, 80) : color.new(color.red, 80)
fill(sa, sb, color=cloud_color, title="Cloud")
"""


def _pine_obv() -> str:
    return """\
//@version=6
indicator("OBV - On-Balance Volume", overlay=false, max_bars_back=500)

// ── Inputs ────────────────────────────────────────────────────────
ma_len   = input.int(20, "Signal MA Length", minval=1)
show_ma  = input.bool(true, "Show Signal MA")

// ── OBV ───────────────────────────────────────────────────────────
obv = ta.obv

// ── Signal line ───────────────────────────────────────────────────
sig = ta.ema(obv, ma_len)

// ── Color: bullish if OBV above signal, bearish if below ──────────
bull = obv >= sig
obv_color = bull ? color.new(color.teal, 0) : color.new(color.red, 0)

// ── Plots ─────────────────────────────────────────────────────────
plot(obv,              "OBV",    color=obv_color, linewidth=2)
plot(show_ma ? sig : na,"Signal",color=color.new(color.orange, 0), linewidth=1)

// ── Fill between OBV and signal ───────────────────────────────────
p1 = plot(obv, display=display.none)
p2 = plot(show_ma ? sig : na, display=display.none)
fill_color = bull ? color.new(color.teal, 85) : color.new(color.red, 85)
fill(p1, p2, color=fill_color)

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
    trend     = bull ? "Bullish flow" : "Bearish flow"
    lbl_color = bull ? color.new(color.teal, 30) : color.new(color.red, 30)
    label.new(bar_index, obv, trend,
              style=label.style_label_left, size=size.small,
              color=lbl_color,
              textcolor=color.white)
"""


def _pine_bb_squeeze() -> str:
    return """\
//@version=6
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
    state     = squeeze ? "In Squeeze" : no_sqz ? "Fired" : "Normal"
    lbl_color = squeeze ? color.new(color.orange, 30) : no_sqz ? color.new(color.blue, 30) : color.new(color.gray, 30)
    label.new(bar_index, val,
              state,
              style=label.style_label_left, size=size.small,
              color=lbl_color,
              textcolor=color.white)
"""


def _pine_unusual_options() -> str:
    return """\
//@version=6
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
    zone      = is_unusual ? "Unusual" : is_high ? "Elevated" : "Normal"
    lbl_color = is_unusual ? color.new(color.red, 30) : is_high ? color.new(color.orange, 30) : color.new(color.gray, 30)
    label.new(bar_index, ratio,
              zone + "  " + str.tostring(math.round(ratio, 2)) + "x avg",
              style=label.style_label_left, size=size.small,
              color=lbl_color,
              textcolor=color.white)
"""


def _pine_feargreed() -> str:
    return """\
//@version=6
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


def _pine_smallcap_pullback() -> str:
    return """\
//@version=6
indicator("Small Cap Micro Pullback", overlay=true, max_bars_back=500)

// ── Inputs ──────────────────────────────────────────────────────────────────
ema_fast  = input.int(9,   "Fast EMA",              minval=1)
ema_mid   = input.int(20,  "Mid EMA",               minval=1)
ema_slow  = input.int(50,  "Slow EMA",              minval=1)
vol_mult  = input.float(1.5, "Volume surge ×",      minval=1.0, step=0.1, tooltip="Signal only fires when volume exceeds this multiple of the 20-bar average — filters noise.")
max_pb    = input.int(5,   "Max pullback bars",     minval=1,  maxval=15,  tooltip="How many consecutive pullback bars are allowed before the setup is invalidated.")
atr_mult  = input.float(1.5, "Stop ATR ×",         minval=0.5, step=0.1,  tooltip="Suggested stop = pullback low minus this many ATRs.")
show_stop = input.bool(true, "Show suggested stop")

// ── Core calculations ────────────────────────────────────────────────────────
e_fast = ta.ema(close, ema_fast)
e_mid  = ta.ema(close, ema_mid)
e_slow = ta.ema(close, ema_slow)
atr    = ta.atr(14)

vol_avg   = ta.sma(volume, 20)
vol_surge = volume >= vol_avg * vol_mult

// ── Trend filter: EMAs stacked bullishly, price above fast EMA ───────────────
uptrend = e_fast > e_mid and e_mid > e_slow and close > e_fast

// ── Relative volume label (bottom-right table) ───────────────────────────────
rvol = volume / vol_avg

// ── Micro pullback detection ─────────────────────────────────────────────────
// A micro pullback is a shallow retracement toward the fast EMA while the
// broader uptrend structure (EMA stack) remains intact.
// Conditions per pullback bar:
//   1. EMA stack still bullish
//   2. Close is below the prior bar's close  -OR-  low wicks into the fast EMA
//   3. Volume is at or below average (healthy, low-volume consolidation)

touching_ema  = low <= e_fast * 1.003              // low within 0.3% of fast EMA
weak_candle   = close < close[1] or close < open   // bearish or doji candle
low_vol_bar   = volume <= vol_avg * 1.1            // volume not spiking

pullback_bar  = uptrend and (touching_ema or weak_candle) and low_vol_bar

// Count consecutive pullback bars
var int pb_count = 0
pb_count := pullback_bar ? pb_count + 1 : 0

in_pb_zone = pb_count > 0 and pb_count <= max_pb

// ── Entry signal ─────────────────────────────────────────────────────────────
// Reclaim of fast EMA on above-average volume after a micro pullback
reclaim = close > e_fast and close > open              // bullish close above EMA
entry   = in_pb_zone[1] and reclaim and vol_surge and uptrend

// ── Pullback low for stop placement ──────────────────────────────────────────
var float pb_low = na
if entry
    pb_low := low[1]
    for i = 2 to max_pb
        if pb_count[i] > 0
            pb_low := math.min(pb_low, low[i])
        else
            break

stop_price = pb_low - atr * atr_mult

// ── Plots ────────────────────────────────────────────────────────────────────
plot(e_fast, "9 EMA",  color=color.new(color.yellow, 0), linewidth=1)
plot(e_mid,  "20 EMA", color=color.new(color.orange, 0), linewidth=1)
plot(e_slow, "50 EMA", color=color.new(color.red,   20), linewidth=2)

// Soft blue background while in pullback zone
bgcolor(in_pb_zone ? color.new(color.blue, 90) : na, title="Pullback Zone")

// Green triangle below entry bar
plotshape(entry, "Entry Signal",
          style    = shape.triangleup,
          location = location.belowbar,
          color    = color.new(color.lime, 0),
          size     = size.normal)

// Entry label
if entry
    label.new(bar_index, low - atr * 0.4,
              "PULLBACK\\n" + str.tostring(math.round(rvol, 1)) + "× vol",
              style     = label.style_label_up,
              color     = color.new(color.lime, 20),
              textcolor = color.white,
              size      = size.small)

// Suggested stop line
plot(show_stop and entry ? stop_price : na,
     "Suggested Stop", color=color.new(color.red, 20),
     style=plot.style_linebr, linewidth=1)

if show_stop and entry
    label.new(bar_index, stop_price,
              "stop " + str.tostring(math.round(stop_price, 2)),
              style     = label.style_label_right,
              color     = color.new(color.red, 30),
              textcolor = color.white,
              size      = size.tiny)

// ── RVOL table (top-right) ────────────────────────────────────────────────────
var table t = table.new(position.top_right, 2, 3,
                        bgcolor=color.new(color.black, 70), border_width=1,
                        border_color=color.new(color.gray, 60))
if barstate.islast
    rvol_color = rvol >= 2.0 ? color.new(color.red,   20)
               : rvol >= 1.5 ? color.new(color.orange, 20)
               : rvol >= 1.0 ? color.new(color.teal,   20)
               :                color.new(color.gray,   30)

    table.cell(t, 0, 0, "RVOL",   text_color=color.new(color.gray,  0), text_size=size.tiny)
    table.cell(t, 1, 0, str.tostring(math.round(rvol, 2)) + "×",
               text_color=rvol_color, text_size=size.small)

    trend_str   = uptrend ? "UPTREND" : "NO TREND"
    trend_color = uptrend ? color.new(color.lime, 20) : color.new(color.gray, 20)
    table.cell(t, 0, 1, "TREND",  text_color=color.new(color.gray, 0), text_size=size.tiny)
    table.cell(t, 1, 1, trend_str, text_color=trend_color, text_size=size.small)

    pb_str   = in_pb_zone ? "PULLBACK (" + str.tostring(pb_count) + ")" : "-"
    pb_color = in_pb_zone ? color.new(color.blue, 10) : color.new(color.gray, 30)
    table.cell(t, 0, 2, "SETUP",  text_color=color.new(color.gray, 0), text_size=size.tiny)
    table.cell(t, 1, 2, pb_str,   text_color=pb_color, text_size=size.small)

// ── Alerts ───────────────────────────────────────────────────────────────────
alertcondition(entry, "Micro Pullback Entry",
    "Small Cap Micro Pullback setup on {{ticker}} — {{interval}} | RVOL {{volume}}")
alertcondition(in_pb_zone and not in_pb_zone[1], "Pullback Started",
    "Micro pullback forming on {{ticker}} — watch for entry")
"""


def _pine_patterns() -> str:
    return """//@version=6
indicator("Chart Pattern Scanner", overlay=true, max_bars_back=500)

// ── Settings ──────────────────────────────────────────────────────────────────
int   pLen   = input.int(5,    "Pivot Length", minval=2,   maxval=20,  group="Detection")
float tolPct = input.float(2.0,"Tolerance %",  minval=0.5, maxval=8.0, group="Detection") / 100
bool showHS   = input.bool(true,"Head & Shoulders",  group="Patterns")
bool showIHS  = input.bool(true,"Inverse H&S",       group="Patterns")
bool showDT   = input.bool(true,"Double Top",        group="Patterns")
bool showDB   = input.bool(true,"Double Bottom",     group="Patterns")
bool showBF   = input.bool(true,"Bull Flag",         group="Patterns")
bool showBrF  = input.bool(true,"Bear Flag",         group="Patterns")

// ── Pivot detection ───────────────────────────────────────────────────────────
float ph = ta.pivothigh(high, pLen, pLen)
float pl = ta.pivotlow(low,  pLen, pLen)

var array<float> phV = array.new_float(0)
var array<int>   phI = array.new_int(0)
var array<float> plV = array.new_float(0)
var array<int>   plI = array.new_int(0)

if not na(ph)
    array.unshift(phV, ph)
    array.unshift(phI, bar_index - pLen)
    if array.size(phV) > 8
        array.pop(phV)
        array.pop(phI)

if not na(pl)
    array.unshift(plV, pl)
    array.unshift(plI, bar_index - pLen)
    if array.size(plV) > 8
        array.pop(plV)
        array.pop(plI)

within(a, b) => math.abs(a - b) / math.max(a, b) < tolPct

var int lHS  = -9999
var int lIHS = -9999
var int lDT  = -9999
var int lDB  = -9999
var int lBF  = -9999
var int lBrF = -9999

int nh = array.size(phV)
int nl = array.size(plV)

// ── HEAD & SHOULDERS ──────────────────────────────────────────────────────────
if showHS and nh >= 3 and barstate.isconfirmed
    float rs  = array.get(phV, 0)
    int   rsi = array.get(phI, 0)
    float hd  = array.get(phV, 1)
    int   hdi = array.get(phI, 1)
    float ls  = array.get(phV, 2)
    int   lsi = array.get(phI, 2)
    float hsNeck = math.min(ls, rs)
    if hd > ls * (1 + tolPct) and hd > rs * (1 + tolPct) and within(ls, rs) and close < hsNeck and rsi != lHS
        lHS := rsi
        line.new(lsi, ls,  hdi, hd, color=color.new(color.red, 20), width=2)
        line.new(hdi, hd,  rsi, rs, color=color.new(color.red, 20), width=2)
        line.new(lsi, hsNeck, rsi, hsNeck, color=color.new(color.red, 40), width=1, style=line.style_dashed)
        label.new(rsi, rs, "H&S  ▼", style=label.style_label_down, color=color.new(color.red, 10), textcolor=color.white, size=size.normal)

// ── INVERSE HEAD & SHOULDERS ──────────────────────────────────────────────────
if showIHS and nl >= 3 and barstate.isconfirmed
    float rs  = array.get(plV, 0)
    int   rsi = array.get(plI, 0)
    float hd  = array.get(plV, 1)
    int   hdi = array.get(plI, 1)
    float ls  = array.get(plV, 2)
    int   lsi = array.get(plI, 2)
    float ihsNeck = math.max(ls, rs)
    if hd < ls * (1 - tolPct) and hd < rs * (1 - tolPct) and within(ls, rs) and close > ihsNeck and rsi != lIHS
        lIHS := rsi
        line.new(lsi, ls,  hdi, hd, color=color.new(color.green, 20), width=2)
        line.new(hdi, hd,  rsi, rs, color=color.new(color.green, 20), width=2)
        line.new(lsi, ihsNeck, rsi, ihsNeck, color=color.new(color.green, 40), width=1, style=line.style_dashed)
        label.new(rsi, rs, "IH&S  ▲", style=label.style_label_up, color=color.new(color.green, 10), textcolor=color.white, size=size.normal)

// ── DOUBLE TOP ────────────────────────────────────────────────────────────────
if showDT and nh >= 2 and barstate.isconfirmed
    float p2  = array.get(phV, 0)
    int   p2i = array.get(phI, 0)
    float p1  = array.get(phV, 1)
    int   p1i = array.get(phI, 1)
    float dtMid = (p1 + p2) / 2 * (1 - tolPct)
    if within(p1, p2) and (p2i - p1i) >= pLen * 3 and close < dtMid and p2i != lDT
        lDT := p2i
        float res = math.max(p1, p2)
        line.new(p1i, res, p2i, res, color=color.new(color.red, 20), width=2, style=line.style_dashed)
        line.new(p1i, p1,  p2i, p2,  color=color.new(color.red, 40), width=1)
        label.new(p2i, p2, "Double Top  ▼", style=label.style_label_down, color=color.new(color.red, 10), textcolor=color.white, size=size.normal)

// ── DOUBLE BOTTOM ─────────────────────────────────────────────────────────────
if showDB and nl >= 2 and barstate.isconfirmed
    float p2  = array.get(plV, 0)
    int   p2i = array.get(plI, 0)
    float p1  = array.get(plV, 1)
    int   p1i = array.get(plI, 1)
    float dbMid = (p1 + p2) / 2 * (1 + tolPct)
    if within(p1, p2) and (p2i - p1i) >= pLen * 3 and close > dbMid and p2i != lDB
        lDB := p2i
        float sup = math.min(p1, p2)
        line.new(p1i, sup, p2i, sup, color=color.new(color.green, 20), width=2, style=line.style_dashed)
        line.new(p1i, p1,  p2i, p2,  color=color.new(color.green, 40), width=1)
        label.new(p2i, p2, "Double Bottom  ▲", style=label.style_label_up, color=color.new(color.green, 10), textcolor=color.white, size=size.normal)

// ── FLAGS ─────────────────────────────────────────────────────────────────────
if (showBF or showBrF) and barstate.isconfirmed
    float poleUp = (close[8]  - close[15]) / math.max(close[15], 0.01)
    float poleDn = (close[15] - close[8])  / math.max(close[15], 0.01)
    float fHigh  = ta.highest(high, 8)
    float fLow   = ta.lowest(low,  8)
    float fRange = (fHigh - fLow) / math.max(fLow, 0.01)
    float fSlope = (close - close[8]) / math.max(close[8], 0.01)
    if showBF and poleUp > 0.04 and fSlope < -0.003 and fSlope > -0.04 and fRange < poleUp * 0.5 and bar_index != lBF
        lBF := bar_index
        line.new(bar_index - 15, close[15], bar_index - 8, close[8], color=color.new(color.green, 10), width=3)
        line.new(bar_index - 8,  fHigh,     bar_index,     fLow,     color=color.new(color.green, 40), width=1, style=line.style_dashed)
        label.new(bar_index, fLow, "Bull Flag  ▲", style=label.style_label_up, color=color.new(color.green, 10), textcolor=color.white, size=size.normal)
    if showBrF and poleDn > 0.04 and fSlope > 0.003 and fSlope < 0.04 and fRange < poleDn * 0.5 and bar_index != lBrF
        lBrF := bar_index
        line.new(bar_index - 15, close[15], bar_index - 8, close[8], color=color.new(color.red,   10), width=3)
        line.new(bar_index - 8,  fLow,      bar_index,     fHigh,    color=color.new(color.red,   40), width=1, style=line.style_dashed)
        label.new(bar_index, fHigh, "Bear Flag  ▼", style=label.style_label_down, color=color.new(color.red, 10), textcolor=color.white, size=size.normal)



"""


def _pine_triangles() -> str:
    return """//@version=6
indicator("Triangle & Wedge Scanner", overlay=true, max_bars_back=500)

// ── Settings ──────────────────────────────────────────────────────────────────
int   pLen   = input.int(5,    "Pivot Length", minval=2,   maxval=20,  group="Detection")
float tolPct = input.float(2.0,"Tolerance %",  minval=0.5, maxval=8.0, group="Detection") / 100
bool showAT   = input.bool(true,"Ascending Triangle",  group="Patterns")
bool showDTri = input.bool(true,"Descending Triangle", group="Patterns")
bool showST   = input.bool(true,"Sym. Triangle",       group="Patterns")
bool showRW   = input.bool(true,"Rising Wedge",        group="Patterns")
bool showFW   = input.bool(true,"Falling Wedge",       group="Patterns")

// ── Pivot detection ───────────────────────────────────────────────────────────
float ph = ta.pivothigh(high, pLen, pLen)
float pl = ta.pivotlow(low,  pLen, pLen)

var array<float> phV = array.new_float(0)
var array<int>   phI = array.new_int(0)
var array<float> plV = array.new_float(0)
var array<int>   plI = array.new_int(0)

if not na(ph)
    array.unshift(phV, ph)
    array.unshift(phI, bar_index - pLen)
    if array.size(phV) > 8
        array.pop(phV)
        array.pop(phI)

if not na(pl)
    array.unshift(plV, pl)
    array.unshift(plI, bar_index - pLen)
    if array.size(plV) > 8
        array.pop(plV)
        array.pop(plI)

slp(i1, v1, i2, v2) => (v2 - v1) / math.max(i2 - i1, 1)

var int lAT  = -9999
var int lDTr = -9999
var int lST  = -9999
var int lRW  = -9999
var int lFW  = -9999

int nh = array.size(phV)
int nl = array.size(plV)

// ── TRIANGLES & WEDGES ────────────────────────────────────────────────────────
if nh >= 2 and nl >= 2 and barstate.isconfirmed
    float h1  = array.get(phV, 1)
    int   h1i = array.get(phI, 1)
    float h2  = array.get(phV, 0)
    int   h2i = array.get(phI, 0)
    float l1  = array.get(plV, 1)
    int   l1i = array.get(plI, 1)
    float l2  = array.get(plV, 0)
    int   l2i = array.get(plI, 0)
    float sH  = slp(h1i, h1, h2i, h2)
    float sL  = slp(l1i, l1, l2i, l2)
    float ft  = close * 0.00005
    bool  hFlat = math.abs(sH) < ft
    bool  lFlat = math.abs(sL) < ft
    bool  hUp   = sH > ft
    bool  lUp   = sL > ft
    bool  hDn   = sH < -ft
    bool  lDn   = sL < -ft
    int   ref   = math.max(h2i, l2i)
    if showAT and hFlat and lUp and close > h2 and ref != lAT
        lAT := ref
        line.new(h1i, h1, h2i, h2, color=color.new(color.teal, 20), width=2)
        line.new(l1i, l1, l2i, l2, color=color.new(color.teal, 20), width=2)
        label.new(ref, h2, "Asc. Triangle  ▲", style=label.style_label_down, color=color.new(color.teal, 10), textcolor=color.white, size=size.normal)
    else if showDTri and lFlat and hDn and close < l2 and ref != lDTr
        lDTr := ref
        line.new(h1i, h1, h2i, h2, color=color.new(color.orange, 20), width=2)
        line.new(l1i, l1, l2i, l2, color=color.new(color.orange, 20), width=2)
        label.new(ref, l2, "Desc. Triangle  ▼", style=label.style_label_up, color=color.new(color.orange, 10), textcolor=color.white, size=size.normal)
    else if showST and hDn and lUp and (close > h2 or close < l2) and ref != lST
        lST := ref
        line.new(h1i, h1, h2i, h2, color=color.new(color.blue, 20), width=2)
        line.new(l1i, l1, l2i, l2, color=color.new(color.blue, 20), width=2)
        if close > h2
            label.new(ref, h2, "Sym. Triangle  ▲", style=label.style_label_down, color=color.new(color.green, 10), textcolor=color.white, size=size.normal)
        else
            label.new(ref, l2, "Sym. Triangle  ▼", style=label.style_label_up, color=color.new(color.red, 10), textcolor=color.white, size=size.normal)
    else if showRW and hUp and lUp and math.abs(sL) > math.abs(sH) * 1.2 and close < l2 and ref != lRW
        lRW := ref
        line.new(h1i, h1, h2i, h2, color=color.new(color.red, 20), width=2)
        line.new(l1i, l1, l2i, l2, color=color.new(color.red, 20), width=2)
        label.new(ref, h2, "Rising Wedge  ▼", style=label.style_label_down, color=color.new(color.red, 10), textcolor=color.white, size=size.normal)
    else if showFW and hDn and lDn and math.abs(sH) > math.abs(sL) * 1.2 and close > h2 and ref != lFW
        lFW := ref
        line.new(h1i, h1, h2i, h2, color=color.new(color.green, 20), width=2)
        line.new(l1i, l1, l2i, l2, color=color.new(color.green, 20), width=2)
        label.new(ref, l2, "Falling Wedge  ▲", style=label.style_label_up, color=color.new(color.green, 10), textcolor=color.white, size=size.normal)
"""


def _pine_prev_day() -> str:
    return """//@version=6
indicator("Prev Day High / Low", overlay=true, max_lines_count=100, max_labels_count=50)

// ── Settings ──────────────────────────────────────────────────────────────────
bool showHigh  = input.bool(true, "Show PDH",    group="Levels")
bool showLow   = input.bool(true, "Show PDL",    group="Levels")
bool showLabel = input.bool(true, "Show Labels", group="Levels")
bool showTable = input.bool(true, "Show Table",  group="Levels")
color highCol  = input.color(color.new(color.red,   0), "PDH Color", group="Style")
color lowCol   = input.color(color.new(color.green, 0), "PDL Color", group="Style")
int   lw       = input.int(1, "Line Width", minval=1, maxval=4, group="Style")

// ── Previous day values via daily security ─────────────────────────────────────
float pdHigh = request.security(syminfo.tickerid, "D", high[1], lookahead=barmerge.lookahead_on)
float pdLow  = request.security(syminfo.tickerid, "D", low[1],  lookahead=barmerge.lookahead_on)

// ── Draw on the first bar of each new day ─────────────────────────────────────
bool newDay = ta.change(time("D")) != 0

if newDay and not na(pdHigh) and not na(pdLow)
    if showHigh
        line.new(bar_index, pdHigh, bar_index + 1, pdHigh, color=highCol, width=lw, style=line.style_solid, extend=extend.right)
        if showLabel
            label.new(bar_index, pdHigh, "PDH  " + str.tostring(pdHigh, format.mintick), style=label.style_label_right, color=color.new(highCol, 20), textcolor=color.white, size=size.small)
    if showLow
        line.new(bar_index, pdLow, bar_index + 1, pdLow, color=lowCol, width=lw, style=line.style_solid, extend=extend.right)
        if showLabel
            label.new(bar_index, pdLow, "PDL  " + str.tostring(pdLow, format.mintick), style=label.style_label_right, color=color.new(lowCol, 20), textcolor=color.white, size=size.small)

// ── Info table ─────────────────────────────────────────────────────────────────
if barstate.islast and showTable and not na(pdHigh)
    var table t = table.new(position.top_right, 2, 3, bgcolor=color.new(color.black, 80), border_width=1, border_color=color.new(color.gray, 60))
    table.cell(t, 0, 0, "Level", text_color=color.gray, text_size=size.small, bgcolor=color.new(color.gray, 85))
    table.cell(t, 1, 0, "Price", text_color=color.gray, text_size=size.small, bgcolor=color.new(color.gray, 85))
    table.cell(t, 0, 1, "PDH",   text_color=highCol,    text_size=size.small)
    table.cell(t, 1, 1, str.tostring(pdHigh, format.mintick), text_color=highCol, text_size=size.small)
    table.cell(t, 0, 2, "PDL",   text_color=lowCol,     text_size=size.small)
    table.cell(t, 1, 2, str.tostring(pdLow,  format.mintick), text_color=lowCol,  text_size=size.small)
"""

def _pine_premarket() -> str:
    return """//@version=6
indicator("Pre-Market High / Low", overlay=true, max_lines_count=50, max_labels_count=20)

// ── Settings ──────────────────────────────────────────────────────────────────
bool showHigh  = input.bool(true,  "Show PM High",    group="Levels")
bool showLow   = input.bool(true,  "Show PM Low",     group="Levels")
bool showLabel = input.bool(true,  "Show Labels",     group="Levels")
bool showTable = input.bool(true,  "Show Info Table", group="Levels")
color highCol  = input.color(color.new(color.blue,    0), "High Color", group="Style")
color lowCol   = input.color(color.new(color.fuchsia, 0), "Low Color",  group="Style")
int   lw       = input.int(2, "Line Width", minval=1, maxval=4, group="Style")

// ── Session detection (ET) ─────────────────────────────────────────────────────
bool isPM  = not na(time(timeframe.period, "0400-0930", "America/New_York"))
bool isReg = not na(time(timeframe.period, "0930-1600", "America/New_York"))

// ── Track PM high/low ──────────────────────────────────────────────────────────
var float pmHigh     = na
var float pmLow      = na
var int   pmStartBar = na
var bool  pmCaptured = false

bool newDay = ta.change(dayofweek) != 0

if newDay
    pmHigh     := na
    pmLow      := na
    pmStartBar := na
    pmCaptured := false

if isPM and not isPM[1]
    pmHigh     := high
    pmLow      := low
    pmStartBar := bar_index

if isPM and not na(pmHigh)
    pmHigh     := math.max(pmHigh, high)
    pmLow      := math.min(pmLow,  low)
    pmCaptured := true

// ── Draw lines at market open ──────────────────────────────────────────────────
bool firstRegBar = isReg and not isReg[1]

if firstRegBar and pmCaptured
    if showHigh
        line.new(pmStartBar, pmHigh, bar_index, pmHigh, color=highCol, width=lw, style=line.style_solid, extend=extend.right)
        if showLabel
            label.new(bar_index, pmHigh, "PM High  " + str.tostring(pmHigh, format.mintick), style=label.style_label_left, color=color.new(highCol, 20), textcolor=color.white, size=size.small)
    if showLow
        line.new(pmStartBar, pmLow, bar_index, pmLow, color=lowCol, width=lw, style=line.style_solid, extend=extend.right)
        if showLabel
            label.new(bar_index, pmLow, "PM Low  " + str.tostring(pmLow, format.mintick), style=label.style_label_left, color=color.new(lowCol, 20), textcolor=color.white, size=size.small)

// ── Info table ─────────────────────────────────────────────────────────────────
if barstate.islast and showTable
    var table t = table.new(position.top_right, 2, 3, bgcolor=color.new(color.black, 80), border_width=1, border_color=color.new(color.gray, 60))
    table.cell(t, 0, 0, "Level",   text_color=color.gray, text_size=size.small, bgcolor=color.new(color.gray, 85))
    table.cell(t, 1, 0, "Price",   text_color=color.gray, text_size=size.small, bgcolor=color.new(color.gray, 85))
    if pmCaptured
        table.cell(t, 0, 1, "PM High", text_color=highCol, text_size=size.small)
        table.cell(t, 1, 1, str.tostring(pmHigh, format.mintick), text_color=highCol, text_size=size.small)
        table.cell(t, 0, 2, "PM Low",  text_color=lowCol,  text_size=size.small)
        table.cell(t, 1, 2, str.tostring(pmLow,  format.mintick), text_color=lowCol,  text_size=size.small)
    else
        table.cell(t, 0, 1, "Enable", text_color=color.orange, text_size=size.small)
        table.cell(t, 1, 1, "Ext Hours", text_color=color.orange, text_size=size.small)
        table.cell(t, 0, 2, "in chart", text_color=color.gray, text_size=size.small)
        table.cell(t, 1, 2, "settings", text_color=color.gray, text_size=size.small)
"""


def _pine_macd_hist() -> str:
    return """//@version=6
indicator("MACD Histogram", "MACD Hist", overlay=false)

float sourceInput  = input.source(close, "Source")
int   fastLenInput = input.int(12, "Fast length",   minval=1)
int   slowLenInput = input.int(26, "Slow length",   minval=1)
int   sigLenInput  = input.int(9,  "Signal length", minval=1)
string oscType     = input.string("EMA", "Oscillator MA", ["EMA", "SMA"])
string sigType     = input.string("EMA", "Signal MA",     ["EMA", "SMA"])

ma(float src, int len, simple string t) =>
    switch t
        "EMA" => ta.ema(src, len)
        "SMA" => ta.sma(src, len)

float maFast = ma(sourceInput, fastLenInput, oscType)
float maSlow = ma(sourceInput, slowLenInput, oscType)
float macd   = maFast - maSlow
float signal = ma(macd, sigLenInput, sigType)
float hist   = macd - signal

color hColor = hist >= 0 ? hist > hist[1] ? #26a69a : #b2dfdb : hist > hist[1] ? #ffcdd2 : #ff5252

bool showLines = input.bool(false, "Show MACD / Signal lines")

hline(0, "Zero", #787b8680)
plot(hist, "Histogram", hColor, style=plot.style_columns)
plot(showLines ? macd   : na, "MACD",   color=#2962ff, linewidth=2)
plot(showLines ? signal : na, "Signal", color=#ff6d00, linewidth=1)

alertcondition(hist[1] >= 0 and hist < 0, "Rising to falling", "MACD histogram switched from rising to falling")
alertcondition(hist[1] <= 0 and hist > 0, "Falling to rising", "MACD histogram switched from falling to rising")
"""


def _pine_ema200_band() -> str:
    return """//@version=6
indicator("200 EMA Band (High / Close / Low)", overlay=true, max_bars_back=500)

// ── Settings ──────────────────────────────────────────────────────────────────
int    emaLen   = input.int(200, "EMA Length", minval=1, group="Settings")
bool   showFill = input.bool(true, "Fill between bands", group="Settings")
color  highCol  = input.color(color.new(color.red,   60), "High EMA",   group="Colors")
color  closeCol = input.color(color.new(color.white,  0), "Close EMA",  group="Colors")
color  lowCol   = input.color(color.new(color.green, 60), "Low EMA",    group="Colors")
color  fillCol  = input.color(color.new(color.gray,  85), "Fill",       group="Colors")

// ── Calculations ──────────────────────────────────────────────────────────────
float emaHigh  = ta.ema(high,  emaLen)
float emaClose = ta.ema(close, emaLen)
float emaLow   = ta.ema(low,   emaLen)

// ── Plots ─────────────────────────────────────────────────────────────────────
pH = plot(emaHigh,  "EMA High",  color=highCol,  linewidth=1)
pC = plot(emaClose, "EMA Close", color=closeCol, linewidth=2)
pL = plot(emaLow,   "EMA Low",   color=lowCol,   linewidth=1)

fill(pH, pL, color=showFill ? fillCol : na)
"""


def _pine_multi_vwap() -> str:
    return """//@version=6
indicator("Multi-Period VWAP", overlay=true, max_bars_back=500)

// ── Visibility ────────────────────────────────────────────────────────────────
bool showD = input.bool(true,  "Daily VWAP",   group="Visibility")
bool showW = input.bool(true,  "Weekly VWAP",  group="Visibility")
bool showM = input.bool(true,  "Monthly VWAP", group="Visibility")
bool showY = input.bool(false, "Yearly VWAP",  group="Visibility")

// ── Colors ────────────────────────────────────────────────────────────────────
color dCol = input.color(color.new(color.aqua,   0), "Daily",   group="Colors")
color wCol = input.color(color.new(color.purple, 0), "Weekly",  group="Colors")
color mCol = input.color(color.new(color.orange, 0), "Monthly", group="Colors")
color yCol = input.color(color.new(color.yellow, 0), "Yearly",  group="Colors")

// ── VWAP calculations ─────────────────────────────────────────────────────────
float pv = hlc3 * volume

// Daily
bool newD     = ta.change(time("D")) != 0
var float dPV = 0.0
var float dV  = 0.0
dPV := newD ? pv : dPV + pv
dV  := newD ? volume : dV + volume
float dVwap = dPV / dV

// Weekly
bool newW     = ta.change(time("W")) != 0
var float wPV = 0.0
var float wV  = 0.0
wPV := newW ? pv : wPV + pv
wV  := newW ? volume : wV + volume
float wVwap = wPV / wV

// Monthly
bool newM     = ta.change(time("M")) != 0
var float mPV = 0.0
var float mV  = 0.0
mPV := newM ? pv : mPV + pv
mV  := newM ? volume : mV + volume
float mVwap = mPV / mV

// Yearly
bool newY     = year != year[1]
var float yPV = 0.0
var float yV  = 0.0
yPV := newY ? pv : yPV + pv
yV  := newY ? volume : yV + volume
float yVwap = yPV / yV

// ── Plots ─────────────────────────────────────────────────────────────────────
plot(showD ? dVwap : na, "Daily VWAP",   color=dCol, linewidth=1)
plot(showW ? wVwap : na, "Weekly VWAP",  color=wCol, linewidth=2)
plot(showM ? mVwap : na, "Monthly VWAP", color=mCol, linewidth=2)
plot(showY ? yVwap : na, "Yearly VWAP",  color=yCol, linewidth=3)
"""


# Each entry: (name, fn, short_description, category, how_to_use, tag)
# tag: "" = none, "beta" = BETA badge
INDICATORS = {
    "volume":    ("24h Volume",      _pine_volume,    "Cumulative volume since market open vs 20-day average daily volume. Red = above average day.",          "volume",     "Add to any chart. Appears as a separate panel below the price. Green bars = normal up-volume, red bars = unusually high volume (1.5× avg). Orange line is the average daily volume benchmark — when the bars are tracking above it, today is an active day.", ""),
    "vwap":      ("VWAP + Bands",    _pine_vwap,      "VWAP with configurable ±1, ±2, ±3 standard deviation bands. Overlaid directly on the price chart.",    "volume",     "Select which bands to show, then copy and paste onto your chart. Price above VWAP = buyers in control. Price near the ±2 red band = overextended, often snaps back. Use on intraday charts (1m–1h) — VWAP resets each day.", ""),
    "vwap_only": ("VWAP Only",       lambda: "//@version=6\nindicator(\"VWAP\", overlay=true, max_bars_back=500)\n\nplot(ta.vwap(hlc3), \"VWAP\", color=color.new(color.blue, 0), linewidth=2)\n", "Just the VWAP line, no bands. Clean and simple, overlaid on the price chart.", "volume", "Paste onto any intraday chart. A single blue line shows the volume-weighted average price for the session. Price above = bullish bias, price below = bearish bias. Resets at market open each day.", ""),
    "multivwap":  ("Multi-Period VWAP", _pine_multi_vwap, "Daily, weekly, monthly, and yearly VWAP lines on one chart. Toggle each period on/off independently.", "volume", "Paste onto your price chart. Daily VWAP (aqua) resets each session — best for intraday bias. Weekly (purple) shows the medium-term mean. Monthly (orange) and Yearly (yellow) act as major support/resistance magnets. Turn each one on or off in TradingView settings. Price above all VWAPs = strong bullish bias.", ""),
    "atr":       ("ATR",             _pine_atr,       "Average True Range (14). Shows how much the asset moves per bar. Red = elevated volatility.",           "volatility", "Add to any chart as a separate panel. The red line shows raw volatility per bar. Toggle the orange average line to see whether current volatility is above or below normal. High ATR = bigger stops needed. Low ATR = tight, choppy market.", ""),
    "rsi":       ("RSI",               _pine_rsi,       "Relative Strength Index with overbought/oversold zones, signal MA, and automatic bullish/bearish divergence labels.", "momentum", "Add as a separate panel. Above 70 = overbought (red zone), below 30 = oversold (green zone). The orange signal line is a 9-bar EMA of RSI — crossovers can signal entries. D+ labels mark bullish divergence (price falling but RSI rising), D- marks bearish divergence. Best used with a trend indicator to filter signals.", ""),
    "ema":       ("EMA Ribbon",       _pine_ema,       "8, 21, 50, 100, and 200 EMAs overlaid on the price chart. Green/red fill between the 50 and 200 shows trend direction.", "trend", "Paste onto your price chart. Toggle which EMAs you want in TradingView settings. Green fill between the 50 and 200 EMA = bullish trend, red = bearish. The 8/21 EMAs react fast and are good for short-term entries. The 50/200 are slower and better for trend confirmation. A label on the last bar shows the current trend.", ""),
    "relvol":    ("Relative Volume", _pine_relvol,    "Today's volume vs average. RVOL > 2 = unusually active. Threshold is adjustable in TradingView.",      "volume",     "Add as a separate panel. RVOL of 1.0 = exactly average. Teal bars = above average. Red bars = unusually high (default threshold: 2×). High RVOL on a breakout confirms the move. High RVOL on a reversal signals a strong change. Low RVOL moves are often noise.", ""),
    "macross":   ("MA Cross",        _pine_ma_cross,  "50/200 MA crossover. Labels Golden Cross (bullish) and Death Cross (bearish) directly on the chart.",  "trend",      "Paste onto your price chart. Green background = uptrend (50 above 200). Red background = downtrend. A 'Golden' label appears when the 50 crosses above the 200 — historically a strong bullish signal. 'Death' appears on the cross below. Best used on daily charts.", ""),
    "macd":      ("MACD",            _pine_macd,      "MACD line, signal line, and histogram. ▲/▼ labels mark bullish and bearish crossovers.",                  "trend",      "Add as a separate panel. The blue MACD line crossing above the orange signal line = bullish momentum. Crossing below = bearish. The histogram shows the gap between the two — green bars growing = strengthening uptrend, red bars growing = strengthening downtrend.", ""),
    "supertrend":("Supertrend",      _pine_supertrend,"Dynamic support/resistance line that flips direction. BUY/SELL labels on every trend change.",             "trend",      "Paste onto your price chart. Green line below price = uptrend. Red line above price = downtrend. BUY label appears when trend flips bullish, SELL when it flips bearish. Adjust the ATR multiplier in settings — higher = fewer signals, less noise.", ""),
    "ichimoku":  ("Ichimoku Cloud",  _pine_ichimoku,  "Full Ichimoku system: Tenkan, Kijun, cloud (Senkou A/B), and Chikou span overlaid on price.",             "trend",      "Paste onto your price chart. Green cloud = bullish, red cloud = bearish. Price above the cloud = strong uptrend. Price inside the cloud = consolidation. Price below = downtrend. The Tenkan/Kijun cross is a short-term signal. Best used on daily or 4h charts.", ""),
    "feargreed":     ("Fear & Greed",           _pine_feargreed,     "Composite 0–100 index built from RSI, trend strength, volatility, VIX, and 52-week momentum.",         "momentum", "Add as a separate panel on any chart. Reads 0–100: below 25 = Extreme Fear (often a buying opportunity), above 75 = Extreme Greed (market may be overheated). The label on the last bar shows the current reading and zone. Works on any ticker — uses VIX as one of its inputs.", ""),
    "obv":           ("OBV",               _pine_obv,       "On-Balance Volume — tracks whether volume is flowing into or out of a stock. Divergence labels flag when price and OBV disagree.", "volume", "Add as a separate panel. OBV rising = money flowing in (bullish). OBV falling = money flowing out (bearish). When OBV diverges from price — price falling but OBV rising (D+ label) = smart money buying the dip. Orange signal line helps confirm trend direction.", ""),
    "bbsqueeze":     ("Bollinger Squeeze",      _pine_bb_squeeze,      "Detects when Bollinger Bands contract inside Keltner Channels — a coiling signal before a big move. Orange dot = in squeeze, blue = fired.", "volatility", "Add as a separate panel. Orange dots on the zero line = squeeze is active (market coiling). Blue dots = squeeze just fired (potential breakout). Green histogram = bullish momentum building, red = bearish. The bigger the histogram bars after the squeeze fires, the stronger the move.", ""),
    "unusualopts":   ("Unusual Options Volume", _pine_unusual_options, "Flags bars where volume spikes above a multiple of the 20-bar average. Red = unusual, orange = elevated.", "volume",   "Add as a separate panel. Each bar shows today's volume as a multiple of the average (1.0 = normal). Red bars with a ⚡ label = unusual spike (default 2× threshold). Orange = elevated but not extreme. Adjust the threshold in TradingView settings. High spikes often precede big moves — watch for them before earnings or news.", ""),
    "prevday":       ("Prev Day High / Low",      _pine_prev_day,          "Draws the previous day's high and low as horizontal lines on the current day's chart. Works on any timeframe.", "smallcap", "Paste onto any chart. Red line = previous day high (PDH), green line = previous day low (PDL). These are the most watched key levels intraday — a break and hold above PDH is a strong bullish signal, below PDL is bearish. Works on all timeframes and instruments. No extended hours required.", ""),
    "premarket":     ("Pre-Market High / Low",   _pine_premarket,         "Draws horizontal lines at the pre-market high and low (4–9:30 AM ET) when regular session opens. Key levels for gap plays and breakouts.", "smallcap", "Paste onto any intraday chart (1m–15m). Works on US equities. Blue line = pre-market high, magenta line = pre-market low. These are the first key levels price tests at the open. A break above PM High = bullish gap continuation. A rejection at PM High = potential fade. The info table top-right shows both levels numerically. Best used on 1m–5m charts alongside VWAP.", ""),
    "smallcap_pb":   ("Small Cap Micro Pullback", _pine_smallcap_pullback, "Detects shallow pullbacks to the 9 EMA in an uptrending small cap. Green triangle = reclaim signal. Includes RVOL table and suggested stop.", "smallcap", "Paste onto a 1m, 2m, or 5m intraday chart. Works best on small/mid-cap stocks showing a strong pre-market gap or morning momentum run. The blue shading shows the pullback zone. A green triangle fires when price reclaims the 9 EMA with above-average volume after the pullback. The red dashed line is a suggested ATR-based stop below the pullback low. Adjust 'Max pullback bars' and 'Volume surge ×' in TradingView's indicator settings to tune sensitivity. Best used alongside the VWAP indicator to confirm the reclaim happens above VWAP.", ""),
    "patterns":      ("Chart Pattern Scanner", _pine_patterns, "Detects the 6 most reliable chart patterns: H&S, Inverse H&S, Double Top, Double Bottom, Bull Flag, Bear Flag. Each pattern is drawn on the chart with its shape.", "patterns", "Paste onto any price chart. Each pattern is drawn directly — outline lines and a neckline or resistance level. Green labels = bullish, red = bearish. Use Pivot Length to tune sensitivity: lower catches more patterns on short timeframes, higher finds only major swings.", ""),
    "triangles":     ("Triangle & Wedge Scanner", _pine_triangles, "Detects 5 continuation and neutral patterns: Ascending Triangle, Descending Triangle, Sym. Triangle, Rising Wedge, Falling Wedge. Draws both trendlines on the chart.", "patterns", "Paste onto any price chart. Both bounding trendlines are drawn for each pattern. Teal = Ascending Triangle (bullish breakout expected), Orange = Descending Triangle (bearish), Blue = Sym. Triangle (wait for breakout), Red = Rising Wedge (bearish reversal), Green = Falling Wedge (bullish reversal). Use alongside the Chart Pattern Scanner.", ""),
    # ── Beta indicators ───────────────────────────────────────────────────────
    "bbands":        ("Bollinger Bands",      lambda: "//@version=6\nindicator(\"Bollinger Bands\", overlay=true)\nint   len   = input.int(20, \"Length\", minval=1)\nfloat mult  = input.float(2.0, \"Multiplier\", step=0.5)\nint   off   = input.int(0,    \"Offset\",     minval=-500, maxval=500)\nfloat basis = ta.sma(close, len)\nfloat dev   = mult * ta.stdev(close, len)\nfloat upper = basis + dev\nfloat lower = basis - dev\np = plot(upper, \"Upper\", color=color.new(color.blue, 40), offset=off)\nq = plot(lower, \"Lower\", color=color.new(color.blue, 40), offset=off)\nplot(basis, \"Basis\", color=color.new(color.orange, 0), offset=off)\nfill(p, q, color=color.new(color.blue, 90))\nbgcolor(close > upper ? color.new(color.red, 92) : close < lower ? color.new(color.green, 92) : na)\n", "Upper/lower bands ±2 standard deviations from a 20-period MA. Price touching the bands signals overextension.", "volatility", "Paste onto your price chart. Price touching the upper band = overbought, lower band = oversold. Bands squeezing together = low volatility coiling before a big move. Bands expanding = momentum. Best combined with RSI to confirm entries.", ""),
    "stochrsi":      ("Stochastic RSI",       lambda: "//@version=6\nindicator(\"Stochastic RSI\", overlay=false)\nint rsiLen   = input.int(14, \"RSI Length\")\nint stochLen = input.int(14, \"Stoch Length\")\nint smooth_k = input.int(3,  \"Smooth K\")\nint smooth_d = input.int(3,  \"Smooth D\")\nfloat rsi    = ta.rsi(close, rsiLen)\nfloat rsiHi  = ta.highest(rsi, stochLen)\nfloat rsiLo  = ta.lowest(rsi,  stochLen)\nfloat k      = ta.sma(100 * (rsi - rsiLo) / math.max(rsiHi - rsiLo, 0.001), smooth_k)\nfloat d      = ta.sma(k, smooth_d)\nplot(k, \"%K\", color=color.new(color.blue,   0), linewidth=2)\nplot(d, \"%D\", color=color.new(color.orange, 0), linewidth=1)\nhline(80, \"OB\", color=color.new(color.red,   50), linestyle=hline.style_dashed)\nhline(20, \"OS\", color=color.new(color.green, 50), linestyle=hline.style_dashed)\nbgcolor(k > 80 ? color.new(color.red, 92) : k < 20 ? color.new(color.green, 92) : na)\n", "RSI smoothed through the Stochastic formula. More sensitive than RSI — great for timing entries.", "momentum", "Add as a separate panel. K line above 80 = overbought (red zone), below 20 = oversold (green zone). K crossing above D in the oversold zone = buy signal. K crossing below D in the overbought zone = sell signal. Reacts faster than plain RSI.", ""),
    "psar":          ("Parabolic SAR",        lambda: "//@version=6\nindicator(\"Parabolic SAR\", overlay=true)\nfloat start = input.float(0.02, \"Start\",     step=0.01)\nfloat inc   = input.float(0.02, \"Increment\", step=0.01)\nfloat maxv  = input.float(0.2,  \"Max\",       step=0.05)\nfloat psar  = ta.sar(start, inc, maxv)\nbool  bull  = close > psar\nplotshape(psar, style=shape.circle, location=location.absolute,\n    color=bull ? color.new(color.green, 0) : color.new(color.red, 0), size=size.tiny)\n", "Trailing dots that flip above/below price to signal trend reversals. Doubles as a dynamic stop-loss.", "trend", "Paste onto your price chart. Green dots below price = uptrend, use as a trailing stop. Red dots above price = downtrend. When dots flip sides, the trend has reversed. Tighten the Start value for more signals, loosen for fewer.", ""),
    "ema200band":    ("200 EMA Band",         _pine_ema200_band,       "Three 200 EMAs plotted on high, close, and low — forming a dynamic channel. Price inside the band = in the zone. Above = bullish, below = bearish.", "trend", "Paste onto your price chart. The band shows the 200 EMA range across high, close, and low. When price is inside the channel it's at the 200 EMA decision zone. Price cleanly above the band = strong uptrend. Below = downtrend. The close EMA (white) is the main reference line.", ""),
    "ema200":        ("200 EMA",              lambda: "//@version=6\nindicator(\"200 EMA + Filtered SAR\", overlay=true)\nfloat start = input.float(0.02, \"SAR Start\",     step=0.01)\nfloat inc   = input.float(0.02, \"SAR Increment\", step=0.01)\nfloat maxv  = input.float(0.2,  \"SAR Max\",       step=0.05)\nfloat e200  = ta.ema(close, 200)\nfloat psar  = ta.sar(start, inc, maxv)\nbool  above = close > e200\nbool  bull  = close > psar\nbool  show  = (bull and above) or (not bull and not above)\nplot(e200, \"200 EMA\", color=above ? color.new(color.green, 0) : color.new(color.red, 0), linewidth=2)\nplotshape(show ? psar : na, style=shape.circle, location=location.absolute,\n    color=bull ? color.new(color.green, 0) : color.new(color.red, 0), size=size.tiny)\nbgcolor(above ? color.new(color.green, 95) : color.new(color.red, 95))\n", "200 EMA with a Parabolic SAR overlay — SAR dots are filtered to only show signals that align with the 200 EMA trend direction.", "trend", "Paste onto your price chart. Green background = price above 200 EMA (uptrend) — only bullish SAR dots shown below price. Red background = downtrend — only bearish SAR dots shown above price. Counter-trend dots are hidden to reduce noise. Use the dots as trailing stops.", ""),
    "macd_hist":     ("MACD Histogram",       _pine_macd_hist, "Just the MACD histogram — green bars growing above zero = strengthening uptrend, red bars growing below = strengthening downtrend.", "momentum", "Add as a separate panel. Growing green bars = bullish momentum building. Shrinking green bars = momentum fading. Growing red bars = bearish pressure. First green bar after red = early bullish signal. Cleaner than the full MACD for momentum-only analysis.", ""),
    "volprofile":    ("Volume Profile",       lambda: """//@version=6\nindicator("Volume Profile (Session)", overlay=true, max_bars_back=500)\nint bins = input.int(20, "Bins", minval=5, maxval=50)\nfloat hi = ta.highest(high, 100)\nfloat lo = ta.lowest(low,  100)\nfloat binSz = (hi - lo) / bins\nif barstate.islast\n    for i = 0 to bins - 1\n        float blo = lo + i * binSz\n        float bhi = blo + binSz\n        float vol = 0.0\n        for j = 0 to 99\n            if high[j] >= blo and low[j] <= bhi\n                vol += volume[j]\n        int w = int(vol / ta.highest(vol, 1) * 15)\n        color bc = bhi > close ? color.new(color.red, 60) : color.new(color.green, 60)\n        box.new(bar_index - w, blo, bar_index, bhi, border_color=na, bgcolor=bc)\n""", "Horizontal bars showing where the most volume traded over the last 100 bars. High volume = key S/R.", "volume", "Paste onto your price chart. Wide green bars below price = strong support (buyers were active there). Wide red bars above = strong resistance. The widest bar is the Point of Control (POC) — the most traded price. Works best on daily and 4h charts.", ""),
    "stoch":         ("Stochastic Oscillator", lambda: """//@version=6\nindicator("Stochastic Oscillator", overlay=false)\nint kLen   = input.int(14, "%K Length")\nint smooth = input.int(3,  "%K Smooth")\nint dLen   = input.int(3,  "%D Length")\nfloat k = ta.sma(ta.stoch(close, high, low, kLen), smooth)\nfloat d = ta.sma(k, dLen)\nplot(k, "%K", color=color.new(color.blue,   0), linewidth=2)\nplot(d, "%D", color=color.new(color.orange, 0), linewidth=1)\nhline(80, "OB", color=color.new(color.red,   50), linestyle=hline.style_dashed)\nhline(20, "OS", color=color.new(color.green, 50), linestyle=hline.style_dashed)\nhline(50, "Mid", color=color.new(color.gray, 60), linestyle=hline.style_dotted)\nbgcolor(k > 80 ? color.new(color.red, 92) : k < 20 ? color.new(color.green, 92) : na)\n""", "Classic %K/%D oscillator. Above 80 = overbought, below 20 = oversold. One of the most widely used indicators.", "momentum", "Add as a separate panel. %K (blue) crossing above %D (orange) in the oversold zone (below 20) = buy signal. Crossing below %D in overbought (above 80) = sell signal. Red background = overbought zone, green = oversold. Best used with a trend filter — only take buy signals in uptrends.", ""),
}

CATEGORIES = {
    "all":        "All",
    "trend":      "Trend",
    "momentum":   "Momentum",
    "volume":     "Volume",
    "volatility": "Volatility",
    "smallcap":   "Small Cap",
    "patterns":   "Patterns",
}


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    ref = request.args.get("ref", "")
    if request.method == "POST":
        username   = request.form.get("username", "").strip()
        password   = request.form.get("password", "")
        email      = request.form.get("email", "").strip()
        ref        = request.form.get("ref", "").strip()
        hcaptcha_token = request.form.get("h-captcha-response", "")
        if not hcaptcha_token:
            error = "Please complete the CAPTCHA."
        elif not username or not password or not email:
            error = "Username, email, and password are required."
        elif "@" not in email or "." not in email:
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            # Verify hCaptcha token
            import urllib.request, urllib.parse
            try:
                data = urllib.parse.urlencode({
                    "secret":   HCAPTCHA_SECRET_KEY,
                    "response": hcaptcha_token,
                    "remoteip": request.remote_addr,
                    "sitekey":  HCAPTCHA_SITE_KEY,
                }).encode()
                req = urllib.request.Request("https://api.hcaptcha.com/siteverify",
                                             data=data, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result = __import__("json").loads(resp.read())
                if not result.get("success"):
                    error = "CAPTCHA verification failed. Please try again."
            except Exception:
                error = "CAPTCHA check failed. Please try again."
            if not error:
                try:
                    ref_code = secrets.token_urlsafe(6).upper()
                    _run(
                        "INSERT INTO users (username, pw_hash, email, referral_code, referred_by) VALUES (%s, %s, %s, %s, %s)",
                        (username, generate_password_hash(password), email or None, ref_code, ref or None)
                    )
                    row = _one("SELECT id FROM users WHERE username=%s", (username,))
                    user_id = row["id"]
                    session["user_id"] = user_id
                    session["username"] = username

                    # If referred by someone, give new user 7 days Pro trial + reward referrer
                    if ref:
                        from datetime import datetime, timezone, timedelta
                        referrer = _one("SELECT id, plan FROM users WHERE referral_code=%s", (ref,))
                        if referrer:
                            trial_expires = datetime.now(timezone.utc) + timedelta(days=7)
                            _run("UPDATE users SET plan='pro', plan_expires=%s WHERE id=%s",
                                 (trial_expires, user_id))
                            ref_count = _scalar("SELECT COUNT(*) FROM users WHERE referred_by=%s", (ref,))
                            referrer_plan = referrer["plan"]
                            ref_expires = datetime.now(timezone.utc) + timedelta(days=7)
                            if ref_count >= 4:
                                _run("UPDATE users SET plan='pro', plan_expires=%s WHERE id=%s",
                                     (ref_expires, referrer["id"]))
                            elif ref_count >= 2:
                                _run("UPDATE users SET plan='basic', plan_expires=%s WHERE id=%s",
                                     (ref_expires, referrer["id"]))

                    # Send welcome email
                    if email:
                        send_welcome_email(email, username, ref_code)

                    return redirect("/indicators")
                except psycopg2.errors.UniqueViolation:
                    error = "Username already taken."
    return render_template_string(AUTH_HTML, mode="register", error=error, ref=ref)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        row = _one("SELECT * FROM users WHERE username=%s", (username,))
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


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    success = None
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        if not email:
            error = "Please enter your email address."
        else:
            row = _one("SELECT id, username FROM users WHERE email=%s", (email,))
            if row:
                from datetime import datetime, timezone, timedelta
                token = secrets.token_urlsafe(32)
                expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
                _run(
                    "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (%s, %s, %s)",
                    (token, row["id"], expires_at),
                )
                reset_url = f"{APP_URL}/reset-password?token={token}"
                send_password_reset_email(email, row["username"], reset_url)
            success = "If that email is on file, a reset link is on its way. Check your inbox (and spam folder)."
    return render_template_string(AUTH_HTML, mode="forgot", error=error, success=success)


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    from datetime import datetime, timezone
    token = request.args.get("token", "") or request.form.get("token", "")
    error = None
    if not token:
        return render_template_string(AUTH_HTML, mode="reset", error="Missing reset token. Request a new link below.", success=None, token="")

    row = _one("SELECT * FROM password_reset_tokens WHERE token=%s", (token,))
    if not row or row["used"] or row["expires_at"].replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return render_template_string(AUTH_HTML, mode="reset", error="This reset link is invalid or has expired. Request a new one below.", success=None, token="")

    if request.method == "POST":
        password = request.form.get("password", "")
        if len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            _run("UPDATE users SET pw_hash=%s WHERE id=%s", (generate_password_hash(password), row["user_id"]))
            _run("UPDATE password_reset_tokens SET used=1 WHERE token=%s", (token,))
            return render_template_string(AUTH_HTML, mode="reset", error=None, success="Password updated. You can sign in with your new password now.", token="")

    return render_template_string(AUTH_HTML, mode="reset", error=error, success=None, token=token)



@app.route("/login/google")
def google_login():
    redirect_uri = "https://chartedge.trade/auth/google/callback"
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

        row = _one("SELECT * FROM users WHERE google_id=%s", (google_id,))
        if not row:
            base = name.replace(" ", "_").lower()[:20]
            username = base
            i = 1
            while _one("SELECT 1 FROM users WHERE username=%s", (username,)):
                username = f"{base}{i}"
                i += 1
            _run("INSERT INTO users (username, pw_hash, google_id) VALUES (%s, '', %s)", (username, google_id))
            row = _one("SELECT * FROM users WHERE google_id=%s", (google_id,))

        session["user_id"]  = row["id"]
        session["username"] = row["username"]
        return redirect(request.args.get("next", "/indicators"))
    except Exception:
        log.exception("Google OAuth error")
        return redirect("/login?error=google")


# ── Favorites API ─────────────────────────────────────────────────────────────

@app.route("/api/favorites")
def api_favorites_list():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify([])
    rows = _q("SELECT indicator_key FROM favorites WHERE user_id=%s ORDER BY id DESC", (user_id,))
    return jsonify([{"indicator_key": r["indicator_key"]} for r in rows])


@app.route("/api/favorite/<key>", methods=["POST"])
@login_required
def toggle_favorite(key):
    user_id = session["user_id"]
    existing = _one("SELECT 1 FROM favorites WHERE user_id=%s AND indicator_key=%s", (user_id, key))
    if existing:
        _run("DELETE FROM favorites WHERE user_id=%s AND indicator_key=%s", (user_id, key))
        return jsonify({"favorited": False})
    else:
        _run("INSERT INTO favorites (user_id, indicator_key) VALUES (%s, %s)", (user_id, key))
        return jsonify({"favorited": True})


@app.route("/api/favorite/<key>", methods=["DELETE"])
@login_required
def delete_favorite(key):
    user_id = session["user_id"]
    _run("DELETE FROM favorites WHERE user_id=%s AND indicator_key=%s", (user_id, key))
    return jsonify({"ok": True})


@app.route("/favorites")
@login_required
def favorites_page():
    user_id = session["user_id"]
    rows = _q("SELECT indicator_key FROM favorites WHERE user_id=%s", (user_id,))
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
            _run("INSERT INTO requests (author, name, description) VALUES (%s, %s, %s)", (author, name, desc))
            success = "Request submitted! Thanks."
    reqs = _q("SELECT * FROM requests ORDER BY votes DESC, created ASC")
    user_votes = set()
    if "user_id" in session:
        rows = _q("SELECT request_id FROM request_votes WHERE user_id=%s", (session["user_id"],))
        user_votes = {r["request_id"] for r in rows}
    return render_template_string(REQUEST_HTML,
        reqs=reqs, user_votes=user_votes,
        error=error, success=success, current_user=current_user())


@app.route("/api/request/<int:req_id>/vote", methods=["POST"])
@login_required
def vote_request(req_id):
    user_id = session["user_id"]
    with _tx() as cur:
        cur.execute("SELECT 1 FROM request_votes WHERE user_id=%s AND request_id=%s", (user_id, req_id))
        existing = cur.fetchone()
        if existing:
            cur.execute("DELETE FROM request_votes WHERE user_id=%s AND request_id=%s", (user_id, req_id))
            cur.execute("UPDATE requests SET votes = votes - 1 WHERE id=%s", (req_id,))
            voted = False
        else:
            cur.execute("INSERT INTO request_votes (user_id, request_id) VALUES (%s, %s)", (user_id, req_id))
            cur.execute("UPDATE requests SET votes = votes + 1 WHERE id=%s", (req_id,))
            voted = True
        cur.execute("SELECT votes FROM requests WHERE id=%s", (req_id,))
        row = cur.fetchone()
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
        name, fn, description, cat, how_to, tag = INDICATORS[kind]
        if kind == "vwap":
            pine_code = fn(band1=band1, band2=band2, band3=band3)
        elif kind == "atr":
            pine_code = fn(show_avg=atr_avg)
        else:
            pine_code = fn()
        pine_code = _inject_expiry(pine_code, session.get("username", "user"))

    # Filter indicators for display
    def visible(k, v):
        iname, _, idesc, icat, _, _tag = v
        if category != "all" and icat != category:
            return False
        if search and search not in iname.lower() and search not in idesc.lower():
            return False
        return True

    filtered = {k: v for k, v in INDICATORS.items() if visible(k, v)}

    # Favorites
    is_favorited = False
    if kind and "user_id" in session:
        row = _one("SELECT 1 FROM favorites WHERE user_id=%s AND indicator_key=%s",
                   (session["user_id"], kind))
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

    copy_toast = 1
    if user_id:
        pref = _one("SELECT copy_toast FROM users WHERE id=%s", (user_id,))
        if pref and pref["copy_toast"] is not None:
            copy_toast = pref["copy_toast"]

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
        copy_toast=copy_toast,
        current_user=current_user())


# ── Stripe / Pricing ──────────────────────────────────────────────────────────

@app.route("/subscribe/<plan>")
@login_required
def subscribe(plan):
    billing = request.args.get("billing", "monthly")
    price_map = {
        ("basic",  "monthly"): STRIPE_BASIC_PRICE,
        ("basic",  "yearly"):  STRIPE_BASIC_PRICE_YEARLY,
        ("pro",    "monthly"): STRIPE_PRO_PRICE,
        ("pro",    "yearly"):  STRIPE_PRO_PRICE_YEARLY,
    }
    price_id = price_map.get((plan, billing))
    if not price_id:
        return redirect("/pricing?error=1")
    user_id = session["user_id"]
    # Only offer trial if user hasn't used one before
    row = _one("SELECT trial_used FROM users WHERE id=%s", (user_id,))
    trial_used = row["trial_used"] if row else 1
    trial_days = 7 if not trial_used else 0
    try:
        params = dict(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=APP_URL + "/billing?success=1",
            cancel_url=APP_URL + "/pricing",
            client_reference_id=str(user_id),
        )
        if trial_days:
            params["subscription_data"] = {"trial_period_days": trial_days}
        checkout = stripe.checkout.Session.create(**params)
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
        user_id  = cs.get("client_reference_id")
        customer = cs.get("customer")
        sub_id   = cs.get("subscription")
        if user_id:
            try:
                sub   = stripe.Subscription.retrieve(sub_id)
                price = sub["items"]["data"][0]["price"]["id"]
                plan  = "pro" if price in (STRIPE_PRO_PRICE, STRIPE_PRO_PRICE_YEARLY) else "basic"
            except Exception:
                plan = "basic"
            _run(
                "UPDATE users SET plan=%s, stripe_customer_id=%s, stripe_subscription_id=%s, trial_used=1 WHERE id=%s",
                (plan, customer, sub_id, int(user_id))
            )
            # Referral reward: give the other person same plan for 7 days
            from datetime import datetime, timezone, timedelta
            expires = datetime.now(timezone.utc) + timedelta(days=7)
            buyer = _one("SELECT referred_by, referral_code FROM users WHERE id=%s", (int(user_id),))
            if buyer:
                # If buyer was referred, reward the referrer
                if buyer["referred_by"]:
                    referrer = _one("SELECT id, plan FROM users WHERE referral_code=%s", (buyer["referred_by"],))
                    if referrer:
                        _run("UPDATE users SET plan=%s, plan_expires=%s WHERE id=%s",
                             (plan, expires, referrer["id"]))
                # If buyer referred others, reward referred users who are still free
                if buyer["referral_code"]:
                    referred = _one("SELECT id FROM users WHERE referred_by=%s AND plan='free'",
                                    (buyer["referral_code"],))
                    if referred:
                        _run("UPDATE users SET plan=%s, plan_expires=%s WHERE id=%s",
                             (plan, expires, referred["id"]))

    elif event["type"] == "customer.subscription.deleted":
        customer = event["data"]["object"].get("customer")
        if customer:
            _run(
                "UPDATE users SET plan='free', stripe_subscription_id=NULL, plan_expires=NULL WHERE stripe_customer_id=%s",
                (customer,)
            )

    return "ok", 200


@app.route("/billing")
@login_required
def billing():
    user_id = session["user_id"]
    row = _one("SELECT plan, stripe_customer_id, plan_expires FROM users WHERE id=%s", (user_id,))
    plan        = _get_user_plan(user_id)
    customer_id = row["stripe_customer_id"] if row else None
    plan_expires = row["plan_expires"] if row else None
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
        plan_expires=plan_expires, current_user=current_user())


# BILLING_HTML defined below after _META/_NAV_LINKS/_THEME_JS


# PRICING_HTML defined below after _META/_NAV_LINKS/_THEME_JS


@app.route("/pricing")
def pricing():
    error = request.args.get("error") == "1"
    trial_used = 1
    if session.get("user_id"):
        row = _one("SELECT trial_used FROM users WHERE id=%s", (session["user_id"],))
        trial_used = row["trial_used"] if row else 1
    return render_template_string(PRICING_HTML, current_user=current_user(),
                                  error=error, trial_used=trial_used)


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

    try:
        # Generate a new code
        new_code = None
        new_plan = None
        if request.method == "POST" and request.form.get("action") == "generate":
            new_plan = request.form.get("plan", "pro")
            new_code = secrets.token_urlsafe(8).upper()
            _run("INSERT INTO promo_codes (code, plan) VALUES (%s, %s)", (new_code, new_plan))

        # Delete a code
        if request.method == "POST" and request.form.get("action") == "delete":
            _run("DELETE FROM promo_codes WHERE code=%s", (request.form.get("code"),))
    except Exception as e:
        return f"Action error: {e}", 500

    codes = _q("SELECT * FROM promo_codes ORDER BY created DESC")

    # Dashboard stats
    try:
        total_users   = _scalar("SELECT COUNT(*) FROM users")
        free_users    = _scalar("SELECT COUNT(*) FROM users WHERE plan='free' OR plan IS NULL")
        basic_users   = _scalar("SELECT COUNT(*) FROM users WHERE plan='basic'")
        pro_users     = _scalar("SELECT COUNT(*) FROM users WHERE plan='pro'")
        new_today     = _scalar("SELECT COUNT(*) FROM users WHERE created::date = CURRENT_DATE")
        new_week      = _scalar("SELECT COUNT(*) FROM users WHERE created >= NOW() - INTERVAL '7 days'")
        total_copies  = _scalar("SELECT COALESCE(SUM(count), 0) FROM copy_log")
        copies_today  = _scalar("SELECT COALESCE(SUM(count), 0) FROM copy_log WHERE date=CURRENT_DATE::text")
        recent_users  = _q("SELECT username, plan, email, created FROM users ORDER BY created DESC LIMIT 10")
    except Exception as e:
        log.error("Studio stats error: %s", e)
        return f"Stats error: {e}", 500

    return render_template_string(ADMIN_CODES_HTML,
        codes=codes, new_code=new_code, new_plan=new_plan,
        total_users=total_users, free_users=free_users, basic_users=basic_users, pro_users=pro_users,
        new_today=new_today, new_week=new_week,
        total_copies=total_copies, copies_today=copies_today,
        recent_users=recent_users,
        current_user=current_user())


@app.route("/redeem", methods=["GET", "POST"])
@login_required
def redeem():
    message = None
    error = None
    if request.method == "POST":
        code = request.form.get("code", "").strip().upper()
        user_id = session["user_id"]
        row = _one("SELECT * FROM promo_codes WHERE code=%s", (code,))
        if not row:
            error = "Invalid code."
        elif row["used"]:
            error = "This code has already been used."
        else:
            plan = row["plan"]
            _run("UPDATE users SET plan=%s WHERE id=%s", (plan, user_id))
            _run("UPDATE promo_codes SET used=1, used_by=%s WHERE code=%s", (user_id, code))
            message = f"Success! Your account has been upgraded to {plan.upper()}."
    return render_template_string(REDEEM_HTML, message=message, error=error, current_user=current_user())


@app.route("/refer")
@login_required
def refer():
    user_id = session["user_id"]
    row = _one("SELECT referral_code, plan FROM users WHERE id=%s", (user_id,))
    ref_code = row["referral_code"] if row else None
    if not ref_code:
        ref_code = secrets.token_urlsafe(6).upper()
        _run("UPDATE users SET referral_code=%s WHERE id=%s", (ref_code, user_id))
    referral_count = _scalar("SELECT COUNT(*) FROM users WHERE referred_by=%s", (ref_code,))
    referral_link = f"{APP_URL}/register?ref={ref_code}"
    return render_template_string(REFER_HTML,
        ref_code=ref_code, referral_link=referral_link,
        referral_count=referral_count, current_user=current_user())


@app.route("/profile/update", methods=["POST"])
@login_required
def profile_update():
    import base64
    user_id = session["user_id"]
    error = None

    new_username = request.form.get("username", "").strip()
    if new_username and new_username != session.get("username", ""):
        existing = _one("SELECT id FROM users WHERE username=%s AND id!=%s", (new_username, user_id))
        if existing:
            error = "Username already taken."
        elif len(new_username) < 3 or len(new_username) > 30:
            error = "Username must be 3–30 characters."
        else:
            _run("UPDATE users SET username=%s WHERE id=%s", (new_username, user_id))
            session["username"] = new_username

    new_email = request.form.get("email", "").strip()
    if new_email:
        if "@" not in new_email or "." not in new_email or len(new_email) > 120:
            error = error or "Please enter a valid email address."
        else:
            _run("UPDATE users SET email=%s WHERE id=%s", (new_email, user_id))

    pic_file = request.files.get("profile_pic")
    if pic_file and pic_file.filename:
        data = pic_file.read(2 * 1024 * 1024)  # max 2MB
        mime = pic_file.content_type or "image/jpeg"
        b64 = base64.b64encode(data).decode()
        data_url = f"data:{mime};base64,{b64}"
        _run("UPDATE users SET profile_pic=%s WHERE id=%s", (data_url, user_id))

    if error:
        session["profile_error"] = error
    return redirect("/me")


@app.route("/profile")
@login_required
def profile():
    return redirect("/me")


ADMIN_LOGIN_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head><meta charset="UTF-8"><title>Admin · ChartEdge.trade</title>
<style>
  :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);display:flex;align-items:center;justify-content:center;min-height:100vh;}
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
<head><meta charset="UTF-8"><title>Studio · ChartEdge.trade</title>
<style>
  :root{--bg:#0d1117;--bg2:#161b22;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;--green:#3fb950;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);padding:40px 24px;}
  h1{margin-bottom:6px;font-size:1.6rem;} h1 span{color:var(--accent);}
  h2{font-size:1rem;color:var(--muted);font-weight:500;margin:28px 0 14px;}
  .stats{display:flex;gap:14px;flex-wrap:wrap;max-width:860px;margin-bottom:8px;}
  .stat{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:18px 22px;flex:1;min-width:120px;}
  .stat .num{font-size:1.8rem;font-weight:800;color:var(--accent);}
  .stat .lbl{font-size:.75rem;color:var(--muted);margin-top:2px;}
  .stat.green .num{color:var(--green);}
  .card{background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px;max-width:860px;}
  .gen-form{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
  select,button{padding:9px 14px;border-radius:6px;font-size:.9rem;cursor:pointer;}
  select{background:var(--bg);border:1px solid var(--border);color:var(--text);}
  .btn-gen{background:var(--accent);color:#fff;border:none;}
  .new-code{background:#1f2d1f;border:1px solid var(--green);border-radius:8px;padding:14px 18px;margin-bottom:16px;max-width:860px;}
  .new-code strong{font-size:1.3rem;letter-spacing:2px;color:var(--green);}
  table{width:100%;max-width:860px;border-collapse:collapse;font-size:.85rem;}
  th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--border);}
  th{color:var(--muted);font-weight:500;font-size:.78rem;text-transform:uppercase;}
  .badge{padding:2px 8px;border-radius:10px;font-size:.72rem;font-weight:600;}
  .badge-pro{background:#1f2d1f;color:var(--green);} .badge-basic{background:#1f3a5f;color:var(--accent);} .badge-free{background:var(--bg);color:var(--muted);border:1px solid var(--border);}
  .used{opacity:.45;} .btn-del{background:none;border:1px solid #f85149;color:#f85149;padding:3px 9px;border-radius:6px;font-size:.75rem;cursor:pointer;}
  .section-title{font-size:.95rem;font-weight:600;margin-bottom:12px;color:var(--text);}
</style></head>
<body>
<h1>ChartEdge.trade <span>Studio</span></h1>
<p style="color:var(--muted);font-size:.85rem;margin-bottom:24px;">Admin dashboard · promo codes · user management</p>

<!-- Stats -->
<h2>Overview</h2>
<div class="stats">
  <div class="stat"><div class="num">{{ total_users }}</div><div class="lbl">Total Users</div></div>
  <div class="stat green"><div class="num">{{ new_today }}</div><div class="lbl">New Today</div></div>
  <div class="stat green"><div class="num">{{ new_week }}</div><div class="lbl">New This Week</div></div>
  <div class="stat"><div class="num">{{ free_users }}</div><div class="lbl">Free</div></div>
  <div class="stat"><div class="num">{{ basic_users }}</div><div class="lbl">Basic</div></div>
  <div class="stat"><div class="num">{{ pro_users }}</div><div class="lbl">Pro</div></div>
  <div class="stat"><div class="num">{{ copies_today }}</div><div class="lbl">Copies Today</div></div>
  <div class="stat"><div class="num">{{ total_copies }}</div><div class="lbl">Total Copies</div></div>
</div>

<!-- Recent users -->
<h2>Recent Signups</h2>
<table>
  <tr><th>Username</th><th>Plan</th><th>Email</th><th>Joined</th></tr>
  {% for u in recent_users %}
  <tr>
    <td>{{ u.username }}</td>
    <td><span class="badge badge-{{ u.plan or 'free' }}">{{ (u.plan or 'free')|upper }}</span></td>
    <td style="color:var(--muted)">{{ u.email or '—' }}</td>
    <td style="color:var(--muted)">{{ u.created|string|truncate(10, killwords=True, end='') }}</td>
  </tr>
  {% endfor %}
</table>

<!-- Promo codes -->
<h2>Promo Codes</h2>
{% if new_code %}
<div class="new-code">New code — share this:<br><br><strong>{{ new_code }}</strong> &nbsp;·&nbsp; {{ new_plan|upper }}</div>
{% endif %}
<div class="card">
  <form method="POST" class="gen-form">
    <input type="hidden" name="action" value="generate">
    <select name="plan"><option value="pro">Pro</option><option value="basic">Basic</option></select>
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
    <td>{{ c.created|string|truncate(10, killwords=True, end='') }}</td>
    <td>{% if not c.used %}
      <form method="POST" style="display:inline">
        <input type="hidden" name="action" value="delete">
        <input type="hidden" name="code" value="{{ c.code }}">
        <button type="submit" class="btn-del">Delete</button>
      </form>{% endif %}</td>
  </tr>
  {% endfor %}
  {% if not codes %}<tr><td colspan="5" style="color:var(--muted)">No codes yet.</td></tr>{% endif %}
</table>
</body></html>"""


# REDEEM_HTML defined below after _META/_NAV_LINKS/_THEME_JS

# ── Watchlist ─────────────────────────────────────────────────────────────────



# ── Ratings API ──────────────────────────────────────────────────────────────

def _get_ratings(keys: list[str], user_id: int | None) -> dict:
    """Return {key: {"ups": N, "downs": N, "user_vote": 0/1/-1}} for each key."""
    result = {k: {"ups": 0, "downs": 0, "user_vote": 0} for k in keys}
    if not keys:
        return result
    placeholders = ",".join(["%s"] * len(keys))
    rows = _q(
        f"SELECT indicator_key, vote FROM indicator_ratings WHERE indicator_key IN ({placeholders})", keys
    )
    for r in rows:
        if r["vote"] == 1:
            result[r["indicator_key"]]["ups"] += 1
        else:
            result[r["indicator_key"]]["downs"] += 1
    if user_id:
        urows = _q(
            f"SELECT indicator_key, vote FROM indicator_ratings WHERE user_id=%s AND indicator_key IN ({placeholders})",
            [user_id] + keys
        )
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
    existing = _one("SELECT vote FROM indicator_ratings WHERE user_id=%s AND indicator_key=%s", (user_id, key))
    if existing and existing["vote"] == vote_val:
        _run("DELETE FROM indicator_ratings WHERE user_id=%s AND indicator_key=%s", (user_id, key))
        user_vote = 0
    else:
        _run("""
            INSERT INTO indicator_ratings (user_id, indicator_key, vote) VALUES (%s, %s, %s)
            ON CONFLICT (user_id, indicator_key) DO UPDATE SET vote=%s
        """, (user_id, key, vote_val, vote_val))
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
                def _first(val):
                    return val[0] if isinstance(val, list) and val else val

                eps_est = _first(cal.get("EPS Estimate") or cal.get("Earnings Average") or cal.get("epsEstimate"))
                rev_est = _first(cal.get("Revenue Estimate") or cal.get("Revenue Average") or cal.get("revenueEstimate"))

                import math
                def _is_num(v):
                    try: return v is not None and not math.isnan(float(v))
                    except: return False

                results.append({
                    "ticker":   ticker,
                    "date":     earn_date.strftime("%b %d, %Y"),
                    "date_ord": earn_date.toordinal(),
                    "eps_est":  (f"-${abs(float(eps_est)):.2f}" if float(eps_est) < 0 else f"${float(eps_est):.2f}") if _is_num(eps_est) else "—",
                    "rev_est":  f"${float(rev_est)/1e9:.1f}B" if _is_num(rev_est) else "—",
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


# ── Backtester ────────────────────────────────────────────────────────────────

@app.route("/backtest")
@login_required
def backtest_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=backtest")
    return render_template_string(BACKTEST_HTML, current_user=current_user())


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    if not session.get("user_id"):
        return jsonify({"ok": False}), 401
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"ok": False, "error": "Pro required"}), 403

    import math as _math
    import numpy as _np
    import pandas as _pd
    import yfinance as _yf

    def _sf(v):
        """Safe float — returns None for NaN/inf."""
        try:
            f = float(v)
            return None if (_math.isnan(f) or _math.isinf(f)) else round(f, 4)
        except Exception:
            return None

    data = request.get_json(silent=True) or {}
    ticker   = str(data.get("ticker", "AAPL")).upper().strip()
    strategy = str(data.get("strategy", "rsi")).lower().strip()
    period   = str(data.get("period", "1y")).lower().strip()
    params   = data.get("params", {}) or {}

    if period not in ("1y", "2y", "5y"):
        period = "1y"
    if strategy not in ("rsi", "macd", "bbands", "ema_cross", "sma_cross", "uvol"):
        return jsonify({"ok": False, "error": "Unknown strategy"}), 400

    try:
        raw = _yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if hasattr(raw.columns, "levels"):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.dropna(subset=["Close", "Open", "High", "Low"])
        if len(raw) < 5:
            return jsonify({"ok": False, "error": "Not enough data"}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    closes  = raw["Close"].astype(float)
    volumes = raw["Volume"].astype(float) if "Volume" in raw.columns else _pd.Series([0.0]*len(raw), index=raw.index)
    dates  = [str(d)[:10] for d in raw.index]
    n      = len(closes)

    # ── compute signals ───────────────────────────────────────────────────────
    signals = [0] * n   # +1 = buy, -1 = sell, 0 = hold

    if strategy == "rsi":
        rsi_period = int(params.get("rsi_period", 14))
        oversold   = float(params.get("oversold",   30))
        overbought = float(params.get("overbought",  70))
        delta = closes.diff()
        gain  = delta.clip(lower=0).rolling(rsi_period).mean()
        loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
        rs    = gain / loss.replace(0, _np.nan)
        rsi   = 100 - (100 / (1 + rs))
        rsi   = rsi.fillna(50)
        for i in range(1, n):
            if rsi.iloc[i-1] >= oversold > rsi.iloc[i]:
                signals[i] = 1    # cross below oversold → wait for bounce up
            elif rsi.iloc[i-1] <= oversold < rsi.iloc[i]:
                signals[i] = 1    # RSI crosses above oversold (buy)
            elif rsi.iloc[i-1] <= overbought < rsi.iloc[i]:
                signals[i] = -1   # RSI crosses above overbought (sell)
        # re-do: standard logic — buy cross UP through oversold, sell cross UP through overbought
        signals = [0] * n
        for i in range(1, n):
            prev, curr = float(rsi.iloc[i-1]), float(rsi.iloc[i])
            if prev <= oversold and curr > oversold:
                signals[i] = 1
            elif prev <= overbought and curr > overbought:
                signals[i] = -1

    elif strategy == "macd":
        fast   = int(params.get("fast",   12))
        slow   = int(params.get("slow",   26))
        signal_p = int(params.get("signal",  9))
        ema_fast = closes.ewm(span=fast,   adjust=False).mean()
        ema_slow = closes.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_p, adjust=False).mean()
        for i in range(1, n):
            pm, ps = float(macd_line.iloc[i-1]), float(signal_line.iloc[i-1])
            cm, cs = float(macd_line.iloc[i]),   float(signal_line.iloc[i])
            if pm <= ps and cm > cs:
                signals[i] = 1
            elif pm >= ps and cm < cs:
                signals[i] = -1

    elif strategy == "bbands":
        bb_period = int(params.get("period", 20))
        bb_std    = float(params.get("std",  2.0))
        rolling_mean = closes.rolling(bb_period).mean()
        rolling_std  = closes.rolling(bb_period).std()
        upper = rolling_mean + bb_std * rolling_std
        lower = rolling_mean - bb_std * rolling_std
        for i in range(1, n):
            pc, cc = float(closes.iloc[i-1]), float(closes.iloc[i])
            pl, cl = float(lower.iloc[i-1])  if not _math.isnan(float(lower.iloc[i-1]))  else cc, float(lower.iloc[i])  if not _math.isnan(float(lower.iloc[i]))  else cc
            pu, cu = float(upper.iloc[i-1])  if not _math.isnan(float(upper.iloc[i-1]))  else cc, float(upper.iloc[i])  if not _math.isnan(float(upper.iloc[i]))  else cc
            if pc >= pl and cc < cl:
                signals[i] = 1
            elif pc <= pu and cc > cu:
                signals[i] = -1

    elif strategy == "ema_cross":
        fast_p = int(params.get("fast",  9))
        slow_p = int(params.get("slow", 21))
        ema_f = closes.ewm(span=fast_p, adjust=False).mean()
        ema_s = closes.ewm(span=slow_p, adjust=False).mean()
        for i in range(1, n):
            pf, ps2 = float(ema_f.iloc[i-1]), float(ema_s.iloc[i-1])
            cf, cs2 = float(ema_f.iloc[i]),   float(ema_s.iloc[i])
            if pf <= ps2 and cf > cs2:
                signals[i] = 1
            elif pf >= ps2 and cf < cs2:
                signals[i] = -1

    elif strategy == "sma_cross":
        fast_p = int(params.get("fast",  50))
        slow_p = int(params.get("slow", 200))
        sma_f = closes.rolling(fast_p).mean()
        sma_s = closes.rolling(slow_p).mean()
        for i in range(1, n):
            pf = float(sma_f.iloc[i-1]) if not _pd.isna(sma_f.iloc[i-1]) else None
            ps2 = float(sma_s.iloc[i-1]) if not _pd.isna(sma_s.iloc[i-1]) else None
            cf = float(sma_f.iloc[i])   if not _pd.isna(sma_f.iloc[i])   else None
            cs2 = float(sma_s.iloc[i])   if not _pd.isna(sma_s.iloc[i])   else None
            if None in (pf, ps2, cf, cs2):
                continue
            if pf <= ps2 and cf > cs2:
                signals[i] = 1
            elif pf >= ps2 and cf < cs2:
                signals[i] = -1

    elif strategy == "uvol":
        # Unusual Volume: buy when volume spikes above multiplier × avg, sell after hold_days
        vol_period  = int(params.get("vol_period", 20))
        multiplier  = float(params.get("multiplier", 2.0))
        hold_days   = int(params.get("hold_days", 5))
        avg_vol = volumes.rolling(vol_period).mean()
        hold_until = -1
        for i in range(vol_period, n):
            avg = float(avg_vol.iloc[i]) if not _pd.isna(avg_vol.iloc[i]) else 0
            vol = float(volumes.iloc[i])
            if not (hold_until >= 0):
                if avg > 0 and vol >= multiplier * avg:
                    signals[i] = 1
                    hold_until = i + hold_days
            else:
                if i >= hold_until:
                    signals[i] = -1
                    hold_until = -1

    # ── simulate trades ───────────────────────────────────────────────────────
    INITIAL = 10000.0
    cash    = INITIAL
    shares  = 0.0
    in_pos  = False
    trades  = []
    equity  = []

    bh_shares = INITIAL / float(closes.iloc[0]) if float(closes.iloc[0]) > 0 else 0.0

    for i in range(n):
        price = float(closes.iloc[i])
        sig   = signals[i]

        if sig == 1 and not in_pos and cash > 0:
            shares   = cash / price
            cash     = 0.0
            in_pos   = True
            trades.append({"date": dates[i], "type": "buy", "price": _sf(price), "shares": _sf(shares)})
        elif sig == -1 and in_pos and shares > 0:
            cash     = shares * price
            shares   = 0.0
            in_pos   = False
            trades.append({"date": dates[i], "type": "sell", "price": _sf(price), "shares": _sf(cash / price if price > 0 else 0)})

        port_val = cash + shares * price
        bh_val   = bh_shares * price
        equity.append({"date": dates[i], "value": _sf(port_val), "bh": _sf(bh_val)})

    # final portfolio value (if still in position)
    final_price = float(closes.iloc[-1])
    final_val   = cash + shares * final_price
    bh_final    = bh_shares * final_price

    total_return = _sf((final_val - INITIAL) / INITIAL * 100)
    bh_return    = _sf((bh_final  - INITIAL) / INITIAL * 100)

    # win rate
    buy_prices, wins, losses = [], 0, 0
    for t in trades:
        if t["type"] == "buy":
            buy_prices.append(float(t["price"]) if t["price"] else 0)
        elif t["type"] == "sell" and buy_prices:
            bp = buy_prices.pop()
            sp = float(t["price"]) if t["price"] else 0
            if sp > bp:
                wins += 1
            else:
                losses += 1
    total_closed = wins + losses
    win_rate = _sf(wins / total_closed * 100) if total_closed > 0 else None

    # max drawdown on equity curve
    vals = [e["value"] for e in equity if e["value"] is not None]
    max_dd = 0.0
    if vals:
        peak = vals[0]
        for v in vals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

    # Sharpe ratio (annualised, daily returns, rf=0)
    sharpe = None
    if len(vals) > 2:
        daily_rets = []
        for j in range(1, len(vals)):
            if vals[j-1] > 0:
                daily_rets.append((vals[j] - vals[j-1]) / vals[j-1])
        if daily_rets:
            arr = _np.array(daily_rets, dtype=float)
            std = float(_np.std(arr, ddof=1))
            if std > 0:
                sharpe = _sf(float(_np.mean(arr)) / std * _math.sqrt(252))

    return jsonify({
        "ok":           True,
        "ticker":       ticker,
        "strategy":     strategy,
        "total_return": total_return,
        "bh_return":    bh_return,
        "win_rate":     win_rate,
        "num_trades":   len(trades),
        "max_drawdown": _sf(max_dd),
        "sharpe":       sharpe,
        "equity_curve": equity,
        "trades":       trades,
    })


# ── Economic Calendar ────────────────────────────────────────────────────────

_econ_cache = {"data": None, "ts": 0}

@app.route("/economic")
def economic_page():
    return render_template_string(ECONOMIC_HTML, current_user=current_user())

@app.route("/api/economic")
def economic_api():
    import requests as _req, time as _time
    now = _time.time()
    if _econ_cache["data"] and now - _econ_cache["ts"] < 3600:
        return jsonify(_econ_cache["data"])
    if not FINNHUB_API_KEY:
        return jsonify({"error": "FINNHUB_API_KEY not configured"}), 503
    try:
        from datetime import date, timedelta
        today    = date.today()
        # Start from Monday of this week so "This Week" filter shows all events
        week_start = today - timedelta(days=today.weekday())  # weekday() 0=Mon
        to    = today + timedelta(days=30)
        url   = (f"https://finnhub.io/api/v1/calendar/economic"
                 f"?from={week_start.isoformat()}&to={to.isoformat()}&token={FINNHUB_API_KEY}")
        r = _req.get(url, timeout=10)
        raw = r.json().get("economicCalendar", [])
        # Filter to US events and sort by time
        events = []
        for ev in raw:
            if ev.get("country", "").upper() != "US":
                continue
            events.append({
                "event":    ev.get("event", ""),
                "time":     ev.get("time", ""),
                "impact":   ev.get("impact", "").lower(),
                "prev":     ev.get("prev"),
                "estimate": ev.get("estimate"),
                "actual":   ev.get("actual"),
                "unit":     ev.get("unit", ""),
            })
        events.sort(key=lambda x: x["time"])
        result = {"events": events}
        _econ_cache["data"] = result
        _econ_cache["ts"]   = now
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Dividends Calendar ───────────────────────────────────────────────────────

_DIVIDEND_TICKERS = [
    # Dividend Aristocrats & high-yield blue chips
    "AAPL","MSFT","JNJ","PG","KO","PEP","WMT","MCD","V","MA","JPM","BAC","WFC",
    "XOM","CVX","COP","VZ","T","IBM","INTC","QCOM","TXN","AVGO",
    "MO","PM","ABBV","PFE","MRK","AMGN","ABT","LLY","TMO",
    "HD","LOW","TGT","COST","NKE","SBUX",
    "GS","MS","BLK","AXP","MMC","TRV",
    "CAT","HON","GE","UPS","FDX","LMT","RTX","NOC","GD",
    "NEE","DUK","SO","D","AEP","EXC",
    "AMT","PLD","O","SPG","EQIX","CCI","WPC",
    "LIN","APD","SHW","ECL",
    # Additional dividend payers
    "USB","PNC","TFC","COF","RF","FITB","KEY","HBAN","CFG","MTB",  # regional banks
    "MET","PRU","AFL","ALL","CB","PGR","HIG","UNM","LNC",           # insurance
    "ENB","ET","EPD","MMP","KMI","OKE","WMB","TRGP","PAA",          # midstream energy/MLPs
    "BP","SHEL","TOT","SLB","HAL","BKR",                            # energy
    "CSCO","HPQ","HPE","ORCL","PAYX","ADP",                         # tech dividends
    "CMI","EMR","ETN","PH","ITW","MMM","ROK","IR","XYL",            # industrials
    "ADM","BG","MOS","NTR","CF",                                     # agriculture/materials
    "VFC","RL","HAS","MAT","CLX","CHD","CPB","GIS","K","SJM","MKC", # consumer staples
    "MDT","SYK","BSX","BDX","ZBH","EW","BAX","RMD","HOLX",          # medical devices
    "WPC","NNN","STAG","LTC","OHI","IIPR","HR","MPW","GOOD",        # more REITs
    "BX","KKR","APO","ARCC","MAIN","BXSL","GBDC","HTGC","TPVG",    # BDCs / alternatives
    # Income ETFs
    "SPY","QQQ","DIA","IWM","VYM","SCHD","HDV","DVY","JEPI","JEPQ",
    "DIVO","PFF","PGX","PFFD","SDOG","NOBL","SDY","VIG","DGRO","IDV",
    "FDVV","SPYD","RDVY","XYLD","QYLD","RYLD","NUSI","QQQX",
]

import threading as _div_threading
import time as _div_time
_div_cache = {"data": None, "ts": 0}
_div_fetching = False

def _fetch_dividends():
    import yfinance as yf
    from datetime import date, timedelta, datetime, timezone
    from concurrent.futures import ThreadPoolExecutor

    today  = date.today()
    cutoff = today + timedelta(days=60)

    def _get_one(ticker):
        try:
            info = yf.Ticker(ticker).info
            ex_ts = info.get("exDividendDate")
            if not ex_ts:
                return None
            ex_date = date.fromtimestamp(float(ex_ts))
            if not (today <= ex_date <= cutoff):
                return None
            div_rate  = info.get("dividendRate")   or info.get("lastDividendValue")
            div_yield = info.get("dividendYield")
            name      = info.get("shortName") or info.get("longName") or ticker
            price     = info.get("currentPrice") or info.get("regularMarketPrice")
            freq_map  = {1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}
            freq      = freq_map.get(info.get("dividendFrequency") or 4, "Quarterly")
            # Estimate next payment per period
            per_share = None
            if div_rate and info.get("dividendFrequency"):
                per_share = round(div_rate / info["dividendFrequency"], 4)
            elif div_rate:
                per_share = round(div_rate / 4, 4)

            import math
            def _safe(v):
                try: return None if (v is None or math.isnan(float(v))) else v
                except: return None

            return {
                "ticker":    ticker,
                "name":      name[:30],
                "ex_date":   ex_date.strftime("%b %d, %Y"),
                "ex_ord":    ex_date.toordinal(),
                "per_share": f"${per_share:.4f}".rstrip("0").rstrip(".") if _safe(per_share) else "—",
                "yield_pct": f"{round(float(div_yield)*100, 2)}%" if _safe(div_yield) else "—",
                "freq":      freq,
                "price":     f"${round(float(price),2)}" if _safe(price) else "—",
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(_get_one, _DIVIDEND_TICKERS))

    rows = [r for r in results if r]
    rows.sort(key=lambda x: x["ex_ord"])
    return rows


def _div_refresh_bg():
    global _div_fetching
    if _div_fetching:
        return
    _div_fetching = True
    def _run():
        global _div_fetching
        try:
            data = _fetch_dividends()
            _div_cache["data"] = data
            _div_cache["ts"]   = _div_time.time()
        except Exception:
            pass
        finally:
            _div_fetching = False
    _div_threading.Thread(target=_run, daemon=True).start()

_div_refresh_bg()  # pre-fetch at startup


@app.route("/dividends")
def dividends_page():
    return render_template_string(DIVIDENDS_HTML, current_user=current_user())


@app.route("/api/dividends")
def dividends_api():
    now = _div_time.time()
    if _div_cache["data"] is None or now - _div_cache["ts"] > 3600:
        _div_refresh_bg()
    if _div_cache["data"] is None:
        return jsonify({"loading": True})
    return jsonify({"dividends": _div_cache["data"], "ts": int(_div_cache["ts"])})


# ── Options Flow ──────────────────────────────────────────────────────────────

@app.route("/flow")
@login_required
def flow_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=flow")
    return render_template_string(FLOW_HTML, current_user=current_user())


@app.route("/api/flow")
@login_required
def flow_api():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "upgrade_required"}), 403
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    ticker = request.args.get("ticker", "SPY").upper()[:10]
    try:
        req_exp = request.args.get("exp", "")

        def _fetch():
            t = yf.Ticker(ticker)
            exps = t.options
            if not exps:
                return None, None, None
            exp = req_exp if req_exp in exps else exps[0]
            chain = t.option_chain(exp)
            return exps, exp, chain

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=10)
        except FuturesTimeout:
            executor.shutdown(wait=False)
            return jsonify({"error": "Yahoo Finance timed out — try again in a moment."}), 504
        except Exception as exc:
            executor.shutdown(wait=False)
            return jsonify({"error": str(exc)}), 500
        executor.shutdown(wait=False)

        expirations, exp, chain = result
        if expirations is None:
            return jsonify({"error": "No options data found for " + ticker}), 404

        calls = chain.calls
        puts  = chain.puts

        # Filter params
        sort_by      = (request.args.get("sort", "volume") or "volume").lower()
        type_filter  = (request.args.get("type", "all") or "all").lower()
        strike_filter = (request.args.get("strike", "all") or "all").lower()
        if sort_by not in ("volume", "oi", "premium", "iv"):
            sort_by = "volume"
        if type_filter not in ("all", "calls", "puts"):
            type_filter = "all"
        if strike_filter not in ("all", "itm", "otm"):
            strike_filter = "all"

        def top_contracts(df, kind, n=20):
            df = df.copy()
            df = df[df["volume"].notna() & (df["volume"] > 0)]
            if strike_filter == "itm":
                df = df[df.get("inTheMoney", False) == True]
            elif strike_filter == "otm":
                df = df[df.get("inTheMoney", False) == False]
            if df.empty:
                return []
            df["_premium"] = df["volume"].fillna(0) * df.get("lastPrice", 0).fillna(0)
            sort_col = {"volume": "volume", "oi": "openInterest", "premium": "_premium", "iv": "impliedVolatility"}[sort_by]
            if sort_col not in df.columns:
                sort_col = "volume"
            df = df.sort_values(sort_col, ascending=False).head(n)
            return [
                {
                    "type":           kind,
                    "strike":         round(float(r["strike"]), 2),
                    "expiry":         exp,
                    "volume":         int(r["volume"]),
                    "openInterest":   int(r.get("openInterest", 0) or 0),
                    "iv":             round(float(r.get("impliedVolatility", 0) or 0) * 100, 1),
                    "lastPrice":      round(float(r.get("lastPrice", 0) or 0), 2),
                    "premium":        round(float(r["volume"]) * float(r.get("lastPrice", 0) or 0) * 100, 0),
                    "inTheMoney":     bool(r.get("inTheMoney", False)),
                }
                for _, r in df.iterrows()
            ]

        call_vol = int(calls["volume"].fillna(0).sum())
        put_vol  = int(puts["volume"].fillna(0).sum())
        call_oi  = int(calls["openInterest"].fillna(0).sum())
        put_oi   = int(puts["openInterest"].fillna(0).sum())
        total_vol = call_vol + put_vol or 1

        top_calls = top_contracts(calls, "call") if type_filter != "puts" else []
        top_puts  = top_contracts(puts,  "put")  if type_filter != "calls" else []
        sort_key = {"volume": "volume", "oi": "openInterest", "premium": "premium", "iv": "iv"}[sort_by]
        top = sorted(top_calls + top_puts, key=lambda x: x.get(sort_key, 0), reverse=True)[:20]

        return jsonify({
            "ticker":      ticker,
            "expiry":      exp,
            "expirations": list(expirations[:20]),
            "call_volume": call_vol,
            "put_volume":  put_vol,
            "call_oi":     call_oi,
            "put_oi":      put_oi,
            "put_call_ratio": round(put_vol / (call_vol or 1), 2),
            "call_pct":    round(call_vol / total_vol * 100, 1),
            "put_pct":     round(put_vol  / total_vol * 100, 1),
            "top_contracts": top,
            "sort_by":     sort_by,
            "type_filter": type_filter,
            "strike_filter": strike_filter,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Gamma ─────────────────────────────────────────────────────────────────────

@app.route("/gamma")
@login_required
def gamma_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan == "free":
        return redirect("/pricing?upgrade=gamma")
    return render_template_string(GAMMA_HTML, current_user=current_user())


@app.route("/api/gamma")
@login_required
def gamma_api():
    plan = _get_user_plan(session.get("user_id"))
    if plan == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import yfinance as yf
    import numpy as np
    from scipy.stats import norm
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    ticker = request.args.get("ticker", "SPY").upper()[:10]
    try:
        req_exp = request.args.get("exp", "")

        def _fetch():
            t = yf.Ticker(ticker)
            spot_ = round(float(t.fast_info.last_price), 2)
            exps  = t.options
            if not exps:
                return None, None, None, None
            exp_ = req_exp if req_exp in exps else exps[0]
            chain_ = t.option_chain(exp_)
            return spot_, exps, exp_, chain_

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_fetch)
        try:
            result = future.result(timeout=10)
        except FuturesTimeout:
            executor.shutdown(wait=False)
            return jsonify({"error": "Yahoo Finance timed out — try again in a moment."}), 504
        except Exception as exc:
            executor.shutdown(wait=False)
            return jsonify({"error": str(exc)}), 500
        executor.shutdown(wait=False)

        spot, expirations, exp, chain = result
        if expirations is None:
            return jsonify({"error": "No options data for " + ticker}), 404

        from datetime import date
        today = date.today()
        exp_date = date.fromisoformat(exp)
        T = max((exp_date - today).days / 365, 1/365)
        r = 0.05  # risk-free rate

        def black_scholes_gamma(S, K, T, r, sigma):
            if sigma <= 0 or T <= 0:
                return 0.0
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))

        gex_by_strike = {}
        for _, row in chain.calls.iterrows():
            K  = float(row["strike"])
            iv = float(row.get("impliedVolatility") or 0)
            oi = float(row.get("openInterest") or 0)
            if iv > 0 and oi > 0:
                g = black_scholes_gamma(spot, K, T, r, iv)
                gex_by_strike[K] = gex_by_strike.get(K, 0) + g * oi * 100  # calls add gamma

        for _, row in chain.puts.iterrows():
            K  = float(row["strike"])
            iv = float(row.get("impliedVolatility") or 0)
            oi = float(row.get("openInterest") or 0)
            if iv > 0 and oi > 0:
                g = black_scholes_gamma(spot, K, T, r, iv)
                gex_by_strike[K] = gex_by_strike.get(K, 0) - g * oi * 100  # puts subtract gamma

        if not gex_by_strike:
            return jsonify({"error": "Not enough data to compute GEX"}), 404

        strikes = sorted(gex_by_strike.keys())
        gex     = [round(gex_by_strike[k], 4) for k in strikes]

        # Gamma flip: strike closest to where GEX crosses zero
        flip_strike = None
        for i in range(len(strikes) - 1):
            if gex[i] * gex[i + 1] < 0:
                flip_strike = round((strikes[i] + strikes[i + 1]) / 2, 2)
                break

        # Largest positive and negative GEX strikes
        max_call_gex = max(gex_by_strike, key=lambda k: gex_by_strike[k])
        max_put_gex  = min(gex_by_strike, key=lambda k: gex_by_strike[k])
        total_gex    = round(sum(gex), 4)

        return jsonify({
            "ticker":       ticker,
            "spot":         float(spot),
            "expiry":       exp,
            "expirations":  list(expirations[:20]),
            "strikes":      [float(s) for s in strikes],
            "gex":          [float(v) for v in gex],
            "flip_strike":  float(flip_strike) if flip_strike is not None else None,
            "max_call_gex": float(max_call_gex),
            "max_put_gex":  float(max_put_gex),
            "total_gex":    float(total_gex),
            "positive_gamma": bool(total_gex >= 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Greeks Dashboard ─────────────────────────────────────────────────────────

@app.route("/greeks")
@login_required
def greeks_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan == "free":
        return redirect("/pricing?upgrade=greeks")
    return render_template_string(GREEKS_HTML, current_user=current_user())


@app.route("/api/greeks")
@login_required
def greeks_api():
    plan = _get_user_plan(session.get("user_id"))
    if plan == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import yfinance as yf
    import numpy as np
    from scipy.stats import norm
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    ticker = request.args.get("ticker", "SPY").upper()[:10]
    req_exp = request.args.get("exp", "")

    try:
        def _fetch():
            t = yf.Ticker(ticker)
            spot_ = round(float(t.fast_info.last_price), 2)
            exps_ = t.options
            return t, spot_, list(exps_)

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_fetch)
            try:
                t, spot, expirations = fut.result(timeout=20)
            except FuturesTimeout:
                return jsonify({"error": "yfinance timed out — try again"}), 504

        if not expirations:
            return jsonify({"error": f"No options data found for {ticker}"}), 404

        exp = req_exp if req_exp in expirations else expirations[0]
        chain = t.option_chain(exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # Risk-free rate approximation
        r = 0.045
        from datetime import datetime
        T = max((datetime.strptime(exp, "%Y-%m-%d") - datetime.utcnow()).days / 365.0, 1e-6)

        def bs_greeks(S, K, T, r, iv, is_call):
            if iv <= 0 or T <= 0:
                return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
            d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
            d2 = d1 - iv * np.sqrt(T)
            nd1 = norm.pdf(d1)
            if is_call:
                delta = float(norm.cdf(d1))
                theta = float((-(S * nd1 * iv) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
                rho   = float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
            else:
                delta = float(norm.cdf(d1) - 1)
                theta = float((-(S * nd1 * iv) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
                rho   = float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)
            gamma = float(nd1 / (S * iv * np.sqrt(T)))
            vega  = float(S * nd1 * np.sqrt(T) / 100)
            return {
                "delta": round(delta, 4),
                "gamma": round(gamma, 6),
                "theta": round(theta, 4),
                "vega":  round(vega, 4),
                "rho":   round(rho, 4),
            }

        def _safe_int(v):
            try:
                f = float(v)
                return 0 if np.isnan(f) else int(f)
            except Exception:
                return 0

        def row(r_ser, is_call):
            iv   = float(r_ser.get("impliedVolatility", 0) or 0)
            K    = float(r_ser["strike"])
            mid  = round((float(r_ser.get("bid", 0) or 0) + float(r_ser.get("ask", 0) or 0)) / 2, 2)
            oi   = _safe_int(r_ser.get("openInterest", 0))
            vol  = _safe_int(r_ser.get("volume", 0))
            g    = bs_greeks(spot, K, T, r, iv, is_call)
            return {
                "strike": K,
                "mid":    mid,
                "iv":     round(iv * 100, 1),
                "oi":     oi,
                "volume": vol,
                **g,
            }

        calls_data = [row(r, True)  for _, r in calls.iterrows()]
        puts_data  = [row(r, False) for _, r in puts.iterrows()]

        # ATM row for summary cards
        atm_call = min(calls_data, key=lambda x: abs(x["strike"] - spot), default={})
        atm_put  = min(puts_data,  key=lambda x: abs(x["strike"] - spot), default={})

        return jsonify({
            "ticker":      ticker,
            "spot":        spot,
            "expiry":      exp,
            "expirations": list(expirations[:20]),
            "T_days":      round(T * 365),
            "calls":       calls_data,
            "puts":        puts_data,
            "atm_call":    atm_call,
            "atm_put":     atm_put,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Heatmap ──────────────────────────────────────────────────────────────────

_HEATMAP_SECTORS = {
    "Technology":     ["AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN","INTC","AMAT","MU","CRM","NOW","PANW","CRWD","NET","LRCX","KLAC","SNOW","PLTR"],
    "Communication":  ["GOOGL","META","NFLX","DIS","T","VZ","SNAP","PINS","RDDT","RBLX"],
    "Consumer Disc.": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","ABNB","BKNG","DASH","GM","F","RIVN"],
    "Financials":     ["JPM","BAC","WFC","GS","MS","V","MA","C","BLK","AXP","PYPL","COIN"],
    "Healthcare":     ["UNH","JNJ","LLY","ABBV","PFE","MRK","AMGN","TMO","ABT","ISRG"],
    "Energy":         ["XOM","CVX","COP","SLB","EOG","PSX","MPC","OXY"],
    "Industrials":    ["CAT","BA","GE","RTX","HON","UPS","FDX","LMT","DE"],
    "Consumer Stap.": ["WMT","PG","COST","KO","PEP","PM","MO","CL"],
    "Materials":      ["LIN","APD","SHW","FCX","NEM","AA"],
    "Real Estate":    ["AMT","PLD","EQIX","CCI","SPG","O"],
    "Utilities":      ["NEE","DUK","SO","D","AEP","EXC"],
}

import time as _hm_time
import threading as _hm_threading

_heatmap_cache    = {"data": None, "ts": 0}
_heatmap_fetching = False

# Hardcoded approximate market caps in dollars (used for treemap sizing + display)
_MCAP_STATIC = {
    "AAPL":3.3e12,"MSFT":3.1e12,"NVDA":2.9e12,"GOOGL":2.1e12,"AMZN":2.2e12,
    "META":1.5e12,"TSLA":8.5e11,"AVGO":9.0e11,"ORCL":4.5e11,"AMD":2.5e11,
    "QCOM":1.7e11,"TXN":1.7e11,"INTC":9.0e10,"AMAT":1.8e11,"MU":1.1e11,
    "CRM":2.9e11,"NOW":2.0e11,"PANW":1.2e11,"CRWD":9.0e10,"NET":7.0e10,
    "LRCX":1.0e11,"KLAC":9.0e10,"SNOW":5.0e10,"PLTR":1.5e11,
    "NFLX":3.8e11,"DIS":1.8e11,"T":1.4e11,"VZ":1.6e11,"SNAP":1.5e10,
    "PINS":2.0e10,"RDDT":2.0e10,"RBLX":3.0e10,
    "HD":4.0e11,"MCD":2.3e11,"NKE":9.0e10,"SBUX":9.0e10,"TGT":6.0e10,
    "ABNB":8.0e10,"BKNG":1.4e11,"DASH":5.5e10,"GM":4.5e10,"F":4.0e10,"RIVN":1.2e10,
    "JPM":7.5e11,"BAC":3.3e11,"WFC":2.5e11,"GS":1.9e11,"MS":2.0e11,
    "V":6.0e11,"MA":4.9e11,"C":1.3e11,"BLK":1.4e11,"AXP":2.0e11,
    "PYPL":6.5e10,"COIN":6.0e10,
    "UNH":5.0e11,"JNJ":3.8e11,"LLY":7.5e11,"ABBV":3.2e11,"PFE":1.4e11,
    "MRK":2.8e11,"AMGN":1.6e11,"TMO":2.0e11,"ABT":2.0e11,"ISRG":1.5e11,
    "XOM":4.9e11,"CVX":2.8e11,"COP":1.3e11,"SLB":5.5e10,"EOG":6.0e10,
    "PSX":5.0e10,"MPC":4.5e10,"OXY":4.5e10,
    "CAT":1.9e11,"BA":1.0e11,"GE":2.1e11,"RTX":1.6e11,"HON":1.3e11,
    "UPS":1.0e11,"FDX":6.0e10,"LMT":1.1e11,"DE":1.3e11,
    "WMT":7.0e11,"PG":3.8e11,"COST":4.0e11,"KO":2.6e11,"PEP":2.2e11,
    "PM":1.7e11,"MO":8.0e10,"CL":6.0e10,
    "LIN":2.2e11,"APD":6.0e10,"SHW":9.0e10,"FCX":5.5e10,"NEM":5.5e10,"AA":1.2e10,
    "AMT":9.5e10,"PLD":1.0e11,"EQIX":8.0e10,"CCI":4.5e10,"SPG":6.0e10,"O":5.0e10,
    "NEE":1.3e11,"DUK":7.0e10,"SO":6.5e10,"D":4.0e10,"AEP":4.5e10,"EXC":4.0e10,
}

def _fmt_mcap(v):
    if not v:
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.2f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"

def _fetch_heatmap_data():
    import yfinance as yf
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor

    all_tickers = [t for tickers in _HEATMAP_SECTORS.values() for t in tickers]

    # Fetch prices and market caps in parallel
    def _get_mcap(ticker):
        try:
            mc = yf.Ticker(ticker).fast_info.market_cap
            return ticker, float(mc) if mc else None
        except Exception:
            return ticker, None

    with ThreadPoolExecutor(max_workers=25) as pool:
        price_future = pool.submit(
            lambda: yf.download(all_tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
        )
        mcap_futures = {t: pool.submit(_get_mcap, t) for t in all_tickers}

        raw      = price_future.result()
        mcap_map = {t: f.result()[1] for t, f in mcap_futures.items()}

    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    result = []
    for sector, tickers in _HEATMAP_SECTORS.items():
        for ticker in tickers:
            try:
                col  = closes[ticker] if ticker in closes.columns else None
                if col is None:
                    continue
                vals = col.dropna()
                if len(vals) == 0:
                    continue
                price = round(float(vals.iloc[-1]), 2)
                pct   = round((float(vals.iloc[-1]) - float(vals.iloc[-2])) / float(vals.iloc[-2]) * 100, 2) if len(vals) >= 2 else 0.0
                # Use live market cap, fall back to hardcoded if unavailable
                mc = mcap_map.get(ticker) or _MCAP_STATIC.get(ticker, 1e9)
                result.append({
                    "ticker":     ticker,
                    "sector":     sector,
                    "price":      price,
                    "change_pct": pct,
                    "mcap":       _fmt_mcap(mc),
                    "mcap_num":   mc,
                })
            except Exception:
                continue
    return result

def _heatmap_refresh_bg():
    global _heatmap_fetching
    if _heatmap_fetching:
        return
    _heatmap_fetching = True
    def _run():
        global _heatmap_fetching
        try:
            data = _fetch_heatmap_data()
            _heatmap_cache["data"] = data
            _heatmap_cache["ts"]   = _hm_time.time()
        except Exception:
            pass
        finally:
            _heatmap_fetching = False
    _hm_threading.Thread(target=_run, daemon=True).start()

# Pre-fetch at server start so first visitor gets instant data
_heatmap_refresh_bg()


@app.route("/heatmap")
def heatmap_page():
    return render_template_string(HEATMAP_HTML, current_user=current_user())


@app.route("/api/heatmap")
def heatmap_api():
    now = _hm_time.time()
    # Trigger background refresh if stale (non-blocking)
    if _heatmap_cache["data"] is None or now - _heatmap_cache["ts"] > 300:
        _heatmap_refresh_bg()
    if _heatmap_cache["data"] is None:
        return jsonify({"loading": True})
    return jsonify({"sectors": _HEATMAP_SECTORS, "stocks": _heatmap_cache["data"], "ts": int(_heatmap_cache["ts"])})


# ── Trump Tracker ────────────────────────────────────────────────────────────

_TRUMP_INSTRUMENTS = [
    {"label": "S&P 500 (SPY)",       "ticker": "SPY"},
    {"label": "NASDAQ 100 (QQQ)",    "ticker": "QQQ"},
    {"label": "Dow Jones (DIA)",      "ticker": "DIA"},
    {"label": "Russell 2000 (IWM)",  "ticker": "IWM"},
    {"label": "Trump Media (DJT)",   "ticker": "DJT"},
    {"label": "Bonds 20Y (TLT)",     "ticker": "TLT"},
    {"label": "Gold (GLD)",           "ticker": "GLD"},
    {"label": "Oil (USO)",            "ticker": "USO"},
    {"label": "Energy (XLE)",         "ticker": "XLE"},
    {"label": "Financials (XLF)",    "ticker": "XLF"},
    {"label": "Tech (XLK)",           "ticker": "XLK"},
    {"label": "VIX (UVIX)",           "ticker": "UVIX"},
    {"label": "Bitcoin (IBIT)",       "ticker": "IBIT"},
    {"label": "China (FXI)",          "ticker": "FXI"},
]

def _parse_ts(entry) -> float:
    """Best-effort Unix timestamp from a feedparser entry."""
    import calendar, email.utils, time as _t
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try: return float(calendar.timegm(entry.published_parsed))
        except Exception: pass
    for attr in ("updated_parsed", "created_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try: return float(calendar.timegm(val))
            except Exception: pass
    for attr in ("published", "updated"):
        s = getattr(entry, attr, None)
        if s:
            try: return float(email.utils.mktime_tz(email.utils.parsedate_tz(s)))
            except Exception: pass
    return 0.0


def _rel_time(ts: float) -> str:
    """Return '2h ago', 'Apr 4' etc."""
    import time as _t
    from datetime import datetime, timezone
    if not ts:
        return ""
    diff = _t.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff//60)}m ago"
    if diff < 86400:
        return f"{int(diff//3600)}h ago"
    if diff < 86400 * 2:
        return "yesterday"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{_MONTHS[dt.month-1]} {dt.day}"


def _fetch_trump_news():
    import feedparser, re
    keywords = ["trump", "tariff", "white house", "executive order", "maga", "trade war",
                "mar-a-lago", "truth social", "federal reserve", "doge", "elon musk"]
    articles = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title   = entry.get("title", "") or ""
                summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", "") or "")[:300]
                combined = (title + " " + summary).lower()
                if not any(k in combined for k in keywords):
                    continue
                ts = _parse_ts(entry)
                articles.append({
                    "source":    source,
                    "title":     title,
                    "link":      entry.get("link", "#"),
                    "published": _rel_time(ts),
                    "summary":   summary[:200],
                    "ts":        ts,
                })
        except Exception:
            pass
    articles.sort(key=lambda a: a["ts"], reverse=True)
    return articles[:40]


@app.route("/trump")
@login_required
def trump_page():
    if _get_user_plan(session.get("user_id")) == "free":
        return redirect("/pricing?upgrade=trump")
    return render_template_string(TRUMP_HTML, current_user=current_user())


@app.route("/api/trump/news")
def trump_news_api():
    try:
        return jsonify({"articles": _fetch_trump_news()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trump/chart")
def trump_chart_api():
    import yfinance as yf
    import numpy as np
    ticker = request.args.get("ticker", "SPY").upper()[:10]
    period = request.args.get("period", "1W")

    period_map  = {"1D": ("1d",  "5m"),  "1W": ("5d",  "30m"),
                   "1M": ("1mo", "1d"),  "3M": ("3mo", "1d")}
    yf_period, interval = period_map.get(period, ("5d", "30m"))

    try:
        raw = yf.download(ticker, period=yf_period, interval=interval,
                          progress=False, auto_adjust=True)
        if raw is None or len(raw) < 2:
            return jsonify({"error": "No data"}), 404

        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        closes = raw["Close"].dropna()
        times  = [int(t.timestamp() * 1000) for t in closes.index]
        prices = [round(float(v), 4) for v in closes.values]

        price_now  = prices[-1]
        price_open = prices[0]
        chg_pct    = round((price_now - price_open) / price_open * 100, 2)

        # Multi-period stats
        def _pct(n):
            if len(prices) < n + 1: return None
            return round((prices[-1] - prices[-(n+1)]) / prices[-(n+1)] * 100, 2)

        # Include recent news timestamps for event markers
        try:
            articles = _fetch_trump_news()
            chart_start = times[0]
            chart_end   = times[-1]
            events = []
            for a in articles:
                ts = a.get("ts", 0)
                if ts and chart_start <= ts * 1000 <= chart_end:
                    events.append({"ts": ts * 1000, "title": a.get("title", "")[:60]})
        except Exception:
            events = []

        return jsonify({
            "ticker":    ticker,
            "period":    period,
            "times":     times,
            "prices":    prices,
            "price":     price_now,
            "chg_pct":   chg_pct,
            "events":    events,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Scanner ticker universe (used by Pre-market scanner) ─────────────────────

_SCANNER_TICKERS = [
    # ETFs
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","HYG","XLF","XLK","XLE","XLV","XLI","XLC","XLB","XLU","XLP","XLRE","ARKK","VXX","SQQQ","TQQQ","SPXU","SPXL","UVXY",
    # Mega cap tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL","CRM","ADBE","CSCO","IBM","NOW","SNOW","PLTR","PANW","CRWD","DDOG","NET","ZS","OKTA","FTNT","GTLB",
    # Semiconductors
    "AMD","INTC","MU","QCOM","TXN","AMAT","LRCX","KLAC","ASML","TSM","ARM","SMCI","MRVL","MCHP","ADI","NXPI","ON","WOLF","SWKS","QRVO",
    # Financials
    "JPM","BAC","C","WFC","GS","MS","BLK","AXP","V","MA","PYPL","SQ","COIN","HOOD","SOFI","NU","SCHW","USB","PNC","TFC","COF","DFS","SYF","ALLY",
    # Healthcare
    "JNJ","UNH","PFE","MRK","LLY","ABBV","AMGN","BMY","GILD","REGN","VRTX","MRNA","BNTX","CVS","CI","HUM","MOH","CNC","ISRG","MDT","ABT","TMO","DHR","A","IQV",
    # Consumer
    "AMZN","WMT","TGT","COST","HD","LOW","MCD","SBUX","NKE","LULU","TJX","ROST","DG","DLTR","EBAY","ETSY","W","CHWY","CVNA","KO","PEP","PG","CL","KMB","GIS","HSY","MO","PM",
    # Energy
    "XOM","CVX","COP","EOG","PXD","SLB","HAL","BKR","MPC","VLO","PSX","OXY","DVN","FANG","HES","APA","WMB","KMI","ET","MPLX",
    # Media/Entertainment
    "NFLX","DIS","PARA","WBD","CMCSA","CHTR","FOXA","FOX","RBLX","U","EA","TTWO","ATVI","MTCH","IAC","ZM",
    # Auto/EV
    "F","GM","RIVN","LCID","NIO","LI","XPEV","TM","HMC","STLA",
    # Travel/Leisure
    "ABNB","BKNG","EXPE","UBER","LYFT","DASH","MAR","HLT","CCL","RCL","NCLH","DAL","UAL","AAL","LUV","DKNG","PENN","MGM","LVS","WYNN","CZR",
    # Retail/E-comm
    "SHOP","MELI","SE","GRAB","BABA","JD","PDD","SNAP","PINS","RDDT","APP",
    # Small/mid cap popular
    "GME","AMC","SPCE","BB","NOK","SNDL","TLRY","CGC","ACB",
    # Growth/ARK names
    "ROKU","TDOC","TWLO","DOCU","BILL","HUBS","VEEV","PCTY","PAYC","MDB","ESTC","CFLT",
    # Biotech
    "BIIB","ILMN","ALGN","HOLX","SRPT","BMRN","ALNY","EXEL","ACAD","PTON","HIMS","RXRX",
    # Industrials/Defense
    "BA","LMT","RTX","NOC","GD","LHX","CAT","DE","GE","HON","MMM","UPS","FDX","CSX","UNP","NSC","WM","RSG",
    # Real estate / REITs
    "AMT","PLD","EQIX","CCI","O","SPG","AVB","EQR","MAA","NLY","AGNC","STWD","BXMT",
    # Banks regional
    "ZION","RF","CFG","HBAN","KEY","MTB","FITB","WAL",
    # Crypto miners
    "MSTR","MARA","RIOT","HUT","CLSK","CIFR","WULF","CORZ",
]
# deduplicate preserving order
_seen = set()
_SCANNER_TICKERS = [t for t in _SCANNER_TICKERS if not (t in _seen or _seen.add(t))]


# ── Ticker Tape ──────────────────────────────────────────────────────────────

_TAPE_TICKERS = ["SPY", "AAPL", "NVDA", "TSLA", "MSFT", "META", "AMZN", "QQQ", "AMD", "GOOGL", "BTC-USD", "GC=F"]
_tape_cache   = {"data": None, "ts": 0}

def _fetch_tape():
    import yfinance as yf, time as _time
    out = []
    for sym in _TAPE_TICKERS:
        try:
            t    = yf.Ticker(sym)
            info = t.fast_info
            px   = float(info.last_price or 0)
            prev = float(info.previous_close or px)
            chg  = ((px - prev) / prev * 100) if prev else 0
            out.append({"sym": sym, "price": px, "chg": round(chg, 2)})
        except Exception:
            pass
    return out

@app.route("/api/tickertape")
def tickertape_api():
    import time
    now = time.time()
    if _tape_cache["data"] is None or now - _tape_cache["ts"] > 300:
        try:
            _tape_cache["data"] = _fetch_tape()
            _tape_cache["ts"]   = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"tickers": _tape_cache["data"]})


# ── IPO Calendar ─────────────────────────────────────────────────────────────

_ipo_cache = {"data": None, "ts": 0}

def _fetch_ipos() -> list[dict]:
    import time, requests
    from datetime import datetime, timedelta
    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
    }
    # Fetch current and next month
    today = datetime.today()
    months = [today, today + timedelta(days=32)]
    seen = set()
    for dt in months:
        url = f"https://api.nasdaq.com/api/ipo/calendar?date={dt.strftime('%Y-%m')}"
        try:
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code != 200:
                continue
            payload = r.json()
            rows_data = payload.get("data", {})
            # upcoming: data.upcoming.upcomingTable.rows
            # priced:   data.priced.rows
            section_rows = [
                ("Upcoming", (rows_data.get("upcoming", {}) or {}).get("upcomingTable", {}).get("rows") or []),
                ("Priced",   (rows_data.get("priced", {}) or {}).get("rows") or []),
            ]
            for status, rows in section_rows:
                for row in rows:
                    company = row.get("companyName", "")
                    ticker  = row.get("proposedTickerSymbol", "") or row.get("symbol", "")
                    key     = company + ticker
                    if key in seen:
                        continue
                    seen.add(key)
                    price_range = row.get("proposedSharePrice", "—") or "—"
                    shares      = row.get("sharesOffered", "—") or "—"
                    exchange    = row.get("proposedExchange", "—") or "—"
                    ipo_date    = (row.get("pricedDate", "") or row.get("expectedPriceDate", "")) if status == "Priced" else (row.get("expectedPriceDate", "") or row.get("pricedDate", ""))
                    # parse date for sorting — NASDAQ returns M/D/YYYY (no zero-padding)
                    try:
                        parts = ipo_date.strip().split("/")
                        d = datetime(int(parts[2]), int(parts[0]), int(parts[1]))
                        date_str = d.strftime("%b %d, %Y")
                        date_ord = d.toordinal()
                    except Exception:
                        date_str = ipo_date or "TBD"
                        date_ord = 0
                    # format shares
                    try:
                        sh = float(str(shares).replace(",", ""))
                        shares_fmt = f"{sh/1e6:.1f}M" if sh >= 1e6 else f"{sh/1e3:.0f}K"
                    except Exception:
                        shares_fmt = shares
                    results.append({
                        "company":    company,
                        "ticker":     ticker or "—",
                        "date":       date_str,
                        "date_ord":   date_ord,
                        "price":      price_range,
                        "shares":     shares_fmt,
                        "exchange":   exchange,
                        "status":     status,
                    })
        except Exception:
            continue
    results.sort(key=lambda x: (x["status"] != "Upcoming", x["date_ord"]))
    return results


@app.route("/ipo")
def ipo_page():
    return render_template_string(IPO_HTML, current_user=current_user())


@app.route("/api/ipo")
def ipo_api():
    import time
    now = time.time()
    if _ipo_cache["data"] is None or now - _ipo_cache["ts"] > 3600:
        try:
            _ipo_cache["data"] = _fetch_ipos()
            _ipo_cache["ts"]   = now
        except Exception as e:
            return jsonify({"error": str(e), "ipos": []}), 500
    return jsonify({"ipos": _ipo_cache["data"]})


# ── Crypto Tools ──────────────────────────────────────────────────────────────

_CRYPTO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
_crypto_fg_cache       = {"data": None, "ts": 0}
_crypto_dom_cache      = {"data": None, "ts": 0}
_crypto_hm_cache       = {"data": None, "ts": 0}
_crypto_fund_cache     = {"data": None, "ts": 0}
_crypto_onchain_cache  = {"data": None, "ts": 0}
_crypto_liq_cache      = {}   # keyed by symbol
_crypto_upcoming_cache = {"data": None, "ts": 0}


# ─ Fear & Greed ───────────────────────────────────────────────────────────────

def _fetch_crypto_fg():
    import requests
    r = requests.get("https://api.alternative.me/fng/?limit=30&format=json",
                     headers=_CRYPTO_HEADERS, timeout=8)
    r.raise_for_status()
    return [{"value": int(d["value"]), "label": d["value_classification"],
             "ts": int(d["timestamp"])} for d in r.json()["data"]]

@app.route("/crypto/feargreed")
def crypto_feargreed_page():
    return render_template_string(CRYPTO_FEARGREED_HTML, current_user=current_user())

@app.route("/api/crypto/feargreed")
def crypto_feargreed_api():
    import time
    now = time.time()
    if _crypto_fg_cache["data"] is None or now - _crypto_fg_cache["ts"] > 3600:
        try:
            _crypto_fg_cache["data"] = _fetch_crypto_fg()
            _crypto_fg_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"data": _crypto_fg_cache["data"]})


# ─ BTC Dominance ──────────────────────────────────────────────────────────────

def _fetch_crypto_dom():
    import requests
    r = requests.get("https://api.coingecko.com/api/v3/global",
                     headers=_CRYPTO_HEADERS, timeout=8)
    r.raise_for_status()
    d = r.json()["data"]
    pcts = d.get("market_cap_percentage", {})
    top = sorted(pcts.items(), key=lambda x: -x[1])[:10]
    return {
        "btc": round(pcts.get("btc", 0), 2),
        "eth": round(pcts.get("eth", 0), 2),
        "total_mcap": d.get("total_market_cap", {}).get("usd", 0),
        "total_volume": d.get("total_volume", {}).get("usd", 0),
        "active_coins": d.get("active_cryptocurrencies", 0),
        "top": [{"symbol": s.upper(), "pct": round(p, 2)} for s, p in top],
    }

@app.route("/crypto/dominance")
@login_required
def crypto_dominance_page():
    if _get_user_plan(session.get("user_id")) == "free":
        return redirect("/pricing?upgrade=crypto")
    return render_template_string(CRYPTO_DOMINANCE_HTML, current_user=current_user())

@app.route("/api/crypto/dominance")
def crypto_dominance_api():
    if "user_id" not in session:
        return jsonify({"error": "login_required"}), 401
    if _get_user_plan(session.get("user_id")) == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import time
    now = time.time()
    if _crypto_dom_cache["data"] is None or now - _crypto_dom_cache["ts"] > 1800:
        try:
            _crypto_dom_cache["data"] = _fetch_crypto_dom()
            _crypto_dom_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(_crypto_dom_cache["data"])


# ─ Crypto Heatmap ─────────────────────────────────────────────────────────────

def _fetch_crypto_hm():
    import requests
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
        "&price_change_percentage=24h&sparkline=false",
        headers=_CRYPTO_HEADERS, timeout=10)
    r.raise_for_status()
    out = []
    for c in r.json():
        chg = c.get("price_change_percentage_24h") or 0
        out.append({
            "symbol": (c.get("symbol") or "").upper(),
            "name":   c.get("name", ""),
            "mcap":   c.get("market_cap") or 0,
            "price":  c.get("current_price") or 0,
            "chg":    round(float(chg), 2),
        })
    return out

@app.route("/crypto/heatmap")
def crypto_heatmap_page():
    return render_template_string(CRYPTO_HEATMAP_HTML, current_user=current_user())

@app.route("/api/crypto/heatmap")
def crypto_heatmap_api():
    import time
    now = time.time()
    if _crypto_hm_cache["data"] is None or now - _crypto_hm_cache["ts"] > 1800:
        try:
            _crypto_hm_cache["data"] = _fetch_crypto_hm()
            _crypto_hm_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"coins": _crypto_hm_cache["data"]})


# ─ Funding Rates ──────────────────────────────────────────────────────────────

def _fetch_crypto_funding():
    import requests
    # Hyperliquid — decentralized perps, fully open API, no IP blocks
    r = requests.post("https://api.hyperliquid.xyz/info",
                      json={"type": "metaAndAssetCtxs"},
                      headers={**_CRYPTO_HEADERS, "Content-Type": "application/json"},
                      timeout=10)
    r.raise_for_status()
    payload = r.json()
    universe = payload[0].get("universe", [])
    ctxs     = payload[1]
    out = []
    for meta, ctx in zip(universe, ctxs):
        sym = meta.get("name", "")
        try:
            rate  = float(ctx.get("funding", 0))
            price = float(ctx.get("markPx", 0))
        except (ValueError, TypeError):
            continue
        out.append({
            "symbol": sym,
            "rate":   round(rate * 100, 4),
            "price":  round(price, 4),
        })
    out.sort(key=lambda x: abs(x["rate"]), reverse=True)
    return out[:60]

@app.route("/crypto/funding")
@login_required
def crypto_funding_page():
    if _get_user_plan(session.get("user_id")) == "free":
        return redirect("/pricing?upgrade=crypto")
    return render_template_string(CRYPTO_FUNDING_HTML, current_user=current_user())

@app.route("/api/crypto/funding")
def crypto_funding_api():
    if "user_id" not in session:
        return jsonify({"error": "login_required"}), 401
    if _get_user_plan(session.get("user_id")) == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import time
    now = time.time()
    if _crypto_fund_cache["data"] is None or now - _crypto_fund_cache["ts"] > 300:
        try:
            _crypto_fund_cache["data"] = _fetch_crypto_funding()
            _crypto_fund_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"rates": _crypto_fund_cache["data"]})


# ─ On-chain Metrics ───────────────────────────────────────────────────────────

def _fetch_crypto_onchain():
    import requests
    result = {}
    try:
        r = requests.get("https://api.blockchain.info/stats",
                         headers=_CRYPTO_HEADERS, timeout=8)
        if r.status_code == 200:
            b = r.json()
            result["btc_chain"] = {
                "hash_rate":    round(b.get("hash_rate", 0) / 1e9, 2),
                "difficulty":   b.get("difficulty", 0),
                "n_tx":         b.get("n_tx", 0),
                "fees_btc":     round(b.get("total_fees_btc", 0) / 1e8, 4),
                "block_time":   round(b.get("minutes_between_blocks", 0), 2),
                "blocks_mined": b.get("n_blocks_mined", 0),
            }
    except Exception:
        pass
    try:
        r2 = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&ids=bitcoin,ethereum&order=market_cap_desc"
            "&sparkline=false&price_change_percentage=24h,7d",
            headers=_CRYPTO_HEADERS, timeout=8)
        if r2.status_code == 200:
            for c in r2.json():
                result[c["id"]] = {
                    "price":    c.get("current_price", 0),
                    "mcap":     c.get("market_cap", 0),
                    "vol_24h":  c.get("total_volume", 0),
                    "chg_24h":  round(c.get("price_change_percentage_24h") or 0, 2),
                    "chg_7d":   round(c.get("price_change_percentage_7d_in_currency") or 0, 2),
                    "ath":      c.get("ath", 0),
                    "ath_chg":  round(c.get("ath_change_percentage") or 0, 2),
                    "supply":   c.get("circulating_supply", 0),
                    "max_supply": c.get("max_supply"),
                }
    except Exception:
        pass
    return result

@app.route("/crypto/onchain")
@login_required
def crypto_onchain_page():
    if _get_user_plan(session.get("user_id")) == "free":
        return redirect("/pricing?upgrade=crypto")
    return render_template_string(CRYPTO_ONCHAIN_HTML, current_user=current_user())

@app.route("/api/crypto/onchain")
def crypto_onchain_api():
    if "user_id" not in session:
        return jsonify({"error": "login_required"}), 401
    if _get_user_plan(session.get("user_id")) == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import time
    now = time.time()
    if _crypto_onchain_cache["data"] is None or now - _crypto_onchain_cache["ts"] > 1800:
        try:
            _crypto_onchain_cache["data"] = _fetch_crypto_onchain()
            _crypto_onchain_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(_crypto_onchain_cache["data"])


# ─ Liquidation Map ────────────────────────────────────────────────────────────

def _fetch_crypto_liq(symbol="BTC"):
    import requests, time as _time
    # Hyperliquid — funding history as liquidation proxy (open API, no IP blocks)
    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - 48 * 3600 * 1000
    r = requests.post("https://api.hyperliquid.xyz/info",
                      json={"type": "fundingHistory", "coin": symbol,
                            "startTime": start_ms, "endTime": now_ms},
                      headers={**_CRYPTO_HEADERS, "Content-Type": "application/json"},
                      timeout=10)
    r.raise_for_status()
    rows = r.json()
    out = []
    for row in rows:
        try:
            ts   = int(row.get("time", 0))
            rate = round(float(row.get("fundingRate", 0)) * 100, 4)
            px   = round(float(row.get("premium", 0)) * 100, 4)
        except (ValueError, TypeError):
            continue
        out.append({"ts": ts, "rate": rate, "premium": px})
    # Also grab current OI + mark price
    r2 = requests.post("https://api.hyperliquid.xyz/info",
                       json={"type": "metaAndAssetCtxs"},
                       headers={**_CRYPTO_HEADERS, "Content-Type": "application/json"},
                       timeout=10)
    oi = 0
    price = 0
    if r2.status_code == 200:
        payload = r2.json()
        universe = payload[0].get("universe", [])
        ctxs     = payload[1]
        for meta, ctx in zip(universe, ctxs):
            if meta.get("name", "").upper() == symbol.upper():
                try:
                    oi    = round(float(ctx.get("openInterest", 0)), 2)
                    price = round(float(ctx.get("markPx", 0)), 4)
                except (ValueError, TypeError):
                    pass
                break
    out.sort(key=lambda x: x["ts"])
    return {"history": out, "oi": oi, "price": price, "symbol": symbol}

@app.route("/crypto/liquidations")
@login_required
def crypto_liquidations_page():
    if _get_user_plan(session.get("user_id")) not in ("basic", "pro"):
        return redirect("/pricing?upgrade=crypto")
    return render_template_string(CRYPTO_LIQUIDATIONS_HTML, current_user=current_user())

@app.route("/api/crypto/liquidations")
def crypto_liquidations_api():
    if "user_id" not in session:
        return jsonify({"error": "login_required"}), 401
    if _get_user_plan(session.get("user_id")) not in ("basic", "pro"):
        return jsonify({"error": "upgrade_required"}), 403
    symbol = request.args.get("symbol", "BTC").upper()[:10]
    import time
    now = time.time()
    entry = _crypto_liq_cache.get(symbol, {"data": None, "ts": 0})
    if entry["data"] is None or now - entry["ts"] > 600:
        try:
            entry = {"data": _fetch_crypto_liq(symbol), "ts": now}
            _crypto_liq_cache[symbol] = entry
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    result = entry["data"]
    if isinstance(result, dict):
        return jsonify(result)
    return jsonify({"history": result, "symbol": symbol})


# ─ Upcoming Coins ─────────────────────────────────────────────────────────────

def _fetch_crypto_upcoming():
    import requests
    # CoinGecko trending (free, no key) — top 15 coins by search volume in 24h
    r = requests.get("https://api.coingecko.com/api/v3/search/trending",
                     headers=_CRYPTO_HEADERS, timeout=8)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("coins", []):
        c = item.get("item", {})
        out.append({
            "symbol": (c.get("symbol") or "").upper(),
            "name":   c.get("name", ""),
            "rank":   c.get("market_cap_rank") or "—",
            "score":  c.get("score", 0) + 1,  # 1-indexed rank in trending list
            "price_btc": c.get("price_btc", 0),
        })
    return out

@app.route("/crypto/upcoming")
def crypto_upcoming_page():
    return render_template_string(CRYPTO_UPCOMING_HTML, current_user=current_user())

@app.route("/api/crypto/upcoming")
def crypto_upcoming_api():
    import time
    now = time.time()
    if _crypto_upcoming_cache["data"] is None or now - _crypto_upcoming_cache["ts"] > 3600:
        try:
            _crypto_upcoming_cache["data"] = _fetch_crypto_upcoming()
            _crypto_upcoming_cache["ts"] = now
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"coins": _crypto_upcoming_cache["data"]})


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
  <meta property="og:title" content="ChartEdge.trade — Free TradingView Indicators">
  <meta property="og:description" content="Free Pine Script indicators for TradingView. No paid plan required.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://chartedge.trade">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="ChartEdge.trade — Free TradingView Indicators">
  <meta name="twitter:description" content="Free Pine Script indicators for TradingView. No paid plan required.">{% if show_email_banner %}
  <meta name="email-missing" content="1">{% endif %}""" + _GA_SCRIPT + """
  <script>
    (function() {
      var t = localStorage.getItem('theme') || 'dark';
      document.documentElement.setAttribute('data-theme', t);
    })();
  </script>
"""

# ── Shared nav macro ─────────────────────────────────────────────────────────

_NAV_CSS = """
  nav {
    position: fixed; top: 0; left: 0; right: 0;
    z-index: 1000; overflow: visible !important;
    transition: transform .35s cubic-bezier(.4,0,.2,1), background-color .3s ease, -webkit-backdrop-filter .3s ease, backdrop-filter .3s ease, box-shadow .3s ease, border-color .3s ease, padding .25s ease;
    will-change: transform;
  }
  nav.nav-hidden { transform: translateY(-110%); }
  nav.nav-scrolled {
    background: rgba(13,17,23,.72);
    -webkit-backdrop-filter: saturate(180%) blur(14px);
    backdrop-filter: saturate(180%) blur(14px);
    box-shadow: 0 8px 32px rgba(0,0,0,.35);
    border-bottom-color: rgba(48,54,61,.4);
    padding-top: 10px; padding-bottom: 10px;
  }
  :root[data-theme="light"] nav.nav-scrolled {
    background: rgba(255,255,255,.78);
    box-shadow: 0 8px 32px rgba(140,150,160,.18);
    border-bottom-color: rgba(208,215,222,.55);
  }
  body { padding-top: 56px; }
  @media (max-width: 640px) { body { padding-top: 52px; } }
  @media (prefers-reduced-motion: reduce) {
    nav { transition: background-color .2s ease, box-shadow .2s ease; }
    nav.nav-hidden { transform: none; }
  }
  .nav-links { display: flex; align-items: center; gap: 2px; }
  @media (min-width: 641px) {
    .nav-links {
      background: rgba(13,17,23,.45);
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 999px;
      padding: 4px;
      -webkit-backdrop-filter: blur(12px) saturate(140%);
              backdrop-filter: blur(12px) saturate(140%);
      box-shadow: 0 4px 14px rgba(0,0,0,.25);
    }
    :root[data-theme="light"] .nav-links {
      background: rgba(255,255,255,.55);
      border-color: rgba(208,215,222,.7);
      box-shadow: 0 4px 14px rgba(140,150,160,.15);
    }
  }
  .nav-links a {
    color: var(--muted); text-decoration: none;
    font-size: 0.88rem; padding: 8px 16px; border-radius: 999px;
    position: relative; transition: color .2s ease;
  }
  .nav-links a:hover { color: var(--text); }
  .dropdown { position: relative; }
  .dropdown > .drop-btn {
    position: relative;
    background: none; border: none; color: var(--muted);
    padding: 8px 16px; border-radius: 999px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 0.88rem; font-weight: 600;
    cursor: pointer; display: flex; align-items: center; gap: 4px;
    transition: color .2s ease, background-color .2s ease;
  }
  /* Tubelight glow bar — emerges from top of active button */
  .dropdown > .drop-btn::before,
  .nav-links > a:not(.logo)::before {
    content: '';
    position: absolute;
    top: -5px; left: 50%;
    width: 24px; height: 3px;
    transform: translateX(-50%) scaleX(.4);
    background: var(--accent);
    border-radius: 0 0 4px 4px;
    opacity: 0;
    box-shadow:
      0 0 8px var(--accent),
      0 6px 14px var(--accent),
      0 10px 24px rgba(88,166,255,.5);
    transition: opacity .25s ease, transform .25s cubic-bezier(.4,0,.2,1);
    pointer-events: none;
  }
  .dropdown > .drop-btn:hover, .dropdown > .drop-btn.open,
  .nav-links > a:not(.logo):hover {
    color: var(--text); background: rgba(88,166,255,.08);
  }
  .dropdown > .drop-btn:hover::before, .dropdown > .drop-btn.open::before,
  .nav-links > a:not(.logo):hover::before {
    opacity: 1; transform: translateX(-50%) scaleX(1);
  }
  @media (prefers-reduced-motion: reduce) {
    .dropdown > .drop-btn::before, .nav-links > a:not(.logo)::before { transition: opacity .15s ease; }
  }
  .drop-menu {
    position: absolute; top: calc(100% + 6px); left: 0;
    background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
    min-width: 180px; z-index: 9999; overflow: hidden;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    opacity: 0; transform: translateY(-6px); pointer-events: none;
    transition: opacity 0.18s ease, transform 0.18s ease;
  }
  .dropdown:last-child .drop-menu, .dropdown:nth-last-child(2) .drop-menu { left: auto; right: 0; }
  .drop-menu.open { opacity: 1; transform: translateY(0); pointer-events: auto; }
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
    .drop-menu { position: static; box-shadow: none; border: none; border-radius: 0; background: var(--bg3); transform: none; max-height: 0; transition: opacity 0.18s ease, max-height 0.22s ease; }
    .drop-menu.open { max-height: 600px; }
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
        <a href="/heatmap">Market Heatmap</a>
        <a href="/earnings">Earnings Calendar</a>
        <a href="/dividends">Dividends Calendar</a>
        <a href="/economic">Economic Calendar</a>
        <a href="/news">News</a>
        <a href="/ipo">IPO Calendar</a>
        {% if nav_plan == 'free' %}
        <a href="/pricing?upgrade=trump" style="opacity:.4;">Trump Tracker 🔒</a>
        <a href="/pricing?upgrade=gamma" style="opacity:.4;">Gamma Exposure 🔒</a>
        <a href="/pricing?upgrade=greeks" style="opacity:.4;">Greeks Dashboard 🔒</a>
        <a href="/pricing?upgrade=volforecast" style="opacity:.4;">Volatility Forecast 🔒</a>
        <a href="/pricing?upgrade=flow" style="opacity:.4;">Options Flow 🔒</a>
        <a href="/pricing?upgrade=insider" style="opacity:.4;">Insider Trading 🔒</a>
        <a href="/pricing?upgrade=premarket" style="opacity:.4;">Pre-Market Scanner 🔒</a>
        <a href="/pricing?upgrade=backtest" style="opacity:.4;">Backtester 🔒</a>
        <a href="/pricing?upgrade=darkpool" style="opacity:.4;">Dark Pool 🔒</a>
        {% elif nav_plan == 'basic' %}
        <a href="/trump">Trump Tracker</a>
        <a href="/gamma">Gamma Exposure</a>
        <a href="/greeks">Greeks Dashboard</a>
        <a href="/volforecast">Volatility Forecast</a>
        <a href="/pricing?upgrade=flow" style="opacity:.4;">Options Flow 🔒</a>
        <a href="/pricing?upgrade=insider" style="opacity:.4;">Insider Trading 🔒</a>
        <a href="/pricing?upgrade=premarket" style="opacity:.4;">Pre-Market Scanner 🔒</a>
        <a href="/pricing?upgrade=backtest" style="opacity:.4;">Backtester 🔒</a>
        <a href="/pricing?upgrade=darkpool" style="opacity:.4;">Dark Pool 🔒</a>
        {% else %}
        <a href="/trump">Trump Tracker</a>
        <a href="/gamma">Gamma Exposure</a>
        <a href="/greeks">Greeks Dashboard</a>
        <a href="/volforecast">Volatility Forecast</a>
        <a href="/flow">Options Flow</a>
        <a href="/insider">Insider Trading</a>
        <a href="/premarket">Pre-Market Scanner</a>
        <a href="/backtest">Backtester</a>
        <a href="/darkpool">Dark Pool</a>
        {% endif %}
      </div>
    </div>
    <div class="dropdown">
      <button class="drop-btn" onclick="toggleDrop(this, event)">Crypto ▾</button>
      <div class="drop-menu">
        <a href="/crypto/feargreed">Fear &amp; Greed</a>
        <a href="/crypto/heatmap">Crypto Heatmap</a>
        <a href="/crypto/upcoming">Trending Coins</a>
        {% if nav_plan == 'free' %}
        <a href="/pricing?upgrade=crypto" style="opacity:.4;">BTC Dominance 🔒</a>
        <a href="/pricing?upgrade=crypto" style="opacity:.4;">Funding Rates 🔒</a>
        <a href="/pricing?upgrade=crypto" style="opacity:.4;">On-Chain Metrics 🔒</a>
        <a href="/pricing?upgrade=crypto" style="opacity:.4;">Liquidation Map 🔒</a>
        {% else %}
        <a href="/crypto/dominance">BTC Dominance</a>
        <a href="/crypto/funding">Funding Rates</a>
        <a href="/crypto/onchain">On-Chain Metrics</a>
        {% if nav_plan == 'pro' %}
        <a href="/crypto/liquidations">Liquidation Map</a>
        {% else %}
        <a href="/pricing?upgrade=crypto" style="opacity:.4;">Liquidation Map 🔒</a>
        {% endif %}
        {% endif %}
      </div>
    </div>
    <div class="dropdown">
      <button class="drop-btn" onclick="toggleDrop(this, event)">Community ▾</button>
      <div class="drop-menu">
        <a href="/request">Request Indicator</a>
        <a href="/faq">FAQ</a>
        {% if current_user %}<a href="/favorites">♥ My Favorites</a>{% endif %}
      </div>
    </div>
    <a href="/pricing" style="font-size:.85rem;color:var(--accent);padding:6px 10px;text-decoration:none;font-weight:600;">Pricing</a>
    <div class="dropdown">
      {% if current_user %}
      <button class="drop-btn" onclick="toggleDrop(this, event)">{{ current_user }} ▾</button>
      <div class="drop-menu">
        <a href="/me">👤 Profile</a>
        <a href="/settings">⚙️ Settings</a>
        <a href="/favorites">♥ Favorites</a>
        <a href="/billing">Billing</a>
        <a href="/redeem">Redeem Code</a>
        <a href="/refer">Refer a Friend</a>
        <a href="/logout">Logout</a>
      </div>
      {% else %}
      <button class="drop-btn" onclick="toggleDrop(this, event)">Account ▾</button>
      <div class="drop-menu">
        <a href="/login">Login</a>
        <a href="/register">Register</a>
        <a href="/forgot-password">Forgot password?</a>
      </div>
      {% endif %}
    </div>
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
(function initScrollNav() {
  let lastY = window.scrollY || 0;
  let ticking = false;
  function update() {
    const nav = document.querySelector('nav');
    if (!nav) { ticking = false; return; }
    const y = window.scrollY;
    nav.classList.toggle('nav-scrolled', y > 8);
    if (y > lastY + 4 && y > 80) {
      if (!nav.classList.contains('nav-hidden')) {
        nav.classList.add('nav-hidden');
        document.querySelectorAll('.drop-menu.open').forEach(m => m.classList.remove('open'));
        document.querySelectorAll('.drop-btn.open').forEach(b => b.classList.remove('open'));
      }
    } else if (y < lastY - 4 || y <= 80) {
      nav.classList.remove('nav-hidden');
    }
    lastY = y;
    ticking = false;
  }
  window.addEventListener('scroll', function() {
    if (!ticking) { requestAnimationFrame(update); ticking = true; }
  }, { passive: true });
})();
(function initEmailBanner() {
  if (!document.querySelector('meta[name="email-missing"]')) return;
  if (sessionStorage.getItem('ce_email_banner_dismissed') === '1') return;
  function mount() {
    if (document.getElementById('ce-email-banner')) return;
    var b = document.createElement('div');
    b.id = 'ce-email-banner';
    b.style.cssText = 'position:fixed;top:56px;left:0;right:0;z-index:999;background:linear-gradient(90deg,#1f6feb 0%,#388bfd 100%);color:#fff;padding:10px 16px;display:flex;align-items:center;justify-content:center;gap:14px;font-size:.85rem;box-shadow:0 4px 14px rgba(0,0,0,.25);flex-wrap:wrap;';
    b.innerHTML = '<span>📧 Add a recovery email so you can reset your password if you forget it.</span><a href="/me#edit-section" style="background:rgba(255,255,255,.18);color:#fff;padding:5px 14px;border-radius:6px;text-decoration:none;font-weight:600;">Add email →</a><button aria-label="Dismiss" id="ce-email-banner-x" style="background:none;border:none;color:#fff;font-size:1.1rem;cursor:pointer;padding:0 6px;line-height:1;">×</button>';
    document.body.insertBefore(b, document.body.firstChild);
    document.body.style.paddingTop = '108px';
    document.getElementById('ce-email-banner-x').addEventListener('click', function() {
      sessionStorage.setItem('ce_email_banner_dismissed', '1');
      b.remove();
      document.body.style.paddingTop = '';
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
"""

PROFILE_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Profile · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .container { max-width: 720px; margin: 0 auto; padding: 36px 24px; }
    .profile-header { display: flex; align-items: center; gap: 20px; margin-bottom: 32px; padding: 28px; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; }
    .avatar { width: 72px; height: 72px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 700; color: #fff; flex-shrink: 0; }
    .profile-info h1 { font-size: 1.4rem; margin-bottom: 4px; }
    .profile-info .meta { color: var(--muted); font-size: .85rem; }
    .plan-pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: .75rem; font-weight: 700; text-transform: uppercase; margin-left: 8px; }
    .section { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 22px; margin-bottom: 20px; }
    .section-title { font-size: .75rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 16px; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; }
    .stat-box { background: var(--bg3); border-radius: 8px; padding: 14px; text-align: center; }
    .stat-num { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
    .stat-lbl { font-size: .72rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
    .badges-grid { display: flex; flex-wrap: wrap; gap: 10px; }
    .badge-card { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border); min-width: 160px; }
    .badge-icon { font-size: 1.4rem; }
    .badge-label { font-size: .82rem; font-weight: 600; }
    .no-badges { color: var(--muted); font-size: .85rem; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
    .avatar img { width: 72px; height: 72px; border-radius: 50%; object-fit: cover; }
    .edit-form { display: flex; flex-direction: column; gap: 14px; }
    .edit-row { display: flex; flex-direction: column; gap: 6px; }
    .edit-row label { font-size: .78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .edit-row input[type=text], .edit-row input[type=file] { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); font-size: .9rem; width: 100%; }
    .btn-save { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 9px 22px; font-size: .9rem; font-weight: 600; cursor: pointer; align-self: flex-start; }
    .btn-save:hover { opacity: .85; }
    .error-msg { background: #2d1515; border: 1px solid var(--red); color: var(--red); border-radius: 6px; padding: 10px 14px; font-size: .85rem; }
    .pencil-btn { background: none; border: none; cursor: pointer; font-size: .85rem; color: var(--muted); padding: 2px 5px; border-radius: 4px; vertical-align: middle; margin-left: 6px; display: inline-block; transform: scaleX(-1); }
    .pencil-btn:hover { color: var(--accent); background: var(--bg3); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="container">
  <div class="profile-header">
    <div class="avatar">{% if profile_pic %}<img src="{{ profile_pic }}" alt="avatar">{% else %}{{ username[0].upper() }}{% endif %}</div>
    <div class="profile-info">
      <h1>{{ username }}
        <span class="plan-pill" style="background:{{ {'free':'#21262d','basic':'#0d1a2d','pro':'#2a2000'}[plan] }};color:{{ {'free':'#8b949e','basic':'#58a6ff','pro':'#e3b341'}[plan] }};">{{ plan.upper() }}</span>
        <button class="pencil-btn" onclick="document.getElementById('edit-section').style.display=document.getElementById('edit-section').style.display==='none'?'block':'none'" title="Edit profile">✏️</button>
      </h1>
      <div class="meta">Member since {{ member_since }}</div>
    </div>
  </div>

  <div id="edit-section" style="display:{% if profile_error %}block{% else %}none{% endif %}">
    <div class="section">
      <div class="section-title">Edit Profile</div>
      {% if profile_error %}<div class="error-msg" style="margin-bottom:14px;">{{ profile_error }}</div>{% endif %}
      <form class="edit-form" method="POST" action="/profile/update" enctype="multipart/form-data">
        <div class="edit-row">
          <label>Username</label>
          <input type="text" name="username" value="{{ username }}" maxlength="30" placeholder="New username">
        </div>
        <div class="edit-row">
          <label>Profile Picture</label>
          <input type="file" name="profile_pic" accept="image/*">
        </div>
        <button class="btn-save" type="submit">Save Changes</button>
      </form>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Stats</div>
    <div class="stats-grid">
      <div class="stat-box"><div class="stat-num">{{ stats.copies }}</div><div class="stat-lbl">Indicators Copied</div></div>
      <div class="stat-box"><div class="stat-num">{{ stats.favorites }}</div><div class="stat-lbl">Favorites</div></div>
      <div class="stat-box"><div class="stat-num">{{ stats.referrals }}</div><div class="stat-lbl">Referrals</div></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Badges</div>
    {% if badges %}
    <div class="badges-grid">
      {% for b in badges %}
      <div class="badge-card" style="background:{{ b.bg }};border-color:{{ b.color }}30;">
        <span class="badge-icon">{{ b.icon }}</span>
        <span class="badge-label" style="color:{{ b.color }};">{{ b.label }}</span>
      </div>
      {% endfor %}
    </div>
    {% else %}
    <div class="no-badges">No badges yet — start copying indicators to earn some!</div>
    {% endif %}
  </div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;">
    <a href="/billing" style="color:var(--accent);font-size:.85rem;">Manage Plan</a>
    <a href="/refer"   style="color:var(--accent);font-size:.85rem;">Refer a Friend</a>
    <a href="/favorites" style="color:var(--accent);font-size:.85rem;">My Favorites</a>
  </div>
</div>

<footer>© 2026 ChartEdge.trade</footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""

# ── Pricing / Billing / Redeem HTML ──────────────────────────────────────────

REDEEM_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Redeem Code · ChartEdge.trade</title>
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
<footer>© 2026 ChartEdge.trade · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


BILLING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Billing · ChartEdge.trade</title>
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Your <span>Billing</span></h1>
  {% if success %}<div class="success-box">✓ Payment successful! Your plan has been upgraded.</div>{% endif %}
  <div class="card">
    <div class="plan-badge badge-{{ plan }}">{{ plan|upper }}</div>
    <h3 style="margin-bottom:8px;">Current plan</h3>
    <p style="color:var(--muted);font-size:.9rem;margin-bottom:{% if plan_expires %}12px{% else %}20px{% endif %};">
      {% if plan == 'free' %}3 copies/day · Free forever
      {% elif plan == 'basic' %}8 copies/day · $9.99/month
      {% else %}Unlimited copies · $15.99/month{% endif %}
    </p>
    {% if plan_expires and plan != 'free' %}
    <p style="color:#e3b341;font-size:.82rem;margin-bottom:20px;">⏳ Referral reward — access expires {{ plan_expires.strftime('%b %d, %Y') }}</p>
    {% endif %}
    {% if portal_url %}<a href="{{ portal_url }}" class="btn btn-secondary">Manage / Cancel Subscription →</a>
    {% elif plan == 'free' %}<a href="/pricing" class="btn btn-primary">Upgrade Plan</a>{% endif %}
  </div>
  <a href="/indicators" style="color:var(--muted);font-size:.85rem;">← Back to indicators</a>
</div>
<footer>© 2026 ChartEdge.trade · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


PRICING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Pricing · ChartEdge.trade</title>
""" + _META + """
  <style>
    :root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--text:#e6edf3;--muted:#8b949e;--border:#30363d;--accent:#58a6ff;}
    *{box-sizing:border-box;margin:0;padding:0;}
    html,body{background:#000;color:#fff;}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow-x:hidden;}
    a{color:inherit;}
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """

    /* ---- Pricing shell ---- */
    .pricing-shell { position: relative; min-height: 100vh; background: #000; overflow: hidden; padding-bottom: 96px; }

    /* Grid pattern overlay */
    .px-grid {
      position: absolute; inset: 0; pointer-events: none;
      background-image:
        linear-gradient(to right, rgba(255,255,255,.045) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(255,255,255,.025) 1px, transparent 1px);
      background-size: 70px 80px;
      -webkit-mask-image: radial-gradient(50% 40% at 50% 22%, white, transparent);
              mask-image: radial-gradient(50% 40% at 50% 22%, white, transparent);
    }

    /* Sparkles canvas */
    .px-sparkles {
      position: absolute; inset: 0; pointer-events: none;
      -webkit-mask-image: radial-gradient(50% 50% at 50% 25%, white, transparent 85%);
              mask-image: radial-gradient(50% 50% at 50% 25%, white, transparent 85%);
    }

    /* Big blue radial glow */
    .px-glow {
      position: absolute; top: -180px; left: 8%; right: 8%; height: 760px;
      background: radial-gradient(circle at 50% 30%, #3131f5 0%, transparent 60%);
      filter: blur(80px); opacity: .42; pointer-events: none;
    }

    /* ---- Heading ---- */
    .px-head { position: relative; z-index: 5; max-width: 820px; margin: 0 auto; padding: clamp(56px, 12vh, 110px) 24px 0; text-align: center; }
    .px-head h1 {
      font-size: clamp(2rem, 4.4vw, 3rem);
      font-weight: 600; letter-spacing: -.02em; line-height: 1.1;
      color: #fff; margin-bottom: 16px;
    }
    .px-sub { color: #c9d1d9; font-size: 1rem; max-width: 560px; margin: 0 auto 30px; line-height: 1.65; }

    /* Vertical cut reveal */
    .vcr { display: inline-flex; flex-wrap: wrap; justify-content: center; gap: 0 .35em; }
    .vcr-w { display: inline-block; overflow: hidden; line-height: 1.1; vertical-align: bottom; }
    .vcr-w > span { display: inline-block; transform: translateY(110%); animation: vcr-up .9s cubic-bezier(.45,1.25,.5,1) var(--d,0s) forwards; }
    @keyframes vcr-up { to { transform: translateY(0); } }

    /* Fade-in-blur (TimelineContent) */
    .tl { opacity: 0; transform: translateY(-18px); filter: blur(10px); animation: tl-in .75s ease-out forwards; animation-delay: var(--d,0s); }
    @keyframes tl-in { to { opacity: 1; transform: translateY(0); filter: blur(0); } }

    /* ---- Pill toggle ---- */
    .bill-pill {
      position: relative;
      display: inline-grid; grid-template-columns: 1fr 1fr;
      background: #0a0a0a; border: 1px solid #2a2d34; border-radius: 999px; padding: 4px;
      box-shadow: 0 6px 24px rgba(0,0,0,.5);
    }
    .bp-btn {
      z-index: 2; padding: 9px 22px; border: none; background: none;
      color: #c9d1d9; font-weight: 600; font-size: .92rem; border-radius: 999px;
      cursor: pointer; transition: color .25s; width: 100%;
      display: inline-flex; align-items: center; justify-content: center; gap: 6px;
      white-space: nowrap;
    }
    .bp-btn.active { color: #fff; }
    .bp-slider {
      position: absolute; top: 4px; left: 4px;
      height: calc(100% - 8px); width: calc(50% - 4px);
      background: linear-gradient(180deg, #4493f8 0%, #1f6feb 100%);
      border: 1px solid #1f6feb; border-radius: 999px;
      box-shadow: 0 6px 22px rgba(31,111,235,.45);
      transition: transform .35s cubic-bezier(.4,0,.2,1);
      z-index: 1;
    }
    .bill-pill[data-bill="1"] .bp-slider { transform: translateX(100%); }
    .save-chip {
      display: inline-flex; align-items: center;
      background: rgba(63,185,80,.18); color: #56d364;
      font-size: .66rem; font-weight: 700; padding: 1px 7px; border-radius: 6px;
      letter-spacing: .03em;
    }

    /* ---- Plans grid ---- */
    .plans-grid {
      position: relative; z-index: 5;
      display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      max-width: 1080px; gap: 16px;
      padding: 36px 24px 0; margin: 0 auto;
    }
    .plan {
      position: relative; overflow: hidden;
      background: linear-gradient(180deg, #0f1117 0%, #1a1d24 55%, #0f1117 100%);
      border: 1px solid #2a2d34; border-radius: 18px;
      padding: 28px 26px 30px; color: #fff;
    }
    .plan.popular {
      border-color: rgba(88,166,255,.45);
      box-shadow: 0 0 0 1px rgba(88,166,255,.22), 0 -10px 90px rgba(9,0,255,.38), 0 24px 60px rgba(0,0,0,.65);
      transform: translateY(-6px); z-index: 2;
    }
    .pop-tag {
      position: absolute; top: 14px; right: 14px;
      background: linear-gradient(135deg, #4493f8 0%, #1f6feb 100%);
      color: #fff; font-size: .62rem; font-weight: 800;
      padding: 4px 10px; border-radius: 999px; letter-spacing: .08em;
      box-shadow: 0 4px 14px rgba(31,111,235,.45);
    }
    .plan-name { font-size: 1.7rem; font-weight: 700; margin-bottom: 10px; }
    .plan-price-wrap { display: flex; align-items: baseline; }
    .plan-price-wrap .currency { font-size: 1.6rem; font-weight: 700; }
    .plan-price { font-size: 2.6rem; font-weight: 700; letter-spacing: -.025em; font-variant-numeric: tabular-nums; }
    .plan-per { color: #c9d1d9; font-size: .92rem; margin-left: 6px; }
    .plan-desc { color: #c9d1d9; font-size: .82rem; margin: 10px 0 18px; line-height: 1.5; }
    .trial-line {
      display: inline-block;
      background: rgba(63,185,80,.14); color: #56d364;
      font-size: .72rem; font-weight: 700;
      padding: 3px 9px; border-radius: 6px;
      margin: 2px 0 14px;
    }
    .btn-plan {
      display: block; text-align: center; padding: 14px;
      border-radius: 12px; font-size: 1.05rem; font-weight: 600;
      text-decoration: none; margin-bottom: 22px;
      transition: transform .15s, box-shadow .2s, border-color .2s;
    }
    .btn-pop {
      background: linear-gradient(180deg, #4493f8 0%, #1f6feb 100%);
      border: 1px solid #1f6feb; color: #fff;
      box-shadow: 0 10px 28px rgba(9,0,255,.35);
    }
    .btn-pop:hover { transform: translateY(-2px); box-shadow: 0 14px 36px rgba(9,0,255,.5); }
    .btn-ghost {
      background: linear-gradient(180deg, #0a0a0a 0%, #2b2b2b 100%);
      border: 1px solid #2a2d34; color: #fff;
      box-shadow: 0 6px 20px rgba(0,0,0,.45);
    }
    .btn-ghost:hover { transform: translateY(-2px); border-color: rgba(255,255,255,.18); }

    .feat-head { font-size: .9rem; font-weight: 600; color: #fff; padding-top: 16px; border-top: 1px solid rgba(255,255,255,.08); margin-bottom: 12px; }
    .feat-list { list-style: none; display: flex; flex-direction: column; gap: 7px; }
    .feat-list li { display: flex; align-items: center; gap: 10px; color: #c9d1d9; font-size: .85rem; line-height: 1.5; }
    .feat-list li::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: rgba(255,255,255,.32); flex-shrink: 0; }
    .plan.popular .feat-list li::before { background: rgba(88,166,255,.7); box-shadow: 0 0 6px rgba(88,166,255,.5); }

    .error-box {
      max-width: 600px; margin: 24px auto 0;
      background: rgba(248,81,73,.15); border: 1px solid #f85149; border-radius: 8px;
      padding: 12px 16px; color: #ff8e88; font-size: .9rem; text-align: center;
      position: relative; z-index: 5;
    }

    footer { text-align: center; padding: 32px 24px; color: #8b949e; font-size: .8rem; border-top: 1px solid #1a1d24; background: #000; position: relative; z-index: 5; }

    @media (max-width: 640px) {
      .plan.popular { transform: none; }
      .px-glow { height: 580px; }
      .px-head { padding-top: 64px; }
    }
    @media (prefers-reduced-motion: reduce) {
      .vcr-w > span, .tl { animation: none; opacity: 1; transform: none; filter: none; }
    }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:#e6edf3">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:#8b949e;font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="pricing-shell">
  <div class="px-grid"></div>
  <canvas class="px-sparkles" id="px-sparkles"></canvas>
  <div class="px-glow"></div>

  {% if error %}<div class="error-box">Something went wrong with checkout. Please try again.</div>{% endif %}

  <header class="px-head">
    <h1 class="vcr">
      <span class="vcr-w"><span style="--d:0s">Plans</span></span>
      <span class="vcr-w"><span style="--d:.08s">that</span></span>
      <span class="vcr-w"><span style="--d:.16s">fit</span></span>
      <span class="vcr-w"><span style="--d:.24s">how</span></span>
      <span class="vcr-w"><span style="--d:.32s">you</span></span>
      <span class="vcr-w"><span style="--d:.40s">trade.</span></span>
    </h1>
    <p class="px-sub tl" style="--d:.55s">Start free, upgrade when you need more. 7-day trial on every paid plan — cancel anytime.</p>

    <div class="tl" style="--d:.75s; display:flex; justify-content:center;">
      <div class="bill-pill" id="bill-pill" data-bill="0">
        <div class="bp-slider"></div>
        <button type="button" class="bp-btn active" onclick="setBill(0)">Monthly</button>
        <button type="button" class="bp-btn" onclick="setBill(1)">Yearly <span class="save-chip">25% OFF</span></button>
      </div>
    </div>
  </header>

  <div class="plans-grid">
    <!-- Free -->
    <div class="plan tl" style="--d:.95s">
      <h3 class="plan-name">Free</h3>
      <div class="plan-price-wrap">
        <span class="currency">$</span>
        <span class="plan-price" data-price="0">0</span>
        <span class="plan-per" data-per>/mo</span>
      </div>
      <p class="plan-desc">Get started with no commitment. No credit card required.</p>
      <a href="/indicators" class="btn-plan btn-ghost">Start Free</a>
      <div class="feat-head">Free includes:</div>
      <ul class="feat-list">
        <li>3 copies per day</li>
        <li>All indicators</li>
        <li>Live chart &amp; forecast</li>
        <li>Market news</li>
        <li>Earnings calendar</li>
        <li>Market heatmap</li>
        <li>Dividends calendar</li>
        <li>IPO calendar</li>
        <li>Economic calendar</li>
        <li>Crypto fear &amp; greed</li>
        <li>Crypto heatmap</li>
        <li>Trending coins</li>
      </ul>
    </div>

    <!-- Basic -->
    <div class="plan tl" style="--d:1.08s">
      <h3 class="plan-name">Basic</h3>
      <div class="plan-price-wrap">
        <span class="currency">$</span>
        <span class="plan-price" data-price="9.99" data-yearly="89.99">9.99</span>
        <span class="plan-per" data-per>/mo</span>
      </div>
      {% if not trial_used or not current_user %}<div class="trial-line">✓ 7-day free trial — no charge until then</div>{% endif %}
      <p class="plan-desc">For active traders who copy often.</p>
      {% if current_user %}
      <a href="/subscribe/basic" class="btn-plan btn-ghost" data-base="/subscribe/basic">{% if not trial_used %}Start 7-Day Free Trial{% else %}Get Basic{% endif %}</a>
      {% else %}
      <a href="/login?next=/subscribe/basic" class="btn-plan btn-ghost" data-base="/login?next=/subscribe/basic">Start 7-Day Free Trial</a>
      {% endif %}
      <div class="feat-head">Basic includes:</div>
      <ul class="feat-list">
        <li>8 copies per day</li>
        <li>All indicators</li>
        <li>Live chart &amp; forecast</li>
        <li>Market news</li>
        <li>Earnings calendar</li>
        <li>Market heatmap</li>
        <li>Dividends calendar</li>
        <li>IPO calendar</li>
        <li>Economic calendar</li>
        <li>Crypto fear &amp; greed</li>
        <li>Crypto heatmap</li>
        <li>Trending coins</li>
        <li>Trump tracker</li>
        <li>Ticker news</li>
        <li>Gamma exposure</li>
        <li>Greeks dashboard</li>
        <li>Volatility forecast</li>
        <li>BTC dominance</li>
        <li>Funding rates</li>
        <li>On-chain metrics</li>
      </ul>
    </div>

    <!-- Pro -->
    <div class="plan popular tl" style="--d:1.22s">
      <span class="pop-tag">MOST POPULAR</span>
      <h3 class="plan-name">Pro</h3>
      <div class="plan-price-wrap">
        <span class="currency">$</span>
        <span class="plan-price" data-price="15.99" data-yearly="149.99">15.99</span>
        <span class="plan-per" data-per>/mo</span>
      </div>
      {% if not trial_used or not current_user %}<div class="trial-line">✓ 7-day free trial — no charge until then</div>{% endif %}
      <p class="plan-desc">Unlimited access for power users.</p>
      {% if current_user %}
      <a href="/subscribe/pro" class="btn-plan btn-pop" data-base="/subscribe/pro">{% if not trial_used %}Start 7-Day Free Trial{% else %}Get Pro{% endif %}</a>
      {% else %}
      <a href="/login?next=/subscribe/pro" class="btn-plan btn-pop" data-base="/login?next=/subscribe/pro">Start 7-Day Free Trial</a>
      {% endif %}
      <div class="feat-head">Pro includes:</div>
      <ul class="feat-list">
        <li>Unlimited copies</li>
        <li>All indicators</li>
        <li>Live chart &amp; forecast</li>
        <li>Market news</li>
        <li>Earnings calendar</li>
        <li>Market heatmap</li>
        <li>Dividends calendar</li>
        <li>IPO calendar</li>
        <li>Economic calendar</li>
        <li>Crypto fear &amp; greed</li>
        <li>Crypto heatmap</li>
        <li>Trending coins</li>
        <li>Trump tracker</li>
        <li>Ticker news</li>
        <li>Gamma exposure</li>
        <li>Greeks dashboard</li>
        <li>Volatility forecast</li>
        <li>BTC dominance</li>
        <li>Funding rates</li>
        <li>On-chain metrics</li>
        <li>LSTM forecast</li>
        <li>Options flow</li>
        <li>Insider trading</li>
        <li>Pre-market scanner</li>
        <li>Liquidation map</li>
      </ul>
    </div>
  </div>
</div>

<footer>© 2026 ChartEdge.trade · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
// Billing toggle with animated number
function setBill(mode){
  var pill = document.getElementById('bill-pill');
  var was = parseInt(pill.getAttribute('data-bill'), 10);
  if (was === mode) return;
  pill.setAttribute('data-bill', mode);
  pill.querySelectorAll('.bp-btn').forEach(function(b, i){ b.classList.toggle('active', i === mode); });
  document.querySelectorAll('.plan-price').forEach(function(el){
    var monthly = parseFloat(el.dataset.price);
    var yearly  = parseFloat(el.dataset.yearly);
    if (isNaN(yearly)) { el.textContent = monthly === 0 ? '0' : monthly.toFixed(2); return; }
    var from = parseFloat(el.textContent.replace(/,/g, ''));
    var to   = mode === 1 ? yearly : monthly;
    animateNum(el, from, to, 480);
  });
  document.querySelectorAll('[data-per]').forEach(function(el){ el.textContent = mode === 1 ? '/yr' : '/mo'; });
  document.querySelectorAll('[data-base]').forEach(function(a){
    var base = a.getAttribute('data-base');
    a.href = mode === 1 ? base + (base.indexOf('?') >= 0 ? '&' : '?') + 'billing=yearly' : base;
  });
}
function animateNum(el, from, to, dur){
  var start = performance.now();
  function step(now){
    var t = Math.min(1, (now - start) / dur);
    var k = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
    var v = from + (to - from) * k;
    if (t >= 1) {
      el.textContent = to === 0 ? '0' : to.toFixed(2);
    } else {
      el.textContent = v.toFixed(2);
    }
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// Sparkles (white drifting dots)
(function(){
  var c = document.getElementById('px-sparkles');
  if (!c) return;
  var ctx = c.getContext('2d');
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  function resize(){
    c.width  = Math.floor(c.offsetWidth  * dpr);
    c.height = Math.floor(c.offsetHeight * dpr);
  }
  resize();
  window.addEventListener('resize', resize);
  var N = 140;
  var P = [];
  for (var i = 0; i < N; i++) {
    P.push({
      x: Math.random() * c.width,
      y: Math.random() * c.height,
      r: (Math.random() * 1.6 + 0.4) * dpr,
      vx: (Math.random() - 0.5) * 0.22 * dpr,
      vy: (Math.random() * 0.45 + 0.12) * dpr,
      a: Math.random() * 0.6 + 0.2,
      da: (Math.random() - 0.5) * 0.018
    });
  }
  function loop(){
    ctx.clearRect(0, 0, c.width, c.height);
    for (var i = 0; i < P.length; i++) {
      var p = P[i];
      p.x += p.vx; p.y += p.vy; p.a += p.da;
      if (p.a < 0.1 || p.a > 0.85) p.da = -p.da;
      if (p.y > c.height + 10) { p.y = -10; p.x = Math.random() * c.width; }
      if (p.x < -10) p.x = c.width + 10;
      if (p.x > c.width + 10) p.x = -10;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,' + p.a + ')';
      ctx.fill();
    }
    requestAnimationFrame(loop);
  }
  loop();
})();
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Auth HTML ─────────────────────────────────────────────────────────────────

AUTH_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>{% if mode == 'register' %}Register{% elif mode == 'forgot' %}Forgot password{% elif mode == 'reset' %}Reset password{% else %}Login{% endif %} — ChartEdge.trade</title>
  {% if mode == 'register' %}<script src="https://js.hcaptcha.com/1/api.js" async defer></script>{% endif %}
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; --green:#3fb950; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; --green:#1a7f37; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """

    .auth-shell {
      flex: 1; display: flex; align-items: center; justify-content: center;
      padding: 56px 24px 80px;
      background:
        radial-gradient(ellipse 50% 40% at 50% 0%, rgba(88,166,255,.06), transparent 70%),
        var(--bg);
    }
    .auth-card {
      width: 100%; max-width: 440px;
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 36px 36px 28px;
      box-shadow: 0 24px 60px rgba(0,0,0,.4), 0 0 0 1px rgba(255,255,255,.02);
    }
    .auth-head { margin-bottom: 26px; }
    .auth-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -.015em; margin-bottom: 8px; color: var(--text); }
    .auth-desc { color: var(--muted); font-size: .88rem; line-height: 1.55; }
    .auth-error { background: rgba(248,81,73,.12); border: 1px solid var(--red); border-radius: 8px; padding: 10px 14px; color: var(--red); font-size: .85rem; margin-bottom: 16px; }
    .auth-social {
      display: flex; align-items: center; justify-content: center; gap: 10px;
      width: 100%; padding: 11px 16px; text-decoration: none;
      background: var(--bg3); color: var(--text);
      border: 1px solid var(--border); border-radius: 9px;
      font-weight: 600; font-size: .92rem;
      transition: background .15s, border-color .15s;
      margin-bottom: 4px;
    }
    .auth-social:hover { background: var(--bg); border-color: var(--accent); }
    .auth-social svg { flex-shrink: 0; }
    .auth-divider {
      display: flex; align-items: center; gap: 12px;
      color: var(--muted); font-size: .72rem; letter-spacing: .14em;
      margin: 20px 0;
    }
    .auth-divider::before, .auth-divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .auth-form { display: flex; flex-direction: column; gap: 14px; }
    .auth-field { display: flex; flex-direction: column; gap: 6px; }
    .auth-field label { font-size: .78rem; color: var(--text); font-weight: 600; }
    .auth-opt { color: var(--muted); font-weight: 400; font-size: .72rem; margin-left: 4px; }
    .auth-input { position: relative; display: flex; align-items: center; }
    .auth-input-icon {
      position: absolute; left: 12px; top: 50%; transform: translateY(-50%);
      width: 16px; height: 16px; color: var(--muted); pointer-events: none;
    }
    .auth-input input {
      width: 100%; background: var(--bg); color: var(--text);
      border: 1px solid var(--border); border-radius: 9px;
      padding: 10px 14px 10px 38px; font-size: .92rem;
      transition: border-color .15s, box-shadow .15s;
    }
    .auth-input input::placeholder { color: var(--muted); opacity: .6; }
    .auth-input input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(88,166,255,.18); }
    .auth-input input.has-toggle { padding-right: 42px; }
    .auth-pw-toggle {
      position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
      background: none; border: none; color: var(--muted); cursor: pointer;
      width: 30px; height: 30px; border-radius: 6px;
      display: flex; align-items: center; justify-content: center;
      transition: color .15s, background .15s;
    }
    .auth-pw-toggle:hover { background: var(--bg3); color: var(--text); }
    .auth-ref {
      background: rgba(63,185,80,.12); border: 1px solid var(--green); border-radius: 8px;
      padding: 9px 12px; color: var(--green); font-size: .8rem; font-weight: 600;
    }
    .auth-success { background: rgba(63,185,80,.12); border: 1px solid var(--green); border-radius: 8px; padding: 10px 14px; color: var(--green); font-size: .85rem; margin-bottom: 16px; }
    .auth-label-row { display: flex; justify-content: space-between; align-items: baseline; }
    .auth-label-row a { color: var(--accent); font-size: .76rem; text-decoration: none; font-weight: 500; }
    .auth-label-row a:hover { text-decoration: underline; }
    .auth-submit {
      width: 100%; background: var(--accent); color: #fff; border: none;
      padding: 12px; border-radius: 9px;
      font-weight: 700; font-size: .95rem; cursor: pointer;
      transition: opacity .15s, transform .15s, box-shadow .2s;
      margin-top: 6px;
      box-shadow: 0 8px 20px rgba(88,166,255,.22);
    }
    .auth-submit:hover { opacity: .94; transform: translateY(-1px); box-shadow: 0 12px 28px rgba(88,166,255,.34); }
    .auth-switch { color: var(--muted); font-size: .85rem; text-align: center; margin-top: 24px; }
    .auth-switch a { color: var(--accent); text-decoration: none; font-weight: 600; }
    .auth-switch a:hover { text-decoration: underline; }
    .auth-tos { color: var(--muted); font-size: .72rem; text-align: center; margin-top: 14px; line-height: 1.6; }
    .auth-tos a { color: var(--muted); text-decoration: underline; }
    .auth-tos a:hover { color: var(--text); }
    @media (max-width: 480px) {
      .auth-card { padding: 28px 22px 22px; border-radius: 14px; }
    }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="auth-shell">
  <div class="auth-card">
    <div class="auth-head">
      <h1 class="auth-title">
        {% if mode == 'register' %}Create your account
        {% elif mode == 'forgot' %}Forgot password?
        {% elif mode == 'reset' %}Set a new password
        {% else %}Welcome back{% endif %}
      </h1>
      <p class="auth-desc">
        {% if mode == 'register' %}
          Start with the free tier — 20+ Pine Script indicators, live charts, and market data. Upgrade anytime.
        {% elif mode == 'forgot' %}
          Enter the email tied to your account. We'll send you a one-time link to set a new password.
        {% elif mode == 'reset' %}
          Choose a new password to finish resetting your account.
        {% else %}
          Sign in to access your indicators, watchlists, and Pro tools.
        {% endif %}
      </p>
    </div>

    {% if error %}<div class="auth-error">{{ error }}</div>{% endif %}
    {% if success %}<div class="auth-success">{{ success }}</div>{% endif %}

    {% if mode in ('register', 'login') %}
    <a class="auth-social" href="/login/google">
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
        <path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/>
        <path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.258c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/>
        <path fill="#FBBC05" d="M3.964 10.707A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.707V4.961H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.039l3.007-2.332z"/>
        <path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.961L3.964 6.293C4.672 4.169 6.656 3.58 9 3.58z"/>
      </svg>
      {{ 'Sign up with Google' if mode == 'register' else 'Continue with Google' }}
    </a>

    <div class="auth-divider"><span>OR</span></div>
    {% endif %}

    {% if not (mode == 'reset' and success) %}
    <form method="POST" class="auth-form">
      {% if mode in ('register', 'login') %}
      <div class="auth-field">
        <label for="auth-username">Username</label>
        <div class="auth-input">
          <svg class="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
          </svg>
          <input id="auth-username" type="text" name="username" required autofocus placeholder="your_username">
        </div>
      </div>
      {% endif %}

      {% if mode == 'register' %}
      <div class="auth-field">
        <label for="auth-email">Email<span class="auth-opt">needed for password recovery</span></label>
        <div class="auth-input">
          <svg class="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="2" y="4" width="20" height="16" rx="2"/>
            <path d="m22 7-10 5L2 7"/>
          </svg>
          <input id="auth-email" type="email" name="email" required placeholder="you@example.com">
        </div>
      </div>
      {% endif %}

      {% if mode == 'forgot' %}
      <div class="auth-field">
        <label for="auth-email">Email</label>
        <div class="auth-input">
          <svg class="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="2" y="4" width="20" height="16" rx="2"/>
            <path d="m22 7-10 5L2 7"/>
          </svg>
          <input id="auth-email" type="email" name="email" required autofocus placeholder="you@example.com">
        </div>
      </div>
      {% endif %}

      {% if mode in ('register', 'login', 'reset') %}
      {% if mode == 'reset' %}<input type="hidden" name="token" value="{{ token }}">{% endif %}
      <div class="auth-field">
        <div class="auth-label-row">
          <label for="auth-pw">
            {% if mode == 'reset' %}New password{% else %}Password{% endif %}{% if mode == 'register' or mode == 'reset' %}<span class="auth-opt">min 6 chars</span>{% endif %}
          </label>
          {% if mode == 'login' %}<a href="/forgot-password">Forgot password?</a>{% endif %}
        </div>
        <div class="auth-input">
          <svg class="auth-input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="3" y="11" width="18" height="11" rx="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
          </svg>
          <input id="auth-pw" type="password" name="password" required placeholder="••••••••" class="has-toggle"{% if mode == 'reset' %} autofocus{% endif %}>
          <button type="button" class="auth-pw-toggle" onclick="togglePw()" aria-label="Show or hide password">
            <svg id="auth-pw-eye" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
          </button>
        </div>
      </div>
      {% endif %}

      {% if mode == 'register' and ref %}
      <input type="hidden" name="ref" value="{{ ref }}">
      <div class="auth-ref">✓ Referral code applied — you'll get 7 days of Pro free!</div>
      {% endif %}

      {% if mode == 'register' %}
      <div class="h-captcha" data-sitekey="b06c575c-981a-4884-bbc0-5aa8c1d8aaa0"></div>
      {% endif %}

      <button class="auth-submit" type="submit">
        {% if mode == 'register' %}Create account
        {% elif mode == 'forgot' %}Send reset link
        {% elif mode == 'reset' %}Update password
        {% else %}Sign in{% endif %}
      </button>
    </form>
    {% endif %}

    <p class="auth-switch">
      {% if mode == 'register' %}
        Already have an account? <a href="/login">Sign in</a>
      {% elif mode in ('forgot', 'reset') %}
        <a href="/login">← Back to sign in</a>
      {% else %}
        New to ChartEdge.trade? <a href="/register">Create one</a>
      {% endif %}
    </p>

    {% if mode in ('register', 'login') %}
    <p class="auth-tos">
      By {{ 'creating an account' if mode == 'register' else 'signing in' }}, you agree to our
      <a href="/terms">Terms</a> &amp; <a href="/privacy">Privacy Policy</a>.
    </p>
    {% endif %}
  </div>
</div>
<script>
function togglePw(){
  var i = document.getElementById('auth-pw');
  var e = document.getElementById('auth-pw-eye');
  if (i.type === 'password') {
    i.type = 'text';
    e.innerHTML = '<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/>';
  } else {
    i.type = 'password';
    e.innerHTML = '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>';
  }
}
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Favorites HTML ────────────────────────────────────────────────────────────

FAVORITES_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Favorites — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
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
    .ind-tag { display: inline-block; font-size: 0.58rem; font-weight: 700; padding: 1px 5px; border-radius: 3px; vertical-align: middle; margin-left: 4px; letter-spacing: 0.05em; }
    .ind-tag-beta { background: #1a1a3a; color: #7b8cff; border: 1px solid #3a3a7a; }
    .empty { text-align: center; padding: 60px 24px; color: var(--muted); }
    .empty a { color: var(--accent); text-decoration: none; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
    {% for key, (iname, _, idesc, icat, _, itag) in indicators.items() %}
    <a class="ind-card" href="/indicators?kind={{ key }}">
      <div class="cat-tag">{{ icat }}{% if itag %} <span class="ind-tag ind-tag-{{ itag }}">{{ itag.upper() }}</span>{% endif %}</div>
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
<footer>© 2026 ChartEdge.trade · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


# ── Request HTML ──────────────────────────────────────────────────────────────

REQUEST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Request an Indicator — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
<footer>© 2026 ChartEdge.trade · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
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
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }

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
    .ind-tag { display: inline-block; font-size: 0.58rem; font-weight: 700; padding: 1px 5px; border-radius: 3px; vertical-align: middle; margin-left: 4px; letter-spacing: 0.05em; }
    .ind-tag-beta { background: #1a1a3a; color: #7b8cff; border: 1px solid #3a3a7a; }
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

    /* Smooth output panel */
    .output-wrap { max-height: 0; overflow: hidden; transition: max-height 0.38s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease; opacity: 0; }
    .output-wrap.visible { max-height: 1400px; opacity: 1; }
    .output-loading { text-align: center; padding: 28px; color: var(--muted); font-size: 0.85rem; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 8px; }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }

    /* Copy toast */
    .copy-toast { position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%) translateY(20px); background: #1a2d1a; border: 1px solid var(--green); color: var(--green); padding: 10px 22px; border-radius: 8px; font-size: .88rem; font-weight: 600; opacity: 0; pointer-events: none; transition: opacity .2s, transform .2s; z-index: 9999; }
    .copy-toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

    /* Page tabs */
    .page-tabs { display: flex; gap: 4px; margin-bottom: 24px; border-bottom: 1px solid var(--border); }
    .page-tab { background: none; border: none; color: var(--muted); font-family: monospace; font-size: .88rem; padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px; }
    .page-tab:hover { color: var(--text); }
    .page-tab.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }

    /* Tutorial */
    .tutorial-wrap { display: none; }
    .tutorial-wrap.active { display: block; }
    .video-container { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 28px; }
    .video-container video { display: block; width: 100%; height: auto; background: #000; }
    .steps { list-style: none; counter-reset: steps; display: flex; flex-direction: column; gap: 12px; }
    .steps li { counter-increment: steps; display: flex; gap: 14px; align-items: flex-start; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; font-size: .88rem; line-height: 1.6; }
    .steps li::before { content: counter(steps); background: var(--accent); color: #fff; font-weight: 700; font-size: .78rem; min-width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }
    .steps li strong { display: block; margin-bottom: 2px; color: var(--text); }
    .steps li span { color: var(--muted); }
    .steps a { color: var(--accent); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="hero">
  <h1>Free <span>TradingView</span> Indicators</h1>
  <p>Pick an indicator and copy the Pine Script — works on any TradingView chart, no paid plan needed.</p>
  <div style="margin:18px auto 0;max-width:560px;display:flex;align-items:center;gap:10px;padding:10px 14px;background:rgba(88,166,255,0.08);border:1px solid rgba(88,166,255,0.28);border-radius:8px;font-size:0.82rem;color:var(--text);text-align:left;">
    <span style="font-size:1rem;">⏱</span>
    <span><strong>Indicators refresh daily.</strong> Each copied script expires at midnight — come back the next day to grab an updated version.</span>
  </div>
</div>

<div class="container">

  <!-- Tab bar -->
  <div class="page-tabs">
    <button class="page-tab active" id="tab-indicators-btn" onclick="switchTab('indicators')">Indicators</button>
    <button class="page-tab" id="tab-tutorial-btn" onclick="switchTab('tutorial')">Tutorial</button>
  </div>

  <!-- Indicators tab -->
  <div id="tab-indicators">
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
    {% for key, (iname, _, idesc, icat, _, itag) in indicators.items() %}
    <a class="ind-card {{ 'active' if key == kind else '' }}"
       href="/indicators?kind={{ key }}&cat={{ category }}{{ ('&q=' + search) if search else '' }}"
       data-key="{{ key }}" data-name="{{ iname.lower() }}" data-desc="{{ idesc.lower() }}"
       onclick="selectIndicator(event, '{{ key }}')">
      <div class="cat-tag">{{ icat }}{% if itag %} <span class="ind-tag ind-tag-{{ itag }}">{{ itag.upper() }}</span>{% endif %}</div>
      <div class="ind-name">{{ iname }}</div>
      <div class="ind-desc">{{ idesc[:65] }}…</div>
    </a>
    {% endfor %}
    {% if not indicators %}
    <div class="no-results">No indicators match "{{ search }}".</div>
    {% endif %}
  </div>
  <div class="output-wrap" id="output-wrap">
    <div id="output-inner"></div>
  </div>
  </div><!-- end #tab-indicators -->

  <!-- Tutorial tab -->
  <div id="tab-tutorial" class="tutorial-wrap">
    <div class="video-container">
      <video src="/static/tutorial.mp4" muted controls playsinline preload="metadata"></video>
    </div>
    <ol class="steps">
      <li>
        <div><strong>Pick an indicator</strong><span>Click any card in the Indicators tab — VWAP, RSI, Bollinger Bands, etc. The script will appear below the grid.</span></div>
      </li>
      <li>
        <div><strong>Copy the Pine Script</strong><span>Hit the <em>Copy</em> button on the right side of the output panel. The script is copied to your clipboard.</span></div>
      </li>
      <li>
        <div><strong>Open TradingView</strong><span>Go to <a href="https://tradingview.com" target="_blank" rel="noopener">tradingview.com</a> and open any chart. A free account is all you need.</span></div>
      </li>
      <li>
        <div><strong>Open the Pine Script Editor</strong><span>At the bottom of the chart, click <em>Pine Editor</em>. If you don't see it, click the <em>+</em> icon in the bottom toolbar.</span></div>
      </li>
      <li>
        <div><strong>Paste &amp; Add to Chart</strong><span>Select all the existing code in the editor (Ctrl+A / Cmd+A), paste your copied script, then click <em>Add to chart</em> in the top-right of the editor.</span></div>
      </li>
      <li>
        <div><strong>Done!</strong><span>The indicator will appear on your chart. You can adjust the settings by clicking the gear icon next to the indicator name.</span></div>
      </li>
    </ol>
  </div>

</div>

<!-- Upgrade modal -->
<div class="modal-bg" id="upgrade-modal" onclick="if(event.target===this)this.style.display='none'">
  <div class="modal-box">
    <h2 style="margin-bottom:8px">Daily limit reached</h2>
    <p style="color:var(--muted);margin-bottom:24px">Upgrade to copy more indicators every day.</p>
    <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
      <a href="/pricing" class="plan-cta basic-cta">Basic — $9.99/mo<br><small>8 copies/day</small></a>
      <a href="/pricing" class="plan-cta pro-cta">Pro — $15.99/mo<br><small>Unlimited copies</small></a>
    </div>
    <button onclick="document.getElementById('upgrade-modal').style.display='none'"
            style="margin-top:16px;background:none;border:none;color:var(--muted);cursor:pointer;font-size:0.85rem;">
      Maybe later
    </button>
  </div>
</div>

<div class="copy-toast" id="copy-toast">✓ Copied to clipboard</div>

<footer>© 2026 ChartEdge.trade · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
const COPY_TOAST_ENABLED = {{ copy_toast }};

function showToast() {
  if (!COPY_TOAST_ENABLED) return;
  const t = document.getElementById('copy-toast');
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

function switchTab(name) {
  document.getElementById('tab-indicators').style.display = name === 'indicators' ? 'block' : 'none';
  document.getElementById('tab-tutorial').classList.toggle('active', name === 'tutorial');
  document.getElementById('tab-indicators-btn').classList.toggle('active', name === 'indicators');
  document.getElementById('tab-tutorial-btn').classList.toggle('active', name === 'tutorial');
}

let _currentKind = '{{ kind }}';
let _vwapOpts = {band1: true, band2: true, band3: false};
let _atrOpts  = {atr_avg: false};

async function selectIndicator(e, key) {
  e.preventDefault();
  if (key === _currentKind) return;
  _currentKind = key;
  history.pushState({}, '', '/indicators?kind=' + key);
  document.querySelectorAll('.ind-card').forEach(c => c.classList.toggle('active', c.dataset.key === key));
  await loadOutput(key);
}

async function loadOutput(key, extraParams) {
  const wrap = document.getElementById('output-wrap');
  const inner = document.getElementById('output-inner');
  inner.innerHTML = '<div class="output-loading"><span class="spinner"></span>Loading…</div>';
  if (!wrap.classList.contains('visible')) {
    wrap.classList.add('visible');
  }
  let url = '/api/indicator?kind=' + encodeURIComponent(key);
  if (extraParams) url += '&' + extraParams;
  const data = await fetch(url).then(r => r.json());
  if (!data.ok) { inner.innerHTML = '<div class="output-loading">Error loading indicator.</div>'; return; }
  inner.innerHTML = buildOutput(data);
  wrap.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

function buildOutput(d) {
  const blurred = d.user_plan === 'free';
  const badge   = d.copies_remaining < 0 ? '' :
    `<span class="copies-badge" id="copies-badge">${d.copies_remaining === 0 ? '0 copies left' : d.copies_remaining + ' copies left today'}</span>`;
  const overlay = blurred ? `
    <div class="pine-overlay" id="pine-overlay">
      <div class="overlay-icon">🔒</div>
      <div class="overlay-msg"><strong id="overlay-count">${d.copies_remaining}</strong> free copies left today</div>
      <button class="btn-copy" onclick="copyPine()">Reveal &amp; Copy</button>
      <a href="/pricing" class="overlay-upgrade">Upgrade for more →</a>
    </div>` : '';

  const vwapOpts = d.has_vwap_options ? `
    <div class="card options">
      <span style="color:var(--muted);font-size:.85rem;">Bands:</span>
      <label style="color:var(--green);margin-left:16px;"><input type="checkbox" id="band1" ${_vwapOpts.band1?'checked':''} onchange="reloadVwap()"> ±1 StDev</label>
      <label style="color:var(--red);margin-left:16px;"><input type="checkbox" id="band2" ${_vwapOpts.band2?'checked':''} onchange="reloadVwap()"> ±2 StDev</label>
      <label style="color:var(--purple);margin-left:16px;"><input type="checkbox" id="band3" ${_vwapOpts.band3?'checked':''} onchange="reloadVwap()"> ±3 StDev</label>
    </div>` : '';

  const atrOpts = d.has_atr_options ? `
    <div class="card options">
      <span style="color:var(--muted);font-size:.85rem;">Options:</span>
      <label style="color:var(--orange);margin-left:16px;"><input type="checkbox" id="atr_avg" ${_atrOpts.atr_avg?'checked':''} onchange="reloadAtr()"> Show 20-bar avg (orange)</label>
    </div>` : '';

  const howTo = d.how_to ? `
    <div class="how-to">
      <div class="how-to-title" onclick="toggleHowTo()">▶ How to use</div>
      <div class="how-to-body" id="howto-body">${d.how_to}</div>
    </div>` : '';

  const about = d.description ? `
    <div style="margin-top:14px;border-top:1px solid var(--border);padding-top:14px;">
      <div style="font-size:.75rem;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">About this indicator</div>
      <div style="color:var(--muted);font-size:.85rem;line-height:1.65;">${d.description}</div>
    </div>` : '';

  const heartBtn = `<button class="btn-copy ${d.is_favorited?'copied':''}" id="heart-btn" onclick="toggleFavorite('${d.kind}')">${d.is_favorited?'♥ Saved':'♡ Save'}</button>`;

  return `
    ${vwapOpts}${atrOpts}
    <div style="display:flex;gap:10px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">
      <button class="btn-copy" onclick="copyLink(this)">🔗 Copy link</button>
      ${heartBtn}
    </div>
    <div class="card ${d.has_vwap_options||d.has_atr_options?'output':''}">
      <div class="pine-label">
        <span>Pine Script v6 — works on any chart, no ticker needed</span>
        ${badge}
        <button class="btn-copy" id="copy-btn" onclick="copyPine()">Copy</button>
      </div>
      <div class="pine-wrap${blurred?' pine-blur':''}" id="pine-wrap">
        <textarea id="pine-out" readonly>${escHtml(d.pine_code)}</textarea>
        ${overlay}
      </div>
      ${howTo}${about}
    </div>`;
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function reloadVwap() {
  _vwapOpts.band1 = document.getElementById('band1').checked;
  _vwapOpts.band2 = document.getElementById('band2').checked;
  _vwapOpts.band3 = document.getElementById('band3').checked;
  const p = `band1=${_vwapOpts.band1?'on':'off'}&band2=${_vwapOpts.band2?'on':'off'}&band3=${_vwapOpts.band3?'on':'off'}`;
  await loadOutput('vwap', p);
}
async function reloadAtr() {
  _atrOpts.atr_avg = document.getElementById('atr_avg').checked;
  await loadOutput('atr', `atr_avg=${_atrOpts.atr_avg?'on':'off'}`);
}

async function copyPine() {
  const res  = await fetch('/api/copy', {method: 'POST'});
  const data = await res.json();
  if (!data.ok) { document.getElementById('upgrade-modal').style.display = 'flex'; return; }
  const wrap    = document.getElementById('pine-wrap');
  const overlay = document.getElementById('pine-overlay');
  if (wrap)    wrap.classList.remove('pine-blur');
  if (overlay) overlay.style.display = 'none';
  const ta = document.getElementById('pine-out');
  ta.select(); document.execCommand('copy');
  const btn = document.getElementById('copy-btn');
  btn.textContent = 'Copied!'; btn.classList.add('copied');
  setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  showToast();
  const badge = document.getElementById('copies-badge');
  const oc    = document.getElementById('overlay-count');
  if (data.remaining >= 0) {
    if (badge) badge.textContent = data.remaining + ' copies left today';
    if (oc)    oc.textContent    = data.remaining;
  } else { if (badge) badge.textContent = '∞'; }
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
  const body  = document.getElementById('howto-body');
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
// Auto-load if a kind was pre-selected via URL on page load
if (_currentKind) { loadOutput(_currentKind); }
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
  <title>ChartEdge.trade — Market Intelligence for Retail Traders</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

    nav {
      background: var(--bg2); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; justify-content: space-between;
    }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
""" + _NAV_CSS + """

    /* Ticker tape */
    .ticker-tape {
      background: var(--bg3); border-bottom: 1px solid var(--border);
      overflow: hidden; white-space: nowrap; padding: 7px 0; font-size: 0.75rem;
    }
    .ticker-inner { display: inline-block; animation: ticker-scroll 40s linear infinite; }
    .ticker-inner:hover { animation-play-state: paused; }
    @keyframes ticker-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .tick { display: inline-block; margin: 0 28px; color: var(--muted); }
    .tick .sym { color: var(--text); font-weight: 700; margin-right: 6px; }
    .tick .up { color: #3fb950; }
    .tick .down { color: #f85149; }
    @media(max-width:640px){.ticker-tape{display:none;}}

    /* Edge hero card — spotlight + animated candlestick */
    .edge-hero { max-width: 1180px; margin: 36px auto 8px; padding: 0 16px; }
    .edge-card {
      position: relative; overflow: hidden; border-radius: 16px;
      background:
        radial-gradient(ellipse 60% 50% at 50% 0%, rgba(88,166,255,0.10), transparent 70%),
        #050810;
      border: 1px solid #1a1f26;
      min-height: 460px;
      display: flex; flex-wrap: wrap;
      box-shadow: 0 20px 60px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.02) inset;
    }
    .edge-card::before {
      content: ''; position: absolute; inset: -30%;
      background: radial-gradient(circle 340px at var(--mx,30%) var(--my,25%),
                  rgba(88,166,255,.18), rgba(88,166,255,.05) 35%, transparent 60%);
      pointer-events: none;
      opacity: var(--spot-op, 1);
      transition: opacity .4s ease;
    }
    .edge-card::after {
      content: ''; position: absolute; inset: 0; pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
      background-size: 44px 44px;
      -webkit-mask-image: linear-gradient(180deg, transparent, #000 30%, #000 70%, transparent);
              mask-image: linear-gradient(180deg, transparent, #000 30%, #000 70%, transparent);
    }
    .edge-left {
      flex: 1 1 360px; padding: 52px 44px; text-align: left;
      position: relative; z-index: 3;
      display: flex; flex-direction: column; justify-content: center;
    }
    .edge-eyebrow {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 5px 14px; margin-bottom: 22px; width: fit-content;
      background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
      border-radius: 20px; font-size: .72rem; color: #8b949e;
    }
    .edge-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: #3fb950; animation: pulse 2s infinite; }
    .edge-title {
      font-size: clamp(2rem, 4.4vw, 3.1rem); line-height: 1.05;
      letter-spacing: -.025em; font-weight: 800; margin-bottom: 18px;
      background: linear-gradient(180deg, #fafafa 0%, #8a8f97 100%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .edge-title .accent {
      background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 100%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .edge-sub { color: #b6bcc6; font-size: 1rem; line-height: 1.7; max-width: 460px; margin-bottom: 30px; }
    .edge-btns { display: flex; gap: 12px; flex-wrap: wrap; }
    .edge-btn-primary {
      background: linear-gradient(135deg, #58a6ff 0%, #4493f8 100%); color: #fff;
      padding: 13px 26px; border-radius: 9px; font-weight: 700; font-size: .92rem;
      text-decoration: none; box-shadow: 0 8px 24px rgba(88,166,255,.28);
      transition: transform .2s, box-shadow .2s;
    }
    .edge-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(88,166,255,.4); }
    .edge-btn-ghost {
      color: #e6edf3; padding: 13px 26px; border-radius: 9px; font-size: .92rem;
      border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.02);
      text-decoration: none; transition: background .2s, border-color .2s;
    }
    .edge-btn-ghost:hover { background: rgba(255,255,255,.06); border-color: rgba(255,255,255,.22); }

    .edge-right {
      flex: 1 1 380px; position: relative; z-index: 2;
      min-height: 320px; padding: 28px 32px 28px 0;
      display: flex; align-items: stretch;
    }
    .edge-chart { width: 100%; height: 100%; min-height: 320px; position: relative; }
    .edge-chart svg { width: 100%; height: 100%; display: block; overflow: visible; }
    .edge-ticker {
      position: absolute; top: 12px; left: 8px; right: 12px;
      display: flex; align-items: center; justify-content: space-between;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .82rem; z-index: 2;
    }
    .edge-ticker .sym { color: #e6edf3; font-weight: 700; margin-right: 6px; }
    .edge-ticker .px  { color: #58a6ff; }
    .edge-ticker .chg { font-size: .72rem; padding: 2px 8px; border-radius: 6px; font-weight: 700; }
    @media (max-width: 760px) {
      .edge-card { min-height: auto; }
      .edge-left { padding: 36px 28px 8px; }
      .edge-right { padding: 0 16px 28px; min-height: 240px; }
      .edge-chart { min-height: 220px; }
    }

    /* Shader hero — bull/bear cosmic WebGL2 background */
    .shader-hero {
      position: relative; width: 100%;
      min-height: calc(100vh - 56px);
      display: flex; align-items: center; justify-content: center;
      overflow: hidden; background: #000; border-bottom: 1px solid var(--border);
    }
    .shader-hero canvas {
      position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none;
    }
    .shader-hero::after {
      content: ''; position: absolute; inset: 0; pointer-events: none;
      background:
        radial-gradient(ellipse 70% 55% at 50% 50%, transparent 30%, rgba(0,0,0,0.55) 100%),
        linear-gradient(180deg, transparent 60%, rgba(13,17,23,0.85) 100%);
    }
    .shader-overlay {
      position: relative; z-index: 2;
      max-width: 760px;
      padding: 48px 40px 44px;
      text-align: center; color: #fff;
      background: linear-gradient(180deg, rgba(255,255,255,.10) 0%, rgba(255,255,255,.04) 100%);
      -webkit-backdrop-filter: blur(14px) saturate(140%);
              backdrop-filter: blur(14px) saturate(140%);
      border: 1px solid rgba(255,255,255,.14);
      border-radius: 22px;
      box-shadow:
        0 24px 60px rgba(0,0,0,.45),
        inset 0 1px 0 rgba(255,255,255,.10);
      margin: 0 24px;
    }
    .shader-badge {
      display: inline-flex; align-items: center; gap: 10px;
      padding: 7px 16px; margin-bottom: 28px;
      background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.14);
      -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
      border-radius: 999px; font-size: .82rem; color: #d6dde6;
      opacity: 0; animation: shader-fade-down .8s .1s ease-out forwards;
    }
    .shader-badge .icon { color: #3fb950; }
    .shader-headline {
      font-weight: 800; letter-spacing: -.025em; line-height: 1.02;
      font-size: clamp(2.6rem, 6vw, 5rem); margin-bottom: 22px;
    }
    .shader-headline .bear, .shader-headline .bull { display: block; }
    .shader-headline .bear {
      background: linear-gradient(180deg, #ffb1ad 0%, #f85149 55%, #b8302a 100%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
      opacity: 0; animation: shader-fade-up .9s .25s ease-out forwards;
    }
    .shader-headline .bull {
      background: linear-gradient(180deg, #98ecaa 0%, #3fb950 55%, #237a36 100%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
      opacity: 0; animation: shader-fade-up .9s .45s ease-out forwards;
    }
    .shader-sub {
      max-width: 680px; margin: 0 auto 32px;
      color: rgba(230,237,243,0.85); font-size: clamp(1rem, 1.5vw, 1.18rem);
      line-height: 1.65; font-weight: 400;
      opacity: 0; animation: shader-fade-up .9s .65s ease-out forwards;
    }
    .shader-btns {
      display: flex; gap: 14px; justify-content: center; flex-wrap: wrap;
      opacity: 0; animation: shader-fade-up .9s .85s ease-out forwards;
    }
    .shader-btn-primary {
      padding: 14px 30px; border-radius: 999px;
      font-weight: 700; font-size: .98rem; color: #06120a;
      background: linear-gradient(135deg, #56d364, #3fb950);
      text-decoration: none; box-shadow: 0 12px 36px rgba(63,185,80,.35);
      transition: transform .2s, box-shadow .2s;
    }
    .shader-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 16px 44px rgba(63,185,80,.5); }
    .shader-btn-ghost {
      padding: 14px 30px; border-radius: 999px;
      font-weight: 600; font-size: .98rem; color: #fff;
      background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.18);
      -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
      text-decoration: none; transition: background .2s, border-color .2s, transform .2s;
    }
    .shader-btn-ghost:hover { background: rgba(255,255,255,0.12); border-color: rgba(255,255,255,0.34); transform: translateY(-2px); }
    @keyframes shader-fade-down {
      from { opacity: 0; transform: translateY(-12px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shader-fade-up {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 640px) {
      .shader-hero { min-height: calc(100vh - 52px); }
      .shader-overlay { padding: 32px 22px 30px; margin: 0 16px; border-radius: 18px; }
    }
    @media (prefers-reduced-motion: reduce) {
      .shader-badge, .shader-headline .bear, .shader-headline .bull, .shader-sub, .shader-btns { animation: none; opacity: 1; }
    }

    /* Hero */
    .hero {
      text-align: center; padding: 40px 24px 56px;
      background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(88,166,255,0.04) 0%, transparent 70%), var(--bg);
      border-bottom: 1px solid var(--border);
    }
    .hero-eyebrow {
      display: inline-flex; align-items: center; gap: 8px; margin-bottom: 24px;
      background: var(--bg2); border: 1px solid var(--border);
      padding: 5px 16px; border-radius: 20px; font-size: 0.75rem; color: var(--muted);
    }
    .hero-eyebrow .dot { width: 7px; height: 7px; border-radius: 50%; background: #3fb950; display: inline-block; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
    .hero h1 { font-size: 2.8rem; line-height: 1.18; margin-bottom: 20px; letter-spacing: -0.02em; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 1rem; max-width: 560px; margin: 0 auto 32px; line-height: 1.75; }
    .hero-btns { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 52px; }
    .btn-primary {
      background: var(--accent); color: #fff; border: none;
      padding: 12px 30px; border-radius: 8px; font-family: monospace;
      font-size: 0.95rem; font-weight: bold; text-decoration: none; cursor: pointer;
    }
    .btn-primary:hover { opacity: 0.88; }
    .btn-secondary {
      background: var(--bg2); color: var(--text); border: 1px solid var(--border);
      padding: 12px 30px; border-radius: 8px; font-family: monospace;
      font-size: 0.95rem; text-decoration: none;
    }
    .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }

    /* Tool cards hero grid */
    .hero-tools { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; max-width: 940px; margin: 0 auto; }
    .hero-tool {
      --ht-c1: #58a6ff; --ht-c2: #79c0ff; --ht-glow: rgba(88,166,255,.20);
      background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
      padding: 20px 22px; text-align: left; text-decoration: none;
      position: relative; overflow: hidden;
      transition: transform .3s cubic-bezier(.2,.7,.3,1), border-color .25s, box-shadow .25s;
    }
    .hero-tool::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, var(--ht-c1), var(--ht-c2));
      opacity: .95;
    }
    .hero-tool::after {
      content: ''; position: absolute; inset: -1px;
      border-radius: 12px; pointer-events: none;
      opacity: 0; transition: opacity .3s;
      background: radial-gradient(75% 65% at 50% 0%, var(--ht-glow), transparent 65%);
    }
    .hero-tool:hover {
      transform: translateY(-4px);
      border-color: var(--ht-c1);
      box-shadow: 0 14px 36px rgba(0,0,0,.28), 0 0 0 1px var(--ht-c1);
    }
    .hero-tool:hover::after { opacity: 1; }
    .hero-tool-icon {
      font-size: 1.55rem; margin-bottom: 10px;
      display: inline-block; transform-origin: center;
      transition: transform .35s cubic-bezier(.3,1.4,.5,1);
    }
    .hero-tool:hover .hero-tool-icon { transform: scale(1.18) rotate(-6deg); }
    .hero-tool-name { font-size: 0.9rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
    .hero-tool-desc { font-size: 0.76rem; color: var(--muted); line-height: 1.5; }
    .hero-tool-tag { display: inline-block; font-size: 0.63rem; font-weight: 700; padding: 1px 7px; border-radius: 8px; margin-top: 10px; }
    .hero-tool.blue   { --ht-c1: #58a6ff; --ht-c2: #79c0ff; --ht-glow: rgba(88,166,255,.22); }
    .hero-tool.green  { --ht-c1: #3fb950; --ht-c2: #56d364; --ht-glow: rgba(63,185,80,.22); }
    .hero-tool.orange { --ht-c1: #e3b341; --ht-c2: #ffa657; --ht-glow: rgba(255,166,87,.22); }
    .hero-tool.purple { --ht-c1: #bc8cff; --ht-c2: #d2a8ff; --ht-glow: rgba(188,140,255,.22); }
    .tag-free   { background: #1f2d1f; color: #3fb950; }
    .tag-basic  { background: #1a2d3d; color: #58a6ff; }
    .tag-pro    { background: #2d1f10; color: #e3b341; }

    /* Stats strip */
    .stats-strip {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      background: linear-gradient(180deg, var(--bg2) 0%, var(--bg) 100%);
      border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
      padding: 36px 24px; gap: 0;
    }
    .stat-item { text-align: center; padding: 8px 24px; border-right: 1px solid var(--border); position: relative; }
    .stat-item:last-child { border-right: none; }
    .stat-num {
      font-size: clamp(1.8rem, 3.2vw, 2.6rem); font-weight: 800;
      background: linear-gradient(180deg, #79c0ff 0%, #58a6ff 60%, #1f6feb 100%);
      -webkit-background-clip: text; background-clip: text; color: transparent;
      letter-spacing: -.025em; font-variant-numeric: tabular-nums; display: inline-block;
    }
    .stat-label { font-size: .8rem; color: var(--muted); margin-top: 6px; text-transform: uppercase; letter-spacing: .05em; font-weight: 500; }
    @media (max-width: 640px) {
      .stat-item { border-right: none; border-bottom: 1px solid var(--border); padding: 16px; }
      .stat-item:last-child { border-bottom: none; }
    }

    /* Scroll reveal */
    .reveal { opacity: 0; transform: translateY(36px); transition: opacity .7s cubic-bezier(.2,.7,.3,1), transform .7s cubic-bezier(.2,.7,.3,1); transition-delay: var(--rd, 0s); }
    .reveal.in { opacity: 1; transform: translateY(0); }
    @media (prefers-reduced-motion: reduce) { .reveal { opacity: 1; transform: none; transition: none; } }

    /* How it works */
    .hiw {
      background: linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
      border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
    }
    .hiw-inner { max-width: 1040px; margin: 0 auto; padding: 64px 24px; }
    .hiw-inner h2 { font-size: 1.6rem; text-align: center; margin-bottom: 44px; }
    .hiw-inner h2 span { color: var(--accent); }
    .hiw-grid { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 18px; align-items: stretch; }
    .hiw-step {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 14px;
      padding: 32px 24px 24px; position: relative; text-align: left;
      transition: border-color .25s, transform .25s, box-shadow .25s;
    }
    .hiw-step:hover { border-color: var(--accent); transform: translateY(-3px); box-shadow: 0 14px 32px rgba(0,0,0,.28); }
    .hiw-num {
      position: absolute; top: -18px; left: 22px;
      width: 40px; height: 40px; border-radius: 50%;
      background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
      color: #fff; font-weight: 800; font-size: 1rem;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 6px 18px rgba(88,166,255,.45);
    }
    .hiw-icon { font-size: 1.7rem; margin: 10px 0 12px; }
    .hiw-step h3 { font-size: 1.02rem; font-weight: 700; margin-bottom: 6px; color: var(--text); }
    .hiw-step p { color: var(--muted); font-size: .86rem; line-height: 1.6; }
    .hiw-arrow {
      display: flex; align-items: center; justify-content: center;
      color: var(--muted); font-size: 1.6rem; opacity: .5;
    }
    @media (max-width: 760px) {
      .hiw-grid { grid-template-columns: 1fr; gap: 22px; }
      .hiw-arrow { transform: rotate(90deg); }
    }

    /* Section */
    .section { max-width: 900px; margin: 0 auto; padding: 60px 24px; }
    .section h2 { font-size: 1.4rem; margin-bottom: 32px; text-align: center; }
    .section h2 span { color: var(--accent); }

    /* Intelligence grid */
    .intel-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
    .intel-card {
      background: var(--bg2); border: 1px solid var(--border); border-radius: 10px;
      padding: 24px; position: relative; overflow: hidden; transition: border-color 0.15s;
    }
    .intel-card:hover { border-color: var(--accent); }
    .intel-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
    .intel-card.blue::before   { background: linear-gradient(90deg, #58a6ff, #79c0ff); }
    .intel-card.green::before  { background: linear-gradient(90deg, #3fb950, #56d364); }
    .intel-card.orange::before { background: linear-gradient(90deg, #e3b341, #ffa657); }
    .intel-card.purple::before { background: linear-gradient(90deg, #bc8cff, #d2a8ff); }
    .intel-card.red::before    { background: linear-gradient(90deg, #f85149, #ff7b72); }
    .intel-tag { display: inline-block; font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; margin-bottom: 12px; }
    .intel-card-icon { font-size: 1.6rem; margin-bottom: 10px; }
    .intel-card h3 { font-size: 0.95rem; margin-bottom: 6px; }
    .intel-card p { color: var(--muted); font-size: 0.82rem; line-height: 1.55; }

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

    /* Pricing teaser */
    .pricing-teaser {
      background: linear-gradient(135deg, var(--bg2) 0%, var(--bg) 100%);
      border: 1px solid var(--border); border-radius: 12px;
      padding: 40px 32px; text-align: center;
    }
    .pricing-teaser h2 { font-size: 1.4rem; margin-bottom: 8px; }
    .pricing-teaser h2 span { color: var(--accent); }
    .pricing-teaser p { color: var(--muted); font-size: 0.9rem; margin-bottom: 24px; }
    .pricing-row { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 16px; }
    .price-pill {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 14px 24px; min-width: 140px; text-align: center;
    }
    .price-pill.featured { border-color: var(--accent); background: rgba(88,166,255,0.06); }
    .price-pill .p-name { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
    .price-pill .p-price { font-size: 1.4rem; font-weight: 800; }
    .price-pill .p-price span { font-size: 0.8rem; font-weight: 400; color: var(--muted); }
    .trial-note { color: var(--muted); font-size: 0.8rem; }
    .trial-note strong { color: #3fb950; }

    /* FAQ */
    .faq-list { margin-top: 0; }
    .faq-item { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
    .faq-q { padding: 16px 20px; cursor: pointer; font-size: 0.9rem; font-weight: 600; display: flex; justify-content: space-between; align-items: center; user-select: none; }
    .faq-q:hover { background: var(--bg3); }
    .faq-chevron { color: var(--muted); font-size: 0.8rem; transition: transform 0.28s cubic-bezier(0.4,0,0.2,1); }
    .faq-a { max-height: 0; overflow: hidden; padding: 0 20px; color: var(--muted); font-size: 0.85rem; line-height: 1.65; transition: max-height 0.32s cubic-bezier(0.4,0,0.2,1), padding 0.2s ease; }
    .faq-item.open .faq-a { max-height: 400px; padding: 0 20px 16px; }
    .faq-item.open .faq-chevron { transform: rotate(180deg); }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<!-- Bull/bear shader hero -->
<section class="shader-hero">
  <canvas id="shader-canvas"></canvas>
  <div class="shader-overlay">
    <h1 class="shader-headline">
      <span class="bear">Trade the Bear.</span>
      <span class="bull">Ride the Bull.</span>
    </h1>
    <p class="shader-sub">20+ Pine Script indicators, real-time options flow, gamma exposure, insider trades, AI volatility forecasts — and so much more.</p>
    <div class="shader-btns">
      {% if not current_user %}<a class="shader-btn-primary" href="/register">Get Started Free</a>{% endif %}
      <a class="shader-btn-ghost" href="/indicators">Browse Indicators →</a>
    </div>
  </div>
</section>
<script>
(function(){
  var canvas = document.getElementById('shader-canvas');
  if (!canvas) return;
  var gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false });
  if (!gl) { canvas.style.display = 'none'; return; }

  var VS = `#version 300 es
precision highp float;
in vec4 position;
void main(){ gl_Position = position; }`;

  var FS = `#version 300 es
precision highp float;
out vec4 O;
uniform vec2 resolution;
uniform float time;
#define FC gl_FragCoord.xy
#define T time
#define R resolution
#define MN min(R.x,R.y)

float rnd(vec2 p){ p=fract(p*vec2(12.9898,78.233)); p+=dot(p,p+34.56); return fract(p.x*p.y); }
float noise(in vec2 p){
  vec2 i=floor(p), f=fract(p), u=f*f*(3.-2.*f);
  float a=rnd(i), b=rnd(i+vec2(1,0)), c=rnd(i+vec2(0,1)), d=rnd(i+1.);
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p){
  float t=.0, a=1.; mat2 m=mat2(1.,-.5,.2,1.2);
  for(int i=0;i<5;i++){ t+=a*noise(p); p*=2.*m; a*=.5; }
  return t;
}
float clouds(vec2 p){
  float d=1., t=.0;
  for(float i=.0;i<3.;i++){
    float a=d*fbm(i*10.+p.x*.2+.2*(1.+i)*p.y+d+i*i+p);
    t=mix(t,d,a); d=a; p*=2./(i+1.);
  }
  return t;
}
float hash11(float n){
  // Hoskins-style hash — stable for large inputs (no sin precision loss)
  n = fract(n * 0.1031);
  n *= n + 33.33;
  n *= n + n;
  return fract(n);
}
float sdBox(vec2 p, vec2 b){
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// One layer of scrolling candlesticks
vec3 candleLayer(vec2 uv, float dx, float bw, float ww, float speed, vec2 offset, float seed, vec3 bull, vec3 bear, float alpha){
  float xs = uv.x + T * speed + offset.x;
  float i = floor(xs / dx);
  float localX = xs - (i + 0.5) * dx;

  float o = (hash11(i*1.3 + seed) - 0.5) * 0.55;
  float c = o + (hash11(i*2.7 + seed + 9.1) - 0.45) * 0.42;
  float h = max(o, c) + hash11(i*3.5 + seed + 19.7) * 0.18 + 0.012;
  float l = min(o, c) - hash11(i*4.9 + seed + 29.3) * 0.18 - 0.012;
  vec3 cc = c > o ? bull : bear;

  float yShift = offset.y + (hash11(i*0.7 + seed + 3.0) - 0.5) * 0.05;
  float bodyCenter = (o + c) * 0.5 + yShift;
  float bodyHalfH = max(abs(c - o) * 0.5, 0.006);
  float dBody = sdBox(vec2(localX, uv.y - bodyCenter), vec2(bw, bodyHalfH));

  float wickCenter = (h + l) * 0.5 + yShift;
  float wickHalfH = (h - l) * 0.5;
  float dWick = sdBox(vec2(localX, uv.y - wickCenter), vec2(ww, wickHalfH));

  float d = min(dBody, dWick);
  float aa = 1.5 / MN;
  float mask = 1.0 - smoothstep(-aa, aa, d);
  float glow = exp(-d * 38.0) * 0.30;
  return cc * (mask * 0.90 + glow) * alpha;
}

void main(void){
  vec2 uv = (FC - .5*R) / MN;
  vec2 st = uv * vec2(2, 1);

  // Bull/bear cosmic cloud background
  float bg = clouds(vec2(st.x + T*.5, -st.y));
  float bullbear = smoothstep(-1.0, 1.0, st.x);
  vec3 bear = vec3(1.0, 0.32, 0.28);
  vec3 bull = vec3(0.25, 0.93, 0.40);
  vec3 cloudCol = mix(bear, bull, bullbear) * bg * 0.24;
  vec3 col = cloudCol;

  // Back layer: small dim candles, slow scroll
  col += candleLayer(uv * 1.4, 0.13, 0.018, 0.0030, 0.045, vec2(0.0, 0.15), 11.0, bull, bear, 0.45);
  // Mid layer: medium candles
  col += candleLayer(uv * 1.0, 0.20, 0.028, 0.0045, 0.085, vec2(7.3, -0.05), 23.0, bull, bear, 0.75);
  // Front layer: larger crisp candles, faster
  col += candleLayer(uv * 0.75, 0.30, 0.045, 0.0070, 0.140, vec2(3.1, 0.10), 47.0, bull, bear, 0.95);

  O = vec4(col, 1);
}`;

  function mk(type, src){
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(s));
    }
    return s;
  }
  var prog = gl.createProgram();
  gl.attachShader(prog, mk(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, mk(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(prog));
    canvas.style.display = 'none';
    return;
  }
  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,1,-1,-1,1,1,1,-1]), gl.STATIC_DRAW);
  var posLoc = gl.getAttribLocation(prog, 'position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
  gl.useProgram(prog);
  var uRes  = gl.getUniformLocation(prog, 'resolution');
  var uTime = gl.getUniformLocation(prog, 'time');

  function resize(){
    var dpr = Math.max(1, 0.5 * (window.devicePixelRatio || 1));
    var r = canvas.getBoundingClientRect();
    canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
    canvas.height = Math.max(1, Math.floor(r.height * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  resize();
  window.addEventListener('resize', resize);

  function loop(t){
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1f(uTime, t * 0.001);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
})();
</script>

<!-- Trust bar -->
<div style="background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 24px;display:flex;justify-content:center;gap:32px;flex-wrap:wrap;">
  <span style="font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:6px;"><span style="color:#3fb950;">✓</span> No credit card to start</span>
  <span style="font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:6px;"><span style="color:#3fb950;">✓</span> 7-day free trial on paid plans</span>
  <span style="font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:6px;"><span style="color:#3fb950;">✓</span> Cancel anytime</span>
  <span style="font-size:.8rem;color:var(--muted);display:flex;align-items:center;gap:6px;"><span style="color:#3fb950;">✓</span> Live market data</span>
</div>

<!-- Stats strip -->
<div class="stats-strip">
  <div class="stat-item reveal">
    <div class="stat-num" data-count="25" data-suffix="+">0+</div>
    <div class="stat-label">Market intelligence tools</div>
  </div>
  <div class="stat-item reveal" style="--rd:.08s">
    <div class="stat-num" data-count="20" data-suffix="+">0+</div>
    <div class="stat-label">Pine Script indicators</div>
  </div>
  <div class="stat-item reveal" style="--rd:.16s">
    <div class="stat-num" data-count="15.99" data-decimals="2" data-prefix="$">$0.00</div>
    <div class="stat-label">Pro plan / month</div>
  </div>
  <div class="stat-item reveal" style="--rd:.24s">
    <div class="stat-num" data-count="7">0</div>
    <div class="stat-label">Day free trial</div>
  </div>
</div>

<!-- How it works -->
<div class="hiw">
  <div class="hiw-inner">
    <h2 class="reveal">Get started in <span>3 steps</span></h2>
    <div class="hiw-grid">
      <div class="hiw-step reveal" style="--rd:0s">
        <div class="hiw-num">1</div>
        <div class="hiw-icon">🎯</div>
        <h3>Pick an indicator</h3>
        <p>Browse 20+ free Pine Script indicators — VWAP, MACD, Bollinger Bands, and more.</p>
      </div>
      <div class="hiw-arrow reveal" style="--rd:.1s">→</div>
      <div class="hiw-step reveal" style="--rd:.15s">
        <div class="hiw-num">2</div>
        <div class="hiw-icon">📋</div>
        <h3>Copy the script</h3>
        <p>Click any indicator card to instantly copy the Pine Script v6 code to your clipboard.</p>
      </div>
      <div class="hiw-arrow reveal" style="--rd:.25s">→</div>
      <div class="hiw-step reveal" style="--rd:.3s">
        <div class="hiw-num">3</div>
        <div class="hiw-icon">⚡</div>
        <h3>Paste in TradingView</h3>
        <p>Open your chart, paste in the Pine Script editor, and click Add to chart — done.</p>
      </div>
    </div>
  </div>
</div>

<!-- Intelligence tools -->
<div class="section">
  <h2>Everything in one <span>dashboard</span></h2>
  <div class="intel-grid">
    <div class="intel-card orange">
      <div class="intel-card-icon">🌊</div>
      <span class="intel-tag tag-pro">Pro</span>
      <h3>Options Flow</h3>
      <p>Track net call vs put activity across any options chain. Visualize where large traders are positioning before the move happens.</p>
    </div>
    <div class="intel-card blue">
      <div class="intel-card-icon">🏛</div>
      <span class="intel-tag tag-pro">Pro</span>
      <h3>Insider Trading</h3>
      <p>SEC Form 4 filings from major S&P 500 companies plus Congress STOCK Act disclosures. Filter by name, ticker, or source.</p>
    </div>
    <div class="intel-card green">
      <div class="intel-card-icon">📊</div>
      <span class="intel-tag tag-basic">Basic</span>
      <h3>Gamma Exposure (GEX)</h3>
      <p>Black-Scholes powered dealer hedging calculation from live options chains. Pinpoints price levels where the market tends to stall or reverse.</p>
    </div>
    <div class="intel-card blue">
      <div class="intel-card-icon">🌅</div>
      <span class="intel-tag tag-pro">Pro</span>
      <h3>Pre-Market Scanner</h3>
      <p>Catch gap ups and gap downs before the open. Filter by sector, float, and volume to find the best setups before 9:30.</p>
    </div>
    <div class="intel-card purple">
      <div class="intel-card-icon">🧮</div>
      <span class="intel-tag tag-basic">Basic</span>
      <h3>Greeks Dashboard</h3>
      <p>Full options chain with Delta, Gamma, Theta, Vega, and Rho for any ticker. Understand your risk before you enter.</p>
    </div>
    <div class="intel-card orange">
      <div class="intel-card-icon">📉</div>
      <span class="intel-tag tag-basic">Basic</span>
      <h3>Volatility Forecast</h3>
      <p>Realized vol, EWMA, and IV vs RV comparison charts. Regime detection flags Low / Medium / High / Extreme conditions in real time.</p>
    </div>
    <div class="intel-card red">
      <div class="intel-card-icon">🌍</div>
      <span class="intel-tag tag-free">Free</span>
      <h3>Market Heatmap</h3>
      <p>D3 squarified treemap of S&P 500 sectors. See at a glance which sectors are leading, lagging, and rotating.</p>
    </div>
    <div class="intel-card green">
      <div class="intel-card-icon">📅</div>
      <span class="intel-tag tag-free">Free</span>
      <h3>Earnings & Dividends</h3>
      <p>Upcoming earnings with EPS and revenue estimates. Ex-dividend dates for 140+ income tickers across the next 60 days.</p>
    </div>
    <div class="intel-card blue">
      <div class="intel-card-icon">🇺🇸</div>
      <span class="intel-tag tag-basic">Basic</span>
      <h3>Trump Tracker</h3>
      <p>Chart any asset against Trump-related news events with timestamped markers. See exactly how markets reacted to each announcement.</p>
    </div>
    <div class="intel-card purple">
      <div class="intel-card-icon">🌑</div>
      <span class="intel-tag tag-pro">Pro</span>
      <h3>Dark Pool Flow</h3>
      <p>Weekly off-exchange institutional block trade volume from FINRA ATS data. Spot where big money is quietly accumulating or distributing.</p>
    </div>
    <div class="intel-card orange">
      <div class="intel-card-icon">⚡</div>
      <span class="intel-tag tag-pro">Pro</span>
      <h3>Backtester</h3>
      <p>Test RSI, MACD, Bollinger Bands, EMA cross, and SMA cross strategies on any ticker. Get return, win rate, Sharpe ratio, and equity curve.</p>
    </div>
  </div>
</div>

<hr class="divider">

<!-- Crypto section -->
<div style="background:linear-gradient(135deg,#0d1a2d 0%,#0d1117 60%,#1a0d2d 100%);border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:60px 24px;">
  <div style="max-width:900px;margin:0 auto;">
    <div style="text-align:center;margin-bottom:36px;">
      <span style="background:#1a0d2d;border:1px solid #6e40c9;color:#bc8cff;font-size:.72rem;font-weight:700;padding:3px 12px;border-radius:20px;letter-spacing:.06em;">CRYPTO</span>
      <h2 style="font-size:1.5rem;margin-top:14px;margin-bottom:10px;">Crypto intelligence, <span style="color:#bc8cff;">built in.</span></h2>
      <p style="color:var(--muted);font-size:.9rem;max-width:540px;margin:0 auto;line-height:1.7;">From fear & greed to funding rates and liquidation maps — all crypto data via Hyperliquid and CoinGecko, no geo-blocks.</p>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;">
      <a href="/crypto/feargreed" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">😰</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">Fear &amp; Greed</div>
        <div style="font-size:.75rem;color:var(--muted);">Crypto sentiment index · Free</div>
      </a>
      <a href="/crypto/heatmap" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">🗺</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">Crypto Heatmap</div>
        <div style="font-size:.75rem;color:var(--muted);">Market cap & 24h % change · Free</div>
      </a>
      <a href="/crypto/dominance" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">₿</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">BTC Dominance</div>
        <div style="font-size:.75rem;color:var(--muted);">Altcoin cycle tracker · Basic+</div>
      </a>
      <a href="/crypto/funding" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">💰</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">Funding Rates</div>
        <div style="font-size:.75rem;color:var(--muted);">Hyperliquid perps · Basic+</div>
      </a>
      <a href="/crypto/onchain" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">⛓</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">On-Chain Metrics</div>
        <div style="font-size:.75rem;color:var(--muted);">Active addresses, tx vol · Basic+</div>
      </a>
      <a href="/crypto/liquidations" style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:20px;text-decoration:none;transition:border-color .15s;" onmouseover="this.style.borderColor='#bc8cff'" onmouseout="this.style.borderColor='#21262d'">
        <div style="font-size:1.3rem;margin-bottom:8px;">💥</div>
        <div style="font-size:.88rem;font-weight:700;color:var(--text);margin-bottom:3px;">Liquidation Map</div>
        <div style="font-size:.75rem;color:var(--muted);">Funding rate history · Pro</div>
      </a>
    </div>
  </div>
</div>

<hr class="divider">

<!-- AI section -->
<div style="background:linear-gradient(135deg,#0d1f33 0%,#0d1117 60%,#0d2210 100%);border-top:1px solid var(--border);border-bottom:1px solid var(--border);padding:64px 24px;">
  <div style="max-width:860px;margin:0 auto;">
    <div style="display:flex;align-items:center;gap:10px;justify-content:center;margin-bottom:12px;">
      <span style="background:#0d3349;border:1px solid #1f6feb;color:#58a6ff;font-size:.72rem;font-weight:700;padding:3px 12px;border-radius:20px;letter-spacing:.06em;">AI-POWERED</span>
    </div>
    <h2 style="font-size:1.6rem;text-align:center;margin-bottom:12px;">Not just indicators. <span style="color:#58a6ff;">Machine learning</span> built in.</h2>
    <p style="text-align:center;color:var(--muted);font-size:.93rem;max-width:600px;margin:0 auto 48px;line-height:1.7;">The forecast engine running behind ChartEdge.trade was trained on years of real market data using an LSTM neural network — the same architecture used in institutional quant research.</p>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:20px;">
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">🧠</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">LSTM Neural Network</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">Long Short-Term Memory networks are purpose-built for time-series data. The model learns temporal patterns in price, volume, and volatility that simple rules-based indicators miss entirely.</p>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">📉</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">Trained on Real Market Data</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">Trained across hundreds of tickers and multiple years of 5-minute intraday bars — including bull runs, crashes, and choppy consolidation. The model has seen it all.</p>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">🔁</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">Walk-Forward Backtested</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">Validated using walk-forward backtesting — training only on past data, never peeking at the future. No curve-fitting. No cherry-picked results. Just honest out-of-sample performance.</p>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">📊</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">42 Engineered Features</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">The model ingests 42 hand-crafted features per bar — spanning momentum, volume profile, volatility regime, ATR, VWAP deviation, and rolling return distributions.</p>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">🎯</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">Confidence Bands</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">Every forecast comes with uncertainty bounds computed via Monte Carlo dropout — so you see not just where volatility is heading, but how confident the model is in that prediction.</p>
      </div>
      <div style="background:rgba(255,255,255,.03);border:1px solid #21262d;border-radius:10px;padding:24px;">
        <div style="font-size:1.5rem;margin-bottom:10px;">⚡</div>
        <h3 style="font-size:.95rem;margin-bottom:6px;">Live Inference</h3>
        <p style="color:var(--muted);font-size:.82rem;line-height:1.55;">Forecasts are generated on demand using fresh market data pulled in real time. The output is a Pine Script indicator you paste directly into TradingView — no coding required.</p>
      </div>
    </div>
    <p style="text-align:center;color:var(--muted);font-size:.78rem;margin-top:32px;">Not financial advice. Past model performance does not guarantee future results.</p>
  </div>
</div>

<hr class="divider">

<!-- Pricing teaser -->
<div class="section">
  <div class="pricing-teaser">
    <h2>Simple, transparent <span>pricing</span></h2>
    <p>Start free. Upgrade anytime. Cancel anytime. First paid month includes a 7-day free trial.</p>
    <div class="pricing-row">
      <div class="price-pill">
        <div class="p-name">Free</div>
        <div class="p-price">$0<span>/mo</span></div>
      </div>
      <div class="price-pill featured">
        <div class="p-name">Basic</div>
        <div class="p-price">$9.99<span>/mo</span></div>
      </div>
      <div class="price-pill">
        <div class="p-name">Pro</div>
        <div class="p-price">$15.99<span>/mo</span></div>
      </div>
    </div>
    <p class="trial-note"><strong>7-day free trial</strong> on your first subscription · Annual plans save up to 25%</p>
    <br>
    <a class="btn-primary" href="/pricing">See full pricing →</a>
  </div>
</div>

<hr class="divider">

<!-- Why ChartEdge.trade -->
<div class="section">
  <h2>Why <span>ChartEdge.trade</span>?</h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px;">
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:22px;">
      <div style="font-size:1.4rem;margin-bottom:10px;">🔓</div>
      <h3 style="font-size:.9rem;font-weight:700;margin-bottom:6px;">Everything in one place</h3>
      <p style="color:var(--muted);font-size:.8rem;line-height:1.55;">Options flow, dark pool, insider trades, gamma, Greeks, crypto — no juggling 6 different sites and subscriptions.</p>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:22px;">
      <div style="font-size:1.4rem;margin-bottom:10px;">💸</div>
      <h3 style="font-size:.9rem;font-weight:700;margin-bottom:6px;">Fraction of the cost</h3>
      <p style="color:var(--muted);font-size:.8rem;line-height:1.55;">Comparable platforms charge $50–$200/mo. ChartEdge.trade Pro is $15.99/mo with a free trial — no commitment required.</p>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:22px;">
      <div style="font-size:1.4rem;margin-bottom:10px;">⚡</div>
      <h3 style="font-size:.9rem;font-weight:700;margin-bottom:6px;">Built for speed</h3>
      <p style="color:var(--muted);font-size:.8rem;line-height:1.55;">No bloated dashboards. Tools load fast, data refreshes automatically, and Pine Scripts copy in one click.</p>
    </div>
    <div style="background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:22px;">
      <div style="font-size:1.4rem;margin-bottom:10px;">📡</div>
      <h3 style="font-size:.9rem;font-weight:700;margin-bottom:6px;">Real data sources</h3>
      <p style="color:var(--muted);font-size:.8rem;line-height:1.55;">FINRA, SEC EDGAR, Hyperliquid, CoinGecko, Finnhub — not delayed or fabricated data. Live options chains, real filings.</p>
    </div>
  </div>
</div>

<hr class="divider">

<!-- FAQ -->
<div class="section">
  <h2>Frequently asked <span>questions</span></h2>
  <div class="faq-list">
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        Do the indicators work on a free TradingView account?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Yes — all Pine Script indicators on ChartEdge.trade are built with TradingView's free tier in mind. They use Pine Script v6 and do not require any TradingView paid plan to run.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        What is the free plan limit?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Free accounts can copy up to 3 indicators per day. Basic and Pro plans raise or remove those limits and unlock additional tools like Gamma Exposure, Options Flow, and the Earnings Calendar.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        Is there a free trial for paid plans?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Yes — your first subscription (Basic or Pro, monthly or yearly) includes a 7-day free trial. You won't be charged until the trial ends, and you can cancel any time before that with no cost.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        How does the referral program work?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Share your unique referral link from the Refer a Friend page. When either you or someone you referred upgrades to a paid plan, both of you receive 7 days of that plan for free — automatically, no codes needed.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        What is Options Flow / Gamma Exposure?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Options Flow shows the net call vs put activity across a ticker's options chain — useful for spotting where large traders are positioning. Gamma Exposure (GEX) uses Black-Scholes to calculate where market makers are likely to hedge, highlighting key support and resistance price levels. Both pull live data from real options chains.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        Can I cancel my subscription?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Absolutely. You can manage or cancel your subscription at any time from the Billing page. There are no cancellation fees and your access continues until the end of the billing period you already paid for.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        What is Pine Script?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">Pine Script is TradingView's built-in scripting language for creating custom indicators and strategies directly on your charts. ChartEdge.trade generates ready-to-use Pine Script v6 code — you just copy it and paste it into TradingView's Pine Script editor, and it appears on your chart instantly. No coding knowledge required.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        What is ChartEdge.trade?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">ChartEdge.trade is a market intelligence dashboard built for retail traders. It combines 20+ free Pine Script indicators with professional-grade tools including live options flow, insider trading disclosures, gamma exposure charts, LSTM volatility forecasts, a pre-market scanner, and more — all in one place. Most tools are available free or with a 7-day trial on paid plans.</div>
    </div>
    <div class="faq-item">
      <div class="faq-q" onclick="toggleFaq(this)">
        Can I use a different trading platform?
        <span class="faq-chevron">▼</span>
      </div>
      <div class="faq-a">The Pine Script indicators are exclusive to TradingView — they won't work on ThinkOrSwim, MetaTrader, or other platforms. However, all of ChartEdge.trade's other tools (options flow, insider trading, gamma exposure, market heatmap, earnings calendar, etc.) run entirely in your browser and work alongside any broker or trading platform you use.</div>
    </div>
  </div>
</div>

<footer>© 2026 ChartEdge.trade · Free Pine Script indicators · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
""" + _THEME_JS + """
function toggleFaq(el) {
  el.closest('.faq-item').classList.toggle('open');
}

// Auto-tag existing cards as reveal targets + observe
(function(){
  ['.hero-tool', '.intel-card', '.faq-item'].forEach(function(sel){
    document.querySelectorAll(sel).forEach(function(el, i){
      el.classList.add('reveal');
      el.style.setProperty('--rd', (i * 0.05) + 's');
    });
  });
  if (!('IntersectionObserver' in window)) {
    document.querySelectorAll('.reveal').forEach(function(el){ el.classList.add('in'); });
    return;
  }
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  document.querySelectorAll('.reveal').forEach(function(el){ io.observe(el); });
})();

// Animated stat counters
(function(){
  var stats = document.querySelectorAll('.stat-num[data-count]');
  if (!stats.length) return;
  function animate(el){
    var target = parseFloat(el.dataset.count);
    var decimals = parseInt(el.dataset.decimals || '0', 10);
    var suffix = el.dataset.suffix || '';
    var prefix = el.dataset.prefix || '';
    var dur = 1400;
    var start = performance.now();
    function step(now){
      var t = Math.min(1, (now - start) / dur);
      var k = 1 - Math.pow(1 - t, 3);
      var v = target * k;
      el.textContent = prefix + v.toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = prefix + target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(step);
  }
  if (!('IntersectionObserver' in window)) { stats.forEach(animate); return; }
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(e){
      if (e.isIntersecting) { animate(e.target); io.unobserve(e.target); }
    });
  }, { threshold: 0.35 });
  stats.forEach(function(el){ io.observe(el); });
})();
</script>
</body>
</html>"""


# ── Generator HTML ───────────────────────────────────────────────────────────

GENERATOR_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Volatility Forecast — TradingView Pine Script Generator</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }

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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
  </div>
</nav>

<div class="hero">
  <h1>TradingView<br><span>Volatility Forecast</span> Indicator</h1>
  <p>Generate a custom Pine Script indicator with an LSTM-powered volatility forecast — a TradingView Plus subscription or higher is required to use this tool.</p>
  <span class="badge-free">⚡ TradingView Plus required</span>
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
      <span>Pine Script v6 — paste into TradingView Pine Editor</span>
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
  © 2026 ChartEdge.trade · LSTM volatility prediction · Not financial advice
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


# ── Backtester HTML ──────────────────────────────────────────────────────────

BACKTEST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Backtester · ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; color:var(--text); }
    """ + _NAV_CSS + """
    .page { max-width:1100px; margin:0 auto; padding:32px 20px; }
    h1 { font-size:1.6rem; margin-bottom:8px; }
    .subtitle { color:var(--muted); margin-bottom:28px; font-size:.95rem; }
    .form-card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:24px; margin-bottom:28px; }
    .form-row { display:flex; flex-wrap:wrap; gap:16px; margin-bottom:18px; }
    .form-group { display:flex; flex-direction:column; gap:6px; flex:1; min-width:160px; }
    .form-group label { font-size:.82rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
    .form-group input, .form-group select { background:var(--bg3); border:1px solid var(--border); color:var(--text); border-radius:6px; padding:9px 12px; font-size:.95rem; outline:none; }
    .form-group input:focus, .form-group select:focus { border-color:var(--accent); }
    .params-section { margin-top:4px; }
    .params-group { display:none; }
    .params-group.active { display:flex; flex-wrap:wrap; gap:16px; }
    .run-btn { background:var(--accent); color:#fff; border:none; border-radius:8px; padding:11px 28px; font-size:1rem; font-weight:600; cursor:pointer; transition:opacity .15s; }
    .run-btn:hover { opacity:.85; }
    .run-btn:disabled { opacity:.5; cursor:not-allowed; }
    .loader { display:none; color:var(--muted); font-size:.9rem; margin-top:12px; }
    .error-msg { display:none; color:var(--red); font-size:.9rem; margin-top:12px; padding:10px 14px; background:rgba(248,81,73,.1); border-radius:6px; border:1px solid rgba(248,81,73,.3); }
    .results { display:none; }
    .stats-row { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:20px; }
    .stat-card { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:18px 22px; flex:1; min-width:140px; }
    .stat-label { font-size:.78rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px; }
    .stat-value { font-size:1.55rem; font-weight:700; }
    .stat-value.pos { color:var(--green); }
    .stat-value.neg { color:var(--red); }
    .stat-value.neutral { color:var(--text); }
    .chart-wrap { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:20px; }
    .chart-title { font-size:1rem; font-weight:600; margin-bottom:12px; }
    #equity-chart { width:100%; height:360px; }
    .table-wrap { background:var(--bg2); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    .table-header { padding:14px 20px; border-bottom:1px solid var(--border); font-weight:600; display:flex; justify-content:space-between; align-items:center; }
    .trade-count { font-size:.82rem; color:var(--muted); font-weight:400; }
    table { width:100%; border-collapse:collapse; font-size:.9rem; }
    th { padding:10px 16px; text-align:left; color:var(--muted); font-weight:600; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; border-bottom:1px solid var(--border); }
    td { padding:10px 16px; border-bottom:1px solid var(--border); }
    tr:last-child td { border-bottom:none; }
    .type-buy  { color:var(--green); font-weight:600; }
    .type-sell { color:var(--red);   font-weight:600; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/">📈 ChartEdge.trade</a>
  <div class="nav-links">""" + _NAV_LINKS + """
  </div>
</nav>
<div class="page">
  <h1>Strategy Backtester</h1>
  <p class="subtitle">Simulate trading strategies on historical data. Pro feature.</p>

  <div class="form-card">
    <div class="form-row">
      <div class="form-group">
        <label>Ticker</label>
        <input type="text" id="bt-ticker" value="AAPL" placeholder="e.g. AAPL" style="text-transform:uppercase">
      </div>
      <div class="form-group">
        <label>Strategy</label>
        <select id="bt-strategy" onchange="updateParams()">
          <option value="rsi">RSI</option>
          <option value="macd">MACD</option>
          <option value="bbands">Bollinger Bands</option>
          <option value="ema_cross">EMA Crossover</option>
          <option value="sma_cross">SMA Crossover</option>
          <option value="uvol">Unusual Volume</option>
        </select>
      </div>
      <div class="form-group">
        <label>Period</label>
        <select id="bt-period">
          <option value="1y">1 Year</option>
          <option value="2y">2 Years</option>
          <option value="5y">5 Years</option>
        </select>
      </div>
    </div>

    <div class="params-section">
      <!-- RSI params -->
      <div class="params-group active" id="params-rsi">
        <div class="form-group">
          <label>RSI Period</label>
          <input type="number" id="rsi-period" value="14" min="2" max="100">
        </div>
        <div class="form-group">
          <label>Oversold</label>
          <input type="number" id="rsi-oversold" value="30" min="1" max="49">
        </div>
        <div class="form-group">
          <label>Overbought</label>
          <input type="number" id="rsi-overbought" value="70" min="51" max="99">
        </div>
      </div>
      <!-- MACD params -->
      <div class="params-group" id="params-macd">
        <div class="form-group">
          <label>Fast EMA</label>
          <input type="number" id="macd-fast" value="12" min="2" max="50">
        </div>
        <div class="form-group">
          <label>Slow EMA</label>
          <input type="number" id="macd-slow" value="26" min="5" max="100">
        </div>
        <div class="form-group">
          <label>Signal</label>
          <input type="number" id="macd-signal" value="9" min="2" max="30">
        </div>
      </div>
      <!-- Bollinger Bands params -->
      <div class="params-group" id="params-bbands">
        <div class="form-group">
          <label>Period</label>
          <input type="number" id="bb-period" value="20" min="5" max="100">
        </div>
        <div class="form-group">
          <label>Std Dev</label>
          <input type="number" id="bb-std" value="2.0" min="0.5" max="5" step="0.1">
        </div>
      </div>
      <!-- EMA Cross params -->
      <div class="params-group" id="params-ema_cross">
        <div class="form-group">
          <label>Fast EMA</label>
          <input type="number" id="ema-fast" value="9" min="2" max="50">
        </div>
        <div class="form-group">
          <label>Slow EMA</label>
          <input type="number" id="ema-slow" value="21" min="5" max="200">
        </div>
      </div>
      <!-- SMA Cross params -->
      <div class="params-group" id="params-sma_cross">
        <div class="form-group">
          <label>Fast SMA</label>
          <input type="number" id="sma-fast" value="50" min="2" max="100">
        </div>
        <div class="form-group">
          <label>Slow SMA</label>
          <input type="number" id="sma-slow" value="200" min="10" max="500">
        </div>
      </div>
      <!-- Unusual Volume params -->
      <div class="params-group" id="params-uvol">
        <div class="form-group">
          <label>Avg Period</label>
          <input type="number" id="uvol-period" value="20" min="5" max="100">
        </div>
        <div class="form-group">
          <label>Volume Multiplier</label>
          <input type="number" id="uvol-mult" value="2.0" min="1.1" max="10" step="0.1">
        </div>
        <div class="form-group">
          <label>Hold Days</label>
          <input type="number" id="uvol-hold" value="5" min="1" max="30">
        </div>
      </div>
    </div>

    <div style="margin-top:20px;">
      <button class="run-btn" id="run-btn" onclick="runBacktest()">Run Backtest</button>
      <div class="loader" id="loader">⏳ Running backtest…</div>
      <div class="error-msg" id="error-msg"></div>
    </div>
  </div>

  <!-- Results -->
  <div class="results" id="results">
    <div class="stats-row" id="stats-row"></div>
    <div class="chart-wrap">
      <div class="chart-title">Equity Curve</div>
      <div id="equity-chart"></div>
    </div>
    <div class="table-wrap">
      <div class="table-header">
        <span>Trade Log</span>
        <span class="trade-count" id="trade-count"></span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Type</th>
            <th>Price</th>
            <th>Shares</th>
          </tr>
        </thead>
        <tbody id="trade-log"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
""" + _THEME_JS + """

function updateParams() {
  const strat = document.getElementById('bt-strategy').value;
  document.querySelectorAll('.params-group').forEach(function(el) {
    el.classList.remove('active');
  });
  const target = document.getElementById('params-' + strat);
  if (target) target.classList.add('active');
}

function getParams() {
  const strat = document.getElementById('bt-strategy').value;
  if (strat === 'rsi') {
    return {
      rsi_period: parseInt(document.getElementById('rsi-period').value) || 14,
      oversold:   parseFloat(document.getElementById('rsi-oversold').value) || 30,
      overbought: parseFloat(document.getElementById('rsi-overbought').value) || 70
    };
  } else if (strat === 'macd') {
    return {
      fast:   parseInt(document.getElementById('macd-fast').value) || 12,
      slow:   parseInt(document.getElementById('macd-slow').value) || 26,
      signal: parseInt(document.getElementById('macd-signal').value) || 9
    };
  } else if (strat === 'bbands') {
    return {
      period: parseInt(document.getElementById('bb-period').value) || 20,
      std:    parseFloat(document.getElementById('bb-std').value) || 2.0
    };
  } else if (strat === 'ema_cross') {
    return {
      fast: parseInt(document.getElementById('ema-fast').value) || 9,
      slow: parseInt(document.getElementById('ema-slow').value) || 21
    };
  } else if (strat === 'sma_cross') {
    return {
      fast: parseInt(document.getElementById('sma-fast').value) || 50,
      slow: parseInt(document.getElementById('sma-slow').value) || 200
    };
  } else if (strat === 'uvol') {
    return {
      vol_period: parseInt(document.getElementById('uvol-period').value) || 20,
      multiplier: parseFloat(document.getElementById('uvol-mult').value) || 2.0,
      hold_days:  parseInt(document.getElementById('uvol-hold').value) || 5
    };
  }
  return {};
}

function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  var sign = v >= 0 ? '+' : '';
  return sign + v.toFixed(2) + '%';
}
function fmtNum(v) {
  if (v === null || v === undefined) return '—';
  return v.toFixed(2);
}
function colorClass(v) {
  if (v === null || v === undefined) return 'neutral';
  return v >= 0 ? 'pos' : 'neg';
}

async function runBacktest() {
  const ticker   = document.getElementById('bt-ticker').value.trim().toUpperCase();
  const strategy = document.getElementById('bt-strategy').value;
  const period   = document.getElementById('bt-period').value;
  const params   = getParams();

  if (!ticker) { showError('Please enter a ticker.'); return; }

  const btn = document.getElementById('run-btn');
  const loader = document.getElementById('loader');
  const errDiv = document.getElementById('error-msg');
  const results = document.getElementById('results');

  btn.disabled = true;
  loader.style.display = 'block';
  errDiv.style.display = 'none';
  results.style.display = 'none';

  try {
    const resp = await fetch('/api/backtest', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ticker: ticker, strategy: strategy, period: period, params: params})
    });
    const d = await resp.json();

    if (!d.ok) { showError(d.error || 'Backtest failed.'); return; }

    // Stats cards
    var statsHTML = [
      makeCard('Total Return', fmtPct(d.total_return), colorClass(d.total_return)),
      makeCard('Buy &amp; Hold Return', fmtPct(d.bh_return), colorClass(d.bh_return)),
      makeCard('Win Rate', d.win_rate !== null ? d.win_rate.toFixed(1) + '%' : '—', 'neutral'),
      makeCard('Max Drawdown', fmtPct(d.max_drawdown), d.max_drawdown <= 0 ? 'neg' : 'neutral'),
      makeCard('Trades', String(d.num_trades), 'neutral'),
      makeCard('Sharpe Ratio', d.sharpe !== null ? fmtNum(d.sharpe) : '—', d.sharpe !== null ? colorClass(d.sharpe) : 'neutral')
    ].join('');
    document.getElementById('stats-row').innerHTML = statsHTML;

    // Reveal results BEFORE plotting so the container has real dimensions
    results.style.display = 'block';

    // Equity chart
    var dates = d.equity_curve.map(function(e) { return e.date; });
    var vals   = d.equity_curve.map(function(e) { return e.value; });
    var bh     = d.equity_curve.map(function(e) { return e.bh; });

    var allVals = vals.concat(bh).filter(function(v) { return v !== null; });
    var minV = Math.min.apply(null, allVals);
    var maxV = Math.max.apply(null, allVals);
    var pad  = (maxV - minV) * 0.03 || 100;

    var isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    var bgColor   = isDark ? '#0d1117' : '#ffffff';
    var gridColor = isDark ? '#21262d' : '#eaeef2';
    var textColor = isDark ? '#8b949e' : '#636c76';

    Plotly.newPlot('equity-chart', [
      {
        x: dates, y: vals, name: 'Strategy',
        type: 'scatter', mode: 'lines',
        line: {color: '#58a6ff', width: 2}
      },
      {
        x: dates, y: bh, name: 'Buy & Hold',
        type: 'scatter', mode: 'lines',
        line: {color: '#8b949e', width: 1.5, dash: 'dot'}
      }
    ], {
      paper_bgcolor: bgColor,
      plot_bgcolor:  bgColor,
      font: {color: textColor, size: 11},
      margin: {t: 10, r: 16, b: 40, l: 64},
      xaxis: {gridcolor: gridColor, linecolor: gridColor, tickfont: {color: textColor}},
      yaxis: {gridcolor: gridColor, linecolor: gridColor, tickfont: {color: textColor},
              range: [minV - pad, maxV + pad], tickprefix: '$'},
      legend: {bgcolor: 'rgba(0,0,0,0)', font: {color: textColor}},
      hovermode: 'x unified'
    }, {responsive: true, displayModeBar: false});

    // Trade log
    document.getElementById('trade-count').textContent = d.num_trades + ' trades';
    var tbody = document.getElementById('trade-log');
    if (d.trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted);text-align:center;padding:20px;">No trades generated</td></tr>';
    } else {
      tbody.innerHTML = d.trades.map(function(t) {
        var cls = t.type === 'buy' ? 'type-buy' : 'type-sell';
        var price = t.price !== null ? '$' + t.price.toFixed(2) : '—';
        var shares = t.shares !== null ? t.shares.toFixed(4) : '—';
        return '<tr><td>' + t.date + '</td><td class="' + cls + '">' + t.type.toUpperCase() + '</td><td>' + price + '</td><td>' + shares + '</td></tr>';
      }).join('');
    }

  } catch(err) {
    showError('Network error: ' + err.message);
  } finally {
    btn.disabled = false;
    loader.style.display = 'none';
  }
}

function makeCard(label, value, cls) {
  return '<div class="stat-card"><div class="stat-label">' + label + '</div><div class="stat-value ' + cls + '">' + value + '</div></div>';
}

function showError(msg) {
  var e = document.getElementById('error-msg');
  e.textContent = msg;
  e.style.display = 'block';
  document.getElementById('loader').style.display = 'none';
  document.getElementById('run-btn').disabled = false;
}
</script>
</body>
</html>"""


# ── Economic Calendar HTML ───────────────────────────────────────────────────

ECONOMIC_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Economic Calendar — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; --yellow:#d29922; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; --yellow:#9a6700; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 960px; margin: 0 auto; padding: 28px 24px; }
    .filter-row { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .filter-btn { background: var(--bg2); border: 1px solid var(--border); color: var(--muted); padding: 6px 16px; border-radius: 20px; font-size: .82rem; cursor: pointer; transition: all .15s; }
    .filter-btn.active, .filter-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--bg3); }
    table { width: 100%; border-collapse: collapse; font-size: .86rem; }
    th { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; padding: 8px 14px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
    td { padding: 12px 14px; border-bottom: 1px solid var(--border); vertical-align: middle; }
    tr:hover td { background: var(--bg2); }
    .impact-high   { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; background:rgba(248,81,73,.15); color:var(--red); }
    .impact-medium { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; background:rgba(210,153,34,.15); color:var(--yellow); }
    .impact-low    { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.72rem; font-weight:700; background:rgba(139,148,158,.12); color:var(--muted); }
    .date-cell { color: var(--muted); font-size: .8rem; white-space: nowrap; }
    .time-cell { color: var(--muted); font-size: .8rem; white-space: nowrap; }
    .event-name { font-weight: 600; color: var(--text); }
    .val { font-variant-numeric: tabular-nums; }
    .val.actual { color: var(--green); font-weight: 600; }
    .val.miss { color: var(--red); font-weight: 600; }
    .day-header td { background: var(--bg2); color: var(--muted); font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; padding: 6px 14px; border-bottom: 1px solid var(--border); }
    .loading { text-align: center; padding: 60px; color: var(--muted); }
    .spin { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; margin: 0 auto 16px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .empty { text-align: center; padding: 60px; color: var(--muted); }
    .disclaimer { color: var(--muted); font-size: .78rem; margin-top: 20px; padding: 12px; background: var(--bg2); border-radius: 6px; border: 1px solid var(--border); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Economic <span>Calendar</span></h1>
  <p>Upcoming US economic events — Fed meetings, CPI, jobs, GDP &amp; more</p>
</div>
<div class="container">
  <div class="filter-row">
    <button class="filter-btn active" onclick="setFilter('week', this)">This Week</button>
    <button class="filter-btn" onclick="setFilter('next', this)">Next Week</button>
    <button class="filter-btn" onclick="setFilter('month', this)">This Month</button>
    <button class="filter-btn" onclick="setFilter('high', this)">High Impact Only</button>
  </div>
  <div id="econ-content">
    <div class="loading"><div class="spin"></div>Loading economic events…</div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Data via Finnhub · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>
let _allEvents = [];
let _activeFilter = 'week';

function setFilter(f, btn) {
  _activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
}

function dateStr(d) {
  // Returns YYYY-MM-DD string in local time
  return d.getFullYear() + '-'
    + String(d.getMonth() + 1).padStart(2, '0') + '-'
    + String(d.getDate()).padStart(2, '0');
}
function getWeekRange(offset) {
  const now = new Date();
  const day = now.getDay(); // 0=Sun, 6=Sat
  // Normalize to midnight to avoid DST edge cases
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const daysSinceMonday = day === 0 ? 6 : day - 1;
  // On weekends bump "this week" forward to the upcoming Mon-Fri
  const weekendShift = (day === 0 || day === 6) ? 1 : 0;
  const monday = new Date(today);
  monday.setDate(today.getDate() - daysSinceMonday + (offset + weekendShift) * 7);
  const friday = new Date(monday);
  friday.setDate(monday.getDate() + 4); // Mon-Fri only, no weekends
  return [dateStr(monday), dateStr(friday)];
}

function filterEvents(events) {
  const now  = new Date();
  const today = dateStr(now);
  if (_activeFilter === 'week') {
    const [start, end] = getWeekRange(0);
    return events.filter(e => { const d = e.time.slice(0,10); return d >= start && d <= end; });
  }
  if (_activeFilter === 'next') {
    const [start, end] = getWeekRange(1);
    return events.filter(e => { const d = e.time.slice(0,10); return d >= start && d <= end; });
  }
  if (_activeFilter === 'month') {
    const end = new Date(now); end.setDate(now.getDate() + 30);
    return events.filter(e => e.time.slice(0,10) <= dateStr(end));
  }
  if (_activeFilter === 'high') {
    return events.filter(e => e.impact === 'high');
  }
  return events;
}

function fmtDate(iso) {
  // Parse YYYY-MM-DD portion only to avoid timezone shifts
  const parts = iso.slice(0,10).split('-');
  const d = new Date(parseInt(parts[0]), parseInt(parts[1])-1, parseInt(parts[2]));
  return d.toLocaleDateString('en-US', {weekday:'short', month:'short', day:'numeric'});
}
function fmtTime(iso) {
  const timePart = iso.slice(11,16);
  if (!timePart || timePart === '00:00') return 'All Day';
  const [hStr, mStr] = timePart.split(':');
  const h = parseInt(hStr), m = parseInt(mStr);
  const ampm = h >= 12 ? 'PM' : 'AM';
  return ((h % 12) || 12) + ':' + String(m).padStart(2,'0') + ' ' + ampm + ' ET';
}
function fmtVal(v, unit) {
  if (v === null || v === undefined) return '<span style="color:var(--muted)">—</span>';
  return '<span class="val">' + v + (unit ? ' ' + unit : '') + '</span>';
}
function fmtActual(actual, estimate, unit) {
  if (actual === null || actual === undefined) return '<span style="color:var(--muted)">—</span>';
  let cls = 'actual';
  if (estimate !== null && estimate !== undefined) {
    cls = actual >= estimate ? 'actual' : 'miss';
  }
  return '<span class="val ' + cls + '">' + actual + (unit ? ' ' + unit : '') + '</span>';
}

function render() {
  const events = filterEvents(_allEvents);
  const el = document.getElementById('econ-content');
  if (!events.length) {
    el.innerHTML = '<div class="empty">No events found for this period.</div>';
    return;
  }
  let html = '<table><thead><tr><th>Date</th><th>Time (ET)</th><th>Event</th><th>Impact</th><th>Previous</th><th>Estimate</th><th>Actual</th></tr></thead><tbody>';
  let lastDate = '';
  events.forEach(e => {
    const d = new Date(e.time);
    const dateStr = d.toDateString();
    if (dateStr !== lastDate) {
      lastDate = dateStr;
      html += '<tr class="day-header"><td colspan="7">' + fmtDate(e.time) + '</td></tr>';
    }
    const impactClass = e.impact === 'high' ? 'impact-high' : e.impact === 'medium' ? 'impact-medium' : 'impact-low';
    const impactLabel = e.impact === 'high' ? 'High' : e.impact === 'medium' ? 'Medium' : 'Low';
    html += '<tr>'
      + '<td class="date-cell">' + fmtDate(e.time) + '</td>'
      + '<td class="time-cell">' + fmtTime(e.time) + '</td>'
      + '<td class="event-name">' + e.event + '</td>'
      + '<td><span class="' + impactClass + '">' + impactLabel + '</span></td>'
      + '<td>' + fmtVal(e.prev, e.unit) + '</td>'
      + '<td>' + fmtVal(e.estimate, e.unit) + '</td>'
      + '<td>' + fmtActual(e.actual, e.estimate, e.unit) + '</td>'
      + '</tr>';
  });
  html += '</tbody></table>';
  html += '<div class="disclaimer">📅 US economic events only · Data via Finnhub · Times in ET · Actual values turn green (beat) or red (miss) vs estimate</div>';
  el.innerHTML = html;
}

async function loadEconomic() {
  try {
    const res = await fetch('/api/economic');
    const data = await res.json();
    if (data.error) {
      document.getElementById('econ-content').innerHTML = '<div class="empty">Unable to load economic data.<br><span style="font-size:.8rem;color:var(--muted)">' + data.error + '</span></div>';
      return;
    }
    _allEvents = data.events || [];
    render();
  } catch(e) {
    document.getElementById('econ-content').innerHTML = '<div class="empty" style="color:var(--red)">Failed to load economic calendar.</div>';
  }
}
loadEconomic();
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
  <title>Earnings Calendar — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
<footer>© 2026 ChartEdge.trade · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
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


DIVIDENDS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Dividends Calendar · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 960px; margin: 0 auto; padding: 28px 24px; }
    table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    th { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; padding: 8px 14px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
    td { padding: 11px 14px; border-bottom: 1px solid var(--border); }
    tr:hover td { background: var(--bg2); }
    .ticker { color: var(--accent); font-weight: 700; font-size: .92rem; }
    .name { color: var(--muted); font-size: .78rem; }
    .yield-val { color: var(--green); font-weight: 600; }
    .soon { color: var(--accent); font-weight: 600; }
    .spin { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; margin: 0 auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading { text-align: center; padding: 60px; color: var(--muted); }
    .empty { text-align: center; padding: 60px; color: var(--muted); }
    .disclaimer { color: var(--muted); font-size: .78rem; margin-top: 20px; padding: 12px; background: var(--bg2); border-radius: 6px; border: 1px solid var(--border); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Dividends <span>Calendar</span></h1>
  <p>Upcoming ex-dividend dates for major stocks &amp; ETFs — next 60 days</p>
</div>
<div class="container">
  <div id="div-content">
    <div class="loading"><div class="spin"></div><p style="margin-top:14px">Loading dividend data…</p></div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Dividend data from Yahoo Finance · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a></footer>
<script>
function loadDividends() {
  fetch('/api/dividends')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.loading) {
        setTimeout(loadDividends, 3000);
        return;
      }
      if (d.error) {
        document.getElementById('div-content').innerHTML = '<div class="empty">Error: ' + d.error + '</div>';
        return;
      }
      if (!d.dividends || d.dividends.length === 0) {
        document.getElementById('div-content').innerHTML = '<div class="empty">No upcoming ex-dividend dates found in the next 60 days.</div>';
        return;
      }
      var today = new Date();
      var html = '<table><thead><tr>'
        + '<th>Ticker</th><th>Company</th><th>Ex-Date</th><th>Per Share</th><th>Yield</th><th>Frequency</th><th>Price</th>'
        + '</tr></thead><tbody>';
      d.dividends.forEach(function(row) {
        var exDate  = new Date(row.ex_date);
        var daysOut = Math.round((exDate - today) / 86400000);
        var dateCls = daysOut <= 7 ? ' class="soon"' : '';
        var dateStr = daysOut === 0 ? 'Today' : daysOut === 1 ? 'Tomorrow' : row.ex_date;
        html += '<tr>'
          + '<td><span class="ticker">' + row.ticker + '</span></td>'
          + '<td><span class="name">' + row.name + '</span></td>'
          + '<td><span' + dateCls + '>' + dateStr + '</span></td>'
          + '<td>' + row.per_share + '</td>'
          + '<td><span class="yield-val">' + row.yield_pct + '</span></td>'
          + '<td>' + row.freq + '</td>'
          + '<td>' + row.price + '</td>'
          + '</tr>';
      });
      html += '</tbody></table>';
      html += '<div class="disclaimer">⚠ Ex-dividend dates and amounts from Yahoo Finance. Always verify before trading. Dates may shift.</div>';
      document.getElementById('div-content').innerHTML = html;
    })
    .catch(function(){
      document.getElementById('div-content').innerHTML = '<div class="empty">Failed to load dividend data.</div>';
    });
}
loadDividends();
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Crypto HTML shared CSS ────────────────────────────────────────────────────

_CRYPTO_PAGE_CSS = """
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; --orange:#e3b341; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; --orange:#9a6700; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; }
    .page { max-width:1000px; margin:0 auto; padding:28px 20px 60px; }
    h1 { font-size:1.5rem; margin-bottom:4px; } h1 span { color:var(--accent); }
    .sub { color:var(--muted); font-size:.83rem; margin-bottom:20px; }
    .stats-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
    .stat { background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; flex:1; min-width:130px; }
    .stat .sv { font-size:1.25rem; font-weight:800; }
    .stat .sl { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }
    .card { background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:20px; margin-bottom:16px; }
    table { width:100%; border-collapse:collapse; font-size:.88rem; }
    th { text-align:left; padding:8px 12px; color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid var(--border); }
    td { padding:9px 12px; border-bottom:1px solid var(--border); }
    tr:last-child td { border-bottom:none; }
    tr:hover td { background:var(--bg3); }
    .pos { color:var(--green); } .neg { color:var(--red); }
    .badge { display:inline-block; padding:2px 8px; border-radius:4px; font-size:.75rem; font-weight:600; }
    .badge-green { background:#1f2d1f; color:var(--green); }
    .badge-red   { background:#2d1f1f; color:var(--red); }
    .badge-orange{ background:#2d2510; color:var(--orange); }
    .loading { color:var(--muted); padding:40px; text-align:center; }
    .err { color:var(--red); padding:20px; }
    .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:16px; }
    .controls select, .controls input { background:var(--bg2); border:1px solid var(--border); color:var(--text); padding:7px 12px; border-radius:7px; font-size:.88rem; outline:none; }
    .controls select:focus, .controls input:focus { border-color:var(--accent); }
    .btn { background:var(--accent); color:#fff; border:none; border-radius:7px; padding:8px 18px; font-size:.88rem; font-weight:600; cursor:pointer; }
    .btn:hover { opacity:.85; }
    footer { text-align:center; padding:32px 24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }
"""

# ── Fear & Greed HTML ─────────────────────────────────────────────────────────

CRYPTO_FEARGREED_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>Crypto Fear &amp; Greed · ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    .gauge-wrap { display:flex; flex-direction:column; align-items:center; padding:28px 0 16px; }
    .gauge-score { font-size:4rem; font-weight:900; line-height:1; }
    .gauge-label { font-size:1.1rem; font-weight:600; margin-top:6px; }
    .gauge-date  { font-size:.78rem; color:var(--muted); margin-top:4px; }
    #gauge-arc   { width:280px; height:160px; }
    #hist-chart  { min-height:260px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Crypto <span>Fear &amp; Greed</span></h1>
  <p class="sub">Sentiment index 0–100 · sourced from alternative.me · updates daily</p>
  <div class="card" style="text-align:center">
    <canvas id="gauge-arc"></canvas>
    <div class="gauge-wrap">
      <div class="gauge-score" id="fg-score">—</div>
      <div class="gauge-label" id="fg-label">Loading…</div>
      <div class="gauge-date"  id="fg-date"></div>
    </div>
  </div>
  <div class="card">
    <div style="font-size:.85rem;font-weight:600;margin-bottom:12px;color:var(--muted)">30-DAY HISTORY</div>
    <div id="hist-chart"><div class="loading">Loading…</div></div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
fetch('/api/crypto/feargreed').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.querySelector('.page').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  var items=d.data;
  var cur=items[0];
  var score=cur.value;
  var label=cur.label;
  // color
  var color=score<25?'#f85149':score<45?'#e3b341':score<55?'#8b949e':score<75?'#3fb950':'#00c853';
  document.getElementById('fg-score').textContent=score;
  document.getElementById('fg-score').style.color=color;
  document.getElementById('fg-label').textContent=label;
  document.getElementById('fg-label').style.color=color;
  document.getElementById('fg-date').textContent='Updated '+new Date(cur.ts*1000).toLocaleDateString();
  // Arc gauge via canvas
  var canvas=document.getElementById('gauge-arc');
  var ctx=canvas.getContext('2d');
  canvas.width=280;canvas.height=160;
  var cx=140,cy=150,r=120,lw=22;
  var colors=['#f85149','#e3b341','#8b949e','#3fb950','#00c853'];
  var segs=[0.25,0.20,0.10,0.20,0.25];
  var start=Math.PI;
  segs.forEach(function(s,i){
    ctx.beginPath();ctx.arc(cx,cy,r,start,start+s*Math.PI);
    ctx.lineWidth=lw;ctx.strokeStyle=colors[i];ctx.lineCap='butt';ctx.stroke();
    start+=s*Math.PI;
  });
  // needle
  var angle=Math.PI+(score/100)*Math.PI;
  ctx.beginPath();
  ctx.moveTo(cx,cy);
  ctx.lineTo(cx+Math.cos(angle)*100,cy+Math.sin(angle)*100);
  ctx.strokeStyle='var(--text)';ctx.lineWidth=3;ctx.lineCap='round';ctx.stroke();
  ctx.beginPath();ctx.arc(cx,cy,8,0,2*Math.PI);ctx.fillStyle='var(--text)';ctx.fill();
  // history chart
  var dates=items.map(function(x){return new Date(x.ts*1000).toLocaleDateString();}).reverse();
  var vals=items.map(function(x){return x.value;}).reverse();
  var barColors=vals.map(function(v){return v<25?'#f85149':v<45?'#e3b341':v<55?'#8b949e':v<75?'#3fb950':'#00c853';});
  Plotly.react('hist-chart',[{x:dates,y:vals,type:'bar',marker:{color:barColors},name:'F&G'}],{
    paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',
    font:{color:'#e6edf3',size:11},margin:{t:10,b:40,l:40,r:10},
    xaxis:{gridcolor:'#21262d'},
    yaxis:{gridcolor:'#21262d',range:[0,100]},
    showlegend:false
  },{responsive:true,displayModeBar:true,modeBarButtonsToRemove:['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'],displaylogo:false});
}).catch(function(e){document.querySelector('.page').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── BTC Dominance HTML ────────────────────────────────────────────────────────

CRYPTO_DOMINANCE_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>BTC Dominance · ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    #dom-chart { min-height:340px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>BTC <span>Dominance</span></h1>
  <p class="sub">Market cap share across top cryptocurrencies · updates every 30 min</p>
  <div class="stats-row" id="stats-row"><div class="loading">Loading…</div></div>
  <div class="card">
    <div id="dom-chart"><div class="loading">Loading…</div></div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
function fmt(n){if(!n)return'—';if(n>=1e12)return'$'+(n/1e12).toFixed(2)+'T';if(n>=1e9)return'$'+(n/1e9).toFixed(2)+'B';return'$'+n.toLocaleString();}
fetch('/api/crypto/dominance').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.querySelector('.page').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  var sr=document.getElementById('stats-row');
  sr.innerHTML=[
    {v:d.btc+'%',l:'BTC Dominance'},{v:d.eth+'%',l:'ETH Dominance'},
    {v:fmt(d.total_mcap),l:'Total Mkt Cap'},{v:fmt(d.total_volume),l:'24h Volume'},
    {v:(d.active_coins||0).toLocaleString(),l:'Active Coins'}
  ].map(function(s){return'<div class="stat"><div class="sv">'+s.v+'</div><div class="sl">'+s.l+'</div></div>';}).join('');
  var labels=d.top.map(function(x){return x.symbol;});
  var vals=d.top.map(function(x){return x.pct;});
  var others=Math.max(0,100-vals.reduce(function(a,b){return a+b;},0));
  if(others>0.1){labels.push('OTHERS');vals.push(parseFloat(others.toFixed(2)));}
  Plotly.react('dom-chart',[{labels:labels,values:vals,type:'pie',hole:.45,
    textinfo:'label+percent',textfont:{size:12},
    marker:{colors:['#f7931a','#627eea','#26a17b','#e84142','#2775ca','#f0b90b','#8247e5','#00adef','#d9534f','#6f42c1','#aaaaaa']}}],{
    paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',
    font:{color:'#e6edf3',size:12},margin:{t:20,b:20,l:20,r:20},
    showlegend:true,legend:{orientation:'v',x:1.02,y:.5,font:{size:11}}
  },{responsive:true,displayModeBar:true,modeBarButtonsToRemove:['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'],displaylogo:false});
}).catch(function(e){document.querySelector('.page').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── Crypto Heatmap HTML ───────────────────────────────────────────────────────

CRYPTO_HEATMAP_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>Crypto Heatmap · ChartEdge.trade</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    #hm-wrap { width:100%; min-height:520px; background:var(--bg2); border:1px solid var(--border); border-radius:8px; overflow:hidden; position:relative; }
    .hm-cell { position:absolute; display:flex; flex-direction:column; align-items:center; justify-content:center; overflow:hidden; cursor:default; border:1px solid var(--bg); border-radius:3px; transition:opacity .15s; }
    .hm-cell:hover { opacity:.82; }
    .hm-sym { font-size:.78rem; font-weight:700; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.5); }
    .hm-chg { font-size:.7rem; color:rgba(255,255,255,.85); }
    .legend { display:flex; gap:6px; align-items:center; margin-top:10px; font-size:.75rem; color:var(--muted); }
    .lsq { width:14px; height:14px; border-radius:2px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Crypto <span>Heatmap</span></h1>
  <p class="sub">Top 100 coins by market cap · sized by market cap · colored by 24h change</p>
  <div id="hm-wrap"><div class="loading">Loading…</div></div>
  <div class="legend">
    <div class="lsq" style="background:#b91c1c"></div>≤−5%
    <div class="lsq" style="background:#f85149"></div>−2 to −5%
    <div class="lsq" style="background:#3d2b2b"></div>~0%
    <div class="lsq" style="background:#3fb950"></div>+2 to +5%
    <div class="lsq" style="background:#22c55e"></div>≥+5%
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
function chgColor(v){
  if(v<=-5)return'#b91c1c';if(v<=-2)return'#f85149';if(v<0)return'#6b2020';
  if(v===0)return'#2d2d2d';if(v<=2)return'#1a3d1a';if(v<=5)return'#3fb950';return'#22c55e';
}
fetch('/api/crypto/heatmap').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.getElementById('hm-wrap').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  var coins=d.coins;
  var wrap=document.getElementById('hm-wrap');
  wrap.innerHTML='';
  var W=wrap.offsetWidth||960, H=Math.max(520,window.innerHeight*0.55);
  wrap.style.height=H+'px';
  var root=d3.hierarchy({children:coins}).sum(function(c){return c.mcap||0;});
  d3.treemap().size([W,H]).paddingInner(2)(root);
  root.leaves().forEach(function(node){
    var c=node.data;
    var w=node.x1-node.x0, h=node.y1-node.y0;
    if(w<4||h<4)return;
    var el=document.createElement('div');
    el.className='hm-cell';
    el.style.cssText='left:'+node.x0+'px;top:'+node.y0+'px;width:'+w+'px;height:'+h+'px;background:'+chgColor(c.chg)+';';
    el.title=c.name+' | '+c.chg+'% | $'+c.price;
    if(w>30&&h>22){
      el.innerHTML='<span class="hm-sym">'+c.symbol+'</span>'+(h>36?'<span class="hm-chg">'+(c.chg>=0?'+':'')+c.chg+'%</span>':'');
    }
    wrap.appendChild(el);
  });
}).catch(function(e){document.getElementById('hm-wrap').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── Funding Rates HTML ────────────────────────────────────────────────────────

CRYPTO_FUNDING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>Funding Rates · ChartEdge.trade</title>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    .rate-pos { color:var(--green); font-weight:700; }
    .rate-neg { color:var(--red);   font-weight:700; }
    .rate-neu { color:var(--muted); }
    input#search { width:220px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Funding <span>Rates</span></h1>
  <p class="sub">Binance perpetual futures · positive = longs pay shorts · negative = shorts pay longs · refreshes every 5 min</p>
  <div class="controls">
    <input id="search" type="text" placeholder="Search symbol…" oninput="filterTable()">
  </div>
  <div class="card" style="padding:0">
    <div id="tbl-wrap"><div class="loading">Loading…</div></div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
var _rows=[];
function filterTable(){
  var q=document.getElementById('search').value.trim().toUpperCase();
  var filtered=q?_rows.filter(function(r){return r.symbol.includes(q);}):_rows;
  renderTable(filtered);
}
function renderTable(rows){
  if(!rows.length){document.getElementById('tbl-wrap').innerHTML='<div class="loading">No results.</div>';return;}
  var html='<table><thead><tr><th>#</th><th>Symbol</th><th>Funding Rate</th><th>Mark Price</th><th>Sentiment</th></tr></thead><tbody>';
  rows.forEach(function(r,i){
    var cls=r.rate>0?'rate-pos':r.rate<0?'rate-neg':'rate-neu';
    var sign=r.rate>0?'+':'';
    var sentiment=r.rate>0.05?'<span class="badge badge-red">Overleveraged Long</span>':r.rate<-0.05?'<span class="badge badge-green">Overleveraged Short</span>':'<span class="badge badge-orange">Neutral</span>';
    html+='<tr><td style="color:var(--muted)">'+(i+1)+'</td><td><strong>'+r.symbol+'</strong></td>'
      +'<td class="'+cls+'">'+sign+r.rate+'%</td>'
      +'<td>$'+r.price.toLocaleString()+'</td>'
      +'<td>'+sentiment+'</td></tr>';
  });
  html+='</tbody></table>';
  document.getElementById('tbl-wrap').innerHTML=html;
}
fetch('/api/crypto/funding').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.getElementById('tbl-wrap').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  _rows=d.rates;
  renderTable(_rows);
}).catch(function(e){document.getElementById('tbl-wrap').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── On-chain Metrics HTML ─────────────────────────────────────────────────────

CRYPTO_ONCHAIN_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>On-Chain Metrics · ChartEdge.trade</title>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    .section-title { font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:24px 0 10px; }
    .coin-header { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
    .coin-logo { width:32px; height:32px; border-radius:50%; }
    .chg-pos { color:var(--green); } .chg-neg { color:var(--red); }
    .supply-bar-wrap { background:var(--bg3); border-radius:4px; height:8px; margin-top:6px; overflow:hidden; }
    .supply-bar { height:100%; background:var(--accent); border-radius:4px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>On-Chain <span>Metrics</span></h1>
  <p class="sub">Bitcoin network stats + BTC &amp; ETH market data · updates every 30 min</p>
  <div id="content"><div class="loading">Loading…</div></div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
function fmt(n){if(!n)return'—';if(n>=1e12)return'$'+(n/1e12).toFixed(2)+'T';if(n>=1e9)return'$'+(n/1e9).toFixed(2)+'B';if(n>=1e6)return'$'+(n/1e6).toFixed(2)+'M';return'$'+n.toLocaleString();}
function fmtN(n){if(!n)return'—';if(n>=1e9)return(n/1e9).toFixed(2)+'B';if(n>=1e6)return(n/1e6).toFixed(2)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n.toLocaleString();}
function chgHtml(v){var c=v>=0?'chg-pos':'chg-neg';var s=v>=0?'+':'';return'<span class="'+c+'">'+s+v+'%</span>';}
fetch('/api/crypto/onchain').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.getElementById('content').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  var html='';
  // BTC network
  if(d.btc_chain){
    var b=d.btc_chain;
    html+='<div class="section-title">⛏ Bitcoin Network</div>';
    html+='<div class="stats-row">';
    [{v:b.hash_rate+'B GH/s',l:'Hash Rate'},{v:fmtN(b.difficulty),l:'Difficulty'},
     {v:fmtN(b.n_tx),l:"Today's Transactions"},{v:b.fees_btc+' BTC',l:'Total Fees (24h)'},
     {v:b.block_time+' min',l:'Avg Block Time'},{v:b.blocks_mined,l:'Blocks Mined (24h)'}
    ].forEach(function(s){html+='<div class="stat"><div class="sv">'+s.v+'</div><div class="sl">'+s.l+'</div></div>';});
    html+='</div>';
  }
  // BTC market
  if(d.bitcoin){
    var btc=d.bitcoin;
    var supplyPct=btc.max_supply?Math.round(btc.supply/btc.max_supply*100):null;
    html+='<div class="section-title">₿ Bitcoin Market</div>';
    html+='<div class="stats-row">';
    [{v:'$'+btc.price.toLocaleString(),l:'Price'},{v:fmt(btc.mcap),l:'Market Cap'},
     {v:fmt(btc.vol_24h),l:'24h Volume'},{v:chgHtml(btc.chg_24h),l:'24h Change'},
     {v:chgHtml(btc.chg_7d),l:'7d Change'},{v:btc.ath_chg+'%',l:'From ATH'}
    ].forEach(function(s){html+='<div class="stat"><div class="sv">'+s.v+'</div><div class="sl">'+s.l+'</div></div>';});
    html+='</div>';
    if(supplyPct!==null){
      html+='<div class="card" style="padding:14px 16px"><div style="display:flex;justify-content:space-between;font-size:.83rem"><span>Circulating Supply</span><span>'+fmtN(btc.supply)+' / 21M BTC ('+supplyPct+'%)</span></div><div class="supply-bar-wrap"><div class="supply-bar" style="width:'+supplyPct+'%"></div></div></div>';
    }
  }
  // ETH market
  if(d.ethereum){
    var eth=d.ethereum;
    html+='<div class="section-title">Ξ Ethereum Market</div>';
    html+='<div class="stats-row">';
    [{v:'$'+eth.price.toLocaleString(),l:'Price'},{v:fmt(eth.mcap),l:'Market Cap'},
     {v:fmt(eth.vol_24h),l:'24h Volume'},{v:chgHtml(eth.chg_24h),l:'24h Change'},
     {v:chgHtml(eth.chg_7d),l:'7d Change'},{v:eth.ath_chg+'%',l:'From ATH'}
    ].forEach(function(s){html+='<div class="stat"><div class="sv">'+s.v+'</div><div class="sl">'+s.l+'</div></div>';});
    html+='</div>';
  }
  document.getElementById('content').innerHTML=html||'<div class="err">No data available.</div>';
}).catch(function(e){document.getElementById('content').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── Liquidation Map HTML ──────────────────────────────────────────────────────

CRYPTO_LIQUIDATIONS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>Liquidation Map · ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    #liq-chart { min-height:360px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Liquidation <span>Map</span></h1>
  <p class="sub">Long/short ratio &amp; open interest · Binance perpetuals · last 48 hours</p>
  <div class="controls">
    <select id="sym-sel" onchange="load()">
      <option>BTC</option><option>ETH</option><option>SOL</option><option>XRP</option>
    </select>
  </div>
  <div class="stats-row" id="stats-row"></div>
  <div class="card"><div id="liq-chart"><div class="loading">Loading…</div></div></div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
function load(){
  var sym=document.getElementById('sym-sel').value;
  document.getElementById('liq-chart').innerHTML='<div class="loading">Loading '+sym+'…</div>';
  document.getElementById('stats-row').innerHTML='';
  fetch('/api/crypto/liquidations?symbol='+sym).then(function(r){return r.json();}).then(function(d){
    if(d.error){document.getElementById('liq-chart').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
    var rows=d.history||[];
    var sr=document.getElementById('stats-row');
    var curRate=rows.length?rows[rows.length-1].rate:0;
    sr.innerHTML=[
      {v:(d.price||0).toLocaleString(),l:'Mark Price ($)'},
      {v:(d.oi||0).toLocaleString(),l:'Open Interest'},
      {v:(curRate>=0?'+':'')+curRate+'%',l:'Latest Funding Rate'},
      {v:curRate>0?'<span class="pos">Longs Pay</span>':'<span class="neg">Shorts Pay</span>',l:'Sentiment'}
    ].map(function(s){return'<div class="stat"><div class="sv">'+s.v+'</div><div class="sl">'+s.l+'</div></div>';}).join('');
    if(!rows.length){document.getElementById('liq-chart').innerHTML='<div class="loading">No history available.</div>';return;}
    var times=rows.map(function(r){return new Date(r.ts).toLocaleString([],{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});});
    // Convert to basis points (×10000) so values are readable (e.g. 1.2 bps instead of 0.00012%)
    var rates=rows.map(function(r){return parseFloat((r.rate*100).toFixed(4));});
    var colors=rates.map(function(v){return v>=0?'#3fb950':'#f85149';});
    var absMax=Math.max.apply(null,rates.map(Math.abs))||0.01;
    document.getElementById('liq-chart').innerHTML='';
    Plotly.newPlot('liq-chart',[
      {x:times,y:rates,type:'bar',marker:{color:colors},name:'Funding Rate (bps)'},
    ],{
      paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',
      font:{color:'#e6edf3',size:11},
      height:360,
      margin:{t:10,b:80,l:60,r:10},
      xaxis:{gridcolor:'#21262d',tickangle:-45,nticks:12},
      yaxis:{gridcolor:'#21262d',title:'Funding Rate (bps)',range:[-absMax*1.3,absMax*1.3]},
      shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:0,y1:0,line:{color:'#8b949e',width:1,dash:'dot'}}],
      showlegend:false
    },{responsive:true,displayModeBar:true,modeBarButtonsToRemove:['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'],displaylogo:false});
  }).catch(function(e){document.getElementById('liq-chart').innerHTML='<div class="err">Failed to load: '+e+'</div>';});
}
load();
</script>
</body></html>"""


# ── Upcoming Coins HTML ───────────────────────────────────────────────────────

CRYPTO_UPCOMING_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">""" + _META + """
  <title>Trending Crypto · ChartEdge.trade</title>
  <style>""" + _CRYPTO_PAGE_CSS + _NAV_CSS + """
    input#search { width:220px; }
    .trend-rank { font-size:1.1rem; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Trending <span>Crypto</span></h1>
  <p class="sub">Top coins by search volume in the last 24h · sourced from CoinGecko · updates hourly</p>
  <div class="card" style="padding:0">
    <div id="tbl-wrap"><div class="loading">Loading…</div></div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
fetch('/api/crypto/upcoming').then(function(r){return r.json();}).then(function(d){
  if(d.error){document.getElementById('tbl-wrap').innerHTML='<div class="err">Error: '+d.error+'</div>';return;}
  var rows=d.coins;
  if(!rows.length){document.getElementById('tbl-wrap').innerHTML='<div class="loading">No data.</div>';return;}
  var html='<table><thead><tr><th>Trending</th><th>Symbol</th><th>Name</th><th>Market Cap Rank</th><th>Price (BTC)</th></tr></thead><tbody>';
  rows.forEach(function(r){
    html+='<tr>'
      +'<td class="trend-rank" style="color:var(--orange)">#'+r.score+'</td>'
      +'<td><strong>'+r.symbol+'</strong></td>'
      +'<td>'+r.name+'</td>'
      +'<td style="color:var(--muted)">'+(r.rank||'—')+'</td>'
      +'<td style="color:var(--muted)">'+parseFloat(r.price_btc).toExponential(4)+'</td>'
      +'</tr>';
  });
  html+='</tbody></table>';
  document.getElementById('tbl-wrap').innerHTML=html;
}).catch(function(e){document.getElementById('tbl-wrap').innerHTML='<div class="err">Failed: '+e+'</div>';});
</script>
</body></html>"""


# ── FAQ HTML ──────────────────────────────────────────────────────────────────

IPO_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Upcoming IPOs — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; --orange:#e3b341; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; --orange:#bc4c00; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 44px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.7rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: 0.9rem; max-width: 480px; margin: 0 auto; }
    .container { max-width: 960px; margin: 0 auto; padding: 28px 24px; }

    /* Filter tabs */
    .tabs { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .tab { background: var(--bg2); border: 1px solid var(--border); color: var(--muted); padding: 5px 16px; border-radius: 20px; font-size: 0.8rem; cursor: pointer; font-family: monospace; }
    .tab:hover { border-color: var(--accent); color: var(--text); }
    .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }

    /* IPO table */
    .ipo-table { width: 100%; border-collapse: collapse; }
    .ipo-table th { text-align: left; padding: 8px 14px; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); border-bottom: 1px solid var(--border); }
    .ipo-table td { padding: 13px 14px; border-bottom: 1px solid var(--border); font-size: 0.87rem; vertical-align: middle; }
    .ipo-table tr:last-child td { border-bottom: none; }
    .ipo-table tr:hover td { background: var(--bg2); }
    .company-name { color: var(--text); font-weight: 600; }
    .ticker-badge { display: inline-block; background: var(--bg3); border: 1px solid var(--border); color: var(--accent); padding: 2px 8px; border-radius: 5px; font-size: 0.78rem; font-weight: 700; margin-top: 3px; }
    .status-upcoming { display: inline-block; background: #0d3349; border: 1px solid #1f6feb; color: #58a6ff; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 700; }
    .status-priced   { display: inline-block; background: #0d2a14; border: 1px solid #238636; color: #3fb950; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 700; }
    .exch { color: var(--muted); font-size: 0.8rem; }

    /* Loading / empty */
    .state-box { text-align: center; padding: 60px 24px; color: var(--muted); }
    .state-box .icon { font-size: 2rem; margin-bottom: 12px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }

    /* Mobile */
    @media (max-width: 600px) {
      .ipo-table th:nth-child(4), .ipo-table td:nth-child(4),
      .ipo-table th:nth-child(5), .ipo-table td:nth-child(5) { display: none; }
    }

    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="hero">
  <h1>Upcoming <span>IPOs</span></h1>
  <p>Newly public and soon-to-price companies. Data sourced from NASDAQ's IPO calendar.</p>
</div>

<div class="container">
  <div class="tabs">
    <button class="tab active" onclick="filterTab(this,'all')">All</button>
    <button class="tab" onclick="filterTab(this,'Upcoming')">Upcoming</button>
    <button class="tab" onclick="filterTab(this,'Priced')">Recently Priced</button>
  </div>

  <div id="ipo-body">
    <div class="state-box"><div class="spinner"></div></div>
  </div>
</div>

<footer>© 2026 ChartEdge.trade · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
let _ipos = [];
let _filter = 'all';

async function loadIpos() {
  const res  = await fetch('/api/ipo');
  const data = await res.json();
  if (data.error || !data.ipos.length) {
    document.getElementById('ipo-body').innerHTML =
      '<div class="state-box"><div class="icon">📭</div><div>No IPO data available right now. Check back soon.</div></div>';
    return;
  }
  _ipos = data.ipos;
  render();
}

function filterTab(btn, val) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  _filter = val;
  render();
}

function render() {
  const rows = _filter === 'all' ? _ipos : _ipos.filter(i => i.status === _filter);
  if (!rows.length) {
    document.getElementById('ipo-body').innerHTML =
      '<div class="state-box"><div class="icon">🔍</div><div>No IPOs in this category.</div></div>';
    return;
  }
  let html = `<table class="ipo-table">
    <thead><tr>
      <th>Company</th>
      <th>Date</th>
      <th>Status</th>
      <th>Price Range</th>
      <th>Shares</th>
      <th>Exchange</th>
    </tr></thead><tbody>`;
  rows.forEach(r => {
    const badge = r.status === 'Priced'
      ? `<span class="status-priced">Priced</span>`
      : `<span class="status-upcoming">Upcoming</span>`;
    html += `<tr>
      <td>
        <div class="company-name">${esc(r.company)}</div>
        ${r.ticker !== '—' ? `<div class="ticker-badge">${esc(r.ticker)}</div>` : ''}
      </td>
      <td>${esc(r.date)}</td>
      <td>${badge}</td>
      <td>${esc(r.price)}</td>
      <td>${esc(r.shares)}</td>
      <td><span class="exch">${esc(r.exchange)}</span></td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('ipo-body').innerHTML = html;
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

loadIpos();
""" + _THEME_JS + """
</script>
</body>
</html>"""


FAQ_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>FAQ — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
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
    .faq-q .arrow { color: var(--muted); font-size: 0.8rem; transition: transform 0.28s cubic-bezier(0.4,0,0.2,1); }
    .faq-q.open .arrow { transform: rotate(90deg); }
    .faq-a { color: var(--muted); font-size: 0.88rem; line-height: 1.7; max-height: 0; overflow: hidden; transition: max-height 0.32s cubic-bezier(0.4,0,0.2,1), padding 0.2s ease; padding: 0; }
    .faq-a.open { max-height: 600px; padding-bottom: 18px; }
    .faq-a a { color: var(--accent); text-decoration: none; }
    .section-title { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 32px 0 8px; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Frequently Asked <span>Questions</span></h1>
  <p>Everything you need to know about ChartEdge.trade and our indicators</p>
</div>
<div class="container">

  <div class="section-title">Getting Started</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is ChartEdge.trade? <span class="arrow">▶</span></button>
    <div class="faq-a">ChartEdge.trade is a market intelligence dashboard built for retail traders. It combines 20+ free Pine Script indicators with professional-grade tools including live options flow, insider trading disclosures, gamma exposure charts, LSTM volatility forecasts, a pre-market scanner, and more — all in one place. Most tools are available free or with a 7-day trial on paid plans.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">What is Pine Script? <span class="arrow">▶</span></button>
    <div class="faq-a">Pine Script is TradingView's built-in scripting language for creating custom indicators and strategies directly on your charts. ChartEdge.trade generates ready-to-use Pine Script v6 code — you just copy it and paste it into TradingView's Pine Script editor, and it appears on your chart instantly. No coding knowledge required.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">How do I add an indicator to TradingView? <span class="arrow">▶</span></button>
    <div class="faq-a">1. Go to the Indicators page and pick an indicator.<br>2. Click <strong>Copy</strong> to copy the Pine Script code.<br>3. In TradingView, open the Pine Script editor (bottom of the chart).<br>4. Paste the code and click <strong>Add to chart</strong>.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Can I use a different trading platform? <span class="arrow">▶</span></button>
    <div class="faq-a">The Pine Script indicators are exclusive to TradingView — they won't work on ThinkOrSwim, MetaTrader, or other platforms. However, all of ChartEdge.trade's other tools (options flow, insider trading, gamma exposure, market heatmap, earnings calendar, etc.) run entirely in your browser and work alongside any broker or trading platform you use.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Do I need a paid TradingView account? <span class="arrow">▶</span></button>
    <div class="faq-a">No. All indicators on ChartEdge.trade are written in Pine Script v6 and work on free TradingView accounts.</div>
  </div>

  <div class="section-title">Account & Pricing</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Is ChartEdge.trade free? <span class="arrow">▶</span></button>
    <div class="faq-a">The core features are free — all Pine Script indicators, the market heatmap, earnings calendar, dividends calendar, unusual volume scanner, and news feed are available without paying. Basic ($9.99/mo) and Pro ($15.99/mo) plans unlock advanced tools like options flow, insider trading, gamma exposure, and the LSTM forecast. Your first paid subscription includes a 7-day free trial.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Is there a free trial for paid plans? <span class="arrow">▶</span></button>
    <div class="faq-a">Yes — your first subscription (Basic or Pro, monthly or yearly) includes a 7-day free trial. You won't be charged until the trial ends, and you can cancel any time before that with no cost.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Do I need an account? <span class="arrow">▶</span></button>
    <div class="faq-a">Yes — you need a free account to copy indicators. Sign up takes under a minute and gives you 3 free copies per day. An account also lets you save favorites and access paid plan features if you subscribe.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Can I cancel my subscription? <span class="arrow">▶</span></button>
    <div class="faq-a">Absolutely. You can manage or cancel your subscription at any time from the Billing page. There are no cancellation fees and your access continues until the end of the billing period you already paid for.</div>
  </div>
  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">How do I request a new indicator? <span class="arrow">▶</span></button>
    <div class="faq-a">Go to <a href="/request">Community → Request Indicator</a> and submit your idea. Other users can upvote requests — the most popular ones get built first.</div>
  </div>

  <div class="section-title">Other</div>

  <div class="faq-item">
    <button class="faq-q" onclick="toggleFaq(this)">Is this financial advice? <span class="arrow">▶</span></button>
    <div class="faq-a">No. ChartEdge.trade provides tools for analysis only. Nothing on this site should be considered financial advice. Always do your own research before making trading decisions.</div>
  </div>

</div>
<footer>© 2026 ChartEdge.trade · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
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
  <title>Privacy Policy — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
  <p>Questions? Email: <a href="mailto:shopverdict@gmail.com">shopverdict@gmail.com</a></p>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""


TERMS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Terms of Service — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Terms of <span>Service</span></h1>
</div>
<div class="container">
  <p class="updated">Last updated: April 3, 2026</p>

  <h2>1. Acceptance</h2>
  <p>By using ChartEdge.trade you agree to these terms. If you do not agree, please do not use the site.</p>

  <h2>2. Use of the service</h2>
  <ul>
    <li>ChartEdge.trade is provided free of charge for personal, non-commercial use</li>
    <li>You may copy and use any Pine Script code from this site on TradingView</li>
    <li>You may not scrape, reproduce, or redistribute the site's content or code for commercial purposes without permission</li>
  </ul>

  <h2>3. Not financial advice</h2>
  <p>Nothing on ChartEdge.trade constitutes financial, investment, or trading advice. All indicators are tools for analysis only. You are solely responsible for your own trading decisions. Past performance of any indicator does not guarantee future results.</p>

  <h2>4. Accounts</h2>
  <p>You are responsible for keeping your account credentials secure. We reserve the right to suspend accounts that abuse the service.</p>

  <h2>5. Disclaimer of warranties</h2>
  <p>ChartEdge.trade is provided "as is" without warranty of any kind. We do not guarantee uptime, accuracy of data, or fitness for any particular purpose.</p>

  <h2>6. Limitation of liability</h2>
  <p>ChartEdge.trade and its operators are not liable for any trading losses or damages arising from use of this site.</p>

  <h2>7. Changes</h2>
  <p>We may update these terms at any time. Continued use of the site after changes constitutes acceptance.</p>

  <h2>8. Contact</h2>
  <p>Questions? Email: <a href="mailto:shopverdict@gmail.com">shopverdict@gmail.com</a></p>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>
<script>""" + _THEME_JS + """</script>
</body>
</html>"""

# ── Insider Trading ───────────────────────────────────────────────────────────

_SP500_MAJOR = {
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","BRK.B","BRK.A",
    "UNH","JPM","JNJ","V","XOM","PG","MA","HD","CVX","MRK","ABBV","LLY",
    "PEP","KO","AVGO","COST","TMO","MCD","WMT","CSCO","ACN","ABT","CRM","BAC",
    "DHR","TXN","NEE","PM","LIN","ADBE","NKE","ORCL","AMD","QCOM","HON","UPS",
    "AMGN","IBM","LOW","SBUX","GS","MS","CAT","RTX","SPGI","BLK","AXP",
    "ISRG","GILD","MDT","ADP","BKNG","TJX","DE","MMC","SYK","ZTS","CI",
    "CB","MO","REGN","AON","SO","DUK","CL","BMY","MDLZ","GE","F","GM",
    "INTC","WFC","C","USB","PNC","TFC","SCHW","COF","BK","STT","DIS","NFLX",
    "PYPL","UBER","LYFT","SNAP","SPOT","SQ","COIN","HOOD","PLTR","RBLX",
    "RIVN","LCID","NIO","BIDU","BABA","JD","PDD","TSM","ASML","SAP",
}

_insider_cache = {"data": [], "ts": 0}

@app.route("/insider")
@login_required
def insider_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=insider")
    return render_template_string(INSIDER_HTML, current_user=current_user())

@app.route("/api/insider")
@login_required
def api_insider():
    import time, requests as _req, xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403

    now = time.time()
    if now - _insider_cache["ts"] < 600 and _insider_cache["data"]:
        return jsonify(_insider_cache["data"])

    headers = {"User-Agent": "ChartEdge.trade ayden.j.folkerts@gmail.com"}
    results = []

    # ── Part 1: SEC Form 4s — look up company CIKs first, then pull their filings ──
    try:
        # SEC's official ticker→CIK map (one request, instant lookup after)
        tickers_data = _req.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers, timeout=10
        ).json()
        cik_map = {}  # ticker → zero-padded CIK
        for entry in tickers_data.values():
            t = (entry.get("ticker") or "").upper()
            if t in _SP500_MAJOR:
                cik_map[t] = str(entry["cik_str"]).zfill(10)

        # Query submissions for each target company to find their recent Form 4 filings
        TARGET_TICKERS = [t for t in [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","BAC","GS",
            "WFC","MS","V","MA","XOM","CVX","JNJ","LLY","UNH","AMD",
            "INTC","CSCO","ORCL","DIS","NFLX","COIN","PLTR","GM","F","GE",
        ] if t in cik_map]

        def _get_recent_form4s(ticker):
            cik_padded = cik_map[ticker]
            cik        = cik_padded.lstrip("0") or "0"
            try:
                subs = _req.get(
                    f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
                    headers=headers, timeout=8
                ).json()
                recent    = subs.get("filings", {}).get("recent", {})
                forms     = recent.get("form", [])
                accnos    = recent.get("accessionNumber", [])
                dates     = recent.get("filingDate", [])
                prim_docs = recent.get("primaryDocument", [])
                out = []
                for i, form in enumerate(forms):
                    if form == "4":
                        doc = prim_docs[i] or ""
                        # Strip XSLT prefix e.g. "xslF345X06/form4.xml" → "form4.xml"
                        if "/" in doc:
                            doc = doc.split("/")[-1]
                        if not doc.endswith(".xml"):
                            doc = "form4.xml"
                        out.append({
                            "ticker":   ticker,
                            "date":     dates[i],
                            "cik":      cik,
                            "ac":       accnos[i].replace("-", ""),
                            "xml_file": doc,
                        })
                        if len(out) >= 3:
                            break
                return out
            except Exception:
                return []

        with ThreadPoolExecutor(max_workers=5) as pool:
            batches = list(pool.map(_get_recent_form4s, TARGET_TICKERS))
        all_filings = [f for batch in batches for f in batch]

        def _parse_form4(f):
            try:
                xml_text = _req.get(
                    f"https://www.sec.gov/Archives/edgar/data/{f['cik']}/{f['ac']}/{f['xml_file']}",
                    headers=headers, timeout=8
                ).text
                root = ET.fromstring(xml_text)

                company = root.findtext(".//issuerName") or f["ticker"]
                insider = root.findtext(".//rptOwnerName") or ""

                role_parts = []
                if root.findtext(".//isDirector") == "1":
                    role_parts.append("Director")
                if root.findtext(".//isOfficer") == "1":
                    role_parts.append(root.findtext(".//officerTitle") or "Officer")
                if root.findtext(".//isTenPercentOwner") == "1":
                    role_parts.append("10% Owner")
                role = ", ".join(role_parts) or "Insider"

                sh_tot, val_tot, txn_code = 0.0, 0.0, None
                for txn in root.findall(".//nonDerivativeTransaction"):
                    sh = txn.find(".//transactionShares/value")
                    pr = txn.find(".//transactionPricePerShare/value")
                    cd = txn.find(".//transactionAcquiredDisposedCode/value")
                    if sh is not None and sh.text:
                        s = float(sh.text)
                        sh_tot += s
                        if pr is not None and pr.text:
                            val_tot += s * float(pr.text)
                        if txn_code is None and cd is not None:
                            txn_code = cd.text

                return {
                    "company": company,
                    "insider": insider,
                    "role":    role,
                    "ticker":  f["ticker"],
                    "link":    f"https://www.sec.gov/Archives/edgar/data/{f['cik']}/{f['ac']}/",
                    "date":    f["date"],
                    "shares":  int(sh_tot) if sh_tot else None,
                    "value":   round(val_tot) if val_tot else None,
                    "txn":     txn_code,
                    "source":  "corporate",
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as pool:
            parsed = list(pool.map(_parse_form4, all_filings))
        results = [r for r in parsed if r]

    except Exception:
        pass

    # ── Part 2: Congressional trades via QuiverQuant ──
    try:
        cr = _req.get("https://api.quiverquant.com/beta/live/congresstrading", timeout=15)
        if cr.status_code == 200:
            congress_data = cr.json()
            congress_data.sort(key=lambda x: x.get("TransactionDate", ""), reverse=True)
            for h in congress_data[:60]:
                ticker = (h.get("Ticker") or "").strip().upper()
                if not ticker or ticker in ("", "N/A", "NONE"):
                    continue
                txn_type = h.get("Transaction", "")
                txn_code = "A" if "purchase" in txn_type.lower() else ("D" if "sale" in txn_type.lower() else None)
                house = h.get("House", "")
                role = "U.S. Senator" if house == "Senate" else "U.S. Representative"
                results.append({
                    "company": h.get("Description") or ticker,
                    "insider": h.get("Representative", ""),
                    "role":    role,
                    "ticker":  ticker,
                    "link":    "",
                    "date":    (h.get("TransactionDate") or "")[:10],
                    "shares":  None,
                    "value":   None,
                    "amount":  h.get("Range", ""),
                    "txn":     txn_code,
                    "source":  "congress",
                })
    except Exception:
        pass

    if not results:
        return jsonify({"error": "No data available — SEC or congressional sources may be temporarily unavailable."}), 503

    results.sort(key=lambda x: x.get("date", ""), reverse=True)
    _insider_cache["data"] = results
    _insider_cache["ts"]   = now
    return jsonify(results)

@app.route("/api/insider/debug")
@login_required
def api_insider_debug():
    import time, requests as _req, xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403
    headers = {"User-Agent": "ChartEdge.trade ayden.j.folkerts@gmail.com"}
    out = {}
    # Test 1: company_tickers.json
    try:
        r = _req.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        out["tickers_status"] = r.status_code
        data = r.json()
        aapl = next((e for e in data.values() if e.get("ticker","").upper() == "AAPL"), None)
        out["aapl_cik"] = aapl["cik_str"] if aapl else "not found"
    except Exception as e:
        out["tickers_error"] = str(e)
    # Test 2: submissions for AAPL
    try:
        cik = str(out.get("aapl_cik", "320193")).zfill(10)
        r2 = _req.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers, timeout=8)
        out["submissions_status"] = r2.status_code
        recent = r2.json().get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        form4_idx = next((i for i, f in enumerate(forms) if f == "4"), None)
        if form4_idx is not None:
            out["aapl_form4_date"] = recent["filingDate"][form4_idx]
            out["aapl_form4_doc"]  = recent["primaryDocument"][form4_idx]
            out["aapl_form4_accno"] = recent["accessionNumber"][form4_idx]
        else:
            out["aapl_form4"] = "none found"
    except Exception as e:
        out["submissions_error"] = str(e)
    # Test 3: AAPL XML fetch
    try:
        accno = out.get("aapl_form4_accno", "0001140361-26-013192")
        ac = accno.replace("-", "")
        cik_str = str(out.get("aapl_cik", 320193))
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_str}/{ac}/form4.xml"
        out["xml_url_tested"] = xml_url
        r4 = _req.get(xml_url, headers=headers, timeout=8)
        out["xml_status"] = r4.status_code
        out["xml_preview"] = r4.text[:300] if r4.status_code == 200 else r4.text[:200]
    except Exception as e:
        out["xml_error"] = str(e)
    # Test 4: congressional — QuiverQuant free endpoint
    try:
        r5 = _req.get("https://api.quiverquant.com/beta/live/congresstrading", timeout=10)
        out["quiver_status"] = r5.status_code
        if r5.status_code == 200:
            d = r5.json()
            out["quiver_count"] = len(d)
            out["quiver_sample"] = d[0] if d else None
    except Exception as e:
        out["quiver_error"] = str(e)
    return jsonify(out)

@app.route("/api/insider/ticker")
@login_required
def api_insider_ticker():
    import feedparser
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403
    ticker = request.args.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify([])
    try:
        import requests as _req, feedparser
        headers = {"User-Agent": "ChartEdge.trade ayden.j.folkerts@gmail.com", "Accept-Encoding": "gzip, deflate"}
        resp = _req.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&CIK=&type=4&dateb=&owner=include&count=20&search_text=&output=atom",
            headers=headers, timeout=15
        )
        rss = feedparser.parse(resp.text)
        results = []
        for entry in rss.entries[:20]:
            results.append({
                "title":   entry.get("title", ""),
                "link":    entry.get("link", ""),
                "date":    entry.get("updated", "")[:10] if entry.get("updated") else "",
                "summary": entry.get("summary", ""),
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


INSIDER_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Insider Trading · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; }
    """ + _NAV_CSS + """
    .page { max-width:900px; margin:0 auto; padding:36px 24px; }
    h1 { font-size:1.6rem; margin-bottom:6px; } h1 span { color:var(--accent); }
    .sub { color:var(--muted); font-size:.85rem; margin-bottom:24px; }
    .filter-section { margin-bottom:20px; display:flex; flex-direction:column; gap:10px; }
    .search-row { display:flex; gap:10px; }
    .search-row input { flex:1; background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:9px 14px; color:var(--text); font-size:.9rem; outline:none; }
    .search-row input:focus { border-color:var(--accent); }
    .search-row button { background:var(--accent); color:#fff; border:none; border-radius:6px; padding:9px 18px; font-size:.9rem; font-weight:600; cursor:pointer; }
    .search-row button:hover { opacity:.85; }
    .filter-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .filter-label { font-size:.75rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; white-space:nowrap; }
    .chip { background:var(--bg2); border:1px solid var(--border); color:var(--muted); padding:4px 12px; border-radius:20px; font-size:.78rem; cursor:pointer; white-space:nowrap; transition:all .15s; }
    .chip:hover { border-color:var(--accent); color:var(--accent); }
    .chip.active { background:#1a2a4a; border-color:var(--accent); color:var(--accent); font-weight:600; }
    .source-tabs { display:flex; gap:4px; }
    .stab { background:var(--bg2); border:1px solid var(--border); color:var(--muted); padding:5px 14px; border-radius:6px; font-size:.8rem; cursor:pointer; }
    .stab.active { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
    .back-btn { background:none; border:none; color:var(--accent); font-size:.85rem; cursor:pointer; margin-bottom:16px; padding:0; }
    .filing-table { width:100%; border-collapse:collapse; }
    .filing-table th { text-align:left; font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; padding:8px 12px; border-bottom:1px solid var(--border); }
    .filing-table td { padding:10px 12px; border-bottom:1px solid var(--border); font-size:.85rem; vertical-align:top; }
    .filing-table tr:hover td { background:var(--bg2); }
    .filing-link { color:var(--accent); text-decoration:none; font-size:.8rem; }
    .filing-link:hover { text-decoration:underline; }
    .date-badge { color:var(--muted); font-size:.78rem; white-space:nowrap; }
    .filing-table td:nth-child(3), .filing-table td:nth-child(4) { white-space:nowrap; }
    .loading { color:var(--muted); padding:40px; text-align:center; }
    .pro-badge { display:inline-block; background:#2a2000; color:#e3b341; border:1px solid #e3b34140; border-radius:4px; font-size:.7rem; font-weight:700; padding:2px 8px; margin-left:8px; vertical-align:middle; }
    footer { text-align:center; padding:32px 24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="page">
  <h1>Insider Trading <span>Feed</span> <span class="pro-badge">PRO</span></h1>
  <p class="sub">SEC Form 4 filings for major companies · U.S. Congressional STOCK Act disclosures</p>

  <div class="filter-section">
    <div class="search-row">
      <input type="text" id="ticker-input" placeholder="Filter by ticker (e.g. AAPL)" oninput="applyFilters()">
      <input type="text" id="name-input" placeholder="Filter by name (e.g. Pelosi)" oninput="applyFilters()" style="max-width:220px;">
    </div>
    <div class="filter-row">
      <span class="filter-label">Source:</span>
      <div class="source-tabs">
        <button class="stab active" data-src="all" onclick="setSource(this)">All</button>
        <button class="stab" data-src="congress" onclick="setSource(this)">Congress</button>
        <button class="stab" data-src="sec" onclick="setSource(this)">SEC</button>
      </div>
    </div>
  </div>

  <div id="content" class="loading">Loading filings…</div>
</div>

<footer>© 2026 ChartEdge.trade · Data via SEC EDGAR · Not financial advice</footer>
<script>""" + _THEME_JS + """
var allFilings = [];
var currentPage = 1;
var PAGE_SIZE = 15;
var _sourceFilter = 'all';
var _activeChip = null;

function setSource(btn) {
  _sourceFilter = btn.getAttribute('data-src');
  document.querySelectorAll('.stab').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  currentPage = 1;
  applyFilters();
}

function setPolitician(btn, name) {
  var nameInput = document.getElementById('name-input');
  if (_activeChip === btn) {
    // deselect
    btn.classList.remove('active');
    _activeChip = null;
    nameInput.value = '';
  } else {
    document.querySelectorAll('.chip').forEach(function(c){ c.classList.remove('active'); });
    btn.classList.add('active');
    _activeChip = btn;
    nameInput.value = name;
    // auto-switch to congress filter
    document.querySelectorAll('.stab').forEach(function(b){ b.classList.remove('active'); });
    document.querySelector('[data-src="congress"]').classList.add('active');
    _sourceFilter = 'congress';
  }
  currentPage = 1;
  applyFilters();
}

function applyFilters() {
  var ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
  var name   = document.getElementById('name-input').value.trim().toLowerCase();
  if (!name && _activeChip) { _activeChip.classList.remove('active'); _activeChip = null; }

  var filtered = allFilings.filter(function(f) {
    if (_sourceFilter === 'congress' && f.source !== 'congress') return false;
    if (_sourceFilter === 'sec'      && f.source === 'congress') return false;
    if (ticker && !(f.ticker || '').toUpperCase().includes(ticker)) return false;
    if (name   && !(f.insider || '').toLowerCase().includes(name)) return false;
    return true;
  });
  currentPage = 1;
  var label = '';
  if (name) label = 'Trades by "' + document.getElementById('name-input').value.trim() + '"';
  else if (ticker) label = 'Filings for ' + ticker;
  renderTable(filtered, label);
}

function renderTable(filings, title) {
  if (!filings.length) {
    document.getElementById('content').innerHTML = '<p style="color:var(--muted);padding:20px 0;">No filings found.</p>';
    return;
  }
  var totalPages = Math.ceil(filings.length / PAGE_SIZE);
  if (currentPage > totalPages) currentPage = totalPages;
  var start = (currentPage - 1) * PAGE_SIZE;
  var page  = filings.slice(start, start + PAGE_SIZE);

  var html = title ? '<h2 style="font-size:1rem;margin-bottom:14px;color:var(--muted);">' + title + '</h2>' : '';
  html += '<table class="filing-table"><thead><tr><th>Insider</th><th>Company</th><th>Role</th><th>Action</th><th>Value</th><th>Date</th><th>Link</th></tr></thead><tbody>';
  for (var i = 0; i < page.length; i++) {
    var f = page[i];
    var txnColor = f.txn === 'A' ? 'var(--green)' : f.txn === 'D' ? 'var(--red)' : 'var(--muted)';
    var txnLabel = f.txn === 'A' ? '▲ Buy' : f.txn === 'D' ? '▼ Sell' : '—';
    var sharesStr = f.shares != null ? (f.shares).toLocaleString() + ' shares' : (f.amount || txnLabel);
    var valueStr  = f.value  != null ? '$' + Number(f.value).toLocaleString() : (f.amount ? f.amount : '—');
    var sourceBadge = f.source === 'congress'
      ? '<span style="background:#1a2a4a;color:#58a6ff;border:1px solid #58a6ff40;border-radius:4px;font-size:.68rem;font-weight:700;padding:1px 6px;margin-left:6px;">CONGRESS</span>'
      : '<span style="background:#1a2a1a;color:#3fb950;border:1px solid #3fb95040;border-radius:4px;font-size:.68rem;font-weight:700;padding:1px 6px;margin-left:6px;">SEC</span>';
    var companyStr = (f.ticker ? '<strong>' + f.ticker + '</strong> · ' : '') + (f.company || '—');
    html += '<tr>';
    html += '<td>' + (f.insider || '—') + sourceBadge + '</td>';
    html += '<td>' + companyStr + '</td>';
    html += '<td style="color:var(--muted);font-size:.8rem">' + (f.role || '—') + '</td>';
    html += '<td style="font-weight:600;color:' + txnColor + '">' + sharesStr + '</td>';
    html += '<td style="color:' + txnColor + '">' + valueStr + '</td>';
    html += '<td class="date-badge">' + (f.date || '—') + '</td>';
    html += '<td><a class="filing-link" href="' + (f.link || '#') + '" target="_blank" rel="noopener">View →</a></td>';
    html += '</tr>';
  }
  html += '</tbody></table>';
  html += '<div style="display:flex;align-items:center;gap:12px;margin-top:16px;font-size:.85rem;">';
  html += '<button onclick="changePage(-1)" style="background:var(--bg2);border:1px solid var(--border);color:var(--text);padding:6px 14px;border-radius:6px;cursor:pointer;" ' + (currentPage <= 1 ? 'disabled style="opacity:.4;cursor:default;background:var(--bg2);border:1px solid var(--border);color:var(--muted);padding:6px 14px;border-radius:6px;"' : '') + '>← Prev</button>';
  html += '<span style="color:var(--muted)">Page ' + currentPage + ' of ' + totalPages + '</span>';
  html += '<button onclick="changePage(1)" style="background:var(--bg2);border:1px solid var(--border);color:var(--text);padding:6px 14px;border-radius:6px;cursor:pointer;" ' + (currentPage >= totalPages ? 'disabled style="opacity:.4;cursor:default;background:var(--bg2);border:1px solid var(--border);color:var(--muted);padding:6px 14px;border-radius:6px;"' : '') + '>Next →</button>';
  html += '</div>';
  document.getElementById('content').innerHTML = html;
}

function changePage(dir) {
  currentPage += dir;
  applyFilters();
}

async function loadAll() {
  try {
    var res = await fetch('/api/insider');
    var data = await res.json();
    if (data.error) { document.getElementById('content').innerHTML = '<p style="color:var(--red)">Error: ' + data.error + '</p>'; return; }
    allFilings = data;
    applyFilters();
  } catch(e) {
    document.getElementById('content').innerHTML = '<p style="color:var(--red)">Failed to load filings.</p>';
  }
}

loadAll();
</script>
</body>
</html>"""


# ── Pre/After Market Scanner ──────────────────────────────────────────────────

import threading as _threading

_premarket_cache    = {"data": [], "ts": 0}
_premarket_fetching = _threading.Event()

def _premarket_fetch_bg():
    import time, requests as _req
    from concurrent.futures import ThreadPoolExecutor
    headers = {"User-Agent": "Mozilla/5.0"}

    def _fetch_one(ticker):
        try:
            r = _req.get(
                f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=2d",
                headers=headers, timeout=6
            )
            result = r.json()["chart"]["result"][0]
            meta   = result["meta"]
            closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            price  = meta.get("regularMarketPrice")
            # Use second-to-last close as previous day, fall back to chartPreviousClose
            prev = closes[-2] if len(closes) >= 2 else meta.get("chartPreviousClose")
            if price is None or not prev:
                return None
            chg     = round(float(price) - float(prev), 2)
            chg_pct = round((chg / float(prev)) * 100, 2)
            return {"ticker": ticker, "price": round(float(price), 2),
                    "prev": round(float(prev), 2), "change": chg, "change_pct": chg_pct}
        except Exception:
            return None

    try:
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = [r for r in pool.map(_fetch_one, _SCANNER_TICKERS) if r]
        results.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        _premarket_cache["data"] = results
        _premarket_cache["ts"]   = time.time()
    except Exception:
        pass
    finally:
        _premarket_fetching.clear()

@app.route("/premarket")
@login_required
def premarket_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=premarket")
    return render_template_string(PREMARKET_HTML, current_user=current_user())

@app.route("/api/premarket")
@login_required
def api_premarket():
    import time
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403

    now = time.time()
    if now - _premarket_cache["ts"] < 300 and _premarket_cache["data"]:
        return jsonify(_premarket_cache["data"])

    if not _premarket_fetching.is_set():
        _premarket_fetching.set()
        _threading.Thread(target=_premarket_fetch_bg, daemon=True).start()

    if _premarket_cache["data"]:
        return jsonify(_premarket_cache["data"])
    return jsonify({"loading": True})


PREMARKET_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Pre/After Market · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; }
    """ + _NAV_CSS + """
    .page { max-width:960px; margin:0 auto; padding:36px 24px; }
    h1 { font-size:1.6rem; margin-bottom:6px; } h1 span { color:var(--accent); }
    .sub { color:var(--muted); font-size:.85rem; margin-bottom:8px; }
    .filter-row { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; align-items:center; }
    .filter-btn { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:6px 14px; color:var(--muted); font-size:.82rem; cursor:pointer; }
    .filter-btn.active { background:var(--accent); border-color:var(--accent); color:#fff; }
    .refresh-btn { background:none; border:1px solid var(--border); border-radius:6px; padding:6px 14px; color:var(--muted); font-size:.82rem; cursor:pointer; margin-left:auto; }
    .refresh-btn:hover { border-color:var(--accent); color:var(--accent); }
    .scanner-table { width:100%; border-collapse:collapse; }
    .scanner-table th { text-align:left; font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; padding:8px 12px; border-bottom:1px solid var(--border); cursor:pointer; user-select:none; }
    .scanner-table th:hover { color:var(--accent); }
    .scanner-table td { padding:10px 12px; border-bottom:1px solid var(--border); font-size:.88rem; }
    .scanner-table tr:hover td { background:var(--bg2); }
    .up { color:var(--green); font-weight:600; }
    .dn { color:var(--red);   font-weight:600; }
    .neutral { color:var(--muted); }
    .loading { color:var(--muted); padding:40px; text-align:center; }
    .pro-badge { display:inline-block; background:#2a2000; color:#e3b341; border:1px solid #e3b34140; border-radius:4px; font-size:.7rem; font-weight:700; padding:2px 8px; margin-left:8px; vertical-align:middle; }
    .last-updated { color:var(--muted); font-size:.75rem; margin-left:8px; }
    footer { text-align:center; padding:32px 24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="page">
  <h1>Pre / After-Hours <span>Scanner</span> <span class="pro-badge">PRO</span></h1>
  <p class="sub">Biggest movers sorted by % change from previous close · Refreshes every 5 min</p>

  <div class="filter-row">
    <button class="filter-btn active" onclick="setFilter('all', this)">All</button>
    <button class="filter-btn" onclick="setFilter('up', this)">Gainers</button>
    <button class="filter-btn" onclick="setFilter('down', this)">Losers</button>
    <span class="last-updated" id="updated-label"></span>
    <button class="refresh-btn" onclick="loadData(true)">↻ Refresh</button>
  </div>

  <div id="content" class="loading">Loading scanner…</div>
</div>

<footer>© 2026 ChartEdge.trade · Data via yfinance · Not financial advice</footer>
<script>""" + _THEME_JS + """
var allData = [];
var currentFilter = 'all';
var sortCol = 'change_pct';
var sortDir = -1;

function setFilter(f, btn) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  renderTable();
}

function renderTable() {
  var data = allData.filter(function(r) {
    if (currentFilter === 'up')   return r.change_pct > 0;
    if (currentFilter === 'down') return r.change_pct < 0;
    return true;
  });
  data.sort(function(a, b) { return (a[sortCol] > b[sortCol] ? 1 : -1) * sortDir; });

  if (!data.length) {
    document.getElementById('content').innerHTML = '<p style="color:var(--muted);padding:20px 0;">No data.</p>';
    return;
  }
  var html = '<table class="scanner-table"><thead><tr>';
  html += '<th onclick="sort(&quot;ticker&quot;)">Ticker</th>';
  html += '<th onclick="sort(&quot;price&quot;)">Price</th>';
  html += '<th onclick="sort(&quot;change&quot;)">Change</th>';
  html += '<th onclick="sort(&quot;change_pct&quot;)">% Change</th>';
  html += '<th onclick="sort(&quot;prev&quot;)">Prev Close</th>';
  html += '</tr></thead><tbody>';
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    var cls = r.change_pct > 0 ? 'up' : r.change_pct < 0 ? 'dn' : 'neutral';
    var arrow = r.change_pct > 0 ? '▲' : r.change_pct < 0 ? '▼' : '—';
    html += '<tr>';
    html += '<td><strong>' + r.ticker + '</strong></td>';
    html += '<td>$' + r.price.toFixed(2) + '</td>';
    html += '<td class="' + cls + '">' + arrow + ' $' + Math.abs(r.change).toFixed(2) + '</td>';
    html += '<td class="' + cls + '">' + arrow + ' ' + Math.abs(r.change_pct).toFixed(2) + '%</td>';
    html += '<td style="color:var(--muted)">$' + r.prev.toFixed(2) + '</td>';
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('content').innerHTML = html;
}

function sort(col) {
  if (sortCol === col) { sortDir *= -1; } else { sortCol = col; sortDir = -1; }
  renderTable();
}

async function loadData(force) {
  if (!force) document.getElementById('content').innerHTML = '<div class="loading">Loading scanner…</div>';
  try {
    var controller = new AbortController();
    var tid = setTimeout(function() { controller.abort(); }, 20000);
    var res  = await fetch('/api/premarket', {signal: controller.signal});
    clearTimeout(tid);
    var text = await res.text();
    var data;
    try { data = JSON.parse(text); } catch(je) {
      document.getElementById('content').innerHTML = '<p style="color:var(--red)">Bad response (HTTP ' + res.status + '): ' + text.slice(0,200) + '</p>';
      return;
    }
    if (data.loading) { document.getElementById('content').innerHTML = '<p style="color:var(--muted);padding:20px 0;">Fetching prices… auto-refreshing in 15s.</p>'; setTimeout(function(){ loadData(true); }, 15000); return; }
    if (data.error) { document.getElementById('content').innerHTML = '<p style="color:var(--red)">Error: ' + data.error + '</p>'; return; }
    allData = data;
    var now = new Date();
    document.getElementById('updated-label').textContent = 'Updated ' + now.toLocaleTimeString();
    renderTable();
  } catch(e) {
    document.getElementById('content').innerHTML = '<p style="color:var(--red)">Request failed: ' + e.message + '</p>';
  }
}

loadData(false);
</script>
</body>
</html>"""


# ── Dark Pool ─────────────────────────────────────────────────────────────────

_darkpool_cache = {"data": [], "ts": 0}

@app.route("/darkpool")
@login_required
def darkpool_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=darkpool")
    return render_template_string(DARKPOOL_HTML, current_user=current_user())

@app.route("/api/darkpool")
@login_required
def api_darkpool():
    import time, requests as _req, datetime as _dt
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403

    now = time.time()
    if now - _darkpool_cache["ts"] < 1800 and _darkpool_cache["data"]:
        return jsonify(_darkpool_cache["data"])

    headers = {"User-Agent": "ChartEdge.trade ayden.j.folkerts@gmail.com"}

    # FINRA OTC / ATS transparency data — weekly aggregated dark pool prints
    # Published biweekly at: https://ats.finra.org/
    # We use the JSON API that powers their public dashboard
    results = []
    try:
        # FINRA ATS Transparency Data - biweekly aggregate by security
        # Each row: ticker, shortName, totalWeeklyShareQuantity, totalWeeklyTradeCount
        # We pull the most recent two-week period available
        r = _req.get(
            "https://ats.finra.org/api/download/ats-transparency-data/all-ats",
            headers=headers, timeout=20
        )
        if r.status_code == 200:
            rows = r.json()
            # rows is a list of dicts with keys like:
            # issueSymbolIdentifier, issueName, totalWeeklyShareQuantity, totalWeeklyTradeCount,
            # lastSalePrice, weekDate, atsName
            # Aggregate by ticker across all ATSs for each week
            agg = {}
            for row in rows:
                tk  = (row.get("issueSymbolIdentifier") or "").strip().upper()
                if not tk:
                    continue
                shares = int(row.get("totalWeeklyShareQuantity") or 0)
                trades = int(row.get("totalWeeklyTradeCount") or 0)
                price  = row.get("lastSalePrice")
                week   = row.get("weekDate") or ""
                name   = row.get("issueName") or tk
                if tk not in agg:
                    agg[tk] = {"ticker": tk, "name": name, "shares": 0, "trades": 0,
                               "price": price, "week": week}
                agg[tk]["shares"] += shares
                agg[tk]["trades"] += trades

            # compute notional value
            for tk, d in agg.items():
                if d["price"]:
                    try:
                        d["notional"] = round(float(d["price"]) * d["shares"])
                    except Exception:
                        d["notional"] = None
                else:
                    d["notional"] = None

            results = sorted(agg.values(), key=lambda x: x["shares"], reverse=True)[:200]

    except Exception:
        pass

    # Fallback: use FINRA weekly ATS CSV if JSON fails
    if not results:
        try:
            # Try alternate FINRA endpoint
            import csv, io
            today = _dt.date.today()
            # FINRA publishes every other Monday; try last few weeks
            for delta in [0, 7, 14, 21]:
                d = today - _dt.timedelta(days=today.weekday() + delta)
                url = f"https://ats.finra.org/api/download/ats-transparency-data/{d.strftime('%Y-%m-%d')}"
                try:
                    r = _req.get(url, headers=headers, timeout=10)
                    if r.status_code == 200:
                        reader = csv.DictReader(io.StringIO(r.text))
                        agg = {}
                        for row in reader:
                            tk = (row.get("Issue Symbol") or row.get("issueSymbolIdentifier") or "").strip().upper()
                            if not tk:
                                continue
                            try:
                                shares = int(row.get("Total Weekly Share Quantity") or row.get("totalWeeklyShareQuantity") or 0)
                                trades = int(row.get("Total Weekly Trade Count") or row.get("totalWeeklyTradeCount") or 0)
                            except Exception:
                                continue
                            price_raw = row.get("Last Sale Price") or row.get("lastSalePrice")
                            price = None
                            try:
                                price = float(price_raw) if price_raw else None
                            except Exception:
                                pass
                            name = row.get("Issue Name") or row.get("issueName") or tk
                            week = row.get("Week Date") or row.get("weekDate") or ""
                            if tk not in agg:
                                agg[tk] = {"ticker": tk, "name": name, "shares": 0,
                                           "trades": 0, "price": price, "week": week}
                            agg[tk]["shares"] += shares
                            agg[tk]["trades"] += trades

                        for d2 in agg.values():
                            if d2["price"]:
                                try:
                                    d2["notional"] = round(float(d2["price"]) * d2["shares"])
                                except Exception:
                                    d2["notional"] = None
                            else:
                                d2["notional"] = None

                        results = sorted(agg.values(), key=lambda x: x["shares"], reverse=True)[:200]
                        if results:
                            break
                except Exception:
                    continue
        except Exception:
            pass

    # Last resort: well-known large dark pool prints from popular tickers using yfinance volume
    if not results:
        import yfinance as _yf
        BIG_TICKERS = [
            "SPY","QQQ","AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM",
            "AMD","INTC","BAC","C","GS","MS","XOM","CVX","V","MA","NFLX","COIN",
            "PLTR","RIVN","SOFI","MARA","F","GM","DIS","PYPL","SQ","UBER","HOOD",
        ]
        try:
            raw = _yf.download(
                " ".join(BIG_TICKERS), period="5d", interval="1d",
                group_by="ticker", auto_adjust=True, progress=False
            )
            today_str = str(_dt.date.today())
            for tk in BIG_TICKERS:
                try:
                    df = raw[tk].dropna()
                    if df.empty:
                        continue
                    last = df.iloc[-1]
                    price  = float(last["Close"])
                    volume = int(last["Volume"])
                    # Estimate dark pool ~30% of total volume (industry average)
                    dp_shares = int(volume * 0.30)
                    results.append({
                        "ticker":   tk,
                        "name":     tk,
                        "shares":   dp_shares,
                        "trades":   None,
                        "price":    round(price, 2),
                        "notional": round(price * dp_shares),
                        "week":     today_str,
                        "estimated": True,
                    })
                except Exception:
                    continue
            results.sort(key=lambda x: x["shares"], reverse=True)
        except Exception:
            pass

    if not results:
        return jsonify({"error": "Dark pool data temporarily unavailable."}), 503

    _darkpool_cache["data"] = results
    _darkpool_cache["ts"]   = now
    return jsonify(results)


@app.route("/api/darkpool/ticker")
@login_required
def api_darkpool_ticker():
    """Historical dark pool volume for a specific ticker (uses FINRA ATS weekly data)."""
    import time, requests as _req
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return jsonify({"error": "Pro required"}), 403

    ticker = (request.args.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    headers = {"User-Agent": "ChartEdge.trade ayden.j.folkerts@gmail.com"}
    weeks = []
    shares_list = []
    notional_list = []

    try:
        r = _req.get(
            "https://ats.finra.org/api/download/ats-transparency-data/all-ats",
            headers=headers, timeout=20
        )
        if r.status_code == 200:
            rows = r.json()
            # group by weekDate for this ticker
            by_week = {}
            for row in rows:
                tk = (row.get("issueSymbolIdentifier") or "").strip().upper()
                if tk != ticker:
                    continue
                week = row.get("weekDate") or ""
                shares = int(row.get("totalWeeklyShareQuantity") or 0)
                price  = row.get("lastSalePrice")
                if week not in by_week:
                    by_week[week] = {"shares": 0, "price": price}
                by_week[week]["shares"] += shares

            for wk in sorted(by_week.keys()):
                d = by_week[wk]
                weeks.append(wk)
                shares_list.append(d["shares"])
                try:
                    notional_list.append(round(float(d["price"]) * d["shares"]) if d["price"] else None)
                except Exception:
                    notional_list.append(None)
    except Exception:
        pass

    return jsonify({"ticker": ticker, "weeks": weeks,
                    "shares": shares_list, "notional": notional_list})


DARKPOOL_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Dark Pool · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; --purple:#bc8cff; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; --purple:#8250df; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; }
    """ + _NAV_CSS + """
    .page { max-width:1000px; margin:0 auto; padding:36px 24px; }
    h1 { font-size:1.6rem; margin-bottom:6px; } h1 span { color:var(--accent); }
    .sub { color:var(--muted); font-size:.85rem; margin-bottom:24px; }
    .toolbar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:20px; }
    .toolbar input { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:8px 14px; color:var(--text); font-size:.9rem; outline:none; width:200px; }
    .toolbar input:focus { border-color:var(--accent); }
    .toolbar select { background:var(--bg2); border:1px solid var(--border); border-radius:6px; padding:8px 12px; color:var(--text); font-size:.85rem; outline:none; cursor:pointer; }
    .refresh-btn { background:none; border:1px solid var(--border); border-radius:6px; padding:7px 14px; color:var(--muted); font-size:.82rem; cursor:pointer; margin-left:auto; }
    .refresh-btn:hover { border-color:var(--accent); color:var(--accent); }
    .dp-table { width:100%; border-collapse:collapse; }
    .dp-table th { text-align:left; font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; padding:8px 12px; border-bottom:1px solid var(--border); cursor:pointer; user-select:none; white-space:nowrap; }
    .dp-table th:hover { color:var(--accent); }
    .dp-table td { padding:10px 12px; border-bottom:1px solid var(--border); font-size:.85rem; vertical-align:middle; }
    .dp-table tr:hover td { background:var(--bg2); }
    .dp-table tr.selected td { background:#1a1f2e; }
    .ticker-link { color:var(--accent); text-decoration:none; font-weight:700; cursor:pointer; }
    .ticker-link:hover { text-decoration:underline; }
    .vol-bar { display:inline-block; height:6px; border-radius:3px; background:var(--purple); opacity:.7; vertical-align:middle; margin-left:6px; }
    .loading { color:var(--muted); padding:40px; text-align:center; }
    .pro-badge { display:inline-block; background:#2a2000; color:#e3b341; border:1px solid #e3b34140; border-radius:4px; font-size:.7rem; font-weight:700; padding:2px 8px; margin-left:8px; vertical-align:middle; }
    .info-banner { background:#161b22; border:1px solid var(--border); border-radius:8px; padding:14px 18px; margin-bottom:24px; font-size:.82rem; color:var(--muted); line-height:1.6; }
    .info-banner strong { color:var(--text); }
    .detail-panel { background:var(--bg2); border:1px solid var(--border); border-radius:10px; padding:22px; margin-bottom:24px; display:none; }
    .detail-panel.visible { display:block; }
    .detail-header { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:16px; }
    .detail-ticker { font-size:1.4rem; font-weight:700; }
    .detail-name { color:var(--muted); font-size:.85rem; margin-top:2px; }
    .close-btn { background:none; border:none; color:var(--muted); font-size:1.2rem; cursor:pointer; }
    .close-btn:hover { color:var(--text); }
    .detail-stats { display:flex; gap:20px; flex-wrap:wrap; margin-bottom:16px; }
    .stat-box { background:var(--bg3); border-radius:8px; padding:12px 16px; min-width:120px; }
    .stat-label { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
    .stat-val { font-size:1.1rem; font-weight:700; color:var(--text); margin-top:3px; }
    #detail-chart { height:180px; }
    .sort-arrow { color:var(--accent); margin-left:4px; font-size:.75rem; }
    footer { text-align:center; padding:32px 24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }
  </style>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>

<div class="page">
  <h1>Dark Pool <span>Flow</span> <span class="pro-badge">PRO</span></h1>
  <p class="sub">Weekly off-exchange block trade volume by security · Source: FINRA ATS Transparency Data</p>

  <div class="info-banner">
    <strong>What is dark pool trading?</strong> Dark pools are private exchanges where large institutional investors
    (hedge funds, banks, pension funds) execute block trades away from public markets to avoid price impact.
    FINRA requires all ATS (Alternative Trading System) operators to report weekly aggregate volume.
    High dark pool volume relative to normal volume often signals institutional accumulation or distribution.
  </div>

  <div id="detail-panel" class="detail-panel">
    <div class="detail-header">
      <div>
        <div class="detail-ticker" id="detail-ticker-label">—</div>
        <div class="detail-name" id="detail-name-label"></div>
      </div>
      <button class="close-btn" onclick="closeDetail()">✕</button>
    </div>
    <div class="detail-stats" id="detail-stats"></div>
    <div id="detail-chart"></div>
  </div>

  <div class="toolbar">
    <input type="text" id="search-input" placeholder="Search ticker or name…" oninput="renderTable()">
    <select id="sort-select" onchange="handleSortChange()">
      <option value="shares">Sort: Dark Pool Shares</option>
      <option value="notional">Sort: Notional Value</option>
      <option value="trades">Sort: Trade Count</option>
    </select>
    <button class="refresh-btn" onclick="loadData(true)">↻ Refresh</button>
  </div>

  <div id="content" class="loading">Loading dark pool data…</div>
</div>

<footer>© 2026 ChartEdge.trade · Data: FINRA ATS Transparency · Biweekly aggregate · Not financial advice</footer>
<script>""" + _THEME_JS + """
var allData = [];
var sortCol = 'shares';
var sortDir = -1;
var maxShares = 0;
var selectedTicker = null;

function handleSortChange() {
  sortCol = document.getElementById('sort-select').value;
  sortDir = -1;
  renderTable();
}

function fmt(n) {
  if (n == null) return '—';
  if (n >= 1e9) return '$' + (n/1e9).toFixed(1) + 'B';
  if (n >= 1e6) return '$' + (n/1e6).toFixed(1) + 'M';
  if (n >= 1e3) return '$' + (n/1e3).toFixed(1) + 'K';
  return '$' + n.toFixed(0);
}
function fmtShares(n) {
  if (n == null) return '—';
  if (n >= 1e9) return (n/1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n/1e3).toFixed(0) + 'K';
  return String(n);
}

function renderTable() {
  var q = (document.getElementById('search-input').value || '').trim().toUpperCase();
  var data = allData.filter(function(r) {
    if (!q) return true;
    return (r.ticker || '').includes(q) || (r.name || '').toUpperCase().includes(q);
  });
  data.sort(function(a, b) {
    var av = a[sortCol] == null ? -Infinity : a[sortCol];
    var bv = b[sortCol] == null ? -Infinity : b[sortCol];
    return (av > bv ? 1 : -1) * sortDir;
  });

  if (!data.length) {
    document.getElementById('content').innerHTML = '<p style="color:var(--muted);padding:20px 0;">No results.</p>';
    return;
  }

  var html = '<table class="dp-table"><thead><tr>';
  html += '<th onclick="sortBy(&quot;ticker&quot;)">Ticker</th>';
  html += '<th onclick="sortBy(&quot;shares&quot;)">Bought or Sold Shares' + (sortCol==='shares'?' <span class=sort-arrow>' + (sortDir>0?'▲':'▼') + '</span>':'') + '</th>';
  html += '<th onclick="sortBy(&quot;notional&quot;)">Notional' + (sortCol==='notional'?' <span class=sort-arrow>' + (sortDir>0?'▲':'▼') + '</span>':'') + '</th>';
  html += '<th onclick="sortBy(&quot;trades&quot;)">Trades' + (sortCol==='trades'?' <span class=sort-arrow>' + (sortDir>0?'▲':'▼') + '</span>':'') + '</th>';
  html += '<th>Price</th>';
  html += '<th>Week</th>';
  html += '</tr></thead><tbody>';

  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    var barW = maxShares > 0 ? Math.round((r.shares / maxShares) * 80) : 0;
    var isSel = r.ticker === selectedTicker;
    html += '<tr class="' + (isSel ? 'selected' : '') + '" onclick="showDetail(&quot;' + r.ticker + '&quot;)">';
    html += '<td><span class="ticker-link">' + r.ticker + '</span>';
    if (r.name && r.name !== r.ticker) html += ' <span style="color:var(--muted);font-size:.78rem;">' + r.name.slice(0,30) + '</span>';
    if (r.estimated) html += ' <span style="background:#1a1a30;color:#8b8bff;border:1px solid #8b8bff40;border-radius:3px;font-size:.65rem;padding:1px 5px;margin-left:4px;">EST</span>';
    html += '</td>';
    html += '<td>' + fmtShares(r.shares) + '<span class="vol-bar" style="width:' + barW + 'px;"></span></td>';
    html += '<td>' + fmt(r.notional) + '</td>';
    html += '<td>' + (r.trades != null ? r.trades.toLocaleString() : '—') + '</td>';
    html += '<td>' + (r.price != null ? '$' + Number(r.price).toFixed(2) : '—') + '</td>';
    html += '<td style="color:var(--muted);font-size:.8rem;">' + (r.week || '—') + '</td>';
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('content').innerHTML = html;
}

function sortBy(col) {
  if (sortCol === col) { sortDir *= -1; } else { sortCol = col; sortDir = -1; }
  document.getElementById('sort-select').value = col === 'trades' ? 'trades' : col === 'notional' ? 'notional' : 'shares';
  renderTable();
}

function closeDetail() {
  selectedTicker = null;
  document.getElementById('detail-panel').classList.remove('visible');
  renderTable();
}

async function showDetail(ticker) {
  if (selectedTicker === ticker) { closeDetail(); return; }
  selectedTicker = ticker;
  renderTable();

  var row = allData.find(function(r) { return r.ticker === ticker; });
  if (!row) return;

  var panel = document.getElementById('detail-panel');
  panel.classList.add('visible');
  document.getElementById('detail-ticker-label').textContent = ticker;
  document.getElementById('detail-name-label').textContent = row.name || '';

  var statsHtml = '';
  statsHtml += '<div class="stat-box"><div class="stat-label">Bought or Sold Shares</div><div class="stat-val">' + fmtShares(row.shares) + '</div></div>';
  statsHtml += '<div class="stat-box"><div class="stat-label">Notional</div><div class="stat-val">' + fmt(row.notional) + '</div></div>';
  if (row.trades != null) statsHtml += '<div class="stat-box"><div class="stat-label">Trades</div><div class="stat-val">' + row.trades.toLocaleString() + '</div></div>';
  if (row.price != null) statsHtml += '<div class="stat-box"><div class="stat-label">Last Price</div><div class="stat-val">$' + Number(row.price).toFixed(2) + '</div></div>';
  document.getElementById('detail-stats').innerHTML = statsHtml;

  // Fetch historical chart
  document.getElementById('detail-chart').innerHTML = '<p style="color:var(--muted);font-size:.82rem;padding:10px 0;">Loading history…</p>';
  try {
    var res = await fetch('/api/darkpool/ticker?ticker=' + encodeURIComponent(ticker));
    var hist = await res.json();
    if (hist.weeks && hist.weeks.length > 1) {
      var isDark = document.documentElement.getAttribute('data-theme') !== 'light';
      var layout = {
        margin:{t:10,b:40,l:60,r:10}, paper_bgcolor:'transparent', plot_bgcolor:'transparent',
        font:{color: isDark ? '#8b949e' : '#636c76', size:11},
        xaxis:{showgrid:false, linecolor: isDark ? '#30363d' : '#d0d7de', tickfont:{size:10}},
        yaxis:{gridcolor: isDark ? '#21262d' : '#eaeef2', tickfont:{size:10}},
        showlegend:false,
        hovermode:'x unified',
      };
      var trace = {
        x: hist.weeks,
        y: hist.shares,
        type: 'bar',
        marker: {color: isDark ? '#bc8cff' : '#8250df', opacity: 0.8},
        name: 'Bought or Sold Shares',
        hovertemplate: '%{y:,.0f} shares<extra></extra>',
      };
      Plotly.newPlot('detail-chart', [trace], layout, {displayModeBar:false, responsive:true});
    } else {
      document.getElementById('detail-chart').innerHTML = '<p style="color:var(--muted);font-size:.82rem;padding:10px 0;">Historical data not available for this ticker.</p>';
    }
  } catch(e) {
    document.getElementById('detail-chart').innerHTML = '<p style="color:var(--muted);font-size:.82rem;padding:10px 0;">Could not load history.</p>';
  }
}

async function loadData(force) {
  if (!force && allData.length) return;
  document.getElementById('content').innerHTML = '<div class="loading">Loading dark pool data…</div>';
  try {
    var res = await fetch('/api/darkpool');
    var data = await res.json();
    if (data.error) {
      document.getElementById('content').innerHTML = '<p style="color:var(--red)">Error: ' + data.error + '</p>';
      return;
    }
    allData = Array.isArray(data) ? data : [];
    maxShares = allData.length ? Math.max.apply(null, allData.map(function(r){return r.shares||0;})) : 0;
    renderTable();
  } catch(e) {
    document.getElementById('content').innerHTML = '<p style="color:var(--red)">Failed to load: ' + e.message + '</p>';
  }
}

loadData(false);
</script>
</body>
</html>"""


# ── Priority Service ──────────────────────────────────────────────────────────

PRIORITY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>""" + _META + """
  <title>Priority Service · ChartEdge.trade</title>
  <style>""" + _NAV_CSS + """
  body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; }
  .priority-wrap { max-width: 760px; margin: 0 auto; padding: 48px 24px 80px; }
  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }
  h1 span { color: var(--accent); }
  .pro-badge { background: linear-gradient(135deg,#f0c040,#e07b00); color: #000;
               font-size: .65rem; font-weight: 800; padding: 2px 7px; border-radius: 4px;
               vertical-align: middle; margin-left: 8px; letter-spacing: .06em; }
  .subtitle { color: var(--muted); font-size: .95rem; margin-bottom: 32px; }
  .form-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
               padding: 28px; margin-bottom: 40px; }
  .form-card h2 { font-size: 1.1rem; font-weight: 600; margin: 0 0 20px; }
  label { display: block; font-size: .8rem; color: var(--muted); margin-bottom: 4px; font-weight: 500; }
  input[type=text], textarea { width: 100%; box-sizing: border-box; background: var(--bg);
    border: 1px solid var(--border); color: var(--fg); border-radius: 8px;
    padding: 10px 14px; font-size: .9rem; font-family: inherit; outline: none; }
  input[type=text]:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 100px; }
  .form-row { margin-bottom: 18px; }
  .submit-btn { background: linear-gradient(135deg,#f0c040,#e07b00); color: #000;
                border: none; border-radius: 8px; padding: 10px 24px;
                font-weight: 700; font-size: .9rem; cursor: pointer; }
  .submit-btn:hover { opacity: .88; }
  .msg-ok  { background: rgba(63,185,80,.12); border: 1px solid rgba(63,185,80,.3);
             color: #3fb950; border-radius: 8px; padding: 10px 14px; margin-bottom: 20px; font-size: .9rem; }
  .msg-err { background: rgba(248,81,73,.12); border: 1px solid rgba(248,81,73,.3);
             color: #f85149; border-radius: 8px; padding: 10px 14px; margin-bottom: 20px; font-size: .9rem; }
  .reqs-section h2 { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; }
  .req-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
              padding: 18px 20px; margin-bottom: 14px; }
  .req-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
  .req-title { font-weight: 600; font-size: 1rem; margin-bottom: 4px; }
  .req-desc { color: var(--muted); font-size: .85rem; line-height: 1.5; }
  .req-meta { font-size: .75rem; color: var(--muted); margin-top: 10px; }
  .status-badge { display: inline-block; font-size: .7rem; font-weight: 700; padding: 3px 9px;
                  border-radius: 20px; white-space: nowrap; flex-shrink: 0; }
  .status-pending  { background: rgba(210,160,40,.15); color: #d2a028; border: 1px solid rgba(210,160,40,.3); }
  .status-review   { background: rgba(88,166,255,.15); color: #58a6ff; border: 1px solid rgba(88,166,255,.3); }
  .status-building { background: rgba(63,185,80,.15); color: #3fb950; border: 1px solid rgba(63,185,80,.3); }
  .status-done     { background: rgba(139,148,158,.15); color: #8b949e; border: 1px solid rgba(139,148,158,.3); }
  .empty { color: var(--muted); font-size: .9rem; text-align: center; padding: 32px 0; }
  </style>
</head>
<body>
""" + _NAV_LINKS + """
<div class="priority-wrap">
  <h1>Priority <span>Service</span> <span class="pro-badge">PRO</span></h1>
  <p class="subtitle">Skip the queue — Pro members get their indicator requests reviewed first and built sooner.</p>

  {% if msg_ok %}  <div class="msg-ok">{{ msg_ok }}</div>  {% endif %}
  {% if msg_err %} <div class="msg-err">{{ msg_err }}</div> {% endif %}

  <div class="form-card">
    <h2>Submit a Request</h2>
    <form method="POST" action="/priority">
      <div class="form-row">
        <label for="title">Indicator name or idea</label>
        <input type="text" id="title" name="title" placeholder="e.g. VWAP Anchored to Earnings" maxlength="120" required>
      </div>
      <div class="form-row">
        <label for="description">What should it do? How would you use it?</label>
        <textarea id="description" name="description" placeholder="Describe the logic, timeframes, inputs..." maxlength="1200" required></textarea>
      </div>
      <button type="submit" class="submit-btn">⚡ Submit Priority Request</button>
    </form>
  </div>

  <div class="reqs-section">
    <h2>Your Requests</h2>
    {% if my_reqs %}
      {% for r in my_reqs %}
      <div class="req-card">
        <div class="req-top">
          <div>
            <div class="req-title">{{ r.title }}</div>
            <div class="req-desc">{{ r.description }}</div>
          </div>
          <span class="status-badge status-{{ r.status }}">
            {% if r.status == 'pending' %}⏳ Pending
            {% elif r.status == 'review' %}🔍 In Review
            {% elif r.status == 'building' %}🔨 Building
            {% else %}✅ Done
            {% endif %}
          </span>
        </div>
        <div class="req-meta">Submitted {{ r.created.strftime('%b %d, %Y') if r.created else '' }}</div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty">No requests yet — submit your first one above.</div>
    {% endif %}
  </div>
</div>
""" + _THEME_JS + """
</body>
</html>"""


ADMIN_PRIORITY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>""" + _META + """
  <title>Priority Queue · Admin</title>
  <style>""" + _NAV_CSS + """
  body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 40px 24px 80px; }
  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { text-align: left; padding: 8px 12px; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 500; }
  td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
  .req-title { font-weight: 600; margin-bottom: 3px; }
  .req-desc { color: var(--muted); font-size: .8rem; line-height: 1.4; max-width: 360px; }
  select { background: var(--card); color: var(--fg); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: .8rem; cursor: pointer; }
  .user-col { font-size: .8rem; color: var(--muted); }
  .empty { color: var(--muted); padding: 32px 0; text-align: center; }
  </style>
</head>
<body>
""" + _NAV_LINKS + """
<div class="wrap">
  <h1>⚡ Priority Queue</h1>
  {% if reqs %}
  <table>
    <thead><tr><th>Request</th><th>User</th><th>Submitted</th><th>Status</th></tr></thead>
    <tbody>
    {% for r in reqs %}
    <tr>
      <td>
        <div class="req-title">{{ r.title }}</div>
        <div class="req-desc">{{ r.description }}</div>
      </td>
      <td class="user-col">{{ r.username }}<br>#{{ r.user_id }}</td>
      <td class="user-col">{{ r.created.strftime('%b %d, %Y') if r.created else '' }}</td>
      <td>
        <form method="POST" action="/admin/priority/{{ r.id }}/status" style="display:inline">
          <select name="status" onchange="this.form.submit()">
            <option value="pending"  {% if r.status=='pending'  %}selected{% endif %}>⏳ Pending</option>
            <option value="review"   {% if r.status=='review'   %}selected{% endif %}>🔍 In Review</option>
            <option value="building" {% if r.status=='building' %}selected{% endif %}>🔨 Building</option>
            <option value="done"     {% if r.status=='done'     %}selected{% endif %}>✅ Done</option>
          </select>
        </form>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No priority requests yet.</div>
  {% endif %}
</div>
""" + _THEME_JS + """
</body>
</html>"""


@app.route("/priority", methods=["GET", "POST"])
@login_required
def priority_page():
    plan = _get_user_plan(session.get("user_id"))
    if plan != "pro":
        return redirect("/pricing?upgrade=priority")
    user_id  = session["user_id"]
    username = session.get("username", "")
    msg_ok = msg_err = None
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        desc  = request.form.get("description", "").strip()
        if not title or not desc:
            msg_err = "Please fill in both fields."
        elif len(title) > 120 or len(desc) > 1200:
            msg_err = "Input too long."
        else:
            _run("INSERT INTO priority_requests (user_id, username, title, description) VALUES (%s, %s, %s, %s)",
                 (user_id, username, title, desc))
            msg_ok = "Request submitted! We'll review it within 24 hours."
    my_reqs = _q("SELECT * FROM priority_requests WHERE user_id=%s ORDER BY created DESC", (user_id,))
    return render_template_string(PRIORITY_HTML, my_reqs=my_reqs,
                                  msg_ok=msg_ok, msg_err=msg_err, current_user=current_user())


@app.route("/admin/priority")
@login_required
def admin_priority_page():
    if session.get("username") not in ("ayden", "admin"):
        return redirect("/")
    reqs = _q("SELECT * FROM priority_requests ORDER BY created DESC")
    return render_template_string(ADMIN_PRIORITY_HTML, reqs=reqs, current_user=current_user())


@app.route("/admin/priority/<int:req_id>/status", methods=["POST"])
@login_required
def admin_priority_update(req_id):
    if session.get("username") not in ("ayden", "admin"):
        return redirect("/")
    status = request.form.get("status", "pending")
    if status not in ("pending", "review", "building", "done"):
        status = "pending"
    _run("UPDATE priority_requests SET status=%s, updated=NOW() WHERE id=%s", (status, req_id))
    return redirect("/admin/priority")


# ── News ─────────────────────────────────────────────────────────────────────

NEWS_FEEDS = [
    ("Yahoo Finance",  "https://finance.yahoo.com/news/rssindex"),
    ("Reuters",        "https://feeds.reuters.com/reuters/businessNews"),
    ("MarketWatch",    "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Seeking Alpha",  "https://seekingalpha.com/market_currents.xml"),
    ("CNBC",           "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Benzinga",       "https://www.benzinga.com/feed"),
    ("Investopedia",   "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline"),
    ("Bloomberg",      "https://feeds.bloomberg.com/markets/news.rss"),
    ("Forbes Finance", "https://www.forbes.com/investing/feed/"),
    ("Motley Fool",    "https://www.fool.com/feeds/index.aspx"),
    ("WSJ Markets",    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("FT Markets",     "https://www.ft.com/rss/home/uk"),
]

_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _fmt_ts(ts: float) -> str:
    """Format a UTC unix timestamp as 'Apr 4, 2026 · 2:30 PM'."""
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{_MONTHS[dt.month-1]} {dt.day}, {dt.year} \u00b7 {hour}:{dt.minute:02d} {ampm}"

def _fetch_news(max_per_feed: int = 15) -> list[dict]:
    import feedparser, re, calendar
    articles = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                ts = 0
                published = ""
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    ts = calendar.timegm(entry.published_parsed)
                    published = _fmt_ts(ts)
                summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", "") or "")[:200]
                articles.append({
                    "source":    source,
                    "title":     entry.get("title", ""),
                    "link":      entry.get("link", "#"),
                    "published": published,
                    "summary":   summary,
                    "_ts":       ts,
                    "ts":        ts,
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
    plan = _get_user_plan(session.get("user_id")) if "user_id" in session else "free"
    return render_template_string(NEWS_HTML, articles=articles, current_user=current_user(), plan=plan)


@app.route("/api/news")
def api_news():
    return jsonify(_fetch_news())


@app.route("/api/news/ticker")
@login_required
def api_news_ticker():
    plan = _get_user_plan(session.get("user_id"))
    if plan == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import yfinance as yf, time as _time
    ticker = request.args.get("t", "").upper().strip()[:10]
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        raw = yf.Ticker(ticker).news or []
        articles = []
        for item in raw[:20]:
            # Handle both old and new yfinance news structure
            content = item.get("content", item)
            title   = content.get("title", item.get("title", ""))
            link    = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
            pub     = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
            ts      = content.get("pubDate", "") or item.get("providerPublishTime", 0)
            if isinstance(ts, (int, float)) and ts:
                published = _fmt_ts(float(ts))
            elif isinstance(ts, str) and ts:
                published = ts[:16]
            else:
                published = ""
            summary = content.get("summary", item.get("summary", ""))
            if not title:
                continue
            articles.append({"title": title, "link": link, "source": pub,
                              "published": published, "summary": summary[:200],
                              "ts": float(ts) if isinstance(ts, (int, float)) else 0})
        return jsonify({"ticker": ticker, "articles": articles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


NEWS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Market News — ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
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
    .pagination { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 24px; flex-wrap: wrap; }
    .pg-btn { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 6px 14px; color: var(--muted); cursor: pointer; font-family: monospace; font-size: .85rem; }
    .pg-btn:hover { border-color: var(--accent); color: var(--accent); }
    .pg-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; cursor: default; }
    .pg-btn:disabled { opacity: .4; cursor: not-allowed; }
    .pg-info { color: var(--muted); font-size: .8rem; }
    .ticker-search { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 18px 20px; margin-bottom: 24px; }
    .ticker-search h3 { font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 12px; }
    .ticker-search .row { display: flex; gap: 10px; }
    .ticker-search input { flex: 1; max-width: 200px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); font-family: monospace; font-size: .9rem; text-transform: uppercase; outline: none; }
    .ticker-search input:focus { border-color: var(--accent); }
    .ticker-search button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 18px; font-family: monospace; font-size: .85rem; cursor: pointer; font-weight: 700; }
    .ticker-search button:hover { opacity: .88; }
    .ticker-results { margin-top: 14px; display: none; }
    .upgrade-hint { color: var(--muted); font-size: .82rem; }
    .upgrade-hint a { color: var(--accent); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 40px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <div class="nav-links">
""" + _NAV_LINKS + """
  </div>
</nav>

<div class="hero">
  <h1><span>Market</span> News</h1>
  <p>Live financial news from Yahoo Finance, Reuters, and MarketWatch. Auto-refreshes every 5 minutes.</p>
</div>

<div class="container">
  {% if plan in ('basic', 'pro') %}
  <div class="ticker-search">
    <h3>Ticker News — Basic &amp; Pro</h3>
    <div class="row">
      <input id="ticker-news-input" placeholder="e.g. NVDA" maxlength="10"
             onkeydown="if(event.key==='Enter'){event.preventDefault();loadTickerNews();}">
      <button onclick="loadTickerNews()">Search</button>
    </div>
    <div class="ticker-results" id="ticker-results"></div>
  </div>
  {% else %}
  <div class="ticker-search">
    <h3>Ticker News</h3>
    <p class="upgrade-hint">Search for news on any specific stock — available on <a href="/pricing">Basic &amp; Pro</a>.</p>
  </div>
  {% endif %}
  <div class="filter-bar">
    <input id="search" placeholder="Search news…" oninput="filterNews()">
    <button class="src-btn active" onclick="filterSource(this, 'all')">All</button>
    <button class="src-btn" onclick="filterSource(this, 'Yahoo Finance')">Yahoo</button>
    <button class="src-btn" onclick="filterSource(this, 'Reuters')">Reuters</button>
    <button class="src-btn" onclick="filterSource(this, 'CNBC')">CNBC</button>
    <button class="src-btn" onclick="filterSource(this, 'MarketWatch')">MarketWatch</button>
    <button class="src-btn" onclick="filterSource(this, 'Benzinga')">Benzinga</button>
    <button class="src-btn" onclick="filterSource(this, 'WSJ Markets')">WSJ</button>
    <button class="refresh-btn" onclick="loadNews()">&#8635; Refresh</button>
  </div>

  <div class="news-list" id="news-list"></div>
  <div class="pagination" id="pagination"></div>
  <div id="status"></div>
</div>

<footer>© 2026 ChartEdge.trade · News sourced from public RSS feeds · Not financial advice · <a href="/privacy" style="color:inherit">Privacy</a> · <a href="/terms" style="color:inherit">Terms</a></footer>

<script>
function closeTickerNews() {
  var box = document.getElementById('ticker-results');
  box.style.display = 'none';
  box.innerHTML = '';
  document.getElementById('ticker-news-input').value = '';
}

function loadTickerNews() {
  var input  = document.getElementById('ticker-news-input');
  if (!input) return;
  var ticker = input.value.trim().toUpperCase();
  if (!ticker) return;
  input.value = ticker;
  var box = document.getElementById('ticker-results');
  box.style.display = 'block';
  box.innerHTML = '<div style="color:var(--muted);font-size:.85rem;padding:8px 0;">Loading ' + ticker + ' news...</div>';
  fetch('/api/news/ticker?t=' + encodeURIComponent(ticker))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) {
        box.innerHTML = '<div style="color:var(--red);font-size:.85rem;">\u26a0 ' + data.error + '</div>';
        return;
      }
      if (!data.articles || data.articles.length === 0) {
        box.innerHTML = '<div style="color:var(--muted);font-size:.85rem;">No news found for ' + ticker + '.</div>';
        return;
      }
      var html = '<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">'
        + '<button onclick="closeTickerNews()" style="background:none;border:none;color:var(--accent);cursor:pointer;font-size:1rem;padding:0;font-family:monospace;">\u2190 All news</button>'
        + '<span style="font-size:.78rem;color:var(--muted);">' + data.articles.length + ' articles for ' + data.ticker + '</span>'
        + '</div>';
      for (var i = 0; i < data.articles.length; i++) {
        var a = data.articles[i];
        html += '<div class="news-card" style="margin-bottom:10px;">'
          + '<div class="news-meta"><span class="source-tag">' + (a.source || 'News') + '</span>'
          + '<span class="news-time">' + (a.ts ? fmtDate(a.ts) : (a.published || '')) + '</span></div>'
          + '<div class="news-title"><a href="' + a.link + '" target="_blank" rel="noopener">' + a.title + '</a></div>'
          + (a.summary ? '<div class="news-summary">' + a.summary + '</div>' : '')
          + '</div>';
      }
      box.innerHTML = html;
    })
    .catch(function() {
      box.innerHTML = '<div style="color:var(--red);font-size:.85rem;">\u26a0 Failed to load ticker news.</div>';
    });
}

var allArticles  = [];
var activeSource = 'all';

function fmtDate(ts) {
  if (!ts) return '';
  var d = new Date(ts * 1000);
  var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var h = d.getHours() % 12 || 12;
  var m = d.getMinutes().toString().padStart(2, '0');
  var ampm = d.getHours() < 12 ? 'AM' : 'PM';
  return months[d.getMonth()] + ' ' + d.getDate() + ', ' + d.getFullYear() + ' \u00b7 ' + h + ':' + m + ' ' + ampm;
}
var currentPage  = 1;
var PAGE_SIZE    = 20;

function filterSource(btn, src) {
  activeSource = src;
  currentPage  = 1;
  document.querySelectorAll('.src-btn').forEach(function(b) { b.classList.remove('active'); });
  btn.classList.add('active');
  renderPage();
}

function filterNews() {
  currentPage = 1;
  renderPage();
}

function filteredArticles() {
  var q = (document.getElementById('search').value || '').toLowerCase();
  return allArticles.filter(function(a) {
    var srcMatch  = activeSource === 'all' || a.source === activeSource;
    var textMatch = !q || a.title.toLowerCase().includes(q) || (a.summary || '').toLowerCase().includes(q);
    return srcMatch && textMatch;
  });
}

function renderPage() {
  var items     = filteredArticles();
  var totalPages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  var start = (currentPage - 1) * PAGE_SIZE;
  var page  = items.slice(start, start + PAGE_SIZE);

  var list = document.getElementById('news-list');
  if (page.length === 0) {
    list.innerHTML = '<div class="no-news">No articles match.</div>';
  } else {
    var html = '';
    for (var i = 0; i < page.length; i++) {
      var a = page[i];
      html += '<div class="news-card">'
        + '<div class="news-meta"><span class="source-tag">' + a.source + '</span>'
        + '<span class="news-time">' + (a.ts ? fmtDate(a.ts) : (a.published || '')) + '</span></div>'
        + '<div class="news-title"><a href="' + a.link + '" target="_blank" rel="noopener">' + a.title + '</a></div>'
        + (a.summary ? '<div class="news-summary">' + a.summary + '...</div>' : '')
        + '</div>';
    }
    list.innerHTML = html;
  }

  // Pagination bar
  var pg = document.getElementById('pagination');
  if (totalPages <= 1) { pg.innerHTML = ''; return; }
  var pgHtml = '<button class="pg-btn" onclick="goPage(' + (currentPage - 1) + ')" ' + (currentPage === 1 ? 'disabled' : '') + '>\u2190 Prev</button>';
  var lo = Math.max(1, currentPage - 2);
  var hi = Math.min(totalPages, currentPage + 2);
  if (lo > 1)          pgHtml += '<span class="pg-info">1 \u2026</span>';
  for (var p = lo; p <= hi; p++) {
    pgHtml += '<button class="pg-btn' + (p === currentPage ? ' active' : '') + '" onclick="goPage(' + p + ')">' + p + '</button>';
  }
  if (hi < totalPages) pgHtml += '<span class="pg-info">\u2026 ' + totalPages + '</span>';
  pgHtml += '<button class="pg-btn" onclick="goPage(' + (currentPage + 1) + ')" ' + (currentPage === totalPages ? 'disabled' : '') + '>Next \u2192</button>';
  pgHtml += '<span class="pg-info">' + items.length + ' articles</span>';
  pg.innerHTML = pgHtml;
}

function goPage(n) {
  currentPage = n;
  renderPage();
  window.scrollTo(0, 0);
}

function loadNews() {
  document.getElementById('status').textContent = 'Refreshing...';
  fetch('/api/news')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      allArticles = data;
      currentPage = 1;
      renderPage();
      document.getElementById('status').textContent = 'Updated ' + new Date().toLocaleTimeString();
    })
    .catch(function(e) {
      document.getElementById('status').textContent = 'Error: ' + e.message;
    });
}

// Auto-refresh every 5 min
setInterval(loadNews, 5 * 60 * 1000);
loadNews();

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
  <title>Live Chart — ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; transition: background 0.2s, color 0.2s; }
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
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
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
    xaxis: { gridcolor: '#21262d', tickformat: '%m/%d', range: [new Date(Date.now() - 7*24*60*60*1000), new Date(futureDates[futureDates.length-1])] },
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

  Plotly.react('chart', traces, layout, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'], displaylogo: false });
}

// Initial load
refresh();
""" + _THEME_JS + """
</script>
</div>
</body>
</html>"""


GAMMA_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Gamma Exposure · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 900px; margin: 0 auto; padding: 28px 24px; }
    .search-row { display: flex; gap: 10px; margin-bottom: 24px; }
    .search-row input { flex: 1; background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; color: var(--text); font-size: .95rem; font-family: monospace; text-transform: uppercase; outline: none; }
    .search-row input:focus { border-color: var(--accent); }
    .search-row button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 10px 22px; cursor: pointer; font-weight: 600; font-size: .9rem; }
    .search-row button:hover { opacity: .88; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .summary-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
    .summary-card .val { font-size: 1.3rem; font-weight: 700; }
    .summary-card .lbl { font-size: .75rem; color: var(--muted); margin-top: 4px; }
    .pos { color: var(--green); } .neg { color: var(--red); } .neutral { color: var(--accent); }
    .regime-box { border-radius: 8px; padding: 14px 18px; margin-bottom: 24px; font-size: .9rem; }
    .regime-pos { background: #1f2d1f; border: 1px solid var(--green); color: var(--green); }
    .regime-neg { background: #2d1f1f; border: 1px solid var(--red);   color: var(--red); }
    .chart-section { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 24px; }
    .chart-section h3 { font-size: .85rem; color: var(--muted); margin-bottom: 16px; text-transform: uppercase; letter-spacing: .05em; }
    #gex-chart { width: 100%; height: 360px; }
    .expiry-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
    .exp-tab { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; font-size: .8rem; cursor: pointer; color: var(--muted); }
    .exp-tab.active { border-color: var(--accent); color: var(--accent); }
    .loading { text-align: center; padding: 60px; color: var(--muted); }
    .error-msg { background: #2d1f1f; border: 1px solid var(--red); border-radius: 8px; padding: 14px 18px; color: var(--red); margin-bottom: 20px; display: none; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
  </style>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Gamma <span>Exposure</span></h1>
  <p>Dealer GEX by strike — gamma flip level and hedging zones</p>
</div>
<div class="container">
  <div class="error-msg" id="error-msg"></div>
  <div class="search-row">
    <input id="ticker-input" placeholder="Enter ticker… (e.g. SPY, QQQ)" maxlength="10"
           onkeydown="if(event.key==='Enter') loadGamma()">
    <button onclick="loadGamma()">Search</button>
  </div>
  <div id="gamma-content" class="loading">Enter a ticker above to load gamma exposure.</div>
</div>
<footer>© 2026 ChartEdge.trade · GEX computed via Black-Scholes · Not financial advice</footer>
<script>
function loadGamma(exp) {
  var input  = document.getElementById('ticker-input');
  var ticker = input.value.trim().toUpperCase();
  if (!ticker) {
    document.getElementById('error-msg').textContent = 'Enter a ticker first (e.g. SPY).';
    document.getElementById('error-msg').style.display = 'block';
    return;
  }
  input.value = ticker;
  var url = '/api/gamma?ticker=' + encodeURIComponent(ticker) + (exp ? '&exp=' + encodeURIComponent(exp) : '');
  document.getElementById('gamma-content').innerHTML = '<div class="loading">Computing gamma exposure for ' + ticker + '...</div>';
  document.getElementById('error-msg').style.display = 'none';
  fetch(url)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.error) {
        document.getElementById('error-msg').textContent = data.error;
        document.getElementById('error-msg').style.display = 'block';
        document.getElementById('gamma-content').innerHTML = '';
        return;
      }
      renderGamma(data);
    })
    .catch(function() {
      document.getElementById('error-msg').textContent = 'Failed to load data.';
      document.getElementById('error-msg').style.display = 'block';
      document.getElementById('gamma-content').innerHTML = '';
    });
}

function loadGammaExp(el) {
  document.querySelectorAll('.exp-tab').forEach(function(t) { t.classList.remove('active'); });
  el.classList.add('active');
  loadGamma(el.getAttribute('data-exp'));
}

function renderGamma(d) {
  var tabs = '';
  for (var i = 0; i < d.expirations.length; i++) {
    var e = d.expirations[i];
    var active = e === d.expiry ? ' active' : '';
    tabs += '<span class="exp-tab' + active + '" data-exp="' + e + '" onclick="loadGammaExp(this)">' + e + '</span>';
  }

  var flipText   = d.flip_strike ? '$' + d.flip_strike : 'N/A';
  var regimeClass = d.positive_gamma ? 'regime-pos' : 'regime-neg';
  var regimeText  = d.positive_gamma
    ? '\u25b2 Positive Gamma \u2014 Dealers buy dips / sell rips. Expect mean-reversion and lower volatility.'
    : '\u25bc Negative Gamma \u2014 Dealers sell dips / buy rips. Expect trend acceleration and higher volatility.';
  var gammaLabel = d.positive_gamma ? 'Positive' : 'Negative';
  var gammaClass = d.positive_gamma ? 'pos' : 'neg';

  var summary = '<div class="summary-grid">'
    + '<div class="summary-card"><div class="val neutral">$' + d.spot + '</div><div class="lbl">Spot Price</div></div>'
    + '<div class="summary-card"><div class="val ' + gammaClass + '">' + gammaLabel + '</div><div class="lbl">Gamma Regime</div></div>'
    + '<div class="summary-card"><div class="val neutral">' + flipText + '</div><div class="lbl">Gamma Flip</div></div>'
    + '<div class="summary-card"><div class="val pos">$' + d.max_call_gex + '</div><div class="lbl">Max Call GEX</div></div>'
    + '<div class="summary-card"><div class="val neg">$' + d.max_put_gex  + '</div><div class="lbl">Max Put GEX</div></div>'
    + '</div>';

  var regime = '<div class="regime-box ' + regimeClass + '">' + regimeText + '</div>';

  var chart = '<div class="chart-section">'
    + '<h3>GEX by Strike \u00b7 ' + d.ticker + ' \u00b7 ' + d.expiry + '</h3>'
    + '<div id="gex-chart"></div>'
    + '</div>';

  document.getElementById('gamma-content').innerHTML =
    '<div class="expiry-tabs">' + tabs + '</div>' + summary + regime + chart;

  // Plotly bar chart
  var colors = d.gex.map(function(v) { return v >= 0 ? '#3fb950' : '#f85149'; });
  Plotly.newPlot('gex-chart', [{
    type: 'bar',
    x: d.strikes,
    y: d.gex,
    marker: { color: colors },
    hovertemplate: '$%{x}<br>GEX: %{y:.4f}<extra></extra>'
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: { color: '#8b949e', size: 11 },
    margin: { t: 20, r: 20, b: 60, l: 60 },
    xaxis: {
      gridcolor: '#21262d', tickformat: '.0f', title: 'Strike',
      tickprefix: '$'
    },
    yaxis: { gridcolor: '#21262d', title: 'GEX', zeroline: true, zerolinecolor: '#58a6ff', zerolinewidth: 2 },
    shapes: d.spot ? [{
      type: 'line', x0: d.spot, x1: d.spot, y0: 0, y1: 1, yref: 'paper',
      line: { color: '#58a6ff', width: 2, dash: 'dash' }
    }] : [],
    annotations: d.spot ? [{
      x: d.spot, y: 1, yref: 'paper', text: 'SPOT', showarrow: false,
      font: { color: '#58a6ff', size: 11 }, xanchor: 'left', yanchor: 'top'
    }] : []
  }, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'], displaylogo: false });
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('gamma-content').innerHTML = '<div class="loading">Enter a ticker above to load gamma exposure.</div>';
});
""" + _THEME_JS + """
</script>
</body>
</html>"""


GREEKS_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Greeks Dashboard · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 1100px; margin: 0 auto; padding: 28px 24px; }
    .search-row { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
    .search-row input { flex: 1; min-width: 140px; background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; color: var(--text); font-size: .95rem; font-family: monospace; text-transform: uppercase; outline: none; }
    .search-row input:focus { border-color: var(--accent); }
    .search-row button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 10px 22px; cursor: pointer; font-weight: 600; font-size: .9rem; }
    .search-row button:hover { opacity: .88; }
    .exp-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
    .exp-tab { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; font-size: .8rem; cursor: pointer; color: var(--muted); }
    .exp-tab.active { border-color: var(--accent); color: var(--accent); }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .scard { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }
    .scard .val { font-size: 1.25rem; font-weight: 700; }
    .scard .lbl { font-size: .72rem; color: var(--muted); margin-top: 4px; }
    .scard .sub { font-size: .7rem; color: var(--muted); margin-top: 2px; }
    .pos { color: var(--green); } .neg { color: var(--red); } .neutral { color: var(--accent); }
    .tabs { display: flex; gap: 0; margin-bottom: 0; border-bottom: 1px solid var(--border); }
    .tab-btn { background: none; border: none; color: var(--muted); padding: 10px 22px; cursor: pointer; font-size: .88rem; font-family: monospace; border-bottom: 2px solid transparent; margin-bottom: -1px; }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
    .table-wrap { background: var(--bg2); border: 1px solid var(--border); border-radius: 0 0 8px 8px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right; white-space: nowrap; }
    th:first-child { text-align: left; }
    td { padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: right; }
    td:first-child { text-align: left; font-weight: 600; }
    tr:hover td { background: var(--bg3); }
    tr.atm td { background: rgba(88,166,255,.08); }
    tr.atm td:first-child { color: var(--accent); }
    .loading { text-align: center; padding: 60px; color: var(--muted); }
    .error-msg { background: #2d1f1f; border: 1px solid var(--red); border-radius: 8px; padding: 14px 18px; color: var(--red); margin-bottom: 20px; display: none; }
    .disclaimer { color: var(--muted); font-size: .78rem; margin-top: 16px; padding: 12px; background: var(--bg2); border-radius: 6px; border: 1px solid var(--border); }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
    .greek-delta { color: var(--accent); }
    .greek-theta { color: var(--red); }
    .greek-vega  { color: #d29922; }
    .greek-rho   { color: var(--muted); }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Options <span>Greeks</span></h1>
  <p>Delta · Gamma · Theta · Vega · Rho — computed via Black-Scholes</p>
</div>
<div class="container">
  <div class="error-msg" id="error-msg"></div>
  <div class="search-row">
    <input id="ticker-input" placeholder="Ticker (e.g. AAPL, SPY)" maxlength="10"
           onkeydown="if(event.key==='Enter') loadGreeks()">
    <button onclick="loadGreeks()">Load Greeks</button>
  </div>
  <div id="greeks-content" class="loading">Enter a ticker above to load options Greeks.</div>
</div>
<footer>© 2026 ChartEdge.trade · Greeks via Black-Scholes · Not financial advice</footer>
<script>
var _greeksData = null;
var _activeTab  = 'calls';

function loadGreeks(exp) {
  var ticker = document.getElementById('ticker-input').value.trim().toUpperCase();
  if (!ticker) {
    showErr('Enter a ticker first.');
    return;
  }
  document.getElementById('ticker-input').value = ticker;
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('greeks-content').innerHTML = '<div class="loading">Fetching options chain for ' + ticker + '…</div>';
  var url = '/api/greeks?ticker=' + encodeURIComponent(ticker) + (exp ? '&exp=' + encodeURIComponent(exp) : '');
  fetch(url)
    .then(function(r) { return r.json(); })
    .then(function(d) {
      if (d.error) { showErr(d.error); document.getElementById('greeks-content').innerHTML = ''; return; }
      _greeksData = d;
      renderGreeks(d, _activeTab);
    })
    .catch(function() { showErr('Failed to load data.'); document.getElementById('greeks-content').innerHTML = ''; });
}

function showErr(msg) {
  var el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

function switchTab(tab) {
  _activeTab = tab;
  if (_greeksData) renderGreeks(_greeksData, tab);
}

function renderGreeks(d, tab) {
  // Expiration tabs
  var tabs = '';
  for (var i = 0; i < d.expirations.length; i++) {
    var e = d.expirations[i];
    var cls = e === d.expiry ? ' active' : '';
    tabs += '<span class="exp-tab' + cls + '" onclick="loadGreeks(&quot;' + e + '&quot;)">' + e + '</span>';
  }

  // ATM summary cards
  var atm = tab === 'calls' ? d.atm_call : d.atm_put;
  var cards = [
    { lbl: 'Spot',   val: '$' + d.spot,          cls: 'neutral', sub: ticker() },
    { lbl: 'Expiry', val: d.expiry,               cls: 'neutral', sub: d.T_days + ' days' },
    { lbl: 'Delta',  val: fmt(atm.delta, 4),      cls: atm.delta >= 0 ? 'pos' : 'neg', sub: 'ATM ' + (tab==='calls'?'call':'put') },
    { lbl: 'Gamma',  val: fmt(atm.gamma, 5),      cls: 'neutral', sub: 'per $1 move' },
    { lbl: 'Theta',  val: fmt(atm.theta, 4),      cls: 'neg',     sub: 'per day' },
    { lbl: 'Vega',   val: fmt(atm.vega, 4),       cls: 'neutral', sub: 'per 1% IV' },
    { lbl: 'Rho',    val: fmt(atm.rho, 4),        cls: atm.rho >= 0 ? 'pos' : 'neg', sub: 'per 1% rate' },
    { lbl: 'IV',     val: atm.iv + '%',            cls: 'neutral', sub: 'implied vol' },
  ];

  var cardsHtml = '<div class="summary-grid">';
  for (var i = 0; i < cards.length; i++) {
    var c = cards[i];
    cardsHtml += '<div class="scard"><div class="val ' + c.cls + '">' + c.val + '</div>'
              +  '<div class="lbl">' + c.lbl + '</div><div class="sub">' + c.sub + '</div></div>';
  }
  cardsHtml += '</div>';

  // Tab buttons
  var tabBtns = '<div class="tabs">'
    + '<button class="tab-btn' + (tab==='calls'?' active':'') + '" onclick="switchTab(&quot;calls&quot;)">Calls</button>'
    + '<button class="tab-btn' + (tab==='puts'?' active':'') + '" onclick="switchTab(&quot;puts&quot;)">Puts</button>'
    + '</div>';

  // Table
  var rows = tab === 'calls' ? d.calls : d.puts;
  var tableHtml = '<div class="table-wrap"><table><thead><tr>'
    + '<th>Strike</th><th>Mid</th><th>IV %</th><th>OI</th><th>Vol</th>'
    + '<th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th><th>Rho</th>'
    + '</tr></thead><tbody>';

  for (var i = 0; i < rows.length; i++) {
    var row = rows[i];
    var isAtm = Math.abs(row.strike - d.spot) === Math.min.apply(null, rows.map(function(r){ return Math.abs(r.strike - d.spot); }));
    var cls = isAtm ? ' class="atm"' : '';
    tableHtml += '<tr' + cls + '>'
      + '<td>' + row.strike + '</td>'
      + '<td>' + (row.mid > 0 ? '$' + row.mid : '—') + '</td>'
      + '<td>' + (row.iv > 0 ? row.iv + '%' : '—') + '</td>'
      + '<td>' + fmtInt(row.oi) + '</td>'
      + '<td>' + fmtInt(row.volume) + '</td>'
      + '<td class="greek-delta">' + (row.delta !== null ? row.delta : '—') + '</td>'
      + '<td>' + (row.gamma !== null ? row.gamma : '—') + '</td>'
      + '<td class="greek-theta">' + (row.theta !== null ? row.theta : '—') + '</td>'
      + '<td class="greek-vega">' + (row.vega !== null ? row.vega : '—') + '</td>'
      + '<td class="greek-rho">' + (row.rho !== null ? row.rho : '—') + '</td>'
      + '</tr>';
  }
  tableHtml += '</tbody></table></div>';

  document.getElementById('greeks-content').innerHTML =
    '<div class="exp-tabs">' + tabs + '</div>'
    + cardsHtml
    + tabBtns
    + tableHtml
    + '<div class="disclaimer">⚠ Greeks computed via Black-Scholes using implied volatility from Yahoo Finance. r = 4.5% risk-free rate assumed. For educational use only — not financial advice.</div>';
}

function ticker() {
  return document.getElementById('ticker-input').value.trim().toUpperCase();
}

function fmt(v, dec) {
  if (v === null || v === undefined) return '—';
  return parseFloat(v).toFixed(dec);
}

function fmtInt(v) {
  if (!v) return '—';
  if (v >= 1000000) return (v/1000000).toFixed(1) + 'M';
  if (v >= 1000) return (v/1000).toFixed(0) + 'K';
  return v;
}
""" + _THEME_JS + """
</script>
</body>
</html>"""


HEATMAP_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <title>Market Heatmap · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 24px 24px 16px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .82rem; }
    .page { max-width: 1400px; margin: 0 auto; padding: 14px 12px 32px; }
    .toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
    .toolbar .ts { color: var(--muted); font-size: .76rem; }
    .toolbar button { background: var(--bg2); border: 1px solid var(--border); color: var(--text); padding: 5px 12px; border-radius: 5px; cursor: pointer; font-size: .78rem; font-family: monospace; }
    .toolbar button:hover { border-color: var(--accent); color: var(--accent); }
    .legend { display: flex; align-items: center; gap: 5px; font-size: .7rem; color: var(--muted); }
    .legend-bar { display: flex; height: 10px; border-radius: 2px; overflow: hidden; }
    .legend-bar span { width: 20px; }
    .detail-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 18px; margin-bottom: 10px; display: none; align-items: center; gap: 24px; flex-wrap: wrap; }
    .detail-card.visible { display: flex; }
    .detail-ticker { font-size: 1.4rem; font-weight: 800; min-width: 60px; }
    .detail-field { display: flex; flex-direction: column; }
    .detail-field .lbl { font-size: .65rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 1px; }
    .detail-field .val { font-size: .95rem; font-weight: 700; }
    .detail-close { margin-left: auto; cursor: pointer; color: var(--muted); font-size: 1.1rem; padding: 3px 7px; border-radius: 4px; }
    .detail-close:hover { background: var(--bg3); color: var(--text); }
    .pos { color: var(--green); } .neg { color: var(--red); }
    #treemap-wrap { position: relative; width: 100%; background: #0a0e14; border-radius: 8px; overflow: hidden; }
    .tm-cell { position: absolute; cursor: pointer; overflow: hidden; display: flex; flex-direction: column; justify-content: center; align-items: center; transition: filter .12s; }
    .tm-cell:hover { filter: brightness(1.35); z-index: 5; }
    .tm-cell.active { outline: 2px solid rgba(255,255,255,.9); outline-offset: -2px; }
    .tm-cell .tc { font-weight: 800; color: #fff; line-height: 1; text-align: center; }
    .tm-cell .tp { color: rgba(255,255,255,.85); font-weight: 600; text-align: center; margin-top: 3px; }
    .tm-sector { position: absolute; display: flex; align-items: flex-start; pointer-events: none; overflow: hidden; }
    .tm-sector span { font-size: .62rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: rgba(255,255,255,.45); padding: 3px 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .loading { text-align: center; padding: 80px; color: var(--muted); font-size: .9rem; }
    .error-msg { background: #2d1f1f; border: 1px solid var(--red); border-radius: 8px; padding: 10px 14px; color: var(--red); margin-bottom: 10px; display: none; }
    .disclaimer { color: var(--muted); font-size: .72rem; margin-top: 10px; text-align: center; }
    footer { text-align: center; padding: 24px; color: var(--muted); font-size: .78rem; border-top: 1px solid var(--border); margin-top: 16px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Market <span>Heatmap</span></h1>
  <p>Cell size = market cap · Color = daily % change · Click for details</p>
</div>
<div class="page">
  <div class="error-msg" id="error-msg"></div>
  <div class="toolbar">
    <div class="ts" id="ts-label">Loading…</div>
    <div style="display:flex;gap:10px;align-items:center;">
      <div class="legend">
        <div class="legend-bar">
          <span style="background:#8b1a1a"></span><span style="background:#6b2020"></span>
          <span style="background:#3a2020"></span><span style="background:#1a1e27"></span>
          <span style="background:#1e3a1e"></span><span style="background:#1f5c1f"></span>
          <span style="background:#1a7a1a"></span>
        </div>
        <span style="margin-left:4px">-3%  0  +3%</span>
      </div>
      <button onclick="loadHeatmap()">Refresh</button>
    </div>
  </div>
  <div class="detail-card" id="detail-card">
    <div class="detail-ticker" id="dp-ticker">—</div>
    <div class="detail-field"><span class="lbl">Price</span><span class="val" id="dp-price">—</span></div>
    <div class="detail-field"><span class="lbl">Change</span><span class="val" id="dp-change">—</span></div>
    <div class="detail-field"><span class="lbl">Market Cap</span><span class="val" id="dp-mcap">—</span></div>
    <span class="detail-close" id="detail-close">&#x2715;</span>
  </div>
  <div id="treemap-wrap"><div class="loading" id="hm-loading">Fetching market data…</div></div>
  <p class="disclaimer">Data from Yahoo Finance · Size = market cap · Cached 5 min · Not financial advice</p>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>
var _lookup = {};
var _hmData = null;

function pctColor(pct) {
  if (Math.abs(pct) < 0.05) return '#1a1e27';
  var t = Math.max(-5, Math.min(5, pct)) / 5;
  if (t > 0) {
    var l = Math.round(14 + t * 16), s = Math.round(35 + t * 55);
    return 'hsl(120,' + s + '%,' + l + '%)';
  } else {
    var l = Math.round(14 + (-t) * 16), s = Math.round(35 + (-t) * 55);
    return 'hsl(0,' + s + '%,' + l + '%)';
  }
}

function loadHeatmap(silent) {
  if (!silent) {
    document.getElementById('error-msg').style.display = 'none';
    document.getElementById('hm-loading').style.display = 'block';
    document.getElementById('ts-label').textContent = 'Loading…';
    document.getElementById('detail-card').classList.remove('visible');
    // remove old cells
    document.querySelectorAll('.tm-cell,.tm-sector').forEach(function(el){ el.remove(); });
  }
  fetch('/api/heatmap')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.loading) {
        document.getElementById('ts-label').textContent = 'Warming up…';
        setTimeout(function(){ loadHeatmap(true); }, 3000);
        return;
      }
      if (d.error) {
        document.getElementById('error-msg').textContent = d.error;
        document.getElementById('error-msg').style.display = 'block';
        document.getElementById('hm-loading').style.display = 'none';
        return;
      }
      _hmData = d;
      renderTreemap(d);
    })
    .catch(function() {
      document.getElementById('error-msg').textContent = 'Failed to load data.';
      document.getElementById('error-msg').style.display = 'block';
      document.getElementById('hm-loading').style.display = 'none';
    });
}

function renderTreemap(d) {
  _lookup = {};
  d.stocks.forEach(function(s){ _lookup[s.ticker] = s; });

  var wrap   = document.getElementById('treemap-wrap');
  var W      = wrap.offsetWidth || 1200;
  var H      = Math.max(560, Math.round(W * 0.58));
  wrap.style.height = H + 'px';

  // Build D3 hierarchy
  var sectors = Object.keys(d.sectors);
  var children = sectors.map(function(sec) {
    var leaves = d.sectors[sec]
      .map(function(t){ return _lookup[t]; })
      .filter(Boolean)
      .map(function(st){ return { name: st.ticker, value: Math.max(st.mcap_num || 5e9, 5e9), st: st }; });
    return { name: sec, children: leaves };
  });

  var root = d3.hierarchy({ name: 'root', children: children })
    .sum(function(n){ return n.value || 0; })
    .sort(function(a, b){ return b.value - a.value; });

  d3.treemap()
    .size([W, H])
    .paddingOuter(3)
    .paddingTop(16)
    .paddingInner(1)
    .round(true)(root);

  // Remove old elements
  document.querySelectorAll('.tm-cell,.tm-sector').forEach(function(el){ el.remove(); });
  document.getElementById('hm-loading').style.display = 'none';

  // Render sector labels
  root.children.forEach(function(node) {
    var sw = node.x1 - node.x0, sh = node.y1 - node.y0;
    if (sw < 20 || sh < 14) return;
    var el = document.createElement('div');
    el.className = 'tm-sector';
    el.style.cssText = 'left:' + node.x0 + 'px;top:' + node.y0 + 'px;width:' + sw + 'px;height:16px;';
    el.innerHTML = '<span>' + node.data.name + '</span>';
    wrap.appendChild(el);
  });

  // Render leaf cells
  root.leaves().forEach(function(node) {
    var cw = node.x1 - node.x0, ch = node.y1 - node.y0;
    if (cw < 3 || ch < 3) return;
    var st   = node.data.st;
    var pct  = st.change_pct;
    var sign = pct >= 0 ? '+' : '';

    var el = document.createElement('div');
    el.className = 'tm-cell';
    el.setAttribute('data-t', st.ticker);
    el.style.cssText = 'left:' + node.x0 + 'px;top:' + node.y0 + 'px;width:' + cw + 'px;height:' + ch + 'px;background:' + pctColor(pct) + ';';

    var fontSize = Math.min(14, Math.max(7, Math.floor(Math.min(cw, ch) / 5)));
    var pctSize  = Math.max(6, fontSize - 2);
    var showPct  = ch > fontSize * 2.5;

    el.innerHTML = cw > 20 && ch > 14
      ? '<div class="tc" style="font-size:' + fontSize + 'px">' + st.ticker + '</div>'
        + (showPct ? '<div class="tp" style="font-size:' + pctSize + 'px">' + sign + pct + '%</div>' : '')
      : '';
    wrap.appendChild(el);
  });

  document.getElementById('ts-label').textContent = 'Updated ' + new Date(d.ts * 1000).toLocaleTimeString();
}

function showDetail(t) {
  var st = _lookup[t];
  if (!st) return;
  var sign = st.change_pct >= 0 ? '+' : '';
  var cls  = st.change_pct >= 0 ? 'pos' : 'neg';
  document.getElementById('dp-ticker').textContent = t;
  document.getElementById('dp-price').textContent  = '$' + st.price;
  document.getElementById('dp-change').className   = 'val ' + cls;
  document.getElementById('dp-change').textContent = sign + st.change_pct + '%';
  document.getElementById('dp-mcap').textContent   = st.mcap || '—';
  document.getElementById('detail-card').classList.add('visible');
  document.querySelectorAll('.tm-cell.active').forEach(function(c){ c.classList.remove('active'); });
  var el = document.querySelector('.tm-cell[data-t="' + t + '"]');
  if (el) el.classList.add('active');
}

document.addEventListener('click', function(e) {
  var cell = e.target.closest('.tm-cell');
  if (cell) { showDetail(cell.getAttribute('data-t')); return; }
  if (e.target.id === 'detail-close') {
    document.getElementById('detail-card').classList.remove('visible');
    document.querySelectorAll('.tm-cell.active').forEach(function(c){ c.classList.remove('active'); });
  }
});

window.addEventListener('resize', function() {
  if (_hmData) renderTreemap(_hmData);
});

loadHeatmap();
setTimeout(function(){ loadHeatmap(); }, 5 * 60 * 1000);
""" + _THEME_JS + """
</script>
</body>
</html>"""


TRUMP_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <title>Trump Tracker · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 28px 24px 16px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .hero h1 span { color: #e8543a; }
    .hero p { color: var(--muted); font-size: .82rem; }
    .layout { display: grid; grid-template-columns: 1fr 380px; gap: 16px; max-width: 1300px; margin: 0 auto; padding: 16px 14px 40px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

    /* Chart panel */
    .chart-panel { display: flex; flex-direction: column; gap: 12px; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    select.inst-select { background: var(--bg2); border: 1px solid var(--border); color: var(--text); padding: 8px 12px; border-radius: 7px; font-size: .9rem; font-family: monospace; outline: none; cursor: pointer; flex: 1; min-width: 180px; }
    select.inst-select:focus { border-color: var(--accent); }
    .period-btns { display: flex; gap: 4px; }
    .pb { background: var(--bg2); border: 1px solid var(--border); color: var(--muted); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: .78rem; font-family: monospace; }
    .pb.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .stats-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .stat-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; flex: 1; min-width: 100px; }
    .stat-card .sv { font-size: 1.2rem; font-weight: 800; }
    .stat-card .sl { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }
    .pos { color: var(--green); } .neg { color: var(--red); }
    #price-chart { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; min-height: 340px; }
    .chart-status { color: var(--muted); font-size: .78rem; padding: 4px 2px; }

    /* News panel */
    .news-panel { display: flex; flex-direction: column; gap: 0; }
    .news-header { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); padding: 0 0 8px; }
    .news-list { display: flex; flex-direction: column; gap: 0; max-height: 700px; overflow-y: auto; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; }
    .news-item { padding: 12px 14px; border-bottom: 1px solid var(--border); text-decoration: none; display: block; }
    .news-item:last-child { border-bottom: none; }
    .news-item:hover { background: var(--bg3); }
    .news-src { font-size: .65rem; color: var(--accent); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
    .news-title { font-size: .83rem; color: var(--text); font-weight: 600; line-height: 1.35; margin-bottom: 4px; }
    .news-time { font-size: .68rem; color: var(--muted); }
    .news-loading { padding: 40px; text-align: center; color: var(--muted); font-size: .85rem; }
    .disclaimer { color: var(--muted); font-size: .72rem; margin-top: 8px; }
    footer { text-align: center; padding: 24px; color: var(--muted); font-size: .78rem; border-top: 1px solid var(--border); margin-top: 8px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Trump <span>Tracker</span></h1>
  <p>Market reactions to Trump news, policies &amp; announcements</p>
</div>
<div class="layout">
  <!-- Left: chart -->
  <div class="chart-panel">
    <div class="controls">
      <select class="inst-select" id="inst-select">
        <option value="BTC-USD" selected>Bitcoin (BTC)</option>
        <option value="SPY">S&amp;P 500 (SPY)</option>
        <option value="QQQ">NASDAQ 100 (QQQ)</option>
        <option value="DIA">Dow Jones (DIA)</option>
        <option value="IWM">Russell 2000 (IWM)</option>
        <option value="DJT">Trump Media (DJT)</option>
        <option value="TLT">Bonds 20Y (TLT)</option>
        <option value="GC=F">Gold (GC=F)</option>
        <option value="USO">Oil (USO)</option>
        <option value="XLE">Energy (XLE)</option>
        <option value="XLF">Financials (XLF)</option>
        <option value="XLK">Tech (XLK)</option>
        <option value="IBIT">Bitcoin ETF (IBIT)</option>
        <option value="FXI">China (FXI)</option>
      </select>
      <div class="period-btns">
        <button class="pb active" data-p="1D" onclick="setPeriod(this)">1D</button>
        <button class="pb" data-p="1W" onclick="setPeriod(this)">1W</button>
        <button class="pb" data-p="1M" onclick="setPeriod(this)">1M</button>
        <button class="pb" data-p="3M" onclick="setPeriod(this)">3M</button>
      </div>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="sv" id="sc-price">—</div><div class="sl">Price</div></div>
      <div class="stat-card"><div class="sv" id="sc-chg">—</div><div class="sl">Period Change</div></div>
    </div>
    <div style="position:relative;">
      <button onclick="loadChart()" style="position:absolute;top:8px;left:8px;z-index:10;background:var(--bg3);border:1px solid var(--border);color:var(--muted);border-radius:5px;padding:3px 10px;font-size:.75rem;cursor:pointer;" title="Reset chart">↺ Reset</button>
      <div id="price-chart"></div>
    </div>
    <div class="chart-status" id="chart-status"></div>
    <p class="disclaimer">Data: Yahoo Finance · Not financial advice</p>
  </div>

  <!-- Right: news -->
  <div class="news-panel">
    <div class="news-header">Trump &amp; Policy News</div>
    <div class="news-list" id="news-list">
      <div class="news-loading">Loading news…</div>
    </div>
  </div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>
var _curPeriod = '1D';

function setPeriod(btn) {
  _curPeriod = btn.getAttribute('data-p');
  document.querySelectorAll('.pb').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  loadChart();
}

function fmtPrice(p, ticker) {
  if (p > 999) {
    return '$' + p.toLocaleString('en-US', {minimumFractionDigits: 0, maximumFractionDigits: 0});
  }
  return '$' + p.toFixed(2);
}

function loadChart() {
  var ticker = document.getElementById('inst-select').value;
  document.getElementById('chart-status').textContent = 'Loading ' + ticker + '…';
  fetch('/api/trump/chart?ticker=' + encodeURIComponent(ticker) + '&period=' + _curPeriod)
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.error) { document.getElementById('chart-status').textContent = 'Error: ' + d.error; return; }
      renderChart(d);
    })
    .catch(function(){ document.getElementById('chart-status').textContent = 'Failed to load chart.'; });
}

function renderChart(d) {
  var sign  = d.chg_pct >= 0 ? '+' : '';
  var color = d.chg_pct >= 0 ? '#3fb950' : '#f85149';
  document.getElementById('sc-price').textContent = fmtPrice(d.price, d.ticker);
  document.getElementById('sc-chg').className  = 'sv ' + (d.chg_pct >= 0 ? 'pos' : 'neg');
  document.getElementById('sc-chg').textContent = sign + d.chg_pct + '%';
  document.getElementById('chart-status').textContent = '';

  var dates = d.times.map(function(t){ return new Date(t); });
  var minP  = Math.min.apply(null, d.prices);
  var maxP  = Math.max.apply(null, d.prices);

  // Baseline shape
  var shapes = [{
    type: 'line', xref: 'paper', x0: 0, x1: 1,
    y0: d.prices[0], y1: d.prices[0],
    line: { color: 'rgba(139,148,158,0.4)', width: 1, dash: 'dot' },
  }];

  // Event marker shapes (vertical lines at news timestamps)
  var annotations = [];
  var events = d.events || [];
  events.forEach(function(ev, i) {
    var evDate = new Date(ev.ts);
    shapes.push({
      type: 'line', xref: 'x', yref: 'paper',
      x0: evDate, x1: evDate, y0: 0, y1: 1,
      line: { color: 'rgba(255,165,0,0.55)', width: 1.5, dash: 'dot' },
    });
    // Only label the top few to avoid clutter
    if (i < 6) {
      annotations.push({
        x: evDate, y: 1, xref: 'x', yref: 'paper',
        text: '📰', showarrow: false,
        font: { size: 12 },
        yanchor: 'top', xanchor: 'center',
        hovertext: ev.title,
      });
    }
  });

  Plotly.purge('price-chart');
  Plotly.newPlot('price-chart', [{
    x: dates, y: d.prices, type: 'scatter', mode: 'lines',
    line: { color: color, width: 2 },
    name: d.ticker,
    hovertemplate: fmtPrice(0, d.ticker).replace('0', '%{y:.2f}') + '<extra></extra>',
  }], {
    paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
    font: { color: '#e6edf3', family: 'monospace', size: 11 },
    xaxis: { gridcolor: '#21262d', tickformat: _curPeriod === '1D' ? '%H:%M' : '%m/%d' },
    yaxis: {
      gridcolor: '#21262d', side: 'right',
      range: [minP * 0.998, maxP * 1.002],
    },
    shapes: shapes,
    annotations: annotations,
    margin: { t: 20, r: 50, b: 36, l: 10 },
    showlegend: false,
  }, { responsive: true, displayModeBar: false });
}

function loadNews() {
  fetch('/api/trump/news')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.error || !d.articles || d.articles.length === 0) {
        document.getElementById('news-list').innerHTML = '<div class="news-loading">No Trump-related news found.</div>';
        return;
      }
      var html = '';
      d.articles.forEach(function(a) {
        html += '<a class="news-item" href="' + a.link + '" target="_blank" rel="noopener">'
              + '<div class="news-src">' + a.source + '</div>'
              + '<div class="news-title">' + a.title + '</div>'
              + '<div class="news-time">' + (a.published || '') + '</div>'
              + '</a>';
      });
      document.getElementById('news-list').innerHTML = html;
    })
    .catch(function(){
      document.getElementById('news-list').innerHTML = '<div class="news-loading">Failed to load news.</div>';
    });
}

document.getElementById('inst-select').addEventListener('change', loadChart);

loadChart();
loadNews();
setTimeout(loadNews, 5 * 60 * 1000);
""" + _THEME_JS + """
</script>
</body>
</html>"""


FLOW_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Options Flow · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 900px; margin: 0 auto; padding: 28px 24px; }
    .search-row { display: flex; gap: 10px; margin-bottom: 14px; }
    .search-row input { flex: 1; background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; color: var(--text); font-size: .95rem; font-family: monospace; text-transform: uppercase; outline: none; }
    .search-row input:focus { border-color: var(--accent); }
    .search-row button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 10px 22px; cursor: pointer; font-weight: 600; font-size: .9rem; }
    .search-row button:hover { opacity: .88; }
    .filter-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 24px; }
    .filter-row label { display: flex; flex-direction: column; gap: 4px; font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }
    .filter-row select { background: var(--bg2); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 7px 10px; font-size: .85rem; cursor: pointer; outline: none; }
    .filter-row select:focus { border-color: var(--accent); }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
    .summary-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }
    .summary-card .val { font-size: 1.4rem; font-weight: 700; }
    .summary-card .lbl { font-size: .75rem; color: var(--muted); margin-top: 4px; }
    .call-val { color: var(--green); }
    .put-val  { color: var(--red); }
    .neutral  { color: var(--accent); }
    .bar-section { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 24px; }
    .bar-section h3 { font-size: .85rem; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .05em; }
    .flow-bar { display: flex; height: 28px; border-radius: 6px; overflow: hidden; margin-bottom: 8px; }
    .flow-bar .call-seg { background: var(--green); display: flex; align-items: center; justify-content: center; font-size: .75rem; color: #000; font-weight: 700; transition: width .5s; }
    .flow-bar .put-seg  { background: var(--red);   display: flex; align-items: center; justify-content: center; font-size: .75rem; color: #fff; font-weight: 700; transition: width .5s; }
    .bar-legend { display: flex; gap: 20px; font-size: .8rem; color: var(--muted); }
    .bar-legend span { display: flex; align-items: center; gap: 6px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .table-section { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
    .table-section h3 { font-size: .85rem; color: var(--muted); padding: 14px 18px; border-bottom: 1px solid var(--border); text-transform: uppercase; letter-spacing: .05em; }
    table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    th { padding: 10px 14px; text-align: left; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }
    td { padding: 10px 14px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--bg3); }
    .badge-call { background: #1f2d1f; color: var(--green); padding: 2px 8px; border-radius: 4px; font-size: .75rem; font-weight: 700; }
    .badge-put  { background: #2d1f1f; color: var(--red);   padding: 2px 8px; border-radius: 4px; font-size: .75rem; font-weight: 700; }
    .itm { color: var(--green); font-size: .7rem; }
    .otm { color: var(--muted); font-size: .7rem; }
    .expiry-tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
    .exp-tab { background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 12px; font-size: .8rem; cursor: pointer; color: var(--muted); }
    .exp-tab.active { border-color: var(--accent); color: var(--accent); }
    .loading { text-align: center; padding: 60px; color: var(--muted); }
    .error-msg { background: #2d1f1f; border: 1px solid var(--red); border-radius: 8px; padding: 14px 18px; color: var(--red); margin-bottom: 20px; display: none; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Options <span>Flow</span></h1>
  <p>Calls vs puts volume, open interest, and top contracts</p>
</div>
<div class="container">
  <div class="search-row">
    <input id="ticker-input" placeholder="Ticker (e.g. SPY, AAPL)" maxlength="10"
           onkeydown="if(event.key==='Enter'){event.preventDefault();loadFlow();}">
    <button type="button" onclick="loadFlow()">Search</button>
  </div>
  <div class="filter-row">
    <label>Sort by
      <select id="flow-sort" onchange="loadFlow()">
        <option value="volume">Volume</option>
        <option value="oi">Open Interest</option>
        <option value="premium">Premium ($)</option>
        <option value="iv">Implied Vol</option>
      </select>
    </label>
    <label>Type
      <select id="flow-type" onchange="loadFlow()">
        <option value="all">All</option>
        <option value="calls">Calls only</option>
        <option value="puts">Puts only</option>
      </select>
    </label>
    <label>Moneyness
      <select id="flow-strike" onchange="loadFlow()">
        <option value="all">All strikes</option>
        <option value="itm">ITM only</option>
        <option value="otm">OTM only</option>
      </select>
    </label>
  </div>
  <div id="flow-content" style="min-height:60px;text-align:center;padding:32px;color:var(--muted);">JavaScript did not run — check browser console</div>
</div>
<footer>© 2026 ChartEdge.trade · Options data via yfinance · Not financial advice</footer>
<script>
function setContent(html) { document.getElementById('flow-content').innerHTML = html; }
function setError(msg)    { setContent('<div style="background:#2d1f1f;border:1px solid #f85149;border-radius:8px;padding:16px 20px;color:#f85149;margin-top:8px;">&#9888; ' + msg + '</div>'); }

function loadFlow(exp) {
  var ticker = (document.getElementById('ticker-input').value || '').trim().toUpperCase();
  if (!ticker) { setError('Enter a ticker first (e.g. SPY).'); return; }
  document.getElementById('ticker-input').value = ticker;
  var sort = (document.getElementById('flow-sort')||{}).value || 'volume';
  var type = (document.getElementById('flow-type')||{}).value || 'all';
  var strike = (document.getElementById('flow-strike')||{}).value || 'all';
  var url = '/api/flow?ticker=' + encodeURIComponent(ticker)
    + (exp ? '&exp=' + encodeURIComponent(exp) : '')
    + '&sort=' + sort + '&type=' + type + '&strike=' + strike;
  setContent('<div style="text-align:center;padding:48px;color:var(--muted);">Loading ' + ticker + '...</div>');

  var killTimer = setTimeout(function() {
    var c = document.getElementById('flow-content');
    if (c && c.innerHTML.indexOf('Loading') !== -1) {
      setError('Timed out — Yahoo Finance is not responding on this server. Try again.');
    }
  }, 13000);

  fetch(url)
    .then(function(r) { return r.text(); })
    .then(function(text) {
      clearTimeout(killTimer);
      try {
        var data = JSON.parse(text);
        if (data.error) { setError(data.error); return; }
        renderFlow(data);
      } catch(e) {
        setError('Error: ' + e.message + ' — server said: ' + text.slice(0, 120));
      }
    })
    .catch(function(err) {
      clearTimeout(killTimer);
      setError('Fetch failed: ' + err.message);
    });
}

function renderFlow(d) {
  var pcrColor = d.put_call_ratio > 1.2 ? 'put-val' : d.put_call_ratio < 0.8 ? 'call-val' : 'neutral';

  // Expiry tabs
  var tabs = '';
  for (var i = 0; i < d.expirations.length; i++) {
    var e = d.expirations[i];
    var active = e === d.expiry ? ' active' : '';
    tabs += '<span class="exp-tab' + active + '" data-exp="' + e + '" onclick="loadFlowExp(this)">' + e + '</span>';
  }

  // Summary cards
  var summary = '<div class="summary-grid">'
    + '<div class="summary-card"><div class="val call-val">' + fmt(d.call_volume) + '</div><div class="lbl">Call Volume</div></div>'
    + '<div class="summary-card"><div class="val put-val">'  + fmt(d.put_volume)  + '</div><div class="lbl">Put Volume</div></div>'
    + '<div class="summary-card"><div class="val call-val">' + fmt(d.call_oi)     + '</div><div class="lbl">Call OI</div></div>'
    + '<div class="summary-card"><div class="val put-val">'  + fmt(d.put_oi)      + '</div><div class="lbl">Put OI</div></div>'
    + '<div class="summary-card"><div class="val ' + pcrColor + '">' + d.put_call_ratio + '</div><div class="lbl">Put/Call Ratio</div></div>'
    + '</div>';

  // Volume bar
  var callLabel = d.call_pct > 10 ? d.call_pct + '%' : '';
  var putLabel  = d.put_pct  > 10 ? d.put_pct  + '%' : '';
  var bar = '<div class="bar-section">'
    + '<h3>Volume Sentiment</h3>'
    + '<div class="flow-bar">'
    + '<div class="call-seg" style="width:' + d.call_pct + '%">' + callLabel + '</div>'
    + '<div class="put-seg"  style="width:' + d.put_pct  + '%">' + putLabel  + '</div>'
    + '</div>'
    + '<div class="bar-legend">'
    + '<span><span class="dot" style="background:var(--green)"></span>Calls ' + d.call_pct + '%</span>'
    + '<span><span class="dot" style="background:var(--red)"></span>Puts '   + d.put_pct  + '%</span>'
    + '</div></div>';

  // Top contracts table
  var rows = '';
  for (var j = 0; j < d.top_contracts.length; j++) {
    var c = d.top_contracts[j];
    var itmClass = c.inTheMoney ? 'itm' : 'otm';
    var itmText  = c.inTheMoney ? 'ITM' : 'OTM';
    rows += '<tr>'
      + '<td><span class="badge-' + c.type + '">' + c.type.toUpperCase() + '</span></td>'
      + '<td>$' + c.strike + '</td>'
      + '<td>' + c.expiry + '</td>'
      + '<td>' + fmt(c.volume) + '</td>'
      + '<td>' + fmt(c.openInterest) + '</td>'
      + '<td>' + c.iv + '%</td>'
      + '<td>$' + c.lastPrice + '</td>'
      + '<td>$' + fmt(c.premium || 0) + '</td>'
      + '<td><span class="' + itmClass + '">' + itmText + '</span></td>'
      + '</tr>';
  }
  var sortLabel = {volume:'Volume', oi:'Open Interest', premium:'Premium', iv:'Implied Vol'}[d.sort_by || 'volume'];
  var typeLabel = d.type_filter === 'calls' ? ' \u00b7 calls' : d.type_filter === 'puts' ? ' \u00b7 puts' : '';
  var strikeLabel = d.strike_filter === 'itm' ? ' \u00b7 ITM' : d.strike_filter === 'otm' ? ' \u00b7 OTM' : '';
  var table = '<div class="table-section">'
    + '<h3>Top Contracts by ' + sortLabel + typeLabel + strikeLabel + ' \u2014 ' + d.ticker + ' \u00b7 ' + d.expiry + '</h3>'
    + '<table><thead><tr><th>Type</th><th>Strike</th><th>Expiry</th><th>Volume</th><th>OI</th><th>IV</th><th>Last</th><th>Premium</th><th></th></tr></thead>'
    + '<tbody>' + rows + '</tbody></table></div>';

  document.getElementById('flow-content').innerHTML =
    '<div class="expiry-tabs">' + tabs + '</div>' + summary + bar + table;
}

function loadFlowExp(el) {
  document.querySelectorAll('.exp-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  loadFlow(el.getAttribute('data-exp'));
}

function fmt(n) { return n >= 1000 ? (n/1000).toFixed(1) + 'K' : String(n); }

document.addEventListener('DOMContentLoaded', function() {
  setContent('<div style="text-align:center;padding:48px;color:var(--muted);">Enter a ticker above and click Search.</div>');
});
""" + _THEME_JS + """
</script>
</body>
</html>"""


REFER_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Refer a Friend · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .hero { text-align: center; padding: 40px 24px 28px; border-bottom: 1px solid var(--border); }
    .hero h1 { font-size: 1.6rem; margin-bottom: 8px; }
    .hero h1 span { color: var(--accent); }
    .hero p { color: var(--muted); font-size: .9rem; }
    .container { max-width: 600px; margin: 0 auto; padding: 32px 24px; }
    .stats-row { display: flex; gap: 16px; margin-bottom: 28px; }
    .stat-box { flex: 1; background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 18px; text-align: center; }
    .stat-box .num { font-size: 2rem; font-weight: 700; color: var(--accent); }
    .stat-box .lbl { font-size: .8rem; color: var(--muted); margin-top: 4px; }
    .section { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 22px; margin-bottom: 20px; }
    .section h3 { font-size: .95rem; margin-bottom: 16px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .link-row { display: flex; gap: 8px; }
    .link-row input { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; color: var(--text); font-size: .85rem; outline: none; min-width: 0; }
    .link-row button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 10px 18px; cursor: pointer; font-weight: 600; font-size: .85rem; white-space: nowrap; }
    .link-row button:hover { opacity: .85; }
    .copy-msg { font-size: .8rem; color: var(--green); margin-top: 8px; height: 16px; }
    .reward-row { margin-bottom: 14px; }
    .reward-row:last-child { margin-bottom: 0; }
    .reward-label { display: flex; justify-content: space-between; font-size: .85rem; margin-bottom: 6px; }
    .reward-label .badge { background: var(--bg3); border-radius: 4px; padding: 1px 7px; font-size: .75rem; color: var(--muted); }
    .progress-track { background: var(--bg3); border-radius: 4px; height: 8px; }
    .progress-fill { height: 8px; border-radius: 4px; transition: width .4s; }
    .step-row { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 14px; }
    .step-row:last-child { margin-bottom: 0; }
    .step-num { width: 28px; height: 28px; border-radius: 50%; background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; flex-shrink: 0; }
    .step-text { font-size: .9rem; line-height: 1.5; padding-top: 4px; }
    .step-text .sub { color: var(--muted); font-size: .85rem; }
    footer { text-align: center; padding: 32px 24px; color: var(--muted); font-size: .8rem; border-top: 1px solid var(--border); margin-top: 20px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="hero">
  <h1>Refer a <span>Friend</span></h1>
  <p>Share your link and earn free upgrades</p>
</div>
<div class="container">

  <div class="stats-row">
    <div class="stat-box">
      <div class="num">{{ referral_count }}</div>
      <div class="lbl">Friends Referred</div>
    </div>
    <div class="stat-box">
      <div class="num" style="font-size:1.1rem;letter-spacing:2px;padding-top:8px;">{{ ref_code }}</div>
      <div class="lbl">Your Code</div>
    </div>
  </div>

  <div class="section">
    <h3>Your Referral Link</h3>
    <div class="link-row">
      <input id="ref-link" type="text" value="{{ referral_link }}" readonly>
      <button onclick="copyLink()">Copy</button>
    </div>
    <div class="copy-msg" id="copy-msg"></div>
  </div>

  <div class="section">
    <h3>Rewards Progress</h3>
    <div class="reward-row">
      <div class="reward-label">
        <span>Basic free &nbsp;<span class="badge">2 referrals</span></span>
        <span>{{ [referral_count, 2]|min }}/2</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="background:var(--accent);width:{{ [referral_count * 50, 100]|min }}%;"></div>
      </div>
    </div>
    <div class="reward-row">
      <div class="reward-label">
        <span>Pro free &nbsp;<span class="badge">4 referrals</span></span>
        <span>{{ [referral_count, 4]|min }}/4</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" style="background:var(--green);width:{{ [referral_count * 25, 100]|min }}%;"></div>
      </div>
    </div>
  </div>

  <div class="section">
    <h3>How it works</h3>
    <div class="step-row">
      <div class="step-num">1</div>
      <div class="step-text">
        <strong>Share your link</strong>
        <div class="sub">Send it to a friend or post it anywhere.</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">2</div>
      <div class="step-text">
        <strong>They sign up</strong>
        <div class="sub">Your friend gets 7 days of Pro free when they register.</div>
      </div>
    </div>
    <div class="step-row">
      <div class="step-num">3</div>
      <div class="step-text">
        <strong>You get rewarded</strong>
        <div class="sub">2 referrals → Basic free &nbsp;·&nbsp; 4 referrals → Pro free</div>
      </div>
    </div>
  </div>

</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>
function copyLink() {
  const inp = document.getElementById('ref-link');
  navigator.clipboard.writeText(inp.value).then(() => {
    document.getElementById('copy-msg').textContent = '✓ Copied to clipboard';
    setTimeout(() => document.getElementById('copy-msg').textContent = '', 3000);
  });
}
""" + _THEME_JS + """
</script>
</body>
</html>"""


# ── Volatility Forecast ───────────────────────────────────────────────────────

@app.route("/volforecast")
@login_required
def volforecast_page():
    if _get_user_plan(session.get("user_id")) == "free":
        return redirect("/pricing?upgrade=volforecast")
    return render_template_string(VOLFORECAST_HTML, current_user=current_user())


@app.route("/api/volforecast")
@login_required
def volforecast_api():
    if _get_user_plan(session.get("user_id")) == "free":
        return jsonify({"error": "upgrade_required"}), 403
    import yfinance as yf
    import numpy as np
    from datetime import date

    ticker  = request.args.get("ticker", "SPY").upper()[:10]
    horizon = int(request.args.get("horizon", 10))

    try:
        raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if raw is None or len(raw) < 30:
            return jsonify({"error": "Not enough data for " + ticker}), 404

        if isinstance(raw.columns, __import__("pandas").MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        closes = raw["Close"].dropna()
        prices = closes.values.flatten().astype(float)
        dates  = [str(d.date()) for d in closes.index]

        # Daily log returns
        rets = np.diff(np.log(prices))

        # Rolling 20-day realized vol (annualized)
        window = 20
        rv = []
        for i in range(window, len(rets) + 1):
            rv.append(float(np.std(rets[i - window:i]) * np.sqrt(252) * 100))
        rv_dates = dates[window:]

        # EWMA volatility forecast (lambda=0.94, RiskMetrics standard)
        lam = 0.94
        ewma_var = float(np.var(rets[-30:], ddof=1))
        ewma_series = []
        for r_t in rets[-(len(rv)):]:
            ewma_var = lam * ewma_var + (1 - lam) * r_t ** 2
            ewma_series.append(float(np.sqrt(ewma_var) * np.sqrt(252) * 100))

        # Forecast next N days (EWMA mean-reverts toward long-run vol)
        long_run_vol = float(np.std(rets, ddof=1) * np.sqrt(252) * 100)
        reversion_speed = 0.10
        forecast = []
        v = ewma_series[-1]
        for _ in range(horizon):
            v = v + reversion_speed * (long_run_vol - v)
            forecast.append(round(v, 2))

        # Volatility percentile rank — strict-less-than for true rank
        current_rv = rv[-1]
        pct_rank = round(float(np.mean(np.array(rv) < current_rv)) * 100, 1)

        # Regime
        p25, p75 = np.percentile(rv, 25), np.percentile(rv, 75)
        if current_rv <= p25:
            regime, regime_color = "Low", "#3fb950"
        elif current_rv <= p75:
            regime, regime_color = "Medium", "#e3b341"
        else:
            regime, regime_color = "High", "#f85149"
        if current_rv > np.percentile(rv, 90):
            regime, regime_color = "Extreme", "#ff4444"

        # IV from options (ATM, nearest expiry) — average of nearest call+put for stability
        iv_current = None
        try:
            t = yf.Ticker(ticker)
            exps = t.options
            if exps:
                chain = t.option_chain(exps[0])
                spot  = float(t.fast_info.last_price)
                ivs = []
                for side in (chain.calls, chain.puts):
                    if side is None or side.empty:
                        continue
                    nearest = side.iloc[(side["strike"] - spot).abs().argsort()[:1]]
                    iv_val = float(nearest.iloc[0].get("impliedVolatility") or 0)
                    if not (0 < iv_val < 5):   # 5.0 = 500% — anything above is bad data
                        continue
                    ivs.append(iv_val)
                if ivs:
                    iv_pct = (sum(ivs) / len(ivs)) * 100
                    # Sanity bound: realised vol is up to ~100% for normal tickers
                    if 1 <= iv_pct <= 300:
                        iv_current = round(iv_pct, 1)
        except Exception:
            pass

        return jsonify({
            "ticker":       ticker,
            "dates":        rv_dates,
            "rv":           [round(v, 2) for v in rv],
            "ewma":         [round(v, 2) for v in ewma_series],
            "forecast":     forecast,
            "horizon":      horizon,
            "current_rv":   round(current_rv, 2),
            "long_run_vol": round(long_run_vol, 2),
            "pct_rank":     pct_rank,
            "regime":       regime,
            "regime_color": regime_color,
            "iv_current":   iv_current,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


VOLFORECAST_HTML = """<!DOCTYPE html>
<html data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">""" + _META + """
  <title>Volatility Forecast · ChartEdge.trade</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
    nav { background:var(--bg2); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; }
    .logo { font-size:1.1rem; font-weight:bold; text-decoration:none; }
    """ + _NAV_CSS + """
    .tv-banner { background:#1c2333; border:1px solid #2d3a52; border-left:3px solid #58a6ff; border-radius:8px; padding:10px 16px; margin:16px 0; display:flex; align-items:center; gap:10px; font-size:.83rem; color:#8b949e; }
    .tv-banner strong { color:#e6edf3; }
    .tv-logo { font-size:1rem; }
    .page { max-width:960px; margin:0 auto; padding:28px 20px 60px; }
    h1 { font-size:1.5rem; margin-bottom:4px; } h1 span { color:var(--accent); }
    .sub { color:var(--muted); font-size:.83rem; margin-bottom:8px; }
    .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:16px; }
    .controls input, .controls select { background:var(--bg2); border:1px solid var(--border); color:var(--text); padding:8px 12px; border-radius:7px; font-size:.9rem; outline:none; }
    .controls input:focus, .controls select:focus { border-color:var(--accent); }
    .controls input { width:110px; }
    .btn-load { background:var(--accent); color:#fff; border:none; border-radius:7px; padding:8px 20px; font-size:.9rem; font-weight:600; cursor:pointer; }
    .btn-load:hover { opacity:.85; }
    .stats-bar { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
    .stat { background:var(--bg2); border:1px solid var(--border); border-radius:8px; padding:12px 16px; flex:1; min-width:120px; }
    .stat .sv { font-size:1.2rem; font-weight:800; }
    .stat .sl { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:3px; }
    #vol-chart { background:var(--bg2); border:1px solid var(--border); border-radius:8px; margin-bottom:16px; }
    .loading { color:var(--muted); padding:40px; text-align:center; }
    .err { color:var(--red); padding:20px; }
    footer { text-align:center; padding:32px 24px; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:20px; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>Volatility <span>Forecast</span></h1>
  <p class="sub">Realized volatility, EWMA forecast &amp; regime detection</p>

  <div class="tv-banner">
    <span class="tv-logo">📊</span>
    <span><strong>TradingView subscription required</strong> — A paid TradingView plan is required to use this tool.</span>
  </div>

  <div class="controls">
    <input type="text" id="ticker-in" value="SPY" placeholder="Ticker" onkeydown="if(event.key==='Enter')load()">
    <select id="horizon-sel">
      <option value="5">5-day forecast</option>
      <option value="10" selected>10-day forecast</option>
      <option value="20">20-day forecast</option>
    </select>
    <button class="btn-load" onclick="load()">Load</button>
  </div>

  <div class="stats-bar" id="stats-bar" style="display:none">
    <div class="stat"><div class="sv" id="st-rv">—</div><div class="sl">Current RV (20d)</div></div>
    <div class="stat"><div class="sv" id="st-iv">—</div><div class="sl">Implied Vol (ATM)</div></div>
    <div class="stat"><div class="sv" id="st-fcast">—</div><div class="sl">Forecast End</div></div>
    <div class="stat"><div class="sv" id="st-lrv">—</div><div class="sl">1-Year Avg Vol</div></div>
    <div class="stat"><div class="sv" id="st-pct">—</div><div class="sl">Vol Percentile</div></div>
    <div class="stat"><div class="sv" id="st-regime">—</div><div class="sl">Regime</div></div>
  </div>

  <div id="vol-chart" style="min-height:360px"><div class="loading">Enter a ticker and click Load</div></div>
</div>
<footer>© 2026 ChartEdge.trade · Not financial advice</footer>
<script>""" + _THEME_JS + """
function load() {
  var ticker  = document.getElementById('ticker-in').value.trim().toUpperCase();
  var horizon = document.getElementById('horizon-sel').value;
  if (!ticker) return;
  document.getElementById('vol-chart').innerHTML = '<div class="loading">Loading ' + ticker + '…</div>';
  document.getElementById('stats-bar').style.display = 'none';
  fetch('/api/volforecast?ticker=' + encodeURIComponent(ticker) + '&horizon=' + horizon)
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.error) { document.getElementById('vol-chart').innerHTML = '<div class="err">Error: ' + d.error + '</div>'; return; }
      render(d);
    })
    .catch(function(){ document.getElementById('vol-chart').innerHTML = '<div class="err">Failed to load.</div>'; });
}

function render(d) {
  // Stats
  document.getElementById('st-rv').textContent     = d.current_rv + '%';
  document.getElementById('st-iv').textContent     = d.iv_current != null ? d.iv_current + '%' : '—';
  document.getElementById('st-fcast').textContent  = d.forecast[d.forecast.length - 1] + '%';
  document.getElementById('st-lrv').textContent    = d.long_run_vol + '%';
  document.getElementById('st-pct').textContent    = d.pct_rank + 'th %ile';
  document.getElementById('st-regime').textContent = d.regime;
  document.getElementById('st-regime').style.color = d.regime_color;
  document.getElementById('st-fcast').style.color  = d.forecast[d.forecast.length-1] > d.current_rv ? 'var(--red)' : 'var(--green)';
  document.getElementById('stats-bar').style.display = 'flex';

  // Build forecast dates (business days approx)
  var lastDate = new Date(d.dates[d.dates.length - 1]);
  var fcastDates = [];
  var d2 = new Date(lastDate);
  for (var i = 0; i < d.horizon; i++) {
    d2.setDate(d2.getDate() + 1);
    if (d2.getDay() === 0) d2.setDate(d2.getDate() + 1);
    if (d2.getDay() === 6) d2.setDate(d2.getDate() + 1);
    fcastDates.push(new Date(d2).toISOString().slice(0,10));
  }

  var traces = [
    {
      x: d.dates, y: d.rv, name: 'Realized Vol (20d)',
      type: 'scatter', mode: 'lines',
      line: { color: '#58a6ff', width: 2 },
    },
    {
      x: d.dates, y: d.ewma, name: 'EWMA Vol',
      type: 'scatter', mode: 'lines',
      line: { color: '#e3b341', width: 1.5, dash: 'dot' },
    },
    {
      x: [d.dates[d.dates.length-1]].concat(fcastDates),
      y: [d.ewma[d.ewma.length-1]].concat(d.forecast),
      name: 'Forecast',
      type: 'scatter', mode: 'lines+markers',
      line: { color: '#f85149', width: 2, dash: 'dash' },
      marker: { size: 4 },
    },
  ];

  if (d.iv_current != null) {
    traces.push({
      x: d.dates, y: Array(d.dates.length).fill(d.iv_current),
      name: 'Implied Vol (ATM)',
      type: 'scatter', mode: 'lines',
      line: { color: '#3fb950', width: 1.5, dash: 'dot' },
    });
  }

  Plotly.react('vol-chart', traces, {
    paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
    font: { color: '#e6edf3', family: 'monospace', size: 11 },
    xaxis: { gridcolor: '#21262d', tickformat: '%m/%d/%y' },
    yaxis: { gridcolor: '#21262d', title: 'Annualized Vol %', side: 'right' },
    legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
    shapes: [
      {
        type: 'line', xref: 'x', yref: 'y',
        x0: d.dates[d.dates.length-1], x1: d.dates[d.dates.length-1],
        y0: 0, y1: Math.max.apply(null, d.rv) * 1.1,
        line: { color: 'rgba(139,148,158,0.4)', width: 1, dash: 'dot' },
      },
      {
        type: 'line', xref: 'paper', yref: 'y',
        x0: 0, x1: 1,
        y0: d.ewma[d.ewma.length-1], y1: d.ewma[d.ewma.length-1],
        line: { color: 'rgba(248,81,73,0.35)', width: 1.5, dash: 'dot' },
      },
    ],
    margin: { t: 20, r: 60, b: 60, l: 10 },
  }, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['zoom2d','pan2d','select2d','lasso2d','zoomIn2d','zoomOut2d','autoScale2d','hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'], displaylogo: false });
}

load();
</script>
</body>
</html>"""


SETTINGS_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Settings · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .page { max-width: 640px; margin: 0 auto; padding: 36px 24px; }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    h1 span { color: var(--accent); }
    .sub { color: var(--muted); font-size: .88rem; margin-bottom: 32px; }
    .section { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 20px; overflow: hidden; }
    .section-title { font-size: .72rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; padding: 14px 20px; border-bottom: 1px solid var(--border); }
    .setting-row { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); gap: 20px; }
    .setting-row:last-child { border-bottom: none; }
    .setting-label { font-size: .9rem; font-weight: 600; }
    .setting-desc { font-size: .78rem; color: var(--muted); margin-top: 3px; }
    /* Toggle switch */
    .toggle { position: relative; width: 44px; height: 24px; flex-shrink: 0; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .toggle-track { position: absolute; inset: 0; background: var(--bg3); border: 1px solid var(--border); border-radius: 12px; cursor: pointer; transition: background .2s; }
    .toggle input:checked + .toggle-track { background: var(--accent); border-color: var(--accent); }
    .toggle-track::after { content: ''; position: absolute; width: 18px; height: 18px; background: #fff; border-radius: 50%; top: 2px; left: 2px; transition: transform .2s; }
    .toggle input:checked + .toggle-track::after { transform: translateX(20px); }
    /* Inputs */
    .setting-input { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); font-size: .88rem; width: 200px; outline: none; font-family: monospace; }
    .setting-input:focus { border-color: var(--accent); }
    .btn { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 18px; font-size: .85rem; font-weight: 600; cursor: pointer; }
    .btn:hover { opacity: .88; }
    .btn-danger { background: var(--red); }
    .btn-muted { background: var(--bg3); color: var(--text); border: 1px solid var(--border); }
    .btn-muted:hover { border-color: var(--accent); }
    .flash { padding: 10px 14px; border-radius: 6px; font-size: .85rem; margin-bottom: 16px; display: none; }
    .flash.ok  { background: #1a2d1a; border: 1px solid var(--green); color: var(--green); }
    .flash.err { background: #2d1515; border: 1px solid var(--red);   color: var(--red); }
    /* Delete modal */
    .modal-bg { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.75); z-index: 1000; align-items: center; justify-content: center; }
    .modal-bg.open { display: flex; }
    .modal { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 32px 28px; max-width: 420px; width: 90%; text-align: center; }
    .modal h2 { margin-bottom: 10px; }
    .modal p { color: var(--muted); font-size: .88rem; margin-bottom: 24px; line-height: 1.5; }
    .modal-btns { display: flex; gap: 12px; justify-content: center; }
    /* Theme selector */
    .theme-opts { display: flex; gap: 8px; }
    .theme-opt { padding: 7px 16px; border-radius: 6px; border: 1px solid var(--border); cursor: pointer; font-size: .83rem; background: var(--bg3); color: var(--muted); }
    .theme-opt.active { border-color: var(--accent); color: var(--accent); background: var(--bg2); font-weight: 600; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">
  <h1>⚙️ <span>Settings</span></h1>
  <p class="sub">Manage your account preferences</p>

  <div id="flash" class="flash"></div>

  <!-- Appearance -->
  <div class="section">
    <div class="section-title">Appearance</div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Theme</div>
        <div class="setting-desc">Choose your preferred color scheme</div>
      </div>
      <div class="theme-opts">
        <button class="theme-opt {{ 'active' if user_theme == 'dark' else '' }}" onclick="setTheme('dark')">☾ Dark</button>
        <button class="theme-opt {{ 'active' if user_theme == 'light' else '' }}" onclick="setTheme('light')">☀ Light</button>
      </div>
    </div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Copy Confirmation Toast</div>
        <div class="setting-desc">Show a toast notification when you copy a script</div>
      </div>
      <label class="toggle">
        <input type="checkbox" id="toast-toggle" {{ 'checked' if copy_toast else '' }} onchange="savePref('copy_toast', this.checked ? 1 : 0)">
        <span class="toggle-track"></span>
      </label>
    </div>
  </div>

  <!-- Chart defaults -->
  <div class="section">
    <div class="section-title">Chart Defaults</div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Default Ticker</div>
        <div class="setting-desc">Pre-filled on Live Chart, Greeks, Gamma &amp; Trump Tracker</div>
      </div>
      <div style="display:flex;gap:8px;">
        <input class="setting-input" id="default-ticker" type="text" value="{{ default_ticker }}" maxlength="12" placeholder="e.g. AAPL" style="width:120px;text-transform:uppercase;">
        <button class="btn" onclick="saveTicker()">Save</button>
      </div>
    </div>
  </div>

  <!-- Account -->
  <div class="section">
    <div class="section-title">Account</div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Change Email</div>
        <div class="setting-desc">Current: {{ email or 'Not set' }}</div>
      </div>
      <div style="display:flex;gap:8px;">
        <input class="setting-input" id="new-email" type="text" placeholder="New email" style="width:180px;">
        <button class="btn" onclick="saveEmail()">Save</button>
      </div>
    </div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Change Password</div>
        <div class="setting-desc">Must be at least 6 characters</div>
      </div>
      <button class="btn btn-muted" onclick="document.getElementById('pw-row').style.display=document.getElementById('pw-row').style.display==='none'?'block':'none'">Change</button>
    </div>
    <div id="pw-row" style="display:none;padding:16px 20px;border-bottom:1px solid var(--border);">
      <div style="display:flex;flex-direction:column;gap:10px;max-width:300px;">
        <input class="setting-input" id="pw-current" type="password" placeholder="Current password" style="width:100%;">
        <input class="setting-input" id="pw-new" type="password" placeholder="New password" style="width:100%;">
        <input class="setting-input" id="pw-confirm" type="password" placeholder="Confirm new password" style="width:100%;">
        <button class="btn" onclick="changePassword()">Update Password</button>
      </div>
    </div>
  </div>

  <!-- Danger zone -->
  <div class="section">
    <div class="section-title" style="color:var(--red);">Danger Zone</div>
    <div class="setting-row">
      <div>
        <div class="setting-label">Delete Account</div>
        <div class="setting-desc">Permanently delete your account and cancel any active subscription</div>
      </div>
      <button class="btn btn-danger" onclick="document.getElementById('delete-modal').classList.add('open')">Delete</button>
    </div>
  </div>

</div>

<!-- Delete confirmation modal -->
<div class="modal-bg" id="delete-modal">
  <div class="modal">
    <h2>Delete Account?</h2>
    <p>This will permanently delete your account, favorites, watchlist, and copy history. If you have an active subscription it will be cancelled. This cannot be undone.</p>
    <div class="modal-btns">
      <button class="btn btn-muted" onclick="document.getElementById('delete-modal').classList.remove('open')">Cancel</button>
      <button class="btn btn-danger" onclick="deleteAccount()">Yes, Delete</button>
    </div>
  </div>
</div>

<script>
""" + _THEME_JS + """

function flash(msg, ok) {
  const el = document.getElementById('flash');
  el.textContent = msg;
  el.className = 'flash ' + (ok ? 'ok' : 'err');
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 3500);
}

function setTheme(t) {
  document.documentElement.setAttribute('data-theme', t);
  document.querySelectorAll('.theme-opt').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  savePref('theme', t);
}

function savePref(key, val) {
  fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key, val})
  }).then(r => r.json()).then(d => {
    if (!d.ok) flash(d.error || 'Error saving', false);
  }).catch(() => flash('Error saving', false));
}

function saveTicker() {
  const val = document.getElementById('default-ticker').value.trim().toUpperCase();
  if (!val) return;
  document.getElementById('default-ticker').value = val;
  fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: 'default_ticker', val})
  }).then(r => r.json()).then(d => {
    if (d.ok) flash('Default ticker saved', true);
    else flash(d.error || 'Error', false);
  }).catch(() => flash('Error saving', false));
}

function saveEmail() {
  const val = document.getElementById('new-email').value.trim();
  if (!val) return;
  fetch('/api/settings/email', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({email: val})
  }).then(r => r.json()).then(d => {
    if (d.ok) { flash('Email updated', true); document.getElementById('new-email').value = ''; }
    else flash(d.error || 'Error', false);
  }).catch(() => flash('Error saving', false));
}

function changePassword() {
  const cur  = document.getElementById('pw-current').value;
  const nw   = document.getElementById('pw-new').value;
  const conf = document.getElementById('pw-confirm').value;
  if (!cur || !nw || !conf) { flash('Fill in all password fields', false); return; }
  if (nw !== conf) { flash('Passwords do not match', false); return; }
  if (nw.length < 6) { flash('Password must be at least 6 characters', false); return; }
  fetch('/api/settings/password', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({current: cur, new_pw: nw})
  }).then(r => r.json()).then(d => {
    if (d.ok) {
      flash('Password updated', true);
      ['pw-current','pw-new','pw-confirm'].forEach(id => document.getElementById(id).value = '');
      document.getElementById('pw-row').style.display = 'none';
    } else flash(d.error || 'Error', false);
  }).catch(() => flash('Error saving', false));
}

function deleteAccount() {
  fetch('/api/settings/delete', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.ok) window.location.href = '/';
      else flash(d.error || 'Error deleting account', false);
    }).catch(() => flash('Error', false));
}
</script>
</body>
</html>"""


@app.route("/settings")
@login_required
def settings_page():
    user_id = session["user_id"]
    row = _one("SELECT email, theme, default_ticker, copy_toast FROM users WHERE id=%s", (user_id,))
    return render_template_string(
        SETTINGS_HTML,
        current_user=current_user(),
        email=row["email"] if row else None,
        user_theme=row["theme"] if row and row["theme"] else "dark",
        default_ticker=row["default_ticker"] if row and row["default_ticker"] else "AAPL",
        copy_toast=row["copy_toast"] if row and row["copy_toast"] is not None else 1,
    )


@app.route("/api/settings", methods=["POST"])
@login_required
def api_settings():
    user_id = session["user_id"]
    data = request.get_json(silent=True) or {}
    key  = data.get("key", "")
    val  = data.get("val")
    allowed = {"theme", "default_ticker", "copy_toast"}
    if key not in allowed:
        return jsonify({"ok": False, "error": "Invalid setting"})
    if key == "default_ticker":
        val = str(val).upper().strip()[:12]
    elif key == "copy_toast":
        val = 1 if val else 0
    elif key == "theme":
        val = "light" if val == "light" else "dark"
    _run(f"UPDATE users SET {key}=%s WHERE id=%s", (val, user_id))
    return jsonify({"ok": True})


@app.route("/api/settings/email", methods=["POST"])
@login_required
def api_settings_email():
    user_id = session["user_id"]
    data  = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    if not email or "@" not in email:
        return jsonify({"ok": False, "error": "Invalid email"})
    existing = _one("SELECT id FROM users WHERE email=%s AND id!=%s", (email, user_id))
    if existing:
        return jsonify({"ok": False, "error": "Email already in use"})
    _run("UPDATE users SET email=%s WHERE id=%s", (email, user_id))
    return jsonify({"ok": True})


@app.route("/api/settings/password", methods=["POST"])
@login_required
def api_settings_password():
    import bcrypt
    user_id = session["user_id"]
    data    = request.get_json(silent=True) or {}
    current = data.get("current", "")
    new_pw  = data.get("new_pw", "")
    if not current or not new_pw or len(new_pw) < 6:
        return jsonify({"ok": False, "error": "Invalid input"})
    row = _one("SELECT pw_hash FROM users WHERE id=%s", (user_id,))
    if not row or not bcrypt.checkpw(current.encode(), row["pw_hash"].encode()):
        return jsonify({"ok": False, "error": "Current password is incorrect"})
    new_hash = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    _run("UPDATE users SET pw_hash=%s WHERE id=%s", (new_hash, user_id))
    return jsonify({"ok": True})


@app.route("/api/settings/delete", methods=["POST"])
@login_required
def api_settings_delete():
    user_id = session["user_id"]
    row = _one("SELECT stripe_subscription_id FROM users WHERE id=%s", (user_id,))
    if row and row["stripe_subscription_id"]:
        try:
            stripe.Subscription.cancel(row["stripe_subscription_id"])
        except Exception:
            pass
    # Delete all user data
    for table, col in [
        ("favorites", "user_id"),
        ("watchlist", "user_id"),
        ("copy_log", "user_id"),
        ("copy_history", "user_id"),
        ("indicator_ratings", "user_id"),
        ("request_votes", "user_id"),
    ]:
        try:
            _run(f"DELETE FROM {table} WHERE {col}=%s", (user_id,))
        except Exception:
            pass
    _run("DELETE FROM users WHERE id=%s", (user_id,))
    session.clear()
    return jsonify({"ok": True})


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML, current_user=current_user())


@app.route("/")
def index():
    return render_template_string(HOME_HTML, current_user=current_user())


# ── My Dashboard ──────────────────────────────────────────────────────────────

@app.route("/api/watchlist", methods=["GET"])
def api_watchlist_get():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "not_logged_in"}), 401
    rows = _q("SELECT symbol, added FROM watchlist WHERE user_id=%s ORDER BY added DESC", (user_id,))
    symbols = [r["symbol"] for r in rows]
    # Fetch live prices
    prices = {}
    if symbols:
        import yfinance as yf
        try:
            tickers = yf.Tickers(" ".join(symbols))
            for sym in symbols:
                try:
                    info = tickers.tickers[sym].fast_info
                    px   = float(info.last_price or 0)
                    prev = float(info.previous_close or px)
                    chg  = ((px - prev) / prev * 100) if prev else 0
                    prices[sym] = {"price": round(px, 4), "chg": round(chg, 2)}
                except Exception:
                    prices[sym] = {"price": None, "chg": None}
        except Exception:
            pass
    result = []
    for r in rows:
        sym = r["symbol"]
        result.append({
            "symbol": sym,
            "added":  r["added"].strftime("%Y-%m-%d") if r["added"] else "",
            **(prices.get(sym, {"price": None, "chg": None}))
        })
    return jsonify(result)


@app.route("/api/watchlist", methods=["POST"])
def api_watchlist_add():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "not_logged_in"}), 401
    data   = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "missing symbol"}), 400
    count = _one("SELECT COUNT(*) as c FROM watchlist WHERE user_id=%s", (user_id,))
    if count and count["c"] >= 20:
        return jsonify({"error": "Watchlist limit is 20 symbols"}), 400
    _run("INSERT INTO watchlist (user_id, symbol) VALUES (%s, %s) ON CONFLICT DO NOTHING", (user_id, symbol))
    return jsonify({"ok": True, "symbol": symbol})


@app.route("/api/watchlist/<symbol>", methods=["DELETE"])
def api_watchlist_delete(symbol):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "not_logged_in"}), 401
    _run("DELETE FROM watchlist WHERE user_id=%s AND symbol=%s", (user_id, symbol.upper()))
    return jsonify({"ok": True})


MY_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>My Dashboard · ChartEdge.trade</title>
  <style>
    :root[data-theme="dark"]  { --bg:#0d1117; --bg2:#161b22; --bg3:#21262d; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; --green:#3fb950; --red:#f85149; }
    :root[data-theme="light"] { --bg:#ffffff; --bg2:#f6f8fa; --bg3:#eaeef2; --border:#d0d7de; --text:#1f2328; --muted:#636c76; --accent:#0969da; --green:#1a7f37; --red:#cf222e; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }
    nav { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    .logo { color: var(--accent); font-size: 1.1rem; font-weight: bold; text-decoration: none; }
    """ + _NAV_CSS + """
    .page { max-width: 1000px; margin: 0 auto; padding: 32px 24px; }
    /* Profile header */
    .profile-header { display: flex; align-items: center; gap: 20px; margin-bottom: 28px; padding: 24px; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; flex-wrap: wrap; }
    .avatar { width: 68px; height: 68px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 1.8rem; font-weight: 700; color: #fff; flex-shrink: 0; overflow: hidden; }
    .avatar img { width: 68px; height: 68px; border-radius: 50%; object-fit: cover; }
    .profile-info { flex: 1; min-width: 0; }
    .profile-name { font-size: 1.3rem; font-weight: 700; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .plan-pill { padding: 3px 10px; border-radius: 20px; font-size: .72rem; font-weight: 700; text-transform: uppercase; }
    .pencil-btn { background: none; border: none; cursor: pointer; font-size: .85rem; color: var(--muted); padding: 2px 5px; border-radius: 4px; }
    .pencil-btn:hover { color: var(--accent); background: var(--bg3); }
    .profile-meta { color: var(--muted); font-size: .83rem; margin-top: 5px; }
    /* Edit form */
    .edit-section { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 22px; margin-bottom: 20px; }
    .edit-section-title { font-size: .75rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 14px; }
    .edit-form { display: flex; flex-direction: column; gap: 12px; }
    .edit-row { display: flex; flex-direction: column; gap: 5px; }
    .edit-row label { font-size: .75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
    .edit-row input[type=text], .edit-row input[type=file] { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); font-size: .9rem; width: 100%; }
    .btn-save { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 9px 22px; font-size: .9rem; font-weight: 600; cursor: pointer; align-self: flex-start; }
    .error-msg { background: #2d1515; border: 1px solid var(--red); color: var(--red); border-radius: 6px; padding: 10px 14px; font-size: .85rem; margin-bottom: 12px; }
    /* Grid */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width:700px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 22px; }
    .card-title { font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 14px; font-weight: 700; }
    .full-card { grid-column: 1 / -1; }
    /* Stats */
    .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .stat-box { background: var(--bg3); border-radius: 8px; padding: 14px; text-align: center; }
    .stat-num { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
    .stat-lbl { font-size: .7rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
    /* Badges */
    .badges-wrap { display: flex; flex-wrap: wrap; gap: 8px; }
    .badge-card { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 8px; border: 1px solid var(--border); font-size: .8rem; font-weight: 600; }
    .no-badges { color: var(--muted); font-size: .85rem; }
    /* Plan card */
    .plan-name { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
    .plan-detail { font-size: .82rem; color: var(--muted); margin-top: 6px; }
    .upgrade-btn { display: inline-block; margin-top: 12px; background: var(--accent); color: #fff; border-radius: 6px; padding: 8px 16px; font-size: .83rem; font-weight: 600; text-decoration: none; }
    /* Usage bar */
    .usage-label { display: flex; justify-content: space-between; font-size: .8rem; color: var(--muted); margin-bottom: 6px; }
    .usage-bar { height: 8px; border-radius: 4px; background: var(--bg3); overflow: hidden; }
    .usage-fill { height: 100%; border-radius: 4px; background: var(--accent); transition: width .4s; }
    .usage-fill.warn { background: #d29922; }
    .usage-fill.full { background: var(--red); }
    /* List items */
    .list-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: .85rem; }
    .list-item:last-child { border-bottom: none; }
    .item-main { font-weight: 600; }
    .item-sub { color: var(--muted); font-size: .75rem; }
    .fav-link { color: var(--accent); text-decoration: none; font-weight: 600; }
    .fav-link:hover { text-decoration: underline; }
    .remove-btn { background: none; border: none; color: var(--red); cursor: pointer; font-size: .8rem; padding: 2px 6px; }
    /* Watchlist */
    .wl-chg-pos { color: var(--green); }
    .wl-chg-neg { color: var(--red); }
    .wl-add-row { display: flex; gap: 8px; margin-top: 14px; }
    .wl-add-row input { flex: 1; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; color: var(--text); font-size: .85rem; font-family: monospace; text-transform: uppercase; outline: none; }
    .wl-add-row input:focus { border-color: var(--accent); }
    .wl-add-row button { background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 8px 16px; font-size: .85rem; font-weight: 600; cursor: pointer; }
    .empty { color: var(--muted); font-size: .85rem; text-align: center; padding: 20px 0; }
  </style>
</head>
<body>
<nav>
  <a class="logo" href="/"><span style="color:var(--text)">Chart</span><span style="color:#58a6ff">Edge</span><span style="color:var(--muted);font-weight:500;">.trade</span></a>
  <button class="hamburger" onclick="toggleMobileNav(event)">☰</button>
  <div class="nav-links" id="mobile-nav">""" + _NAV_LINKS + """</div>
</nav>
<div class="page">

  <!-- Profile header -->
  <div class="profile-header">
    <div class="avatar">{% if profile_pic %}<img src="{{ profile_pic }}" alt="avatar">{% else %}{{ username[0].upper() }}{% endif %}</div>
    <div class="profile-info">
      <div class="profile-name">
        {{ username }}
        <span class="plan-pill" style="background:{{ {'free':'#21262d','basic':'#0d1a2d','pro':'#2a2000'}[plan] }};color:{{ {'free':'#8b949e','basic':'#58a6ff','pro':'#e3b341'}[plan] }};">{{ plan.upper() }}</span>
        <span class="plan-pill" style="background:#21262d;color:#e6edf3;">#{{ signup_rank }}</span>
        {% if signup_tier %}<span class="plan-pill" style="background:{{ signup_tier.bg }};color:{{ signup_tier.color }};">{{ signup_tier.icon }} {{ signup_tier.title }}</span>{% endif %}
        <button class="pencil-btn" onclick="document.getElementById('edit-section').style.display=document.getElementById('edit-section').style.display==='none'?'block':'none'" title="Edit profile">✏️</button>
      </div>
      <div class="profile-meta">Member since {{ member_since }}{% if plan_expires %} &nbsp;·&nbsp; Renews {{ plan_expires }}{% endif %}</div>
    </div>
  </div>

  <!-- Edit profile (hidden by default) -->
  <div id="edit-section" style="display:{% if profile_error %}block{% else %}none{% endif %}">
    <div class="edit-section">
      <div class="edit-section-title">Edit Profile</div>
      {% if profile_error %}<div class="error-msg">{{ profile_error }}</div>{% endif %}
      <form class="edit-form" method="POST" action="/profile/update" enctype="multipart/form-data">
        <div class="edit-row">
          <label>Username</label>
          <input type="text" name="username" value="{{ username }}" maxlength="30" placeholder="New username">
        </div>
        <div class="edit-row">
          <label>Email {% if not email %}<span style="color:#f0883e;text-transform:none;font-weight:500;letter-spacing:0;">— needed for password recovery</span>{% endif %}</label>
          <input type="email" name="email" value="{{ email or '' }}" maxlength="120" placeholder="you@example.com">
        </div>
        <div class="edit-row">
          <label>Profile Picture</label>
          <input type="file" name="profile_pic" accept="image/*">
        </div>
        <button class="btn-save" type="submit">Save Changes</button>
      </form>
    </div>
  </div>

  <div class="grid">

    <!-- Stats -->
    <div class="card">
      <div class="card-title">Stats</div>
      <div class="stats-grid">
        <div class="stat-box"><div class="stat-num">{{ stats.copies }}</div><div class="stat-lbl">Copied</div></div>
        <div class="stat-box"><div class="stat-num">{{ stats.favorites }}</div><div class="stat-lbl">Favorites</div></div>
        <div class="stat-box"><div class="stat-num">{{ stats.referrals }}</div><div class="stat-lbl">Referrals</div></div>
      </div>
    </div>

    <!-- Copy usage today -->
    <div class="card">
      <div class="card-title">Copies Today</div>
      <div class="usage-label">
        <span id="copies-used-lbl">—</span>
        <span id="copies-limit-lbl">—</span>
      </div>
      <div class="usage-bar"><div class="usage-fill" id="copies-bar" style="width:0%"></div></div>
      <div style="margin-top:12px;font-size:.82rem;color:var(--muted);">
        All-time total: <strong id="copies-total" style="color:var(--text)">—</strong>
      </div>
    </div>

    <!-- Badges (full width) -->
    <div class="card full-card">
      <div class="card-title">Badges</div>
      {% if badges %}
      <div class="badges-wrap">
        {% for b in badges %}
        <div class="badge-card" style="background:{{ b.bg }};border-color:{{ b.color }}40;">
          <span>{{ b.icon }}</span>
          <span style="color:{{ b.color }};">{{ b.label }}</span>
        </div>
        {% endfor %}
      </div>
      {% else %}
      <div class="no-badges">No badges yet — start copying indicators to earn some!</div>
      {% endif %}
    </div>

    <!-- Recent copies -->
    <div class="card">
      <div class="card-title">Recent Copies</div>
      <div id="hist-list"><div class="empty">Loading…</div></div>
    </div>

    <!-- Favorites -->
    <div class="card">
      <div class="card-title">Saved Favorites</div>
      <div id="fav-list"><div class="empty">Loading…</div></div>
    </div>

    <!-- Watchlist (full width) -->
    <div class="card full-card">
      <div class="card-title">Watchlist <span style="color:var(--muted);font-size:.72rem;font-weight:400;">(up to 20)</span></div>
      <div id="wl-list"><div class="empty">Loading…</div></div>
      <div class="wl-add-row">
        <input type="text" id="wl-input" placeholder="Add ticker, e.g. AAPL" maxlength="12">
        <button onclick="wlAdd()">Add</button>
      </div>
      <div id="wl-error" style="color:var(--red);font-size:.78rem;margin-top:6px;display:none;"></div>
    </div>

    <!-- Plan status -->
    <div class="card">
      <div class="card-title">Plan</div>
      <div class="plan-name">{{ plan | upper }}</div>
      {% if plan == 'free' %}
        <div class="plan-detail">Unlock all tools with Basic or Pro.</div>
        <a href="/pricing" class="upgrade-btn">Upgrade Plan</a>
      {% elif plan == 'basic' %}
        <div class="plan-detail">Upgrade to Pro for Options Flow, Insider Trading &amp; more.</div>
        <a href="/pricing" class="upgrade-btn">Go Pro</a>
      {% else %}
        <div class="plan-detail" style="color:var(--green);margin-top:6px;">✓ Full access to all features</div>
      {% endif %}
    </div>

    <!-- Account links -->
    <div class="card">
      <div class="card-title">Account</div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <a href="/settings" style="color:var(--accent);font-size:.88rem;text-decoration:none;">⚙️ Settings</a>
        <a href="/billing" style="color:var(--accent);font-size:.88rem;text-decoration:none;">💳 Manage Billing</a>
        <a href="/refer"   style="color:var(--accent);font-size:.88rem;text-decoration:none;">🤝 Refer a Friend</a>
        <a href="/redeem"  style="color:var(--accent);font-size:.88rem;text-decoration:none;">🎁 Redeem Code</a>
        <a href="/favorites" style="color:var(--accent);font-size:.88rem;text-decoration:none;">♥ All Favorites</a>
        <a href="/logout"  style="color:var(--muted);font-size:.88rem;text-decoration:none;">→ Logout</a>
      </div>
    </div>

  </div>
</div>
<script>
""" + _THEME_JS + """

const PLAN      = {{ plan | tojson }};
const PLAN_LIMS = { free: 3, basic: 8, pro: null };
const limit     = PLAN_LIMS[PLAN];

// ── Copy usage ─────────────────────────────────────────────────────────────
fetch('/api/me/stats')
  .then(r => r.json())
  .then(d => {
    const used  = d.copies_today || 0;
    const total = d.copies_total || 0;
    document.getElementById('copies-total').textContent = total.toLocaleString();
    if (PLAN === 'pro') {
      document.getElementById('copies-used-lbl').textContent   = used + ' used';
      document.getElementById('copies-limit-lbl').textContent  = 'Unlimited';
      const bar = document.getElementById('copies-bar');
      bar.style.width = '100%';
      bar.style.background = 'var(--green)';
    } else {
      const lim = limit || 3;
      const pct = Math.min(used / lim * 100, 100);
      document.getElementById('copies-used-lbl').textContent  = used + ' used';
      document.getElementById('copies-limit-lbl').textContent = lim + ' daily limit';
      const bar = document.getElementById('copies-bar');
      bar.style.width = pct + '%';
      if (pct >= 100) bar.classList.add('full');
      else if (pct >= 70) bar.classList.add('warn');
    }
  }).catch(() => {});

// ── Copy history ───────────────────────────────────────────────────────────
fetch('/api/me/history')
  .then(r => r.json())
  .then(rows => {
    const el = document.getElementById('hist-list');
    if (!rows.length) { el.innerHTML = '<div class="empty">No copies yet</div>'; return; }
    el.innerHTML = rows.map(r =>
      '<div class="hist-item">' +
        '<span class="hist-sym">' + r.indicator + '</span>' +
        '<span class="hist-ts">'  + r.ts + '</span>' +
      '</div>'
    ).join('');
  }).catch(() => { document.getElementById('hist-list').innerHTML = '<div class="empty">Could not load</div>'; });

// ── Favorites ──────────────────────────────────────────────────────────────
fetch('/api/favorites')
  .then(r => r.json())
  .then(rows => {
    const el = document.getElementById('fav-list');
    if (!rows || !rows.length) { el.innerHTML = '<div class="empty">No favorites saved</div>'; return; }
    el.innerHTML = rows.slice(0, 10).map(r =>
      '<div class="fav-item">' +
        '<a class="fav-link" href="/indicators?kind=' + encodeURIComponent(r.indicator_key) + '">' + r.indicator_key + '</a>' +
        '<button class="remove-btn" onclick="removeFav(' + JSON.stringify(r.indicator_key) + ', this)">✕</button>' +
      '</div>'
    ).join('');
  }).catch(() => { document.getElementById('fav-list').innerHTML = '<div class="empty">Could not load</div>'; });

function removeFav(key, btn) {
  fetch('/api/favorite/' + encodeURIComponent(key), {method: 'DELETE'})
    .then(r => r.json())
    .then(() => { btn.closest('.fav-item').remove(); })
    .catch(() => {});
}

// ── Watchlist ──────────────────────────────────────────────────────────────
function loadWatchlist() {
  fetch('/api/watchlist')
    .then(r => r.json())
    .then(rows => {
      const el = document.getElementById('wl-list');
      if (!rows.length) { el.innerHTML = '<div class="empty">No symbols yet — add one below</div>'; return; }
      el.innerHTML = rows.map(r => {
        const chgClass = (r.chg > 0) ? 'wl-chg-pos' : (r.chg < 0) ? 'wl-chg-neg' : '';
        const chgStr   = r.chg !== null ? ((r.chg >= 0 ? '+' : '') + r.chg.toFixed(2) + '%') : '—';
        const priceStr = r.price !== null ? (r.price >= 1000 ? r.price.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : r.price.toFixed(r.price < 1 ? 4 : 2)) : '—';
        return '<div class="wl-row">' +
          '<span class="wl-sym">' + r.symbol + '</span>' +
          '<span style="display:flex;gap:18px;align-items:center;">' +
            '<span class="wl-price">' + priceStr + '</span>' +
            '<span class="' + chgClass + '" style="min-width:60px;text-align:right;">' + chgStr + '</span>' +
            '<button class="remove-btn" onclick="wlRemove(' + JSON.stringify(r.symbol) + ', this)">✕</button>' +
          '</span>' +
        '</div>';
      }).join('');
    }).catch(() => { document.getElementById('wl-list').innerHTML = '<div class="empty">Could not load watchlist</div>'; });
}

function wlAdd() {
  const inp = document.getElementById('wl-input');
  const sym = inp.value.trim().toUpperCase();
  const errEl = document.getElementById('wl-error');
  if (!sym) return;
  errEl.style.display = 'none';
  fetch('/api/watchlist', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({symbol: sym})
  }).then(r => r.json()).then(d => {
    if (d.error) { errEl.textContent = d.error; errEl.style.display = 'block'; return; }
    inp.value = '';
    loadWatchlist();
  }).catch(() => {});
}

document.getElementById('wl-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') wlAdd();
});

function wlRemove(sym, btn) {
  fetch('/api/watchlist/' + encodeURIComponent(sym), {method: 'DELETE'})
    .then(r => r.json())
    .then(() => { btn.closest('.wl-row').remove(); })
    .catch(() => {});
}

loadWatchlist();
</script>
</body>
</html>"""


@app.route("/api/me/stats")
def api_me_stats():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "not_logged_in"}), 401
    today        = date.today().isoformat()
    copies_today = _copies_used_today(user_id)
    row_total    = _one("SELECT COALESCE(SUM(count),0) as total FROM copy_log WHERE user_id=%s", (user_id,))
    copies_total = int(row_total["total"]) if row_total else 0
    return jsonify({"copies_today": copies_today, "copies_total": copies_total})


@app.route("/api/me/history")
def api_me_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "not_logged_in"}), 401
    rows = _q(
        "SELECT indicator, ts FROM copy_history WHERE user_id=%s ORDER BY ts DESC LIMIT 15",
        (user_id,)
    )
    result = []
    for r in rows:
        ts = r["ts"]
        result.append({
            "indicator": r["indicator"],
            "ts": ts.strftime("%b %d, %H:%M") if ts else ""
        })
    return jsonify(result)


@app.route("/me")
@login_required
def me():
    from datetime import datetime, timezone
    user_id  = session["user_id"]
    username = session.get("username", "")
    plan     = _get_user_plan(user_id)
    row      = _one("SELECT created, plan_expires, referral_code, profile_pic, email FROM users WHERE id=%s", (user_id,))

    # Plan expiry
    expires = None
    if row and row["plan_expires"] and plan != "free":
        expires = row["plan_expires"].strftime("%b %d, %Y")

    # Stats
    copies    = _scalar("SELECT COALESCE(SUM(count),0) FROM copy_log WHERE user_id=%s", (user_id,)) or 0
    fav_count = _scalar("SELECT COUNT(*) FROM favorites WHERE user_id=%s", (user_id,)) or 0
    ref_code  = (row["referral_code"] or "") if row else ""
    referrals = _scalar("SELECT COUNT(*) FROM users WHERE referred_by=%s", (ref_code,)) or 0 if ref_code else 0

    # Member duration
    now     = datetime.now(timezone.utc)
    created = row["created"] if row else None
    if created:
        created_aware = created.replace(tzinfo=timezone.utc) if created.tzinfo is None else created
        delta = now - created_aware
        days  = delta.days
        member_since = created_aware.strftime("%b %d, %Y")
        if days < 7:
            duration_label = "New Member"; duration_color = "#8b949e"
        elif days < 30:
            duration_label = f"{days} Days"; duration_color = "#58a6ff"
        elif days < 90:
            m = days // 30; duration_label = f"{m} Month{'s' if m>1 else ''}"; duration_color = "#3fb950"
        elif days < 365:
            m = days // 30; duration_label = f"{m} Months"; duration_color = "#e3b341"
        elif days < 730:
            duration_label = "1 Year"; duration_color = "#f0883e"
        else:
            y = days // 365; duration_label = f"{y} Years"; duration_color = "#bc8cff"
    else:
        days = 0; member_since = "Unknown"; duration_label = "Member"; duration_color = "#8b949e"

    # Signup rank — actual ordinal among all users (handles deletions)
    signup_rank = _scalar("SELECT COUNT(*) FROM users WHERE id <= %s", (user_id,)) or user_id
    if signup_rank <= 10:
        signup_tier = {"title": "Founding Member", "color": "#e3b341", "bg": "#2a2000", "icon": "👑"}
    elif signup_rank <= 100:
        signup_tier = {"title": "Pioneer",        "color": "#bc8cff", "bg": "#1a0d2d", "icon": "🚀"}
    elif signup_rank <= 1000:
        signup_tier = {"title": "Early Adopter",  "color": "#58a6ff", "bg": "#0d1a2d", "icon": "⭐"}
    else:
        signup_tier = None

    # Badges
    badges = []
    badges.append({"icon":"👤","label":f"User #{signup_rank}","color":"#e6edf3","bg":"#21262d"})
    if signup_tier:
        badges.append({"icon": signup_tier["icon"], "label": signup_tier["title"],
                       "color": signup_tier["color"], "bg": signup_tier["bg"]})
    plan_colors = {"free":("#8b949e","#21262d"),"basic":("#58a6ff","#0d1a2d"),"pro":("#e3b341","#2a2000")}
    pc, pbg = plan_colors.get(plan, plan_colors["free"])
    badges.append({"icon":"💎" if plan=="pro" else ("⭐" if plan=="basic" else "🆓"),
                   "label":plan.capitalize()+" Plan","color":pc,"bg":pbg})
    if days >= 7:
        dur_icons = {7:"🌿",30:"🏅",90:"🥈",365:"🥇",730:"👑"}
        dur_icon = "🌿"
        for t, ic in sorted(dur_icons.items()):
            if days >= t: dur_icon = ic
        badges.append({"icon":dur_icon,"label":duration_label+" Member","color":duration_color,"bg":"#0d1117"})
    if copies >= 1:  badges.append({"icon":"📋","label":"First Copy","color":"#3fb950","bg":"#1a2d1a"})
    if copies >= 10: badges.append({"icon":"⚡","label":"Power User","color":"#e3b341","bg":"#2a2000"})
    if copies >= 50: badges.append({"icon":"🚀","label":"Super Trader","color":"#f0883e","bg":"#2d1a0d"})
    if fav_count >= 3:  badges.append({"icon":"❤️","label":"Collector","color":"#f85149","bg":"#2d1515"})
    if fav_count >= 10: badges.append({"icon":"💫","label":"Indicator Fan","color":"#bc8cff","bg":"#1a0d2d"})
    if referrals >= 1: badges.append({"icon":"🤝","label":"Referrer","color":"#3fb950","bg":"#1a2d1a"})
    if referrals >= 3: badges.append({"icon":"🌟","label":"Ambassador","color":"#e3b341","bg":"#2a2000"})
    if 0 <= days <= 30: badges.append({"icon":"🎉","label":"Early Adopter","color":"#bc8cff","bg":"#1a0d2d"})

    profile_pic   = row["profile_pic"] if row else None
    email         = (row["email"] if row else "") or ""
    profile_error = session.pop("profile_error", None)
    stats = {"copies": copies, "favorites": fav_count, "referrals": referrals}

    return render_template_string(
        MY_DASHBOARD_HTML,
        current_user=username,
        username=username,
        plan=plan,
        plan_expires=expires,
        member_since=member_since,
        badges=badges,
        stats=stats,
        profile_pic=profile_pic,
        profile_error=profile_error,
        signup_rank=signup_rank,
        signup_tier=signup_tier,
        email=email,
    )


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
