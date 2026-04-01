"""
server.py — FastAPI bridge between the Python model and TradingView.

How it works
────────────
1. Server loads saved models on startup.
2. TradingView alert fires → POST /webhook  (sends ticker + bar data)
3. Server runs the ML prediction pipeline.
4. Result is stored in memory and visible at GET /predict and /dashboard.

Run with:
    python3 server.py

Then expose publicly with ngrok:
    ngrok http 8000

The ngrok URL (e.g. https://abc123.ngrok.io) is what you paste into
TradingView's alert webhook field.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add project root to path so imports work when run from any directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from prediction.predictor import Predictor, classify_volatility
from utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(title="Stock Volatility API", version="1.0")

# ── Global state ──────────────────────────────────────────────────────────────

predictor: Optional[Predictor] = None
_latest: dict = {}          # last prediction result
_history: list[dict] = []   # rolling history (last 100)
_lock = threading.Lock()


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global predictor
    log.info("Loading models from %s …", config.MODEL_DIR)
    try:
        predictor = Predictor(config.MODEL_DIR)
        predictor.load()
        log.info("Models loaded — server ready.")
    except FileNotFoundError as exc:
        log.error(
            "Model files not found: %s\n"
            "Train the model first with: python3 main.py → option 2",
            exc,
        )
        # Server still starts so health check works; /predict will return 503.


# ── Models ────────────────────────────────────────────────────────────────────

class WebhookPayload(BaseModel):
    """
    TradingView sends a JSON body when an alert fires.
    You configure the message format in the TradingView alert dialog.

    Recommended TradingView alert message:
        {"ticker": "{{ticker}}", "close": {{close}}, "volume": {{volume}}}
    """
    ticker: str = config.TICKER
    close:  Optional[float] = None
    volume: Optional[float] = None
    note:   Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Simple liveness check."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None and predictor._loaded,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/predict")
def predict(ticker: str = config.TICKER):
    """
    Run a prediction for *ticker* and return JSON.

    Example:
        GET /predict?ticker=SPY
    """
    if predictor is None or not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python3 main.py → option 2",
        )

    try:
        result = predictor.predict(ticker)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        _store(result)
        return JSONResponse(result)
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/webhook")
async def webhook(request: Request):
    """
    TradingView alert endpoint.

    Paste this URL into TradingView → Create Alert → Webhook URL:
        https://<your-ngrok-url>/webhook

    TradingView alert message (JSON format):
        {"ticker": "{{ticker}}", "close": {{close}}, "volume": {{volume}}}
    """
    if predictor is None or not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        body = await request.json()
    except Exception:
        body = {}

    ticker = body.get("ticker", config.TICKER).replace("NASDAQ:", "").replace("NYSE:", "")

    log.info("Webhook received — ticker=%s  body=%s", ticker, body)

    try:
        result = predictor.predict(ticker)
        result["timestamp"]    = datetime.now(timezone.utc).isoformat()
        result["trigger_data"] = body
        _store(result)
        log.info(
            "Prediction complete — %s  vol=%.6f  level=%s",
            ticker, result["predicted_volatility"], result["level"],
        )
        return JSONResponse(result)
    except Exception as exc:
        log.exception("Webhook prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/latest")
def latest():
    """Return the most recent prediction (used by the dashboard auto-refresh)."""
    with _lock:
        if not _latest:
            return JSONResponse({"message": "No predictions yet."})
        return JSONResponse(_latest)


@app.get("/history")
def history():
    """Return the last 100 predictions."""
    with _lock:
        return JSONResponse(_history)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """
    Simple HTML dashboard — auto-refreshes every 30 seconds.
    Open in browser: http://localhost:8000/dashboard
    """
    return HTMLResponse(_dashboard_html())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _store(result: dict) -> None:
    with _lock:
        _latest.update(result)
        _history.append(dict(result))
        if len(_history) > 100:
            _history.pop(0)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

def _dashboard_html() -> str:
    level_colors = {
        "Low":     "#2ecc71",
        "Medium":  "#f1c40f",
        "High":    "#e67e22",
        "Extreme": "#e74c3c",
    }

    with _lock:
        data = dict(_latest)

    if not data:
        body = "<p style='color:#aaa'>No predictions yet. Hit <code>/predict?ticker=SPY</code> or wait for a TradingView alert.</p>"
    else:
        vol   = data.get("predicted_volatility", 0)
        level = data.get("level", "—")
        color = level_colors.get(level, "#aaa")
        ts    = data.get("timestamp", "—")
        tick  = data.get("ticker", "—")
        conf  = data.get("confidence", "—")
        desc  = data.get("description", "")

        rows = "".join(
            f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
            for k, v in [
                ("Ticker",    tick),
                ("Timestamp", ts),
                ("Volatility", f"{vol:.8f}"),
                ("Level",     f"<span style='color:{color};font-weight:bold'>{level}</span>"),
                ("Confidence", conf),
                ("Note",      desc),
            ]
        )
        body = f"<table>{rows}</table>"

        # History table
        with _lock:
            hist = list(reversed(_history[-20:]))
        hrows = "".join(
            "<tr>"
            f"<td>{h.get('timestamp','')[:19]}</td>"
            f"<td>{h.get('ticker','')}</td>"
            f"<td>{h.get('predicted_volatility',0):.8f}</td>"
            "<td style='color:" + level_colors.get(h.get('level',''), '#aaa') + "'>"
            f"{h.get('level','')}</td>"
            "</tr>"
            for h in hist
        )
        body += f"""
        <h3 style='margin-top:2rem'>Recent predictions</h3>
        <table>
          <tr><th>Time (UTC)</th><th>Ticker</th><th>Volatility</th><th>Level</th></tr>
          {hrows}
        </table>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <title>Volatility Dashboard</title>
  <meta http-equiv="refresh" content="30">
  <style>
    body {{ font-family: monospace; background: #1a1a2e; color: #eee;
           max-width: 800px; margin: 40px auto; padding: 0 20px; }}
    h1   {{ color: #e94560; }}
    h3   {{ color: #aaa; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    td, th {{ padding: 8px 12px; border: 1px solid #333; text-align: left; }}
    th   {{ background: #16213e; color: #aaa; font-weight: normal; }}
    code {{ background: #16213e; padding: 2px 6px; border-radius: 3px; }}
    .refresh {{ color: #555; font-size: 0.85rem; margin-top: 2rem; }}
  </style>
</head>
<body>
  <h1>Stock Volatility Prediction</h1>
  <p>Auto-refreshes every 30 seconds &nbsp;|&nbsp;
     <a href="/predict?ticker={data.get('ticker', config.TICKER)}" style="color:#e94560">
       force refresh
     </a>
  </p>
  {body}
  <p class="refresh">Server time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║         Stock Volatility — API Server                ║
╠══════════════════════════════════════════════════════╣
║  Local dashboard : http://localhost:8000/dashboard   ║
║  Predict endpoint: http://localhost:8000/predict     ║
║  Webhook endpoint: http://localhost:8000/webhook     ║
║  Health check    : http://localhost:8000/health      ║
╠══════════════════════════════════════════════════════╣
║  To expose publicly:                                 ║
║    ngrok http 8000                                   ║
║  Paste the ngrok URL into TradingView alert webhook  ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
