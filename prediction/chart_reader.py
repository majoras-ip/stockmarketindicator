"""
prediction/chart_reader.py — Read a chart screenshot with local OCR (free).

The screenshot is only an *input method*: we OCR the image to pull the ticker
symbol (and, if visible, the timeframe), then the caller fetches real OHLCV for
that symbol and runs the normal expected-move math. No prices or direction are
read off the pixels.

This uses Tesseract locally — $0 per screenshot, no API key. Requires the
`tesseract` binary on the host (see Dockerfile / nixpacks.toml) plus the
`pytesseract` + `Pillow` Python packages.
"""

from __future__ import annotations

import io
import re

_VALID_INTERVALS = {"1m", "5m", "15m", "1h", "1d"}

# exchange-prefixed tickers are the strongest signal (e.g. "NASDAQ:AAPL")
_EXCHANGE_RE = re.compile(
    r"(?:NASDAQ|NYSE|AMEX|ARCA|BATS|CBOE|OTC|NYSEARCA)[:\s]+([A-Z]{1,6})"
)
_CAND_RE = re.compile(r"\b[A-Z]{2,5}\b")

# uppercase tokens that show up on charts but are never the ticker
_STOP = {
    "RSI", "MACD", "EMA", "SMA", "MA", "VOL", "BB", "ATR", "ADX", "OBV", "VWAP",
    "OHLC", "AVG", "STD", "BUY", "SELL", "LOG", "AUTO", "USD", "EUR", "GBP", "JPY",
    "HIGH", "LOW", "OPEN", "CLOSE", "VOLUME", "PRICE", "CHART", "AM", "PM", "UTC",
    "EST", "EDT", "GMT", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
    "SEP", "OCT", "NOV", "DEC", "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN",
    "AND", "THE", "FOR", "NEW", "ALL", "ADD", "COMP",
}


def _detect_interval(text: str) -> str:
    """Best-effort timeframe from OCR text. Falls back to '1d' (most robust for
    an expected-move estimate) when nothing legible is found."""
    t = text.lower()
    # ordered so more specific tokens win
    for pat, code in (
        (r"\b15\s?m\b|\b15\s?min", "15m"),
        (r"\b5\s?m\b|\b5\s?min", "5m"),
        (r"\b1\s?m\b|\b1\s?min", "1m"),
        (r"\b1\s?h\b|\b60\s?min|\bhourly", "1h"),
        (r"\b1\s?d\b|\bdaily|\b1\s?day", "1d"),
    ):
        if re.search(pat, t):
            return code
    return "1d"


def _valid_ticker(sym: str) -> bool:
    """True if yfinance returns any recent data for the symbol."""
    try:
        import yfinance as yf
        df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=True)
        return df is not None and len(df) > 0
    except Exception:
        return False


def read_chart(image_bytes: bytes, media_type: str = "image/png") -> dict:
    """
    OCR an uploaded chart image and return
    {is_chart, ticker, interval, timeframe_detected, confidence}.

    `ticker` is "" if nothing legible/valid was found. `interval` is normalised
    to an app code (defaults to '1d').
    """
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L")   # grayscale
    # upscale small images — OCR is much more reliable above ~1000px wide
    if img.width < 1000:
        scale = 1000 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    # image_to_data gives per-word boxes so we can prefer top-of-chart tokens
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = data["text"]
    tops = data["top"]
    full_text = " ".join(w for w in words if w.strip())

    if not full_text.strip():
        return {"is_chart": False, "ticker": "", "interval": "1d",
                "timeframe_detected": "unknown", "confidence": 0.0}

    # 1) exchange-prefixed ticker wins outright
    ticker = ""
    m = _EXCHANGE_RE.search(full_text)
    if m and _valid_ticker(m.group(1)):
        ticker = m.group(1)

    # 2) otherwise gather candidates, prefer ones nearer the top, validate
    if not ticker:
        cands = []  # (top_y, symbol)
        for w, y in zip(words, tops):
            for c in _CAND_RE.findall(w.upper()):
                if c not in _STOP:
                    cands.append((y, c))
        cands.sort(key=lambda p: p[0])          # topmost first
        seen = set()
        for _, c in cands:
            if c in seen:
                continue
            seen.add(c)
            if _valid_ticker(c):
                ticker = c
                break
            if len(seen) >= 6:                   # cap validation lookups
                break

    interval = _detect_interval(full_text)
    return {
        "is_chart": bool(ticker) or len(full_text) > 20,
        "ticker": ticker,
        "interval": interval if interval in _VALID_INTERVALS else "1d",
        "timeframe_detected": interval,
        "confidence": 0.9 if ticker else 0.0,
    }
