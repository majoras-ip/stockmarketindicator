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

_VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}

# exchange-prefixed tickers are the strongest signal (e.g. "NASDAQ:AAPL").
# Capture up to 8 chars because OCR often glues the adjacent timeframe onto the
# symbol ("NASDAQ:AAPL1D" -> "AAPLID"); _resolve() trims it back to a real ticker.
_EXCHANGE_RE = re.compile(
    r"(?:NASDAQ|NYSE|AMEX|ARCA|BATS|CBOE|OTC|NYSEARCA)[:\s]*([A-Z]{1,8})"
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
    """Best-effort timeframe from OCR text.

    A charting toolbar usually lists *every* timeframe (1m 5m 15m 1h D W M), and
    OCR can't see which one is highlighted — so if several are present we can't
    tell the active one and fall back to '1d' (the most robust horizon for an
    expected-move estimate). We only trust a timeframe when exactly one is
    visible (e.g. a cropped chart).

    Case matters: minutes render lowercase ('1m','30m') but day/week/month render
    uppercase ('D','W','M'), so lowercasing would collide 1-month with 1-minute.
    Patterns are matched against the original-case text accordingly."""
    checks = [
        (r"\b30\s?m(in)?\b", "30m", re.I),
        (r"\b15\s?m(in)?\b", "15m", re.I),
        (r"\b5\s?m(in)?\b", "5m", re.I),
        (r"\b1\s?m(in)?\b", "1m", 0),                       # lowercase m → minute
        (r"\b(1\s?h|60\s?m(in)?|hourly)\b", "1h", re.I),
        (r"\b(1?\s?D|daily|1\s?day)\b", "1d", 0),           # uppercase D → day
        (r"\b(1?\s?W|weekly|1\s?wk)\b", "1wk", 0),          # uppercase W → week
        (r"(\b1?\s?M\b|Monthly|\b1\s?mo\b)", "1mo", 0),     # uppercase M → month
    ]
    found = []
    for pat, code, flags in checks:
        if re.search(pat, text, flags) and code not in found:
            found.append(code)
    return found[0] if len(found) == 1 else "1d"


def _valid_ticker(sym: str) -> bool:
    """True if yfinance returns any recent data for the symbol."""
    try:
        import yfinance as yf
        df = yf.download(sym, period="5d", interval="1d", progress=False, auto_adjust=True)
        return df is not None and len(df) > 0
    except Exception:
        return False


def _resolve(token: str) -> str:
    """Return the longest prefix of `token` (len 2–5) that is a real ticker, or
    "". Handles OCR gluing the timeframe onto the symbol ("AAPLID" -> "AAPL")."""
    token = token.upper()
    for n in range(min(len(token), 5), 1, -1):
        cand = token[:n]
        if cand in _STOP:
            continue
        if _valid_ticker(cand):
            return cand
    return ""


def read_chart(image_bytes: bytes, media_type: str = "image/png") -> dict:
    """
    OCR an uploaded chart image and return
    {is_chart, ticker, interval, timeframe_detected, confidence}.

    `ticker` is "" if nothing legible/valid was found. `interval` is normalised
    to an app code (defaults to '1d').
    """
    import pytesseract
    from PIL import Image

    from PIL import ImageOps, ImageStat

    img = Image.open(io.BytesIO(image_bytes)).convert("L")   # grayscale
    # upscale — OCR is much more reliable on larger text
    if img.width < 1600:
        scale = 1600 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    # dark-mode charts are light text on a dark background; Tesseract expects the
    # opposite, so invert when the image is mostly dark
    if ImageStat.Stat(img).mean[0] < 128:
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)   # stretch contrast so faint labels read

    # image_to_data gives per-word boxes so we can prefer top-of-chart tokens
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = data["text"]
    tops = data["top"]
    full_text = " ".join(w for w in words if w.strip())

    if not full_text.strip():
        return {"is_chart": False, "ticker": "", "interval": "1d",
                "timeframe_detected": "unknown", "confidence": 0.0,
                "ocr_preview": "", "candidates": []}

    # 1) exchange-prefixed capture is the strongest signal — resolve it first
    #    (trims any glued-on timeframe). 2) otherwise the topmost uppercase
    #    tokens, resolved in order. Only genuinely uppercase tokens qualify, so
    #    lowercase axis labels / prose never become candidates.
    ticker = ""
    m = _EXCHANGE_RE.search(full_text)
    if m:
        ticker = _resolve(m.group(1))

    cands = []  # (top_y, symbol)
    for w, y in zip(words, tops):
        for c in _CAND_RE.findall(w):
            if c not in _STOP:
                cands.append((y, c))
    cands.sort(key=lambda p: p[0])              # topmost first
    ordered = list(dict.fromkeys(c for _, c in cands))

    if not ticker:
        for c in ordered[:6]:                  # cap validation lookups
            if _valid_ticker(c):
                ticker = c
                break

    interval = _detect_interval(full_text)
    return {
        "is_chart": bool(ticker) or len(full_text) > 20,
        "ticker": ticker,
        "interval": interval if interval in _VALID_INTERVALS else "1d",
        "timeframe_detected": interval,
        "confidence": 0.9 if ticker else 0.0,
        "ocr_preview": full_text[:120],        # diagnostic: what OCR actually saw
        "candidates": ordered[:8],             # diagnostic: uppercase tokens considered
    }
