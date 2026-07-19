"""Local dev preview for the Expected Move feature — run WITHOUT the database.

    python3 preview_expected_move.py   then open http://127.0.0.1:5065/expected-move

Serves the real page + real /api endpoints (reusing prediction.chart_reader and
prediction.expected_move), plus a generated sample chart at /test-chart.png so you
can try the screenshot upload. Add ?demo=1 to auto-upload the sample. Not used in
production — dashboard.py is the real app."""
import ast, io, random, math
from flask import Flask, render_template_string, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFont

# --- pull the real template out of dashboard.py (no DB import) ---
SRC = open("dashboard.py").read(); tree = ast.parse(SRC)
A = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.Add, ast.Mult, ast.Mod,
     ast.Tuple, ast.JoinedStr, ast.FormattedValue)
pure = lambda n: all(isinstance(x, A) for x in ast.walk(n))
ns = {"_GA_ID": "", "GA_ID": "", "_GA_SCRIPT": ""}
cands = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
         and isinstance(n.targets[0], ast.Name) and pure(n.value)]
for _ in range(3):
    for n in cands:
        try: exec(ast.get_source_segment(SRC, n), ns)
        except Exception: pass

def _font(sz):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

def make_chart():
    """A busy, TradingView-ish AAPL daily screenshot."""
    W, H = 1280, 760
    img = Image.new("RGB", (W, H), (13, 17, 23)); d = ImageDraw.Draw(img)
    # header
    d.text((24, 16), "NASDAQ:AAPL", fill=(230, 237, 243), font=_font(30))
    d.text((250, 22), "1D", fill=(88, 166, 255), font=_font(22))
    d.text((300, 22), "·  Apple Inc.", fill=(139, 148, 158), font=_font(20))
    d.text((520, 22), "O 228.14  H 231.02  L 227.55  C 230.61", fill=(139, 148, 158), font=_font(18))
    # timeframe toolbar
    for i, tf in enumerate(["1m", "5m", "15m", "1h", "1D", "1W"]):
        col = (88, 166, 255) if tf == "1D" else (110, 118, 129)
        d.text((24 + i * 46, 58), tf, fill=col, font=_font(16))
    # price axis
    for i in range(6):
        y = 120 + i * 95
        d.line([70, y, W - 70, y], fill=(30, 36, 43), width=1)
        d.text((W - 62, y - 8), f"{235 - i*3:.0f}.0", fill=(110, 118, 129), font=_font(14))
    # candlesticks
    random.seed(7); price = 300.0
    for i in range(70):
        x = 90 + i * 16
        o = price; c = o + random.uniform(-14, 14); price = c
        hi = max(o, c) + random.uniform(0, 8); lo = min(o, c) - random.uniform(0, 8)
        up = c >= o; col = (63, 185, 80) if up else (248, 81, 73)
        sy = lambda v: 120 + (300 - v) * 1.5
        d.line([x, sy(hi), x, sy(lo)], fill=col, width=1)
        d.rectangle([x - 4, sy(max(o, c)), x + 4, sy(min(o, c))], fill=col)
    # indicator subpanel labels
    d.text((24, H - 120), "RSI 14  58.3", fill=(139, 148, 158), font=_font(16))
    d.text((24, H - 96),  "MACD 12 26 9   Vol 47.2M", fill=(139, 148, 158), font=_font(16))
    buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0); return buf

# --- real compute (reuses the real expected_move math) ---
def compute(ticker, interval):
    import numpy as np, pandas as pd, yfinance as yf
    pm = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "2y"}
    raw = yf.download(ticker, period=pm.get(interval, "60d"), interval=interval,
                      progress=False, auto_adjust=True)
    if raw is None or len(raw) < 30: raise ValueError("not_enough_data")
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna(); last = float(close.iloc[-1])
    rv = float(np.log(close/close.shift(1)).dropna().tail(30).std())
    from prediction.expected_move import expected_move
    bm = {"1m":1,"5m":5,"15m":15,"1h":60,"1d":1440}.get(interval,60)
    def lbl(h):
        if interval=="1d": return f"{h} day"+("s" if h>1 else "")
        m=h*bm; return f"{m} min" if m<60 else (f"{m//60}h" if m%60==0 else f"{m//60}h {m%60}m")
    return {"ticker":ticker,"interval":interval,"last_price":round(last,4),
            "moves":[dict(expected_move(last,horizon_bars=h,rv_per_bar=rv),label=lbl(h)) for h in (1,6,24)]}

app = Flask(__name__)

@app.route("/expected-move")
def page():
    html = render_template_string(ns["EXPECTED_MOVE_HTML"], current_user=None, nav_plan="free")
    if request.args.get("demo"):
        inject = """
<script>window.addEventListener('load', () => setTimeout(async () => {
  const blob = await (await fetch('/test-chart.png')).blob();
  loadFromImage(new File([blob], 'chart.png', {type:'image/png'}));
}, 300));</script>
</body>"""
        html = html.replace("</body>", inject)
    return html

@app.route("/test-chart.png")
def chart():
    return send_file(make_chart(), mimetype="image/png")

@app.route("/api/expected_move")
def gm():
    try: return jsonify(compute(request.args.get("ticker","SPY").upper(),
                                 request.args.get("interval","1h")))
    except ValueError: return jsonify({"error":"Not enough data"}),400
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/expected_move_from_image", methods=["POST"])
def im():
    pic = request.files.get("chart")
    if pic is None: return jsonify({"error":"No image uploaded"}),400
    data = pic.read()
    from prediction.chart_reader import read_chart
    det = read_chart(data, media_type=pic.mimetype or "image/png")
    if not det["ticker"]:
        return jsonify({"error":"Couldn't read the ticker — type it above.","manual":True}),422
    try: payload = compute(det["ticker"], det["interval"])
    except ValueError: return jsonify({"error":f"No data for {det['ticker']}.","manual":True}),400
    payload["detected"] = det
    return jsonify(payload)

if __name__ == "__main__":
    app.run(port=5065, use_reloader=False)
