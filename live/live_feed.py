"""
live/live_feed.py — Real-time volatility prediction loop using Alpaca.

Fetches the latest bars every 5 minutes during market hours, runs the
LightGBM model, and writes a fresh Pine Script file ready to paste into
TradingView.

Usage:
    python3 live/live_feed.py --ticker AAPL
    python3 live/live_feed.py --ticker PLTR --interval 5Min --horizon 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load API keys from .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from prediction.pine_exporter import PineExporter
from utils.logger import get_logger

log = get_logger(__name__)


def is_market_open() -> bool:
    """Return True if US equity market is currently open (9:30–16:00 ET)."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        clock = client.get_clock()
        return clock.is_open
    except Exception as exc:
        log.warning("Could not check market clock: %s", exc)
        return True   # assume open if check fails


def fetch_bars(ticker: str, interval: str, lookback: int = 500) -> "pd.DataFrame":
    """
    Fetch the last *lookback* bars for *ticker* at *interval* from Alpaca.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a tz-aware DatetimeIndex (UTC).
    """
    import pandas as pd
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    # Map interval string to Alpaca TimeFrame
    tf_map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1,  TimeFrameUnit.Hour),
    }
    tf = tf_map.get(interval, TimeFrame(5, TimeFrameUnit.Minute))

    from datetime import timedelta
    end   = datetime.now(timezone.utc)
    # For 5Min bars: ~78 bars/day × 60 days ≈ 4,680 bars; use 65 days to be safe
    # For 1Min bars: ~390 bars/day × 10 days ≈ 3,900; cap at 10 days (yfinance limit)
    day_map = {"1Min": 10, "5Min": 65, "15Min": 65, "1Hour": 65}
    days = day_map.get(interval, 65)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=start,
        end=end,
        feed="iex",
    )
    bars = client.get_stock_bars(request).df

    if bars.empty:
        raise ValueError(f"No bars returned for {ticker}.")

    # Alpaca returns a MultiIndex (symbol, timestamp) — drop symbol level
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")

    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    # Alpaca doesn't provide Bid/Ask in bar data — approximate from OHLC
    bars["Bid"] = bars["Low"]
    bars["Ask"] = bars["High"]

    return bars.tail(lookback)


def run_once(ticker: str, interval: str, horizon_mins: int, exporter: PineExporter) -> str:
    """Fetch latest data, run model, write Pine Script. Returns output path."""
    log.info("Fetching latest %s bars for %s …", interval, ticker)
    df = fetch_bars(ticker, interval)

    import joblib
    import pandas as pd
    from data.preprocessor import prepare

    # Set forecast horizon
    interval_minutes = {"1Min": 1, "5Min": 5, "15Min": 15, "1Hour": 60}.get(interval, 5)
    config.FORECAST_HORIZON = max(1, round(horizon_mins / interval_minutes))

    X, _y, _ = prepare(
        df,
        scaler_path=os.path.join(exporter.model_dir, config.SCALER_FILE),
        fit_scaler=False,
    )

    if exporter.feature_names:
        for col in set(exporter.feature_names) - set(X.columns):
            X[col] = 0.0
        X = X[exporter.feature_names]

    preds = exporter.lgb.predict(X)

    import pandas as pd
    pred_series = pd.Series(preds, index=X.index)

    from prediction.pine_exporter import _build_pine_script
    import numpy as np

    timestamps_ms = [int(ts.timestamp() * 1000) for ts in pred_series.index]
    values = [round(float(v), 8) for v in pred_series.values]
    interval_mins = interval_minutes

    pine_code = _build_pine_script(
        ticker=ticker,
        interval=interval.replace("Min", "m").replace("Hour", "h"),
        interval_mins=interval_mins,
        timestamps_ms=timestamps_ms,
        values=values,
    )

    os.makedirs(config.CHART_DIR, exist_ok=True)
    interval_label = interval.replace("Min", "m").replace("Hour", "h").lower()
    filename = f"{ticker.lower()}_live_rv_{interval_label}.pine"
    out_path = os.path.join(config.CHART_DIR, filename)
    with open(out_path, "w") as f:
        f.write(pine_code)

    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n  [{now}] Updated: {out_path}")
    print(f"  Bars: {len(values)} | Last RV: {values[-1]:.6f} | "
          f"Horizon: +{horizon_mins}min")
    print(f"\n  Copy to clipboard:")
    print(f"    cat {out_path} | pbcopy\n")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Live volatility prediction loop")
    parser.add_argument("--ticker",   default="AAPL",  help="Stock ticker")
    parser.add_argument("--interval", default="5Min",
                        choices=["1Min", "5Min", "15Min", "1Hour"],
                        help="Bar interval")
    parser.add_argument("--horizon",  default=30, type=int,
                        help="Forecast horizon in minutes (max 30)")
    parser.add_argument("--refresh",  default=5,  type=int,
                        help="How often to refresh in minutes")
    parser.add_argument("--once",     action="store_true",
                        help="Run once and exit (no loop)")
    args = parser.parse_args()

    ticker       = args.ticker.upper()
    horizon_mins = max(1, min(args.horizon, 30))
    refresh_secs = args.refresh * 60

    print(f"\n  Live Volatility Feed")
    print(f"  Ticker   : {ticker}")
    print(f"  Interval : {args.interval}")
    print(f"  Horizon  : +{horizon_mins} min")
    print(f"  Refresh  : every {args.refresh} min")
    print(f"  Press Ctrl+C to stop\n")

    exporter = PineExporter()
    exporter.load()

    if args.once:
        run_once(ticker, args.interval, horizon_mins, exporter)
        return

    while True:
        try:
            if not is_market_open():
                now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                print(f"  [{now}] Market closed — waiting {args.refresh} min …")
            else:
                run_once(ticker, args.interval, horizon_mins, exporter)
        except KeyboardInterrupt:
            print("\n  Stopped.\n")
            break
        except Exception as exc:
            log.error("Error in live loop: %s", exc)
            print(f"  [!] {exc} — retrying in {args.refresh} min …")

        time.sleep(refresh_secs)


if __name__ == "__main__":
    main()
