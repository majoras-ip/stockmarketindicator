"""
main.py — Entry point for the Stock Volatility Prediction Tool.

Run with:  python main.py
"""

from __future__ import annotations

import json
import os
import sys

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Current active ticker (mutable within a session).
_ticker = config.TICKER


# ─── Menu ─────────────────────────────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════╗
║     Stock Volatility Prediction Tool     ║
╠══════════════════════════════════════════╣
║  1. Download & Prepare Data              ║
║  2. Train Model                          ║
║  3. Evaluate Model                       ║
║  4. Predict Next 10 Minutes              ║
║  5. Load Existing Model                  ║
║  6. Export Model Info                    ║
║  7. Change Stock Ticker                  ║
║  8. View Charts (open folder)            ║
║  9. Export TradingView Pine Script       ║
║  ──────────────────────────────────────  ║
║  11. Train 30-Min Forecaster             ║
║  12. Export Forecast Pine Script         ║
║  13. Auto-Refresh Forecast (every 5min) ║
║  14. Multi-Ticker Forecast               ║
║  15. Start Alert Monitor                 ║
║  ──────────────────────────────────────  ║
║  10. Exit                                ║
╚══════════════════════════════════════════╝"""


def _print_menu() -> None:
    print(MENU)
    print(f"  Active ticker: {_ticker}")
    print()


# ─── Option handlers ──────────────────────────────────────────────────────────

def _download_data() -> None:
    """Option 1 — download and cache raw data."""
    from data.collector import download, best_interval

    global _ticker
    period   = input(f"  Period [default: {config.PERIOD}]: ").strip() or config.PERIOD
    interval = input(
        f"  Interval [leave blank to auto-detect for {period}]: "
    ).strip() or None

    use_csv = input("  Use local CSV file instead? (y/N): ").strip().lower()
    if use_csv == "y":
        path = input("  CSV file path: ").strip()
        from data.collector import load_csv
        df = load_csv(path)
    else:
        df = download(_ticker, period=period, interval=interval)

    print(f"\n  Downloaded {len(df):,} rows  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    print("  Data ready. Choose option 2 to train.\n")
    _session["df"] = df


def _train_model() -> None:
    """Option 2 — full training pipeline."""
    from training.trainer import Trainer

    if "df" not in _session:
        print("  No data loaded. Running download first …\n")
        _download_data()

    trainer = Trainer(ticker=_ticker)
    try:
        metrics = trainer.run(_session["df"])
        _session["trainer"] = trainer
        print("\n  Training complete. Models saved to:", config.MODEL_DIR)
    except MemoryError:
        print(
            "\n  [!] Out of memory. Try reducing SEQUENCE_LENGTH or "
            "LGB_N_ESTIMATORS in config.py."
        )
    except Exception as exc:
        print(f"\n  [!] Training failed: {exc}")
        log.exception("Training error")


def _evaluate_model() -> None:
    """Option 3 — re-run evaluation on a loaded/trained model."""
    if "trainer" not in _session:
        print("  No trained model in session. Load one first (option 5).")
        return

    trainer = _session["trainer"]
    if not hasattr(trainer, "robust_model") or trainer.robust_model is None:
        print("  Model not yet trained.")
        return

    print("  Re-evaluating … (metrics were already printed during training)")
    trainer.evaluator._print_table()


def _predict() -> None:
    """Option 4 — predict next 10-minute volatility."""
    from prediction.predictor import Predictor

    if "predictor" not in _session:
        _session["predictor"] = Predictor()

    ticker = input(f"  Ticker [default: {_ticker}]: ").strip().upper() or _ticker
    try:
        _session["predictor"].predict(ticker)
    except FileNotFoundError as exc:
        print(f"\n  [!] {exc}")
    except Exception as exc:
        print(f"\n  [!] Prediction failed: {exc}")
        log.exception("Prediction error")


def _load_model() -> None:
    """Option 5 — load existing model from disk."""
    from models.robust_model import RobustModel
    from training.trainer import Trainer
    import joblib

    model_dir = input(
        f"  Model directory [default: {config.MODEL_DIR}]: "
    ).strip() or config.MODEL_DIR

    if not os.path.isdir(model_dir):
        print(f"  [!] Directory not found: {model_dir}")
        return

    try:
        robust = RobustModel()
        robust.load_all(model_dir)

        scaler_path = os.path.join(model_dir, config.SCALER_FILE)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        metrics_path = os.path.join(model_dir, config.METRICS_FILE)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print("\n  Saved model performance:")
            rob = metrics.get("robust", {}).get("test", {})
            for k, v in rob.items():
                print(f"    {k:8s}: {v:.6f}")

        # Stash so training session references work.
        trainer = Trainer(ticker=_ticker)
        trainer.robust_model = robust
        trainer.scaler       = scaler
        _session["trainer"]  = trainer
        _session["predictor"] = None   # force reload on next predict call
        print("\n  Model loaded and ready.\n")
    except Exception as exc:
        print(f"\n  [!] Failed to load model: {exc}")
        log.exception("Load error")


def _export_model_info() -> None:
    """Option 6 — display and confirm saved model files."""
    model_dir = config.MODEL_DIR
    if not os.path.isdir(model_dir):
        print(f"  No models saved yet in '{model_dir}'.")
        return

    print(f"\n  Files in {model_dir}:")
    for fname in sorted(os.listdir(model_dir)):
        fpath = os.path.join(model_dir, fname)
        size  = os.path.getsize(fpath) / 1024
        print(f"    {fname:<40}  {size:>8.1f} KB")

    cfg_path = os.path.join(model_dir, config.CONFIG_SNAPSHOT_FILE)
    if os.path.exists(cfg_path):
        print("\n  Config snapshot:")
        with open(cfg_path) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            print(f"    {k:<25}: {v}")
    print()


def _change_ticker() -> None:
    """Option 7 — change the active stock ticker."""
    global _ticker
    new = input("  Enter new ticker symbol: ").strip().upper()
    if new:
        _ticker = new
        _session.clear()   # reset session so stale data is not reused
        print(f"  Ticker changed to {_ticker}. Session data cleared.")
    else:
        print("  No change.")


def _export_pine_script() -> None:
    """Option 9 — generate a TradingView Pine Script with embedded predictions."""
    from prediction.pine_exporter import PineExporter

    if "pine_exporter" not in _session:
        _session["pine_exporter"] = PineExporter()

    ticker   = input(f"  Ticker   [default: {_ticker}]: ").strip().upper() or _ticker
    period   = input("  Period   [default: 5d, options: 1d/5d/60d]: ").strip() or "5d"
    interval = input("  Interval [default: 5m, options: 1m/5m/15m/1h]: ").strip() or "5m"

    horizon_input = input("  Forecast horizon in minutes [default: 30, max: 30]: ").strip()
    if horizon_input:
        try:
            minutes = int(horizon_input)
            minutes = max(1, min(minutes, 30))
        except ValueError:
            print("  Invalid input, using 30 minutes.")
            minutes = 30
    else:
        minutes = 30

    # Convert minutes to bars based on interval
    interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(interval, 5)
    horizon_bars = max(1, round(minutes / interval_minutes))
    config.FORECAST_HORIZON = horizon_bars
    print(f"  Forecast horizon set to {minutes} min ({horizon_bars} bars on {interval} chart)")

    try:
        _session["pine_exporter"].export(
            ticker=ticker,
            period=period,
            interval=interval,
        )
    except FileNotFoundError as exc:
        print(f"\n  [!] {exc}")
    except Exception as exc:
        print(f"\n  [!] Export failed: {exc}")
        log.exception("Pine export error")


def _train_forecaster() -> None:
    """Option 11 — train the LSTM multi-step forecaster on 5m Alpaca data."""
    from training.forecast_trainer import ForecastTrainer

    ticker = input(f"  Ticker [default: {_ticker}]: ").strip().upper() or _ticker

    # Fetch 5m data from Alpaca (60 days) so the scaler matches option 12 inference
    print(f"  Fetching 5m data from Alpaca for {ticker} (up to 60 days) …")
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(config.BASE_DIR, ".env"))
        from live.live_feed import fetch_bars
        df = fetch_bars(ticker, "5Min", lookback=10000)
        print(f"  Fetched {len(df)} bars from Alpaca.")
    except Exception as exc:
        print(f"  [!] Alpaca fetch failed: {exc}")
        print("  Falling back to session data (1h) — scale mismatch may occur.")
        if "df" not in _session:
            print("  No data loaded either. Run option 1 first.")
            return
        df = _session["df"]

    trainer = ForecastTrainer()
    try:
        trainer.run(df)
        print("\n  Forecaster training complete.\n")
    except Exception as exc:
        print(f"\n  [!] Forecaster training failed: {exc}")
        log.exception("Forecaster training error")


def _export_forecast_pine() -> None:
    """Option 12 — export LSTM 30-min forecast as Pine Script."""
    from prediction.forecast_exporter import ForecastExporter

    if "forecast_exporter" not in _session:
        _session["forecast_exporter"] = ForecastExporter()

    ticker   = input(f"  Ticker   [default: {_ticker}]: ").strip().upper() or _ticker
    interval = input("  Interval [default: 5m, options: 1m/5m/15m/1h]: ").strip() or "5m"

    # Fetch fresh data via Alpaca or yfinance
    use_live = input("  Use live Alpaca data? (Y/n): ").strip().lower()
    if use_live != "n":
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(config.BASE_DIR, ".env"))
            from live.live_feed import fetch_bars
            interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
            df = fetch_bars(ticker, interval_map.get(interval, "5Min"))
            print(f"  Fetched {len(df)} bars from Alpaca.")
        except Exception as exc:
            print(f"  [!] Alpaca fetch failed: {exc}")
            print("  Falling back to yfinance …")
            from data.collector import download
            df = download(ticker, period="5d", interval=interval, save_csv=False)
    else:
        from data.collector import download
        df = download(ticker, period="5d", interval=interval, save_csv=False)

    try:
        _session["forecast_exporter"].export(df=df, ticker=ticker, interval=interval)
    except FileNotFoundError as exc:
        print(f"\n  [!] {exc}")
    except Exception as exc:
        print(f"\n  [!] Export failed: {exc}")
        log.exception("Forecast export error")


def _auto_refresh_forecast() -> None:
    """Option 13 — auto-refresh LSTM forecast every N minutes."""
    import time
    from prediction.forecast_exporter import ForecastExporter
    from data.collector import download

    ticker   = input(f"  Ticker   [default: {_ticker}]: ").strip().upper() or _ticker
    interval = input("  Interval [default: 5m]: ").strip() or "5m"
    refresh  = input("  Refresh every N minutes [default: 5]: ").strip()
    refresh_secs = int(refresh) * 60 if refresh.isdigit() else 300

    if "forecast_exporter" not in _session:
        _session["forecast_exporter"] = ForecastExporter()

    exporter = _session["forecast_exporter"]

    interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
    alpaca_interval = interval_map.get(interval, "5Min")

    print(f"\n  Auto-refresh started for {ticker} every {refresh_secs // 60} min.")
    print(f"  Press Ctrl+C to stop.\n")

    while True:
        try:
            try:
                from dotenv import load_dotenv
                load_dotenv(os.path.join(config.BASE_DIR, ".env"))
                from live.live_feed import fetch_bars
                df = fetch_bars(ticker, alpaca_interval)
            except Exception:
                df = download(ticker, period="5d", interval=interval, save_csv=False)

            exporter._loaded = False  # force reload of model each cycle
            exporter.export(df=df, ticker=ticker, interval=interval)

            out_path = os.path.join(config.CHART_DIR, f"{ticker.lower()}_forecast_{interval}.pine")
            print(f"  Copy to clipboard: pbcopy < {out_path}")
            print(f"  Next update in {refresh_secs // 60} min — press Ctrl+C to stop.\n")
            time.sleep(refresh_secs)

        except KeyboardInterrupt:
            print("\n  Auto-refresh stopped.\n")
            break
        except Exception as exc:
            print(f"\n  [!] Error: {exc} — retrying in {refresh_secs // 60} min …\n")
            time.sleep(refresh_secs)


def _multi_ticker_forecast() -> None:
    """Option 14 — generate forecast Pine Scripts for multiple tickers at once."""
    from prediction.forecast_exporter import ForecastExporter
    from data.collector import download

    raw = input("  Tickers (comma-separated, e.g. AAPL,PLTR,NVDA): ").strip().upper()
    tickers = [t.strip() for t in raw.split(",") if t.strip()]
    if not tickers:
        print("  No tickers entered.")
        return

    interval = input("  Interval [default: 5m]: ").strip() or "5m"
    interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
    alpaca_interval = interval_map.get(interval, "5Min")

    use_live = input("  Use live Alpaca data? (Y/n): ").strip().lower()

    exporter = ForecastExporter()

    print()
    for ticker in tickers:
        print(f"  Processing {ticker} …")
        try:
            if use_live != "n":
                try:
                    from dotenv import load_dotenv
                    load_dotenv(os.path.join(config.BASE_DIR, ".env"))
                    from live.live_feed import fetch_bars
                    df = fetch_bars(ticker, alpaca_interval)
                except Exception as exc:
                    print(f"    Alpaca failed ({exc}), falling back to yfinance …")
                    df = download(ticker, period="5d", interval=interval, save_csv=False)
            else:
                df = download(ticker, period="5d", interval=interval, save_csv=False)

            exporter._loaded = False  # reload model for each ticker
            path = exporter.export(df=df, ticker=ticker, interval=interval)
            print(f"    Saved: {path}")
        except Exception as exc:
            print(f"    [!] Failed for {ticker}: {exc}")

    print(f"\n  Done. Files saved in: {config.CHART_DIR}")
    print(f"  Copy any file with: pbcopy < {config.CHART_DIR}/<ticker>_forecast_{interval}.pine\n")


def _alert_monitor() -> None:
    """Option 15 — monitor forecast and send phone alerts via ntfy.sh."""
    import time
    from prediction.forecast_exporter import ForecastExporter
    from data.forecast_preprocessor import FORECAST_STEPS
    from models.forecaster import build_sequences
    from data.preprocessor import _filter_market_hours, _remove_outliers
    from data.features import build_features
    from utils.alerts import check_and_alert
    import numpy as np

    print("\n  Alert Monitor uses ntfy.sh — free push notifications to your phone.")
    print("  1. Install the 'ntfy' app (iOS / Android)")
    print("  2. Subscribe to your topic in the app")
    print("  3. Enter the same topic name below\n")

    raw = input(f"  Tickers (comma-separated) [default: {_ticker}]: ").strip().upper()
    tickers  = [t.strip() for t in raw.split(",") if t.strip()] or [_ticker]
    interval = input("  Interval [default: 5m]: ").strip() or "5m"
    topic    = input("  ntfy topic name (e.g. stockvol-ayden): ").strip()
    refresh  = input("  Check every N minutes [default: 5]: ").strip()
    refresh_secs = int(refresh) * 60 if refresh.isdigit() else 300

    if not topic:
        print("  No topic entered — alerts disabled.")
        return

    interval_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
    alpaca_interval = interval_map.get(interval, "5Min")

    exporter    = ForecastExporter()
    seen_events: set = set()

    print(f"\n  Monitoring {tickers} every {refresh_secs // 60} min → ntfy.sh/{topic}")
    print("  Press Ctrl+C to stop.\n")

    while True:
        for ticker in tickers:
            try:
                try:
                    from dotenv import load_dotenv
                    load_dotenv(os.path.join(config.BASE_DIR, ".env"))
                    from live.live_feed import fetch_bars
                    df = fetch_bars(ticker, alpaca_interval)
                except Exception:
                    from data.collector import download
                    df = download(ticker, period="5d", interval=interval, save_csv=False)

                exporter._loaded = False
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
                    continue

                future = list(exporter.forecaster.predict_last(X_seq) * exporter.y_scale)
                arr    = exporter.forecaster.predict(X_seq)[:, 0] * exporter.y_scale
                high_thresh = float(np.percentile(arr, 75))

                direction_up   = None
                direction_conf = None
                if exporter.direction_model is not None:
                    proba = exporter.direction_model.predict_proba(features.iloc[[-1]].values)[0]
                    direction_up   = bool(proba[1] >= 0.5)
                    direction_conf = float(max(proba))

                check_and_alert(
                    topic=topic,
                    ticker=ticker,
                    future_values=future,
                    high_thresh=high_thresh,
                    direction_up=direction_up,
                    direction_conf=direction_conf,
                    seen_events=seen_events,
                )
                print(f"  [{ticker}] checked — max forecast: {max(future):.5f}  thresh: {high_thresh:.5f}")

            except Exception as exc:
                print(f"  [{ticker}] error: {exc}")

        try:
            time.sleep(refresh_secs)
        except KeyboardInterrupt:
            print("\n  Alert monitor stopped.\n")
            break


def _view_charts() -> None:
    """Option 8 — open the charts directory in Finder / Explorer."""
    import subprocess, platform
    chart_dir = config.CHART_DIR
    if not os.path.isdir(chart_dir):
        print(f"  No charts folder found at '{chart_dir}'. Train a model first.")
        return

    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", chart_dir])
        elif system == "Windows":
            subprocess.Popen(["explorer", chart_dir])
        else:
            subprocess.Popen(["xdg-open", chart_dir])
        print(f"  Opened: {chart_dir}")
    except Exception as exc:
        print(f"  Could not open folder automatically: {exc}")
        print(f"  Charts are saved in: {chart_dir}")


# ─── Main loop ────────────────────────────────────────────────────────────────

_session: dict = {}   # in-memory state for the current run

_HANDLERS = {
    "1": _download_data,
    "2": _train_model,
    "3": _evaluate_model,
    "4": _predict,
    "5": _load_model,
    "6": _export_model_info,
    "7": _change_ticker,
    "8": _view_charts,
    "9": _export_pine_script,
    "11": _train_forecaster,
    "12": _export_forecast_pine,
    "13": _auto_refresh_forecast,
    "14": _multi_ticker_forecast,
    "15": _alert_monitor,
}


def main() -> None:
    print("\nWelcome to the Stock Volatility Prediction Tool")
    print("Based on: Liao, Chen & Cai (2024) — LightGBM-GRU Hybrid Model\n")

    while True:
        _print_menu()
        choice = input("  Select option (1-10): ").strip()

        if choice == "10":
            print("\n  Goodbye!\n")
            sys.exit(0)

        handler = _HANDLERS.get(choice)
        if handler is None:
            print("  Invalid choice. Please enter a number between 1 and 10.\n")
            continue

        print()
        try:
            handler()
        except KeyboardInterrupt:
            print("\n  Interrupted. Returning to menu.\n")
        except Exception as exc:
            print(f"\n  [!] Unexpected error: {exc}\n")
            log.exception("Unhandled error in menu handler")


if __name__ == "__main__":
    main()
