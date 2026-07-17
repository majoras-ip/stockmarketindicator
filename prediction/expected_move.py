"""
prediction/expected_move.py — Honest "expected move ±X%" from volatility.

The directional test showed that predicting *which way* a stock goes has no
skill (worse than an always-up baseline — see training/direction_trainer.py).
Predicting *how much* it moves, however, is exactly what the volatility model
already does well.  This module turns that into a user-facing primitive:

    "SPY — expected move over the next 20 bars: ±3.1%
     68% chance within [$728, $774], 95% within [$705, $799].
     No directional call: up/down at this horizon is a coin flip."

It makes **no** directional claim.  That's the point — it's the version that is
both useful and honest.

Volatility source (in priority order):
  1. a caller-supplied forecast RV (e.g. from the LSTM forecaster), or
  2. a self-contained realised-vol estimate straight from OHLCV.

`rv` here is the per-bar return standard deviation; a horizon of H bars scales
it by sqrt(H) under the standard random-walk assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ~1.96σ for a 95% band; 1σ ≈ 68%
_Z95 = 1.96


def _realised_vol_per_bar(df: pd.DataFrame, window: int = 30) -> float:
    """Per-bar log-return std over the last `window` bars (a vol fallback)."""
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_ret) < 5:
        raise ValueError("Not enough bars to estimate volatility.")
    return float(log_ret.tail(window).std())


def expected_move(
    last_price: float,
    horizon_bars: int,
    rv_per_bar: float | None = None,
    df: pd.DataFrame | None = None,
    window: int = 30,
) -> dict:
    """
    Compute an expected-move band for `horizon_bars` ahead.

    Provide either `rv_per_bar` (e.g. from the trained forecaster) or `df` to
    estimate it from recent OHLCV.  Returns move sizes and price bands with an
    explicit no-direction statement.
    """
    if rv_per_bar is None:
        if df is None:
            raise ValueError("Provide rv_per_bar or df to estimate volatility.")
        rv_per_bar = _realised_vol_per_bar(df, window)

    # random-walk scaling of per-bar vol to the horizon
    sigma_h = rv_per_bar * np.sqrt(horizon_bars)

    move_1sig = float(sigma_h)                 # ~68% expected move (log-return units)
    move_2sig = float(_Z95 * sigma_h)          # ~95% expected move

    return {
        "horizon_bars": horizon_bars,
        "last_price": round(float(last_price), 4),
        "expected_move_pct_68": round(move_1sig * 100, 2),
        "expected_move_pct_95": round(move_2sig * 100, 2),
        "band_68": [round(last_price * np.exp(-move_1sig), 2),
                    round(last_price * np.exp(move_1sig), 2)],
        "band_95": [round(last_price * np.exp(-move_2sig), 2),
                    round(last_price * np.exp(move_2sig), 2)],
        "direction": None,   # intentionally omitted — no skill exists here
        "disclaimer": "Expected magnitude only. Direction at this horizon is not predictable.",
    }


def summarize(result: dict, ticker: str = "") -> str:
    """One-line human-readable rendering for CLI / UI."""
    lo68, hi68 = result["band_68"]
    lo95, hi95 = result["band_95"]
    tag = f"{ticker} " if ticker else ""
    return (
        f"{tag}expected move over next {result['horizon_bars']} bars: "
        f"±{result['expected_move_pct_68']}% (68%)  /  "
        f"±{result['expected_move_pct_95']}% (95%)\n"
        f"    68% within [{lo68}, {hi68}]   95% within [{lo95}, {hi95}]\n"
        f"    {result['disclaimer']}"
    )


if __name__ == "__main__":
    from data.collector import download

    df = download(ticker="SPY", period="60d", interval="1h", save_csv=False)
    last = float(df["Close"].iloc[-1])
    for h in (1, 6, 24):
        print(summarize(expected_move(last, horizon_bars=h, df=df), ticker="SPY"))
        print()
