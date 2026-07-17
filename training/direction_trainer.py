"""
training/direction_trainer.py — Multi-horizon *directional* model for ChartEdge.

What it predicts
────────────────
For each horizon H (in bars), it models the **distribution of the forward
log-return** r_H = log(close[t+H] / close[t]) using LightGBM quantile
regression.  From the predicted quantiles you get, for free:

  • direction  = sign of the median (q50)
  • confidence = how far q50 sits from 0 relative to the q10–q90 spread
  • price cone = quantiles mapped back onto price (target + downside/upside)

Why quantiles instead of a naive up/down classifier: a bare classifier throws
away magnitude and confidence, and its accuracy is nearly uninformative on
near-efficient markets.  A calibrated distribution is both more useful and more
honest — and its median sign still answers "which way".

Honesty guardrails (this is the part that matters)
──────────────────────────────────────────────────
  • Purged walk-forward CV with an embargo of H bars, so the model is never
    validated on a bar whose label overlaps the training window.  This is what
    stops "70% in backtest, 50% live".
  • Per-ticker feature/label construction (shifts never cross ticker
    boundaries) before pooling.
  • Reported metrics include directional accuracy, q10–q90 coverage
    (calibration), pinball loss, and an after-cost hit rate at a confidence
    threshold — the numbers that reveal whether there's a real edge.

Run:  python3 -m training.direction_trainer
"""

from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

# LightGBM's sklearn wrapper warns when predicting on a bare ndarray after
# fitting on one — harmless here (we intentionally use arrays end-to-end).
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import config
from data.pattern_features import build_all_features
from utils.logger import get_logger

log = get_logger(__name__)

# ─── artifact names ───────────────────────────────────────────────────────────
DIRECTION_QUANTILE_FILE = "direction_quantile_model.joblib"
DIRECTION_METRICS_FILE  = "direction_metrics.json"

# ─── defaults ─────────────────────────────────────────────────────────────────
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
# horizons in *bars*; meaning depends on the interval you train on
DEFAULT_HORIZONS = {"h1": 1, "h6": 6, "h24": 24}
DEFAULT_TICKERS  = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN"]
# a "flat" move smaller than this (in the horizon's return units) is treated as
# no-call when scoring directional accuracy, so noise near zero doesn't dominate
FLAT_EPS = 0.0005
# round-trip cost assumption for the after-cost hit-rate sanity check
COST_PER_TRADE = 0.0004
CONF_THRESHOLD = 0.15   # only "take" trades where |q50| exceeds this * spread


# ─── purged walk-forward CV ───────────────────────────────────────────────────

def purged_walk_forward(n: int, n_splits: int, embargo: int):
    """
    Yield (train_idx, test_idx) for expanding-window walk-forward validation.

    Rows are assumed time-ordered.  The last `embargo` training rows before each
    test block are dropped so a training label cannot overlap the test window.
    """
    fold = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_start = train_end
        test_end = fold * (k + 1) if k < n_splits else n
        tr = np.arange(0, max(0, train_end - embargo))
        te = np.arange(test_start, test_end)
        if len(tr) and len(te):
            yield tr, te


# ─── dataset construction ─────────────────────────────────────────────────────

def _build_dataset(tickers, period, interval, horizons):
    """Download each ticker, build features + per-horizon labels, pool them."""
    from data.collector import download

    frames = []
    for tkr in tickers:
        try:
            df = download(ticker=tkr, period=period, interval=interval, save_csv=False)
        except Exception as exc:  # noqa: BLE001 — one bad ticker shouldn't kill the run
            log.warning("Skipping %s: %s", tkr, exc)
            continue

        feats = build_all_features(df)
        close = df["Close"].reindex(feats.index)

        labels = {}
        for name, hbars in horizons.items():
            labels[f"y_{name}"] = np.log(close.shift(-hbars) / close)

        block = feats.copy()
        for col, series in labels.items():
            block[col] = series
        block["__ticker"] = tkr
        block["__ts"] = feats.index
        frames.append(block)
        log.info("%s: %d rows", tkr, len(block))

    if not frames:
        raise RuntimeError("No data downloaded for any ticker.")

    data = pd.concat(frames, ignore_index=True)
    # time-order the pooled rows so walk-forward CV respects chronology
    data = data.sort_values("__ts", kind="mergesort").reset_index(drop=True)
    return data


def _feature_columns(data, horizons):
    non_features = {"__ticker", "__ts"} | {f"y_{n}" for n in horizons}
    return [c for c in data.columns if c not in non_features]


# ─── metrics ──────────────────────────────────────────────────────────────────

def _pinball(y_true, y_pred, q):
    d = y_true - y_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))


def _score_horizon(y_true, preds_by_q):
    """preds_by_q: dict quantile->array. Returns a metrics dict."""
    q50 = preds_by_q[0.5]
    q10, q90 = preds_by_q[0.1], preds_by_q[0.9]

    # directional accuracy on non-flat actual moves
    mask = np.abs(y_true) > FLAT_EPS
    if mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(q50[mask]) == np.sign(y_true[mask])))
    else:
        dir_acc = float("nan")

    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))  # target ≈ 0.80
    pinball = float(np.mean([_pinball(y_true, preds_by_q[q], q) for q in preds_by_q]))

    # after-cost hit rate: only "take" high-confidence calls
    spread = np.maximum(q90 - q10, 1e-9)
    conf = np.abs(q50) / spread
    take = conf > CONF_THRESHOLD
    if take.sum() > 0:
        gross = np.sign(q50[take]) * y_true[take]
        net = gross - COST_PER_TRADE
        hit_after_cost = float(np.mean(net > 0))
        n_trades = int(take.sum())
        avg_net = float(np.mean(net))
    else:
        hit_after_cost, n_trades, avg_net = float("nan"), 0, float("nan")

    return {
        "directional_accuracy": round(dir_acc, 4),
        "q10_q90_coverage": round(coverage, 4),
        "pinball_loss": round(pinball, 6),
        "after_cost_hit_rate": round(hit_after_cost, 4) if n_trades else None,
        "trades_taken": n_trades,
        "avg_net_return_per_trade": round(avg_net, 6) if n_trades else None,
        "n_eval": int(len(y_true)),
    }


def _fit_quantile_models(X, y, quantiles):
    import lightgbm as lgb

    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=400, learning_rate=0.03,
            num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, min_child_samples=40,
            reg_lambda=1.0, random_state=config.LGB_RANDOM_STATE, verbose=-1,
        )
        m.fit(X, y)
        models[q] = m
    return models


def _predict_quantiles(models, X):
    """Predict and enforce monotonic (non-crossing) quantiles per row."""
    qs = sorted(models)
    raw = np.column_stack([models[q].predict(X) for q in qs])
    raw = np.sort(raw, axis=1)                      # fix any quantile crossing
    return {q: raw[:, i] for i, q in enumerate(qs)}


# ─── main entry point ─────────────────────────────────────────────────────────

def train(
    tickers=None,
    period="2y",
    interval="1h",
    horizons=None,
    quantiles=QUANTILES,
    n_splits=5,
    model_dir=config.MODEL_DIR,
):
    tickers  = tickers or DEFAULT_TICKERS
    horizons = horizons or DEFAULT_HORIZONS

    print(f"\n  Building dataset: {tickers}  period={period}  interval={interval}")
    data = _build_dataset(tickers, period, interval, horizons)
    feat_cols = _feature_columns(data, horizons)
    print(f"  {len(data)} pooled rows · {len(feat_cols)} features · horizons={horizons}\n")

    all_metrics = {}
    all_models  = {}

    for name, hbars in horizons.items():
        ycol = f"y_{name}"
        sub = data[feat_cols + [ycol]].dropna()
        X = sub[feat_cols].values
        y = sub[ycol].values
        n = len(X)
        if n < 500:
            log.warning("Horizon %s has only %d usable rows — skipping.", name, n)
            continue

        # ── walk-forward validation (out-of-sample scoring) ──
        fold_scores = []
        for tr, te in purged_walk_forward(n, n_splits, embargo=hbars):
            models = _fit_quantile_models(X[tr], y[tr], quantiles)
            preds = _predict_quantiles(models, X[te])
            fold_scores.append(_score_horizon(y[te], preds))

        # average the numeric metrics across folds
        agg = {}
        keys = ["directional_accuracy", "q10_q90_coverage", "pinball_loss"]
        for kk in keys:
            vals = [f[kk] for f in fold_scores if f[kk] is not None and not np.isnan(f[kk])]
            agg[kk] = round(float(np.mean(vals)), 4) if vals else None
        ac = [f["after_cost_hit_rate"] for f in fold_scores if f["after_cost_hit_rate"] is not None]
        agg["after_cost_hit_rate"] = round(float(np.mean(ac)), 4) if ac else None
        agg["folds"] = len(fold_scores)
        agg["horizon_bars"] = hbars
        all_metrics[name] = agg

        print(f"  ── {name} (H={hbars} bars, n={n}) ──")
        print(f"     directional accuracy : {agg['directional_accuracy']}   (0.50 = coin flip)")
        print(f"     q10–q90 coverage     : {agg['q10_q90_coverage']}   (0.80 = calibrated)")
        print(f"     pinball loss         : {agg['pinball_loss']}")
        print(f"     after-cost hit rate  : {agg['after_cost_hit_rate']}\n")

        # ── final model: refit on ALL rows for deployment ──
        all_models[name] = _fit_quantile_models(X, y, quantiles)

    # ── persist ──
    os.makedirs(model_dir, exist_ok=True)
    artifact = {
        "models": all_models,
        "feature_cols": feat_cols,
        "quantiles": list(quantiles),
        "horizons": horizons,
        "trained_on": {"tickers": tickers, "period": period, "interval": interval},
    }
    joblib.dump(artifact, os.path.join(model_dir, DIRECTION_QUANTILE_FILE))
    with open(os.path.join(model_dir, DIRECTION_METRICS_FILE), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  Saved model  → {os.path.join(model_dir, DIRECTION_QUANTILE_FILE)}")
    print(f"  Saved metrics→ {os.path.join(model_dir, DIRECTION_METRICS_FILE)}\n")
    return all_metrics


# ─── inference helper ─────────────────────────────────────────────────────────

class DirectionModel:
    """Load the trained artifact and produce a directional read for a df."""

    def __init__(self, model_dir: str = config.MODEL_DIR):
        path = os.path.join(model_dir, DIRECTION_QUANTILE_FILE)
        self.art = joblib.load(path)

    def predict(self, df: pd.DataFrame) -> dict:
        """Return per-horizon direction + confidence + price cone for the last bar."""
        feats = build_all_features(df)
        row = feats[self.art["feature_cols"]].iloc[[-1]].fillna(0.0).values
        last_close = float(df["Close"].iloc[-1])

        out = {}
        for name, models in self.art["models"].items():
            preds = _predict_quantiles(models, row)
            q = {k: float(v[0]) for k, v in preds.items()}
            q50 = q[0.5]
            spread = max(q[0.9] - q[0.1], 1e-9)
            out[name] = {
                "direction": "up" if q50 > 0 else "down",
                "expected_return": q50,
                "confidence": round(min(abs(q50) / spread, 1.0), 3),
                "target_price": round(last_close * np.exp(q50), 4),
                "low_price":  round(last_close * np.exp(q[0.1]), 4),
                "high_price": round(last_close * np.exp(q[0.9]), 4),
            }
        return out


if __name__ == "__main__":
    train()
