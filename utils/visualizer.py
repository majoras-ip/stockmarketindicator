"""
utils/visualizer.py — Generate and save all charts.

All plots are saved as PNG files to the /charts directory.
Matplotlib is used in non-interactive (Agg) mode so it works on
headless servers (Colab, SSH) as well as desktops.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")   # must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils.logger import get_logger

log = get_logger(__name__)


def _ensure_chart_dir() -> None:
    os.makedirs(config.CHART_DIR, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> str:
    _ensure_chart_dir()
    path = os.path.join(config.CHART_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Chart saved → %s", path)
    return path


# ─── 1. Predicted vs Actual ───────────────────────────────────────────────────

def plot_predictions(
    y_true: np.ndarray,
    y_pred_lgb: np.ndarray,
    y_pred_robust: np.ndarray,
    title: str = "Predicted vs Actual Volatility",
) -> str:
    n = min(len(y_true), len(y_pred_robust))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y_true[:n],        label="Actual",          linewidth=1.0, color="black")
    ax.plot(x, y_pred_lgb[:n],    label="LightGBM",        linewidth=0.8, alpha=0.7, color="steelblue")
    ax.plot(x, y_pred_robust[:n], label="LightGBM-GRU",    linewidth=0.8, alpha=0.9, color="tomato")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time step")
    ax.set_ylabel("10-min Realised Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "predicted_vs_actual.png")


# ─── 2. Metric Comparison Bar Chart ───────────────────────────────────────────

def plot_metric_comparison(metrics: dict) -> str:
    """
    *metrics* structure:
      {
        "lightgbm": {"train": {...}, "test": {...}},
        "robust":   {"train": {...}, "test": {...}},
      }
    """
    keys   = ["MSE", "MAE", "RMSE", "RMSPE"]
    labels = ["LGB Train", "LGB Test", "Robust Train", "Robust Test"]
    x      = np.arange(len(keys))
    width  = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    values = [
        [metrics["lightgbm"]["train"][k] for k in keys],
        [metrics["lightgbm"]["test"][k]  for k in keys],
        [metrics["robust"]["train"][k]   for k in keys],
        [metrics["robust"]["test"][k]    for k in keys],
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, (vals, label, color) in enumerate(zip(values, labels, colors)):
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_title("Model Metric Comparison", fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "metric_comparison.png")


# ─── 3. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(importance: pd.Series, top_n: int = 20) -> str:
    top = importance.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top.index, top.values, color="steelblue", alpha=0.85)
    ax.set_title(f"Top {top_n} LightGBM Features (Gain)", fontsize=14)
    ax.set_xlabel("Mean Gain")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


# ─── 4. Training Loss Curve ───────────────────────────────────────────────────

def plot_training_loss(history) -> str:
    """*history* is the Keras History object returned by model.fit()."""
    fig, ax = plt.subplots(figsize=(10, 4))

    train_loss = history.history.get("loss", [])
    val_loss   = history.history.get("val_loss", [])

    ax.plot(train_loss, label="Training loss",   color="steelblue")
    if val_loss:
        # val_loss may be recorded every validation_freq epochs.
        val_x = np.linspace(0, len(train_loss) - 1, len(val_loss))
        ax.plot(val_x, val_loss, label="Validation loss", color="tomato", linestyle="--")

    ax.set_title("GRU Training Loss", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    return _save(fig, "training_loss.png")


# ─── 5. Volatility Distribution ───────────────────────────────────────────────

def plot_volatility_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 60
    ax.hist(y_true, bins=bins, alpha=0.6, label="Actual",    color="black",  density=True)
    ax.hist(y_pred, bins=bins, alpha=0.6, label="Predicted", color="tomato", density=True)
    ax.set_title("Volatility Distribution: Actual vs Predicted", fontsize=14)
    ax.set_xlabel("Realised Volatility")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, "volatility_distribution.png")


# ─── Convenience: generate all charts at once ─────────────────────────────────

def generate_all(
    y_true: np.ndarray,
    y_pred_lgb: np.ndarray,
    y_pred_robust: np.ndarray,
    metrics: dict,
    feature_importance: pd.Series,
    gru_history,
) -> list[str]:
    """Generate every chart and return a list of saved file paths."""
    paths = []
    paths.append(plot_predictions(y_true, y_pred_lgb, y_pred_robust))
    paths.append(plot_metric_comparison(metrics))
    paths.append(plot_feature_importance(feature_importance))
    if gru_history is not None:
        paths.append(plot_training_loss(gru_history))
    n = min(len(y_true), len(y_pred_robust))
    paths.append(plot_volatility_distribution(y_true[:n], y_pred_robust[:n]))
    return paths
