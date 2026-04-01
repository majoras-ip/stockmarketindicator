"""
training/evaluator.py — Metric calculation and comparison reporting.

Metrics implemented (all from the paper):
  MSE   — Mean Squared Error
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Squared Error
  RMSPE — Root Mean Squared Percentage Error
  R²    — Coefficient of Determination
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

import config
from utils.logger import get_logger

log = get_logger(__name__)


# ─── Core metric function ─────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return all paper metrics as a dict."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    # RMSPE: percentage error — avoid div-by-zero on near-zero actuals.
    pct_err = (y_true - y_pred) / (np.abs(y_true) + 1e-10)
    rmspe   = float(np.sqrt(np.mean(pct_err ** 2)))

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = float(1 - ss_res / (ss_tot + 1e-10))

    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "RMSPE": rmspe, "R2": r2}


# ─── Full evaluation report ───────────────────────────────────────────────────

class Evaluator:
    """
    Orchestrates evaluation of both sub-models and the combined pipeline,
    producing a side-by-side comparison table.
    """

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def evaluate_all(
        self,
        lgb_model,
        robust_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
    ) -> dict[str, Any]:
        """
        Evaluate both LightGBM-only and full LightGBM-GRU pipeline on
        training and test sets.

        Returns nested dict:
          {
            "lightgbm": {"train": {...}, "test": {...}},
            "robust":   {"train": {...}, "test": {...}},
          }
        """
        log.info("Evaluating LightGBM (standalone) …")

        lgb_train_pred = lgb_model.predict(X_train)
        lgb_test_pred  = lgb_model.predict(X_test)

        lgb_results = {
            "train": compute_metrics(y_train.values, lgb_train_pred),
            "test":  compute_metrics(y_test.values,  lgb_test_pred),
        }

        log.info("Evaluating LightGBM-GRU (robust pipeline) …")

        # The robust model's predict() drops the first SEQUENCE_LENGTH rows.
        seq = config.SEQUENCE_LENGTH

        robust_train_pred = robust_model.predict(X_train)
        robust_test_pred  = robust_model.predict(X_test)

        robust_results = {
            "train": compute_metrics(y_train.values[seq:], robust_train_pred),
            "test":  compute_metrics(y_test.values[seq:],  robust_test_pred),
        }

        self.results = {
            "lightgbm": lgb_results,
            "robust":   robust_results,
        }

        self._print_table()
        return self.results

    def _print_table(self) -> None:
        """Pretty-print a comparison table to stdout."""
        metrics = ["MSE", "MAE", "RMSE", "RMSPE", "R2"]
        col_w = 13  # column width for numeric values

        header = (
            f"{'Metric':<8}  {'LGB Train':>{col_w}}  {'LGB Test':>{col_w}}"
            f"  {'Δ(bias)':>{col_w}}  {'Robust Train':>{col_w}}"
            f"  {'Robust Test':>{col_w}}  {'Δ(bias)':>{col_w}}"
        )
        sep = "═" * len(header)

        print("\n" + sep)
        print("  Evaluation Results")
        print(sep)
        print(header)
        print("─" * len(header))

        def _fmt(v: float, sign: bool = False) -> str:
            prefix = "+" if sign and v >= 0 else ""
            if abs(v) >= 1e4 or (abs(v) < 1e-7 and v != 0):
                return f"{prefix}{v:.4e}"
            return f"{prefix}{v:.6f}"

        for m in metrics:
            lt = self.results["lightgbm"]["train"][m]
            lv = self.results["lightgbm"]["test"][m]
            rt = self.results["robust"]["train"][m]
            rv = self.results["robust"]["test"][m]

            row = (
                f"{m:<8}  {_fmt(lt):>{col_w}}  {_fmt(lv):>{col_w}}"
                f"  {_fmt(lv - lt, sign=True):>{col_w}}"
                f"  {_fmt(rt):>{col_w}}  {_fmt(rv):>{col_w}}"
                f"  {_fmt(rv - rt, sign=True):>{col_w}}"
            )
            print(row)

        print(sep + "\n")

    def save(self, filepath: str) -> None:
        """Write results to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        log.info("Metrics saved → %s", filepath)
