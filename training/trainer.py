"""
training/trainer.py — Orchestrates the full training workflow.

Call `Trainer.run()` to:
  1. Prepare data (clean, feature-engineer, scale, split).
  2. Train the LightGBM base model.
  3. Train the GRU corrector.
  4. Evaluate both models.
  5. Save everything (models, scaler, metrics, config snapshot).
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

import config
from data.preprocessor import prepare, train_test_split
from models.lightgbm_model import LightGBMModel
from models.gru_model import GRUModel, build_sequences
from models.robust_model import RobustModel
from training.evaluator import Evaluator
from utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    """Drives the end-to-end training pipeline."""

    def __init__(self, ticker: str = config.TICKER) -> None:
        self.ticker = ticker
        self.robust_model: RobustModel | None = None
        self.scaler = None
        self.evaluator = Evaluator()

    def run(self, df) -> dict:
        """
        Execute the full training workflow.

        Parameters
        ----------
        df : raw OHLCV DataFrame from data.collector

        Returns
        -------
        dict of evaluation metrics
        """
        log.info("=== Training pipeline started for %s ===", self.ticker)

        # ── 1. Preprocessing ──────────────────────────────────────────────
        log.info("Step 1/5 — Preprocessing …")
        X, y, self.scaler = prepare(df)

        if len(X) < config.SEQUENCE_LENGTH * 3:
            raise ValueError(
                f"Only {len(X)} usable rows after cleaning. Need at least "
                f"{config.SEQUENCE_LENGTH * 3} to train. "
                "Try a longer date range or lower SEQUENCE_LENGTH in config.py."
            )

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Keep a validation slice from the end of training data (10 %).
        n_val   = max(config.SEQUENCE_LENGTH * 2, int(len(X_train) * 0.10))
        X_val   = X_train.iloc[-n_val:]
        y_val   = y_train.iloc[-n_val:]
        X_train = X_train.iloc[:-n_val]
        y_train = y_train.iloc[:-n_val]

        log.info(
            "Data split: train=%d  val=%d  test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # ── 2. Train pipeline ─────────────────────────────────────────────
        log.info("Step 2/5 — Training LightGBM-GRU pipeline …")
        self.robust_model = RobustModel()
        self.robust_model.train(X_train, y_train, X_val, y_val)

        # ── 3. Evaluate ───────────────────────────────────────────────────
        log.info("Step 3/5 — Evaluating …")
        metrics = self.evaluator.evaluate_all(
            lgb_model=self.robust_model.lgb,
            robust_model=self.robust_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        # ── 4. Save models ────────────────────────────────────────────────
        log.info("Step 4/5 — Saving models …")
        self._save_all(X_train, metrics)

        # ── 5. Generate charts ────────────────────────────────────────────
        log.info("Step 5/5 — Generating charts …")
        try:
            from utils.visualizer import generate_all
            seq = config.SEQUENCE_LENGTH
            lgb_test_pred    = self.robust_model.lgb.predict(X_test)
            robust_test_pred = self.robust_model.predict(X_test)
            fi = self.robust_model.lgb.get_feature_importance()
            generate_all(
                y_true=y_test.values[seq:],
                y_pred_lgb=lgb_test_pred[seq:],
                y_pred_robust=robust_test_pred,
                metrics=metrics,
                feature_importance=fi,
                gru_history=self.robust_model.gru.history,
            )
            log.info("Charts saved to %s", config.CHART_DIR)
        except Exception as exc:
            log.warning("Chart generation failed (non-fatal): %s", exc)

        log.info("=== Training complete ===")
        return metrics

    def _save_all(self, X_train, metrics: dict) -> None:
        """Persist models, scaler, feature list, metrics, and config."""
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        # Models
        self.robust_model.save_all(config.MODEL_DIR)

        # Scaler
        scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_FILE)
        joblib.dump(self.scaler, scaler_path)
        log.info("Scaler saved → %s", scaler_path)

        # Feature list
        feat_path = os.path.join(config.MODEL_DIR, config.FEATURE_LIST_FILE)
        with open(feat_path, "w") as f:
            json.dump(list(X_train.columns), f, indent=2)
        log.info("Feature list saved → %s", feat_path)

        # Config snapshot
        cfg_path = os.path.join(config.MODEL_DIR, config.CONFIG_SNAPSHOT_FILE)
        snapshot = {
            "ticker":            self.ticker,
            "lgb_n_estimators":  config.LGB_N_ESTIMATORS,
            "lgb_learning_rate": config.LGB_LEARNING_RATE,
            "lgb_n_fold":        config.LGB_N_FOLD,
            "gru_layer1":        config.GRU_LAYER1_UNITS,
            "gru_layer2":        config.GRU_LAYER2_UNITS,
            "gru_dropout":       config.GRU_DROPOUT,
            "gru_lr":            config.GRU_LEARNING_RATE,
            "gru_epochs":        config.GRU_EPOCHS,
            "sequence_length":   config.SEQUENCE_LENGTH,
            "train_split":       config.TRAIN_SPLIT,
        }
        with open(cfg_path, "w") as f:
            json.dump(snapshot, f, indent=2)

        # Metrics
        metrics_path = os.path.join(config.MODEL_DIR, config.METRICS_FILE)
        self.evaluator.save(metrics_path)

        log.info("All artefacts saved to %s", config.MODEL_DIR)
