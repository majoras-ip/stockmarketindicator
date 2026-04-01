"""
models/robust_model.py — Combined LightGBM-GRU pipeline.

Stage 1: LightGBM produces initial volatility predictions.
Stage 2: GRU uses those predictions + original features to output the
         final refined prediction.

This file is the single object you need for training and inference.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd

import config
from models.lightgbm_model import LightGBMModel
from models.gru_model import GRUModel, build_sequences
from utils.logger import get_logger

log = get_logger(__name__)


class RobustModel:
    """
    Two-stage 'robust learning' pipeline combining LightGBM + GRU.

    Training
    ────────
    1. Train LightGBM on (X_train, y_train) — k-fold cross-validation.
    2. Collect out-of-fold LightGBM predictions for X_train.
    3. Concatenate those predictions with X_train → X_aug.
    4. Build sequences from X_aug.
    5. Train GRU on those sequences.

    Inference
    ─────────
    1. LightGBM predicts on X.
    2. Concatenate with X → X_aug.
    3. Build sequences from X_aug.
    4. GRU refines and outputs final prediction.
    """

    def __init__(self) -> None:
        self.lgb = LightGBMModel()
        self.gru = GRUModel()
        self.feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
    ) -> None:
        """
        Full two-stage training.

        Parameters
        ----------
        X_train / X_val : scaled feature DataFrames
        y_train / y_val : target volatility Series
        """
        self.feature_names = list(X_train.columns)

        # ── Stage 1: LightGBM ─────────────────────────────────────────────
        log.info("=== Stage 1: LightGBM training ===")
        lgb_oof_preds = self.lgb.train(X_train, y_train)
        lgb_val_preds = self.lgb.predict(X_val)

        # ── Stage 2: GRU ──────────────────────────────────────────────────
        log.info("=== Stage 2: GRU training ===")

        # Augment features with LightGBM predictions (paper's key contribution).
        X_aug_train = self._augment(X_train.values, lgb_oof_preds)
        X_aug_val   = self._augment(X_val.values,   lgb_val_preds)

        # Build 3-D sequences for the GRU.
        X_seq_train, y_seq_train = build_sequences(
            X_aug_train, y_train.values, config.SEQUENCE_LENGTH
        )
        X_seq_val, y_seq_val = build_sequences(
            X_aug_val, y_val.values, config.SEQUENCE_LENGTH
        )

        if len(X_seq_train) == 0:
            raise ValueError(
                f"Not enough training samples to build sequences of length "
                f"{config.SEQUENCE_LENGTH}. Try a longer date range or "
                "reduce SEQUENCE_LENGTH in config.py."
            )

        input_shape = (X_seq_train.shape[1], X_seq_train.shape[2])
        self.gru.build_model(input_shape)
        self.gru.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val)

        log.info("Both stages trained successfully.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return final volatility predictions for each valid sequence in X.

        The first (SEQUENCE_LENGTH) rows cannot produce a prediction and
        are silently discarded — the returned array is shorter than X.
        """
        lgb_preds = self.lgb.predict(X)
        X_aug     = self._augment(X.values, lgb_preds)
        # Dummy y just to reuse build_sequences (values unused).
        dummy_y = np.zeros(len(X_aug))
        X_seq, _ = build_sequences(X_aug, dummy_y, config.SEQUENCE_LENGTH)

        if len(X_seq) == 0:
            raise ValueError(
                "Not enough data to form even one sequence. "
                "Provide more rows than SEQUENCE_LENGTH."
            )

        return self.gru.predict(X_seq)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """
        Compute all paper metrics for the combined model on *test* data.

        Returns a dict with keys: MSE, MAE, RMSE, RMSPE, R2.
        """
        from training.evaluator import compute_metrics

        preds = self.predict(X_test)
        # Trim y_test to align with the shorter prediction array.
        actuals = y_test.values[config.SEQUENCE_LENGTH :]
        n = min(len(preds), len(actuals))
        return compute_metrics(actuals[:n], preds[:n])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_all(self, directory: str) -> None:
        """Persist both sub-models to *directory*."""
        os.makedirs(directory, exist_ok=True)
        self.lgb.save(directory)
        gru_path = os.path.join(directory, config.GRU_MODEL_FILE)
        self.gru.save(gru_path)
        # Save feature list for compatibility checks on load.
        feat_path = os.path.join(directory, config.FEATURE_LIST_FILE)
        with open(feat_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)
        log.info("Full pipeline saved → %s", directory)

    def load_all(self, directory: str) -> None:
        """Restore both sub-models from *directory*."""
        self.lgb.load(directory)
        gru_path  = os.path.join(directory, config.GRU_MODEL_FILE)
        self.gru.load(gru_path)
        feat_path = os.path.join(directory, config.FEATURE_LIST_FILE)
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                self.feature_names = json.load(f)
        log.info("Full pipeline loaded ← %s", directory)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _augment(X: np.ndarray, lgb_preds: np.ndarray) -> np.ndarray:
        """
        Append LightGBM predictions as an extra column to X.

        Shape: (n, features) → (n, features + 1)
        """
        return np.column_stack([X, lgb_preds.reshape(-1, 1)])
