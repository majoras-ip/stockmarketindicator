"""
models/lightgbm_model.py — LightGBM base model (Stage 1 of the pipeline).

Paper: "Stock Market Volatility Prediction Based on Robust GBM-GRU Model"
       Liao, Chen & Cai, 2024.

The LightGBM model produces an initial volatility estimate.
Its predictions — alongside the original features — are then fed into
the GRU corrector model.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

import config
from utils.logger import get_logger

log = get_logger(__name__)


class LightGBMModel:
    """
    Wrapper around a LightGBM regressor trained with k-fold
    cross-validation, as described in the paper.
    """

    def __init__(self) -> None:
        self.models: list[lgb.Booster] = []   # one per CV fold
        self.feature_names: list[str]  = []
        self._params = {
            "objective":        config.LGB_OBJECTIVE,
            "boosting_type":    config.LGB_BOOSTING_TYPE,
            "max_depth":        config.LGB_MAX_DEPTH,
            "learning_rate":    config.LGB_LEARNING_RATE,
            "subsample":        config.LGB_SUBSAMPLE,
            "subsample_freq":   config.LGB_SUBSAMPLE_FREQ,
            "feature_fraction": config.LGB_FEATURE_FRACTION,
            "lambda_l1":        config.LGB_LAMBDA_L1,
            "lambda_l2":        config.LGB_LAMBDA_L2,
            "seed":             config.LGB_RANDOM_STATE,
            "verbose":          config.LGB_VERBOSE,
            "n_jobs":           -1,
        }

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> np.ndarray:
        """
        Train with k-fold cross-validation (paper uses 10 folds).

        Returns out-of-fold predictions for the full training set,
        which are later used as GRU inputs during Stage 2 training.
        """
        self.feature_names = list(X_train.columns)
        X = X_train.values
        y = y_train.values

        kf   = KFold(n_splits=config.LGB_N_FOLD, shuffle=False)
        oof  = np.zeros(len(X))   # out-of-fold predictions

        log.info(
            "Training LightGBM with %d-fold CV, up to %d estimators …",
            config.LGB_N_FOLD, config.LGB_N_ESTIMATORS,
        )

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr,  X_val  = X[tr_idx],  X[val_idx]
            y_tr,  y_val  = y[tr_idx],  y[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
            dval   = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names,
                                 reference=dtrain)

            callbacks = [
                lgb.early_stopping(stopping_rounds=config.LGB_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=500),
            ]

            booster = lgb.train(
                self._params,
                dtrain,
                num_boost_round=config.LGB_N_ESTIMATORS,
                valid_sets=[dval],
                callbacks=callbacks,
            )

            oof[val_idx] = booster.predict(X_val)
            self.models.append(booster)

            fold_rmse = np.sqrt(np.mean((oof[val_idx] - y_val) ** 2))
            log.info("  Fold %d/%d  RMSE=%.6f  trees=%d",
                     fold, config.LGB_N_FOLD, fold_rmse, booster.num_trees())

        oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
        log.info("LightGBM OOF RMSE: %.6f", oof_rmse)
        return oof

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Average predictions across all fold models.
        This ensemble reduces variance compared to a single model.
        """
        if not self.models:
            raise RuntimeError("Model has not been trained or loaded yet.")

        arr = X.values if isinstance(X, pd.DataFrame) else X
        preds = np.stack([m.predict(arr) for m in self.models], axis=0)
        return preds.mean(axis=0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save all fold models to *directory*."""
        os.makedirs(directory, exist_ok=True)
        for i, booster in enumerate(self.models):
            path = os.path.join(directory, f"lgb_fold_{i}.lgb")
            booster.save_model(path)
        log.info("LightGBM: saved %d fold models → %s", len(self.models), directory)

    def load(self, directory: str) -> None:
        """Load all fold models from *directory*."""
        self.models = []
        i = 0
        while True:
            path = os.path.join(directory, f"lgb_fold_{i}.lgb")
            if not os.path.exists(path):
                break
            booster = lgb.Booster(model_file=path)
            self.models.append(booster)
            i += 1

        if not self.models:
            raise FileNotFoundError(
                f"No LightGBM fold model files found in '{directory}'."
            )
        # Recover feature names from the first model.
        self.feature_names = self.models[0].feature_name()
        log.info("LightGBM: loaded %d fold models from %s", len(self.models), directory)

    # ── Feature Importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.Series:
        """Return mean feature importance (gain) across all fold models."""
        if not self.models:
            raise RuntimeError("Model has not been trained or loaded yet.")

        importances = np.stack(
            [m.feature_importance(importance_type="gain") for m in self.models],
            axis=0,
        ).mean(axis=0)

        return pd.Series(
            importances, index=self.feature_names, name="importance"
        ).sort_values(ascending=False)
