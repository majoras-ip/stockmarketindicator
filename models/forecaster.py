"""
models/forecaster.py — Multi-step LSTM forecaster.

Takes the last SEQUENCE_LENGTH bars of features and predicts the next
FORECAST_STEPS values of realised volatility.

Architecture: LSTM encoder → RepeatVector → LSTM decoder → Dense output
"""

from __future__ import annotations

import os
import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)

FORECAST_STEPS = 6   # bars ahead to predict (6 × 5min = 30 min)


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping (input, target) pairs for multi-step forecasting.

    X_out shape : (n_samples, seq_len, n_features)
    y_out shape : (n_samples, steps)
    """
    X_out, y_out = [], []
    for i in range(len(X) - seq_len - steps + 1):
        X_out.append(X[i : i + seq_len])
        y_out.append(y[i + seq_len : i + seq_len + steps])
    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.float32)


class LSTMForecaster:
    """
    LSTM encoder-decoder that forecasts the next *steps* bars of RV.
    """

    def __init__(self, steps: int = FORECAST_STEPS) -> None:
        self.steps   = steps
        self.model   = None
        self._built  = False

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, input_shape: tuple[int, int]) -> None:
        """
        Build the Keras model.

        Parameters
        ----------
        input_shape : (seq_len, n_features)
        """
        import tensorflow as tf
        from tensorflow import keras

        seq_len, n_features = input_shape

        inputs = keras.Input(shape=(seq_len, n_features))

        # Encoder
        x = keras.layers.LSTM(64, return_sequences=False)(inputs)
        x = keras.layers.Dropout(0.2)(x)

        # Decoder
        x = keras.layers.RepeatVector(self.steps)(x)
        x = keras.layers.LSTM(64, return_sequences=True)(x)
        x = keras.layers.Dropout(0.2)(x)

        # Output — one value per forecast step (linear; clipped to ≥0 at inference)
        outputs = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
        outputs = keras.layers.Reshape((self.steps,))(outputs)

        self.model = keras.Model(inputs, outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )
        self._built = True
        log.info("LSTMForecaster built: input=%s output=(%d,)", input_shape, self.steps)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        epochs:  int = 200,
        patience: int = 20,
    ) -> None:
        import tensorflow as tf

        if not self._built:
            raise RuntimeError("Call build() before train().")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10,
                min_lr=1e-6, verbose=1,
            ),
        ]

        # Auto batch size
        batch_size = min(256, max(32, len(X_train) // 20))

        log.info("Training forecaster: %d samples, batch=%d", len(X_train), batch_size)
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next *steps* RV values for each sequence in X.

        Returns shape: (n_samples, steps)
        """
        if self.model is None:
            raise RuntimeError("Model not built or loaded.")
        return np.clip(self.model.predict(X, verbose=0), 0, None)

    def predict_last(self, X: np.ndarray) -> np.ndarray:
        """
        Return the forecast for the most recent sequence only.

        Returns shape: (steps,)
        """
        seq = X[-1:] if X.ndim == 3 else X[-config.SEQUENCE_LENGTH:][np.newaxis]
        return self.predict(seq)[0]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        self.model.save(path)
        log.info("LSTMForecaster saved → %s", path)

    def load(self, path: str) -> None:
        import tensorflow as tf
        self.model  = tf.keras.models.load_model(path)
        self._built = True
        log.info("LSTMForecaster loaded ← %s", path)
