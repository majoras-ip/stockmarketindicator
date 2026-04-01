"""
models/gru_model.py — GRU corrector model (Stage 2 of the pipeline).

Architecture (from paper):
  Input  → GRU(100) + Dropout(0.2)
         → GRU(10)  + Dropout(0.2)
         → Dense(1)

The input to this model is the concatenation of:
  • LightGBM predictions for each timestep
  • Original engineered features for each timestep

The GRU receives these as a 3-D tensor:
  (samples, sequence_length, n_features + 1)

Apple Silicon / Colab GPU detection is handled at build time.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config
from utils.logger import get_logger

log = get_logger(__name__)


def _configure_backend() -> str:
    """
    Detect and configure the best available compute backend.

    Priority:
      1. Apple Silicon GPU (tensorflow-metal)
      2. CUDA GPU (Google Colab / Linux)
      3. CPU fallback
    """
    import platform
    import tensorflow as tf

    # ── Apple Silicon ──────────────────────────────────────────────────────
    if platform.processor() == "arm" or platform.machine() == "arm64":
        try:
            physical = tf.config.list_physical_devices("GPU")
            if physical:
                log.info("Apple Silicon GPU detected (%d device(s)).", len(physical))
                # Allow memory growth to avoid grabbing all VRAM at once.
                for dev in physical:
                    tf.config.experimental.set_memory_growth(dev, True)
                return "apple_silicon_gpu"
        except Exception as e:
            log.warning("Apple Silicon GPU check failed (%s). Using CPU.", e)
        return "cpu"

    # ── CUDA (Colab / Linux) ───────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("CUDA GPU detected (%d device(s)).", len(gpus))
        for dev in gpus:
            try:
                tf.config.experimental.set_memory_growth(dev, True)
            except Exception:
                pass
        return "cuda_gpu"

    log.info("No GPU found. Falling back to CPU.")
    return "cpu"


class GRUModel:
    """
    Two-layer GRU network that refines LightGBM volatility predictions.

    The model operates on sequences of shape
    (batch, config.SEQUENCE_LENGTH, n_input_features).
    """

    def __init__(self) -> None:
        self.model = None
        self.history = None
        self._backend = None

    # ── Architecture ──────────────────────────────────────────────────────────

    def build_model(self, input_shape: tuple[int, int]) -> None:
        """
        Construct the Keras model.

        Parameters
        ----------
        input_shape : (sequence_length, n_features)
        """
        import tensorflow as tf
        from tensorflow import keras

        self._backend = _configure_backend()

        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=input_shape),
                keras.layers.GRU(
                    config.GRU_LAYER1_UNITS,
                    return_sequences=True,
                    name="gru_1",
                ),
                keras.layers.Dropout(config.GRU_DROPOUT, name="drop_1"),
                keras.layers.GRU(
                    config.GRU_LAYER2_UNITS,
                    return_sequences=False,
                    name="gru_2",
                ),
                keras.layers.Dropout(config.GRU_DROPOUT, name="drop_2"),
                keras.layers.Dense(1, name="output"),
            ],
            name="LightGBM_GRU",
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.GRU_LEARNING_RATE),
            loss="mean_squared_error",
        )

        log.info("GRU model built  backend=%s  input_shape=%s", self._backend, input_shape)
        self.model.summary(print_fn=log.info)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> None:
        """
        Fit the GRU model.

        Parameters
        ----------
        X_train / X_val : shape (samples, seq_len, features)
        y_train / y_val : shape (samples,)
        """
        import tensorflow as tf
        from tensorflow import keras

        if self.model is None:
            raise RuntimeError("Call build_model() before train().")

        # Adaptive batch size: ~10 % of training set, capped at 4096.
        if config.GRU_BATCH_SIZE is None:
            batch_size = min(4096, max(32, len(X_train) // 10))
            log.info(
                "Auto batch_size=%d  (paper used 30,096 — scaled for this machine).",
                batch_size,
            )
        else:
            batch_size = config.GRU_BATCH_SIZE

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.GRU_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=20,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        log.info(
            "Training GRU: max_epochs=%d  batch=%d  patience=%d …",
            config.GRU_EPOCHS, batch_size, config.GRU_PATIENCE,
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.GRU_EPOCHS,
            batch_size=batch_size,
            validation_freq=config.GRU_VALIDATION_FREQ,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions, shape (samples,)."""
        if self.model is None:
            raise RuntimeError("Model has not been built/loaded yet.")
        return self.model.predict(X, verbose=0).ravel()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save the Keras model in the native .keras format."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model has not been built.")
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.model.save(filepath)
        log.info("GRU model saved → %s", filepath)

    def load(self, filepath: str) -> None:
        """Load a previously saved Keras model."""
        import tensorflow as tf

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GRU model file not found: {filepath}")

        _configure_backend()   # ensure GPU is initialised before loading
        self.model = tf.keras.models.load_model(filepath)
        log.info("GRU model loaded ← %s", filepath)


# ─── Sequence builder ─────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = config.SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 2-D feature matrix into 3-D sequences for the GRU.

    Parameters
    ----------
    X       : (n_samples, n_features)
    y       : (n_samples,)
    seq_len : number of timesteps per sequence

    Returns
    -------
    X_seq : (n_samples - seq_len, seq_len, n_features)
    y_seq : (n_samples - seq_len,)  — label is the value at the END of each window
    """
    n_samples, n_features = X.shape
    n_seq = n_samples - seq_len

    X_seq = np.empty((n_seq, seq_len, n_features), dtype=np.float32)
    y_seq = np.empty(n_seq,                         dtype=np.float32)

    for i in range(n_seq):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len]

    return X_seq, y_seq
