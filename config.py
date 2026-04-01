"""
config.py — Central configuration for all model parameters and paths.
Change values here to tune the model without touching any other file.
"""

import os

# ─── Data Settings ────────────────────────────────────────────────────────────
TICKER = "AAPL"          # Default stock ticker
PERIOD = "3y"            # Historical lookback period
INTERVAL = "1h"          # Default data interval
TRAIN_SPLIT = 0.80       # Fraction of data used for training
SEQUENCE_LENGTH = 30     # Number of timesteps fed into the GRU
FORECAST_HORIZON = 6    # Bars ahead to predict (6 x 5m = 30 min ahead)

# ─── LightGBM Settings (from paper) ───────────────────────────────────────────
LGB_N_ESTIMATORS     = 20_000   # Paper: 20,000
LGB_OBJECTIVE        = "rmse"
LGB_BOOSTING_TYPE    = "gbdt"
LGB_MAX_DEPTH        = -1       # -1 = no limit
LGB_LEARNING_RATE    = 0.01
LGB_SUBSAMPLE        = 0.8
LGB_SUBSAMPLE_FREQ   = 4
LGB_FEATURE_FRACTION = 0.8
LGB_LAMBDA_L1        = 1
LGB_LAMBDA_L2        = 1
LGB_RANDOM_STATE     = 66
LGB_EARLY_STOPPING   = 500      # rounds without improvement
LGB_N_FOLD           = 10       # k-fold cross-validation folds
LGB_VERBOSE          = -1       # suppress per-tree output

# ─── GRU Settings (from paper) ────────────────────────────────────────────────
GRU_LAYER1_UNITS  = 100
GRU_LAYER2_UNITS  = 10
GRU_DROPOUT       = 0.2
GRU_LEARNING_RATE = 0.00001    # Paper: 1e-5
GRU_EPOCHS        = 1000
GRU_PATIENCE      = 50         # early stopping patience
GRU_VALIDATION_FREQ = 5        # validate every N epochs
# Paper batch size was 30,096 — scaled down here for typical machines.
# Set to None to auto-compute as ~10% of training data (capped at 4096).
GRU_BATCH_SIZE    = None

# ─── Volatility Thresholds ────────────────────────────────────────────────────
VOL_LOW     = 0.0130   # below this → Low
VOL_MEDIUM  = 0.0170   # below this → Medium
VOL_HIGH    = 0.0220   # below this → High  (above → Extreme)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "saved_models")
CHART_DIR   = os.path.join(BASE_DIR, "charts")
DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")

# ─── File names inside MODEL_DIR ──────────────────────────────────────────────
LGB_MODEL_FILE       = "lightgbm_model.lgb"
GRU_MODEL_FILE       = "gru_model.keras"
SCALER_FILE          = "scaler.joblib"
FEATURE_LIST_FILE    = "feature_list.json"
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"
METRICS_FILE         = "training_metrics.json"

# ─── Market Hours (Eastern Time) ──────────────────────────────────────────────
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"

# ─── Outlier Removal ──────────────────────────────────────────────────────────
OUTLIER_STD = 3.0   # rows beyond this many std-devs are dropped
