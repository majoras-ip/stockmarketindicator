# Stock Volatility Prediction — LightGBM-GRU Hybrid Model

A Python implementation of the two-stage "robust learning" hybrid model
described in:

> **"Stock Market Volatility Prediction Based on Robust GBM-GRU Model"**
> Liao, Chen & Cai, 2024

The model predicts **10-minute realised volatility** for any stock ticker
using a sequential pipeline:

```
Raw OHLCV data
     │
     ▼
Feature Engineering (40+ features)
     │
     ▼
Stage 1: LightGBM → initial volatility estimate
     │
     ▼
Stage 2: GRU corrector (uses LightGBM output + original features)
     │
     ▼
Final volatility prediction
```

---

## Quick Start

### 1. Install dependencies

**Apple Silicon (M1/M2/M3) Mac:**
```bash
pip install tensorflow-macos tensorflow-metal
pip install -r requirements.txt
```

**Intel Mac / Linux / Windows:**
```bash
pip install -r requirements.txt
```

**Google Colab:**
```python
!pip install lightgbm yfinance joblib scipy
```
TensorFlow is pre-installed in Colab.

### 2. Run the application
```bash
python main.py
```

You'll see an interactive menu:
```
╔══════════════════════════════════════════╗
║     Stock Volatility Prediction Tool     ║
╠══════════════════════════════════════════╣
║  1. Download & Prepare Data              ║
║  2. Train Model                          ║
║  3. Evaluate Model                       ║
║  4. Predict Next 10 Minutes              ║
║  5. Load Existing Model                  ║
║  6. Export Model Info                    ║
║  7. Change Stock Ticker                  ║
║  8. View Charts (open folder)            ║
║  9. Exit                                 ║
╚══════════════════════════════════════════╝
```

### 3. End-to-end workflow

```
1 → Download data for AAPL (default, ~3 years hourly)
2 → Train — takes 15–60 min depending on hardware
3 → Evaluate — view MSE / MAE / RMSE / RMSPE / R²
4 → Predict — enter any ticker for a live forecast
8 → View Charts — opens the /charts folder
```

---

## Project Structure

```
stock_volatility/
├── main.py                 # Entry point — menu driven
├── config.py               # All settings and hyperparameters
├── data/
│   ├── collector.py        # yfinance download + CSV loader
│   ├── preprocessor.py     # Cleaning, target creation, normalisation
│   └── features.py         # 40+ engineered features
├── models/
│   ├── lightgbm_model.py   # Stage 1: LightGBM with k-fold CV
│   ├── gru_model.py        # Stage 2: GRU corrector network
│   └── robust_model.py     # Combined pipeline
├── training/
│   ├── trainer.py          # End-to-end training orchestration
│   └── evaluator.py        # Metric computation and reporting
├── prediction/
│   └── predictor.py        # Live inference pipeline
├── saved_models/           # Model files saved here after training
├── utils/
│   ├── logger.py           # Centralised logging
│   └── visualizer.py       # All chart generation
├── charts/                 # PNG charts saved here
├── requirements.txt
├── README.md
└── LIMITATIONS.md
```

---

## Configuration

All key parameters are in `config.py`. The most important ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TICKER` | `AAPL` | Stock to download |
| `PERIOD` | `3y` | Historical lookback |
| `INTERVAL` | `1h` | Bar size (auto-detected if not set) |
| `TRAIN_SPLIT` | `0.8` | Train / test ratio |
| `SEQUENCE_LENGTH` | `30` | GRU input window (timesteps) |
| `LGB_N_ESTIMATORS` | `20,000` | Max LightGBM trees (paper value) |
| `LGB_N_FOLD` | `10` | Cross-validation folds |
| `GRU_LAYER1_UNITS` | `100` | First GRU layer neurons |
| `GRU_LAYER2_UNITS` | `10` | Second GRU layer neurons |
| `GRU_EPOCHS` | `1,000` | Max GRU training epochs |
| `GRU_PATIENCE` | `50` | Early-stopping patience |
| `VOL_LOW` | `0.0005` | Low/Medium threshold |
| `VOL_MEDIUM` | `0.0015` | Medium/High threshold |
| `VOL_HIGH` | `0.0030` | High/Extreme threshold |

---

## Target Variable

The model predicts **10-minute realised volatility**:

```
RV = sqrt( Σ log(p_t / p_{t-1})² )   over a 10-bar window
```

This matches the definition used in the original paper.

---

## Feature Engineering

40+ features are computed automatically from OHLCV + bid/ask data:

| Category | Features |
|----------|----------|
| Price | Log returns (1, 5, 10, 30-bar), momentum, rate-of-change, MA distances |
| Volatility | Rolling vol (10/20/30-bar), HV ratios, vol-of-vol |
| Volume | Volume momentum, VWAP, relative volume |
| Bid/Ask | Spread, spread %, mid price |
| Time | Hour, minute, day-of-week, mins since open, open/close period flag |
| Statistical | Rolling mean/std/min/max, skewness, kurtosis |

---

## Saved Model Files

After training, the following files appear in `saved_models/`:

| File | Contents |
|------|----------|
| `lgb_fold_0.lgb … lgb_fold_9.lgb` | LightGBM fold models |
| `gru_model.keras` | Trained GRU network |
| `scaler.joblib` | MinMaxScaler fitted on training data |
| `feature_list.json` | Feature names and order |
| `config_snapshot.json` | Hyperparameters used during training |
| `training_metrics.json` | MSE/MAE/RMSE/RMSPE/R² for both models |

---

## Generated Charts

All PNG charts are saved in the `charts/` folder:

| File | Description |
|------|-------------|
| `predicted_vs_actual.png` | Time-series: actual vs LightGBM vs LightGBM-GRU |
| `metric_comparison.png` | Bar chart comparing model errors |
| `feature_importance.png` | Top 20 LightGBM features by gain |
| `training_loss.png` | GRU training and validation loss curves |
| `volatility_distribution.png` | Histogram: actual vs predicted volatility |

---

## Volatility Levels

| Level | Threshold | Interpretation |
|-------|-----------|----------------|
| Low | < 0.0005 | Calm market, typical day |
| Medium | 0.0005–0.0015 | Normal intraday movement |
| High | 0.0015–0.0030 | Elevated volatility, larger moves |
| Extreme | > 0.0030 | Very high volatility, exercise caution |

Thresholds are configurable in `config.py` (`VOL_LOW`, `VOL_MEDIUM`, `VOL_HIGH`).

---

## Google Colab Compatibility

The training pipeline auto-detects Colab CUDA GPUs and Apple Silicon GPUs.
To run on Colab:

```python
# Clone or upload the project, then:
import subprocess
subprocess.run(["pip", "install", "lightgbm", "yfinance", "joblib", "scipy"])

# Then run from the project root:
%cd stock_volatility
from training.trainer import Trainer
from data.collector import download

df = download("AAPL", period="60d", interval="5m")
trainer = Trainer("AAPL")
trainer.run(df)
```

---

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for a full discussion of what the model
can and cannot do.

Key points:
- Predicts **magnitude** of movement, not direction.
- Cannot anticipate news or macro shocks.
- yfinance provides only 7 days of 1-minute data (60 days of 5-minute).
- Bid/ask spread features are synthetic — real data not available from yfinance.
- Retrain periodically as market microstructure evolves.

---

## Reference

Liao, W., Chen, X., & Cai, Y. (2024).
*Stock Market Volatility Prediction Based on Robust GBM-GRU Model.*
