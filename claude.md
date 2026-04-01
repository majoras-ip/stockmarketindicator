# Stock Volatility Project — Claude Progress Log

## Status: FUNCTIONAL — OPEN ITEMS REMAIN

---

## What was built

A full LightGBM-GRU hybrid model for 10-minute stock volatility prediction,
based on Liao, Chen & Cai (2024).

### File inventory

| File | Status | Notes |
|------|--------|-------|
| `config.py` | ✅ | All hyperparameters, paths, thresholds centralised here |
| `data/collector.py` | ✅ | yfinance download with auto interval selection + CSV loader |
| `data/features.py` | ✅ | 42 engineered features across 6 categories |
| `data/preprocessor.py` | ✅ | Cleaning, market-hours filter, outlier removal, RV target, MinMaxScaler |
| `models/lightgbm_model.py` | ✅ | k-fold LightGBM, ensemble predict, save/load per fold |
| `models/gru_model.py` | ✅ | 2-layer GRU, Apple Silicon + CUDA detection, sequence builder |
| `models/robust_model.py` | ✅ | Combined pipeline: augment → sequence → GRU |
| `training/trainer.py` | ✅ | End-to-end orchestration with val split, saves all artefacts, auto-generates charts |
| `training/evaluator.py` | ✅ | MSE/MAE/RMSE/RMSPE/R² table for both models, adaptive number formatting |
| `prediction/predictor.py` | ⚠️ | Written but never run end-to-end through menu |
| `utils/logger.py` | ✅ | Centralised logging to stdout |
| `utils/visualizer.py` | ✅ | 5 chart types saved as PNG |
| `main.py` | ✅ | 9-option interactive menu |
| `requirements.txt` | ✅ | Pinned dependencies |
| `README.md` | ✅ | Full setup, usage, config, file map |
| `LIMITATIONS.md` | ✅ | Honest model limitations |

---

## Known open items (as of 2026-03-30)

### 1. GRU never fully validated
Every test run used reduced epochs (50–300). The GRU needs the full
1,000-epoch run from config.py to converge and improve on LightGBM.
In all smoke tests the GRU underperformed LightGBM significantly.
**Root cause**: lr=1e-5 requires ~500–800 epochs to meaningfully move weights.
With CPU-only training this takes ~20–60 min depending on dataset size.

### 2. Predictor (menu option 4) untested end-to-end
`prediction/predictor.py` is written and logically correct but has never
been exercised through the full menu → download → load → predict flow.

### 3. `training_metrics.json` may not save correctly
`Trainer._save_all()` calls `self.evaluator.save()` but the evaluator's
`results` dict is only populated if `evaluate_all()` was called. Need to
verify the file is non-empty after a full training run.

### 4. Chart auto-generation wired but unverified
Chart generation was added to `trainer.py` Step 5/5 but has not been
run through the menu to confirm it completes without error.

### 5. Model selection not integrated
Benchmarking on SPY (730d, 1h) showed:

| Rank | Model | RMSE | R² | Time |
|------|-------|------|----|------|
| 1 | Gradient Boosting (sklearn, 500 trees) | 0.000145 | 0.9992 | 12s |
| 2 | Extra Trees (100) | 0.000185 | 0.9986 | 0.3s |
| 3 | Random Forest (100) | 0.000236 | 0.9978 | 0.5s |
| 4 | Linear Regression | 0.000436 | 0.9923 | 0.0s |
| 8 | LightGBM (5-fold, 3000 trees) | 0.001383 | 0.9229 | 12s |
| 15 | GRU standalone | 0.015052 | -8.00 | 65s |

Decision pending: keep app as paper-faithful LightGBM-GRU only, add
model selection to menu, or replace LightGBM with sklearn GBM.
**User has not decided yet.**

---

## Environment (macOS, 2026-03-30)

- Python 3.13 via `/Library/Frameworks/Python.framework/`
- LightGBM 4.6.0 — libomp rpath already fixed via install_name_tool
- TensorFlow 2.21.0 — CPU only (tensorflow-macos not available for Python 3.13)
- All other deps installed via pip3

---

## Key design decisions made

1. **LightGBM k-fold ensemble** — predictions averaged across all fold models
2. **GRU input = original features + LightGBM prediction** — paper's "robust learning" augmentation
3. **Scaler fitted on training set only** — no leakage
4. **Market-hours filter** — Eastern Time, 9:30–16:00 only
5. **Batch size auto-scaled** — paper's 30,096 capped at 4,096 for typical hardware
6. **Evaluator uses adaptive formatting** — scientific notation for large/small values

---

## Data limits (yfinance free tier)

| Resolution | Max window | Used for |
|------------|------------|---------|
| 1-min | 7 days | Not used (too short for training) |
| 5-min | 60 days | Smoke tests, live prediction |
| 1-hour | ~730 days | **Training (recommended)** |
| 1-day | Unlimited | Not used (too coarse for 10-min vol) |

---

## How to run

```bash
cd /Users/ayden/code/stock_volatility
python3 main.py
# → 1 (Download, use 730d/1h) → 2 (Train, ~30-60 min full run) → 4 (Predict)
```
