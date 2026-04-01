---
name: Project State
description: Current training status, recent changes, and open items for the stock volatility project
type: project
---

Model is currently training (as of 2026-03-30) for the first time — option 1 (730d/1h AAPL) then option 2.

**Recent changes made this session:**

1. Added `prediction/pine_exporter.py` — generates a ready-to-paste TradingView Pine Script with predictions embedded as arrays. Works on any TradingView timeframe (averages predictions within higher-TF bars, holds last prediction for finer-TF bars).

2. Added menu option 9 "Export TradingView Pine Script" to `main.py` (exit moved to 10).

3. Removed 4 price-level (scale-dependent) features from `data/features.py` to make the model generalize across stocks:
   - Removed: `momentum_10`, `vwap`, `spread`, `mid_price`
   - Kept their normalized equivalents: `roc_10`, `vwap_dist`, `spread_pct`
   - Feature count: 42 → 38, all now scale-free (log returns, ratios, percentages)

**Why:** User wants to train once and predict any stock. With scale-free features, the scaler fitted on AAPL stays valid for other equities.

**How to apply:** After training completes, option 9 → any ticker. Caveat: stocks with structurally different volatility regimes (e.g. MSTR vs SPY) may be off in magnitude but directionally correct.
