# Model Limitations

## What this model CAN do

- Estimate 10-minute realised volatility based on recent price behaviour.
- Outperform a standalone LightGBM model on historical back-tests (per the paper).
- Run inference in near-real time once trained (~1 second per prediction).
- Work on any ticker available through yfinance.

---

## What this model CANNOT do

### 1. Predict direction
The model predicts **magnitude** of price movement (volatility), not whether
prices will go up or down.  A high volatility reading does not tell you which
way the market will move.

### 2. Anticipate news or macro shocks
Earnings surprises, Fed announcements, geopolitical events, and other
discontinuous shocks are not in the feature set.  The model will lag badly
immediately after such events.

### 3. Generalise across market regimes
A model trained on a low-volatility period may underestimate volatility in a
crisis (and vice versa).  Retrain periodically to keep the model current.

### 4. Provide reliable out-of-sample performance beyond the training window
Equity market microstructure changes over time (decimalization, algorithm
proliferation, etc.).  Performance degrades as the training data ages.

### 5. Use real bid/ask data
yfinance does not provide historical bid/ask quotes.  The spread features are
synthetic (±0.05 % of close price) and will not capture real liquidity
conditions.

### 6. Operate on live tick data
The current pipeline polls yfinance on demand; it does not consume a live
WebSocket feed.  True intraday deployment would require a real-time data
source.

### 7. Replace a risk management system
Volatility forecasts should be one input among many in any risk decision.
Do not use this model as the sole basis for trading decisions.

---

## Data limitations

| Resolution  | Max historical window (yfinance free tier) |
|-------------|---------------------------------------------|
| 1-minute    | 7 days                                      |
| 5-minute    | 60 days                                     |
| 1-hour      | ~730 days                                   |
| 1-day       | Unlimited                                   |

Training on 1-hour data gives a longer history but loses intraday
microstructure; training on 5-minute data is richer but covers fewer market
cycles.

---

## Hardware limitations

- Training with the full paper parameters (20,000 LightGBM trees, 1,000 GRU
  epochs) on a large dataset may take 30–90 minutes on a modern laptop.
- GRU batch size is automatically scaled down from the paper's 30,096 to
  fit typical consumer hardware.  This may slightly affect final accuracy.
- Apple Silicon users need `tensorflow-macos` and `tensorflow-metal` installed
  separately for GPU acceleration.
