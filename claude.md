# ChartEdge · Claude Progress Log

## Status: LIVE ON RAILWAY — chartedge.trade

---

## Stack

- **Backend:** Flask + gunicorn on Railway, auto-deploys from GitHub (`majoras-ip/stockmarketindicator`)
- **DB:** PostgreSQL (Railway)
- **Payments:** Stripe (Basic $9.99/mo, Pro $15.99/mo · yearly saves up to 25%)
- **Python:** 3.13 · Key deps: yfinance, lightgbm, tensorflow, scipy, feedparser

---

## Plan tiers & access gates

| Feature | Free | Basic | Pro |
|---------|------|-------|-----|
| Copies/day | 3 | 8 | Unlimited |
| All indicators | ✓ | ✓ | ✓ |
| Live chart & forecast | ✓ | ✓ | ✓ |
| Unusual volume | ✓ | ✓ | ✓ |
| Market news | ✓ | ✓ | ✓ |
| Earnings calendar | ✓ | ✓ | ✓ |
| Market heatmap | ✓ | ✓ | ✓ |
| Dividends calendar | ✓ | ✓ | ✓ |
| IPO calendar | ✓ | ✓ | ✓ |
| Crypto fear & greed | ✓ | ✓ | ✓ |
| Crypto heatmap | ✓ | ✓ | ✓ |
| Trending coins | ✓ | ✓ | ✓ |
| Trump tracker | ✗ | ✓ | ✓ |
| Ticker news | ✗ | ✓ | ✓ |
| Gamma exposure | ✗ | ✓ | ✓ |
| Greeks dashboard | ✗ | ✓ | ✓ |
| Volatility forecast | ✗ | ✓ | ✓ |
| BTC dominance | ✗ | ✓ | ✓ |
| Funding rates | ✗ | ✓ | ✓ |
| On-chain metrics | ✗ | ✓ | ✓ |
| LSTM forecast | ✗ | ✗ | ✓ |
| Options flow | ✗ | ✗ | ✓ |
| Insider trading | ✗ | ✗ | ✓ |
| Pre-market scanner | ✗ | ✗ | ✓ |
| Liquidation map | ✗ | ✗ | ✓ |

---

## File inventory

| File | Notes |
|------|-------|
| `dashboard.py` | Single-file Flask app, ~10500+ lines |
| `Dockerfile` | python:3.13-slim + libgomp1, exposes 5000 |
| `requirements.txt` | Pinned deps |

---

## Key features & routes

| Route | Description | Plan |
|-------|-------------|------|
| `/indicators` | Pine Script generator — all indicators + Tutorial tab | Free+ |
| `/generate` | LSTM forecast | Pro |
| `/dashboard` | Live chart | Free+ |
| `/heatmap` | D3 squarified treemap, S&P 500 sectors | Free+ |
| `/earnings` | Earnings calendar w/ EPS/revenue estimates | Free+ |
| `/dividends` | Upcoming ex-dividend dates (~140 tickers) | Free+ |
| `/ipo` | IPO calendar — upcoming & priced | Free+ |
| `/trump` | Trump Tracker — BTC default, 1D default, event markers | Basic+ |
| `/volforecast` | Volatility forecast (RV, EWMA, regime) | Basic+ |
| `/gamma` | Gamma exposure chart | Basic+ |
| `/greeks` | Greeks dashboard (Delta/Gamma/Theta/Vega/Rho) | Basic+ |
| `/flow` | Options flow | Pro |
| `/insider` | Insider trading — SEC + Congress, name/source filter | Pro |
| `/premarket` | Pre-market scanner | Pro |
| `/volume` | Unusual volume scanner | Free+ |
| `/news` | Market news feed | Free+ |
| `/crypto/feargreed` | Crypto Fear & Greed Index | Free+ |
| `/crypto/heatmap` | Crypto market heatmap | Free+ |
| `/crypto/upcoming` | Trending coins (CoinGecko) | Free+ |
| `/crypto/dominance` | BTC dominance chart | Basic+ |
| `/crypto/funding` | Funding rates (Hyperliquid) | Basic+ |
| `/crypto/onchain` | On-chain metrics | Basic+ |
| `/crypto/liquidations` | Liquidation map (Hyperliquid) | Pro |
| `/me` | My Dashboard — profile + watchlist + favorites + copy history | Logged in |
| `/settings` | Account settings | Logged in |

---

## Important implementation details

### Pine Script expiry injection
Every generated Pine Script goes through `_inject_expiry(script, username)` before being returned. This embeds **4 expiry checks** at different positions (after indicator(), ~33%, ~66%, end) using 3 different syntactic forms so they're hard to find and remove all at once. Scripts break the day after generation with `runtime.error("Expired · Regenerate at chartedge.trade")`. Username watermarked in comments.

### Nav dropdown plan-gating
`_NAV_LINKS` uses `{{ nav_plan }}` (injected via `@app.context_processor`) to grey out locked tools at 40% opacity with 🔒 icon. Free users see Trump/Gamma/Greeks/VolForecast/Flow/Insider/Premarket + most crypto locked. Basic users see Flow/Insider/Premarket + Liquidation Map locked. Nav order: Tools | Crypto | Community | Pricing | Username ▾

### Background pre-fetch pattern
Heatmap and dividends use daemon threads pre-fetched at startup. Returns `{"loading": true}` if cache empty, triggers background refresh if stale (>1h).

### Dividends calendar
~140 tickers across dividend aristocrats, banks, insurance, energy MLPs, industrials, REITs, BDCs, income ETFs. Shows ex-dates within next 60 days only.

### Trump Tracker
- Default ticker: BTC-USD, default period: 1D
- Event markers (📰) on chart at news article timestamps
- Gold uses GC=F (futures ~$3000), not GLD ETF
- Prices >$999 formatted with commas

### Insider Trading
- SEC Form 4 filings (major SP500 companies) + House STOCK Act via QuiverQuant
- Filters: All/Congress/SEC tabs + name search + ticker search (all client-side)

### Volatility Forecast
- Banner at top: "TradingView Plus subscription required"
- Shows: realized vol (20d), EWMA vol, IV vs RV chart, forecast N days ahead
- Regime detection: Low/Medium/High/Extreme

### Crypto tools
- All exchange data via **Hyperliquid** (decentralized, no geo blocks from Railway)
- CoinGecko free tier for heatmap, dominance, trending
- alternative.me for Fear & Greed Index
- Liquidation map shows funding rate history as bar chart (values in basis points ×10000)
- All crypto API endpoints use manual session check (NOT @login_required) to return JSON 401 instead of HTML redirect

### My Dashboard (/me)
- Combined profile + dashboard page. `/profile` redirects here.
- Shows: profile header (avatar, username, plan pill, edit button), stats, badges, copy usage bar, recent copies, favorites, watchlist with live prices, plan card, account links
- DB tables: `watchlist (user_id, symbol, added)`, `copy_history (id, user_id, indicator, ts)`
- API: `GET/POST/DELETE /api/watchlist`, `GET /api/me/stats`, `GET /api/me/history`, `GET /api/favorites`, `DELETE /api/favorite/<key>`

### Settings (/settings)
- Theme (dark/light, saved to DB), default ticker (pre-fills chart pages), copy toast on/off, change email, change password, delete account
- DB columns added via `_migrate_pg()`: `theme TEXT DEFAULT 'dark'`, `default_ticker TEXT DEFAULT 'AAPL'`, `copy_toast INTEGER DEFAULT 1`
- Delete account cancels Stripe subscription and wipes all user data

### hCaptcha
- On register form only. Site key: `b06c575c-981a-4884-bbc0-5aa8c1d8aaa0`
- Secret stored as Railway env var `HCAPTCHA_SECRET`

### Ticker tape
- Live prices via yfinance for: SPY, AAPL, NVDA, TSLA, MSFT, META, AMZN, QQQ, AMD, GOOGL, BTC-USD, GC=F
- 5-minute cache via `_tape_cache`. Endpoint: `GET /api/tickertape`

### Copy tracking
- `/api/copy` accepts `indicator` field and logs to `copy_history` for all logged-in users (including Pro)
- Pro users: `_increment_copy()` is called before returning (was previously skipped — fixed)

### Indicators Tutorial tab
- Two tabs on `/indicators`: Indicators (default) | Tutorial
- Tutorial has YouTube embed (unlisted, anonymous account): `https://www.youtube.com/embed/q8hcp4K4rNM?rel=0&modestbranding=1&iv_load_policy=3&fs=1&controls=1`
- Step-by-step written guide below video

### Homepage
- "Get Started Free" button hidden for logged-in users, shows only for visitors

---

## Known patterns & gotchas

- **JS string escaping in triple-quoted Python:** Use `&quot;` for string args in onclick, or data attributes + event delegation. `\'x\'` → `'x'` breaks JS string literals.
- **Plotly fill:'tozeroy'** makes financial charts flat when price >> 0. Always use tight y-axis range `[min*0.998, max*1.002]`.
- **yfinance NaN in jsonify:** Python's JSON encoder rejects NaN. Always sanitize with `_safe_f()` / `_safe_i()` before returning.
- **pandas itertuples()** returns named tuples — use attribute access (`r.strike`) not string indexing (`r["strike"]`). Or use `.astype()` directly on the column.
- **MultiIndex columns** from `yf.download()` multi-ticker: flatten with `raw.columns = raw.columns.get_level_values(0)`.
- **@login_required on JSON endpoints** returns HTML redirect — `fetch().then(r=>r.json())` throws silently. Always use manual session check on API routes.
- **Plotly.newPlot vs react** — always clear div and use `newPlot` for initial renders; `react` can fail silently on divs with existing innerHTML.
- **Nav dropdowns** — last two dropdowns use `right: 0` alignment to prevent overflow off right edge of screen.

---

## Environment

- Python 3.13
- LightGBM 4.6.0 — libomp rpath fixed via install_name_tool (macOS dev)
- TensorFlow 2.21.0 — CPU only
- Deployed: Railway · Domain: chartedge.trade
- GitHub: majoras-ip/stockmarketindicator
